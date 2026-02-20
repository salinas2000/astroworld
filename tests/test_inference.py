"""
Tests for the P9 candidate searcher inference pipeline.

All tests use tiny synthetic FITS images (64×64 or 128×128) with
Gaussian point sources injected at known positions.  No GPU required.
"""

from pathlib import Path

import numpy as np
import pytest

from astroworld.ml import HAS_TORCH

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

if HAS_TORCH:
    import torch
    from astroworld.ml.inference import (
        Candidate,
        P9Searcher,
        PatchDetection,
        SearchResult,
    )
    from astroworld.ml.model import P9SiameseDetector
    from astroworld.ml.post_processing import (
        detections_to_candidates,
        generate_report,
        non_maximum_suppression,
        plot_heatmap,
        save_diagnostic_cutouts,
    )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_gaussian(size, cx, cy, flux, fwhm=2.5):
    """Create a 2D Gaussian at (cx, cy) with given flux."""
    sigma = fwhm / 2.3548
    y, x = np.mgrid[0:size, 0:size]
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    g = g / g.sum() * flux
    return g


def _make_fits(path, nx=64, ny=64, sources=None, bg_mean=1000, bg_std=100, seed=42):
    """Create a FITS file with optional injected sources.

    Parameters
    ----------
    sources : list of (cx, cy, flux) tuples, or None for clean image
    """
    from astropy.io import fits

    rng = np.random.default_rng(seed)
    image = rng.normal(bg_mean, bg_std, size=(ny, nx)).astype(np.float32)

    if sources:
        for cx, cy, flux in sources:
            image += _make_gaussian(ny, cx, cy, flux).astype(np.float32)

    header = fits.Header()
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CRVAL1"] = 71.3
    header["CRVAL2"] = 14.5
    header["CRPIX1"] = nx / 2.0
    header["CRPIX2"] = ny / 2.0
    header["CDELT1"] = -0.000326
    header["CDELT2"] = 0.000326
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"

    hdu = fits.PrimaryHDU(image, header=header)
    hdu.writeto(str(path), overwrite=True)

    return image


def _make_pair_fits(tmp_path, img_size=128, inject=True, motion_px=8, snr=10,
                    bg_std=100, seed=42):
    """Create epoch 1 and epoch 2 FITS with optional moving source.

    Returns (path_1, path_2, inject_x, inject_y) where (inject_x, inject_y)
    is the source position in epoch 1 (NaN if no injection).
    """
    rng = np.random.default_rng(seed)
    cx = img_size / 2.0
    cy = img_size / 2.0

    sources_1 = None
    sources_2 = None

    if inject:
        flux = snr * bg_std * 6.0  # Total flux for visible source
        angle = rng.uniform(0, 2 * np.pi)
        dx = motion_px * np.cos(angle)
        dy = motion_px * np.sin(angle)
        sources_1 = [(cx, cy, flux)]
        sources_2 = [(cx + dx, cy + dy, flux)]

    path_1 = tmp_path / "epoch1.fits"
    path_2 = tmp_path / "epoch2.fits"

    _make_fits(path_1, nx=img_size, ny=img_size, sources=sources_1,
               bg_std=bg_std, seed=seed)
    _make_fits(path_2, nx=img_size, ny=img_size, sources=sources_2,
               bg_std=bg_std, seed=seed + 1)

    inject_x = cx if inject else float("nan")
    inject_y = cy if inject else float("nan")

    return path_1, path_2, inject_x, inject_y


def _make_dummy_checkpoint(tmp_path):
    """Create a dummy P9SiameseDetector checkpoint."""
    model = P9SiameseDetector()
    path = tmp_path / "test_model.pt"
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_loss": 0.5,
    }, path)
    return path


# ---------------------------------------------------------------------------
# TestTiling
# ---------------------------------------------------------------------------

@requires_torch
class TestTiling:
    """Test the sliding window tiling logic."""

    def test_grid_shape_512(self):
        """512×512 with patch=64, overlap=8 → 9×9 grid."""
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, overlap=8, device="cpu")
        n_rows, n_cols, y_origins, x_origins = searcher._compute_grid_shape(512, 512)
        assert n_rows == 9
        assert n_cols == 9
        assert y_origins[0] == 0
        assert y_origins[-1] == 448  # 448 + 64 = 512

    def test_grid_shape_128(self):
        """128×128 → 3×3 grid (origins at 0, 56, 64 for edge coverage)."""
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, overlap=8, device="cpu")
        n_rows, n_cols, y_origins, x_origins = searcher._compute_grid_shape(128, 128)
        # stride=56: origins [0, 56] → 56+64=120 < 128, so add 128-64=64
        # Final origins: [0, 56, 64]
        assert n_rows == 3
        assert n_cols == 3
        assert y_origins[-1] + 64 == 128  # Edge coverage

    def test_grid_shape_64(self):
        """64×64 → 1×1 grid (single patch)."""
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, overlap=8, device="cpu")
        n_rows, n_cols, y_origins, x_origins = searcher._compute_grid_shape(64, 64)
        assert n_rows == 1
        assert n_cols == 1
        assert y_origins == [0]
        assert x_origins == [0]

    def test_edge_coverage(self):
        """Last patch reaches exactly to image edge."""
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, overlap=8, device="cpu")
        _, _, y_origins, x_origins = searcher._compute_grid_shape(512, 512)
        assert y_origins[-1] + 64 == 512
        assert x_origins[-1] + 64 == 512

    def test_non_divisible_size(self):
        """Non-evenly-divisible image gets edge-anchored extra patch."""
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, overlap=8, device="cpu")
        # 100 pixels: stride=56 → origins [0], but 0+64=64 < 100
        # so extra origin at 100-64=36
        n_rows, n_cols, y_origins, x_origins = searcher._compute_grid_shape(100, 100)
        assert y_origins[-1] + 64 == 100
        assert len(y_origins) >= 2


# ---------------------------------------------------------------------------
# TestBatchInference
# ---------------------------------------------------------------------------

@requires_torch
class TestBatchInference:
    """Test the batch inference path."""

    def test_output_shape(self):
        """Output has one probability per patch."""
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, device="cpu")

        n = 9  # number of patches
        p1 = torch.randn(n, 1, 64, 64)
        p2 = torch.randn(n, 1, 64, 64)
        probs = searcher._batch_inference(p1, p2)

        assert probs.shape == (n,)

    def test_output_bounded(self):
        """All probabilities in [0, 1]."""
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, device="cpu")

        p1 = torch.randn(5, 1, 64, 64)
        p2 = torch.randn(5, 1, 64, 64)
        probs = searcher._batch_inference(p1, p2)

        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_deterministic(self):
        """Same input produces same output in eval mode."""
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, device="cpu")

        p1 = torch.randn(3, 1, 64, 64)
        p2 = torch.randn(3, 1, 64, 64)

        r1 = searcher._batch_inference(p1, p2)
        r2 = searcher._batch_inference(p1, p2)

        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# TestP9Searcher
# ---------------------------------------------------------------------------

@requires_torch
class TestP9Searcher:
    """Test the full search pipeline."""

    def test_search_returns_result(self, tmp_path):
        """search() returns a SearchResult with correct fields."""
        p1, p2, _, _ = _make_pair_fits(tmp_path, img_size=64)
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, device="cpu", threshold=0.0)

        result = searcher.search(p1, p2)

        assert isinstance(result, SearchResult)
        assert result.heatmap.shape == result.grid_shape
        assert result.patch_size == 64
        assert result.stride == 56
        assert result.inference_time_seconds >= 0

    def test_heatmap_shape_128(self, tmp_path):
        """128×128 image produces correct heatmap shape."""
        p1, p2, _, _ = _make_pair_fits(tmp_path, img_size=128)
        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, overlap=8, device="cpu")

        result = searcher.search(p1, p2)

        assert result.heatmap.shape[0] >= 2
        assert result.heatmap.shape[1] >= 2

    def test_from_checkpoint(self, tmp_path):
        """from_checkpoint loads model and creates searcher."""
        ckpt_path = _make_dummy_checkpoint(tmp_path)
        searcher = P9Searcher.from_checkpoint(ckpt_path, device="cpu")

        assert isinstance(searcher.model, P9SiameseDetector)
        assert searcher.patch_size == 64

    def test_search_from_arrays(self, tmp_path):
        """search_from_arrays works with numpy arrays."""
        rng = np.random.default_rng(42)
        img = rng.normal(1000, 100, size=(64, 64)).astype(np.float32)

        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, device="cpu")

        result = searcher.search_from_arrays(img, img)

        assert isinstance(result, SearchResult)
        assert result.image_path_1 == "<array>"

    def test_mismatched_shapes_raises(self):
        """search_from_arrays raises ValueError for mismatched images."""
        img1 = np.zeros((64, 64), dtype=np.float32)
        img2 = np.zeros((128, 128), dtype=np.float32)

        model = P9SiameseDetector()
        searcher = P9Searcher(model, patch_size=64, device="cpu")

        with pytest.raises(ValueError, match="shapes must match"):
            searcher.search_from_arrays(img1, img2)


# ---------------------------------------------------------------------------
# TestNMS
# ---------------------------------------------------------------------------

@requires_torch
class TestNMS:
    """Test Non-Maximum Suppression."""

    def test_empty_input(self):
        """No detections → empty list."""
        result = non_maximum_suppression([], grid_shape=(9, 9))
        assert result == []

    def test_single_detection(self):
        """Single detection passes through."""
        det = PatchDetection(
            patch_row=4, patch_col=4, pixel_x=256, pixel_y=256,
            probability=0.99,
        )
        result = non_maximum_suppression([det], grid_shape=(9, 9))
        assert len(result) == 1
        assert result[0].probability == 0.99

    def test_suppresses_neighbors(self):
        """Adjacent lower-probability detection gets suppressed."""
        det_strong = PatchDetection(
            patch_row=4, patch_col=4, pixel_x=256, pixel_y=256,
            probability=0.99,
        )
        det_weak = PatchDetection(
            patch_row=4, patch_col=5, pixel_x=312, pixel_y=256,
            probability=0.85,
        )
        result = non_maximum_suppression(
            [det_weak, det_strong], grid_shape=(9, 9), suppress_radius=1,
        )
        assert len(result) == 1
        assert result[0].probability == 0.99

    def test_preserves_distant(self):
        """Detections far apart both survive."""
        det_a = PatchDetection(
            patch_row=0, patch_col=0, pixel_x=32, pixel_y=32,
            probability=0.98,
        )
        det_b = PatchDetection(
            patch_row=8, patch_col=8, pixel_x=480, pixel_y=480,
            probability=0.96,
        )
        result = non_maximum_suppression(
            [det_a, det_b], grid_shape=(9, 9), suppress_radius=1,
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TestPostProcessing
# ---------------------------------------------------------------------------

@requires_torch
class TestPostProcessing:
    """Test coordinate conversion, report generation, and visualization."""

    def test_pixel_to_radec_conversion(self):
        """detections_to_candidates converts pixel to RA/Dec."""
        det = PatchDetection(
            patch_row=4, patch_col=4, pixel_x=32.0, pixel_y=32.0,
            probability=0.99,
        )
        header = {
            "CRVAL1": 71.3, "CRVAL2": 14.5,
            "CRPIX1": 32.0, "CRPIX2": 32.0,
            "CDELT1": -0.000326, "CDELT2": 0.000326,
            "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
            "NAXIS1": 64, "NAXIS2": 64,
        }
        candidates = detections_to_candidates(
            [det], header, grid_shape=(9, 9),
        )
        assert len(candidates) == 1
        # At reference pixel, RA/Dec should match CRVAL
        assert abs(candidates[0].ra_deg - 71.3) < 0.01
        assert abs(candidates[0].dec_deg - 14.5) < 0.01

    def test_report_csv_format(self, tmp_path):
        """generate_report creates CSV with expected columns."""
        result = SearchResult(
            image_path_1="a.fits", image_path_2="b.fits",
            heatmap=np.zeros((2, 2)), patch_detections=[],
            candidates=[
                Candidate(
                    pixel_x=100, pixel_y=200, ra_deg=71.3, dec_deg=14.5,
                    probability=0.98, n_patches=2, patch_indices=[(1, 1), (1, 2)],
                ),
            ],
            grid_shape=(2, 2), patch_size=64, stride=56,
            threshold=0.95, inference_time_seconds=0.5,
        )

        csv_path = tmp_path / "report.csv"
        df = generate_report([result], csv_path)

        assert csv_path.exists()
        assert "probability" in df.columns
        assert "ra_deg" in df.columns
        assert len(df) == 1

    def test_heatmap_plot(self, tmp_path):
        """plot_heatmap creates PNG file."""
        image = np.random.default_rng(42).normal(
            1000, 100, size=(128, 128)
        ).astype(np.float32)

        result = SearchResult(
            image_path_1="a.fits", image_path_2="b.fits",
            heatmap=np.random.default_rng(42).random((2, 2)).astype(np.float32),
            patch_detections=[], candidates=[],
            grid_shape=(2, 2), patch_size=64, stride=56,
            threshold=0.95, inference_time_seconds=0.1,
        )

        out_path = tmp_path / "heatmap.png"
        plot_heatmap(result, image, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_diagnostic_cutouts(self, tmp_path):
        """save_diagnostic_cutouts creates PNG for each candidate."""
        rng = np.random.default_rng(42)
        img1 = rng.normal(1000, 100, size=(128, 128)).astype(np.float32)
        img2 = rng.normal(1000, 100, size=(128, 128)).astype(np.float32)

        result = SearchResult(
            image_path_1="a.fits", image_path_2="b.fits",
            heatmap=np.zeros((2, 2)),
            patch_detections=[],
            candidates=[
                Candidate(
                    pixel_x=64, pixel_y=64, ra_deg=71.3, dec_deg=14.5,
                    probability=0.99, n_patches=1, patch_indices=[(1, 1)],
                ),
            ],
            grid_shape=(2, 2), patch_size=64, stride=56,
            threshold=0.95, inference_time_seconds=0.1,
        )

        cutout_dir = tmp_path / "cutouts"
        paths = save_diagnostic_cutouts(result, img1, img2, cutout_dir)

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".png"
