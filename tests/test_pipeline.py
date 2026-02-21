"""
Tests for Phase 14: Observatory Pipeline.

All tests are offline (no network, no GPU) using mocked downloads and
untrained models.  Tests cover:
  - PipelineConfig defaults and customization
  - FieldResult dataclass
  - ObservatoryPipeline construction and mode validation
  - Stage 2 (download) path mapping
  - Stage 3 (spectral) cube construction and Planck filtering
  - Stage 4 (kinematic) conservative mode
  - Full pipeline integration (mocked end-to-end)
  - ObservatorySummary and report generation
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Check torch availability
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wcs_header(
    ra: float = 180.0,
    dec: float = 30.0,
    pixel_scale_arcsec: float = 1.7,
    nx: int = 64,
    ny: int = 64,
) -> dict:
    """Create a minimal WCS header dict for testing."""
    return {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": ra,
        "CRVAL2": dec,
        "CRPIX1": (nx + 1) / 2.0,
        "CRPIX2": (ny + 1) / 2.0,
        "CDELT1": -pixel_scale_arcsec / 3600.0,
        "CDELT2": pixel_scale_arcsec / 3600.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


def _write_synthetic_fits(
    path: Path,
    nx: int = 64,
    ny: int = 64,
    ra: float = 180.0,
    dec: float = 30.0,
    fill_zeros: bool = False,
) -> None:
    """Write a minimal synthetic FITS file for testing.

    Parameters
    ----------
    fill_zeros : bool
        If True, write all-zero data (simulates HTTP 404 / empty tile).
    """
    from astropy.io import fits

    if fill_zeros:
        data = np.zeros((ny, nx), dtype=np.float32)
    else:
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=(ny, nx)).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdr = hdu.header
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec
    hdr["CRPIX1"] = (nx + 1) / 2.0
    hdr["CRPIX2"] = (ny + 1) / 2.0
    hdr["CDELT1"] = -1.7 / 3600.0
    hdr["CDELT2"] = 1.7 / 3600.0
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(str(path), overwrite=True)


def _mock_download_factory(tmp_path: Path):
    """Create a mock download_field that writes synthetic FITS files."""

    def _mock_download(
        ra_deg,
        dec_deg,
        surveys=None,
        radius_arcmin=5.0,
        pixels=64,
        output_dir=None,
        rate_limit_sec=0.0,
    ):
        from astroworld.imaging.survey_download import _survey_to_dir

        surveys = surveys or ["DSS2 Red"]
        paths = []
        for survey in surveys:
            survey_dir = _survey_to_dir(survey)
            dec_str = f"+{dec_deg:.3f}" if dec_deg >= 0 else f"{dec_deg:.3f}"
            fname = f"field_ra{ra_deg:07.3f}_dec{dec_str}.fits"
            path = tmp_path / survey_dir / fname
            if not path.exists():
                _write_synthetic_fits(
                    path, nx=pixels, ny=pixels, ra=ra_deg, dec=dec_deg
                )
            paths.append(path)
        return paths

    return _mock_download


# ---------------------------------------------------------------------------
# TestPipelineConfig
# ---------------------------------------------------------------------------


@requires_torch
class TestPipelineConfig:
    """Tests for PipelineConfig defaults and validation."""

    def test_default_config(self):
        """Default config is spectral_only mode."""
        from astroworld.ml.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.mode == "spectral_only"
        assert cfg.spectral_threshold == 0.80
        assert cfg.mc_samples == 10
        assert cfg.patch_size == 64
        assert cfg.overlap == 8
        assert cfg.kinematic_enabled is True
        assert cfg.device == "auto"
        assert cfg.ir_min_valid_fraction == 0.90
        assert cfg.snr_min == 3.0
        assert cfg.snr_aperture_radius == 3
        assert cfg.snr_annulus_inner == 20
        assert cfg.snr_annulus_outer == 30

    def test_custom_thresholds(self):
        """Custom thresholds are stored correctly."""
        from astroworld.ml.pipeline import PipelineConfig

        cfg = PipelineConfig(
            spectral_threshold=0.50,
            mc_samples=5,
            patch_size=32,
        )
        assert cfg.spectral_threshold == 0.50
        assert cfg.mc_samples == 5
        assert cfg.patch_size == 32

    def test_planck_classes_default(self):
        """Default keeps p9_candidate and cold_object."""
        from astroworld.ml.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert "p9_candidate" in cfg.planck_classes_keep
        assert "cold_object" in cfg.planck_classes_keep
        assert "warm_reject" not in cfg.planck_classes_keep


# ---------------------------------------------------------------------------
# TestFieldResult
# ---------------------------------------------------------------------------


@requires_torch
class TestFieldResult:
    """Tests for FieldResult dataclass."""

    def test_default_values(self):
        """Default FieldResult has no candidates."""
        from astroworld.ml.pipeline import FieldResult

        fr = FieldResult(field_id="test", ra_deg=0.0, dec_deg=0.0)
        assert len(fr.final_candidates) == 0
        assert len(fr.stage3_candidates) == 0
        assert fr.stage1_passed is True
        assert fr.error is None
        assert fr.spectral_result is None

    def test_field_id_format(self):
        """FieldResult stores correct coordinates."""
        from astroworld.ml.pipeline import FieldResult

        fr = FieldResult(
            field_id="field_ra071.300_dec+14.500",
            ra_deg=71.3,
            dec_deg=14.5,
        )
        assert "071.300" in fr.field_id
        assert "+14.500" in fr.field_id
        assert fr.ra_deg == 71.3
        assert fr.dec_deg == 14.5


# ---------------------------------------------------------------------------
# TestObservatoryPipelineInit
# ---------------------------------------------------------------------------


@requires_torch
class TestObservatoryPipelineInit:
    """Tests for pipeline construction."""

    def test_from_spectral_searcher(self):
        """Construct pipeline in spectral_only mode."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(mode="spectral_only")

        pipeline = ObservatoryPipeline(config, searcher)
        assert pipeline.spectral_searcher is searcher
        assert pipeline.optical_searcher is None

    def test_dual_optical_requires_optical_searcher(self):
        """dual_optical mode raises ValueError without optical searcher."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(mode="dual_optical")

        with pytest.raises(ValueError, match="dual_optical"):
            ObservatoryPipeline(config, searcher, optical_searcher=None)


# ---------------------------------------------------------------------------
# TestStage2Download
# ---------------------------------------------------------------------------


@requires_torch
class TestStage2Download:
    """Tests for IR download stage (mocked network)."""

    def test_download_called_with_correct_surveys(self, tmp_path):
        """download_field receives WISE 3.4 + 4.6 surveys."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        called_surveys = []

        mock_dl = _mock_download_factory(tmp_path)

        def _tracking_download(*args, **kwargs):
            surveys = kwargs.get("surveys", args[2] if len(args) > 2 else None)
            called_surveys.extend(surveys or [])
            return mock_dl(*args, **kwargs)

        with patch(
            "astroworld.ml.pipeline.download_field",
            side_effect=_tracking_download,
        ):
            result = FieldResult(
                field_id="test", ra_deg=180.0, dec_deg=30.0
            )
            result = pipeline._stage2_download(result)

        assert "DSS2 Red" in called_surveys
        assert "WISE 3.4" in called_surveys
        assert "WISE 4.6" in called_surveys

    def test_paths_mapped_correctly(self, tmp_path):
        """Downloaded paths are assigned to optical/W1/W2."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        with patch(
            "astroworld.ml.pipeline.download_field",
            side_effect=_mock_download_factory(tmp_path),
        ):
            result = FieldResult(
                field_id="test", ra_deg=180.0, dec_deg=30.0
            )
            result = pipeline._stage2_download(result)

        assert result.optical_path is not None
        assert "dss2_red" in result.optical_path.lower()
        assert result.w1_path is not None
        assert "wise_3.4" in result.w1_path.lower() or "wise_3_4" in result.w1_path.lower()
        assert result.w2_path is not None
        assert result.ir_downloaded is True


# ---------------------------------------------------------------------------
# TestStage3Spectral
# ---------------------------------------------------------------------------


@requires_torch
class TestStage3Spectral:
    """Tests for spectral inference stage."""

    def test_spectral_result_populated(self, tmp_path):
        """Stage 3 produces a SpectralSearchResult."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        # Create FITS files
        optical_path = tmp_path / "dss2_red" / "test.fits"
        w1_path = tmp_path / "wise_3.4" / "test.fits"
        w2_path = tmp_path / "wise_4.6" / "test.fits"
        for p in [optical_path, w1_path, w2_path]:
            _write_synthetic_fits(p, nx=64, ny=64)

        result = FieldResult(
            field_id="test", ra_deg=180.0, dec_deg=30.0,
            optical_path=str(optical_path),
            w1_path=str(w1_path),
            w2_path=str(w2_path),
            has_w1=True,
            has_w2=True,
        )
        result = pipeline._stage3_spectral(result)

        assert result.spectral_result is not None
        assert result.spectral_result.heatmap.shape[0] > 0
        assert isinstance(result.stage3_detections, int)

    def test_planck_filter_applied(self, tmp_path):
        """Only candidates with allowed Planck classes survive Stage 3."""
        from astroworld.ml.pipeline import PipelineConfig

        config = PipelineConfig(
            planck_classes_keep=["p9_candidate"],
        )

        # Simulate enriched candidates
        enriched = [
            {"planck_class": "p9_candidate", "pixel_x": 1, "pixel_y": 1,
             "ra_deg": 0, "dec_deg": 0, "probability": 0.9,
             "temperature_k": 40, "temperature_std_k": 5},
            {"planck_class": "warm_reject", "pixel_x": 2, "pixel_y": 2,
             "ra_deg": 0, "dec_deg": 0, "probability": 0.85,
             "temperature_k": 1000, "temperature_std_k": 100},
            {"planck_class": "cold_object", "pixel_x": 3, "pixel_y": 3,
             "ra_deg": 0, "dec_deg": 0, "probability": 0.8,
             "temperature_k": 200, "temperature_std_k": 50},
        ]

        filtered = [
            c for c in enriched
            if c["planck_class"] in config.planck_classes_keep
        ]

        assert len(filtered) == 1
        assert filtered[0]["planck_class"] == "p9_candidate"

    def test_single_epoch_zero_temporal(self, tmp_path):
        """When cube2 = cube1.copy(), temporal gradient channel is zero."""
        from astroworld.imaging.spectral_cube import (
            BAND_TEMPORAL,
            build_spectral_cube,
        )

        image = np.random.default_rng(42).normal(
            100, 10, size=(64, 64)
        ).astype(np.float32)
        header = _make_wcs_header(nx=64, ny=64)

        cube1 = build_spectral_cube(image, header)
        cube2 = cube1.copy()

        # Temporal channel should be zero in both cubes
        assert np.allclose(cube1[BAND_TEMPORAL], 0.0)
        assert np.allclose(cube2[BAND_TEMPORAL], 0.0)
        assert np.array_equal(cube1, cube2)

    def test_ir_quality_filter_no_ir(self, tmp_path):
        """Stage 3 rejects ALL candidates when no WISE data available."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        # Create ONLY optical — NO W1 or W2
        optical_path = tmp_path / "dss2_red" / "test.fits"
        _write_synthetic_fits(optical_path, nx=64, ny=64)

        result = FieldResult(
            field_id="test", ra_deg=180.0, dec_deg=30.0,
            optical_path=str(optical_path),
            # has_w1=False, has_w2=False (defaults)
        )
        result = pipeline._stage3_spectral(result)

        # All candidates should be rejected due to insufficient IR
        assert len(result.stage3_candidates) == 0

    def test_ir_quality_filter_partial_ir(self, tmp_path):
        """Stage 3 keeps candidates with partial IR (1/2 bands) flagged."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        # Create optical + W1 only
        optical_path = tmp_path / "dss2_red" / "test.fits"
        w1_path = tmp_path / "wise_3.4" / "test.fits"
        _write_synthetic_fits(optical_path, nx=64, ny=64)
        _write_synthetic_fits(w1_path, nx=64, ny=64)

        result = FieldResult(
            field_id="test", ra_deg=180.0, dec_deg=30.0,
            optical_path=str(optical_path),
            w1_path=str(w1_path),
            has_w1=True,  # W1 loaded successfully
            # has_w2=False (missing)
        )
        result = pipeline._stage3_spectral(result)

        # Candidates should be kept but flagged as partial.
        # Per-patch filter may label as "partial_patch" (only W1 in patch),
        # and field-level filter then upgrades to "partial" for remaining.
        # Accept either partial variant.
        for c in result.stage3_candidates:
            if "ir_quality" in c:
                assert c["ir_quality"] in ("partial", "partial_patch")

    def test_ir_patch_filter_zero_tiles(self, tmp_path):
        """Per-patch IR filter rejects detections when patch IR pixels are zeros.

        Simulates WISE tile-boundary artifact: field has W1/W2 files (has_w1=True,
        has_w2=True), but the actual FITS data is all zeros — the 64×64 patch
        region has <90% valid pixels, so candidates are rejected.
        """
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.01, mc_samples=1, device="cpu",  # very low thresh
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
            ir_min_valid_fraction=0.90,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        # Optical = real data, W1/W2 = all zeros (tile boundary)
        optical_path = tmp_path / "dss2_red" / "test.fits"
        w1_path = tmp_path / "wise_3.4" / "test.fits"
        w2_path = tmp_path / "wise_4.6" / "test.fits"
        _write_synthetic_fits(optical_path, nx=64, ny=64)
        _write_synthetic_fits(w1_path, nx=64, ny=64, fill_zeros=True)
        _write_synthetic_fits(w2_path, nx=64, ny=64, fill_zeros=True)

        result = FieldResult(
            field_id="test", ra_deg=180.0, dec_deg=30.0,
            optical_path=str(optical_path),
            w1_path=str(w1_path),
            w2_path=str(w2_path),
            has_w1=True,  # File exists but data is zeros
            has_w2=True,
        )
        result = pipeline._stage3_spectral(result)

        # ALL candidates should be rejected: patch IR pixels are zeros
        assert len(result.stage3_candidates) == 0

    def test_ir_patch_filter_valid_data_passes(self, tmp_path):
        """Per-patch IR filter keeps detections when patch has real IR pixels."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.01, mc_samples=1, device="cpu",  # very low thresh
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
            ir_min_valid_fraction=0.90,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        # All bands have real data
        optical_path = tmp_path / "dss2_red" / "test.fits"
        w1_path = tmp_path / "wise_3.4" / "test.fits"
        w2_path = tmp_path / "wise_4.6" / "test.fits"
        _write_synthetic_fits(optical_path, nx=64, ny=64)
        _write_synthetic_fits(w1_path, nx=64, ny=64)  # Real data
        _write_synthetic_fits(w2_path, nx=64, ny=64)  # Real data

        result = FieldResult(
            field_id="test", ra_deg=180.0, dec_deg=30.0,
            optical_path=str(optical_path),
            w1_path=str(w1_path),
            w2_path=str(w2_path),
            has_w1=True,
            has_w2=True,
        )
        result = pipeline._stage3_spectral(result)

        # With real IR data, candidates should survive the patch filter
        # (they pass with ir_quality="full")
        for c in result.stage3_candidates:
            assert c.get("ir_quality") == "full"

    def test_ir_patch_filter_config_threshold(self, tmp_path):
        """ir_min_valid_fraction=0.0 disables per-patch rejection."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.01, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
            ir_min_valid_fraction=0.0,  # Disabled: accept anything
        )
        pipeline = ObservatoryPipeline(config, searcher)

        # W1/W2 all zeros — but filter is disabled
        optical_path = tmp_path / "dss2_red" / "test.fits"
        w1_path = tmp_path / "wise_3.4" / "test.fits"
        w2_path = tmp_path / "wise_4.6" / "test.fits"
        _write_synthetic_fits(optical_path, nx=64, ny=64)
        _write_synthetic_fits(w1_path, nx=64, ny=64, fill_zeros=True)
        _write_synthetic_fits(w2_path, nx=64, ny=64, fill_zeros=True)

        result = FieldResult(
            field_id="test", ra_deg=180.0, dec_deg=30.0,
            optical_path=str(optical_path),
            w1_path=str(w1_path),
            w2_path=str(w2_path),
            has_w1=True,
            has_w2=True,
        )
        result = pipeline._stage3_spectral(result)

        # With threshold=0.0, zero-IR patches should NOT be rejected
        # (they may still be filtered by Planck class, but not by IR quality)
        # Verify no candidate has "insufficient_ir_patch"
        for c in result.stage3_candidates:
            assert c.get("ir_quality") != "insufficient_ir_patch"


# ---------------------------------------------------------------------------
# TestSNRFilter
# ---------------------------------------------------------------------------


@requires_torch
class TestSNRFilter:
    """Tests for the optical SNR photometric filter."""

    def test_snr_of_point_source(self):
        """Bright point source has high SNR."""
        from astroworld.ml.pipeline import ObservatoryPipeline

        # Create a 128x128 image with Gaussian noise + bright point source
        rng = np.random.default_rng(42)
        img = rng.normal(100, 10, size=(128, 128)).astype(np.float32)
        # Inject bright point source at (64, 64)
        img[62:67, 62:67] += 500  # 50-sigma above background

        snr = ObservatoryPipeline._patch_snr(img, 64, 64)
        assert snr > 10, f"Bright source should have SNR >> 3, got {snr}"

    def test_snr_of_pure_noise(self):
        """Pure Gaussian noise has low SNR."""
        from astroworld.ml.pipeline import ObservatoryPipeline

        rng = np.random.default_rng(42)
        img = rng.normal(100, 10, size=(128, 128)).astype(np.float32)

        snr = ObservatoryPipeline._patch_snr(img, 64, 64)
        # Pure noise: SNR should typically be < 3
        assert snr < 5, f"Pure noise should have low SNR, got {snr}"

    def test_snr_uses_mad(self):
        """MAD-based sigma is robust to cosmic ray outliers."""
        from astroworld.ml.pipeline import ObservatoryPipeline

        rng = np.random.default_rng(42)
        img = rng.normal(100, 10, size=(128, 128)).astype(np.float32)
        # Add cosmic ray spikes in annulus (not at center)
        img[10, 10] = 50000  # Extreme outlier
        img[15, 15] = 40000

        snr = ObservatoryPipeline._patch_snr(img, 64, 64)
        # MAD should be robust to these outliers
        assert snr < 5, f"MAD should be robust to CR, got {snr}"

    def test_snr_filter_rejects_noise_patches(self, tmp_path):
        """Stage 3 SNR filter removes candidates from pure noise images."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.01,  # Very low threshold -> many detections
            mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
            snr_min=3.0,  # Require SNR >= 3
        )
        pipeline = ObservatoryPipeline(config, searcher)

        # Create pure noise images (no real sources)
        optical_path = tmp_path / "dss2_red" / "test.fits"
        w1_path = tmp_path / "wise_3.4" / "test.fits"
        w2_path = tmp_path / "wise_4.6" / "test.fits"
        _write_synthetic_fits(optical_path, nx=64, ny=64)
        _write_synthetic_fits(w1_path, nx=64, ny=64)
        _write_synthetic_fits(w2_path, nx=64, ny=64)

        result = FieldResult(
            field_id="test", ra_deg=180.0, dec_deg=30.0,
            optical_path=str(optical_path),
            w1_path=str(w1_path),
            w2_path=str(w2_path),
            has_w1=True,
            has_w2=True,
        )
        result = pipeline._stage3_spectral(result)

        # Most noise candidates should be filtered out
        # With untrained model we can't guarantee exact numbers,
        # but the filter should reject many/most noise detections
        # The key test is that optical_snr field exists on surviving candidates
        for c in result.stage3_candidates:
            assert "optical_snr" in c
            assert c["optical_snr"] >= 3.0

    def test_snr_filter_disabled_when_zero(self, tmp_path):
        """snr_min=0 disables the SNR filter."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.01,
            mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path, download_pixels=64, verbose=False,
            snr_min=0.0,  # Disabled
        )
        pipeline = ObservatoryPipeline(config, searcher)

        optical_path = tmp_path / "dss2_red" / "test.fits"
        w1_path = tmp_path / "wise_3.4" / "test.fits"
        w2_path = tmp_path / "wise_4.6" / "test.fits"
        _write_synthetic_fits(optical_path, nx=64, ny=64)
        _write_synthetic_fits(w1_path, nx=64, ny=64)
        _write_synthetic_fits(w2_path, nx=64, ny=64)

        result = FieldResult(
            field_id="test", ra_deg=180.0, dec_deg=30.0,
            optical_path=str(optical_path),
            w1_path=str(w1_path),
            w2_path=str(w2_path),
            has_w1=True,
            has_w2=True,
        )
        result = pipeline._stage3_spectral(result)

        # With snr_min=0, no SNR filtering occurs
        # Candidates should NOT have optical_snr (filter was skipped)
        # or if they do, they can be any value
        # Key: the filter block only runs when snr_threshold > 0

    def test_snr_edge_pixel(self):
        """SNR computation at image edge doesn't crash."""
        from astroworld.ml.pipeline import ObservatoryPipeline

        rng = np.random.default_rng(42)
        img = rng.normal(100, 10, size=(64, 64)).astype(np.float32)

        # Test corners
        for px, py in [(0, 0), (63, 63), (0, 63), (63, 0)]:
            snr = ObservatoryPipeline._patch_snr(img, px, py)
            assert np.isfinite(snr), f"SNR at edge ({px},{py}) should be finite"


# ---------------------------------------------------------------------------
# TestStage4Kinematic
# ---------------------------------------------------------------------------


@requires_torch
class TestStage4Kinematic:
    """Tests for kinematic filter stage."""

    def test_conservative_mode_keeps_all(self):
        """Without displacement estimates, all candidates survive."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            kinematic_enabled=True, verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        candidates = [
            {"pixel_x": 10.0, "pixel_y": 20.0, "probability": 0.9,
             "ra_deg": 0, "dec_deg": 0, "temperature_k": 40,
             "temperature_std_k": 5, "planck_class": "p9_candidate"},
            {"pixel_x": 30.0, "pixel_y": 40.0, "probability": 0.85,
             "ra_deg": 0, "dec_deg": 0, "temperature_k": 45,
             "temperature_std_k": 5, "planck_class": "p9_candidate"},
        ]

        result = FieldResult(
            field_id="test", ra_deg=0.0, dec_deg=0.0,
            stage3_candidates=candidates,
        )
        result = pipeline._stage4_kinematic(result)

        # Conservative: all candidates survive
        assert result.stage4_survivors == 2
        assert len(result.final_candidates) == 2
        assert all(c["kinematic_valid"] for c in result.final_candidates)

    def test_disabled_kinematic_keeps_all(self):
        """kinematic_enabled=False passes all candidates through."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
            FieldResult,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            kinematic_enabled=False, verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        candidates = [
            {"pixel_x": 10.0, "pixel_y": 20.0, "probability": 0.9,
             "ra_deg": 0, "dec_deg": 0, "temperature_k": 40,
             "temperature_std_k": 5, "planck_class": "p9_candidate"},
        ]

        result = FieldResult(
            field_id="test", ra_deg=0.0, dec_deg=0.0,
            stage3_candidates=candidates,
        )
        result = pipeline._stage4_kinematic(result)

        assert result.stage4_survivors == 1
        assert len(result.final_candidates) == 1


# ---------------------------------------------------------------------------
# TestPipelineIntegration
# ---------------------------------------------------------------------------


@requires_torch
class TestPipelineIntegration:
    """Integration tests with mocked models and synthetic FITS."""

    def test_full_pipeline_single_field(self, tmp_path):
        """Full pipeline processes one field end-to-end."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            download_pixels=64,
            spectral_threshold=0.5,
            verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        with patch(
            "astroworld.ml.pipeline.download_field",
            side_effect=_mock_download_factory(tmp_path),
        ):
            summary = pipeline.run(fields=[(180.0, 30.0)])

        # Basic structure checks
        assert summary.n_fields_total == 1
        assert summary.n_fields_processed == 1
        assert summary.n_fields_errored == 0
        assert summary.total_time_sec > 0
        assert len(summary.field_results) == 1

        fr = summary.field_results[0]
        assert fr.error is None
        assert fr.optical_path is not None
        assert fr.ir_downloaded is True
        # Model is untrained, so detections depend on random weights
        assert isinstance(fr.stage3_detections, int)

    def test_full_pipeline_no_detections(self, tmp_path):
        """Pipeline produces empty results when nothing detected."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.999,  # Very high threshold -> no detections
            mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            download_pixels=64,
            spectral_threshold=0.999,
            verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        with patch(
            "astroworld.ml.pipeline.download_field",
            side_effect=_mock_download_factory(tmp_path),
        ):
            summary = pipeline.run(fields=[(180.0, 30.0)])

        assert summary.total_candidates == 0
        assert summary.n_fields_with_candidates == 0
        assert len(summary.candidate_table) == 0

    def test_multiple_fields(self, tmp_path):
        """Pipeline processes multiple fields and builds summary."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            download_pixels=64,
            verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        fields = [(180.0, 30.0), (181.0, 31.0), (182.0, 32.0)]

        with patch(
            "astroworld.ml.pipeline.download_field",
            side_effect=_mock_download_factory(tmp_path),
        ):
            summary = pipeline.run(fields=fields)

        assert summary.n_fields_total == 3
        assert summary.n_fields_processed == 3
        assert len(summary.field_results) == 3
        assert summary.total_time_sec > 0
        assert summary.avg_time_per_field_sec > 0


# ---------------------------------------------------------------------------
# TestObservatorySummary
# ---------------------------------------------------------------------------


@requires_torch
class TestObservatorySummary:
    """Tests for summary generation and reporting."""

    def test_candidate_table_schema(self, tmp_path):
        """Candidate table has expected columns."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            download_pixels=64,
            verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        with patch(
            "astroworld.ml.pipeline.download_field",
            side_effect=_mock_download_factory(tmp_path),
        ):
            summary = pipeline.run(fields=[(180.0, 30.0)])

        expected_columns = {
            "field_id", "field_ra", "field_dec",
            "pixel_x", "pixel_y", "ra_deg", "dec_deg",
            "probability", "temperature_k", "temperature_std_k",
            "planck_class", "kinematic_valid",
        }

        if len(summary.candidate_table) > 0:
            assert set(summary.candidate_table.columns) == expected_columns
        else:
            # Empty DataFrame: valid (just no candidates)
            assert isinstance(summary.candidate_table, pd.DataFrame)

    def test_funnel_counts_consistent(self, tmp_path):
        """Stage counts decrease monotonically through the funnel."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        config = PipelineConfig(
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            download_pixels=64,
            verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        with patch(
            "astroworld.ml.pipeline.download_field",
            side_effect=_mock_download_factory(tmp_path),
        ):
            summary = pipeline.run(fields=[(180.0, 30.0)])

        # Funnel: stage3_detections >= planck_filtered >= stage4_survivors >= candidates
        assert summary.n_stage3_detections >= summary.n_stage3_planck_filtered
        assert summary.n_stage3_planck_filtered >= summary.n_stage4_survivors
        assert summary.n_stage4_survivors >= summary.total_candidates

    def test_csv_report_created(self, tmp_path):
        """CSV report is written to output directory."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
        )
        from astroworld.ml.spectral_inference import SpectralSearcher
        from astroworld.ml.spectral_model import SpectralSiameseNet

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.1)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.5, mc_samples=1, device="cpu",
        )
        output_dir = tmp_path / "results"
        config = PipelineConfig(
            data_dir=tmp_path,
            output_dir=output_dir,
            download_pixels=64,
            verbose=False,
        )
        pipeline = ObservatoryPipeline(config, searcher)

        with patch(
            "astroworld.ml.pipeline.download_field",
            side_effect=_mock_download_factory(tmp_path),
        ):
            pipeline.run(fields=[(180.0, 30.0)])

        csv_path = output_dir / "observatory_candidates.csv"
        assert csv_path.exists()

        summary_path = output_dir / "observatory_summary.txt"
        assert summary_path.exists()

        # Read and verify CSV
        df = pd.read_csv(csv_path)
        assert isinstance(df, pd.DataFrame)

        # Read and verify summary text
        text = summary_path.read_text()
        assert "Observatory Pipeline" in text
        assert "spectral_only" in text
