"""
Tests for Phase 12: SSTF (Siamese Spectral-Temporal Fusion).

All tests are offline (no network, no GPU) using synthetic data.
Tests cover:
  - Spectral cube assembly and physics (Planck, temperature)
  - SpectralEncoder, CrossAttentionFusion, SpectralSiameseNet
  - MC Dropout uncertainty estimation
  - PlanckFilter classification
  - MultiSurveyDataset loading and augmentation
  - SpectralSearcher tiling and inference
  - SpectralTrainer combined loss
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Spectral cube (no torch required)
from astroworld.imaging.spectral_cube import (
    BAND_OPTICAL,
    BAND_W1,
    BAND_W2,
    BAND_UNCERTAINTY,
    BAND_TEMPORAL,
    NUM_CHANNELS,
    build_spectral_cube,
    build_spectral_cube_cached,
    compute_uncertainty_map,
    compute_temporal_gradient,
    estimate_color_temperature,
    planck_flux_ratio,
    planck_spectral_radiance,
)

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


def _make_noisy_image(
    shape: tuple[int, int] = (64, 64),
    bkg_mean: float = 100.0,
    bkg_std: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """Create a noisy test image."""
    rng = np.random.default_rng(seed)
    return rng.normal(bkg_mean, bkg_std, shape).astype(np.float32)


def _make_noisy_image_with_source(
    shape: tuple[int, int] = (64, 64),
    bkg_mean: float = 100.0,
    bkg_std: float = 10.0,
    source_flux: float = 500.0,
    source_pos: tuple[int, int] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Create a noisy image with a Gaussian point source."""
    rng = np.random.default_rng(seed)
    image = rng.normal(bkg_mean, bkg_std, shape).astype(np.float32)

    if source_pos is None:
        source_pos = (shape[0] // 2, shape[1] // 2)

    cy, cx = source_pos
    y, x = np.ogrid[:shape[0], :shape[1]]
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    psf = np.exp(-r2 / (2.0 * 2.5 ** 2))
    image += (source_flux * psf).astype(np.float32)

    return image


# ===========================================================================
# Tests: Spectral Cube Assembly
# ===========================================================================

class TestBuildSpectralCube:
    """Tests for build_spectral_cube()."""

    def test_output_shape(self):
        """Cube has shape (5, H, W)."""
        optical = _make_noisy_image((64, 64))
        header = _make_wcs_header(nx=64, ny=64)
        cube = build_spectral_cube(optical, header)
        assert cube.shape == (5, 64, 64)

    def test_optical_only(self):
        """Without W1/W2, IR channels are zeros."""
        optical = _make_noisy_image((64, 64))
        header = _make_wcs_header()
        cube = build_spectral_cube(optical, header)

        # Optical channel has data
        assert cube[BAND_OPTICAL].std() > 0
        # W1, W2 are zeros
        assert cube[BAND_W1].sum() == 0
        assert cube[BAND_W2].sum() == 0
        # Uncertainty has data
        assert cube[BAND_UNCERTAINTY].sum() > 0
        # Temporal is zeros (placeholder)
        assert cube[BAND_TEMPORAL].sum() == 0

    def test_with_ir_channels(self):
        """W1/W2 images are placed in correct channels."""
        optical = _make_noisy_image((64, 64), seed=1)
        w1 = _make_noisy_image((32, 32), seed=2)  # Different resolution
        w2 = _make_noisy_image((32, 32), seed=3)
        header = _make_wcs_header()
        w1_header = _make_wcs_header(pixel_scale_arcsec=2.75, nx=32, ny=32)
        w2_header = _make_wcs_header(pixel_scale_arcsec=2.75, nx=32, ny=32)

        cube = build_spectral_cube(
            optical, header, w1, w1_header, w2, w2_header,
        )
        assert cube.shape == (5, 64, 64)
        # W1/W2 should have non-zero data (from reprojection or resize)
        assert cube[BAND_W1].std() > 0
        assert cube[BAND_W2].std() > 0

    def test_custom_target_shape(self):
        """Target shape overrides optical shape."""
        optical = _make_noisy_image((128, 128))
        header = _make_wcs_header(nx=128, ny=128)
        cube = build_spectral_cube(optical, header, target_shape=(64, 64))
        assert cube.shape == (5, 64, 64)

    def test_cached_cube(self):
        """Cached cube loads from disk on second call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optical = _make_noisy_image((32, 32))
            header = _make_wcs_header(nx=32, ny=32)
            cache_dir = Path(tmpdir) / "cache"

            cube1 = build_spectral_cube_cached(
                optical, header, cache_dir=cache_dir, cache_key="test",
            )
            # Should create cache file
            cache_file = cache_dir / "cube_test.npy"
            assert cache_file.exists()

            # Second call loads from cache
            cube2 = build_spectral_cube_cached(
                optical, header, cache_dir=cache_dir, cache_key="test",
            )
            np.testing.assert_array_equal(cube1, cube2)


class TestUncertaintyMap:
    """Tests for compute_uncertainty_map()."""

    def test_positive_values(self):
        """All uncertainty values are >= 0."""
        image = _make_noisy_image((64, 64))
        unc = compute_uncertainty_map(image)
        assert (unc >= 0).all()

    def test_shape_matches(self):
        """Output shape matches input."""
        image = _make_noisy_image((100, 80))
        unc = compute_uncertainty_map(image)
        assert unc.shape == (100, 80)

    def test_high_noise_low_weight(self):
        """Higher noise regions get lower inverse-variance weight."""
        # Low noise
        low_noise = _make_noisy_image((64, 64), bkg_std=1.0)
        unc_low = compute_uncertainty_map(low_noise)

        # High noise
        high_noise = _make_noisy_image((64, 64), bkg_std=100.0)
        unc_high = compute_uncertainty_map(high_noise)

        # Low noise should have higher weight (inverse variance)
        assert unc_low.mean() > unc_high.mean()


class TestTemporalGradient:
    """Tests for compute_temporal_gradient()."""

    def test_shape_2d(self):
        """Works with 2D images."""
        img1 = _make_noisy_image((64, 64), seed=1)
        img2 = _make_noisy_image((64, 64), seed=2)
        grad = compute_temporal_gradient(img1, img2, delta_t_days=30.0)
        assert grad.shape == (64, 64)

    def test_shape_3d(self):
        """Works with 3D cubes (uses optical channel)."""
        cube1 = np.random.randn(5, 64, 64).astype(np.float32)
        cube2 = np.random.randn(5, 64, 64).astype(np.float32)
        grad = compute_temporal_gradient(cube1, cube2, delta_t_days=30.0)
        assert grad.shape == (64, 64)

    def test_normalized(self):
        """Output is approximately zero-mean, unit-std."""
        img1 = _make_noisy_image((64, 64), seed=1)
        img2 = _make_noisy_image((64, 64), seed=2)
        grad = compute_temporal_gradient(img1, img2, delta_t_days=30.0)
        assert abs(grad.mean()) < 0.5
        assert 0.5 < grad.std() < 1.5

    def test_zero_delta_t_raises(self):
        """delta_t_days <= 0 raises ValueError."""
        img = _make_noisy_image((32, 32))
        with pytest.raises(ValueError):
            compute_temporal_gradient(img, img, delta_t_days=0.0)


# ===========================================================================
# Tests: Planck Physics
# ===========================================================================

class TestPlanckPhysics:
    """Tests for Planck blackbody functions."""

    def test_planck_ratio_monotonic(self):
        """W1/W2 flux ratio increases monotonically with temperature."""
        temps = [100, 500, 1000, 5000, 10000]
        ratios = [planck_flux_ratio(t) for t in temps]
        for i in range(len(ratios) - 1):
            assert ratios[i + 1] > ratios[i], \
                f"Ratio not monotonic at T={temps[i+1]}"

    def test_solar_temperature(self):
        """Estimate ~5778K from solar W1/W2 flux ratio."""
        # Compute expected ratio for the Sun
        ratio_sun = planck_flux_ratio(5778.0)
        # Invert back
        t_est = estimate_color_temperature(ratio_sun, 1.0)
        assert abs(t_est - 5778.0) < 200.0, \
            f"Solar temperature estimate {t_est:.0f}K, expected ~5778K"

    def test_cold_body_temperature(self):
        """Estimate ~30K for a cold body (Sedna-like)."""
        ratio_cold = planck_flux_ratio(30.0)
        t_est = estimate_color_temperature(ratio_cold, 1.0)
        assert abs(t_est - 30.0) < 5.0

    def test_planck_radiance_positive(self):
        """Planck radiance is positive for valid T."""
        assert planck_spectral_radiance(3.4, 300.0) > 0
        assert planck_spectral_radiance(4.6, 300.0) > 0

    def test_planck_radiance_zero_temp(self):
        """Zero temperature gives zero radiance."""
        assert planck_spectral_radiance(3.4, 0.0) == 0.0

    def test_invalid_flux_raises(self):
        """Non-positive fluxes raise ValueError."""
        with pytest.raises(ValueError):
            estimate_color_temperature(-1.0, 1.0)
        with pytest.raises(ValueError):
            estimate_color_temperature(1.0, 0.0)


# ===========================================================================
# Tests: SpectralModel (requires torch)
# ===========================================================================

@requires_torch
class TestSpectralEncoder:
    """Tests for SpectralEncoder CNN backbone."""

    def test_output_shape(self):
        """Encoder outputs (B, 256) for (B, 5, 64, 64) input."""
        from astroworld.ml.spectral_model import SpectralEncoder
        encoder = SpectralEncoder(in_channels=5, feature_dim=256)
        x = torch.randn(2, 5, 64, 64)
        out = encoder(x)
        assert out.shape == (2, 256)

    def test_small_input(self):
        """Works with smaller inputs (AdaptiveAvgPool handles it)."""
        from astroworld.ml.spectral_model import SpectralEncoder
        encoder = SpectralEncoder(in_channels=5, feature_dim=128)
        x = torch.randn(1, 5, 32, 32)
        out = encoder(x)
        assert out.shape == (1, 128)

    def test_param_count(self):
        """Encoder has reasonable parameter count (~400K)."""
        from astroworld.ml.spectral_model import SpectralEncoder
        encoder = SpectralEncoder()
        n_params = encoder.num_parameters
        assert 200_000 < n_params < 1_200_000, \
            f"Unexpected param count: {n_params:,}"


@requires_torch
class TestCrossAttention:
    """Tests for CrossAttentionFusion."""

    def test_output_shape(self):
        """Output is (B, 4*embed_dim)."""
        from astroworld.ml.spectral_model import CrossAttentionFusion
        ca = CrossAttentionFusion(embed_dim=128, num_heads=4)
        f1 = torch.randn(4, 128)
        f2 = torch.randn(4, 128)
        out = ca(f1, f2)
        assert out.shape == (4, 512)  # 4 * 128

    def test_output_dim_property(self):
        """output_dim returns correct value."""
        from astroworld.ml.spectral_model import CrossAttentionFusion
        ca = CrossAttentionFusion(embed_dim=256, num_heads=4)
        assert ca.output_dim == 1024


@requires_torch
class TestSpectralSiameseNet:
    """Tests for the full SpectralSiameseNet."""

    def test_forward_shapes(self):
        """Forward returns (motion_logits, temp_log10k) with correct shapes."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        model = SpectralSiameseNet(feature_dim=128, num_heads=4)
        x1 = torch.randn(3, 5, 64, 64)
        x2 = torch.randn(3, 5, 64, 64)
        motion, temp = model(x1, x2)
        assert motion.shape == (3, 1)
        assert temp.shape == (3, 1)

    def test_predict_proba_range(self):
        """predict_proba returns values in [0, 1]."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        model = SpectralSiameseNet(feature_dim=64, num_heads=2)
        model.eval()
        x1 = torch.randn(5, 5, 32, 32)
        x2 = torch.randn(5, 5, 32, 32)
        probs = model.predict_proba(x1, x2)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_mc_dropout_variance(self):
        """MC Dropout produces non-zero variance."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.5)
        x1 = torch.randn(2, 5, 32, 32)
        x2 = torch.randn(2, 5, 32, 32)
        result = model.predict_with_uncertainty(x1, x2, n_samples=20)
        assert "motion_prob_std" in result
        assert "temp_std" in result
        # With dropout=0.5 and 20 samples, there should be measurable variance
        assert result["motion_prob_std"].mean() > 0

    def test_mc_dropout_restores_eval(self):
        """predict_with_uncertainty restores eval mode."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        model = SpectralSiameseNet(feature_dim=64, num_heads=2)
        model.eval()
        x = torch.randn(1, 5, 32, 32)
        model.predict_with_uncertainty(x, x, n_samples=3)
        assert not model.training, "Model should be in eval mode after MC"

    def test_checkpoint_roundtrip(self):
        """Model can be saved and loaded from checkpoint."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        model = SpectralSiameseNet(feature_dim=64, num_heads=2)
        model.eval()  # Set eval mode before first prediction (BatchNorm)
        x1 = torch.randn(1, 5, 32, 32)
        x2 = torch.randn(1, 5, 32, 32)

        with torch.no_grad():
            out_before = model.predict_proba(x1, x2)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": model.state_dict()}, f.name)

            # Load into new model
            model2 = SpectralSiameseNet(feature_dim=64, num_heads=2)
            ckpt = torch.load(f.name, weights_only=True)
            model2.load_state_dict(ckpt["model_state_dict"])
            model2.eval()

            with torch.no_grad():
                out_after = model2.predict_proba(x1, x2)

        np.testing.assert_allclose(
            out_before.numpy(), out_after.numpy(), atol=1e-6,
        )

    def test_num_parameters(self):
        """Total parameters are reasonable (~1-2M)."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        model = SpectralSiameseNet(feature_dim=256, num_heads=4)
        n = model.num_parameters
        assert 800_000 < n < 3_000_000, f"Unexpected param count: {n:,}"


# ===========================================================================
# Tests: PlanckFilter
# ===========================================================================

@requires_torch
class TestPlanckFilter:
    """Tests for PlanckFilter classification."""

    def test_classify_cold(self):
        """T=30K classified as p9_candidate."""
        from astroworld.ml.spectral_model import PlanckFilter
        assert PlanckFilter.classify(30.0) == "p9_candidate"

    def test_classify_warm(self):
        """T=1000K classified as warm_reject."""
        from astroworld.ml.spectral_model import PlanckFilter
        assert PlanckFilter.classify(1000.0) == "warm_reject"

    def test_classify_cold_object(self):
        """T=200K classified as cold_object."""
        from astroworld.ml.spectral_model import PlanckFilter
        assert PlanckFilter.classify(200.0) == "cold_object"

    def test_classify_uncertain(self):
        """High uncertainty → uncertain."""
        from astroworld.ml.spectral_model import PlanckFilter
        # temp_std > 0.5 * temp_k → uncertain
        assert PlanckFilter.classify(100.0, temp_std_k=60.0) == "uncertain"

    def test_expected_p9_temp_480au(self):
        """T(480 AU) ~ 12.7K (radiative equilibrium)."""
        from astroworld.ml.spectral_model import PlanckFilter
        t = PlanckFilter.expected_p9_temperature(480.0)
        assert 5.0 < t < 20.0, f"T(480 AU) = {t:.1f} K"

    def test_expected_temp_1au(self):
        """T(1 AU) ~ 278K (Earth equilibrium)."""
        from astroworld.ml.spectral_model import PlanckFilter
        t = PlanckFilter.expected_p9_temperature(1.0)
        assert 250 < t < 300, f"T(1 AU) = {t:.1f} K"


# ===========================================================================
# Tests: SpectralSearcher (requires torch)
# ===========================================================================

@requires_torch
class TestSpectralSearcher:
    """Tests for SpectralSearcher sliding-window inference."""

    def test_tile_spectral_cube(self):
        """Tiles a 5-channel cube into (N, 5, ps, ps) patches."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        from astroworld.ml.spectral_inference import SpectralSearcher

        model = SpectralSiameseNet(feature_dim=64, num_heads=2)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0, threshold=0.5,
            mc_samples=1, device="cpu",
        )

        cube = np.random.randn(5, 64, 64).astype(np.float32)
        _, _, y_origins, x_origins = searcher._compute_grid_shape(64, 64)
        patches, origins = searcher._tile_spectral_cube(
            cube, y_origins, x_origins,
        )
        assert patches.shape[1] == 5  # 5 channels
        assert patches.shape[2] == 32  # patch_size
        assert patches.shape[3] == 32
        assert len(origins) == patches.shape[0]

    def test_search_returns_spectral_result(self):
        """search_from_cubes returns SpectralSearchResult."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        from astroworld.ml.spectral_inference import (
            SpectralSearcher,
            SpectralSearchResult,
        )

        model = SpectralSiameseNet(feature_dim=64, num_heads=2)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0, threshold=0.5,
            mc_samples=1, device="cpu",
        )

        cube1 = np.random.randn(5, 64, 64).astype(np.float32)
        cube2 = np.random.randn(5, 64, 64).astype(np.float32)
        result = searcher.search_from_cubes(cube1, cube2)

        assert isinstance(result, SpectralSearchResult)
        assert result.temperature_map.shape == result.grid_shape
        assert result.motion_std_map.shape == result.grid_shape

    def test_mc_dropout_produces_uncertainty(self):
        """MC samples > 1 produces non-trivial uncertainty maps."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        from astroworld.ml.spectral_inference import SpectralSearcher

        model = SpectralSiameseNet(feature_dim=64, num_heads=2, dropout=0.5)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0, threshold=0.5,
            mc_samples=10, device="cpu",
        )

        cube1 = np.random.randn(5, 64, 64).astype(np.float32)
        cube2 = np.random.randn(5, 64, 64).astype(np.float32)
        result = searcher.search_from_cubes(cube1, cube2)

        # With dropout=0.5 and 10 samples, std should be non-zero
        assert result.motion_std_map.max() > 0

    def test_planck_classification_on_detections(self):
        """Detections get Planck classification strings."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        from astroworld.ml.spectral_inference import SpectralSearcher

        model = SpectralSiameseNet(feature_dim=64, num_heads=2)
        searcher = SpectralSearcher(
            model, patch_size=32, overlap=0,
            threshold=0.01,  # Very low to guarantee detections
            mc_samples=1, device="cpu",
        )

        cube1 = np.random.randn(5, 64, 64).astype(np.float32)
        cube2 = np.random.randn(5, 64, 64).astype(np.float32)
        result = searcher.search_from_cubes(cube1, cube2)

        # With threshold=0.01, we should get some detections
        if len(result.planck_classifications) > 0:
            for cls in result.planck_classifications:
                assert cls in {
                    "p9_candidate", "cold_object", "warm_reject", "uncertain"
                }


# ===========================================================================
# Tests: SpectralTrainer combined loss (requires torch)
# ===========================================================================

@requires_torch
class TestSpectralTrainerLoss:
    """Tests for the dual-head combined loss."""

    def test_combined_loss_components(self):
        """Loss is alpha * BCE + beta * MSE(temp) * mask."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        from astroworld.ml.trainer import TrainingConfig

        # Import the trainer
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_spectral import SpectralTrainer

        model = SpectralSiameseNet(feature_dim=64, num_heads=2)
        config = TrainingConfig(epochs=1, device="cpu")
        trainer = SpectralTrainer(model, config, alpha=1.0, beta=0.1)

        # Create fake batch
        motion_logits = torch.randn(4, 1)
        motion_labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        temp_pred = torch.randn(4, 1)
        temp_labels = torch.tensor([
            [np.log10(30.0)],  # positive: valid temp
            [float("nan")],     # negative: NaN
            [np.log10(50.0)],  # positive: valid temp
            [float("nan")],     # negative: NaN
        ])

        import torch.nn as nn
        motion_criterion = nn.BCEWithLogitsLoss()
        temp_criterion = nn.MSELoss(reduction="none")

        loss = trainer._compute_combined_loss(
            motion_logits, motion_labels,
            temp_pred, temp_labels,
            motion_criterion, temp_criterion,
        )
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_temp_loss_masked_for_negatives(self):
        """Temperature loss is zero when all labels are NaN."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        from astroworld.ml.trainer import TrainingConfig

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_spectral import SpectralTrainer

        model = SpectralSiameseNet(feature_dim=64, num_heads=2)
        config = TrainingConfig(epochs=1, device="cpu")
        trainer = SpectralTrainer(model, config, alpha=1.0, beta=0.5)

        motion_logits = torch.zeros(2, 1)
        motion_labels = torch.zeros(2, 1)
        temp_pred = torch.randn(2, 1)
        # All NaN → no temperature loss
        temp_labels = torch.tensor([[float("nan")], [float("nan")]])

        import torch.nn as nn
        motion_criterion = nn.BCEWithLogitsLoss()
        temp_criterion = nn.MSELoss(reduction="none")

        # Loss should be pure BCE (no temp component)
        loss_with_nan = trainer._compute_combined_loss(
            motion_logits, motion_labels,
            temp_pred, temp_labels,
            motion_criterion, temp_criterion,
        )
        # Compare with pure motion loss
        pure_motion = motion_criterion(motion_logits, motion_labels)
        assert abs(loss_with_nan.item() - pure_motion.item()) < 1e-5


# ===========================================================================
# Tests: normalize_spectral_cube
# ===========================================================================

@requires_torch
class TestNormalizeSpectralCube:
    """Tests for per-channel normalization."""

    def test_normalize_shape(self):
        """Output shape matches input."""
        from astroworld.ml.spectral_dataset import normalize_spectral_cube
        cube = np.random.randn(5, 64, 64).astype(np.float32) * 100 + 500
        norm = normalize_spectral_cube(cube)
        assert norm.shape == (5, 64, 64)

    def test_normalize_per_channel(self):
        """Each channel is independently normalized to ~N(0,1)."""
        from astroworld.ml.spectral_dataset import normalize_spectral_cube
        rng = np.random.default_rng(42)
        cube = np.zeros((5, 64, 64), dtype=np.float32)
        for ch in range(5):
            cube[ch] = rng.normal(ch * 100 + 50, ch * 10 + 5, (64, 64))

        norm = normalize_spectral_cube(cube)
        for ch in range(5):
            assert abs(norm[ch].mean()) < 0.5, \
                f"Channel {ch} mean={norm[ch].mean():.2f}"
            assert 0.5 < norm[ch].std() < 2.0, \
                f"Channel {ch} std={norm[ch].std():.2f}"


# ===========================================================================
# Tests: Spectral Training Data Generator
# ===========================================================================

class TestSpectralDataGenerator:
    """Tests for generate_spectral_training_data.py functions."""

    def test_ir_flux_ratio_matches_planck(self):
        """W1/W2 flux ratio should approximate planck_flux_ratio(T) for warm bodies.

        At very low temperatures (T < 100K), the Planck ratio W1/W2 is
        effectively zero (both bands are far from the emission peak).
        We test at temperatures where the ratio is physically meaningful.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from generate_spectral_training_data import compute_ir_fluxes

        rng = np.random.default_rng(42)

        # Test at temperatures where Planck ratio is well-defined (> ~200K)
        for temp in [300.0, 1000.0, 3000.0, 5000.0]:
            flux_w1, flux_w2 = compute_ir_fluxes(
                optical_flux=1000.0, temperature_k=temp,
                ir_boost=5.0, rng=rng, color_noise_frac=0.0,
            )
            actual_ratio = flux_w1 / flux_w2 if flux_w2 > 0 else 0
            expected_ratio = planck_flux_ratio(temp)
            assert abs(actual_ratio - expected_ratio) < 0.02 * expected_ratio + 1e-6, \
                f"T={temp}K: ratio {actual_ratio:.4f} != expected {expected_ratio:.4f}"

    def test_color_noise_adds_variation(self):
        """Color noise produces different ratios across samples.

        Uses T=1000K where the Planck ratio is well-defined (~0.66),
        so the ±5% noise is visible in the W1/W2 ratio spread.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from generate_spectral_training_data import compute_ir_fluxes

        ratios = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            flux_w1, flux_w2 = compute_ir_fluxes(
                optical_flux=1000.0, temperature_k=1000.0,
                ir_boost=5.0, rng=rng, color_noise_frac=0.05,
            )
            ratios.append(flux_w1 / flux_w2)

        # With 5% noise at T=1000K (ratio ~0.66), std should be ~0.03
        assert np.std(ratios) > 0.001, \
            f"Color noise should produce variation, got std={np.std(ratios):.6f}"

    def test_generate_small_dataset(self):
        """Generate a small dataset and check catalog schema."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from generate_spectral_training_data import generate_spectral_training_data

        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = generate_spectral_training_data(
                output_dir=Path(tmpdir),
                n_positive=5, n_negative=5,
                img_size=32, seed=42,
            )

            # Check catalog columns
            required_cols = [
                "example_id", "optical_path", "w1_path", "w2_path",
                "source_present", "temp_kelvin", "epoch_jd", "delta_t_days",
            ]
            for col in required_cols:
                assert col in catalog.columns, f"Missing column: {col}"

            # Should have 20 rows (5 pos + 5 neg) * 2 epochs
            assert len(catalog) == 20

            # Check FITS files exist
            for _, row in catalog.iterrows():
                opt_path = Path(tmpdir) / row["optical_path"]
                assert opt_path.exists(), f"Missing: {opt_path}"
                w1_path = Path(tmpdir) / row["w1_path"]
                assert w1_path.exists(), f"Missing: {w1_path}"

            # Positive rows have valid temperature
            pos = catalog[catalog["source_present"] == True]
            assert pos["temp_kelvin"].notna().all()

            # Negative rows have NaN temperature
            neg = catalog[catalog["source_present"] == False]
            assert neg["temp_kelvin"].isna().all()

    def test_fits_image_shape(self):
        """Generated FITS files have correct shape."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from generate_spectral_training_data import generate_spectral_training_data
        from astropy.io import fits as test_fits

        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = generate_spectral_training_data(
                output_dir=Path(tmpdir),
                n_positive=2, n_negative=1,
                img_size=48, seed=99,
            )

            # Check first positive optical file
            first_opt = Path(tmpdir) / catalog.iloc[0]["optical_path"]
            with test_fits.open(str(first_opt)) as hdu:
                data = hdu[0].data.copy()  # Copy data before closing
            assert data.shape == (48, 48)


@requires_torch
class TestCheckpointArchParams:
    """Tests for architecture parameter storage in checkpoints."""

    def test_checkpoint_stores_arch_params(self):
        """Checkpoint contains feature_dim, num_heads, dropout."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        from astroworld.ml.trainer import TrainingConfig

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_spectral import SpectralTrainer

        model = SpectralSiameseNet(feature_dim=128, num_heads=2, dropout=0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                epochs=1, device="cpu", checkpoint_dir=tmpdir,
            )
            trainer = SpectralTrainer(model, config)
            optimizer = torch.optim.Adam(model.parameters())

            ckpt_path = trainer._save_checkpoint(1, 0.5, optimizer)

            ckpt = torch.load(ckpt_path, weights_only=True)
            assert ckpt["feature_dim"] == 128
            assert ckpt["num_heads"] == 2
            assert abs(ckpt["dropout"] - 0.3) < 1e-6

    def test_from_checkpoint_reads_arch_params(self):
        """from_checkpoint() uses architecture params from checkpoint."""
        from astroworld.ml.spectral_model import SpectralSiameseNet
        from astroworld.ml.spectral_inference import SpectralSearcher

        model = SpectralSiameseNet(feature_dim=128, num_heads=2, dropout=0.3)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": "SpectralSiameseNet",
                "feature_dim": 128,
                "num_heads": 2,
                "dropout": 0.3,
            }, f.name)

            # Load without specifying model_kwargs — should use checkpoint values
            searcher = SpectralSearcher.from_checkpoint(
                f.name, patch_size=32, overlap=0,
                threshold=0.5, mc_samples=1, device="cpu",
            )

            # Verify model was built with correct params
            assert searcher.model.encoder.feature_dim == 128
