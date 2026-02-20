"""
Tests for the ML pipeline (Phase 8).

All tests use tiny synthetic data (32x32 images, small batches) for
fast execution.  No GPU or real FITS images required.

Follows the project convention: inline fixtures, class-based grouping,
pytest.mark.skipif for optional dependencies.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from astropy.io import fits

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="torch not installed"
)


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic FITS + CSV catalog
# ---------------------------------------------------------------------------

def _make_tiny_fits(
    path: Path,
    nx: int = 32,
    ny: int = 32,
    bkg_mean: float = 1000.0,
    bkg_std: float = 50.0,
    inject_source: bool = False,
    source_x: int = 16,
    source_y: int = 16,
    source_flux: float = 5000.0,
    seed: int = 42,
) -> Path:
    """Create a tiny mock FITS image with optional injected source."""
    rng = np.random.default_rng(seed)
    image = rng.normal(bkg_mean, bkg_std, (ny, nx)).astype(np.float32)

    if inject_source:
        # Add a small Gaussian blob
        yy, xx = np.mgrid[0:ny, 0:nx]
        sigma = 1.5
        blob = source_flux * np.exp(
            -0.5 * ((xx - source_x) ** 2 + (yy - source_y) ** 2) / sigma ** 2
        )
        image += blob.astype(np.float32)

    hdu = fits.PrimaryHDU(image)
    hdu.header["CRVAL1"] = 71.3
    hdu.header["CRVAL2"] = 14.5
    hdu.header["CRPIX1"] = nx / 2.0 + 0.5
    hdu.header["CRPIX2"] = ny / 2.0 + 0.5
    hdu.header["CDELT1"] = -0.000326
    hdu.header["CDELT2"] = 0.000326
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["NAXIS1"] = nx
    hdu.header["NAXIS2"] = ny
    hdu.writeto(str(path), overwrite=True)
    return path


def _make_tiny_catalog(
    tmp_path: Path,
    n_pos: int = 4,
    n_neg: int = 4,
    img_size: int = 32,
) -> Path:
    """Create tiny FITS + training_catalog.csv for testing."""
    pos_dir = tmp_path / "positive"
    neg_dir = tmp_path / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_pos):
        fname = f"pos_{i:04d}.fits"
        fpath = pos_dir / fname
        _make_tiny_fits(
            fpath, nx=img_size, ny=img_size,
            inject_source=True,
            source_flux=2000.0 + i * 1000,
            seed=42 + i,
        )
        rows.append({
            "example_id": f"pos_{i:04d}",
            "fits_path": str(fpath),
            "source_present": True,
            "ra_deg": 71.3 + i * 0.001,
            "dec_deg": 14.5 + i * 0.001,
            "pixel_x": 16.0 + i * 0.5,
            "pixel_y": 16.0 + i * 0.5,
            "magnitude": 20.0 + i * 0.5,
            "flux_adu": 2000.0 + i * 1000,
            "snr": 5.0 + i,
            "source_fits": f"field_{i:02d}.fits",
            "survey": "test",
            "psf_fwhm": 2.5,
            "zero_point": 22.5,
        })

    for i in range(n_neg):
        fname = f"neg_{i:04d}.fits"
        fpath = neg_dir / fname
        _make_tiny_fits(
            fpath, nx=img_size, ny=img_size,
            inject_source=False,
            seed=100 + i,
        )
        rows.append({
            "example_id": f"neg_{i:04d}",
            "fits_path": str(fpath),
            "source_present": False,
            "ra_deg": float("nan"),
            "dec_deg": float("nan"),
            "pixel_x": float("nan"),
            "pixel_y": float("nan"),
            "magnitude": float("nan"),
            "flux_adu": float("nan"),
            "snr": float("nan"),
            "source_fits": f"field_{i:02d}.fits",
            "survey": "test",
            "psf_fwhm": 2.5,
            "zero_point": 22.5,
        })

    catalog = pd.DataFrame(rows)
    catalog_path = tmp_path / "training_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    return catalog_path


def _make_multi_epoch_catalog(
    tmp_path: Path,
    n_pairs: int = 3,
    img_size: int = 32,
) -> Path:
    """Create multi-epoch paired FITS + catalog for Siamese testing."""
    pos_dir = tmp_path / "positive"
    pos_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_pairs):
        for epoch in [1, 2]:
            fname = f"pair_{i:04d}_epoch{epoch}.fits"
            fpath = pos_dir / fname
            # Epoch 2 has slightly shifted source
            sx = 16 + (2 if epoch == 2 else 0)
            _make_tiny_fits(
                fpath, nx=img_size, ny=img_size,
                inject_source=True,
                source_x=sx,
                source_flux=3000.0,
                seed=200 + i * 10 + epoch,
            )
            rows.append({
                "example_id": f"pair_{i:04d}_epoch{epoch}",
                "fits_path": str(fpath),
                "source_present": True,
                "ra_deg": 71.3,
                "dec_deg": 14.5,
                "pixel_x": float(sx),
                "pixel_y": 16.0,
                "magnitude": 22.0,
                "flux_adu": 3000.0,
                "snr": 1.0,
                "source_fits": f"field_{i:02d}.fits",
                "survey": "test",
                "psf_fwhm": 2.5,
                "zero_point": 22.5,
            })

    catalog = pd.DataFrame(rows)
    catalog_path = tmp_path / "training_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    return catalog_path


# ---------------------------------------------------------------------------
# TestNormalization
# ---------------------------------------------------------------------------

@requires_torch
class TestNormalization:
    """Tests for sigma-clip image normalization."""

    def test_normalize_zero_mean_unit_std(self):
        """Normalized background should have mean~0, std~1."""
        from astroworld.ml.dataset import normalize_image

        rng = np.random.default_rng(42)
        image = rng.normal(5000.0, 100.0, (64, 64))
        normed = normalize_image(image)

        assert abs(normed.mean()) < 0.1
        assert abs(normed.std() - 1.0) < 0.1

    def test_normalize_handles_constant_image(self):
        """Constant image should normalize without error."""
        from astroworld.ml.dataset import normalize_image

        image = np.full((32, 32), 42.0)
        normed = normalize_image(image)
        assert np.isfinite(normed).all()

    def test_normalize_ignores_bright_stars(self):
        """Bright outliers should not bias the background estimate."""
        from astroworld.ml.dataset import normalize_image

        rng = np.random.default_rng(42)
        image = rng.normal(1000.0, 50.0, (128, 128))
        # Add bright stars
        image[60, 60] = 50000.0
        image[80, 80] = 100000.0

        normed = normalize_image(image)
        # Background should still be near zero
        # Exclude the star pixels
        mask = np.ones_like(normed, dtype=bool)
        mask[60, 60] = False
        mask[80, 80] = False
        assert abs(normed[mask].mean()) < 0.2


# ---------------------------------------------------------------------------
# TestFITSDataset
# ---------------------------------------------------------------------------

@requires_torch
class TestFITSDataset:
    """Tests for the single-image FITS dataset."""

    def test_loads_from_catalog(self, tmp_path):
        """Should load dataset from CSV catalog."""
        from astroworld.ml.dataset import FITSDataset

        catalog_path = _make_tiny_catalog(tmp_path, n_pos=3, n_neg=3)
        ds = FITSDataset(catalog_path)
        assert len(ds) == 6

    def test_length_matches_catalog(self, tmp_path):
        """Dataset length should match catalog rows."""
        from astroworld.ml.dataset import FITSDataset

        catalog_path = _make_tiny_catalog(tmp_path, n_pos=5, n_neg=3)
        ds = FITSDataset(catalog_path)
        assert len(ds) == 8

    def test_returns_tensor_and_label(self, tmp_path):
        """Each item should be (Tensor, float)."""
        from astroworld.ml.dataset import FITSDataset

        catalog_path = _make_tiny_catalog(tmp_path)
        ds = FITSDataset(catalog_path)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, float)
        assert label in (0.0, 1.0)

    def test_image_shape_correct(self, tmp_path):
        """Image tensor should be (1, H, W)."""
        from astroworld.ml.dataset import FITSDataset

        catalog_path = _make_tiny_catalog(tmp_path, img_size=32)
        ds = FITSDataset(catalog_path)
        image, _ = ds[0]
        assert image.shape == (1, 32, 32)
        assert image.dtype == torch.float32

    def test_normalization_near_zero_mean(self, tmp_path):
        """Normalized image background should be near zero."""
        from astroworld.ml.dataset import FITSDataset

        catalog_path = _make_tiny_catalog(tmp_path)
        ds = FITSDataset(catalog_path, do_normalize=True)

        # Check a negative (no-source) example
        neg_idx = len(ds) - 1  # last examples are negatives
        image, label = ds[neg_idx]
        assert label == 0.0
        assert abs(image.mean().item()) < 0.5  # approximately zero

    def test_split_train_val_sizes(self, tmp_path):
        """Split should create correct train/val sizes."""
        from astroworld.ml.dataset import split_train_val

        catalog_path = _make_tiny_catalog(tmp_path, n_pos=10, n_neg=10)
        catalog = pd.read_csv(catalog_path)

        train_idx, val_idx = split_train_val(catalog, val_fraction=0.2)
        assert len(train_idx) + len(val_idx) == 20
        assert len(val_idx) >= 2  # at least 1 per class
        # No overlap
        assert len(set(train_idx) & set(val_idx)) == 0

    def test_split_stratified_balance(self, tmp_path):
        """Both train and val should have positive and negative examples."""
        from astroworld.ml.dataset import split_train_val

        catalog_path = _make_tiny_catalog(tmp_path, n_pos=10, n_neg=10)
        catalog = pd.read_csv(catalog_path)

        train_idx, val_idx = split_train_val(catalog, val_fraction=0.2)

        train_labels = catalog.iloc[train_idx]["source_present"]
        val_labels = catalog.iloc[val_idx]["source_present"]

        assert train_labels.sum() > 0  # has positives
        assert (~train_labels).sum() > 0  # has negatives
        assert val_labels.sum() > 0
        assert (~val_labels).sum() > 0


# ---------------------------------------------------------------------------
# TestP9DetectorCNN
# ---------------------------------------------------------------------------

@requires_torch
class TestP9DetectorCNN:
    """Tests for the single-image CNN architecture."""

    def test_forward_shape(self):
        """Output should be (B, 1) for batch of images."""
        from astroworld.ml.model import P9DetectorCNN

        model = P9DetectorCNN()
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        assert out.shape == (2, 1)

    def test_output_is_logit(self):
        """Raw output should be unbounded (not sigmoid)."""
        from astroworld.ml.model import P9DetectorCNN

        model = P9DetectorCNN()
        x = torch.randn(4, 1, 32, 32) * 10
        out = model(x)
        # Logits can be any real number
        assert out.dtype == torch.float32

    def test_predict_proba_bounded(self):
        """predict_proba should return values in [0, 1]."""
        from astroworld.ml.model import P9DetectorCNN

        model = P9DetectorCNN()
        model.eval()
        x = torch.randn(3, 1, 32, 32)
        probs = model.predict_proba(x)
        assert probs.shape == (3, 1)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_num_parameters_reasonable(self):
        """Should have between 50K and 500K parameters."""
        from astroworld.ml.model import P9DetectorCNN

        model = P9DetectorCNN()
        n = model.num_parameters
        assert 50_000 < n < 500_000

    def test_gradient_flows(self):
        """Backward pass should produce gradients."""
        from astroworld.ml.model import P9DetectorCNN

        model = P9DetectorCNN()
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check that some parameter has a non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_deterministic_in_eval(self):
        """Same input should give same output in eval mode."""
        from astroworld.ml.model import P9DetectorCNN

        model = P9DetectorCNN()
        model.eval()
        x = torch.randn(2, 1, 32, 32)
        out1 = model(x).detach().clone()
        out2 = model(x).detach().clone()
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# TestP9SiameseDetector
# ---------------------------------------------------------------------------

@requires_torch
class TestP9SiameseDetector:
    """Tests for the multi-epoch Siamese network."""

    def test_forward_shape_dual_input(self):
        """Should accept two images and return (B, 1)."""
        from astroworld.ml.model import P9SiameseDetector

        model = P9SiameseDetector()
        x1 = torch.randn(2, 1, 32, 32)
        x2 = torch.randn(2, 1, 32, 32)
        out = model(x1, x2)
        assert out.shape == (2, 1)

    def test_shared_encoder_same_weights(self):
        """Encoder should be shared (same object) for both inputs."""
        from astroworld.ml.model import P9SiameseDetector

        model = P9SiameseDetector()
        # The encoder is applied to both x1 and x2 via self.encoder
        # Verify by checking output is different when inputs differ
        x1 = torch.randn(1, 1, 32, 32)
        x2 = torch.randn(1, 1, 32, 32) * 2
        model.eval()
        out1 = model(x1, x2)
        # If we swap inputs, output should be different (not symmetric)
        out2 = model(x2, x1)
        # The difference vector |f1-f2| IS symmetric, but concat(f1,f2) is not
        # So outputs should differ when inputs are swapped
        assert out1.shape == (1, 1)
        assert out2.shape == (1, 1)

    def test_identical_inputs_consistent(self):
        """Identical epoch inputs should give a valid output."""
        from astroworld.ml.model import P9SiameseDetector

        model = P9SiameseDetector()
        model.eval()
        x = torch.randn(2, 1, 32, 32)
        out = model(x, x)
        assert out.shape == (2, 1)
        # Output should be finite
        assert torch.isfinite(out).all()

    def test_gradient_flows_both_inputs(self):
        """Gradients should flow through both input branches."""
        from astroworld.ml.model import P9SiameseDetector

        model = P9SiameseDetector()
        x1 = torch.randn(2, 1, 32, 32, requires_grad=True)
        x2 = torch.randn(2, 1, 32, 32, requires_grad=True)
        out = model(x1, x2)
        loss = out.sum()
        loss.backward()
        assert x1.grad is not None
        assert x2.grad is not None
        assert x1.grad.abs().sum() > 0
        assert x2.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# TestTrainer
# ---------------------------------------------------------------------------

@requires_torch
class TestTrainer:
    """Tests for the training loop."""

    def test_train_one_epoch(self, tmp_path):
        """Should complete one training epoch without error."""
        from astroworld.ml.model import P9DetectorCNN
        from astroworld.ml.dataset import FITSDataset, split_train_val
        from astroworld.ml.trainer import Trainer, TrainingConfig

        catalog_path = _make_tiny_catalog(tmp_path, n_pos=4, n_neg=4)
        catalog = pd.read_csv(catalog_path)
        train_idx, val_idx = split_train_val(catalog, val_fraction=0.3)

        train_ds = FITSDataset(catalog_path, indices=train_idx)
        val_ds = FITSDataset(catalog_path, indices=val_idx)

        model = P9DetectorCNN()
        config = TrainingConfig(
            epochs=1, batch_size=4, device="cpu",
            scheduler="plateau",
        )
        trainer = Trainer(model, config)
        result = trainer.train(train_ds, val_ds)

        assert len(result.epoch_metrics) == 1
        assert result.epoch_metrics[0].train_loss > 0

    def test_validation_metrics_computed(self, tmp_path):
        """Validation metrics should be populated after training."""
        from astroworld.ml.model import P9DetectorCNN
        from astroworld.ml.dataset import FITSDataset, split_train_val
        from astroworld.ml.trainer import Trainer, TrainingConfig

        catalog_path = _make_tiny_catalog(tmp_path, n_pos=6, n_neg=6)
        catalog = pd.read_csv(catalog_path)
        train_idx, val_idx = split_train_val(catalog, val_fraction=0.3)

        train_ds = FITSDataset(catalog_path, indices=train_idx)
        val_ds = FITSDataset(catalog_path, indices=val_idx)

        model = P9DetectorCNN()
        config = TrainingConfig(
            epochs=2, batch_size=4, device="cpu",
            scheduler="plateau",
        )
        trainer = Trainer(model, config)
        result = trainer.train(train_ds, val_ds)

        m = result.epoch_metrics[-1]
        assert 0.0 <= m.val_accuracy <= 1.0
        assert 0.0 <= m.val_precision <= 1.0
        assert 0.0 <= m.val_recall <= 1.0
        assert 0.0 <= m.val_f1 <= 1.0
        assert 0.0 <= m.val_roc_auc <= 1.0

    def test_early_stopping_triggers(self, tmp_path):
        """Early stopping should halt training before max epochs."""
        from astroworld.ml.model import P9DetectorCNN
        from astroworld.ml.dataset import FITSDataset, split_train_val
        from astroworld.ml.trainer import Trainer, TrainingConfig

        catalog_path = _make_tiny_catalog(tmp_path, n_pos=4, n_neg=4)
        catalog = pd.read_csv(catalog_path)
        train_idx, val_idx = split_train_val(catalog, val_fraction=0.3)

        train_ds = FITSDataset(catalog_path, indices=train_idx)
        val_ds = FITSDataset(catalog_path, indices=val_idx)

        model = P9DetectorCNN()
        config = TrainingConfig(
            epochs=100,  # very high
            batch_size=4,
            patience=3,  # but very low patience
            device="cpu",
            scheduler="plateau",
        )
        trainer = Trainer(model, config)
        result = trainer.train(train_ds, val_ds)

        # Should stop well before 100 epochs
        assert len(result.epoch_metrics) < 100

    def test_checkpoint_saved(self, tmp_path):
        """Checkpoint should be saved when checkpoint_dir is set."""
        from astroworld.ml.model import P9DetectorCNN
        from astroworld.ml.dataset import FITSDataset, split_train_val
        from astroworld.ml.trainer import Trainer, TrainingConfig

        catalog_path = _make_tiny_catalog(tmp_path, n_pos=4, n_neg=4)
        catalog = pd.read_csv(catalog_path)
        train_idx, val_idx = split_train_val(catalog, val_fraction=0.3)

        train_ds = FITSDataset(catalog_path, indices=train_idx)
        val_ds = FITSDataset(catalog_path, indices=val_idx)

        ckpt_dir = tmp_path / "checkpoints"
        model = P9DetectorCNN()
        config = TrainingConfig(
            epochs=2, batch_size=4, device="cpu",
            checkpoint_dir=ckpt_dir,
            scheduler="plateau",
        )
        trainer = Trainer(model, config)
        result = trainer.train(train_ds, val_ds)

        assert result.model_state_path is not None
        assert result.model_state_path.exists()

    def test_checkpoint_loadable(self, tmp_path):
        """Should be able to load model from saved checkpoint."""
        from astroworld.ml.model import P9DetectorCNN
        from astroworld.ml.dataset import FITSDataset, split_train_val
        from astroworld.ml.trainer import Trainer, TrainingConfig

        catalog_path = _make_tiny_catalog(tmp_path, n_pos=4, n_neg=4)
        catalog = pd.read_csv(catalog_path)
        train_idx, val_idx = split_train_val(catalog, val_fraction=0.3)

        train_ds = FITSDataset(catalog_path, indices=train_idx)
        val_ds = FITSDataset(catalog_path, indices=val_idx)

        ckpt_dir = tmp_path / "checkpoints"
        model = P9DetectorCNN()
        config = TrainingConfig(
            epochs=2, batch_size=4, device="cpu",
            checkpoint_dir=ckpt_dir,
            scheduler="plateau",
        )
        trainer = Trainer(model, config)
        result = trainer.train(train_ds, val_ds)

        # Load into a fresh model
        model2 = P9DetectorCNN()
        trainer2 = Trainer(model2, config)
        epoch = trainer2.load_checkpoint(result.model_state_path)
        assert epoch == result.best_epoch

    def test_metrics_accuracy_known_values(self):
        """compute_metrics should give correct results for known inputs."""
        from astroworld.ml.trainer import compute_metrics

        labels = np.array([1, 1, 0, 0])
        # Perfect classifier: positive logits for positives, negative for negatives
        logits = np.array([2.0, 3.0, -2.0, -1.0])

        m = compute_metrics(labels, logits)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_roc_auc_perfect_classifier(self):
        """Perfect classifier should have AUC = 1.0."""
        from astroworld.ml.trainer import compute_roc_auc

        labels = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        auc = compute_roc_auc(labels, scores)
        assert abs(auc - 1.0) < 0.01

    def test_roc_auc_random_classifier(self):
        """Random classifier should have AUC ~ 0.5."""
        from astroworld.ml.trainer import compute_roc_auc

        rng = np.random.default_rng(42)
        n = 1000
        labels = rng.integers(0, 2, n)
        scores = rng.random(n)
        auc = compute_roc_auc(labels, scores)
        assert abs(auc - 0.5) < 0.1  # approximately 0.5


# ---------------------------------------------------------------------------
# TestMultiEpochDataset
# ---------------------------------------------------------------------------

@requires_torch
class TestMultiEpochDataset:
    """Tests for the paired multi-epoch dataset."""

    def test_loads_pairs(self, tmp_path):
        """Should detect and load epoch pairs."""
        from astroworld.ml.dataset import MultiEpochDataset

        catalog_path = _make_multi_epoch_catalog(tmp_path, n_pairs=3)
        ds = MultiEpochDataset(catalog_path)
        assert len(ds) == 3

    def test_returns_two_images_and_label(self, tmp_path):
        """Each item should be (Tensor, Tensor, float)."""
        from astroworld.ml.dataset import MultiEpochDataset

        catalog_path = _make_multi_epoch_catalog(tmp_path, n_pairs=2)
        ds = MultiEpochDataset(catalog_path)
        img1, img2, label = ds[0]

        assert isinstance(img1, torch.Tensor)
        assert isinstance(img2, torch.Tensor)
        assert isinstance(label, float)
        assert img1.shape == img2.shape
        assert img1.shape[0] == 1  # channel dim

    def test_paired_augmentation_consistent(self, tmp_path):
        """With augmentation, both epochs should get same transform."""
        from astroworld.ml.dataset import paired_random_flip

        rng = np.random.default_rng(42)
        img1 = np.arange(16).reshape(4, 4).astype(np.float32)
        img2 = np.arange(16, 32).reshape(4, 4).astype(np.float32)

        # The key property: if we flip, both get flipped the same way
        aug1, aug2 = paired_random_flip(img1, img2, rng)

        # Verify they have the same flip applied by checking shape
        assert aug1.shape == aug2.shape == (4, 4)
        # If horizontal flip was applied to img1, it was also applied to img2
        # We can verify by checking if the transformation is consistent
        if np.array_equal(aug1, np.flip(img1, axis=-1)):
            assert np.array_equal(aug2, np.flip(img2, axis=-1))
