"""
Machine learning pipeline for Planet 9 detection in sky survey images.

Provides CNN-based detectors trained on synthetic FITS images generated
by the imaging pipeline (Phase 7).  Two architectures:
  - P9DetectorCNN: single-image binary classifier (baseline)
  - P9SiameseDetector: multi-epoch network detecting correlated motion

Modules:
  - dataset: FITS image loading and preprocessing for PyTorch
  - model: CNN architectures for single-image and multi-epoch detection
  - trainer: Training loop, metrics, checkpointing
"""

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from astroworld.ml.dataset import FITSDataset, MultiEpochDataset
    from astroworld.ml.model import P9DetectorCNN, P9SiameseDetector
    from astroworld.ml.trainer import Trainer, TrainingConfig, TrainingResult

__all__ = [
    "HAS_TORCH",
    # Dataset
    "FITSDataset",
    "MultiEpochDataset",
    # Models
    "P9DetectorCNN",
    "P9SiameseDetector",
    # Training
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
]
