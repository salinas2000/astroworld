"""
Machine learning pipeline for Planet 9 detection in sky survey images.

Provides CNN-based detectors trained on synthetic FITS images generated
by the imaging pipeline (Phase 7), plus inference tools for searching
survey images using a sliding-window approach.

Modules:
  - dataset: FITS image loading and preprocessing for PyTorch
  - model: CNN architectures for single-image and multi-epoch detection
  - trainer: Training loop, metrics, checkpointing
  - inference: Sliding-window search pipeline (P9Searcher)
  - post_processing: NMS, coordinate conversion, visualization
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
    from astroworld.ml.inference import (
        P9Searcher,
        SearchResult,
        PatchDetection,
        Candidate,
    )
    from astroworld.ml.post_processing import (
        non_maximum_suppression,
        detections_to_candidates,
    )

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
    # Inference
    "P9Searcher",
    "SearchResult",
    "PatchDetection",
    "Candidate",
    # Post-processing
    "non_maximum_suppression",
    "detections_to_candidates",
]
