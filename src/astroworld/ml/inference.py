"""
Sliding-window inference pipeline for Planet 9 detection.

Takes two FITS images (epoch 1 and epoch 2), tiles them into 64×64
patches, runs all patches through P9SiameseDetector in a single batch,
and produces a probability heatmap with candidate detections.

The tiling strategy uses 8-pixel overlap between adjacent patches to
ensure sources at patch boundaries are captured.  For a 512×512 image
with 64×64 patches and stride 56, this produces a 9×9 = 81 patch grid.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from astroworld.imaging.fits_store import load_fits_image
from astroworld.ml.dataset import normalize_image
from astroworld.ml.model import P9SiameseDetector


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PatchDetection:
    """A single patch-level detection from the Siamese network."""

    patch_row: int      # Patch grid row index (0-based)
    patch_col: int      # Patch grid column index (0-based)
    pixel_x: float      # Center pixel x of the patch in the full image
    pixel_y: float      # Center pixel y of the patch in the full image
    probability: float  # Detection probability from model


@dataclass
class Candidate:
    """A grouped detection candidate after NMS."""

    pixel_x: float                          # Best-estimate pixel x
    pixel_y: float                          # Best-estimate pixel y
    ra_deg: float                           # RA in degrees (from WCS)
    dec_deg: float                          # Dec in degrees (from WCS)
    probability: float                      # Peak probability
    n_patches: int                          # Overlapping patches merged
    patch_indices: list[tuple[int, int]]    # (row, col) of merged patches


@dataclass
class SearchResult:
    """Complete result of searching one image pair."""

    image_path_1: str
    image_path_2: str
    heatmap: np.ndarray                     # (n_rows, n_cols) probability grid
    patch_detections: list[PatchDetection]   # Patches above threshold
    candidates: list[Candidate]             # After NMS
    grid_shape: tuple[int, int]             # (n_rows, n_cols)
    patch_size: int
    stride: int
    threshold: float
    inference_time_seconds: float


# ---------------------------------------------------------------------------
# P9Searcher — sliding window + batch inference
# ---------------------------------------------------------------------------

class P9Searcher:
    """Sliding-window inference pipeline for Planet 9 detection.

    Takes two FITS images (epoch 1, epoch 2), tiles them into patches,
    runs all patches through P9SiameseDetector, and produces a
    probability heatmap with candidate detections.

    Parameters
    ----------
    model : P9SiameseDetector instance (already loaded)
    patch_size : Side length of each square patch (default 64)
    overlap : Overlap between adjacent patches in pixels (default 8)
    threshold : Detection probability threshold (default 0.95)
    device : Compute device ("auto", "cpu", "cuda", "mps")
    """

    def __init__(
        self,
        model: P9SiameseDetector,
        patch_size: int = 64,
        overlap: int = 8,
        threshold: float = 0.95,
        device: str = "auto",
    ):
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.threshold = threshold
        self.device = self._select_device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        patch_size: int = 64,
        overlap: int = 8,
        threshold: float = 0.95,
        device: str = "auto",
    ) -> "P9Searcher":
        """Load model from checkpoint and create searcher.

        Parameters
        ----------
        checkpoint_path : Path to ``best_model.pt`` checkpoint
        patch_size, overlap, threshold, device : Passed to constructor

        Returns
        -------
        Configured P9Searcher ready for inference
        """
        checkpoint_path = Path(checkpoint_path)

        # Determine device first
        searcher_device = cls._select_device_static(device)

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=searcher_device, weights_only=True
        )

        # Build model and load weights
        model = P9SiameseDetector()
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(
            model=model,
            patch_size=patch_size,
            overlap=overlap,
            threshold=threshold,
            device=device,
        )

    def search(
        self,
        fits_path_1: Path | str,
        fits_path_2: Path | str,
    ) -> SearchResult:
        """Run full search on an image pair.

        Parameters
        ----------
        fits_path_1 : Path to epoch 1 FITS image
        fits_path_2 : Path to epoch 2 FITS image

        Returns
        -------
        SearchResult with heatmap, detections, and timing
        """
        image_1, _header_1 = load_fits_image(Path(fits_path_1))
        image_2, _header_2 = load_fits_image(Path(fits_path_2))

        return self.search_from_arrays(
            image_1, image_2,
            image_path_1=str(fits_path_1),
            image_path_2=str(fits_path_2),
        )

    def search_from_arrays(
        self,
        image_1: np.ndarray,
        image_2: np.ndarray,
        image_path_1: str = "<array>",
        image_path_2: str = "<array>",
    ) -> SearchResult:
        """Run search on pre-loaded numpy arrays.

        Parameters
        ----------
        image_1, image_2 : 2D numpy arrays (same shape)
        image_path_1, image_path_2 : Labels for the result

        Returns
        -------
        SearchResult
        """
        t0 = time.time()

        # Validate dimensions
        if image_1.shape != image_2.shape:
            raise ValueError(
                f"Image shapes must match: {image_1.shape} != {image_2.shape}"
            )

        h, w = image_1.shape

        # Compute grid layout
        n_rows, n_cols, y_origins, x_origins = self._compute_grid_shape(h, w)

        # Tile both images
        patches_1, origins = self._tile_image(image_1, y_origins, x_origins)
        patches_2, _ = self._tile_image(image_2, y_origins, x_origins)

        # Batch inference
        probs = self._batch_inference(patches_1, patches_2)

        # Build heatmap and detections
        heatmap = np.zeros((n_rows, n_cols), dtype=np.float32)
        detections = []

        for idx, (y0, x0) in enumerate(origins):
            row_idx = y_origins.index(y0)
            col_idx = x_origins.index(x0)

            heatmap[row_idx, col_idx] = probs[idx]

            if probs[idx] >= self.threshold:
                # Center pixel of this patch in the full image
                cx = x0 + self.patch_size / 2.0
                cy = y0 + self.patch_size / 2.0

                detections.append(PatchDetection(
                    patch_row=row_idx,
                    patch_col=col_idx,
                    pixel_x=cx,
                    pixel_y=cy,
                    probability=float(probs[idx]),
                ))

        elapsed = time.time() - t0

        return SearchResult(
            image_path_1=image_path_1,
            image_path_2=image_path_2,
            heatmap=heatmap,
            patch_detections=detections,
            candidates=[],  # Filled by post_processing.detections_to_candidates
            grid_shape=(n_rows, n_cols),
            patch_size=self.patch_size,
            stride=self.stride,
            threshold=self.threshold,
            inference_time_seconds=elapsed,
        )

    def _compute_grid_shape(
        self,
        img_h: int,
        img_w: int,
    ) -> tuple[int, int, list[int], list[int]]:
        """Compute patch grid layout for an image.

        Parameters
        ----------
        img_h, img_w : Image height and width in pixels

        Returns
        -------
        (n_rows, n_cols, y_origins, x_origins) where origins are
        top-left pixel indices for each patch position
        """
        ps = self.patch_size
        stride = self.stride

        def _origins(img_size: int) -> list[int]:
            if img_size < ps:
                return [0]  # Image smaller than patch — single patch, zero-padded
            origins = list(range(0, img_size - ps + 1, stride))
            # Ensure edge coverage
            if origins[-1] + ps < img_size:
                origins.append(img_size - ps)
            return origins

        y_origins = _origins(img_h)
        x_origins = _origins(img_w)

        return len(y_origins), len(x_origins), y_origins, x_origins

    def _tile_image(
        self,
        image: np.ndarray,
        y_origins: list[int],
        x_origins: list[int],
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Tile a 2D image into normalized patches.

        Each patch is independently normalized using sigma-clip
        (matching training-time behavior).

        Parameters
        ----------
        image : 2D numpy array
        y_origins, x_origins : Patch origin coordinates

        Returns
        -------
        patches : (N, 1, patch_size, patch_size) float32 tensor
        origins : list of (y_origin, x_origin) for each patch
        """
        ps = self.patch_size
        h, w = image.shape
        patches = []
        origins = []

        for y0 in y_origins:
            for x0 in x_origins:
                # Extract patch (zero-pad if needed)
                y1 = min(y0 + ps, h)
                x1 = min(x0 + ps, w)
                patch = np.zeros((ps, ps), dtype=np.float64)
                patch[:y1 - y0, :x1 - x0] = image[y0:y1, x0:x1]

                # Normalize per-patch (same as training)
                patch = normalize_image(patch)

                patches.append(patch)
                origins.append((y0, x0))

        # Stack into tensor: (N, 1, ps, ps)
        stacked = np.stack(patches, axis=0)[:, np.newaxis, :, :]
        return torch.from_numpy(stacked).float(), origins

    @torch.no_grad()
    def _batch_inference(
        self,
        patches_1: torch.Tensor,
        patches_2: torch.Tensor,
    ) -> np.ndarray:
        """Run batched inference through the Siamese network.

        Parameters
        ----------
        patches_1, patches_2 : (N, 1, H, W) tensors

        Returns
        -------
        (N,) numpy array of detection probabilities in [0, 1]
        """
        patches_1 = patches_1.to(self.device)
        patches_2 = patches_2.to(self.device)

        probs = self.model.predict_proba(patches_1, patches_2)

        return probs.cpu().numpy().flatten()

    @staticmethod
    def _select_device_static(device: str) -> torch.device:
        """Select compute device (static version for classmethod)."""
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _select_device(self, device: str) -> torch.device:
        """Select compute device."""
        return self._select_device_static(device)
