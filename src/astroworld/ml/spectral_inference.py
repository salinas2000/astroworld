"""
Sliding-window spectral inference pipeline with MC Dropout.

Extends the P9Searcher concept to 5-channel spectral cubes, adding:
  - Per-channel normalization during tiling
  - MC Dropout for Bayesian uncertainty estimation
  - Temperature map alongside detection heatmap
  - PlanckFilter classification for each candidate

For a 512×512 cube with 64×64 patches and stride 56, this produces
a 9×9 = 81 patch grid — identical layout to the original searcher.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from astroworld.imaging.spectral_cube import NUM_CHANNELS
from astroworld.ml.dataset import normalize_image
from astroworld.ml.inference import (
    Candidate,
    PatchDetection,
    SearchResult,
)
from astroworld.ml.spectral_model import PlanckFilter, SpectralSiameseNet


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpectralSearchResult(SearchResult):
    """Search result with spectral temperature and uncertainty maps.

    Extends SearchResult with:
      - temperature_map: Estimated temperature at each patch (Kelvin)
      - temperature_std_map: Temperature uncertainty (Kelvin)
      - motion_std_map: Detection probability uncertainty
      - planck_classifications: PlanckFilter class per candidate
    """

    temperature_map: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    temperature_std_map: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    motion_std_map: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    planck_classifications: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SpectralSearcher — sliding window + MC Dropout + Planck
# ---------------------------------------------------------------------------

class SpectralSearcher:
    """Sliding-window spectral search with MC Dropout uncertainty.

    Takes two 5-channel spectral cubes (epoch 1, epoch 2), tiles them
    into patches, runs MC Dropout inference, and produces probability,
    temperature, and uncertainty heatmaps.

    Parameters
    ----------
    model : SpectralSiameseNet instance
    patch_size : Side length of patches (default 64)
    overlap : Overlap between patches in pixels (default 8)
    threshold : Detection probability threshold (default 0.80)
    mc_samples : Number of MC Dropout forward passes (default 10)
    device : Compute device ("auto", "cpu", "cuda", "mps")
    """

    def __init__(
        self,
        model: SpectralSiameseNet,
        patch_size: int = 64,
        overlap: int = 8,
        threshold: float = 0.80,
        mc_samples: int = 10,
        device: str = "auto",
    ):
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.threshold = threshold
        self.mc_samples = mc_samples
        self.device = self._select_device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        patch_size: int = 64,
        overlap: int = 8,
        threshold: float = 0.80,
        mc_samples: int = 10,
        device: str = "auto",
        **model_kwargs,
    ) -> "SpectralSearcher":
        """Load SpectralSiameseNet from checkpoint and create searcher.

        Parameters
        ----------
        checkpoint_path : Path to ``best_model.pt``
        model_kwargs : Passed to SpectralSiameseNet constructor
            (feature_dim, num_heads, dropout)
        """
        checkpoint_path = Path(checkpoint_path)
        searcher_device = cls._select_device_static(device)

        checkpoint = torch.load(
            checkpoint_path, map_location=searcher_device, weights_only=True
        )

        model = SpectralSiameseNet(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(
            model=model,
            patch_size=patch_size,
            overlap=overlap,
            threshold=threshold,
            mc_samples=mc_samples,
            device=device,
        )

    def search_from_cubes(
        self,
        cube_1: np.ndarray,
        cube_2: np.ndarray,
        label_1: str = "<cube>",
        label_2: str = "<cube>",
    ) -> SpectralSearchResult:
        """Run full spectral search on a cube pair.

        Parameters
        ----------
        cube_1, cube_2 : (5, H, W) spectral cubes
        label_1, label_2 : Labels for the result

        Returns
        -------
        SpectralSearchResult with heatmaps and Planck classifications
        """
        t0 = time.time()

        # Validate dimensions
        if cube_1.shape != cube_2.shape:
            raise ValueError(
                f"Cube shapes must match: {cube_1.shape} != {cube_2.shape}"
            )
        if cube_1.ndim != 3 or cube_1.shape[0] != NUM_CHANNELS:
            raise ValueError(
                f"Expected (5, H, W) cube, got {cube_1.shape}"
            )

        _, h, w = cube_1.shape

        # Compute grid layout
        n_rows, n_cols, y_origins, x_origins = self._compute_grid_shape(h, w)

        # Tile both cubes
        patches_1, origins = self._tile_spectral_cube(
            cube_1, y_origins, x_origins
        )
        patches_2, _ = self._tile_spectral_cube(
            cube_2, y_origins, x_origins
        )

        # MC Dropout inference
        mc_result = self._mc_batch_inference(patches_1, patches_2)

        # Build heatmaps
        heatmap = np.zeros((n_rows, n_cols), dtype=np.float32)
        motion_std_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        temp_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        temp_std_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        detections = []

        for idx, (y0, x0) in enumerate(origins):
            row_idx = y_origins.index(y0)
            col_idx = x_origins.index(x0)

            prob_mean = mc_result["motion_mean"][idx]
            prob_std = mc_result["motion_std"][idx]
            t_mean = mc_result["temp_mean"][idx]
            t_std = mc_result["temp_std"][idx]

            heatmap[row_idx, col_idx] = prob_mean
            motion_std_map[row_idx, col_idx] = prob_std
            temp_map[row_idx, col_idx] = t_mean
            temp_std_map[row_idx, col_idx] = t_std

            if prob_mean >= self.threshold:
                cx = x0 + self.patch_size / 2.0
                cy = y0 + self.patch_size / 2.0

                detections.append(PatchDetection(
                    patch_row=row_idx,
                    patch_col=col_idx,
                    pixel_x=cx,
                    pixel_y=cy,
                    probability=float(prob_mean),
                ))

        # Planck classification for each detection
        planck_filter = PlanckFilter()
        classifications = []
        for det in detections:
            r, c = det.patch_row, det.patch_col
            t_k = float(temp_map[r, c])
            t_s = float(temp_std_map[r, c])
            classifications.append(planck_filter.classify(t_k, t_s))

        elapsed = time.time() - t0

        return SpectralSearchResult(
            image_path_1=label_1,
            image_path_2=label_2,
            heatmap=heatmap,
            patch_detections=detections,
            candidates=[],  # Filled by post_processing
            grid_shape=(n_rows, n_cols),
            patch_size=self.patch_size,
            stride=self.stride,
            threshold=self.threshold,
            inference_time_seconds=elapsed,
            temperature_map=temp_map,
            temperature_std_map=temp_std_map,
            motion_std_map=motion_std_map,
            planck_classifications=classifications,
        )

    def _compute_grid_shape(
        self,
        img_h: int,
        img_w: int,
    ) -> tuple[int, int, list[int], list[int]]:
        """Compute patch grid layout (same logic as P9Searcher)."""
        ps = self.patch_size
        stride = self.stride

        def _origins(img_size: int) -> list[int]:
            if img_size < ps:
                return [0]
            origins = list(range(0, img_size - ps + 1, stride))
            if origins[-1] + ps < img_size:
                origins.append(img_size - ps)
            return origins

        y_origins = _origins(img_h)
        x_origins = _origins(img_w)

        return len(y_origins), len(x_origins), y_origins, x_origins

    def _tile_spectral_cube(
        self,
        cube: np.ndarray,
        y_origins: list[int],
        x_origins: list[int],
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Tile a (5, H, W) cube into normalized patches.

        Each patch is (5, ps, ps) with each channel normalized
        independently via sigma-clipping.

        Returns
        -------
        patches : (N, 5, ps, ps) float32 tensor
        origins : list of (y0, x0) for each patch
        """
        ps = self.patch_size
        _, h, w = cube.shape
        patches = []
        origins = []

        for y0 in y_origins:
            for x0 in x_origins:
                y1 = min(y0 + ps, h)
                x1 = min(x0 + ps, w)

                patch = np.zeros((NUM_CHANNELS, ps, ps), dtype=np.float64)
                patch[:, :y1 - y0, :x1 - x0] = cube[:, y0:y1, x0:x1]

                # Normalize each channel independently
                for ch in range(NUM_CHANNELS):
                    patch[ch] = normalize_image(patch[ch])

                patches.append(patch.astype(np.float32))
                origins.append((y0, x0))

        stacked = np.stack(patches, axis=0)  # (N, 5, ps, ps)
        return torch.from_numpy(stacked).float(), origins

    def _mc_batch_inference(
        self,
        patches_1: torch.Tensor,
        patches_2: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Run MC Dropout inference for uncertainty estimation.

        Parameters
        ----------
        patches_1, patches_2 : (N, 5, H, W) tensors

        Returns
        -------
        Dict with:
            motion_mean, motion_std : (N,) arrays
            temp_mean, temp_std : (N,) arrays (Kelvin)
        """
        patches_1 = patches_1.to(self.device)
        patches_2 = patches_2.to(self.device)

        n_patches = patches_1.shape[0]

        if self.mc_samples <= 1:
            # No MC Dropout — single deterministic pass
            self.model.eval()
            with torch.no_grad():
                motion_logits, temp_log10k = self.model(patches_1, patches_2)
                motion_probs = torch.sigmoid(motion_logits).cpu().numpy().flatten()
                temps = (10.0 ** temp_log10k).cpu().numpy().flatten()

            return {
                "motion_mean": motion_probs,
                "motion_std": np.zeros(n_patches, dtype=np.float32),
                "temp_mean": temps,
                "temp_std": np.zeros(n_patches, dtype=np.float32),
            }

        # MC Dropout: multiple forward passes with dropout active
        self.model.train()  # Enable dropout
        motion_samples = []
        temp_samples = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                motion_logits, temp_log10k = self.model(patches_1, patches_2)
                probs = torch.sigmoid(motion_logits).cpu().numpy().flatten()
                temps = (10.0 ** temp_log10k).cpu().numpy().flatten()
                motion_samples.append(probs)
                temp_samples.append(temps)

        self.model.eval()  # Restore eval mode

        motion_arr = np.stack(motion_samples, axis=0)  # (mc, N)
        temp_arr = np.stack(temp_samples, axis=0)        # (mc, N)

        return {
            "motion_mean": motion_arr.mean(axis=0),
            "motion_std": motion_arr.std(axis=0),
            "temp_mean": temp_arr.mean(axis=0),
            "temp_std": temp_arr.std(axis=0),
        }

    @staticmethod
    def _select_device_static(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _select_device(self, device: str) -> torch.device:
        return self._select_device_static(device)
