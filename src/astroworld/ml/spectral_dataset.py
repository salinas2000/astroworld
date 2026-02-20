"""
PyTorch dataset for 5-channel spectral cubes (SSTF training).

Loads multi-survey image pairs (optical + WISE W1 + WISE W2) from a
training catalog, assembles them into spectral cubes, and returns
``(cube1, cube2, motion_label, temp_kelvin)`` tuples for dual-head
training of SpectralSiameseNet.

Catalog CSV schema
------------------
example_id       : str   — pair identifier (e.g. "pair_0001_epoch1")
optical_path     : str   — path to optical FITS file
w1_path          : str   — path to WISE W1 FITS (or empty)
w2_path          : str   — path to WISE W2 FITS (or empty)
source_present   : bool  — True if moving source injected
temp_kelvin      : float — ground-truth temperature (NaN for negatives)
epoch_jd         : float — Julian date of epoch
delta_t_days     : float — time separation between paired epochs

The temporal gradient channel (C4) is computed online from the paired
optical images divided by ``delta_t_days``.

Cube caching: Pre-assembled cubes are loaded from disk if available
(see ``spectral_cube.build_spectral_cube_cached``), avoiding repeated
WCS reprojection during training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from astroworld.imaging.fits_store import load_fits_image
from astroworld.imaging.spectral_cube import (
    BAND_TEMPORAL,
    NUM_CHANNELS,
    build_spectral_cube,
    build_spectral_cube_cached,
    compute_temporal_gradient,
)
from astroworld.ml.dataset import normalize_image, paired_random_flip


# ---------------------------------------------------------------------------
# Per-channel normalization for spectral cubes
# ---------------------------------------------------------------------------

def normalize_spectral_cube(cube: np.ndarray) -> np.ndarray:
    """Normalize each channel of a spectral cube independently.

    Applies iterative sigma-clipping (from ``normalize_image``) to
    each channel, so all channels have background ~ N(0, 1).

    Parameters
    ----------
    cube : (C, H, W) spectral cube (typically C=5)

    Returns
    -------
    Normalized cube (float32), same shape.
    """
    n_channels = cube.shape[0]
    result = np.empty_like(cube, dtype=np.float32)

    for ch in range(n_channels):
        result[ch] = normalize_image(cube[ch])

    return result


# ---------------------------------------------------------------------------
# MultiSurveyDataset — 5-channel paired cubes for Siamese training
# ---------------------------------------------------------------------------

class MultiSurveyDataset(Dataset):
    """PyTorch Dataset for paired 5-channel spectral cubes.

    Each sample is a pair of spectral cubes (epoch 1, epoch 2) with
    dual labels:
      - motion_label : 1.0 if moving source present, 0.0 otherwise
      - temp_kelvin  : ground-truth blackbody temperature (NaN for negatives)

    Pairs are linked by shared prefix in ``example_id``:
      pair_0001_epoch1 + pair_0001_epoch2

    Parameters
    ----------
    catalog_path : Path to training catalog CSV
    do_normalize : Apply per-channel sigma-clip normalization
    augment : Apply paired random flips during training
    indices : Subset of pair indices (for train/val split)
    use_cache : Load cached cubes from disk if available
    cache_dir : Directory for cube cache files
    """

    def __init__(
        self,
        catalog_path: Path | str,
        do_normalize: bool = True,
        augment: bool = False,
        indices: np.ndarray | None = None,
        use_cache: bool = True,
        cache_dir: Path | str = Path("data/spectral_cache"),
    ):
        self.catalog_path = Path(catalog_path)
        self.catalog = pd.read_csv(self.catalog_path)
        self.base_dir = self.catalog_path.parent
        self.do_normalize = do_normalize
        self.augment = augment
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self._rng = np.random.default_rng()

        # Group pairs by prefix
        self.pairs = self._find_pairs()

        if indices is not None:
            self.pairs = [self.pairs[i] for i in indices]

    def _find_pairs(self) -> list[tuple[int, int]]:
        """Identify epoch-1 / epoch-2 pairs from example_id column."""
        pairs = {}
        for idx, row in self.catalog.iterrows():
            eid = row["example_id"]
            if "_epoch1" in eid:
                pair_key = eid.replace("_epoch1", "")
                pairs.setdefault(pair_key, {})["epoch1"] = idx
            elif "_epoch2" in eid:
                pair_key = eid.replace("_epoch2", "")
                pairs.setdefault(pair_key, {})["epoch2"] = idx

        result = []
        for _key, epochs in sorted(pairs.items()):
            if "epoch1" in epochs and "epoch2" in epochs:
                result.append((epochs["epoch1"], epochs["epoch2"]))

        return result

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Load and return paired spectral cubes with labels.

        Returns
        -------
        cube1 : Tensor (5, H, W) — epoch 1 spectral cube
        cube2 : Tensor (5, H, W) — epoch 2 spectral cube
        motion_label : float — 1.0 if source present, 0.0 otherwise
        temp_kelvin : float — temperature in K (NaN for negatives)
        """
        idx1, idx2 = self.pairs[idx]
        row1 = self.catalog.iloc[idx1]
        row2 = self.catalog.iloc[idx2]

        # Load spectral cubes for both epochs
        cube1 = self._load_cube(row1)
        cube2 = self._load_cube(row2)

        # Fill temporal gradient channel (C4) using both epochs
        delta_t = float(row1.get("delta_t_days", 30.0))
        if delta_t > 0:
            temporal_grad = compute_temporal_gradient(cube1, cube2, delta_t)
            cube1[BAND_TEMPORAL] = temporal_grad
            cube2[BAND_TEMPORAL] = -temporal_grad  # Symmetric

        # Normalize each channel
        if self.do_normalize:
            cube1 = normalize_spectral_cube(cube1)
            cube2 = normalize_spectral_cube(cube2)

        # Paired augmentation (same flip on both cubes, all 5 channels)
        if self.augment:
            cube1, cube2 = _paired_random_flip_cube(cube1, cube2, self._rng)

        tensor1 = torch.from_numpy(cube1).float()
        tensor2 = torch.from_numpy(cube2).float()

        motion_label = 1.0 if row1["source_present"] else 0.0

        # Temperature: NaN for negatives (masked in loss)
        temp_kelvin = float(row1.get("temp_kelvin", float("nan")))

        return tensor1, tensor2, motion_label, temp_kelvin

    def _load_cube(self, row: pd.Series) -> np.ndarray:
        """Load or build a spectral cube for one epoch.

        Tries to load from cache first.  If not cached, builds from
        individual FITS files (optical + W1 + W2).
        """
        # Load optical image
        optical_path = self._resolve_path(row["optical_path"])
        optical_image, optical_header = load_fits_image(optical_path)

        # Load WISE images if available
        w1_image, w1_header = None, None
        w2_image, w2_header = None, None

        w1_path_str = row.get("w1_path", "")
        if pd.notna(w1_path_str) and str(w1_path_str).strip():
            try:
                w1_path = self._resolve_path(str(w1_path_str))
                w1_image, w1_header = load_fits_image(w1_path)
            except Exception:
                pass  # Missing W1 → zeros

        w2_path_str = row.get("w2_path", "")
        if pd.notna(w2_path_str) and str(w2_path_str).strip():
            try:
                w2_path = self._resolve_path(str(w2_path_str))
                w2_image, w2_header = load_fits_image(w2_path)
            except Exception:
                pass  # Missing W2 → zeros

        target_shape = optical_image.shape

        if self.use_cache:
            cube = build_spectral_cube_cached(
                optical_image, dict(optical_header) if hasattr(optical_header, 'items') else optical_header,
                w1_image, dict(w1_header) if w1_header is not None and hasattr(w1_header, 'items') else w1_header,
                w2_image, dict(w2_header) if w2_header is not None and hasattr(w2_header, 'items') else w2_header,
                target_shape=target_shape,
                cache_dir=self.cache_dir,
            )
        else:
            cube = build_spectral_cube(
                optical_image, dict(optical_header) if hasattr(optical_header, 'items') else optical_header,
                w1_image, dict(w1_header) if w1_header is not None and hasattr(w1_header, 'items') else w1_header,
                w2_image, dict(w2_header) if w2_header is not None and hasattr(w2_header, 'items') else w2_header,
                target_shape=target_shape,
            )

        return cube

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a catalog path relative to base directory."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        if p.exists():
            return p
        return self.base_dir / p


# ---------------------------------------------------------------------------
# Augmentation helper
# ---------------------------------------------------------------------------

def _paired_random_flip_cube(
    cube1: np.ndarray,
    cube2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the SAME random flip to both spectral cubes.

    Identical to ``paired_random_flip`` but operates on (C, H, W) cubes
    instead of (H, W) images.  All channels are flipped consistently.

    Parameters
    ----------
    cube1, cube2 : (C, H, W) spectral cubes
    rng : NumPy random generator

    Returns
    -------
    (augmented_cube1, augmented_cube2)
    """
    flip_h = rng.random() > 0.5
    flip_v = rng.random() > 0.5

    if flip_h:
        cube1 = np.flip(cube1, axis=-1)   # Flip W dimension
        cube2 = np.flip(cube2, axis=-1)
    if flip_v:
        cube1 = np.flip(cube1, axis=-2)   # Flip H dimension
        cube2 = np.flip(cube2, axis=-2)

    return np.ascontiguousarray(cube1), np.ascontiguousarray(cube2)
