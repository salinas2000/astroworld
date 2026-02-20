"""
PyTorch datasets for loading FITS images from the training catalog.

Provides two dataset classes:
  - FITSDataset: single-image binary classification (source / no source)
  - MultiEpochDataset: paired multi-epoch images for Siamese detection

Images are normalized via iterative sigma-clipping (matching the
background estimation from psf_extraction.py) so that background
pixels have mean~0, std~1, and sources appear as positive excursions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from astroworld.imaging.fits_store import load_fits_image


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize a FITS image to zero-mean, unit-std using sigma-clipping.

    Iterative 3-sigma clipping (5 iterations) estimates background
    statistics while rejecting bright stars.  This matches the
    ``estimate_background`` approach in ``psf_extraction.py``.

    Parameters
    ----------
    image : 2D numpy array (FITS pixel data)

    Returns
    -------
    Normalized image (float32) with background ~ N(0, 1)
    """
    arr = image.astype(np.float64)
    mask = np.ones(arr.shape, dtype=bool)

    for _ in range(5):
        masked = arr[mask]
        if len(masked) == 0:
            break
        mean = masked.mean()
        std = masked.std()
        if std < 1e-10:
            break
        mask = np.abs(arr - mean) < 3.0 * std

    masked = arr[mask]
    mean = masked.mean() if len(masked) > 0 else 0.0
    std = masked.std() if len(masked) > 0 else 1.0
    std = max(std, 1e-6)  # avoid division by zero

    return ((arr - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentations (numpy-only, no torchvision dependency)
# ---------------------------------------------------------------------------

def random_flip(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random horizontal and/or vertical flips.

    Astronomical images have no preferred orientation, so flips are
    physically valid augmentations for single-image classification.

    Parameters
    ----------
    image : 2D or 3D numpy array
    rng : NumPy random generator

    Returns
    -------
    Augmented image (may share memory with input)
    """
    if rng.random() > 0.5:
        image = np.flip(image, axis=-1)
    if rng.random() > 0.5:
        image = np.flip(image, axis=-2)
    return np.ascontiguousarray(image)


def paired_random_flip(
    image1: np.ndarray,
    image2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the SAME random flip to both images in a pair.

    For multi-epoch Siamese detection, both epochs must receive
    identical augmentation.  Otherwise, flipping one epoch but not
    the other would create a false large-scale motion signal that
    overwhelms the subtle ~8 pixel P9 shift.

    Parameters
    ----------
    image1, image2 : 2D or 3D numpy arrays of the same shape
    rng : NumPy random generator

    Returns
    -------
    (augmented_image1, augmented_image2) with identical flips applied
    """
    flip_h = rng.random() > 0.5
    flip_v = rng.random() > 0.5

    if flip_h:
        image1 = np.flip(image1, axis=-1)
        image2 = np.flip(image2, axis=-1)
    if flip_v:
        image1 = np.flip(image1, axis=-2)
        image2 = np.flip(image2, axis=-2)

    return np.ascontiguousarray(image1), np.ascontiguousarray(image2)


# ---------------------------------------------------------------------------
# Train/val split (pure numpy, no sklearn)
# ---------------------------------------------------------------------------

def split_train_val(
    catalog: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split by ``source_present`` column.

    Parameters
    ----------
    catalog : DataFrame with ``source_present`` boolean column
    val_fraction : Fraction reserved for validation (0-1)
    seed : Random seed for reproducibility

    Returns
    -------
    (train_indices, val_indices) as numpy integer arrays
    """
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []

    for label in [True, False]:
        group = catalog.index[catalog["source_present"] == label].to_numpy().copy()
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_fraction))
        val_idx.extend(group[:n_val])
        train_idx.extend(group[n_val:])

    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)


# ---------------------------------------------------------------------------
# FITSDataset — single-image classification
# ---------------------------------------------------------------------------

class FITSDataset(Dataset):
    """PyTorch Dataset for FITS images from the training catalog.

    Loads FITS images listed in ``training_catalog.csv``, normalizes
    using iterative sigma-clipping, and returns ``(image, label)``
    pairs for binary classification (source present / not present).

    Parameters
    ----------
    catalog_path : Path to ``training_catalog.csv``
    do_normalize : Apply sigma-clip normalization (default True)
    augment : Apply random flips during training (default False)
    indices : Subset of catalog row indices (for train/val split)
    """

    def __init__(
        self,
        catalog_path: Path | str,
        do_normalize: bool = True,
        augment: bool = False,
        indices: np.ndarray | None = None,
    ):
        self.catalog_path = Path(catalog_path)
        self.catalog = pd.read_csv(self.catalog_path)
        self.base_dir = self.catalog_path.parent
        self.do_normalize = do_normalize
        self.augment = augment
        self._rng = np.random.default_rng()

        if indices is not None:
            self.catalog = self.catalog.iloc[indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.catalog)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        """Load and return (image_tensor, label).

        Returns
        -------
        image : Tensor of shape (1, H, W), float32
        label : 1.0 if source present, 0.0 otherwise
        """
        row = self.catalog.iloc[idx]

        # Resolve FITS path
        fits_path = Path(row["fits_path"])
        if not fits_path.is_absolute():
            # Try path as-is first (relative to CWD / project root)
            if not fits_path.exists():
                # Fall back to relative to catalog directory
                fits_path = self.base_dir / fits_path

        image, _header = load_fits_image(fits_path)

        if self.do_normalize:
            image = normalize_image(image)
        else:
            image = image.astype(np.float32)

        if self.augment:
            image = random_flip(image, self._rng)

        # Add channel dimension: (H, W) -> (1, H, W)
        tensor = torch.from_numpy(image).unsqueeze(0)

        label = 1.0 if row["source_present"] else 0.0

        return tensor, label


# ---------------------------------------------------------------------------
# MultiEpochDataset — paired images for Siamese detection
# ---------------------------------------------------------------------------

class MultiEpochDataset(Dataset):
    """PyTorch Dataset for paired multi-epoch images.

    Each sample is a pair of images (epoch 1, epoch 2) with a label
    indicating whether a moving source is present.  Pairs are linked
    by shared ``pair_*`` prefix in the ``example_id`` column.

    Parameters
    ----------
    catalog_path : Path to ``training_catalog.csv`` (multi-epoch format)
    do_normalize : Apply sigma-clip normalization (default True)
    augment : Apply paired random flips (default False)
    indices : Subset of pair indices
    """

    def __init__(
        self,
        catalog_path: Path | str,
        do_normalize: bool = True,
        augment: bool = False,
        indices: np.ndarray | None = None,
    ):
        self.catalog_path = Path(catalog_path)
        self.catalog = pd.read_csv(self.catalog_path)
        self.base_dir = self.catalog_path.parent
        self.do_normalize = do_normalize
        self.augment = augment
        self._rng = np.random.default_rng()

        # Group pairs by prefix: pair_0001_epoch1 + pair_0001_epoch2
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Load and return (image1, image2, label).

        Returns
        -------
        image1 : Tensor (1, H, W) for epoch 1
        image2 : Tensor (1, H, W) for epoch 2
        label : 1.0 if moving source present, 0.0 otherwise
        """
        idx1, idx2 = self.pairs[idx]
        row1 = self.catalog.iloc[idx1]
        row2 = self.catalog.iloc[idx2]

        # Load both images
        path1 = Path(row1["fits_path"])
        path2 = Path(row2["fits_path"])
        if not path1.is_absolute():
            if not path1.exists():
                path1 = self.base_dir / path1
        if not path2.is_absolute():
            if not path2.exists():
                path2 = self.base_dir / path2

        img1, _ = load_fits_image(path1)
        img2, _ = load_fits_image(path2)

        if self.do_normalize:
            img1 = normalize_image(img1)
            img2 = normalize_image(img2)
        else:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        if self.augment:
            # CRITICAL: same augmentation applied to both epochs
            img1, img2 = paired_random_flip(img1, img2, self._rng)

        tensor1 = torch.from_numpy(img1).unsqueeze(0)
        tensor2 = torch.from_numpy(img2).unsqueeze(0)

        label = 1.0 if row1["source_present"] else 0.0

        return tensor1, tensor2, label
