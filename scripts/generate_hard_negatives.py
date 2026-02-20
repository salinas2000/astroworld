#!/usr/bin/env python3
"""
Phase 10 — Generate hard-negative training dataset from real DSS2 images.

Strategy:
  1. HARD NEGATIVES: Extract 64×64 cutout pairs from real DSS2 images
     (self-paired). These contain real field stars, noise patterns, and
     artifacts — exactly the "confounders" the model must learn to ignore.

  2. POSITIVES ON REAL BACKGROUNDS: Extract 64×64 cutouts from real DSS2,
     then inject a synthetic moving source (S/N 3-15, ~8px motion).
     The model must detect the injected motion against real star clutter.

Output: training_catalog.csv with paired 64×64 FITS cutouts ready for
the MultiEpochDataset → P9SiameseDetector fine-tuning pipeline.

Usage:
  uv run python scripts/generate_hard_negatives.py \\
      --image-dir data/survey_images/dss2_red \\
      --output-dir data/training/hard_negatives \\
      --n-positive 500 --n-negative 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def make_gaussian_psf(size: int = 11, fwhm: float = 2.5) -> np.ndarray:
    """Normalized 2D Gaussian PSF."""
    sigma = fwhm / 2.3548
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, y)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def inject_source(image, cx, cy, flux, psf):
    """Inject point source into image."""
    result = image.copy()
    h, w = image.shape
    ph, pw = psf.shape
    ix, iy = int(round(cx)), int(round(cy))
    y0 = iy - ph // 2
    x0 = ix - pw // 2
    for j in range(ph):
        for i in range(pw):
            yy = y0 + j
            xx = x0 + i
            if 0 <= yy < h and 0 <= xx < w:
                result[yy, xx] += flux * psf[j, i]
    return result


def estimate_local_bg_std(cutout):
    """Estimate background std from a cutout using sigma-clipping."""
    arr = cutout.flatten().astype(np.float64)
    mask = np.ones(len(arr), dtype=bool)
    for _ in range(5):
        masked = arr[mask]
        if len(masked) == 0:
            return 1.0
        mean = masked.mean()
        std = masked.std()
        if std < 1e-10:
            return 1.0
        mask = np.abs(arr - mean) < 3.0 * std
    return max(float(arr[mask].std()), 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard-negative training data from real DSS2 images",
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/hard_negatives"))
    parser.add_argument("--n-positive", type=int, default=500)
    parser.add_argument("--n-negative", type=int, default=500)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--snr-min", type=float, default=3.0)
    parser.add_argument("--snr-max", type=float, default=15.0)
    parser.add_argument("--motion-px", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Scan for FITS files
    fits_files = sorted(args.image_dir.glob("*.fits"))
    if not fits_files:
        print(f"ERROR: No FITS files in {args.image_dir}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    ps = args.patch_size
    psf = make_gaussian_psf(size=11, fwhm=2.5)

    pos_dir = args.output_dir / "positive"
    neg_dir = args.output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PHASE 10 — HARD NEGATIVE DATASET GENERATOR")
    print("=" * 60)
    print()
    print(f"  FITS images:  {len(fits_files)}")
    print(f"  Patch size:   {ps}×{ps}")
    print(f"  Positives:    {args.n_positive} (injected on real backgrounds)")
    print(f"  Negatives:    {args.n_negative} (real DSS2 cutout pairs)")
    print(f"  S/N range:    {args.snr_min:.0f} - {args.snr_max:.0f}")
    print(f"  Motion:       {args.motion_px:.0f} px between epochs")
    print(f"  Output:       {args.output_dir}")
    print()

    # Pre-load all images
    print("Loading FITS images...")
    images = []
    for fp in fits_files:
        with fits.open(str(fp)) as hdul:
            data = hdul[0].data.astype(np.float64)
            images.append((fp.stem, data))
    print(f"  Loaded {len(images)} images")
    print()

    records = []

    # =====================================================================
    # NEGATIVES: Real DSS2 cutout pairs (no injected source)
    # =====================================================================
    print(f"Generating {args.n_negative} hard negative pairs...")

    for i in range(args.n_negative):
        # Random field
        field_name, image = images[rng.integers(len(images))]
        h, w = image.shape

        # Random cutout position (avoid edges)
        margin = ps + 10
        cx = rng.integers(margin, w - margin)
        cy = rng.integers(margin, h - margin)

        # Extract cutout from the SAME image for both epochs
        # (self-pair: nothing moved, so label = negative)
        # Add small independent noise to simulate different epoch readout
        cut1 = image[cy:cy + ps, cx:cx + ps].copy().astype(np.float32)
        cut2 = image[cy:cy + ps, cx:cx + ps].copy().astype(np.float32)

        # Add small readout noise difference between epochs
        bg_std = estimate_local_bg_std(cut1)
        cut1 += rng.normal(0, bg_std * 0.1, size=(ps, ps)).astype(np.float32)
        cut2 += rng.normal(0, bg_std * 0.1, size=(ps, ps)).astype(np.float32)

        # Save as paired FITS
        fname1 = f"neg_{i:04d}_epoch1.fits"
        fname2 = f"neg_{i:04d}_epoch2.fits"
        path1 = neg_dir / fname1
        path2 = neg_dir / fname2

        fits.PrimaryHDU(cut1).writeto(str(path1), overwrite=True)
        fits.PrimaryHDU(cut2).writeto(str(path2), overwrite=True)

        records.append({
            "example_id": f"neg_{i:04d}_epoch1",
            "fits_path": str(path1),
            "source_present": False,
            "pixel_x": float("nan"), "pixel_y": float("nan"),
            "snr": float("nan"), "flux_adu": float("nan"),
            "field": field_name,
        })
        records.append({
            "example_id": f"neg_{i:04d}_epoch2",
            "fits_path": str(path2),
            "source_present": False,
            "pixel_x": float("nan"), "pixel_y": float("nan"),
            "snr": float("nan"), "flux_adu": float("nan"),
            "field": field_name,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{args.n_negative}]")

    # =====================================================================
    # POSITIVES: Injected moving source on real DSS2 backgrounds
    # =====================================================================
    print(f"\nGenerating {args.n_positive} positive pairs (injection on real bg)...")

    for i in range(args.n_positive):
        # Random field
        field_name, image = images[rng.integers(len(images))]
        h, w = image.shape

        # Random cutout position
        margin = ps + 20
        cx = rng.integers(margin, w - margin)
        cy = rng.integers(margin, h - margin)

        # Extract cutouts for both epochs
        cut1 = image[cy:cy + ps, cx:cx + ps].copy().astype(np.float32)
        cut2 = image[cy:cy + ps, cx:cx + ps].copy().astype(np.float32)

        # Add readout noise
        bg_std = estimate_local_bg_std(cut1)
        cut1 += rng.normal(0, bg_std * 0.1, size=(ps, ps)).astype(np.float32)
        cut2 += rng.normal(0, bg_std * 0.1, size=(ps, ps)).astype(np.float32)

        # Inject moving source
        snr = rng.uniform(args.snr_min, args.snr_max)
        peak_adu = snr * bg_std
        flux = peak_adu / psf.max()

        # Source near center, random position
        src_x1 = ps / 2 + rng.uniform(-10, 10)
        src_y1 = ps / 2 + rng.uniform(-10, 10)

        # Motion direction
        angle = rng.uniform(0, 2 * np.pi)
        src_x2 = src_x1 + args.motion_px * np.cos(angle)
        src_y2 = src_y1 + args.motion_px * np.sin(angle)

        cut1 = inject_source(cut1, src_x1, src_y1, flux, psf)
        cut2 = inject_source(cut2, src_x2, src_y2, flux, psf)

        # Save
        fname1 = f"pair_{i:04d}_epoch1.fits"
        fname2 = f"pair_{i:04d}_epoch2.fits"
        path1 = pos_dir / fname1
        path2 = pos_dir / fname2

        fits.PrimaryHDU(cut1).writeto(str(path1), overwrite=True)
        fits.PrimaryHDU(cut2).writeto(str(path2), overwrite=True)

        records.append({
            "example_id": f"pair_{i:04d}_epoch1",
            "fits_path": str(path1),
            "source_present": True,
            "pixel_x": src_x1, "pixel_y": src_y1,
            "snr": snr, "flux_adu": flux,
            "field": field_name,
        })
        records.append({
            "example_id": f"pair_{i:04d}_epoch2",
            "fits_path": str(path2),
            "source_present": True,
            "pixel_x": src_x2, "pixel_y": src_y2,
            "snr": snr, "flux_adu": flux,
            "field": field_name,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{args.n_positive}] S/N={snr:.1f}")

    # Save catalog
    catalog = pd.DataFrame(records)
    catalog_path = args.output_dir / "training_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    print()
    print("=" * 60)
    print("  HARD NEGATIVE DATASET COMPLETE")
    print("=" * 60)
    print(f"  Total records:  {len(records)} ({args.n_positive + args.n_negative} pairs)")
    print(f"  Positive pairs: {args.n_positive} (moving source on real bg)")
    print(f"  Negative pairs: {args.n_negative} (real DSS2, no motion)")
    print(f"  Catalog:        {catalog_path}")
    print()
    print("  Fine-tune with:")
    print(f"  uv run python scripts/train_detector.py \\")
    print(f"      --data-dir {args.output_dir} --model siamese \\")
    print(f"      --resume checkpoints/siamese_sanity_64/best_model.pt \\")
    print(f"      --epochs 20 --lr 0.0005 --batch-size 32")


if __name__ == "__main__":
    main()
