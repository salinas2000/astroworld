#!/usr/bin/env python3
"""
Generate multi-epoch Siamese sanity check dataset.

Creates paired 512x512 synthetic images:
  - Positive: same noise + a source that MOVES between epochs (~8 px)
  - Negative: same noise field, no source (or same noise + random stars
    that appear identically in both epochs)

This tests whether the Siamese network can detect the *correlated motion*
between epochs, which is the key signature of Planet 9.

S/N range: 2-10 (deliberately challenging — must detect motion, not brightness)

Usage:
  uv run python scripts/generate_siamese_sanity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def make_gaussian_psf(size: int = 11, fwhm: float = 2.5) -> np.ndarray:
    """Create a 2D Gaussian PSF normalized to sum=1."""
    sigma = fwhm / 2.3548
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, y)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def inject_source(
    image: np.ndarray,
    px: float,
    py: float,
    flux: float,
    psf: np.ndarray,
) -> np.ndarray:
    """Inject a point source into the image at sub-pixel position."""
    result = image.copy()
    h, w = image.shape
    ph, pw = psf.shape
    ix, iy = int(round(px)), int(round(py))
    y0 = iy - ph // 2
    x0 = ix - pw // 2

    for j in range(ph):
        for i in range(pw):
            yy = y0 + j
            xx = x0 + i
            if 0 <= yy < h and 0 <= xx < w:
                result[yy, xx] += flux * psf[j, i]

    return result


def add_static_stars(
    image: np.ndarray,
    n_stars: int,
    rng: np.random.Generator,
    psf: np.ndarray,
    bg_std: float,
) -> np.ndarray:
    """Add random static field stars to an image."""
    h, w = image.shape
    result = image.copy()
    margin = 20

    for _ in range(n_stars):
        px = rng.uniform(margin, w - margin)
        py = rng.uniform(margin, h - margin)
        # Stars with S/N 5-50 (much brighter than the moving source)
        snr = rng.uniform(5, 50)
        peak_adu = snr * bg_std
        flux = peak_adu / psf.max()
        result = inject_source(result, px, py, flux, psf)

    return result


def main():
    output_dir = Path("data/training/siamese_sanity")
    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Parameters
    img_size = 512
    bg_mean = 1000.0
    bg_std = 100.0
    n_positive = 500
    n_negative = 500
    psf_fwhm = 2.5
    psf = make_gaussian_psf(size=11, fwhm=psf_fwhm)
    n_field_stars = 50  # Static stars per image

    # Moving source parameters
    snr_min = 2.0   # Deliberately faint — tests motion detection
    snr_max = 10.0
    motion_px = 8.0  # P9 typical motion between epochs

    print("=" * 60)
    print("  SIAMESE SANITY CHECK DATASET GENERATOR")
    print("=" * 60)
    print()
    print(f"  Image size:    {img_size}x{img_size}")
    print(f"  Background:    {bg_mean:.0f} +/- {bg_std:.0f} ADU")
    print(f"  PSF FWHM:      {psf_fwhm:.1f} px")
    print(f"  Field stars:   {n_field_stars} per image (same in both epochs)")
    print(f"  Moving source: S/N {snr_min:.0f}-{snr_max:.0f}, {motion_px:.0f} px shift")
    print(f"  Positive:      {n_positive} pairs (noise + static stars + moving source)")
    print(f"  Negative:      {n_negative} pairs (noise + static stars, no moving source)")
    print()

    records = []

    # === Positive pairs: static stars + one moving source ===
    print(f"Generating {n_positive} positive pairs...")
    for i in range(n_positive):
        # SAME noise base for both epochs (simulating same field)
        base_noise = rng.normal(bg_mean, bg_std, size=(img_size, img_size)).astype(np.float32)

        # Add independent noise component (readout noise, different epoch)
        noise1 = base_noise + rng.normal(0, bg_std * 0.3, size=(img_size, img_size)).astype(np.float32)
        noise2 = base_noise + rng.normal(0, bg_std * 0.3, size=(img_size, img_size)).astype(np.float32)

        # Same static stars in both epochs (seed-controlled)
        star_seed = rng.integers(0, 2**31)
        star_rng1 = np.random.default_rng(star_seed)
        star_rng2 = np.random.default_rng(star_seed)
        img1 = add_static_stars(noise1, n_field_stars, star_rng1, psf, bg_std)
        img2 = add_static_stars(noise2, n_field_stars, star_rng2, psf, bg_std)

        # Moving source in both epochs
        margin = 30
        px1 = rng.uniform(margin, img_size - margin)
        py1 = rng.uniform(margin, img_size - margin)

        # Random motion direction
        angle = rng.uniform(0, 2 * np.pi)
        px2 = px1 + motion_px * np.cos(angle)
        py2 = py1 + motion_px * np.sin(angle)

        # Same flux in both epochs
        snr = rng.uniform(snr_min, snr_max)
        peak_adu = snr * bg_std
        flux = peak_adu / psf.max()

        img1 = inject_source(img1, px1, py1, flux, psf)
        img2 = inject_source(img2, px2, py2, flux, psf)

        # Save FITS
        fname1 = f"pair_{i:04d}_epoch1.fits"
        fname2 = f"pair_{i:04d}_epoch2.fits"
        path1 = pos_dir / fname1
        path2 = pos_dir / fname2
        fits.PrimaryHDU(img1).writeto(str(path1), overwrite=True)
        fits.PrimaryHDU(img2).writeto(str(path2), overwrite=True)

        records.append({
            "example_id": f"pair_{i:04d}_epoch1",
            "fits_path": str(path1),
            "source_present": True,
            "ra_deg": float("nan"),
            "dec_deg": float("nan"),
            "pixel_x": px1,
            "pixel_y": py1,
            "magnitude": float("nan"),
            "flux_adu": flux,
            "snr": snr,
            "source_fits": "synthetic",
            "survey": "siamese_sanity",
            "psf_fwhm": psf_fwhm,
            "zero_point": float("nan"),
        })
        records.append({
            "example_id": f"pair_{i:04d}_epoch2",
            "fits_path": str(path2),
            "source_present": True,
            "ra_deg": float("nan"),
            "dec_deg": float("nan"),
            "pixel_x": px2,
            "pixel_y": py2,
            "magnitude": float("nan"),
            "flux_adu": flux,
            "snr": snr,
            "source_fits": "synthetic",
            "survey": "siamese_sanity",
            "psf_fwhm": psf_fwhm,
            "zero_point": float("nan"),
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_positive}] S/N={snr:.1f}")

    # === Negative pairs: same static stars, NO moving source ===
    print(f"Generating {n_negative} negative pairs...")
    for i in range(n_negative):
        base_noise = rng.normal(bg_mean, bg_std, size=(img_size, img_size)).astype(np.float32)
        noise1 = base_noise + rng.normal(0, bg_std * 0.3, size=(img_size, img_size)).astype(np.float32)
        noise2 = base_noise + rng.normal(0, bg_std * 0.3, size=(img_size, img_size)).astype(np.float32)

        star_seed = rng.integers(0, 2**31)
        star_rng1 = np.random.default_rng(star_seed)
        star_rng2 = np.random.default_rng(star_seed)
        img1 = add_static_stars(noise1, n_field_stars, star_rng1, psf, bg_std)
        img2 = add_static_stars(noise2, n_field_stars, star_rng2, psf, bg_std)

        fname1 = f"pair_{i:04d}_epoch1.fits"
        fname2 = f"pair_{i:04d}_epoch2.fits"
        path1 = neg_dir / fname1
        path2 = neg_dir / fname2
        fits.PrimaryHDU(img1).writeto(str(path1), overwrite=True)
        fits.PrimaryHDU(img2).writeto(str(path2), overwrite=True)

        records.append({
            "example_id": f"neg_{i:04d}_epoch1",
            "fits_path": str(path1),
            "source_present": False,
            "ra_deg": float("nan"),
            "dec_deg": float("nan"),
            "pixel_x": float("nan"),
            "pixel_y": float("nan"),
            "magnitude": float("nan"),
            "flux_adu": float("nan"),
            "snr": float("nan"),
            "source_fits": "synthetic",
            "survey": "siamese_sanity",
            "psf_fwhm": psf_fwhm,
            "zero_point": float("nan"),
        })
        records.append({
            "example_id": f"neg_{i:04d}_epoch2",
            "fits_path": str(path2),
            "source_present": False,
            "ra_deg": float("nan"),
            "dec_deg": float("nan"),
            "pixel_x": float("nan"),
            "pixel_y": float("nan"),
            "magnitude": float("nan"),
            "flux_adu": float("nan"),
            "snr": float("nan"),
            "source_fits": "synthetic",
            "survey": "siamese_sanity",
            "psf_fwhm": psf_fwhm,
            "zero_point": float("nan"),
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_negative}]")

    # Save catalog
    catalog = pd.DataFrame(records)
    catalog_path = output_dir / "training_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    print()
    print("=" * 60)
    print("  SIAMESE SANITY CHECK DATA COMPLETE")
    print("=" * 60)
    print(f"  Total records: {len(records)} ({n_positive + n_negative} pairs)")
    print(f"  Catalog: {catalog_path}")
    print()

    pos_data = catalog[catalog["source_present"] == True]
    print(f"  Moving source S/N: {pos_data['snr'].min():.1f} - {pos_data['snr'].max():.1f}")
    print(f"  Motion: {motion_px:.0f} pixels between epochs")
    print()
    print("  Train with:")
    print("  uv run python scripts/train_detector.py \\")
    print("      --data-dir data/training/siamese_sanity --model siamese --epochs 30")


if __name__ == "__main__":
    main()
