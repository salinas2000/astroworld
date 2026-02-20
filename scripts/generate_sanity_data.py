#!/usr/bin/env python3
"""
Generate a simple sanity-check dataset for CNN validation.

Creates 512x512 synthetic images on FLAT Gaussian noise backgrounds
with/without a single Gaussian point source. No real field stars —
just noise + optional PSF.

This isolates the CNN learning problem: can it find a single bright
Gaussian bump in a noisy image?

Usage:
  uv run python scripts/generate_sanity_data.py
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

    # Integer position
    ix, iy = int(round(px)), int(round(py))

    # PSF stamp bounds
    y0 = iy - ph // 2
    x0 = ix - pw // 2

    for j in range(ph):
        for i in range(pw):
            yy = y0 + j
            xx = x0 + i
            if 0 <= yy < h and 0 <= xx < w:
                result[yy, xx] += flux * psf[j, i]

    return result


def main():
    output_dir = Path("data/training/sanity_check")
    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Parameters
    img_size = 512
    bg_mean = 1000.0  # ADU background level
    bg_std = 100.0  # ADU background noise
    n_positive = 500
    n_negative = 500
    psf_fwhm = 2.5
    psf = make_gaussian_psf(size=11, fwhm=psf_fwhm)

    # S/N range for sanity check: 5-30 (clearly detectable)
    snr_min = 5.0
    snr_max = 30.0

    print("=" * 60)
    print("  SANITY CHECK DATASET GENERATOR")
    print("=" * 60)
    print()
    print(f"  Image size:  {img_size}x{img_size}")
    print(f"  Background:  {bg_mean:.0f} +/- {bg_std:.0f} ADU")
    print(f"  PSF FWHM:    {psf_fwhm:.1f} px")
    print(f"  S/N range:   {snr_min:.0f} - {snr_max:.0f}")
    print(f"  Positive:    {n_positive}")
    print(f"  Negative:    {n_negative}")
    print()

    records = []

    # Positive examples: noise + single Gaussian source
    print(f"Generating {n_positive} positive examples...")
    for i in range(n_positive):
        # Random noise background
        image = rng.normal(bg_mean, bg_std, size=(img_size, img_size)).astype(np.float32)

        # Random source position (avoid edges)
        margin = 20
        px = rng.uniform(margin, img_size - margin)
        py = rng.uniform(margin, img_size - margin)

        # Random S/N -> flux
        snr = rng.uniform(snr_min, snr_max)
        # peak flux = snr * bg_std (for Gaussian PSF, peak = flux * psf.max())
        peak_adu = snr * bg_std
        flux = peak_adu / psf.max()

        # Inject source
        image = inject_source(image, px, py, flux, psf)

        # Save FITS
        fname = f"pos_{i:04d}.fits"
        path = pos_dir / fname
        hdu = fits.PrimaryHDU(image)
        hdu.writeto(str(path), overwrite=True)

        records.append({
            "example_id": f"pos_{i:04d}",
            "fits_path": str(path),
            "source_present": True,
            "ra_deg": float("nan"),
            "dec_deg": float("nan"),
            "pixel_x": px,
            "pixel_y": py,
            "magnitude": float("nan"),
            "flux_adu": flux,
            "snr": snr,
            "source_fits": "synthetic",
            "survey": "sanity_check",
            "psf_fwhm": psf_fwhm,
            "zero_point": float("nan"),
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_positive}] S/N={snr:.1f}")

    # Negative examples: pure noise
    print(f"Generating {n_negative} negative examples...")
    for i in range(n_negative):
        image = rng.normal(bg_mean, bg_std, size=(img_size, img_size)).astype(np.float32)

        fname = f"neg_{i:04d}.fits"
        path = neg_dir / fname
        hdu = fits.PrimaryHDU(image)
        hdu.writeto(str(path), overwrite=True)

        records.append({
            "example_id": f"neg_{i:04d}",
            "fits_path": str(path),
            "source_present": False,
            "ra_deg": float("nan"),
            "dec_deg": float("nan"),
            "pixel_x": float("nan"),
            "pixel_y": float("nan"),
            "magnitude": float("nan"),
            "flux_adu": float("nan"),
            "snr": float("nan"),
            "source_fits": "synthetic",
            "survey": "sanity_check",
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
    print("  SANITY CHECK DATA COMPLETE")
    print("=" * 60)
    print(f"  Total: {len(records)} examples")
    print(f"  Catalog: {catalog_path}")
    print()

    # Quick verification
    pos_data = catalog[catalog["source_present"] == True]
    print(f"  S/N range: {pos_data['snr'].min():.1f} - {pos_data['snr'].max():.1f}")
    print(f"  S/N mean:  {pos_data['snr'].mean():.1f}")
    print()
    print("  Train with:")
    print("  uv run python scripts/train_detector.py \\")
    print("      --data-dir data/training/sanity_check --epochs 30")


if __name__ == "__main__":
    main()
