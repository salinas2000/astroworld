#!/usr/bin/env python3
"""
Siamese sanity check with 64x64 cutouts (instead of 512x512).

The key insight: full 512x512 images overwhelm the subtle 8px motion signal.
By using 64x64 cutouts centered on the source location, the CNN can focus
on the relevant spatial information.

This simulates a realistic detection pipeline where a pre-filter identifies
candidate regions and the CNN classifies cutouts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def make_gaussian_psf(size: int = 11, fwhm: float = 2.5) -> np.ndarray:
    sigma = fwhm / 2.3548
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, y)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def inject_source(image, px, py, flux, psf):
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


def main():
    output_dir = Path("data/training/siamese_sanity_64")
    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Parameters
    img_size = 64  # Small cutouts!
    bg_mean = 1000.0
    bg_std = 100.0
    n_positive = 500
    n_negative = 500
    psf_fwhm = 2.5
    psf = make_gaussian_psf(size=11, fwhm=psf_fwhm)

    # Moving source: S/N 3-15, motion 8 px
    snr_min = 3.0
    snr_max = 15.0
    motion_px = 8.0

    print("=" * 60)
    print("  SIAMESE SANITY CHECK — 64x64 CUTOUTS")
    print("=" * 60)
    print()
    print(f"  Image size:    {img_size}x{img_size}")
    print(f"  Background:    {bg_mean:.0f} +/- {bg_std:.0f} ADU")
    print(f"  Moving source: S/N {snr_min:.0f}-{snr_max:.0f}, {motion_px:.0f} px shift")
    print(f"  Pairs:         {n_positive} pos + {n_negative} neg")
    print()

    records = []

    # Positive: source moves between epochs
    print(f"Generating {n_positive} positive pairs...")
    for i in range(n_positive):
        img1 = rng.normal(bg_mean, bg_std, size=(img_size, img_size)).astype(np.float32)
        img2 = rng.normal(bg_mean, bg_std, size=(img_size, img_size)).astype(np.float32)

        # Source near center of cutout
        cx, cy = img_size / 2, img_size / 2
        px1 = cx + rng.uniform(-5, 5)
        py1 = cy + rng.uniform(-5, 5)

        angle = rng.uniform(0, 2 * np.pi)
        px2 = px1 + motion_px * np.cos(angle)
        py2 = py1 + motion_px * np.sin(angle)

        snr = rng.uniform(snr_min, snr_max)
        peak_adu = snr * bg_std
        flux = peak_adu / psf.max()

        img1 = inject_source(img1, px1, py1, flux, psf)
        img2 = inject_source(img2, px2, py2, flux, psf)

        path1 = pos_dir / f"pair_{i:04d}_epoch1.fits"
        path2 = pos_dir / f"pair_{i:04d}_epoch2.fits"
        fits.PrimaryHDU(img1).writeto(str(path1), overwrite=True)
        fits.PrimaryHDU(img2).writeto(str(path2), overwrite=True)

        records.append({
            "example_id": f"pair_{i:04d}_epoch1",
            "fits_path": str(path1), "source_present": True,
            "ra_deg": 0, "dec_deg": 0, "pixel_x": px1, "pixel_y": py1,
            "magnitude": 0, "flux_adu": flux, "snr": snr,
            "source_fits": "synthetic", "survey": "siamese_sanity_64",
            "psf_fwhm": psf_fwhm, "zero_point": 0,
        })
        records.append({
            "example_id": f"pair_{i:04d}_epoch2",
            "fits_path": str(path2), "source_present": True,
            "ra_deg": 0, "dec_deg": 0, "pixel_x": px2, "pixel_y": py2,
            "magnitude": 0, "flux_adu": flux, "snr": snr,
            "source_fits": "synthetic", "survey": "siamese_sanity_64",
            "psf_fwhm": psf_fwhm, "zero_point": 0,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_positive}] S/N={snr:.1f}")

    # Negative: same noise, no source
    print(f"Generating {n_negative} negative pairs...")
    for i in range(n_negative):
        img1 = rng.normal(bg_mean, bg_std, size=(img_size, img_size)).astype(np.float32)
        img2 = rng.normal(bg_mean, bg_std, size=(img_size, img_size)).astype(np.float32)

        path1 = neg_dir / f"pair_{i:04d}_epoch1.fits"
        path2 = neg_dir / f"pair_{i:04d}_epoch2.fits"
        fits.PrimaryHDU(img1).writeto(str(path1), overwrite=True)
        fits.PrimaryHDU(img2).writeto(str(path2), overwrite=True)

        records.append({
            "example_id": f"neg_{i:04d}_epoch1",
            "fits_path": str(path1), "source_present": False,
            "ra_deg": 0, "dec_deg": 0, "pixel_x": 0, "pixel_y": 0,
            "magnitude": 0, "flux_adu": 0, "snr": 0,
            "source_fits": "synthetic", "survey": "siamese_sanity_64",
            "psf_fwhm": psf_fwhm, "zero_point": 0,
        })
        records.append({
            "example_id": f"neg_{i:04d}_epoch2",
            "fits_path": str(path2), "source_present": False,
            "ra_deg": 0, "dec_deg": 0, "pixel_x": 0, "pixel_y": 0,
            "magnitude": 0, "flux_adu": 0, "snr": 0,
            "source_fits": "synthetic", "survey": "siamese_sanity_64",
            "psf_fwhm": psf_fwhm, "zero_point": 0,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_negative}]")

    catalog = pd.DataFrame(records)
    catalog_path = output_dir / "training_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    print()
    print("=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Total: {n_positive + n_negative} pairs")
    print(f"  Catalog: {catalog_path}")


if __name__ == "__main__":
    main()
