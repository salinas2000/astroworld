#!/usr/bin/env python3
"""
Generate synthetic training data for Planet 9 ML detection.

Creates labeled FITS images with injected synthetic point sources
at various magnitudes and positions, suitable for training a
convolutional neural network or other ML classifier.

Usage:
  # Generate 100 positive + 100 negative examples from DSS2 images
  uv run python scripts/generate_training_data.py --image-dir data/survey_images/dss2_red

  # Custom magnitude range
  uv run python scripts/generate_training_data.py --image-dir data/survey_images/dss2_red --mag-min 20.0 --mag-max 24.0

  # PSF inspection only (no training data)
  uv run python scripts/generate_training_data.py --image-dir data/survey_images/dss2_red --psf-only

  # Multi-epoch pairs for shift-and-stack training
  uv run python scripts/generate_training_data.py --image-dir data/survey_images/dss2_red --multi-epoch --dt-days 30

  # Small test run
  uv run python scripts/generate_training_data.py --image-dir data/survey_images/dss2_red --n-positive 10 --n-negative 10
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from astroworld.imaging.psf_extraction import extract_psf, HAS_PHOTUTILS
from astroworld.imaging.fits_store import load_fits_image
from astroworld.imaging.synthetic_injection import estimate_zero_point
from astroworld.imaging.training_generator import (
    generate_training_set,
    generate_multi_epoch_pair,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for Planet 9 ML detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image-dir data/survey_images/dss2_red
  %(prog)s --image-dir data/survey_images/dss2_red --n-positive 50 --n-negative 50
  %(prog)s --image-dir data/survey_images/dss2_red --mag-min 20.0 --mag-max 24.0
  %(prog)s --image-dir data/survey_images/dss2_red --psf-only
  %(prog)s --image-dir data/survey_images/dss2_red --multi-epoch --dt-days 30
        """,
    )

    parser.add_argument(
        "--image-dir", type=Path, required=True,
        help="Directory containing FITS images (e.g., data/survey_images/dss2_red)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/training"),
        help="Output directory for training data (default: data/training)",
    )

    # Training set size
    parser.add_argument(
        "--n-positive", type=int, default=100,
        help="Number of positive examples (default: 100)",
    )
    parser.add_argument(
        "--n-negative", type=int, default=100,
        help="Number of negative examples (default: 100)",
    )

    # Magnitude range
    parser.add_argument(
        "--mag-min", type=float, default=18.0,
        help="Minimum injection magnitude (default: 18.0)",
    )
    parser.add_argument(
        "--mag-max", type=float, default=24.0,
        help="Maximum injection magnitude (default: 24.0)",
    )

    # PSF options
    parser.add_argument(
        "--fwhm", type=float, default=2.5,
        help="PSF FWHM estimate in pixels (default: 2.5)",
    )
    parser.add_argument(
        "--psf-size", type=int, default=25,
        help="PSF stamp size in pixels (default: 25)",
    )

    # Modes
    parser.add_argument(
        "--psf-only", action="store_true",
        help="Only extract and display PSF (no training data)",
    )
    parser.add_argument(
        "--multi-epoch", action="store_true",
        help="Generate multi-epoch pairs instead of single images",
    )
    parser.add_argument(
        "--dt-days", type=float, default=30.0,
        help="Epoch separation in days for multi-epoch mode (default: 30.0)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ASTROWORLD - SYNTHETIC TRAINING DATA GENERATOR")
    print("=" * 60)

    print(f"\n  photutils: {'available' if HAS_PHOTUTILS else 'NOT FOUND (using Gaussian PSF)'}")
    print(f"  Image dir: {args.image_dir}")

    # Find FITS files
    fits_files = sorted(args.image_dir.rglob("*.fits"))
    if len(fits_files) == 0:
        print(f"\n  ERROR: No FITS files found in {args.image_dir}")
        return

    print(f"  FITS files: {len(fits_files)}")

    # --- PSF-only mode ---
    if args.psf_only:
        print(f"\n  Mode: PSF extraction only")
        _run_psf_only(fits_files, args)
        return

    # --- Multi-epoch mode ---
    if args.multi_epoch:
        print(f"\n  Mode: Multi-epoch pair generation")
        _run_multi_epoch(fits_files, args)
        return

    # --- Standard training set ---
    print(f"\n  Mode: Training set generation")
    print(f"  Positive: {args.n_positive}")
    print(f"  Negative: {args.n_negative}")
    print(f"  Magnitude: {args.mag_min:.1f} - {args.mag_max:.1f}")
    print(f"  Output: {args.output_dir}")
    print()

    catalog = generate_training_set(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        magnitude_range=(args.mag_min, args.mag_max),
        fwhm_estimate=args.fwhm,
        psf_size=args.psf_size,
        seed=args.seed,
    )


def _run_psf_only(fits_files, args):
    """Extract PSF from first image and display diagnostics."""
    path = fits_files[0]
    print(f"\n  Source: {path.name}")

    image, header = load_fits_image(path)

    # Pixel scale info
    cdelt = abs(header.get("CDELT1", 0.000326))
    pixel_scale = cdelt * 3600.0
    print(f"  Pixel scale: {pixel_scale:.2f} arcsec/pixel")
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # Extract PSF
    print(f"\n  Extracting PSF (FWHM estimate={args.fwhm:.1f} px)...")
    result = extract_psf(
        image, header,
        fwhm_estimate=args.fwhm,
        psf_size=args.psf_size,
    )

    print(f"\n  PSF Extraction Results:")
    print(f"  " + "-" * 40)
    print(f"    Method:         {result.method}")
    print(f"    FWHM:           {result.fwhm_pixels:.2f} px "
          f"({result.fwhm_pixels * pixel_scale:.2f} arcsec)")
    print(f"    Stars detected: {result.n_stars_detected}")
    print(f"    Stars used:     {result.n_stars_used}")
    print(f"    PSF size:       {result.psf_size}x{result.psf_size}")
    print(f"    Background:     {result.background_mean:.1f} +/- "
          f"{result.background_std:.1f} ADU")
    print(f"    PSF peak:       {result.psf_image.max():.4f}")
    print(f"    PSF sum:        {result.psf_image.sum():.2f}")

    # Zero point
    zp = estimate_zero_point(
        image, header,
        star_positions=result.star_positions,
        star_fluxes=result.star_fluxes,
    )
    print(f"    Zero point:     {zp:.2f}")

    # Save PSF stamp
    args.output_dir.mkdir(parents=True, exist_ok=True)
    psf_path = args.output_dir / "psf_stamp.fits"
    psf_hdu = fits.PrimaryHDU(result.psf_image.astype(np.float32))
    psf_hdu.header["FWHM"] = result.fwhm_pixels
    psf_hdu.header["METHOD"] = result.method
    psf_hdu.header["NSTARS"] = result.n_stars_used
    psf_hdu.writeto(str(psf_path), overwrite=True)
    print(f"\n  PSF saved: {psf_path}")


def _run_multi_epoch(fits_files, args):
    """Generate multi-epoch pairs."""
    if len(fits_files) < 2:
        print("  ERROR: Need at least 2 FITS files for multi-epoch mode")
        return

    rng = np.random.default_rng(args.seed)

    # Extract PSF from first image
    image, header = load_fits_image(fits_files[0])
    psf_result = extract_psf(
        image, header,
        fwhm_estimate=args.fwhm,
        psf_size=args.psf_size,
    )

    print(f"  PSF: {psf_result.method} (FWHM={psf_result.fwhm_pixels:.2f} px)")
    print(f"  Dt: {args.dt_days:.1f} days")
    print(f"  Generating {args.n_positive} pairs...")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_positive):
        idx_1 = rng.integers(0, len(fits_files))
        idx_2 = rng.integers(0, len(fits_files))
        mag = rng.uniform(args.mag_min, args.mag_max)

        ex1, ex2 = generate_multi_epoch_pair(
            fits_path_1=fits_files[idx_1],
            fits_path_2=fits_files[idx_2],
            output_dir=args.output_dir,
            psf_result=psf_result,
            magnitude=mag,
            dt_days=args.dt_days,
            rng=rng,
            pair_id=f"pair_{i + 1:04d}",
        )

        if (i + 1) % max(1, args.n_positive // 5) == 0:
            dx = ex2.pixel_x - ex1.pixel_x
            dy = ex2.pixel_y - ex1.pixel_y
            disp = np.sqrt(dx**2 + dy**2)
            print(f"  [{i + 1}/{args.n_positive}] mag={mag:.1f}, "
                  f"displacement={disp:.1f} px")

    print(f"\n  Multi-epoch pairs complete: {args.output_dir}")


if __name__ == "__main__":
    main()
