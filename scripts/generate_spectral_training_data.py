#!/usr/bin/env python
"""
Generate synthetic multi-survey training data for the SSTF spectral model.

Creates paired 64x64 images in 3 bands (optical, WISE W1, WISE W2) with
physically motivated infrared flux ratios based on Planck blackbody law.

Positive pairs: Gaussian noise background + PSF point source injected with
  - Optical flux from S/N ratio
  - W1/W2 flux from Planck law at chosen temperature (25-60 K)
  - Source moves between epochs by ``motion_px`` pixels
  - ±5% color noise on IR flux ratio (simulates non-blackbody effects)

Negative pairs: Pure Gaussian noise in all bands (no source)

Output: FITS files + training_catalog.csv matching MultiSurveyDataset schema.

Usage
-----
    uv run python scripts/generate_spectral_training_data.py \\
        --output-dir data/training/spectral \\
        --n-positive 300 --n-negative 300 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from astroworld.imaging.spectral_cube import (
    planck_spectral_radiance,
    planck_flux_ratio,
    WISE_W1_WAVELENGTH_UM,
    WISE_W2_WAVELENGTH_UM,
)


# ---------------------------------------------------------------------------
# Background parameters per band
# ---------------------------------------------------------------------------

# Optical (DSS2 Red): high background, moderate noise
OPTICAL_BG_MEAN = 1000.0
OPTICAL_BG_STD = 100.0

# WISE W1 (3.4 um): lower background, higher noise (thermal)
W1_BG_MEAN = 500.0
W1_BG_STD = 80.0

# WISE W2 (4.6 um): higher background (thermal), higher noise
W2_BG_MEAN = 800.0
W2_BG_STD = 120.0

# PSF parameters
OPTICAL_PSF_FWHM = 2.5      # arcsec (DSS2/Pan-STARRS typical)
WISE_PSF_FWHM = 6.0         # arcsec (WISE typical ~6 arcsec)

# Epoch parameters
EPOCH1_JD = 2455400.0        # ~2010.7 (WISE all-sky)
DEFAULT_DELTA_T = 30.0       # days between epochs

# Optical wavelength for Planck ratio computation
OPTICAL_WAVELENGTH_UM = 0.65  # DSS2 Red ~0.65 um


# ---------------------------------------------------------------------------
# PSF generation
# ---------------------------------------------------------------------------

def make_gaussian_psf(size: int = 11, fwhm: float = 2.5) -> np.ndarray:
    """Create a normalized Gaussian PSF kernel."""
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
    """Inject a point source into an image at sub-pixel position."""
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


# ---------------------------------------------------------------------------
# IR flux computation with Planck physics + color noise
# ---------------------------------------------------------------------------

def compute_ir_fluxes(
    optical_flux: float,
    temperature_k: float,
    ir_boost: float,
    rng: np.random.Generator,
    color_noise_frac: float = 0.05,
) -> tuple[float, float]:
    """Compute WISE W1/W2 flux from Planck blackbody law.

    The W1/W2 flux **ratio** is determined by the Planck function at the
    given temperature.  The absolute IR flux is scaled relative to the
    optical flux via ``ir_boost``, simulating that cold bodies are much
    brighter in infrared than optical.

    A ±5% random perturbation on the W1/W2 ratio simulates non-blackbody
    spectral features (methane ice, nitrogen ice, atmospheric effects)
    that real Planet 9 might have.

    Parameters
    ----------
    optical_flux : float
        Total optical flux (ADU) of the injected source.
    temperature_k : float
        Blackbody temperature in Kelvin.
    ir_boost : float
        Multiplicative factor for IR brightness relative to optical.
        Typically 2-10 for cold bodies in synthetic training data.
    rng : numpy random Generator
        For reproducible color noise.
    color_noise_frac : float
        Fractional noise on W1/W2 ratio (default 0.05 = ±5%).

    Returns
    -------
    (flux_w1, flux_w2) : tuple of float
        WISE W1 and W2 fluxes in ADU.
    """
    # Theoretical W1/W2 ratio from Planck law
    ratio_w1_w2 = planck_flux_ratio(temperature_k)

    # Add color noise: ±5% perturbation on the ratio
    # Simulates non-blackbody effects (methane, nitrogen ices)
    noise_factor = 1.0 + rng.uniform(-color_noise_frac, color_noise_frac)
    ratio_w1_w2 *= noise_factor

    # Base IR flux (W2 as reference, scaled from optical)
    flux_w2 = optical_flux * ir_boost

    # W1 from the Planck ratio
    flux_w1 = flux_w2 * max(ratio_w1_w2, 0.01)  # Prevent zero

    return flux_w1, flux_w2


# ---------------------------------------------------------------------------
# FITS I/O with minimal WCS header
# ---------------------------------------------------------------------------

def _make_minimal_header(
    nx: int,
    ny: int,
    pixel_scale_arcsec: float = 1.7,
) -> fits.Header:
    """Create a minimal FITS header with WCS placeholder."""
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 30.0
    header["CRPIX1"] = (nx + 1) / 2.0
    header["CRPIX2"] = (ny + 1) / 2.0
    header["CDELT1"] = -pixel_scale_arcsec / 3600.0
    header["CDELT2"] = pixel_scale_arcsec / 3600.0
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    return header


def save_fits(image: np.ndarray, path: Path, pixel_scale: float = 1.7):
    """Save a 2D image as a FITS file with minimal WCS."""
    header = _make_minimal_header(image.shape[1], image.shape[0], pixel_scale)
    hdu = fits.PrimaryHDU(image.astype(np.float32), header=header)
    hdu.writeto(str(path), overwrite=True)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_spectral_training_data(
    output_dir: Path,
    n_positive: int = 300,
    n_negative: int = 300,
    img_size: int = 64,
    snr_min: float = 3.0,
    snr_max: float = 15.0,
    temp_min: float = 25.0,
    temp_max: float = 60.0,
    motion_px: float = 8.0,
    delta_t_days: float = DEFAULT_DELTA_T,
    color_noise_frac: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-survey synthetic training data.

    Creates 6 FITS files per pair (2 epochs x 3 bands) and returns
    a catalog DataFrame matching MultiSurveyDataset schema.

    Parameters
    ----------
    output_dir : Path
        Output directory for FITS files and catalog.
    n_positive : int
        Number of positive (source present) pairs.
    n_negative : int
        Number of negative (no source) pairs.
    img_size : int
        Image size in pixels (square).
    snr_min, snr_max : float
        Optical S/N range for injected sources.
    temp_min, temp_max : float
        Temperature range in Kelvin for blackbody sources.
    motion_px : float
        Source displacement between epochs in pixels.
    delta_t_days : float
        Time separation between epochs in days.
    color_noise_frac : float
        Fractional noise on W1/W2 ratio (default 0.05 = ±5%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with training catalog.
    """
    output_dir = Path(output_dir)
    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # PSFs for optical and WISE bands
    optical_psf = make_gaussian_psf(size=11, fwhm=OPTICAL_PSF_FWHM)
    wise_psf = make_gaussian_psf(size=15, fwhm=WISE_PSF_FWHM)

    records = []

    # -------------------------------------------------------------------
    # Positive pairs: source with Planck-consistent IR signature
    # -------------------------------------------------------------------
    print(f"Generating {n_positive} positive pairs...")
    for i in range(n_positive):
        # Choose physical parameters
        temperature = rng.uniform(temp_min, temp_max)
        snr = rng.uniform(snr_min, snr_max)
        ir_boost = rng.uniform(2.0, 10.0)

        # Source position near center (both epochs)
        cx, cy = img_size / 2, img_size / 2
        px1 = cx + rng.uniform(-5, 5)
        py1 = cy + rng.uniform(-5, 5)

        # Motion direction
        angle = rng.uniform(0, 2 * np.pi)
        px2 = px1 + motion_px * np.cos(angle)
        py2 = py1 + motion_px * np.sin(angle)

        # Optical flux from S/N
        optical_flux = snr * OPTICAL_BG_STD / optical_psf.max()

        # IR fluxes from Planck law + color noise
        flux_w1, flux_w2 = compute_ir_fluxes(
            optical_flux, temperature, ir_boost, rng, color_noise_frac,
        )

        # Generate images for each epoch
        for epoch, (sx, sy) in enumerate([(px1, py1), (px2, py2)], start=1):
            epoch_tag = f"epoch{epoch}"

            # Optical band
            opt_img = rng.normal(
                OPTICAL_BG_MEAN, OPTICAL_BG_STD, (img_size, img_size),
            ).astype(np.float32)
            opt_img = inject_source(opt_img, sx, sy, optical_flux, optical_psf)

            # WISE W1 band
            w1_img = rng.normal(
                W1_BG_MEAN, W1_BG_STD, (img_size, img_size),
            ).astype(np.float32)
            w1_img = inject_source(w1_img, sx, sy, flux_w1, wise_psf)

            # WISE W2 band
            w2_img = rng.normal(
                W2_BG_MEAN, W2_BG_STD, (img_size, img_size),
            ).astype(np.float32)
            w2_img = inject_source(w2_img, sx, sy, flux_w2, wise_psf)

            # Save FITS files
            opt_path = pos_dir / f"pair_{i:04d}_{epoch_tag}_optical.fits"
            w1_path = pos_dir / f"pair_{i:04d}_{epoch_tag}_w1.fits"
            w2_path = pos_dir / f"pair_{i:04d}_{epoch_tag}_w2.fits"

            save_fits(opt_img, opt_path, pixel_scale=1.7)
            save_fits(w1_img, w1_path, pixel_scale=2.75)
            save_fits(w2_img, w2_path, pixel_scale=2.75)

            # Catalog record
            epoch_jd = EPOCH1_JD + (epoch - 1) * delta_t_days
            records.append({
                "example_id": f"pair_{i:04d}_{epoch_tag}",
                "optical_path": str(opt_path.relative_to(output_dir)),
                "w1_path": str(w1_path.relative_to(output_dir)),
                "w2_path": str(w2_path.relative_to(output_dir)),
                "source_present": True,
                "temp_kelvin": temperature,
                "epoch_jd": epoch_jd,
                "delta_t_days": delta_t_days,
            })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_positive}] T={temperature:.1f}K, "
                  f"S/N={snr:.1f}, ir_boost={ir_boost:.1f}")

    # -------------------------------------------------------------------
    # Negative pairs: pure noise (no source)
    # -------------------------------------------------------------------
    print(f"Generating {n_negative} negative pairs...")
    for i in range(n_negative):
        for epoch in [1, 2]:
            epoch_tag = f"epoch{epoch}"

            # Noise-only images
            opt_img = rng.normal(
                OPTICAL_BG_MEAN, OPTICAL_BG_STD, (img_size, img_size),
            ).astype(np.float32)
            w1_img = rng.normal(
                W1_BG_MEAN, W1_BG_STD, (img_size, img_size),
            ).astype(np.float32)
            w2_img = rng.normal(
                W2_BG_MEAN, W2_BG_STD, (img_size, img_size),
            ).astype(np.float32)

            # Save FITS
            opt_path = neg_dir / f"neg_{i:04d}_{epoch_tag}_optical.fits"
            w1_path = neg_dir / f"neg_{i:04d}_{epoch_tag}_w1.fits"
            w2_path = neg_dir / f"neg_{i:04d}_{epoch_tag}_w2.fits"

            save_fits(opt_img, opt_path, pixel_scale=1.7)
            save_fits(w1_img, w1_path, pixel_scale=2.75)
            save_fits(w2_img, w2_path, pixel_scale=2.75)

            epoch_jd = EPOCH1_JD + (epoch - 1) * delta_t_days
            records.append({
                "example_id": f"neg_{i:04d}_{epoch_tag}",
                "optical_path": str(opt_path.relative_to(output_dir)),
                "w1_path": str(w1_path.relative_to(output_dir)),
                "w2_path": str(w2_path.relative_to(output_dir)),
                "source_present": False,
                "temp_kelvin": float("nan"),
                "epoch_jd": epoch_jd,
                "delta_t_days": delta_t_days,
            })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_negative}]")

    catalog = pd.DataFrame(records)
    catalog_path = output_dir / "training_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-survey spectral training data for SSTF"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/training/spectral"),
        help="Output directory for FITS + catalog",
    )
    parser.add_argument("--n-positive", type=int, default=300)
    parser.add_argument("--n-negative", type=int, default=300)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--snr-min", type=float, default=3.0)
    parser.add_argument("--snr-max", type=float, default=15.0)
    parser.add_argument("--temp-min", type=float, default=25.0,
                        help="Min blackbody temperature (K)")
    parser.add_argument("--temp-max", type=float, default=60.0,
                        help="Max blackbody temperature (K)")
    parser.add_argument("--motion-px", type=float, default=8.0)
    parser.add_argument("--color-noise", type=float, default=0.05,
                        help="Fractional color noise on W1/W2 ratio")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("=" * 60)
    print("  SPECTRAL TRAINING DATA GENERATOR — SSTF")
    print("=" * 60)
    print(f"  Output:      {args.output_dir}")
    print(f"  Pairs:       {args.n_positive} pos + {args.n_negative} neg")
    print(f"  Image size:  {args.img_size}x{args.img_size}")
    print(f"  Optical S/N: {args.snr_min:.0f}-{args.snr_max:.0f}")
    print(f"  Temperature: {args.temp_min:.0f}-{args.temp_max:.0f} K")
    print(f"  Motion:      {args.motion_px:.0f} px")
    print(f"  Color noise: +/-{args.color_noise*100:.0f}%")
    print(f"  Seed:        {args.seed}")
    print("=" * 60)
    print()

    catalog = generate_spectral_training_data(
        output_dir=args.output_dir,
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        img_size=args.img_size,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        motion_px=args.motion_px,
        color_noise_frac=args.color_noise,
        seed=args.seed,
    )

    n_pairs = len(catalog) // 2
    n_files = len(catalog) * 3  # 3 FITS per row (opt, w1, w2)

    print()
    print("=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Pairs:    {n_pairs}")
    print(f"  FITS:     {n_files} files")
    print(f"  Catalog:  {args.output_dir / 'training_catalog.csv'}")
    print(f"  Columns:  {list(catalog.columns)}")
    print()

    # Summary statistics for positives
    pos = catalog[catalog["source_present"] == True]
    if len(pos) > 0:
        temps = pos["temp_kelvin"].dropna()
        print(f"  Temperature range: {temps.min():.1f} - {temps.max():.1f} K")
        print(f"  Temperature mean:  {temps.mean():.1f} K")


if __name__ == "__main__":
    main()
