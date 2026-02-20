"""
Batch training data generator for Planet 9 ML detection.

Generates labeled FITS images for training a convolutional neural network
or other ML classifier to detect faint moving point sources in sky survey
images. Produces:
  - Positive examples: real FITS with an injected synthetic source
  - Negative examples: clean FITS cutouts (no injection)
  - Multi-epoch pairs: same source at two positions offset by P9 motion

All generated images preserve the original WCS header and add injection
metadata keywords (INJECTED, INJ_RA, INJ_DEC, INJ_MAG, etc.).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from astroworld.imaging.fits_store import load_fits_image, build_catalog
from astroworld.imaging.psf_extraction import extract_psf, PSFResult
from astroworld.imaging.synthetic_injection import (
    inject_p9_source,
    estimate_zero_point,
    pixel_to_radec,
    InjectionResult,
)


@dataclass
class TrainingExample:
    """A single training example for the ML detector."""

    example_id: str  # Unique ID (pos_0001, neg_0001, etc.)
    fits_path: str  # Path to output FITS
    source_present: bool  # True if source was injected
    ra_deg: float  # Injected RA (NaN if no source)
    dec_deg: float  # Injected Dec (NaN if no source)
    pixel_x: float  # Injected pixel x (NaN if no source)
    pixel_y: float  # Injected pixel y (NaN if no source)
    magnitude: float  # Injected magnitude (NaN if no source)
    flux_adu: float  # Injected flux (NaN if no source)
    snr: float  # Estimated S/N (NaN if no source)
    source_fits: str  # Original FITS filename
    survey: str  # Survey name (e.g., "dss2_red")
    psf_fwhm: float  # PSF FWHM used for injection
    zero_point: float  # Zero point used


def generate_single_example(
    fits_path: Path,
    output_dir: Path,
    psf_result: PSFResult,
    magnitude: float = 22.5,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    inject: bool = True,
    zero_point: float | None = None,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
    example_id: str | None = None,
) -> TrainingExample:
    """
    Generate a single training example from a FITS image.

    Creates a modified FITS with an injected source (positive) or a
    clean copy (negative), and returns metadata for the training catalog.

    Parameters
    ----------
    fits_path : Path to input FITS
    output_dir : Directory for output files
    psf_result : Pre-extracted PSF
    magnitude : Injection magnitude
    ra_deg, dec_deg : Injection position (random if None and inject=True)
    inject : If False, creates a negative example (no source)
    zero_point : Pre-computed zero point
    add_noise : Add local Gaussian noise
    rng : Random generator
    example_id : Unique identifier for this example

    Returns
    -------
    TrainingExample with path and metadata
    """
    if rng is None:
        rng = np.random.default_rng()

    fits_path = Path(fits_path)
    image, header = load_fits_image(fits_path)

    # Determine output subdirectory
    subdir = "positive" if inject else "negative"
    out_subdir = output_dir / subdir
    out_subdir.mkdir(parents=True, exist_ok=True)

    if example_id is None:
        prefix = "pos" if inject else "neg"
        example_id = f"{prefix}_{rng.integers(0, 999999):06d}"

    out_filename = f"{example_id}_{fits_path.stem}.fits"
    out_path = out_subdir / out_filename

    if inject:
        # Inject synthetic source
        result = inject_p9_source(
            image=image,
            header=header,
            psf_result=psf_result,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            magnitude=magnitude,
            zero_point=zero_point,
            add_noise=add_noise,
            random_position=(ra_deg is None and dec_deg is None),
            rng=rng,
        )

        # Save injected FITS with metadata
        _save_fits_with_injection_metadata(
            result.image_injected, header, result, out_path
        )

        return TrainingExample(
            example_id=example_id,
            fits_path=str(out_path),
            source_present=True,
            ra_deg=result.ra_deg,
            dec_deg=result.dec_deg,
            pixel_x=result.pixel_x,
            pixel_y=result.pixel_y,
            magnitude=result.magnitude,
            flux_adu=result.flux_adu,
            snr=result.snr_estimate,
            source_fits=fits_path.name,
            survey=fits_path.parent.name,
            psf_fwhm=psf_result.fwhm_pixels,
            zero_point=zero_point or 0.0,
        )
    else:
        # Negative example: save clean copy
        hdu = fits.PrimaryHDU(image.astype(np.float32), header=_clean_header_for_fits(header))
        hdu.header["INJECTED"] = False
        hdu.writeto(str(out_path), overwrite=True)

        return TrainingExample(
            example_id=example_id,
            fits_path=str(out_path),
            source_present=False,
            ra_deg=float("nan"),
            dec_deg=float("nan"),
            pixel_x=float("nan"),
            pixel_y=float("nan"),
            magnitude=float("nan"),
            flux_adu=float("nan"),
            snr=float("nan"),
            source_fits=fits_path.name,
            survey=fits_path.parent.name,
            psf_fwhm=psf_result.fwhm_pixels,
            zero_point=zero_point or 0.0,
        )


def _clean_header_for_fits(header: dict) -> fits.Header:
    """Build a safe astropy Header from a plain dict.

    Real FITS headers (e.g. from SkyView DSS2) contain COMMENT and
    HISTORY entries that, when stored as ``dict`` values, get
    concatenated with newlines.  ``fits.Header(dict)`` then raises
    ``ValueError`` because those values contain non-printable ASCII.

    This function skips COMMENT, HISTORY, and any value containing
    newlines or non-ASCII characters while preserving all WCS and
    science keywords.
    """
    hdr = fits.Header()
    skip_keys = {"COMMENT", "HISTORY", ""}
    for key, val in header.items():
        if key.upper() in skip_keys:
            continue
        # Skip values that contain newlines or non-ASCII
        if isinstance(val, str):
            if "\n" in val or "\r" in val:
                continue
            try:
                val.encode("ascii")
            except UnicodeEncodeError:
                continue
        try:
            hdr[key] = val
        except (ValueError, TypeError):
            # If astropy rejects the value for any reason, skip it
            continue
    return hdr


def _save_fits_with_injection_metadata(
    image: np.ndarray,
    header: dict,
    result: InjectionResult,
    output_path: Path,
) -> None:
    """Save a FITS image with injection metadata in the header."""
    hdu = fits.PrimaryHDU(image.astype(np.float32), header=_clean_header_for_fits(header))
    hdu.header["INJECTED"] = True
    hdu.header["INJ_RA"] = (result.ra_deg, "Injected source RA (deg)")
    hdu.header["INJ_DEC"] = (result.dec_deg, "Injected source Dec (deg)")
    hdu.header["INJ_MAG"] = (result.magnitude, "Injected magnitude")
    hdu.header["INJ_FLUX"] = (result.flux_adu, "Injected total flux (ADU)")
    hdu.header["INJ_X"] = (result.pixel_x, "Injected pixel X position")
    hdu.header["INJ_Y"] = (result.pixel_y, "Injected pixel Y position")
    hdu.header["INJ_SNR"] = (result.snr_estimate, "Estimated S/N ratio")
    hdu.header["INJ_PSF"] = (result.psf_method, "PSF method used")
    hdu.writeto(str(output_path), overwrite=True)


def generate_training_set(
    image_dir: Path,
    output_dir: Path,
    n_positive: int = 100,
    n_negative: int = 100,
    magnitude_range: tuple[float, float] = (18.0, 24.0),
    fwhm_estimate: float = 2.5,
    psf_size: int = 25,
    seed: int = 42,
    surveys: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate a labeled training dataset for ML-based source detection.

    Scans image_dir for FITS files, extracts the PSF, and generates
    positive (injected) and negative (clean) examples with varying
    magnitudes and positions.

    Parameters
    ----------
    image_dir : Directory containing FITS files (or subdirectories)
    output_dir : Output directory for training data
    n_positive : Number of positive examples (source injected)
    n_negative : Number of negative examples (clean images)
    magnitude_range : (min_mag, max_mag) for random magnitude sampling
    fwhm_estimate : PSF FWHM estimate for extraction
    psf_size : PSF stamp size
    seed : Random seed for reproducibility
    surveys : Filter to specific survey subdirectories

    Returns
    -------
    DataFrame catalog of all generated examples (saved as CSV)
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Collect all FITS files
    fits_files = sorted(image_dir.rglob("*.fits"))
    if surveys:
        fits_files = [
            f for f in fits_files
            if f.parent.name in surveys
        ]

    if len(fits_files) == 0:
        print(f"No FITS files found in {image_dir}")
        return pd.DataFrame()

    print(f"Found {len(fits_files)} FITS files in {image_dir}")

    # Extract PSF from the image with lowest background noise
    print("\nExtracting PSF from best-quality image...")
    best_path, psf_result = _extract_best_psf(
        fits_files, fwhm_estimate, psf_size
    )
    print(f"  PSF method: {psf_result.method}")
    print(f"  FWHM: {psf_result.fwhm_pixels:.2f} pixels")
    print(f"  Stars detected: {psf_result.n_stars_detected}")
    print(f"  Stars used: {psf_result.n_stars_used}")
    print(f"  Background: {psf_result.background_mean:.1f} +/- "
          f"{psf_result.background_std:.1f} ADU")

    # Save PSF stamp
    psf_dir = output_dir / "psf"
    psf_dir.mkdir(parents=True, exist_ok=True)
    psf_hdu = fits.PrimaryHDU(psf_result.psf_image.astype(np.float32))
    psf_hdu.header["FWHM"] = psf_result.fwhm_pixels
    psf_hdu.header["METHOD"] = psf_result.method
    psf_hdu.header["NSTARS"] = psf_result.n_stars_used
    psf_hdu.writeto(str(psf_dir / "psf_stamp.fits"), overwrite=True)

    # Estimate zero point from the PSF extraction image
    img_zp, hdr_zp = load_fits_image(best_path)
    zero_point = estimate_zero_point(
        img_zp, hdr_zp,
        star_positions=psf_result.star_positions,
        star_fluxes=psf_result.star_fluxes,
    )
    print(f"  Zero point: {zero_point:.2f}")

    # Generate examples
    examples = []

    print(f"\nGenerating {n_positive} positive examples (mag {magnitude_range[0]:.0f}-{magnitude_range[1]:.0f})...")
    for i in range(n_positive):
        fits_path = rng.choice(fits_files)
        mag = rng.uniform(magnitude_range[0], magnitude_range[1])
        example_id = f"pos_{i + 1:04d}"

        ex = generate_single_example(
            fits_path=fits_path,
            output_dir=output_dir,
            psf_result=psf_result,
            magnitude=mag,
            inject=True,
            zero_point=zero_point,
            rng=rng,
            example_id=example_id,
        )
        examples.append(ex)

        if (i + 1) % max(1, n_positive // 10) == 0:
            print(f"  [{i + 1}/{n_positive}] mag={mag:.1f}, S/N={ex.snr:.1f}")

    print(f"\nGenerating {n_negative} negative examples...")
    for i in range(n_negative):
        fits_path = rng.choice(fits_files)
        example_id = f"neg_{i + 1:04d}"

        ex = generate_single_example(
            fits_path=fits_path,
            output_dir=output_dir,
            psf_result=psf_result,
            inject=False,
            zero_point=zero_point,
            rng=rng,
            example_id=example_id,
        )
        examples.append(ex)

    # Build catalog DataFrame
    catalog = pd.DataFrame([asdict(ex) for ex in examples])

    # Save catalog
    catalog_path = output_dir / "training_catalog.csv"
    catalog.to_csv(catalog_path, index=False, float_format="%.6f")

    # Summary
    pos_mask = catalog["source_present"]
    print(f"\n{'=' * 60}")
    print(f"  TRAINING SET COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total examples: {len(catalog)}")
    print(f"  Positive: {pos_mask.sum()} (with source)")
    print(f"  Negative: {(~pos_mask).sum()} (clean)")
    if pos_mask.sum() > 0:
        pos_data = catalog[pos_mask]
        print(f"  Magnitude range: {pos_data['magnitude'].min():.1f} - "
              f"{pos_data['magnitude'].max():.1f}")
        print(f"  S/N range: {pos_data['snr'].min():.1f} - "
              f"{pos_data['snr'].max():.1f}")
    print(f"  Catalog: {catalog_path}")
    print(f"  Output: {output_dir}")

    return catalog


def _extract_best_psf(
    fits_files: list[Path],
    fwhm_estimate: float,
    psf_size: int,
) -> tuple[Path, PSFResult]:
    """
    Extract PSF from the FITS image with lowest background noise.

    Samples up to 5 images, picks the one with the lowest background_std,
    and performs full PSF extraction on it.
    """
    # Sample a few images to find the best one
    sample_size = min(5, len(fits_files))
    candidates = []

    for path in fits_files[:sample_size]:
        try:
            image, header = load_fits_image(path)
            if image is None:
                continue
            bkg_mean, bkg_std = extract_psf.__wrapped__(image) if hasattr(extract_psf, '__wrapped__') else (0, 0)
            # Quick background estimate
            from astroworld.imaging.psf_extraction import estimate_background
            bkg_mean, bkg_std = estimate_background(image)
            candidates.append((path, bkg_std))
        except Exception:
            continue

    if not candidates:
        # Fallback: use first file
        path = fits_files[0]
        image, header = load_fits_image(path)
        psf = extract_psf(image, header, fwhm_estimate=fwhm_estimate, psf_size=psf_size)
        return path, psf

    # Sort by background noise (lowest first)
    candidates.sort(key=lambda x: x[1])
    best_path = candidates[0][0]

    image, header = load_fits_image(best_path)
    psf = extract_psf(image, header, fwhm_estimate=fwhm_estimate, psf_size=psf_size)
    return best_path, psf


def generate_multi_epoch_pair(
    fits_path_1: Path,
    fits_path_2: Path,
    output_dir: Path,
    psf_result: PSFResult,
    magnitude: float = 22.5,
    dt_days: float = 30.0,
    zero_point: float | None = None,
    rng: np.random.Generator | None = None,
    pair_id: str | None = None,
) -> tuple[TrainingExample, TrainingExample]:
    """
    Generate a pair of images with a moving source (multi-epoch detection).

    Injects the same source at two different positions corresponding
    to P9's motion between two epochs. The angular displacement is
    computed from the predicted angular velocity (~0.013 arcsec/hr).

    For P9 at 480 AU:
      30 days * 0.013 arcsec/hr * 24 hr/day = 9.4 arcsec = ~8 pixels (DSS2)

    Parameters
    ----------
    fits_path_1, fits_path_2 : Input FITS for epoch 1 and 2
    output_dir : Output directory
    psf_result : Pre-extracted PSF
    magnitude : Source magnitude
    dt_days : Time separation between epochs (default 30 days)
    zero_point : Photometric zero point
    rng : Random generator
    pair_id : Unique pair identifier

    Returns
    -------
    (example_1, example_2) training examples for the two epochs
    """
    if rng is None:
        rng = np.random.default_rng()

    if pair_id is None:
        pair_id = f"pair_{rng.integers(0, 999999):06d}"

    # Load both images
    image_1, header_1 = load_fits_image(fits_path_1)
    image_2, header_2 = load_fits_image(fits_path_2)

    ny, nx = image_1.shape
    margin = psf_result.psf_size + 20  # Extra margin for motion

    # Random starting position (with extra margin for motion)
    px_1 = rng.uniform(margin, nx - margin)
    py_1 = rng.uniform(margin, ny - margin)

    # Compute P9 motion offset
    # Angular velocity: ~0.013 arcsec/hr (from sky_position predictions)
    v_arcsec_hr = 0.013
    displacement_arcsec = v_arcsec_hr * dt_days * 24.0

    # Convert to pixels using pixel scale from header
    cdelt = abs(header_1.get("CDELT1", 0.000326))  # deg/pixel
    pixel_scale_arcsec = cdelt * 3600.0  # arcsec/pixel
    displacement_px = displacement_arcsec / pixel_scale_arcsec

    # Motion direction: approximate P9 sky motion along RA
    # (mostly east along ecliptic at its current position)
    dx_px = displacement_px * 0.97  # ~97% RA component
    dy_px = displacement_px * 0.26  # ~26% Dec component

    px_2 = px_1 + dx_px
    py_2 = py_1 + dy_px

    # Get RA/Dec for both positions
    ra_1, dec_1 = pixel_to_radec(px_1, py_1, header_1)
    ra_2, dec_2 = pixel_to_radec(px_2, py_2, header_2)

    # Inject into both images
    ex_1 = generate_single_example(
        fits_path=fits_path_1,
        output_dir=output_dir,
        psf_result=psf_result,
        magnitude=magnitude,
        ra_deg=ra_1,
        dec_deg=dec_1,
        inject=True,
        zero_point=zero_point,
        rng=rng,
        example_id=f"{pair_id}_epoch1",
    )

    ex_2 = generate_single_example(
        fits_path=fits_path_2,
        output_dir=output_dir,
        psf_result=psf_result,
        magnitude=magnitude,
        ra_deg=ra_2,
        dec_deg=dec_2,
        inject=True,
        zero_point=zero_point,
        rng=rng,
        example_id=f"{pair_id}_epoch2",
    )

    return ex_1, ex_2
