#!/usr/bin/env python
"""
Generate hybrid training data for domain adaptation (Phase 13).

"Smart Crop" strategy: Downloads large fields from the Taurus molecular
cloud region (dense stars, variables, nebulosity), crops hundreds of
aligned 64x64 patches across 3 surveys (DSS2 Red, WISE W1, WISE W2),
and creates two types of training examples:

  **Hard negatives** (real backgrounds, no injection):
    - Both epochs use the SAME real patch (no motion)
    - Model learns that astrophysical artifacts (variable stars,
      inter-survey differences) are NOT moving Planet 9 sources
    - Temporal gradient ~0 -> model learns to suppress false positives

  **Injection positives** (real backgrounds + synthetic source):
    - Real background patch + P9-like source with Planck-consistent
      W1/W2 ratios and +-5% color noise
    - Source moves between epochs by ``motion_px`` pixels
    - Model learns to detect faint moving sources ON real backgrounds

Output: FITS files + training_catalog.csv compatible with
MultiSurveyDataset (--direct-stack mode).

Catalog schema (8 columns):
    example_id, optical_path, w1_path, w2_path,
    source_present, temp_kelvin, epoch_jd, delta_t_days

Usage
-----
    uv run python scripts/generate_hybrid_training_data.py \\
        --output-dir data/training/hybrid \\
        --n-injection 200 --n-negative 200 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from astroworld.imaging.spectral_cube import (
    planck_flux_ratio,
)
from astroworld.imaging.survey_download import download_field


# ---------------------------------------------------------------------------
# Taurus molecular cloud fields (dense stellar backgrounds)
# ---------------------------------------------------------------------------

#: Coordinates (RA, Dec) of large fields in the Taurus molecular cloud.
#: These regions have dense star fields, variable stars, and nebulosity --
#: the most challenging backgrounds for false positive suppression.
TAURUS_FIELDS = [
    (68.0, 22.0),
    (69.5, 24.0),
    (66.5, 26.0),
    (71.0, 20.0),
    (64.0, 28.0),
    (72.5, 22.5),
    (63.0, 24.0),
    (67.5, 19.0),
]

#: Surveys to download for each field
HYBRID_SURVEYS = ["DSS2 Red", "WISE 3.4", "WISE 4.6"]

#: Field download parameters
FIELD_RADIUS_ARCMIN = 15.0
FIELD_PIXELS = 512

# Epoch parameters matching the real DSS2 -> WISE baseline
DSS2_EPOCH_JD = 2449200.0   # ~1993.7
WISE_EPOCH_JD = 2455400.0   # ~2010.7
DEFAULT_DELTA_T = 30.0       # days between "observation" epochs

# Background noise parameters (typical survey statistics)
OPTICAL_PSF_FWHM = 2.5      # DSS2 Red seeing ~2.5 pixels
WISE_PSF_FWHM = 6.0         # WISE PSF ~6 arcsec


# ---------------------------------------------------------------------------
# Field downloading
# ---------------------------------------------------------------------------

def download_large_fields(
    fields: list[tuple[float, float]],
    output_dir: Path,
    surveys: list[str] | None = None,
    radius_arcmin: float = FIELD_RADIUS_ARCMIN,
    pixels: int = FIELD_PIXELS,
    rate_limit_sec: float = 1.5,
) -> list[dict[str, Path | None]]:
    """Download multi-survey FITS images for a list of sky fields.

    Parameters
    ----------
    fields : List of (RA, Dec) coordinate tuples (degrees).
    output_dir : Base output directory for downloaded FITS files.
    surveys : List of survey names (default: DSS2 Red + WISE W1 + W2).
    radius_arcmin : Field of view radius.
    pixels : Image size in pixels.
    rate_limit_sec : Seconds between requests (be nice to servers).

    Returns
    -------
    List of dicts mapping survey name -> downloaded FITS path (or None).
    """
    if surveys is None:
        surveys = HYBRID_SURVEYS

    results = []
    for i, (ra, dec) in enumerate(fields):
        print(f"  [{i + 1}/{len(fields)}] RA={ra:.1f}, Dec={dec:.1f}")
        paths = download_field(
            ra_deg=ra,
            dec_deg=dec,
            surveys=surveys,
            radius_arcmin=radius_arcmin,
            pixels=pixels,
            output_dir=output_dir,
            rate_limit_sec=rate_limit_sec,
        )

        # Map survey names to paths
        field_paths = {}
        for j, survey in enumerate(surveys):
            if j < len(paths) and paths[j].exists():
                field_paths[survey] = paths[j]
            else:
                field_paths[survey] = None

        results.append(field_paths)

    return results


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> np.ndarray:
    """Load a FITS image as float64."""
    with fits.open(str(path)) as hdul:
        data = hdul[0].data
    if data is None:
        return np.zeros((64, 64), dtype=np.float64)
    return data.astype(np.float64)


def _resize_wise_to_optical(
    wise_image: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resize WISE image to match optical pixel grid.

    WISE pixel scale is ~2.75 arcsec/px vs DSS2 ~1.7 arcsec/px.
    Since all our patches will be on the same pixel grid (direct_stack),
    we simply zoom the WISE image to match optical dimensions.

    Uses scipy.ndimage.zoom if available, otherwise nearest-neighbor.
    """
    if wise_image.shape == target_shape:
        return wise_image

    try:
        from scipy.ndimage import zoom
        zy = target_shape[0] / wise_image.shape[0]
        zx = target_shape[1] / wise_image.shape[1]
        return zoom(wise_image, (zy, zx), order=1)
    except ImportError:
        # Nearest-neighbor fallback
        ny, nx = target_shape
        wy, wx = wise_image.shape
        result = np.zeros(target_shape, dtype=wise_image.dtype)
        for y in range(ny):
            for x in range(nx):
                sy = min(int(y * wy / ny), wy - 1)
                sx = min(int(x * wx / nx), wx - 1)
                result[y, x] = wise_image[sy, sx]
        return result


def extract_aligned_patches(
    optical_path: Path,
    w1_path: Path | None,
    w2_path: Path | None,
    n_patches: int,
    patch_size: int = 64,
    rng: np.random.Generator | None = None,
) -> list[dict[str, np.ndarray]]:
    """Extract aligned patches from multi-survey images.

    Loads the optical image and (optionally) WISE W1/W2 images,
    resizes WISE to match optical pixel grid, then crops random
    aligned patches.

    Parameters
    ----------
    optical_path : Path to DSS2 Red FITS file.
    w1_path : Path to WISE W1 FITS file (or None).
    w2_path : Path to WISE W2 FITS file (or None).
    n_patches : Number of patches to extract.
    patch_size : Side length of each patch (pixels).
    rng : NumPy random generator.

    Returns
    -------
    List of dicts with keys: "optical", "w1", "w2" (each 2D array).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Load optical image
    opt_img = _load_image(optical_path)
    h, w = opt_img.shape

    # Load and resize WISE images
    if w1_path is not None and w1_path.exists():
        w1_img = _resize_wise_to_optical(_load_image(w1_path), (h, w))
    else:
        w1_img = np.zeros((h, w), dtype=np.float64)

    if w2_path is not None and w2_path.exists():
        w2_img = _resize_wise_to_optical(_load_image(w2_path), (h, w))
    else:
        w2_img = np.zeros((h, w), dtype=np.float64)

    # Crop random patches (avoid edges)
    margin = patch_size // 2
    max_y = h - patch_size - margin
    max_x = w - patch_size - margin
    if max_y <= margin or max_x <= margin:
        return []  # Image too small

    patches = []
    for _ in range(n_patches):
        y0 = rng.integers(margin, max_y)
        x0 = rng.integers(margin, max_x)

        patch = {
            "optical": opt_img[y0:y0 + patch_size, x0:x0 + patch_size].copy(),
            "w1": w1_img[y0:y0 + patch_size, x0:x0 + patch_size].copy(),
            "w2": w2_img[y0:y0 + patch_size, x0:x0 + patch_size].copy(),
        }
        patches.append(patch)

    return patches


# ---------------------------------------------------------------------------
# PSF and source injection helpers
# ---------------------------------------------------------------------------

def _make_gaussian_psf(fwhm_px: float, size: int = 21) -> np.ndarray:
    """Create a peak-normalized Gaussian PSF."""
    sigma = fwhm_px / 2.3548
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf


def _inject_psf_at(
    image: np.ndarray,
    px: float,
    py: float,
    flux: float,
    psf: np.ndarray,
    bg_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject a PSF at a sub-pixel position with noise."""
    result = image.copy()
    h, w = image.shape
    ph, pw = psf.shape
    ix, iy = int(round(px)), int(round(py))

    # Scale PSF to flux
    scaled = psf * flux

    # Add Gaussian noise (matching background statistics)
    if bg_std > 0:
        noise = rng.normal(0.0, bg_std * 0.3, scaled.shape)
        noise_mask = scaled > 0.01 * scaled.max()
        scaled = scaled + noise * noise_mask

    # Place on image
    y0 = iy - ph // 2
    x0 = ix - pw // 2
    for j in range(ph):
        for i in range(pw):
            yy = y0 + j
            xx = x0 + i
            if 0 <= yy < h and 0 <= xx < w:
                result[yy, xx] += scaled[j, i]

    return result


def _estimate_bg_std(image: np.ndarray) -> float:
    """Quick sigma-clipped background RMS estimate."""
    data = image.ravel()
    for _ in range(3):
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        mask = np.abs(data - mean) < 3 * std
        data = data[mask]
        if len(data) == 0:
            return 0.0
    return float(np.std(data))


# ---------------------------------------------------------------------------
# Injection into real patches
# ---------------------------------------------------------------------------

def inject_source_into_patch(
    optical: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    temp_k: float,
    snr: float,
    motion_px: float,
    rng: np.random.Generator,
    color_noise_frac: float = 0.05,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Inject a synthetic P9-like source into a real background patch.

    Creates two epoch images: epoch 1 with source at position (px1, py1)
    and epoch 2 with source at position (px2, py2 = px1+dx, py1+dy).

    IR fluxes are Planck-consistent with +-color_noise_frac random
    perturbation on the W1/W2 ratio.

    Parameters
    ----------
    optical, w1, w2 : 2D arrays
        Real survey background patches (same pixel grid, same size).
    temp_k : Blackbody temperature in Kelvin.
    snr : Target S/N of injected source (optical band).
    motion_px : Displacement between epochs in pixels.
    rng : NumPy random generator.
    color_noise_frac : Fractional noise on W1/W2 ratio.

    Returns
    -------
    (epoch1_dict, epoch2_dict) each with keys "optical", "w1", "w2".
    """
    h, w_dim = optical.shape
    ps = min(h, w_dim)

    # Source position near center (both epochs)
    cx, cy = ps / 2, ps / 2
    px1 = cx + rng.uniform(-8, 8)
    py1 = cy + rng.uniform(-8, 8)

    # Clamp to safe region
    margin = 12
    px1 = np.clip(px1, margin, ps - margin)
    py1 = np.clip(py1, margin, ps - margin)

    # Motion direction (random angle)
    angle = rng.uniform(0, 2 * np.pi)
    dx = motion_px * np.cos(angle)
    dy = motion_px * np.sin(angle)
    px2 = np.clip(px1 + dx, margin, ps - margin)
    py2 = np.clip(py1 + dy, margin, ps - margin)

    # Background RMS per band
    bg_std_opt = max(_estimate_bg_std(optical), 1e-10)
    bg_std_w1 = max(_estimate_bg_std(w1), 1e-10)
    bg_std_w2 = max(_estimate_bg_std(w2), 1e-10)

    # Optical flux from S/N
    opt_psf = _make_gaussian_psf(OPTICAL_PSF_FWHM)
    optical_flux = snr * bg_std_opt / max(opt_psf.max(), 1e-10)

    # IR fluxes: Planck-consistent with color noise
    ratio_w1_w2 = planck_flux_ratio(temp_k)
    noise_factor = 1.0 + rng.uniform(-color_noise_frac, color_noise_frac)
    ratio_w1_w2 *= noise_factor

    # IR boost: cold bodies are brighter in IR relative to optical
    ir_boost = max(3.0, 10.0 - temp_k / 10.0)
    w2_flux = optical_flux * ir_boost * (bg_std_w2 / max(bg_std_opt, 1e-10))
    w1_flux = w2_flux * max(ratio_w1_w2, 0.01)

    wise_psf = _make_gaussian_psf(WISE_PSF_FWHM)

    # Epoch 1: inject at (px1, py1)
    e1 = {
        "optical": _inject_psf_at(
            optical, px1, py1, optical_flux, opt_psf, bg_std_opt, rng
        ),
        "w1": _inject_psf_at(w1, px1, py1, w1_flux, wise_psf, bg_std_w1, rng),
        "w2": _inject_psf_at(w2, px1, py1, w2_flux, wise_psf, bg_std_w2, rng),
    }

    # Epoch 2: inject at (px2, py2) — motion!
    e2 = {
        "optical": _inject_psf_at(
            optical, px2, py2, optical_flux, opt_psf, bg_std_opt, rng
        ),
        "w1": _inject_psf_at(w1, px2, py2, w1_flux, wise_psf, bg_std_w1, rng),
        "w2": _inject_psf_at(w2, px2, py2, w2_flux, wise_psf, bg_std_w2, rng),
    }

    return e1, e2


# ---------------------------------------------------------------------------
# FITS I/O helper
# ---------------------------------------------------------------------------

def _make_minimal_header(
    nx: int,
    ny: int,
    pixel_scale_arcsec: float = 1.7,
) -> fits.Header:
    """Create a minimal FITS header with TAN-projected WCS."""
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


def _save_patch_fits(image: np.ndarray, path: Path, pixel_scale: float = 1.7):
    """Save a 2D patch as a FITS file with minimal WCS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _make_minimal_header(image.shape[1], image.shape[0], pixel_scale)
    hdu = fits.PrimaryHDU(image.astype(np.float32), header=header)
    hdu.writeto(str(path), overwrite=True)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_hybrid_training_data(
    output_dir: Path,
    fields: list[tuple[float, float]] | None = None,
    n_injection: int = 200,
    n_negative: int = 200,
    patch_size: int = 64,
    snr_min: float = 3.0,
    snr_max: float = 15.0,
    temp_min: float = 25.0,
    temp_max: float = 60.0,
    motion_px: float = 8.0,
    delta_t_days: float = DEFAULT_DELTA_T,
    color_noise_frac: float = 0.05,
    download_dir: Path | None = None,
    seed: int = 42,
    skip_download: bool = False,
) -> pd.DataFrame:
    """Generate hybrid training data from real survey backgrounds.

    Parameters
    ----------
    output_dir : Directory for output FITS files and catalog.
    fields : List of (RA, Dec) sky coordinates for background fields.
        Defaults to TAURUS_FIELDS.
    n_injection : Number of injection-positive pairs to generate.
    n_negative : Number of hard-negative pairs to generate.
    patch_size : Patch side length in pixels.
    snr_min, snr_max : S/N range for injected sources.
    temp_min, temp_max : Temperature range (K) for blackbody sources.
    motion_px : Source displacement between epochs (pixels).
    delta_t_days : Time separation between "epochs" (days).
    color_noise_frac : Fractional noise on W1/W2 ratio.
    download_dir : Directory for raw survey downloads.
        Defaults to output_dir / "raw_fields".
    seed : Random seed for reproducibility.
    skip_download : If True, skip downloading (use cached fields only).

    Returns
    -------
    pd.DataFrame with training catalog (same schema as synthetic data).
    """
    output_dir = Path(output_dir)
    if download_dir is None:
        download_dir = output_dir / "raw_fields"

    if fields is None:
        fields = TAURUS_FIELDS

    rng = np.random.default_rng(seed)

    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Step 1: Download large fields (or use cached)
    # -----------------------------------------------------------------
    if not skip_download:
        print("\n[1/4] Downloading large survey fields...")
        t0 = time.time()
        field_survey_paths = download_large_fields(
            fields, download_dir,
            surveys=HYBRID_SURVEYS,
            radius_arcmin=FIELD_RADIUS_ARCMIN,
            pixels=FIELD_PIXELS,
        )
        elapsed = time.time() - t0
        print(f"  Download complete in {elapsed:.1f}s")
    else:
        print("\n[1/4] Skipping download (using cached fields)...")
        # Reconstruct paths from existing files
        field_survey_paths = _find_cached_fields(fields, download_dir)

    # -----------------------------------------------------------------
    # Step 2: Extract aligned patches from all fields
    # -----------------------------------------------------------------
    print("\n[2/4] Extracting aligned patches from fields...")
    n_total_needed = n_injection + n_negative
    n_per_field = max(1, n_total_needed // len(fields) + 2)

    all_patches = []
    for i, field_paths in enumerate(field_survey_paths):
        opt_path = field_paths.get("DSS2 Red")
        w1_path = field_paths.get("WISE 3.4")
        w2_path = field_paths.get("WISE 4.6")

        if opt_path is None:
            print(f"  [skip] Field {i}: no optical data")
            continue

        patches = extract_aligned_patches(
            opt_path, w1_path, w2_path,
            n_patches=n_per_field,
            patch_size=patch_size,
            rng=rng,
        )
        all_patches.extend(patches)
        print(f"  Field {i}: {len(patches)} patches extracted")

    if len(all_patches) == 0:
        print("  ERROR: No patches extracted. Check survey downloads.")
        return pd.DataFrame()

    print(f"  Total patches: {len(all_patches)}")

    # Shuffle patches
    rng.shuffle(all_patches)

    # -----------------------------------------------------------------
    # Step 3: Generate injection positives
    # -----------------------------------------------------------------
    print(f"\n[3/4] Generating {n_injection} injection positives...")
    records = []
    n_gen_pos = 0

    for i in range(n_injection):
        # Cycle through patches
        patch = all_patches[i % len(all_patches)]

        temperature = rng.uniform(temp_min, temp_max)
        snr = rng.uniform(snr_min, snr_max)

        e1, e2 = inject_source_into_patch(
            patch["optical"], patch["w1"], patch["w2"],
            temp_k=temperature,
            snr=snr,
            motion_px=motion_px,
            rng=rng,
            color_noise_frac=color_noise_frac,
        )

        # Save both epochs
        for epoch, data in enumerate([e1, e2], start=1):
            tag = f"epoch{epoch}"
            opt_path = pos_dir / f"hybrid_pos_{i:04d}_{tag}_optical.fits"
            w1_path = pos_dir / f"hybrid_pos_{i:04d}_{tag}_w1.fits"
            w2_path = pos_dir / f"hybrid_pos_{i:04d}_{tag}_w2.fits"

            _save_patch_fits(data["optical"], opt_path, pixel_scale=1.7)
            _save_patch_fits(data["w1"], w1_path, pixel_scale=1.7)
            _save_patch_fits(data["w2"], w2_path, pixel_scale=1.7)

            epoch_jd = DSS2_EPOCH_JD + (epoch - 1) * delta_t_days
            records.append({
                "example_id": f"hybrid_pos_{i:04d}_{tag}",
                "optical_path": str(opt_path.relative_to(output_dir)),
                "w1_path": str(w1_path.relative_to(output_dir)),
                "w2_path": str(w2_path.relative_to(output_dir)),
                "source_present": True,
                "temp_kelvin": temperature,
                "epoch_jd": epoch_jd,
                "delta_t_days": delta_t_days,
            })

        n_gen_pos += 1
        if (n_gen_pos) % 50 == 0:
            print(f"  [{n_gen_pos}/{n_injection}] T={temperature:.1f}K, "
                  f"S/N={snr:.1f}")

    # -----------------------------------------------------------------
    # Step 4: Generate hard negatives
    # -----------------------------------------------------------------
    print(f"\n[4/4] Generating {n_negative} hard negatives...")
    n_gen_neg = 0

    for i in range(n_negative):
        # Cycle through patches (use SAME patch for both epochs!)
        patch = all_patches[(n_injection + i) % len(all_patches)]

        # Both epochs use the same real patch (no motion => label=0)
        for epoch in [1, 2]:
            tag = f"epoch{epoch}"
            opt_path = neg_dir / f"hybrid_neg_{i:04d}_{tag}_optical.fits"
            w1_path = neg_dir / f"hybrid_neg_{i:04d}_{tag}_w1.fits"
            w2_path = neg_dir / f"hybrid_neg_{i:04d}_{tag}_w2.fits"

            _save_patch_fits(patch["optical"], opt_path, pixel_scale=1.7)
            _save_patch_fits(patch["w1"], w1_path, pixel_scale=1.7)
            _save_patch_fits(patch["w2"], w2_path, pixel_scale=1.7)

            epoch_jd = DSS2_EPOCH_JD + (epoch - 1) * delta_t_days
            records.append({
                "example_id": f"hybrid_neg_{i:04d}_{tag}",
                "optical_path": str(opt_path.relative_to(output_dir)),
                "w1_path": str(w1_path.relative_to(output_dir)),
                "w2_path": str(w2_path.relative_to(output_dir)),
                "source_present": False,
                "temp_kelvin": float("nan"),
                "epoch_jd": epoch_jd,
                "delta_t_days": delta_t_days,
            })

        n_gen_neg += 1
        if (n_gen_neg) % 50 == 0:
            print(f"  [{n_gen_neg}/{n_negative}]")

    # Save catalog
    catalog = pd.DataFrame(records)
    catalog_path = output_dir / "training_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    print(f"\n  Catalog saved: {catalog_path}")

    return catalog


def _find_cached_fields(
    fields: list[tuple[float, float]],
    download_dir: Path,
) -> list[dict[str, Path | None]]:
    """Reconstruct field paths from cached downloads."""
    from astroworld.imaging.survey_download import _survey_to_dir

    results = []
    for ra, dec in fields:
        ra_str = f"{ra:07.3f}"
        dec_str = f"{dec:+07.3f}"
        filename = f"field_ra{ra_str}_dec{dec_str}.fits"

        field_paths: dict[str, Path | None] = {}
        for survey in HYBRID_SURVEYS:
            survey_dir = download_dir / _survey_to_dir(survey)
            filepath = survey_dir / filename
            field_paths[survey] = filepath if filepath.exists() else None

        results.append(field_paths)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid training data for domain adaptation"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/training/hybrid"),
        help="Output directory for hybrid FITS + catalog",
    )
    parser.add_argument(
        "--download-dir", type=Path, default=None,
        help="Directory for raw survey downloads (default: output-dir/raw_fields)",
    )
    parser.add_argument("--n-injection", type=int, default=200,
                        help="Number of injection-positive pairs")
    parser.add_argument("--n-negative", type=int, default=200,
                        help="Number of hard-negative pairs")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--snr-min", type=float, default=3.0)
    parser.add_argument("--snr-max", type=float, default=15.0)
    parser.add_argument("--temp-min", type=float, default=25.0)
    parser.add_argument("--temp-max", type=float, default=60.0)
    parser.add_argument("--motion-px", type=float, default=8.0)
    parser.add_argument("--color-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading (use cached fields)")
    parser.add_argument(
        "--n-fields", type=int, default=None,
        help="Number of Taurus fields to use (default: all 8)",
    )

    args = parser.parse_args()

    # Select fields
    fields = TAURUS_FIELDS
    if args.n_fields is not None:
        fields = fields[:args.n_fields]

    print("=" * 65)
    print("  HYBRID TRAINING DATA GENERATOR — Domain Adaptation")
    print("=" * 65)
    print(f"  Output:       {args.output_dir}")
    print(f"  Download dir: {args.download_dir or 'auto'}")
    print(f"  Fields:       {len(fields)} Taurus regions")
    print(f"  Pairs:        {args.n_injection} inj + {args.n_negative} neg")
    print(f"  Patch size:   {args.patch_size}x{args.patch_size}")
    print(f"  Optical S/N:  {args.snr_min:.0f}-{args.snr_max:.0f}")
    print(f"  Temperature:  {args.temp_min:.0f}-{args.temp_max:.0f} K")
    print(f"  Motion:       {args.motion_px:.0f} px")
    print(f"  Color noise:  +-{args.color_noise * 100:.0f}%")
    print(f"  Seed:         {args.seed}")
    print("=" * 65)

    catalog = generate_hybrid_training_data(
        output_dir=args.output_dir,
        fields=fields,
        n_injection=args.n_injection,
        n_negative=args.n_negative,
        patch_size=args.patch_size,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        motion_px=args.motion_px,
        delta_t_days=DEFAULT_DELTA_T,
        color_noise_frac=args.color_noise,
        download_dir=args.download_dir,
        seed=args.seed,
        skip_download=args.skip_download,
    )

    if len(catalog) == 0:
        print("\n  ERROR: No training data generated.")
        sys.exit(1)

    n_pairs = len(catalog) // 2
    n_files = len(catalog) * 3  # 3 FITS per row

    print()
    print("=" * 65)
    print("  COMPLETE")
    print("=" * 65)
    print(f"  Total pairs: {n_pairs}")
    print(f"  FITS files:  {n_files}")
    print(f"  Catalog:     {args.output_dir / 'training_catalog.csv'}")
    print(f"  Columns:     {list(catalog.columns)}")

    # Summary
    pos = catalog[catalog["source_present"] == True]
    neg = catalog[catalog["source_present"] == False]
    print(f"  Positive rows: {len(pos)} ({len(pos) // 2} pairs)")
    print(f"  Negative rows: {len(neg)} ({len(neg) // 2} pairs)")

    if len(pos) > 0:
        temps = pos["temp_kelvin"].dropna()
        if len(temps) > 0:
            print(f"  Temperature:   {temps.min():.1f} - {temps.max():.1f} K")
    print()


if __name__ == "__main__":
    main()
