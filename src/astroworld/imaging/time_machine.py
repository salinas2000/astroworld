"""
Multi-epoch blink comparison for Planet 9 candidate verification.

Downloads Pan-STARRS r-band (epoch ~2012) cutouts and compares against
existing DSS2 Red images (epoch ~1993) to detect proper motion.

Verdicts:
  STATIC:   Source present in both epochs at same position -> ISM/galaxy
  MOVED:    Source shifted between epochs -> possible proper motion object!
  ABSENT:   No source in either epoch -> pipeline noise artifact
  APPEARED: Source in one epoch only -> transient or moving object

For Planet 9 at ~480 AU: expected PM ~0.5 arcsec/yr -> ~9.5 arcsec in 19 years
-> easily detectable (>30 pixels at PS1 scale of 0.25 arcsec/px).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

from astroworld.imaging.reprojection import build_common_wcs, _sanitize_header


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PS1_FILENAMES_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PS1_FITSCUT_URL = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"

DSS2_EPOCH_YEAR = 1993.0   # Approximate mid-epoch for POSS-II / DSS2 Red
PS1_EPOCH_YEAR = 2012.0    # Approximate mid-epoch for Pan-STARRS DR1

PS1_PIXEL_SCALE_ARCSEC = 0.25
DSS2_PIXEL_SCALE_ARCSEC = 1.7

# Filename pattern for coordinate extraction
_FIELD_RE = re.compile(
    r"field_ra(\d+\.\d+)_dec([+-]?\d+\.\d+)\.fits"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BlinkResult:
    """Result of a multi-epoch blink comparison for a single candidate."""

    ra_deg: float
    dec_deg: float
    verdict: str = "UNKNOWN"           # STATIC, MOVED, ABSENT, APPEARED
    shift_arcsec: float = 0.0
    shift_ra_arcsec: float = 0.0
    shift_dec_arcsec: float = 0.0
    snr_epoch1: float = 0.0
    snr_epoch2: float = 0.0
    flux_ratio: float = 0.0
    epoch1_year: float = DSS2_EPOCH_YEAR
    epoch2_year: float = PS1_EPOCH_YEAR
    baseline_years: float = PS1_EPOCH_YEAR - DSS2_EPOCH_YEAR
    implied_pm_arcsec_yr: float = 0.0
    confidence: str = "low"            # high, medium, low
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pan-STARRS download
# ---------------------------------------------------------------------------

def download_ps1_cutout(
    ra_deg: float,
    dec_deg: float,
    size_pixels: int = 240,
    filter_band: str = "r",
    output_dir: Path = Path("data/survey_images"),
    rate_limit_sec: float = 1.0,
) -> Path | None:
    """
    Download a Pan-STARRS DR1 FITS cutout for a given sky position.

    Uses the PS1 image cutout service at STScI.

    Parameters
    ----------
    ra_deg, dec_deg : Sky coordinates (degrees, ICRS).
    size_pixels : Cutout size in pixels (PS1 pixel = 0.25 arcsec).
    filter_band : PS1 filter (g, r, i, z, y).
    output_dir : Base directory for survey images.
    rate_limit_sec : Delay between HTTP requests.

    Returns
    -------
    Path to downloaded FITS file, or None on failure.
    """
    ps1_dir = output_dir / "ps1_r"
    ps1_dir.mkdir(parents=True, exist_ok=True)

    ra_str = f"{ra_deg:07.3f}"
    dec_str = f"{dec_deg:+07.3f}"
    filename = f"field_ra{ra_str}_dec{dec_str}.fits"
    filepath = ps1_dir / filename

    # Resume support
    if filepath.exists():
        return filepath

    try:
        # Step 1: Query available filenames
        params = {
            "ra": ra_deg,
            "dec": dec_deg,
            "filters": filter_band,
        }
        resp = requests.get(PS1_FILENAMES_URL, params=params, timeout=30)
        resp.raise_for_status()

        # Parse the text table (space-delimited, first line is header)
        lines = resp.text.strip().split("\n")
        if len(lines) < 2:
            return None

        # Extract filename from first data row
        parts = lines[1].split()
        if len(parts) < 8:
            return None

        image_filename = parts[7]  # 'filename' column

        time.sleep(rate_limit_sec)

        # Step 2: Download FITS cutout
        cutout_params = {
            "ra": ra_deg,
            "dec": dec_deg,
            "size": size_pixels,
            "format": "fits",
            "red": image_filename,
        }
        resp2 = requests.get(PS1_FITSCUT_URL, params=cutout_params, timeout=60)
        resp2.raise_for_status()

        # Validate it's a FITS file (starts with 'SIMPLE')
        if not resp2.content[:6] == b"SIMPLE":
            return None

        filepath.write_bytes(resp2.content)
        return filepath

    except Exception:
        return None


# ---------------------------------------------------------------------------
# DSS2 field finder
# ---------------------------------------------------------------------------

def find_dss2_field(
    ra_deg: float,
    dec_deg: float,
    dss2_dir: Path,
    max_separation_arcmin: float = 5.0,
) -> Path | None:
    """
    Find the nearest DSS2 FITS field file for a given sky position.

    Parses RA/Dec from filenames and returns the closest match
    within ``max_separation_arcmin``.
    """
    best_path = None
    best_sep = float("inf")
    cos_dec = np.cos(np.radians(dec_deg))

    for fpath in dss2_dir.glob("field_ra*.fits"):
        m = _FIELD_RE.match(fpath.name)
        if not m:
            continue
        file_ra = float(m.group(1))
        file_dec = float(m.group(2))

        # Angular separation (small-angle approximation)
        dra = (ra_deg - file_ra) * cos_dec
        ddec = dec_deg - file_dec
        sep_deg = np.sqrt(dra**2 + ddec**2)
        sep_arcmin = sep_deg * 60.0

        if sep_arcmin < best_sep:
            best_sep = sep_arcmin
            best_path = fpath

    if best_sep <= max_separation_arcmin:
        return best_path
    return None


# ---------------------------------------------------------------------------
# Image alignment
# ---------------------------------------------------------------------------

def align_epochs(
    ra_deg: float,
    dec_deg: float,
    image1: np.ndarray,
    header1: dict,
    image2: np.ndarray,
    header2: dict,
    cutout_arcsec: float = 30.0,
    target_pixel_scale: float = PS1_PIXEL_SCALE_ARCSEC,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Align two images from different epochs to a common WCS grid and
    extract cutouts centered on the candidate position.

    Parameters
    ----------
    ra_deg, dec_deg : Candidate sky position.
    image1, header1 : Epoch 1 (DSS2) image and FITS header.
    image2, header2 : Epoch 2 (PS1) image and FITS header.
    cutout_arcsec : Cutout size in arcseconds.
    target_pixel_scale : Output pixel scale in arcsec/px.

    Returns
    -------
    (cutout1, cutout2, output_header)
    """
    from reproject import reproject_interp
    from astropy.wcs import WCS

    cutout_pixels = int(cutout_arcsec / target_pixel_scale)
    shape = (cutout_pixels, cutout_pixels)

    # Build target WCS centered on candidate
    target_header = build_common_wcs(
        ra_deg, dec_deg, target_pixel_scale, shape,
    )
    target_wcs = WCS(target_header)

    # Sanitize input headers
    h1 = _sanitize_header(header1)
    h2 = _sanitize_header(header2)

    # Reproject both
    r1, _ = reproject_interp(
        (image1.astype(np.float64), WCS(h1)),
        target_wcs,
        shape_out=shape,
        order=1,
    )
    r2, _ = reproject_interp(
        (image2.astype(np.float64), WCS(h2)),
        target_wcs,
        shape_out=shape,
        order=1,
    )

    r1 = np.nan_to_num(r1, nan=0.0)
    r2 = np.nan_to_num(r2, nan=0.0)

    return r1, r2, dict(target_header)


# ---------------------------------------------------------------------------
# Source SNR measurement
# ---------------------------------------------------------------------------

def measure_source_snr(
    cutout: np.ndarray,
    aperture_radius_pix: int = 5,
) -> dict:
    """
    Measure the SNR of a source at the center of a cutout image.

    Uses circular aperture photometry with an annular background estimate.
    """
    ny, nx = cutout.shape
    cy, cx = ny // 2, nx // 2

    # Build coordinate grids
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Aperture: central circle
    aperture_mask = r <= aperture_radius_pix

    # Background annulus: 2x to 3x aperture radius
    inner = aperture_radius_pix * 2
    outer = aperture_radius_pix * 3
    annulus_mask = (r >= inner) & (r <= outer)

    bg_pixels = cutout[annulus_mask]
    if len(bg_pixels) < 10:
        # Fallback: use outer 25% of image
        border = max(1, ny // 4)
        bg_pixels = np.concatenate([
            cutout[:border, :].ravel(),
            cutout[-border:, :].ravel(),
            cutout[:, :border].ravel(),
            cutout[:, -border:].ravel(),
        ])

    # Sigma-clipped background
    bg_median = float(np.median(bg_pixels))
    mad = float(np.median(np.abs(bg_pixels - bg_median)))
    bg_std = 1.4826 * mad if mad > 0 else float(np.std(bg_pixels))

    # Source flux
    source_pixels = cutout[aperture_mask]
    peak = float(np.max(source_pixels))
    snr = (peak - bg_median) / bg_std if bg_std > 0 else 0.0

    return {
        "snr": snr,
        "peak_flux": peak,
        "bg_median": bg_median,
        "bg_std": bg_std,
        "is_detected": snr >= 3.0,
    }


# ---------------------------------------------------------------------------
# Shift detection via cross-correlation
# ---------------------------------------------------------------------------

def measure_shift(
    cutout1: np.ndarray,
    cutout2: np.ndarray,
    pixel_scale_arcsec: float = PS1_PIXEL_SCALE_ARCSEC,
) -> dict:
    """
    Measure positional shift between two aligned cutouts via
    cross-correlation.

    Returns shift in pixels and arcseconds with sub-pixel precision.
    """
    from scipy.signal import correlate2d

    # Normalize both cutouts (zero-mean, unit-std)
    def _norm(img):
        s = np.std(img)
        if s < 1e-10:
            return img - np.mean(img)
        return (img - np.mean(img)) / s

    c1 = _norm(cutout1)
    c2 = _norm(cutout2)

    # Cross-correlation
    corr = correlate2d(c1, c2, mode="full", boundary="fill", fillvalue=0)

    # Expected peak position (zero shift)
    ny, nx = cutout1.shape
    center_y = ny - 1
    center_x = nx - 1

    # Find peak
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_y, peak_x = peak_idx

    # Sub-pixel refinement via parabolic interpolation
    def _subpix(arr, iy, ix):
        h, w = arr.shape
        dy = 0.0
        dx = 0.0
        if 0 < iy < h - 1:
            a = arr[iy - 1, ix]
            b = arr[iy, ix]
            c = arr[iy + 1, ix]
            denom = 2.0 * (2 * b - a - c)
            if abs(denom) > 1e-10:
                dy = (a - c) / denom
        if 0 < ix < w - 1:
            a = arr[iy, ix - 1]
            b = arr[iy, ix]
            c = arr[iy, ix + 1]
            denom = 2.0 * (2 * b - a - c)
            if abs(denom) > 1e-10:
                dx = (a - c) / denom
        return dy, dx

    sub_dy, sub_dx = _subpix(corr, peak_y, peak_x)

    shift_y_pix = (peak_y + sub_dy) - center_y
    shift_x_pix = (peak_x + sub_dx) - center_x
    shift_pix = np.sqrt(shift_x_pix**2 + shift_y_pix**2)
    shift_arcsec = shift_pix * pixel_scale_arcsec

    # Correlation SNR: peak value vs background std of correlation map
    peak_val = float(corr[peak_y, peak_x])
    # Mask the central peak region for background stats
    mask = np.ones_like(corr, dtype=bool)
    r_mask = 5
    y_lo = max(0, peak_y - r_mask)
    y_hi = min(corr.shape[0], peak_y + r_mask + 1)
    x_lo = max(0, peak_x - r_mask)
    x_hi = min(corr.shape[1], peak_x + r_mask + 1)
    mask[y_lo:y_hi, x_lo:x_hi] = False
    bg_vals = corr[mask]
    corr_bg_std = float(np.std(bg_vals)) if len(bg_vals) > 0 else 1.0
    correlation_snr = (peak_val - float(np.median(bg_vals))) / corr_bg_std if corr_bg_std > 0 else 0.0

    return {
        "shift_x_pix": shift_x_pix,
        "shift_y_pix": shift_y_pix,
        "shift_pix": shift_pix,
        "shift_arcsec": shift_arcsec,
        "shift_ra_arcsec": shift_x_pix * pixel_scale_arcsec,
        "shift_dec_arcsec": shift_y_pix * pixel_scale_arcsec,
        "peak_value": peak_val,
        "correlation_snr": correlation_snr,
    }


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------

def classify_verdict(
    snr1: float,
    snr2: float,
    shift_arcsec: float,
    correlation_snr: float,
    snr_threshold: float = 3.0,
    shift_threshold_arcsec: float = 1.0,
) -> tuple[str, str]:
    """
    Classify the blink comparison result into a verdict.

    Returns
    -------
    (verdict, confidence)
        verdict : STATIC, MOVED, ABSENT, or APPEARED
        confidence : high, medium, or low
    """
    det1 = snr1 >= snr_threshold
    det2 = snr2 >= snr_threshold

    if not det1 and not det2:
        verdict = "ABSENT"
        conf = "high" if (snr1 < 1.0 and snr2 < 1.0) else "medium"
    elif det1 and not det2:
        verdict = "APPEARED"
        conf = "medium" if snr1 > 5.0 else "low"
    elif not det1 and det2:
        verdict = "APPEARED"
        conf = "medium" if snr2 > 5.0 else "low"
    elif shift_arcsec > shift_threshold_arcsec and correlation_snr > 3.0:
        verdict = "MOVED"
        min_snr = min(snr1, snr2)
        conf = "high" if min_snr > 10.0 else ("medium" if min_snr > 5.0 else "low")
    else:
        verdict = "STATIC"
        min_snr = min(snr1, snr2)
        conf = "high" if min_snr > 10.0 else ("medium" if min_snr > 5.0 else "low")

    return verdict, conf


# ---------------------------------------------------------------------------
# Full blink analysis for a single candidate
# ---------------------------------------------------------------------------

def blink_candidate(
    ra_deg: float,
    dec_deg: float,
    dss2_dir: Path = Path("data/survey_images/dss2_red"),
    ps1_dir: Path = Path("data/survey_images/ps1_r"),
    cutout_arcsec: float = 30.0,
    shift_threshold_arcsec: float = 1.0,
) -> tuple[BlinkResult, np.ndarray | None, np.ndarray | None]:
    """
    Run the full blink analysis for a single candidate.

    Returns
    -------
    (result, cutout1, cutout2)
        result : BlinkResult with verdict and metrics.
        cutout1, cutout2 : Aligned cutout arrays (or None on failure).
    """
    from astroworld.imaging.fits_store import load_fits_image

    result = BlinkResult(ra_deg=ra_deg, dec_deg=dec_deg)

    # Find DSS2 field (use generous radius: SkyView fields are 10' FOV)
    dss2_path = find_dss2_field(ra_deg, dec_deg, dss2_dir, max_separation_arcmin=7.0)
    if dss2_path is None:
        result.verdict = "ERROR"
        result.details = {"error": "No DSS2 field found"}
        return result, None, None

    # Find PS1 file
    ra_str = f"{ra_deg:07.3f}"
    dec_str = f"{dec_deg:+07.3f}"
    ps1_path = ps1_dir / f"field_ra{ra_str}_dec{dec_str}.fits"
    if not ps1_path.exists():
        result.verdict = "ERROR"
        result.details = {"error": f"PS1 file not found: {ps1_path}"}
        return result, None, None

    # Load images
    try:
        img1, hdr1 = load_fits_image(dss2_path)
        img2, hdr2 = load_fits_image(ps1_path)
    except Exception as e:
        result.verdict = "ERROR"
        result.details = {"error": f"FITS load failed: {e}"}
        return result, None, None

    # Extract epoch from headers
    for key in ("DATE-OBS", "MJD-OBS"):
        if key in hdr1:
            result.details["dss2_date"] = str(hdr1[key])
        if key in hdr2:
            result.details["ps1_date"] = str(hdr2[key])

    # Align epochs
    try:
        cutout1, cutout2, out_hdr = align_epochs(
            ra_deg, dec_deg, img1, hdr1, img2, hdr2,
            cutout_arcsec=cutout_arcsec,
        )
    except Exception as e:
        result.verdict = "ERROR"
        result.details = {"error": f"Alignment failed: {e}"}
        return result, None, None

    # Measure SNR in both cutouts
    snr_info1 = measure_source_snr(cutout1)
    snr_info2 = measure_source_snr(cutout2)
    result.snr_epoch1 = snr_info1["snr"]
    result.snr_epoch2 = snr_info2["snr"]

    # Flux ratio
    if snr_info1["peak_flux"] > 0 and snr_info2["peak_flux"] > 0:
        result.flux_ratio = snr_info2["peak_flux"] / snr_info1["peak_flux"]

    # Measure shift
    shift_info = measure_shift(cutout1, cutout2)
    result.shift_arcsec = shift_info["shift_arcsec"]
    result.shift_ra_arcsec = shift_info["shift_ra_arcsec"]
    result.shift_dec_arcsec = shift_info["shift_dec_arcsec"]

    # Classify verdict
    verdict, conf = classify_verdict(
        result.snr_epoch1,
        result.snr_epoch2,
        result.shift_arcsec,
        shift_info["correlation_snr"],
        shift_threshold_arcsec=shift_threshold_arcsec,
    )
    result.verdict = verdict
    result.confidence = conf

    # Zero out shift for non-detections (cross-corr of noise is meaningless)
    if verdict in ("ABSENT", "APPEARED"):
        result.shift_arcsec = 0.0
        result.shift_ra_arcsec = 0.0
        result.shift_dec_arcsec = 0.0

    # Implied proper motion
    if result.baseline_years > 0:
        result.implied_pm_arcsec_yr = result.shift_arcsec / result.baseline_years

    result.details.update({
        "dss2_field": str(dss2_path.name),
        "ps1_file": str(ps1_path.name),
        "correlation_snr": shift_info["correlation_snr"],
        "shift_x_pix": shift_info["shift_x_pix"],
        "shift_y_pix": shift_info["shift_y_pix"],
    })

    return result, cutout1, cutout2


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def blink_candidates(
    df,  # pd.DataFrame
    dss2_dir: Path = Path("data/survey_images/dss2_red"),
    ps1_dir: Path = Path("data/survey_images/ps1_r"),
    cutout_arcsec: float = 30.0,
    shift_threshold_arcsec: float = 1.0,
    verbose: bool = True,
) -> list[tuple[BlinkResult, np.ndarray | None, np.ndarray | None]]:
    """
    Run blink analysis on a DataFrame of candidates.

    Parameters
    ----------
    df : DataFrame with ``ra_deg`` and ``dec_deg`` columns.

    Returns
    -------
    List of (BlinkResult, cutout1, cutout2) tuples.
    """
    results = []
    n = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])

        if verbose:
            print(f"  [{i + 1:3d}/{n}] RA={ra:.4f} Dec={dec:+.4f}", end=" ")

        result, c1, c2 = blink_candidate(
            ra, dec, dss2_dir, ps1_dir,
            cutout_arcsec=cutout_arcsec,
            shift_threshold_arcsec=shift_threshold_arcsec,
        )

        if verbose:
            print(f"-> {result.verdict} "
                  f"(shift={result.shift_arcsec:.2f}\", "
                  f"SNR1={result.snr_epoch1:.1f}, "
                  f"SNR2={result.snr_epoch2:.1f}, "
                  f"conf={result.confidence})")

        results.append((result, c1, c2))

    return results


# ---------------------------------------------------------------------------
# Visualization: Blink card
# ---------------------------------------------------------------------------

def make_blink_card(
    result: BlinkResult,
    cutout1: np.ndarray,
    cutout2: np.ndarray,
    save_path: Path | None = None,
    candidate_temp_k: float | None = None,
) -> object:
    """
    Generate a side-by-side blink comparison card.

    Returns matplotlib Figure (or saves to ``save_path``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Color normalization (independent zscale-like per panel)
    def _norm(img):
        med = np.median(img)
        std = np.std(img)
        vmin = med - 2 * std
        vmax = med + 5 * std
        return Normalize(vmin=vmin, vmax=vmax)

    # Panel 1: DSS2 (epoch 1)
    ax1.imshow(cutout1, cmap="gray_r", norm=_norm(cutout1), origin="lower")
    ny, nx = cutout1.shape
    cy, cx = ny // 2, nx // 2
    ax1.axhline(cy, color="cyan", linewidth=0.5, alpha=0.7)
    ax1.axvline(cx, color="cyan", linewidth=0.5, alpha=0.7)
    ax1.set_title(f"DSS2 Red (~{result.epoch1_year:.0f})\n"
                  f"SNR = {result.snr_epoch1:.1f}", fontsize=10)
    ax1.set_xlabel("pixels")
    ax1.set_ylabel("pixels")

    # Panel 2: PS1 (epoch 2)
    ax2.imshow(cutout2, cmap="gray_r", norm=_norm(cutout2), origin="lower")
    ny2, nx2 = cutout2.shape
    cy2, cx2 = ny2 // 2, nx2 // 2
    ax2.axhline(cy2, color="cyan", linewidth=0.5, alpha=0.7)
    ax2.axvline(cx2, color="cyan", linewidth=0.5, alpha=0.7)
    ax2.set_title(f"Pan-STARRS r (~{result.epoch2_year:.0f})\n"
                  f"SNR = {result.snr_epoch2:.1f}", fontsize=10)
    ax2.set_xlabel("pixels")

    # Draw motion vector if MOVED
    if result.verdict == "MOVED":
        dx = result.details.get("shift_x_pix", 0)
        dy = result.details.get("shift_y_pix", 0)
        ax2.annotate("", xy=(cx2 + dx, cy2 + dy), xytext=(cx2, cy2),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2))

    # Verdict color
    verdict_colors = {
        "STATIC": "orange",
        "MOVED": "red",
        "ABSENT": "gray",
        "APPEARED": "lime",
        "ERROR": "darkred",
        "UNKNOWN": "gray",
    }
    color = verdict_colors.get(result.verdict, "white")

    # Bottom annotation
    temp_str = f"T = {candidate_temp_k:.1f} K  " if candidate_temp_k else ""
    fig.suptitle(
        f"RA = {result.ra_deg:.4f}, Dec = {result.dec_deg:+.4f}  |  "
        f"{temp_str}"
        f"Verdict: {result.verdict} ({result.confidence})  |  "
        f"Shift: {result.shift_arcsec:.2f}\"  |  "
        f"PM: {result.implied_pm_arcsec_yr:.3f}\"/yr",
        fontsize=10, fontweight="bold", color=color,
        y=0.02,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

    return fig
