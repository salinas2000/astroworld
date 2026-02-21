"""
Morphological pre-filter + NEOWISE multi-epoch IR motion search for Planet 9.

Stage 1: PSF sharpness filter on WISE W2 cutouts to reject diffuse nebulosity
Stage 2: NEOWISE multi-epoch proper motion search (2014-2024, ~20 epochs)

Verdicts:
  POINT_MOVING: Point source with significant PM -> possible P9 / moving object!
  POINT_STATIC: Point source, no PM -> cold star or compact galaxy
  DIFFUSE:      Extended emission -> ISM / molecular cloud dust
  NO_SOURCE:    SNR < 3 in W2 -> pipeline noise
  ERROR:        Technical failure

For Planet 9 at ~480 AU: expected PM ~0.5 arcsec/yr
NEOWISE resolution: ~6.4 arcsec FWHM in W2
With ~20 epochs over 10 yr: 3-sigma sensitivity ~ 0.03 arcsec/yr
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WISE W2 characteristics
WISE_W2_PIXEL_SCALE_ARCSEC = 2.75
WISE_W2_PSF_FWHM_ARCSEC = 6.4
WISE_W2_PSF_FWHM_PIX = WISE_W2_PSF_FWHM_ARCSEC / WISE_W2_PIXEL_SCALE_ARCSEC

# NEOWISE parameters
NEOWISE_SEARCH_RADIUS_DEG = 12.0 / 3600.0  # 12 arcsec
NEOWISE_CATALOG = "neowiser_p1bs_psd"
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP/sync"

# Morphological thresholds
POINTINESS_THRESHOLD = 0.3
MIN_SNR_MORPHOLOGY = 3.0

# Proper motion thresholds
PM_SIGNIFICANCE_THRESHOLD = 3.0
PM_MIN_ARCSEC_YR = 0.3
PM_MAX_ARCSEC_YR = 5.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MorphologyResult:
    """Result of morphological classification for a single source."""

    ra_deg: float
    dec_deg: float
    is_point_source: bool = False
    pointiness: float = 0.0
    fwhm_x_arcsec: float = 0.0
    fwhm_y_arcsec: float = 0.0
    ellipticity: float = 0.0
    snr: float = 0.0
    peak_flux: float = 0.0
    integrated_flux: float = 0.0
    background_mean: float = 0.0
    background_std: float = 0.0
    fit_residual: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class NEOWISEDetection:
    """Single NEOWISE detection at one epoch."""

    mjd: float
    ra_deg: float
    dec_deg: float
    w2mpro: float
    w2sigmpro: float
    qual_frame: int = 0
    qi_fact: int = 0


@dataclass
class ProperMotionResult:
    """Result of proper motion fitting from NEOWISE multi-epoch data."""

    ra_deg: float
    dec_deg: float
    n_detections: int = 0
    n_epochs: int = 0
    baseline_years: float = 0.0
    mu_ra_arcsec_yr: float = 0.0
    mu_dec_arcsec_yr: float = 0.0
    mu_total_arcsec_yr: float = 0.0
    mu_ra_err: float = 0.0
    mu_dec_err: float = 0.0
    mu_total_err: float = 0.0
    pm_significance: float = 0.0
    chi2_reduced: float = 0.0
    mean_w2mag: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class DustPiercerResult:
    """Combined result from both stages for a single candidate."""

    ra_deg: float
    dec_deg: float
    morphology: MorphologyResult | None = None
    passed_morphology: bool = False
    proper_motion: ProperMotionResult | None = None
    has_significant_pm: bool = False
    verdict: str = "UNKNOWN"
    confidence: str = "low"
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 1: Morphological Pre-Filter
# ---------------------------------------------------------------------------

def _gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian model for PSF fitting."""
    x, y = xy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    a = cos_t**2 / (2 * sigma_x**2) + sin_t**2 / (2 * sigma_y**2)
    b = -sin_t * cos_t / (2 * sigma_x**2) + sin_t * cos_t / (2 * sigma_y**2)
    c = sin_t**2 / (2 * sigma_x**2) + cos_t**2 / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(
        -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    )
    return g.ravel()


def fit_gaussian_2d(cutout: np.ndarray) -> dict | None:
    """
    Fit a 2D Gaussian to a small cutout image.

    Returns dict with fit parameters or None on failure.
    """
    ny, nx = cutout.shape
    cy, cx = ny // 2, nx // 2

    # Background estimate
    border = max(1, min(ny, nx) // 4)
    bg_pixels = np.concatenate([
        cutout[:border, :].ravel(),
        cutout[-border:, :].ravel(),
        cutout[:, :border].ravel(),
        cutout[:, -border:].ravel(),
    ])
    bg_median = float(np.median(bg_pixels))
    bg_mad = float(np.median(np.abs(bg_pixels - bg_median)))
    bg_std = 1.4826 * bg_mad if bg_mad > 0 else float(np.std(bg_pixels))

    # Peak above background
    peak = float(cutout[cy, cx])
    amplitude_guess = max(peak - bg_median, bg_std)

    if amplitude_guess < bg_std:
        return None  # No significant source

    # Build coordinate grids
    y_grid, x_grid = np.mgrid[:ny, :nx]
    xy = (x_grid, y_grid)

    # Initial guess
    sigma_guess = WISE_W2_PSF_FWHM_PIX / 2.355
    p0 = [amplitude_guess, float(cx), float(cy), sigma_guess, sigma_guess, 0.0, bg_median]

    # Bounds
    bounds_lo = [0, 0, 0, 0.5, 0.5, -np.pi, -np.inf]
    bounds_hi = [amplitude_guess * 10, nx, ny, nx / 2, ny / 2, np.pi, np.inf]

    try:
        popt, pcov = curve_fit(
            _gaussian_2d, xy, cutout.ravel(), p0=p0,
            bounds=(bounds_lo, bounds_hi), maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return None

    amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt

    # Compute fit residual
    model = _gaussian_2d(xy, *popt).reshape(ny, nx)
    residual_rms = float(np.sqrt(np.mean((cutout - model) ** 2)))

    return {
        "amplitude": float(amplitude),
        "x0": float(x0),
        "y0": float(y0),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "theta": float(theta),
        "offset": float(offset),
        "residual_rms": residual_rms,
    }


def classify_morphology(
    ra_deg: float,
    dec_deg: float,
    w2_image: np.ndarray,
    w2_header: dict,
    cutout_radius_pix: int = 7,
) -> MorphologyResult:
    """
    Classify whether a candidate is a point source or diffuse emission
    in a WISE W2 image.

    Parameters
    ----------
    ra_deg, dec_deg : Candidate sky coordinates.
    w2_image : WISE W2 image array.
    w2_header : FITS header dict with WCS.
    cutout_radius_pix : Half-size of cutout in pixels.

    Returns
    -------
    MorphologyResult with classification.
    """
    result = MorphologyResult(ra_deg=ra_deg, dec_deg=dec_deg)

    # Convert RA/Dec to pixel coordinates
    try:
        from astroworld.imaging.reprojection import _sanitize_header
        clean_hdr = _sanitize_header(w2_header)
        wcs = WCS(clean_hdr)
        px, py = wcs.world_to_pixel_values(ra_deg, dec_deg)
        px, py = int(round(float(px))), int(round(float(py)))
    except Exception as e:
        result.details = {"error": f"WCS conversion failed: {e}"}
        return result

    # Bounds check
    ny, nx = w2_image.shape
    r = cutout_radius_pix
    if px < r or px >= nx - r or py < r or py >= ny - r:
        result.details = {"error": "Candidate near image edge"}
        return result

    # Extract cutout
    cutout = w2_image[py - r: py + r + 1, px - r: px + r + 1].astype(np.float64)

    # Background from FULL image (robust, avoids source contamination)
    # Use pixels far from the candidate center
    full_img = w2_image.astype(np.float64)
    yy, xx = np.ogrid[:ny, :nx]
    far_mask = np.sqrt((xx - px) ** 2 + (yy - py) ** 2) > 3 * r
    if np.sum(far_mask) > 100:
        bg_pixels = full_img[far_mask]
    else:
        # Fallback: use image corners
        border = max(1, ny // 4)
        bg_pixels = np.concatenate([
            full_img[:border, :].ravel(),
            full_img[-border:, :].ravel(),
        ])
    bg_median = float(np.median(bg_pixels))
    bg_mad = float(np.median(np.abs(bg_pixels - bg_median)))
    bg_std = 1.4826 * bg_mad if bg_mad > 0 else float(np.std(bg_pixels))

    result.background_mean = bg_median
    result.background_std = bg_std

    # Peak flux at center
    center_val = float(cutout[r, r])
    result.peak_flux = center_val
    result.snr = (center_val - bg_median) / bg_std if bg_std > 0 else 0.0

    if result.snr < MIN_SNR_MORPHOLOGY:
        result.details = {"reason": "below_snr"}
        return result

    # Fit 2D Gaussian
    fit = fit_gaussian_2d(cutout)
    if fit is None:
        result.details = {"reason": "fit_failed"}
        return result

    result.fit_residual = fit["residual_rms"]

    # FWHM in arcsec
    fwhm_x = fit["sigma_x"] * 2.355 * WISE_W2_PIXEL_SCALE_ARCSEC
    fwhm_y = fit["sigma_y"] * 2.355 * WISE_W2_PIXEL_SCALE_ARCSEC
    result.fwhm_x_arcsec = fwhm_x
    result.fwhm_y_arcsec = fwhm_y

    # Ellipticity
    major = max(fwhm_x, fwhm_y)
    minor = min(fwhm_x, fwhm_y)
    result.ellipticity = 1.0 - minor / major if major > 0 else 0.0

    # Integrated flux in aperture
    y_grid, x_grid = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
    dist = np.sqrt((x_grid - r) ** 2 + (y_grid - r) ** 2)
    aperture_mask = dist <= r

    total_flux = float(np.sum(cutout[aperture_mask] - bg_median))
    result.integrated_flux = total_flux

    # Pointiness: ratio of PSF FWHM to fitted FWHM (1.0 = perfect point source)
    # Use geometric mean of fitted sigmas for symmetric comparison
    fitted_fwhm = np.sqrt(fwhm_x * fwhm_y)
    result.pointiness = WISE_W2_PSF_FWHM_ARCSEC / fitted_fwhm if fitted_fwhm > 0 else 0.0

    # Classification: point source if FWHM close to PSF (pointiness > threshold)
    max_fwhm = 2.0 * WISE_W2_PSF_FWHM_ARCSEC
    result.is_point_source = bool(
        result.pointiness >= POINTINESS_THRESHOLD
        and max(fwhm_x, fwhm_y) < max_fwhm
    )

    result.details = {
        "fit_amplitude": fit["amplitude"],
        "fit_sigma_x_pix": fit["sigma_x"],
        "fit_sigma_y_pix": fit["sigma_y"],
        "fit_theta": fit["theta"],
        "fit_offset": fit["offset"],
    }

    return result


# ---------------------------------------------------------------------------
# Stage 2: NEOWISE Multi-Epoch Motion Search
# ---------------------------------------------------------------------------

def query_neowise_tap(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = NEOWISE_SEARCH_RADIUS_DEG,
    rate_limit_sec: float = 1.0,
) -> list[NEOWISEDetection] | None:
    """
    Query NEOWISE Single-Exposure Source Table for multi-epoch detections.

    Uses astroquery IRSA as primary, raw TAP/ADQL as fallback.

    Returns list of NEOWISEDetection or None on failure.
    """
    detections = _query_neowise_astroquery(ra_deg, dec_deg, radius_deg)

    if detections is None:
        logger.info("astroquery IRSA failed, trying raw TAP fallback")
        detections = _query_neowise_raw_tap(ra_deg, dec_deg, radius_deg)

    time.sleep(rate_limit_sec)
    return detections


def _query_neowise_astroquery(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
) -> list[NEOWISEDetection] | None:
    """Primary: query via astroquery.ipac.irsa."""
    try:
        from astroquery.ipac.irsa import Irsa
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs")
        table = Irsa.query_region(
            coord,
            catalog=NEOWISE_CATALOG,
            spatial="Cone",
            radius=radius_deg * u.deg,
        )

        if table is None or len(table) == 0:
            return []

        return _parse_neowise_table(table)

    except Exception as e:
        logger.warning("astroquery IRSA query failed: %s", e)
        return None


def _query_neowise_raw_tap(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
) -> list[NEOWISEDetection] | None:
    """Fallback: raw TAP/ADQL query to IRSA."""
    adql = (
        "SELECT ra, dec, mjd, w2mpro, w2sigmpro, qual_frame, qi_fact "
        f"FROM {NEOWISE_CATALOG} "
        "WHERE CONTAINS("
        "POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})"
        ") = 1 "
        "AND qi_fact > 0 "
        "AND qual_frame > 0 "
        "AND w2mpro IS NOT NULL "
        "ORDER BY mjd"
    )

    try:
        resp = requests.get(
            IRSA_TAP_URL,
            params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
            timeout=60,
        )
        resp.raise_for_status()

        lines = resp.text.strip().split("\n")
        if len(lines) < 2:
            return []

        # Parse CSV
        detections = []
        header = lines[0].split(",")
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                det = NEOWISEDetection(
                    ra_deg=float(parts[0]),
                    dec_deg=float(parts[1]),
                    mjd=float(parts[2]),
                    w2mpro=float(parts[3]),
                    w2sigmpro=float(parts[4]),
                    qual_frame=int(parts[5]),
                    qi_fact=int(parts[6]),
                )
                detections.append(det)
            except (ValueError, IndexError):
                continue

        return detections

    except Exception as e:
        logger.warning("Raw TAP query failed: %s", e)
        return None


def _parse_neowise_table(table) -> list[NEOWISEDetection]:
    """Parse an astropy Table from IRSA into NEOWISEDetection list."""
    detections = []
    for row in table:
        try:
            # Filter quality
            qi = int(row.get("qi_fact", 0))
            qf = int(row.get("qual_frame", 0))
            if qi <= 0 or qf <= 0:
                continue

            w2 = float(row["w2mpro"])
            w2sig = float(row.get("w2sigmpro", 0.1))

            # Skip invalid magnitudes
            if np.isnan(w2) or w2 <= 0:
                continue

            det = NEOWISEDetection(
                ra_deg=float(row["ra"]),
                dec_deg=float(row["dec"]),
                mjd=float(row["mjd"]),
                w2mpro=w2,
                w2sigmpro=w2sig,
                qual_frame=qf,
                qi_fact=qi,
            )
            detections.append(det)
        except (KeyError, ValueError, TypeError):
            continue

    return detections


# ---------------------------------------------------------------------------
# Epoch grouping
# ---------------------------------------------------------------------------

def group_by_epoch(
    detections: list[NEOWISEDetection],
    window_days: float = 30.0,
) -> list[dict]:
    """
    Group NEOWISE detections into distinct observing epochs.

    NEOWISE visits each sky position for ~1 day every 6 months.
    Groups within ``window_days`` are merged into single epoch measurements.

    Returns list of epoch dicts: mjd, ra_deg, dec_deg, w2mag, n_frames.
    """
    if not detections:
        return []

    # Sort by MJD
    sorted_dets = sorted(detections, key=lambda d: d.mjd)

    epochs = []
    current_group = [sorted_dets[0]]

    for det in sorted_dets[1:]:
        if det.mjd - current_group[0].mjd < window_days:
            current_group.append(det)
        else:
            epochs.append(_summarize_epoch(current_group))
            current_group = [det]

    if current_group:
        epochs.append(_summarize_epoch(current_group))

    return epochs


def _summarize_epoch(group: list[NEOWISEDetection]) -> dict:
    """Compute median position and weighted mean magnitude for an epoch group."""
    mjds = [d.mjd for d in group]
    ras = [d.ra_deg for d in group]
    decs = [d.dec_deg for d in group]
    w2s = [d.w2mpro for d in group]

    return {
        "mjd": float(np.median(mjds)),
        "ra_deg": float(np.median(ras)),
        "dec_deg": float(np.median(decs)),
        "w2mag": float(np.median(w2s)),
        "n_frames": len(group),
    }


# ---------------------------------------------------------------------------
# Proper motion fitting
# ---------------------------------------------------------------------------

def fit_proper_motion(
    epochs: list[dict],
    ra0_deg: float,
    dec0_deg: float,
) -> ProperMotionResult:
    """
    Fit a linear proper motion model to multi-epoch positions.

    Parameters
    ----------
    epochs : List of epoch dicts from ``group_by_epoch()``.
    ra0_deg, dec0_deg : Reference position (initial candidate coordinates).

    Returns
    -------
    ProperMotionResult with PM fit and significance.
    """
    result = ProperMotionResult(ra_deg=ra0_deg, dec_deg=dec0_deg)
    result.n_detections = sum(e["n_frames"] for e in epochs)
    result.n_epochs = len(epochs)

    if len(epochs) < 3:
        result.details = {"reason": "insufficient_epochs"}
        return result

    # Convert to arcsec offsets from reference position
    cos_dec = np.cos(np.radians(dec0_deg))
    mjds = np.array([e["mjd"] for e in epochs])
    ras = np.array([e["ra_deg"] for e in epochs])
    decs = np.array([e["dec_deg"] for e in epochs])

    # Arcsec offsets
    dra_arcsec = (ras - ra0_deg) * cos_dec * 3600.0
    ddec_arcsec = (decs - dec0_deg) * 3600.0

    # Time in years from mean epoch
    t_mean = float(np.mean(mjds))
    t_yr = (mjds - t_mean) / 365.25

    result.baseline_years = float((mjds[-1] - mjds[0]) / 365.25)
    result.mean_w2mag = float(np.mean([e["w2mag"] for e in epochs]))

    # Linear fit: offset = a + mu * t
    try:
        # RA fit
        coeffs_ra, cov_ra = np.polyfit(t_yr, dra_arcsec, 1, cov=True)
        mu_ra = float(coeffs_ra[0])
        mu_ra_err = float(np.sqrt(cov_ra[0, 0]))

        # Dec fit
        coeffs_dec, cov_dec = np.polyfit(t_yr, ddec_arcsec, 1, cov=True)
        mu_dec = float(coeffs_dec[0])
        mu_dec_err = float(np.sqrt(cov_dec[0, 0]))
    except (np.linalg.LinAlgError, ValueError) as e:
        result.details = {"reason": f"fit_failed: {e}"}
        return result

    result.mu_ra_arcsec_yr = mu_ra
    result.mu_dec_arcsec_yr = mu_dec
    result.mu_ra_err = mu_ra_err
    result.mu_dec_err = mu_dec_err

    # Total PM
    mu_total = np.sqrt(mu_ra**2 + mu_dec**2)
    result.mu_total_arcsec_yr = float(mu_total)

    # Error propagation for total PM
    if mu_total > 0:
        mu_total_err = np.sqrt(
            (mu_ra * mu_ra_err) ** 2 + (mu_dec * mu_dec_err) ** 2
        ) / mu_total
        result.mu_total_err = float(mu_total_err)
        result.pm_significance = mu_total / mu_total_err if mu_total_err > 0 else 0.0
    else:
        result.mu_total_err = float(np.sqrt(mu_ra_err**2 + mu_dec_err**2))

    # Reduced chi-squared
    dra_model = coeffs_ra[1] + mu_ra * t_yr
    ddec_model = coeffs_dec[1] + mu_dec * t_yr
    resid_ra = dra_arcsec - dra_model
    resid_dec = ddec_arcsec - ddec_model
    n_dof = len(epochs) - 2
    if n_dof > 0:
        result.chi2_reduced = float(
            (np.sum(resid_ra**2) + np.sum(resid_dec**2)) / (2 * n_dof)
        )

    result.details = {
        "t_mean_mjd": t_mean,
        "ra_coeffs": coeffs_ra.tolist(),
        "dec_coeffs": coeffs_dec.tolist(),
    }

    return result


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------

def classify_dust_piercer_verdict(
    morph: MorphologyResult,
    pm: ProperMotionResult | None = None,
) -> tuple[str, str]:
    """
    Classify the combined morphology + PM result into a verdict.

    Returns (verdict, confidence).
    """
    if morph.snr < MIN_SNR_MORPHOLOGY:
        conf = "high" if morph.snr < 1.0 else "medium"
        return "NO_SOURCE", conf

    if not morph.is_point_source:
        conf = "high" if morph.snr > 5.0 else "medium"
        return "DIFFUSE", conf

    # Point source — check PM
    if pm is None or pm.n_epochs < 3:
        return "POINT_STATIC", "low"

    if (
        pm.pm_significance >= PM_SIGNIFICANCE_THRESHOLD
        and PM_MIN_ARCSEC_YR <= pm.mu_total_arcsec_yr <= PM_MAX_ARCSEC_YR
    ):
        conf = "high" if pm.pm_significance > 5.0 else "medium"
        return "POINT_MOVING", conf

    conf = "high" if pm.n_epochs >= 5 else "medium"
    return "POINT_STATIC", conf


# ---------------------------------------------------------------------------
# Integration: full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_candidate(
    ra_deg: float,
    dec_deg: float,
    w2_image: np.ndarray,
    w2_header: dict,
    rate_limit_sec: float = 1.0,
    skip_neowise: bool = False,
) -> DustPiercerResult:
    """
    Run the full Dust Piercer analysis for a single candidate.

    Stage 1: Morphological classification in WISE W2.
    Stage 2: NEOWISE multi-epoch proper motion (if point source).

    Parameters
    ----------
    ra_deg, dec_deg : Candidate sky coordinates.
    w2_image : WISE W2 image data.
    w2_header : WISE W2 FITS header.
    rate_limit_sec : Delay between NEOWISE queries.
    skip_neowise : If True, skip Stage 2.

    Returns
    -------
    DustPiercerResult with verdict and all metrics.
    """
    result = DustPiercerResult(ra_deg=ra_deg, dec_deg=dec_deg)

    # Stage 1: Morphology
    morph = classify_morphology(ra_deg, dec_deg, w2_image, w2_header)
    result.morphology = morph
    result.passed_morphology = morph.is_point_source

    # Stage 2: NEOWISE (only if point source and not skipped)
    pm = None
    if morph.is_point_source and not skip_neowise:
        detections = query_neowise_tap(ra_deg, dec_deg, rate_limit_sec=rate_limit_sec)
        if detections is not None and len(detections) > 0:
            epochs = group_by_epoch(detections)
            pm = fit_proper_motion(epochs, ra_deg, dec_deg)
            result.proper_motion = pm
            result.has_significant_pm = (
                pm.pm_significance >= PM_SIGNIFICANCE_THRESHOLD
                and pm.mu_total_arcsec_yr >= PM_MIN_ARCSEC_YR
            )
        elif detections is not None:
            result.details["neowise"] = "no_detections"
        else:
            result.details["neowise"] = "query_failed"

    # Final verdict
    verdict, conf = classify_dust_piercer_verdict(morph, pm)
    result.verdict = verdict
    result.confidence = conf

    return result


def analyze_candidates(
    df: pd.DataFrame,
    w2_dir: Path = Path("data/survey_images/wise_4.6"),
    rate_limit_sec: float = 1.0,
    skip_neowise: bool = False,
    verbose: bool = True,
) -> list[DustPiercerResult]:
    """
    Run Dust Piercer analysis on a DataFrame of candidates.

    Parameters
    ----------
    df : DataFrame with ``ra_deg`` and ``dec_deg`` columns.
    w2_dir : Directory containing WISE W2 FITS files.
    rate_limit_sec : Delay between NEOWISE queries.
    skip_neowise : If True, skip Stage 2 for all candidates.
    verbose : Print progress to stdout.

    Returns
    -------
    List of DustPiercerResult.
    """
    from astroworld.imaging.fits_store import load_fits_image
    from astroworld.imaging.time_machine import find_dss2_field

    results = []
    n = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])

        if verbose:
            print(f"  [{i + 1:3d}/{n}] RA={ra:.4f} Dec={dec:+.4f}", end=" ")

        # Find WISE W2 field
        w2_path = find_dss2_field(ra, dec, w2_dir, max_separation_arcmin=7.0)
        if w2_path is None:
            if verbose:
                print("-> ERROR (no W2 field)")
            res = DustPiercerResult(
                ra_deg=ra, dec_deg=dec, verdict="ERROR", confidence="low",
                details={"error": "No WISE W2 field found"},
            )
            results.append(res)
            continue

        # Load WISE W2 image
        try:
            w2_img, w2_hdr = load_fits_image(w2_path)
        except Exception as e:
            if verbose:
                print(f"-> ERROR (FITS load: {e})")
            res = DustPiercerResult(
                ra_deg=ra, dec_deg=dec, verdict="ERROR", confidence="low",
                details={"error": f"FITS load failed: {e}"},
            )
            results.append(res)
            continue

        # Run analysis
        res = analyze_candidate(
            ra, dec, w2_img, w2_hdr,
            rate_limit_sec=rate_limit_sec,
            skip_neowise=skip_neowise,
        )

        if verbose:
            morph_str = (
                f"point(P={res.morphology.pointiness:.2f})"
                if res.morphology and res.morphology.is_point_source
                else f"diffuse(P={res.morphology.pointiness:.2f})"
                if res.morphology and res.morphology.snr >= MIN_SNR_MORPHOLOGY
                else "no_source"
            )
            pm_str = ""
            if res.proper_motion and res.proper_motion.n_epochs >= 3:
                pm_str = (
                    f" PM={res.proper_motion.mu_total_arcsec_yr:.3f}\"/yr"
                    f"({res.proper_motion.pm_significance:.1f}s)"
                )
            print(f"-> {res.verdict} [{morph_str}{pm_str}] ({res.confidence})")

        results.append(res)

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_dust_piercer_card(
    result: DustPiercerResult,
    w2_cutout: np.ndarray | None = None,
    save_path: Path | None = None,
    candidate_temp_k: float | None = None,
) -> object:
    """
    Generate a 3-panel Dust Piercer analysis card.

    Panel 1: WISE W2 cutout with Gaussian fit contours.
    Panel 2: NEOWISE position vs. time (if available).
    Panel 3: PM vector diagram (if available).

    Returns matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    has_pm = (
        result.proper_motion is not None
        and result.proper_motion.n_epochs >= 3
    )
    n_panels = 3 if has_pm else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Verdict colors
    verdict_colors = {
        "POINT_MOVING": "red",
        "POINT_STATIC": "cyan",
        "DIFFUSE": "gray",
        "NO_SOURCE": "darkgray",
        "ERROR": "darkred",
        "UNKNOWN": "white",
    }
    color = verdict_colors.get(result.verdict, "white")

    # Panel 1: W2 cutout
    ax1 = axes[0]
    if w2_cutout is not None and w2_cutout.size > 0:
        med = np.median(w2_cutout)
        std = np.std(w2_cutout)
        norm = Normalize(vmin=med - 2 * std, vmax=med + 5 * std)
        ax1.imshow(w2_cutout, cmap="inferno", norm=norm, origin="lower")

        ny, nx = w2_cutout.shape
        cy, cx = ny // 2, nx // 2
        ax1.axhline(cy, color="cyan", linewidth=0.5, alpha=0.5)
        ax1.axvline(cx, color="cyan", linewidth=0.5, alpha=0.5)

        # Gaussian contours if fit available
        if result.morphology and result.morphology.details.get("fit_amplitude"):
            d = result.morphology.details
            y_g, x_g = np.mgrid[:ny, :nx]
            xy = (x_g, y_g)
            model = _gaussian_2d(
                xy, d["fit_amplitude"],
                d.get("fit_x0", cx), d.get("fit_y0", cy),
                d["fit_sigma_x_pix"], d["fit_sigma_y_pix"],
                d["fit_theta"], d["fit_offset"],
            ).reshape(ny, nx)
            ax1.contour(model, levels=3, colors="lime", linewidths=0.5, alpha=0.7)

        snr_str = f"SNR={result.morphology.snr:.1f}" if result.morphology else ""
        point_str = f"P={result.morphology.pointiness:.2f}" if result.morphology else ""
        ax1.set_title(f"WISE W2 (4.6 um)\n{snr_str}  {point_str}", fontsize=10)
    else:
        ax1.text(0.5, 0.5, "No W2 cutout", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=12, color="gray")
        ax1.set_title("WISE W2 (4.6 um)", fontsize=10)

    ax1.set_xlabel("pixels")
    ax1.set_ylabel("pixels")

    # Panel 2: Position vs time
    if has_pm and len(axes) > 1:
        ax2 = axes[1]
        pm = result.proper_motion
        epochs = pm.details.get("ra_coeffs", None)

        # Reconstruct epoch data from PM fit
        if "t_mean_mjd" in pm.details:
            t_mean = pm.details["t_mean_mjd"]
            # We don't store epoch positions, so plot the fit line
            t_range = np.linspace(-pm.baseline_years / 2, pm.baseline_years / 2, 100)
            ra_line = pm.mu_ra_arcsec_yr * t_range
            dec_line = pm.mu_dec_arcsec_yr * t_range
            ax2.plot(t_range + 2014 + (t_mean - 56658) / 365.25,
                     ra_line, "b-", label=f"RA: {pm.mu_ra_arcsec_yr:+.3f}\"/yr")
            ax2.plot(t_range + 2014 + (t_mean - 56658) / 365.25,
                     dec_line, "r-", label=f"Dec: {pm.mu_dec_arcsec_yr:+.3f}\"/yr")
            ax2.set_xlabel("Year")
            ax2.set_ylabel('Offset (")')
            ax2.legend(fontsize=8)
            ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

        ax2.set_title(
            f"NEOWISE PM ({pm.n_epochs} epochs)\n"
            f"Total: {pm.mu_total_arcsec_yr:.3f}\"/yr "
            f"({pm.pm_significance:.1f}sigma)",
            fontsize=10,
        )

    # Panel 3: PM vector
    if has_pm and len(axes) > 2:
        ax3 = axes[2]
        pm = result.proper_motion
        ax3.set_aspect("equal")
        ax3.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax3.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

        # Draw PM vector
        ax3.annotate(
            "", xy=(pm.mu_ra_arcsec_yr, pm.mu_dec_arcsec_yr), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )

        # Error ellipse (simplified)
        theta = np.linspace(0, 2 * np.pi, 100)
        ex = pm.mu_ra_err * np.cos(theta) + pm.mu_ra_arcsec_yr
        ey = pm.mu_dec_err * np.sin(theta) + pm.mu_dec_arcsec_yr
        ax3.plot(ex, ey, "--", color=color, alpha=0.5, linewidth=0.5)

        scale = max(abs(pm.mu_ra_arcsec_yr) + pm.mu_ra_err,
                     abs(pm.mu_dec_arcsec_yr) + pm.mu_dec_err, 0.5) * 1.5
        ax3.set_xlim(-scale, scale)
        ax3.set_ylim(-scale, scale)
        ax3.set_xlabel('RA PM ("/yr)')
        ax3.set_ylabel('Dec PM ("/yr)')
        ax3.set_title(f"PM Vector\nW2 = {pm.mean_w2mag:.1f} mag", fontsize=10)

    # Bottom annotation
    temp_str = f"T = {candidate_temp_k:.1f} K  " if candidate_temp_k else ""
    pm_str = ""
    if has_pm:
        pm_str = f"PM: {result.proper_motion.mu_total_arcsec_yr:.3f}\"/yr  "
    fig.suptitle(
        f"RA = {result.ra_deg:.4f}, Dec = {result.dec_deg:+.4f}  |  "
        f"{temp_str}"
        f"Verdict: {result.verdict} ({result.confidence})  |  "
        f"{pm_str}",
        fontsize=10, fontweight="bold", color=color, y=0.02,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return fig
