"""
Empirical PSF extraction from FITS sky survey images.

Detects bright, isolated stars in a FITS image and builds an empirical
Point Spread Function (ePSF) using the Anderson & King (2000) algorithm.
Falls back to a 2D Gaussian model when too few stars are detected or
when photutils is not installed.

The extracted PSF is used by synthetic_injection.py to inject realistic
synthetic point sources for ML training data generation.
"""

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table

try:
    from photutils.detection import DAOStarFinder
    from photutils.psf import EPSFBuilder, extract_stars

    HAS_PHOTUTILS = True
except ImportError:
    HAS_PHOTUTILS = False

try:
    from scipy.ndimage import shift as ndimage_shift

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class PSFResult:
    """Result of PSF extraction from a FITS image."""

    psf_image: np.ndarray  # 2D normalized PSF array (peak=1.0)
    psf_size: int  # Side length in pixels (odd number)
    fwhm_pixels: float  # Measured FWHM in pixels
    n_stars_used: int  # Number of stars used to build ePSF
    n_stars_detected: int  # Total stars detected before filtering
    method: str  # "empirical" or "gaussian_fallback"
    star_positions: np.ndarray  # (N, 2) array of star pixel coords used
    star_fluxes: np.ndarray  # (N,) array of peak fluxes of used stars
    background_mean: float  # Estimated sky background (ADU)
    background_std: float  # Background noise (ADU)


def estimate_background(
    image: np.ndarray,
    n_iterations: int = 5,
    sigma_clip: float = 3.0,
) -> tuple[float, float]:
    """
    Estimate sky background level and noise via iterative sigma clipping.

    Parameters
    ----------
    image : 2D array of pixel values (ADU)
    n_iterations : Number of sigma-clipping iterations
    sigma_clip : Clipping threshold in sigma units

    Returns
    -------
    (background_mean, background_std) in ADU
    """
    data = image.ravel().astype(np.float64)
    mask = np.ones(len(data), dtype=bool)

    for _ in range(n_iterations):
        clipped = data[mask]
        if len(clipped) == 0:
            break
        mean = np.mean(clipped)
        std = np.std(clipped)
        if std == 0:
            break
        mask = np.abs(data - mean) < sigma_clip * std

    clipped = data[mask]
    return float(np.mean(clipped)), float(np.std(clipped))


def detect_stars(
    image: np.ndarray,
    fwhm_estimate: float = 2.5,
    threshold_sigma: float = 5.0,
    background_mean: float | None = None,
    background_std: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect point sources in a FITS image.

    Uses photutils DAOStarFinder when available, otherwise falls back
    to a simple local-maximum peak finder.

    Parameters
    ----------
    image : 2D image array (float64)
    fwhm_estimate : Expected PSF FWHM in pixels (DSS2 ~ 2-3 px)
    threshold_sigma : Detection threshold in units of background std
    background_mean : Pre-computed background (if None, estimated)
    background_std : Pre-computed noise (if None, estimated)

    Returns
    -------
    (positions, fluxes) where:
        positions : (N, 2) array of (x, y) pixel coordinates
        fluxes : (N,) array of peak pixel values above background
    """
    if background_mean is None or background_std is None:
        bkg_mean, bkg_std = estimate_background(image)
        if background_mean is None:
            background_mean = bkg_mean
        if background_std is None:
            background_std = bkg_std

    # Background-subtracted image
    img_sub = image - background_mean
    threshold = threshold_sigma * background_std

    if HAS_PHOTUTILS:
        finder = DAOStarFinder(
            fwhm=fwhm_estimate,
            threshold=threshold,
        )
        sources = finder(img_sub)

        if sources is None or len(sources) == 0:
            return np.empty((0, 2)), np.empty(0)

        positions = np.column_stack(
            [sources["xcentroid"], sources["ycentroid"]]
        )
        fluxes = np.array(sources["peak"])
        return positions, fluxes

    # Fallback: simple peak finder using local maxima
    return _detect_stars_simple(img_sub, threshold, fwhm_estimate)


def _detect_stars_simple(
    img_sub: np.ndarray,
    threshold: float,
    fwhm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple peak finder fallback when photutils is not available."""
    ny, nx = img_sub.shape
    half_w = max(int(fwhm), 1)
    positions = []
    fluxes = []

    # Scan with a sliding window, find local maxima above threshold
    for y in range(half_w, ny - half_w, max(half_w, 1)):
        for x in range(half_w, nx - half_w, max(half_w, 1)):
            val = img_sub[y, x]
            if val < threshold:
                continue
            # Check if this is the local maximum in a box
            box = img_sub[
                y - half_w : y + half_w + 1,
                x - half_w : x + half_w + 1,
            ]
            if val >= np.max(box):
                positions.append([float(x), float(y)])
                fluxes.append(float(val))

    if len(positions) == 0:
        return np.empty((0, 2)), np.empty(0)

    return np.array(positions), np.array(fluxes)


def _select_psf_stars(
    positions: np.ndarray,
    fluxes: np.ndarray,
    image_shape: tuple[int, int],
    cutout_size: int = 25,
    min_separation_px: float = 15.0,
    flux_percentile_range: tuple[float, float] = (30.0, 95.0),
    max_stars: int = 25,
    edge_margin: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select bright, isolated, non-saturated stars for PSF construction.

    Filters:
      1. Exclude stars too close to image edge
      2. Select stars in flux percentile range (reject faint + saturated)
      3. Require minimum separation from all neighbors
      4. Take up to max_stars, sorted by flux (brightest first)

    Returns
    -------
    (selected_positions, selected_fluxes)
    """
    if len(positions) == 0:
        return np.empty((0, 2)), np.empty(0)

    ny, nx = image_shape
    mask = np.ones(len(positions), dtype=bool)

    # Filter 1: edge margin
    mask &= positions[:, 0] >= edge_margin
    mask &= positions[:, 0] < nx - edge_margin
    mask &= positions[:, 1] >= edge_margin
    mask &= positions[:, 1] < ny - edge_margin

    # Filter 2: flux percentile range
    if mask.sum() > 3:
        valid_fluxes = fluxes[mask]
        lo = np.percentile(valid_fluxes, flux_percentile_range[0])
        hi = np.percentile(valid_fluxes, flux_percentile_range[1])
        mask &= (fluxes >= lo) & (fluxes <= hi)

    # Filter 3: minimum separation
    indices = np.where(mask)[0]
    isolated = np.ones(len(indices), dtype=bool)
    for i_idx in range(len(indices)):
        if not isolated[i_idx]:
            continue
        for j_idx in range(i_idx + 1, len(indices)):
            if not isolated[j_idx]:
                continue
            dist = np.linalg.norm(
                positions[indices[i_idx]] - positions[indices[j_idx]]
            )
            if dist < min_separation_px:
                # Remove the fainter one
                if fluxes[indices[i_idx]] >= fluxes[indices[j_idx]]:
                    isolated[j_idx] = False
                else:
                    isolated[i_idx] = False
                    break

    selected_idx = indices[isolated]

    # Sort by flux (brightest first) and take max_stars
    order = np.argsort(-fluxes[selected_idx])
    selected_idx = selected_idx[order[:max_stars]]

    return positions[selected_idx], fluxes[selected_idx]


def make_gaussian_psf(
    fwhm_pixels: float = 2.5,
    size: int = 25,
) -> np.ndarray:
    """
    Create a normalized 2D Gaussian PSF.

    Parameters
    ----------
    fwhm_pixels : Full width at half maximum in pixels
    size : Output array side length (odd integer)

    Returns
    -------
    2D array with peak normalized to 1.0
    """
    if size % 2 == 0:
        size += 1

    sigma = fwhm_pixels / 2.3548  # FWHM = 2*sqrt(2*ln2) * sigma
    center = size // 2
    y, x = np.mgrid[0:size, 0:size] - center
    r_sq = x ** 2 + y ** 2
    psf = np.exp(-0.5 * r_sq / (sigma ** 2))
    psf /= psf.max()  # Peak normalize to 1.0
    return psf


def _measure_fwhm(psf: np.ndarray) -> float:
    """
    Measure FWHM from a 2D PSF image using radial profile interpolation.

    Parameters
    ----------
    psf : 2D PSF array (peak normalized to 1.0)

    Returns
    -------
    FWHM in pixels
    """
    cy, cx = np.array(psf.shape) / 2.0
    y, x = np.mgrid[0 : psf.shape[0], 0 : psf.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
    vals = psf.ravel()

    # Sort by radius
    order = np.argsort(r)
    r_sorted = r[order]
    v_sorted = vals[order]

    # Find where profile crosses 0.5
    # Use a binned radial profile for robustness
    max_r = min(psf.shape) / 2.0
    n_bins = int(max_r * 2)
    if n_bins < 3:
        return 2.5  # fallback

    r_bins = np.linspace(0, max_r, n_bins)
    profile = np.zeros(n_bins - 1)
    for i in range(n_bins - 1):
        in_bin = (r_sorted >= r_bins[i]) & (r_sorted < r_bins[i + 1])
        if in_bin.sum() > 0:
            profile[i] = np.mean(v_sorted[in_bin])

    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Find the half-maximum radius
    above_half = profile >= 0.5
    if not above_half.any():
        return 2.5

    # Find the last radius where profile is above 0.5
    half_idx = np.where(above_half)[0]
    if len(half_idx) == 0:
        return 2.5

    last_above = half_idx[-1]
    if last_above >= len(r_centers) - 1:
        return r_centers[last_above] * 2.0

    # Linear interpolation between last-above and first-below
    r_a = r_centers[last_above]
    r_b = r_centers[last_above + 1]
    v_a = profile[last_above]
    v_b = profile[last_above + 1]

    if v_a == v_b:
        half_r = r_a
    else:
        half_r = r_a + (0.5 - v_a) * (r_b - r_a) / (v_b - v_a)

    return float(half_r * 2.0)  # FWHM = 2 * half-radius


def extract_psf(
    image: np.ndarray,
    header: dict | None = None,
    fwhm_estimate: float = 2.5,
    psf_size: int = 25,
    threshold_sigma: float = 5.0,
    min_stars: int = 5,
    oversampling: int = 2,
) -> PSFResult:
    """
    Extract an empirical PSF from stars in a FITS image.

    Uses the Anderson & King (2000) ePSF algorithm via photutils.
    Falls back to a Gaussian PSF model if too few stars are found
    or if photutils is not installed.

    Parameters
    ----------
    image : 2D image array (float64)
    header : FITS header dict (optional, for pixel scale info)
    fwhm_estimate : Initial guess for PSF FWHM in pixels
    psf_size : Size of output PSF stamp (odd integer, pixels)
    threshold_sigma : Star detection threshold
    min_stars : Minimum stars required for empirical PSF; below this
                triggers Gaussian fallback
    oversampling : ePSF oversampling factor (2 = Nyquist)

    Returns
    -------
    PSFResult with normalized PSF image and diagnostics
    """
    if psf_size % 2 == 0:
        psf_size += 1

    # Step 1: Background estimation
    bkg_mean, bkg_std = estimate_background(image)

    # Step 2: Star detection
    positions, fluxes = detect_stars(
        image,
        fwhm_estimate=fwhm_estimate,
        threshold_sigma=threshold_sigma,
        background_mean=bkg_mean,
        background_std=bkg_std,
    )
    n_detected = len(positions)

    # Step 3: Select isolated PSF stars
    sel_positions, sel_fluxes = _select_psf_stars(
        positions, fluxes, image.shape, cutout_size=psf_size
    )
    n_selected = len(sel_positions)

    # Step 4: Build PSF
    if n_selected >= min_stars and HAS_PHOTUTILS:
        # Empirical ePSF via Anderson & King algorithm
        try:
            psf_img, fwhm = _build_empirical_psf(
                image, bkg_mean, sel_positions, psf_size, oversampling
            )
            method = "empirical"
        except Exception:
            # If ePSF building fails, fallback to Gaussian
            psf_img = make_gaussian_psf(fwhm_estimate, psf_size)
            fwhm = fwhm_estimate
            method = "gaussian_fallback"
    else:
        # Gaussian fallback
        psf_img = make_gaussian_psf(fwhm_estimate, psf_size)
        fwhm = fwhm_estimate
        method = "gaussian_fallback"

    return PSFResult(
        psf_image=psf_img,
        psf_size=psf_size,
        fwhm_pixels=fwhm,
        n_stars_used=n_selected,
        n_stars_detected=n_detected,
        method=method,
        star_positions=sel_positions,
        star_fluxes=sel_fluxes,
        background_mean=bkg_mean,
        background_std=bkg_std,
    )


def _build_empirical_psf(
    image: np.ndarray,
    bkg_mean: float,
    positions: np.ndarray,
    psf_size: int,
    oversampling: int,
) -> tuple[np.ndarray, float]:
    """
    Build an empirical ePSF from selected star positions.

    Uses photutils EPSFBuilder (Anderson & King 2000 algorithm).

    Returns
    -------
    (psf_image, fwhm_pixels) where psf_image is peak-normalized to 1.0
    """
    # Prepare background-subtracted image as NDData
    img_sub = image - bkg_mean
    nddata = NDData(data=img_sub)

    # Build star table for extract_stars
    stars_tbl = Table()
    stars_tbl["x"] = positions[:, 0]
    stars_tbl["y"] = positions[:, 1]

    # Extract star cutouts
    stars = extract_stars(nddata, stars_tbl, size=psf_size)

    # Build the ePSF
    epsf_builder = EPSFBuilder(
        oversampling=oversampling,
        maxiters=10,
        progress_bar=False,
    )
    epsf, fitted_stars = epsf_builder(stars)

    # Get the ePSF data and downsample to native resolution
    psf_oversampled = epsf.data

    if oversampling > 1:
        # Block-average to native resolution
        oy, ox = psf_oversampled.shape
        ty = oy // oversampling
        tx = ox // oversampling
        # Trim to exact multiple
        trimmed = psf_oversampled[: ty * oversampling, : tx * oversampling]
        psf_native = trimmed.reshape(ty, oversampling, tx, oversampling).mean(
            axis=(1, 3)
        )
    else:
        psf_native = psf_oversampled.copy()

    # Resize to target psf_size if needed
    if psf_native.shape[0] != psf_size or psf_native.shape[1] != psf_size:
        result = np.zeros((psf_size, psf_size), dtype=np.float64)
        cy_src, cx_src = np.array(psf_native.shape) // 2
        cy_dst, cx_dst = psf_size // 2, psf_size // 2
        # Copy the overlapping region
        h = min(cy_src, cy_dst)
        w = min(cx_src, cx_dst)
        result[cy_dst - h : cy_dst + h + 1, cx_dst - w : cx_dst + w + 1] = (
            psf_native[cy_src - h : cy_src + h + 1, cx_src - w : cx_src + w + 1]
        )
        psf_native = result

    # Normalize to peak = 1.0
    peak = psf_native.max()
    if peak > 0:
        psf_native /= peak

    # Measure FWHM
    fwhm = _measure_fwhm(psf_native)

    return psf_native, fwhm
