"""
Synthetic point source injection into FITS sky survey images.

Injects realistic synthetic Planet 9-like sources into real FITS images
using the empirical PSF extracted from existing stars. Supports:
  - WCS-based coordinate-to-pixel conversion
  - Photometric zero-point estimation (with DSS2-safe fallbacks)
  - Sub-pixel source placement via interpolation
  - Local Gaussian noise matching the background statistics

Designed for generating ML training data for Planet 9 detection.
"""

from dataclasses import dataclass

import numpy as np
from astropy.wcs import WCS

from astroworld.imaging.psf_extraction import (
    PSFResult,
    detect_stars,
    estimate_background,
)

try:
    from scipy.ndimage import shift as ndimage_shift

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Default zero point if no stars detected and no header calibration.
# Chosen as a reasonable value for DSS2 Red plate scans.
DEFAULT_ZERO_POINT = 22.5

# Keys that are safe for WCS construction. COMMENT/HISTORY entries from
# dict(header) often contain newlines or non-ASCII text that make
# astropy's WCS(header) raise ValueError.
_WCS_SAFE_KEYS = {
    "NAXIS", "NAXIS1", "NAXIS2",
    "CTYPE1", "CTYPE2",
    "CRVAL1", "CRVAL2",
    "CRPIX1", "CRPIX2",
    "CDELT1", "CDELT2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "CROTA1", "CROTA2",
    "LONPOLE", "LATPOLE",
    "EQUINOX", "EPOCH", "RADESYS",
    "PV1_0", "PV1_1", "PV1_2", "PV2_0", "PV2_1", "PV2_2",
    "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
    "WCSAXES", "WCSNAME",
}


def _sanitize_header_for_wcs(header: dict) -> dict:
    """Extract only WCS-relevant keywords from a FITS header dict.

    Real FITS headers (e.g. DSS2 from SkyView) contain COMMENT and
    HISTORY entries with newlines and non-ASCII characters.  When the
    header is stored as a plain dict, these corrupt values cause
    ``astropy.wcs.WCS(header)`` to raise ``ValueError``.

    This function keeps only the keywords needed for coordinate
    transforms, which is both safe and sufficient.
    """
    clean = {}
    for key, val in header.items():
        # Keep known WCS keys
        if key.upper() in _WCS_SAFE_KEYS:
            clean[key] = val
        # Also keep SIP distortion coefficients (A_p_q, B_p_q, etc.)
        elif key and key[0] in ("A", "B") and "_" in key:
            parts = key.split("_")
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                clean[key] = val
    return clean


@dataclass
class InjectionResult:
    """Result of injecting a synthetic source into a FITS image."""

    image_injected: np.ndarray  # 2D modified image
    image_original: np.ndarray  # 2D original (reference copy)
    pixel_x: float  # Sub-pixel x position of injection
    pixel_y: float  # Sub-pixel y position of injection
    ra_deg: float  # Injected RA (degrees)
    dec_deg: float  # Injected Dec (degrees)
    flux_adu: float  # Total injected flux (ADU counts)
    peak_adu: float  # Peak pixel value added
    magnitude: float  # Target magnitude
    snr_estimate: float  # Estimated S/N at injection site
    psf_method: str  # PSF method used


def radec_to_pixel(
    ra_deg: float,
    dec_deg: float,
    header: dict,
) -> tuple[float, float]:
    """
    Convert RA/Dec to pixel coordinates using FITS WCS.

    Parameters
    ----------
    ra_deg : Right ascension (degrees, ICRS)
    dec_deg : Declination (degrees, ICRS)
    header : FITS header dict with WCS keywords (CRVAL, CRPIX, CDELT, CTYPE)

    Returns
    -------
    (pixel_x, pixel_y) as floats (may be sub-pixel)

    Raises
    ------
    ValueError : If the position falls outside the image
    """
    wcs = WCS(_sanitize_header_for_wcs(header))
    px, py = wcs.all_world2pix(ra_deg, dec_deg, 0)
    px, py = float(px), float(py)

    # Bounds check (NaN from WCS also triggers out-of-bounds)
    nx = header.get("NAXIS1", 512)
    ny = header.get("NAXIS2", 512)
    if np.isnan(px) or np.isnan(py) or px < 0 or px >= nx or py < 0 or py >= ny:
        raise ValueError(
            f"Position RA={ra_deg:.4f}, Dec={dec_deg:.4f} maps to "
            f"pixel ({px:.1f}, {py:.1f}) which is outside the "
            f"{nx}x{ny} image"
        )

    return px, py


def pixel_to_radec(
    pixel_x: float,
    pixel_y: float,
    header: dict,
) -> tuple[float, float]:
    """
    Convert pixel coordinates to RA/Dec using FITS WCS.

    Parameters
    ----------
    pixel_x, pixel_y : Pixel coordinates (0-indexed)
    header : FITS header dict with WCS keywords

    Returns
    -------
    (ra_deg, dec_deg) in degrees
    """
    wcs = WCS(_sanitize_header_for_wcs(header))
    ra, dec = wcs.all_pix2world(pixel_x, pixel_y, 0)
    return float(ra), float(dec)


def estimate_zero_point(
    image: np.ndarray,
    header: dict,
    star_positions: np.ndarray | None = None,
    star_fluxes: np.ndarray | None = None,
    reference_magnitude: float = 15.0,
    reference_percentile: float = 75.0,
) -> float:
    """
    Estimate photometric zero point for magnitude-to-flux conversion.

    For DSS2 images without standard photometric headers, estimates
    the zero point from detected star statistics. Uses the Nth
    percentile star flux as a reference point corresponding to a
    reference magnitude typical for that survey depth.

    Strategy:
      ZP = reference_magnitude + 2.5 * log10(reference_flux)

    For training data generation, exact calibration is not critical --
    what matters is that injected sources have realistic appearance
    relative to the existing star field.

    Parameters
    ----------
    image : 2D image array
    header : FITS header (checked for PHOTZPT/MAGZPT first)
    star_positions : Pre-detected star positions (optional)
    star_fluxes : Pre-measured star fluxes (optional)
    reference_magnitude : Assumed mag of the reference percentile star
    reference_percentile : Which flux percentile to use as reference

    Returns
    -------
    zero_point : float (mag = ZP - 2.5 * log10(flux_adu))
    """
    # Check header for standard photometric zero point
    for key in ("PHOTZPT", "MAGZPT", "ZEROPT"):
        if key in header:
            val = header[key]
            if isinstance(val, (int, float)) and np.isfinite(val):
                return float(val)

    # Detect stars if not provided
    if star_fluxes is None or len(star_fluxes) == 0:
        positions, fluxes = detect_stars(image)
        star_fluxes = fluxes

    # Need at least 1 star with positive flux
    positive_fluxes = star_fluxes[star_fluxes > 0]
    if len(positive_fluxes) == 0:
        # No stars detected: use safe default ZP
        return DEFAULT_ZERO_POINT

    # Use the Nth percentile flux as reference
    ref_flux = np.percentile(positive_fluxes, reference_percentile)
    if ref_flux <= 0:
        return DEFAULT_ZERO_POINT

    # ZP = mag_ref + 2.5 * log10(flux_ref)
    zp = reference_magnitude + 2.5 * np.log10(ref_flux)
    return float(zp)


def magnitude_to_flux(
    magnitude: float,
    zero_point: float,
) -> float:
    """
    Convert apparent magnitude to ADU flux using the zero point.

    flux = 10^((ZP - magnitude) / 2.5)

    Parameters
    ----------
    magnitude : Target apparent magnitude
    zero_point : Photometric zero point

    Returns
    -------
    flux_adu : Total flux in ADU counts (always >= 0)
    """
    return float(10.0 ** ((zero_point - magnitude) / 2.5))


def inject_source(
    image: np.ndarray,
    psf: np.ndarray,
    pixel_x: float,
    pixel_y: float,
    flux_adu: float,
    background_std: float = 0.0,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Inject a scaled PSF into an image at a given sub-pixel position.

    Noise model: For calibrated survey images (DSS2, WISE), the pixel
    noise is dominated by sky background and readout, not Poisson
    photon counting. We add local Gaussian noise matching the image's
    background_std rather than Poisson noise on the source signal.
    This is more realistic for faint sources (mag > 20) where
    S/N_source << S/N_background.

    Parameters
    ----------
    image : 2D image array (will NOT be modified in place)
    psf : 2D PSF stamp (peak-normalized to 1.0)
    pixel_x, pixel_y : Sub-pixel injection position
    flux_adu : Total flux to inject (scales the PSF)
    background_std : Background noise level (ADU) for Gaussian noise
    add_noise : If True, add local Gaussian noise to injected source
    rng : NumPy random generator for reproducibility

    Returns
    -------
    Modified image with the injected source (new array, original unchanged)
    """
    if rng is None:
        rng = np.random.default_rng()

    result = image.copy()
    psf_h, psf_w = psf.shape

    # Scale PSF to desired total flux
    psf_sum = psf.sum()
    if psf_sum > 0:
        scaled_psf = psf * flux_adu / psf_sum
    else:
        scaled_psf = psf * flux_adu

    # Sub-pixel shift
    ix = int(np.round(pixel_x))
    iy = int(np.round(pixel_y))
    dx = pixel_x - ix
    dy = pixel_y - iy

    if abs(dx) > 0.01 or abs(dy) > 0.01:
        if HAS_SCIPY:
            # Cubic interpolation for sub-pixel shift
            scaled_psf = ndimage_shift(scaled_psf, (dy, dx), order=3, mode="constant")
        else:
            # Bilinear interpolation fallback
            scaled_psf = _bilinear_shift(scaled_psf, dx, dy)

    # Add local Gaussian noise (not Poisson) to match background statistics
    if add_noise and background_std > 0:
        noise = rng.normal(0.0, background_std, scaled_psf.shape)
        # Only add noise where the PSF has significant signal
        noise_mask = scaled_psf > 0.01 * scaled_psf.max()
        scaled_psf = scaled_psf + noise * noise_mask

    # Compute stamp placement on image
    half_h = psf_h // 2
    half_w = psf_w // 2

    # Image bounds
    ny, nx = result.shape
    y_start = iy - half_h
    y_end = iy - half_h + psf_h
    x_start = ix - half_w
    x_end = ix - half_w + psf_w

    # Clip to image boundaries
    psf_y_start = max(0, -y_start)
    psf_y_end = psf_h - max(0, y_end - ny)
    psf_x_start = max(0, -x_start)
    psf_x_end = psf_w - max(0, x_end - nx)

    img_y_start = max(0, y_start)
    img_y_end = min(ny, y_end)
    img_x_start = max(0, x_start)
    img_x_end = min(nx, x_end)

    # Add the PSF stamp
    if img_y_end > img_y_start and img_x_end > img_x_start:
        result[img_y_start:img_y_end, img_x_start:img_x_end] += (
            scaled_psf[psf_y_start:psf_y_end, psf_x_start:psf_x_end]
        )

    return result


def _bilinear_shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Bilinear interpolation for sub-pixel shifting (no scipy needed)."""
    ny, nx = image.shape
    result = np.zeros_like(image)

    # Weights for bilinear interpolation
    ax = abs(dx)
    ay = abs(dy)

    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1

    for y in range(ny):
        for x in range(nx):
            # Source coordinates
            src_x = x - dx
            src_y = y - dy
            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))
            x1 = x0 + 1
            y1 = y0 + 1
            fx = src_x - x0
            fy = src_y - y0

            val = 0.0
            if 0 <= x0 < nx and 0 <= y0 < ny:
                val += (1 - fx) * (1 - fy) * image[y0, x0]
            if 0 <= x1 < nx and 0 <= y0 < ny:
                val += fx * (1 - fy) * image[y0, x1]
            if 0 <= x0 < nx and 0 <= y1 < ny:
                val += (1 - fx) * fy * image[y1, x0]
            if 0 <= x1 < nx and 0 <= y1 < ny:
                val += fx * fy * image[y1, x1]

            result[y, x] = val

    return result


def inject_p9_source(
    image: np.ndarray,
    header: dict,
    psf_result: PSFResult,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    magnitude: float = 22.5,
    epoch_jd: float | None = None,
    zero_point: float | None = None,
    add_noise: bool = True,
    random_position: bool = False,
    rng: np.random.Generator | None = None,
) -> InjectionResult:
    """
    Inject a synthetic Planet 9-like source into a FITS image.

    Combines WCS conversion, zero-point estimation, flux scaling,
    and PSF injection into a single call.

    Parameters
    ----------
    image : 2D image array
    header : FITS header dict with WCS
    psf_result : PSFResult from extract_psf()
    ra_deg, dec_deg : Sky position for injection (optional)
    magnitude : Target apparent magnitude (default 22.5 for P9)
    epoch_jd : Julian Date for P9 position prediction (uses sky_position)
    zero_point : Photometric zero point (estimated if None)
    add_noise : Add local Gaussian noise to injected source
    random_position : Place at random position (if ra/dec not given)
    rng : NumPy random generator for reproducibility

    Returns
    -------
    InjectionResult with the modified image and metadata
    """
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = image.shape

    # Determine injection position
    if ra_deg is not None and dec_deg is not None:
        px, py = radec_to_pixel(ra_deg, dec_deg, header)
    elif epoch_jd is not None:
        # Use P9 sky position prediction
        from astroworld.imaging.sky_position import (
            predict_p9_sky_position,
            DEFAULT_P9,
        )

        pos = predict_p9_sky_position(epoch_jd=epoch_jd, **DEFAULT_P9)
        ra_deg = pos["ra_deg"]
        dec_deg = pos["dec_deg"]
        px, py = radec_to_pixel(ra_deg, dec_deg, header)
    elif random_position:
        # Random pixel position with edge margin
        margin = psf_result.psf_size
        px = rng.uniform(margin, nx - margin)
        py = rng.uniform(margin, ny - margin)
        ra_deg, dec_deg = pixel_to_radec(px, py, header)
    else:
        # Default: image center
        px = nx / 2.0
        py = ny / 2.0
        ra_deg, dec_deg = pixel_to_radec(px, py, header)

    # Estimate zero point
    if zero_point is None:
        zero_point = estimate_zero_point(
            image,
            header,
            star_positions=psf_result.star_positions,
            star_fluxes=psf_result.star_fluxes,
        )

    # Convert magnitude to flux
    flux_adu = magnitude_to_flux(magnitude, zero_point)

    # Inject source
    original = image.copy()
    injected = inject_source(
        image=image,
        psf=psf_result.psf_image,
        pixel_x=px,
        pixel_y=py,
        flux_adu=flux_adu,
        background_std=psf_result.background_std,
        add_noise=add_noise,
        rng=rng,
    )

    # Peak ADU added
    psf_peak = psf_result.psf_image.max()
    psf_sum = psf_result.psf_image.sum()
    peak_adu = flux_adu * psf_peak / psf_sum if psf_sum > 0 else 0.0

    # S/N estimate
    if psf_result.background_std > 0:
        snr = peak_adu / psf_result.background_std
    else:
        snr = float("inf") if peak_adu > 0 else 0.0

    return InjectionResult(
        image_injected=injected,
        image_original=original,
        pixel_x=float(px),
        pixel_y=float(py),
        ra_deg=float(ra_deg),
        dec_deg=float(dec_deg),
        flux_adu=float(flux_adu),
        peak_adu=float(peak_adu),
        magnitude=float(magnitude),
        snr_estimate=float(snr),
        psf_method=psf_result.method,
    )
