"""
Spectral cube assembly for multi-survey image fusion.

Combines optical (DSS2/Pan-STARRS), infrared (WISE W1/W2), uncertainty,
and temporal gradient channels into a 5-channel cube per epoch.

Two assembly modes:
  ``build_spectral_cube``  — Optical-anchored (default): optical as C0 reference.
  ``build_ir_cube``        — IR-only: W1 as C0 reference, no optical needed.

Channel layout (both modes):
  C0: Reference band (optical or W1 in ir_only mode)
  C1: WISE W1 (3.4 um) — thermal emission
  C2: WISE W2 (4.6 um) — thermal confirmation
  C3: Uncertainty map (inverse background variance)
  C4: Temporal gradient (inter-epoch difference / delta_t, zeros in ir_only)

WISE images (2.75 arcsec/px) are reprojected to the reference WCS
using ``reproject_interp``.

Cube assembly is the slowest step due to WCS reprojection.  Use
``build_spectral_cube_cached()`` / ``build_ir_cube_cached()`` for
persistent on-disk caching that avoids re-alignment between runs.

Physical utilities:
  - Planck blackbody flux ratio (W1/W2) for temperature estimation
  - Color temperature from W1/W2 flux ratio (Wien/Planck inversion)
"""

import hashlib
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Band index constants
# ---------------------------------------------------------------------------

BAND_OPTICAL = 0       # Optical survey (DSS2 Red, Pan-STARRS r)
BAND_W1 = 1            # WISE 3.4 um
BAND_W2 = 2            # WISE 4.6 um
BAND_UNCERTAINTY = 3   # Inverse background variance
BAND_TEMPORAL = 4      # Temporal gradient (filled by dataset)

NUM_CHANNELS = 5

# Physical constants for Planck law
WISE_W1_WAVELENGTH_UM = 3.4
WISE_W2_WAVELENGTH_UM = 4.6
PLANCK_H = 6.626e-34        # J s
PLANCK_C = 2.998e8           # m/s
BOLTZMANN_K = 1.381e-23      # J/K


# ---------------------------------------------------------------------------
# Cube assembly
# ---------------------------------------------------------------------------

def build_spectral_cube(
    optical_image: np.ndarray,
    optical_header: dict,
    w1_image: np.ndarray | None = None,
    w1_header: dict | None = None,
    w2_image: np.ndarray | None = None,
    w2_header: dict | None = None,
    target_shape: tuple[int, int] | None = None,
    target_pixel_scale_arcsec: float | None = None,
) -> np.ndarray:
    """Assemble a 5-channel spectral cube from multi-survey images.

    Reprojects WISE W1/W2 images to the optical WCS frame and stacks
    all channels into a single (5, H, W) array.  Missing IR bands are
    filled with zeros.

    Parameters
    ----------
    optical_image : 2D array
        Optical survey image (reference WCS frame).
    optical_header : dict
        FITS header with WCS for the optical image.
    w1_image, w2_image : 2D array or None
        WISE W1/W2 images (may differ in shape/scale).
    w1_header, w2_header : dict or None
        FITS headers with WCS for WISE images.
    target_shape : tuple of int, optional
        Output spatial shape (ny, nx).  Defaults to optical image shape.
    target_pixel_scale_arcsec : float, optional
        Override output pixel scale.  Defaults to optical scale.

    Returns
    -------
    np.ndarray of shape (5, H, W), float32
        Spectral cube with channels: optical, W1, W2, uncertainty, temporal.
        Temporal channel is initialized to zeros (filled by dataset layer).
    """
    if target_shape is None:
        target_shape = optical_image.shape

    ny, nx = target_shape
    cube = np.zeros((NUM_CHANNELS, ny, nx), dtype=np.float32)

    # C0: Optical — crop/pad to target shape
    cube[BAND_OPTICAL] = _match_shape(optical_image, target_shape)

    # C1: WISE W1 — reproject to optical WCS
    if w1_image is not None and w1_header is not None:
        cube[BAND_W1] = _reproject_to_optical(
            w1_image, w1_header,
            optical_header, target_shape, target_pixel_scale_arcsec,
        )

    # C2: WISE W2 — reproject to optical WCS
    if w2_image is not None and w2_header is not None:
        cube[BAND_W2] = _reproject_to_optical(
            w2_image, w2_header,
            optical_header, target_shape, target_pixel_scale_arcsec,
        )

    # C3: Uncertainty map from optical background
    # Compute from full optical, then crop/pad to target shape
    unc_full = compute_uncertainty_map(optical_image)
    cube[BAND_UNCERTAINTY] = _match_shape(unc_full, target_shape)

    # C4: Temporal gradient — initialized to zeros, filled by dataset
    # (requires the second epoch cube)

    return cube


# ---------------------------------------------------------------------------
# IR-only cube assembly
# ---------------------------------------------------------------------------

def build_ir_cube(
    w1_image: np.ndarray,
    w1_header: dict,
    w2_image: np.ndarray | None = None,
    w2_header: dict | None = None,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Assemble a 5-channel spectral cube from IR-only images.

    For IR-only detection mode: uses WISE W1 as the spatial reference
    frame (instead of optical).  Channel layout matches
    ``build_spectral_cube`` for SSTF model compatibility:

    * C0 (BAND_OPTICAL) — W1 image (broadband reference for encoder)
    * C1 (BAND_W1)      — W1 3.4 µm (thermal emission)
    * C2 (BAND_W2)      — W2 4.6 µm reprojected to W1 WCS (or zeros)
    * C3 (BAND_UNCERTAINTY) — Inverse-variance from W1 background
    * C4 (BAND_TEMPORAL) — Zeros (single-epoch IR, no motion baseline)

    Parameters
    ----------
    w1_image : 2D array
        WISE W1 image (reference WCS frame, typically 2.75 arcsec/px).
    w1_header : dict
        FITS header with WCS for the W1 image.
    w2_image : 2D array or None
        WISE W2 image (may differ in shape/WCS).
    w2_header : dict or None
        FITS header with WCS for the W2 image.
    target_shape : tuple of int, optional
        Output spatial shape ``(ny, nx)``.  Defaults to W1 image shape.

    Returns
    -------
    np.ndarray of shape ``(5, H, W)``, float32
        IR-only spectral cube.
    """
    if target_shape is None:
        target_shape = w1_image.shape

    ny, nx = target_shape
    cube = np.zeros((NUM_CHANNELS, ny, nx), dtype=np.float32)

    # C0: W1 as broadband reference (replaces optical channel)
    cube[BAND_OPTICAL] = _match_shape(w1_image, target_shape)

    # C1: WISE W1 (identical to C0 in IR-only mode)
    cube[BAND_W1] = _match_shape(w1_image, target_shape)

    # C2: WISE W2 — reproject to W1 WCS if available
    if w2_image is not None and w2_header is not None:
        cube[BAND_W2] = _reproject_to_reference(
            w2_image, w2_header,
            w1_header, target_shape,
        )

    # C3: Uncertainty map from W1 background
    unc_full = compute_uncertainty_map(w1_image)
    cube[BAND_UNCERTAINTY] = _match_shape(unc_full, target_shape)

    # C4: Temporal gradient — zeros (no multi-epoch IR baseline)

    return cube


def build_ir_cube_cached(
    w1_image: np.ndarray,
    w1_header: dict,
    w2_image: np.ndarray | None = None,
    w2_header: dict | None = None,
    target_shape: tuple[int, int] | None = None,
    cache_dir: Path = Path("data/spectral_cache"),
    cache_key: str | None = None,
) -> np.ndarray:
    """Build IR-only spectral cube with persistent on-disk cache.

    Same caching strategy as ``build_spectral_cube_cached`` but for
    the IR-only cube.  Cache keys are derived from W1 header
    coordinates + target shape.

    Parameters
    ----------
    cache_dir : Path
        Directory for cached ``.npy`` files.
    cache_key : str, optional
        Explicit cache key.  If None, computed from ``w1_header``.
    Other parameters : same as ``build_ir_cube``.

    Returns
    -------
    np.ndarray of shape ``(5, H, W)``, float32
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_key is None:
        cache_key = _compute_cache_key(
            w1_header, target_shape or w1_image.shape,
            None,  # no target_pixel_scale override for IR
        )

    cache_path = cache_dir / f"ircube_{cache_key}.npy"

    if cache_path.exists():
        return np.load(str(cache_path))

    cube = build_ir_cube(
        w1_image, w1_header,
        w2_image, w2_header,
        target_shape,
    )

    np.save(str(cache_path), cube)
    return cube


def _reproject_to_reference(
    ir_image: np.ndarray,
    ir_header: dict,
    ref_header: dict,
    target_shape: tuple[int, int],
    target_pixel_scale_arcsec: float | None = None,
) -> np.ndarray:
    """Reproject an IR image to a reference WCS frame.

    Thin wrapper around ``_reproject_to_optical`` — identical logic,
    but named to avoid confusion in IR-only mode where the reference
    frame is WISE W1 (not optical).
    """
    return _reproject_to_optical(
        ir_image, ir_header,
        ref_header, target_shape,
        target_pixel_scale_arcsec,
    )


def build_spectral_cube_cached(
    optical_image: np.ndarray,
    optical_header: dict,
    w1_image: np.ndarray | None = None,
    w1_header: dict | None = None,
    w2_image: np.ndarray | None = None,
    w2_header: dict | None = None,
    target_shape: tuple[int, int] | None = None,
    target_pixel_scale_arcsec: float | None = None,
    cache_dir: Path = Path("data/spectral_cache"),
    cache_key: str | None = None,
) -> np.ndarray:
    """Build spectral cube with persistent on-disk cache.

    WCS reprojection of WISE images is the slowest step (~1-2 sec per
    cube).  This wrapper caches the assembled cube as a ``.npy`` file
    so subsequent training runs skip reprojection entirely.

    Parameters
    ----------
    cache_dir : Path
        Directory for cached ``.npy`` files.
    cache_key : str, optional
        Explicit cache key.  If None, computed from ``optical_header``
        coordinates + target shape (deterministic hash).
    Other parameters : same as ``build_spectral_cube``.

    Returns
    -------
    np.ndarray of shape (5, H, W), float32
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_key is None:
        cache_key = _compute_cache_key(
            optical_header, target_shape or optical_image.shape,
            target_pixel_scale_arcsec,
        )

    cache_path = cache_dir / f"cube_{cache_key}.npy"

    if cache_path.exists():
        return np.load(str(cache_path))

    cube = build_spectral_cube(
        optical_image, optical_header,
        w1_image, w1_header,
        w2_image, w2_header,
        target_shape, target_pixel_scale_arcsec,
    )

    np.save(str(cache_path), cube)
    return cube


# ---------------------------------------------------------------------------
# Uncertainty map
# ---------------------------------------------------------------------------

def compute_uncertainty_map(
    image: np.ndarray,
    block_size: int = 32,
) -> np.ndarray:
    """Compute pixel-level uncertainty (inverse background variance).

    Divides the image into blocks and estimates the local background
    variance in each block using sigma-clipping.  The uncertainty at
    each pixel is ``1 / sigma_local^2`` (inverse variance weighting).

    Parameters
    ----------
    image : 2D array
        Input image.
    block_size : int
        Side length of local estimation blocks (pixels).

    Returns
    -------
    2D array (same shape as input) of inverse-variance weights, float32.
    Values are >= 0.  Regions with zero variance get weight 0.
    """
    h, w = image.shape
    result = np.zeros_like(image, dtype=np.float32)

    for y0 in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            y1 = min(y0 + block_size, h)
            x1 = min(x0 + block_size, w)
            block = image[y0:y1, x0:x1].astype(np.float64)

            # Sigma-clipped variance (3 iterations, 3-sigma)
            mask = np.ones(block.shape, dtype=bool)
            for _ in range(3):
                masked = block[mask]
                if len(masked) == 0:
                    break
                mean = masked.mean()
                std = masked.std()
                if std < 1e-10:
                    break
                mask = np.abs(block - mean) < 3.0 * std

            masked = block[mask]
            if len(masked) > 0:
                var = max(masked.var(), 1e-10)
                result[y0:y1, x0:x1] = 1.0 / var
            # else: leave as 0 (no data)

    return result


# ---------------------------------------------------------------------------
# Temporal gradient
# ---------------------------------------------------------------------------

def compute_temporal_gradient(
    cube1: np.ndarray,
    cube2: np.ndarray,
    delta_t_days: float,
) -> np.ndarray:
    """Compute normalized temporal gradient between two epoch cubes.

    Uses the optical channel (C0) to compute pixel-level rate of change:
    ``gradient = (cube2[optical] - cube1[optical]) / delta_t``

    The result is normalized to zero-mean, unit-std for network input.

    Parameters
    ----------
    cube1, cube2 : (5, H, W) or (H, W) arrays
        Spectral cubes or single-band images for epochs 1 and 2.
    delta_t_days : float
        Time separation between epochs in days (must be > 0).

    Returns
    -------
    2D array (H, W) of normalized temporal gradient, float32.
    """
    if delta_t_days <= 0:
        raise ValueError(f"delta_t_days must be positive, got {delta_t_days}")

    # Extract optical channel if cube, else use as-is
    if cube1.ndim == 3:
        img1 = cube1[BAND_OPTICAL]
        img2 = cube2[BAND_OPTICAL]
    else:
        img1 = cube1
        img2 = cube2

    gradient = (img2.astype(np.float64) - img1.astype(np.float64)) / delta_t_days

    # Normalize
    gradient = gradient.astype(np.float32)
    std = gradient.std()
    if std > 1e-10:
        mean = gradient.mean()
        gradient = (gradient - mean) / std

    return gradient


# ---------------------------------------------------------------------------
# Planck blackbody physics
# ---------------------------------------------------------------------------

def planck_spectral_radiance(
    wavelength_um: float,
    temperature_k: float,
) -> float:
    """Planck spectral radiance B(lambda, T).

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers.
    temperature_k : float
        Temperature in Kelvin (must be > 0).

    Returns
    -------
    Spectral radiance in W / (m^2 sr um).
    """
    if temperature_k <= 0:
        return 0.0

    lam = wavelength_um * 1e-6  # um -> m
    # B(lambda, T) = 2hc^2 / lambda^5 * 1 / (exp(hc / lambda kT) - 1)
    exponent = PLANCK_H * PLANCK_C / (lam * BOLTZMANN_K * temperature_k)

    # Prevent overflow
    if exponent > 500:
        return 0.0

    denom = np.exp(exponent) - 1.0
    if denom <= 0:
        return 0.0

    numerator = 2.0 * PLANCK_H * PLANCK_C ** 2 / lam ** 5
    return float(numerator / denom)


def planck_flux_ratio(temperature_k: float) -> float:
    """Theoretical WISE W1/W2 flux ratio for a blackbody at given temperature.

    Parameters
    ----------
    temperature_k : float
        Blackbody temperature in Kelvin.

    Returns
    -------
    Ratio B(3.4 um, T) / B(4.6 um, T).
    Returns 0.0 if temperature <= 0 or computation fails.
    """
    if temperature_k <= 0:
        return 0.0

    b_w1 = planck_spectral_radiance(WISE_W1_WAVELENGTH_UM, temperature_k)
    b_w2 = planck_spectral_radiance(WISE_W2_WAVELENGTH_UM, temperature_k)

    if b_w2 <= 0:
        return 0.0

    return b_w1 / b_w2


def estimate_color_temperature(
    w1_flux: float,
    w2_flux: float,
    t_min: float = 1.0,
    t_max: float = 50000.0,
) -> float:
    """Estimate blackbody temperature from WISE W1/W2 flux ratio.

    Inverts the Planck function by finding the temperature T such that
    ``B(3.4 um, T) / B(4.6 um, T) = w1_flux / w2_flux``.

    Uses Brent's method (scipy.optimize.brentq) for robust root finding.

    Parameters
    ----------
    w1_flux, w2_flux : float
        Observed fluxes in WISE W1 and W2 bands.
        Must both be positive.
    t_min, t_max : float
        Search bounds for temperature (Kelvin).

    Returns
    -------
    Estimated temperature in Kelvin.

    Raises
    ------
    ValueError
        If fluxes are non-positive or solution not found.
    """
    if w1_flux <= 0 or w2_flux <= 0:
        raise ValueError(
            f"Fluxes must be positive: W1={w1_flux}, W2={w2_flux}"
        )

    from scipy.optimize import brentq

    observed_ratio = w1_flux / w2_flux

    def _residual(t):
        return planck_flux_ratio(t) - observed_ratio

    try:
        temperature = brentq(_residual, t_min, t_max, xtol=0.1)
        return float(temperature)
    except ValueError:
        # Ratio outside physical range — return boundary
        r_min = planck_flux_ratio(t_min)
        r_max = planck_flux_ratio(t_max)
        if observed_ratio <= r_min:
            return t_min
        elif observed_ratio >= r_max:
            return t_max
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reproject_to_optical(
    ir_image: np.ndarray,
    ir_header: dict,
    optical_header: dict,
    target_shape: tuple[int, int],
    target_pixel_scale_arcsec: float | None,
) -> np.ndarray:
    """Reproject an IR image to the optical WCS frame.

    Uses ``reproject_interp`` from the ``reproject`` package for
    flux-preserving bilinear interpolation.

    Falls back to simple resize if ``reproject`` is not installed.
    """
    try:
        from astroworld.imaging.reprojection import (
            build_common_wcs,
            _get_pixel_scale,
            _sanitize_header,
        )
        from astropy.wcs import WCS
        from reproject import reproject_interp

        # Target WCS from optical
        if target_pixel_scale_arcsec is None:
            target_pixel_scale_arcsec = _get_pixel_scale(optical_header)

        ra_center = float(optical_header.get("CRVAL1", 0))
        dec_center = float(optical_header.get("CRVAL2", 0))

        target_header = build_common_wcs(
            ra_center, dec_center,
            target_pixel_scale_arcsec,
            target_shape,
        )
        target_wcs = WCS(target_header)

        # Sanitize IR header
        ir_wcs_header = _sanitize_header(ir_header)
        ir_wcs = WCS(ir_wcs_header)

        reproj, _footprint = reproject_interp(
            (ir_image.astype(np.float64), ir_wcs),
            target_wcs,
            shape_out=target_shape,
            order=1,
        )

        return np.nan_to_num(reproj, nan=0.0).astype(np.float32)

    except ImportError:
        # Fallback: simple resize with scipy
        from scipy.ndimage import zoom

        zy = target_shape[0] / ir_image.shape[0]
        zx = target_shape[1] / ir_image.shape[1]
        return zoom(ir_image.astype(np.float32), (zy, zx), order=1)


def _match_shape(
    image: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Crop or zero-pad a 2D image to match target shape."""
    h, w = image.shape
    th, tw = target_shape

    result = np.zeros(target_shape, dtype=np.float32)

    # Copy the overlapping region
    copy_h = min(h, th)
    copy_w = min(w, tw)
    result[:copy_h, :copy_w] = image[:copy_h, :copy_w]

    return result


def _compute_cache_key(
    header: dict,
    target_shape: tuple[int, int],
    pixel_scale: float | None,
) -> str:
    """Compute a deterministic cache key from WCS + shape parameters."""
    ra = header.get("CRVAL1", 0)
    dec = header.get("CRVAL2", 0)
    key_str = f"ra{ra:.6f}_dec{dec:.6f}_shape{target_shape}_scale{pixel_scale}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]
