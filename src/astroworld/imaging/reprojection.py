"""
WCS reprojection for multi-telescope image alignment.

Reprojects two FITS images from different surveys (e.g. DSS2 and SDSS)
to a common WCS grid so the Siamese detector can compare them
pixel-by-pixel.

Uses the ``reproject`` package (astropy-affiliated) for accurate
flux-preserving interpolation.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def build_common_wcs(
    ra_center: float,
    dec_center: float,
    pixel_scale_arcsec: float,
    shape: tuple[int, int],
) -> fits.Header:
    """
    Build a TAN-projected WCS header for a common reference frame.

    Parameters
    ----------
    ra_center, dec_center : float
        Sky coordinates of the image center (degrees).
    pixel_scale_arcsec : float
        Pixel scale in arcseconds per pixel.
    shape : tuple of int
        Output image shape (ny, nx).

    Returns
    -------
    astropy.io.fits.Header
        FITS header with WCS keywords for TAN projection.
    """
    ny, nx = shape
    pixel_scale_deg = pixel_scale_arcsec / 3600.0

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = ra_center
    header["CRVAL2"] = dec_center
    header["CRPIX1"] = (nx + 1) / 2.0
    header["CRPIX2"] = (ny + 1) / 2.0
    header["CDELT1"] = -pixel_scale_deg  # RA increases left (convention)
    header["CDELT2"] = pixel_scale_deg
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["RADESYS"] = "FK5"
    header["EQUINOX"] = 2000.0

    return header


def _get_pixel_scale(header: dict) -> float:
    """
    Extract pixel scale from a FITS header in arcseconds/pixel.

    Handles both CDELT and CD matrix representations.
    """
    if "CDELT1" in header:
        return abs(float(header["CDELT1"])) * 3600.0
    elif "CD1_1" in header:
        cd11 = float(header["CD1_1"])
        cd12 = float(header.get("CD1_2", 0))
        return np.sqrt(cd11**2 + cd12**2) * 3600.0
    else:
        # Fallback to DSS2 typical
        return 1.7


def _get_center_coords(header: dict) -> tuple[float, float]:
    """Extract RA/Dec center from a FITS header."""
    return float(header["CRVAL1"]), float(header["CRVAL2"])


def _sanitize_header(header: dict) -> fits.Header:
    """
    Extract only WCS-safe keywords from a header dict.

    Real FITS headers (especially DSS2) contain COMMENT and HISTORY
    entries that corrupt astropy.wcs.WCS parsing.
    """
    safe_keys = {
        "NAXIS", "NAXIS1", "NAXIS2",
        "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2",
        "CRPIX1", "CRPIX2", "CDELT1", "CDELT2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "CUNIT1", "CUNIT2", "RADESYS", "EQUINOX",
        "CROTA1", "CROTA2", "LONPOLE", "LATPOLE",
    }
    # Also include SIP distortion coefficients
    h = fits.Header()
    for key, val in header.items():
        if key in safe_keys or key.startswith(("A_", "B_", "AP_", "BP_")):
            try:
                h[key] = val
            except (ValueError, TypeError):
                continue
    return h


def reproject_to_common_frame(
    image_1: np.ndarray,
    header_1: dict,
    image_2: np.ndarray,
    header_2: dict,
    target_shape: tuple[int, int] | None = None,
    target_pixel_scale_arcsec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Reproject two images to a common WCS grid.

    The output WCS is centered on the midpoint of the two images'
    centers, with pixel scale matching the finer of the two inputs
    (or ``target_pixel_scale_arcsec`` if specified).

    Parameters
    ----------
    image_1, image_2 : np.ndarray
        Input images (2D arrays, may differ in shape).
    header_1, header_2 : dict
        FITS headers with WCS information.
    target_shape : tuple of int, optional
        Output image shape (ny, nx). Defaults to the larger of the two
        input shapes.
    target_pixel_scale_arcsec : float, optional
        Override output pixel scale. Defaults to the finer (smaller)
        of the two input scales.

    Returns
    -------
    (reprojected_1, reprojected_2, output_header)
        Both images reprojected to the common WCS, plus the output
        FITS header (as dict).

    Raises
    ------
    ImportError
        If ``reproject`` package is not installed.
    """
    try:
        from reproject import reproject_interp
    except ImportError:
        raise ImportError(
            "The 'reproject' package is required for cross-survey alignment. "
            "Install with: uv add 'reproject>=0.13'"
        )

    # Determine pixel scale
    scale_1 = _get_pixel_scale(header_1)
    scale_2 = _get_pixel_scale(header_2)

    if target_pixel_scale_arcsec is None:
        target_pixel_scale_arcsec = min(scale_1, scale_2)

    # Determine output shape
    if target_shape is None:
        target_shape = (
            max(image_1.shape[0], image_2.shape[0]),
            max(image_1.shape[1], image_2.shape[1]),
        )

    # Center: midpoint of the two image centers
    ra1, dec1 = _get_center_coords(header_1)
    ra2, dec2 = _get_center_coords(header_2)
    ra_center = (ra1 + ra2) / 2.0
    dec_center = (dec1 + dec2) / 2.0

    # Build target WCS
    target_header = build_common_wcs(
        ra_center, dec_center,
        target_pixel_scale_arcsec,
        target_shape,
    )
    target_wcs = WCS(target_header)

    # Sanitize input headers for WCS parsing
    wcs_header_1 = _sanitize_header(header_1)
    wcs_header_2 = _sanitize_header(header_2)

    # Reproject both images to the common frame
    reproj_1, footprint_1 = reproject_interp(
        (image_1.astype(np.float64), WCS(wcs_header_1)),
        target_wcs,
        shape_out=target_shape,
        order=1,  # Bilinear interpolation
    )

    reproj_2, footprint_2 = reproject_interp(
        (image_2.astype(np.float64), WCS(wcs_header_2)),
        target_wcs,
        shape_out=target_shape,
        order=1,
    )

    # Replace NaN (outside footprint) with 0
    reproj_1 = np.nan_to_num(reproj_1, nan=0.0)
    reproj_2 = np.nan_to_num(reproj_2, nan=0.0)

    # Convert target header to dict for compatibility
    output_header = dict(target_header)

    return reproj_1, reproj_2, output_header
