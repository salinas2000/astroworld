"""
Sky survey FITS image downloader.

Downloads cutout images from astronomical sky surveys using astroquery's
SkyView interface. Supports multiple surveys (DSS, WISE, SDSS, 2MASS)
with rate limiting and resume capability.

Available surveys (via SkyView, no API keys required):
  Optical:  DSS2 Red, DSS2 Blue, DSS2 IR, SDSSr, SDSSg, SDSSi
  Infrared: WISE 3.4, WISE 4.6, WISE 12, WISE 22, 2MASS-J, 2MASS-H, 2MASS-K

For Planet 9 at ~480 AU:
  - Optical (mag ~22-25): DSS2 too shallow, SDSS marginal, ZTF best
  - Infrared (mag ~17): WISE 3.4 µm is the prime band (P9 emits thermally)
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


# Default surveys: optical + infrared coverage
DEFAULT_SURVEYS = ["DSS2 Red", "WISE 3.4"]

# Survey name → filesystem-safe directory name
_SURVEY_DIR_MAP = {
    "DSS2 Red": "dss2_red",
    "DSS2 Blue": "dss2_blue",
    "DSS2 IR": "dss2_ir",
    "WISE 3.4": "wise_3.4",
    "WISE 4.6": "wise_4.6",
    "WISE 12": "wise_12",
    "WISE 22": "wise_22",
    "SDSSr": "sdss_r",
    "SDSSg": "sdss_g",
    "SDSSi": "sdss_i",
    "2MASS-J": "2mass_j",
    "2MASS-H": "2mass_h",
    "2MASS-K": "2mass_k",
    "PS1 r": "ps1_r",
}


def _survey_to_dir(survey_name: str) -> str:
    """Convert survey name to filesystem-safe directory name."""
    return _SURVEY_DIR_MAP.get(
        survey_name,
        survey_name.lower().replace(" ", "_").replace(".", "_"),
    )


def _skyview_get_with_fallback(
    coord: "SkyCoord",
    survey: str,
    pixels: int,
    radius_arcmin: float,
    max_retries: int = 3,
) -> Optional[list]:
    """Try SkyView download with adaptive parameter fallback.

    SkyView's WISE tile mosaic engine sometimes fails (HTTP 404) for
    certain radius/pixel combinations at specific sky positions.  When
    the primary request fails we retry with progressively smaller
    cutout parameters that are more likely to fit within a single WISE
    tile, avoiding the problematic server-side mosaic.

    Returns
    -------
    HDU list on success, None on failure.
    """
    from astroquery.skyview import SkyView

    # Primary attempt + fallback parameter ladders (pixels, radius_arcmin)
    attempts = [
        (pixels, radius_arcmin),           # original request
        (pixels, radius_arcmin * 0.6),     # smaller radius
        (300, min(radius_arcmin, 3.0)),    # reduced resolution + radius
        (200, 2.0),                        # minimal — fits one WISE tile
    ]

    for px, rad in attempts[:max_retries + 1]:
        try:
            hdu_list = SkyView.get_images(
                position=coord,
                survey=[survey],
                pixels=px,
                radius=rad * u.arcmin,
            )
            if hdu_list and len(hdu_list) > 0:
                return hdu_list
        except Exception:
            time.sleep(0.5)

    return None


def download_field(
    ra_deg: float,
    dec_deg: float,
    surveys: Optional[list[str]] = None,
    radius_arcmin: float = 5.0,
    pixels: int = 512,
    output_dir: Path = Path("data/survey_images"),
    rate_limit_sec: float = 1.0,
) -> list[Path]:
    """
    Download FITS cutouts from SkyView for a single sky position.

    Parameters
    ----------
    ra_deg : Right ascension (degrees, ICRS)
    dec_deg : Declination (degrees, ICRS)
    surveys : List of survey names (default: DSS2 Red + WISE 3.4)
    radius_arcmin : Field of view radius in arcminutes
    pixels : Image size in pixels (square)
    output_dir : Base output directory
    rate_limit_sec : Seconds to wait between requests

    Returns
    -------
    List of paths to downloaded FITS files
    """
    from astroquery.skyview import SkyView

    if surveys is None:
        surveys = DEFAULT_SURVEYS

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    downloaded = []

    for survey in surveys:
        survey_dir = output_dir / _survey_to_dir(survey)
        survey_dir.mkdir(parents=True, exist_ok=True)

        # Filename encodes coordinates for easy identification
        ra_str = f"{ra_deg:07.3f}"
        dec_str = f"{dec_deg:+07.3f}"
        filename = f"field_ra{ra_str}_dec{dec_str}.fits"
        filepath = survey_dir / filename

        # Skip if already downloaded (resume support)
        if filepath.exists():
            print(f"  [skip] {filepath.name} (already exists)")
            downloaded.append(filepath)
            continue

        try:
            hdu_list = SkyView.get_images(
                position=coord,
                survey=[survey],
                pixels=pixels,
                radius=radius_arcmin * u.arcmin,
            )

            if hdu_list and len(hdu_list) > 0:
                hdu_list[0].writeto(str(filepath), overwrite=True)
                print(f"  [done] {survey}: {filepath.name} "
                      f"({hdu_list[0][0].data.shape})")
                downloaded.append(filepath)
                time.sleep(rate_limit_sec)
                continue

        except Exception:
            pass

        # --- Fallback: adaptive parameter retry for WISE surveys ---
        if "WISE" in survey:
            hdu_list = _skyview_get_with_fallback(
                coord, survey, pixels, radius_arcmin,
            )
            if hdu_list is not None:
                hdu_list[0].writeto(str(filepath), overwrite=True)
                shape = hdu_list[0][0].data.shape
                print(f"  [done] {survey}: {filepath.name} "
                      f"({shape}) (fallback)")
                downloaded.append(filepath)
                time.sleep(rate_limit_sec)
                continue

        print(f"  [error] {survey}: no data for "
              f"RA={ra_deg:.3f}, Dec={dec_deg:.3f}")

        # Rate limiting (be nice to servers)
        time.sleep(rate_limit_sec)

    return downloaded


def download_p9_search_grid(
    ra_center: float,
    dec_center: float,
    grid_size_deg: float = 5.0,
    grid_step_deg: float = 1.0,
    surveys: Optional[list[str]] = None,
    radius_arcmin: float = 5.0,
    pixels: int = 512,
    output_dir: Path = Path("data/survey_images"),
    rate_limit_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Download a grid of FITS cutouts centered on the predicted P9 position.

    Creates a square grid of pointings around (ra_center, dec_center) and
    downloads each field from the specified surveys.

    Parameters
    ----------
    ra_center : Center RA of search grid (degrees)
    dec_center : Center Dec of search grid (degrees)
    grid_size_deg : Total grid size in degrees (e.g., 5.0 = 5x5 deg)
    grid_step_deg : Step between grid points (degrees)
    surveys : Survey list (default: DSS2 Red + WISE 3.4)
    radius_arcmin : Field of view per pointing (arcmin)
    pixels : Image size in pixels
    output_dir : Base output directory
    rate_limit_sec : Rate limiting between requests

    Returns
    -------
    DataFrame with columns:
        ra_deg, dec_deg, survey, filepath, status
    """
    if surveys is None:
        surveys = DEFAULT_SURVEYS

    # Generate grid of pointings
    half = grid_size_deg / 2.0
    ra_offsets = np.arange(-half, half + grid_step_deg * 0.5, grid_step_deg)
    dec_offsets = np.arange(-half, half + grid_step_deg * 0.5, grid_step_deg)

    # Correct RA offsets for spherical geometry
    cos_dec = np.cos(np.radians(dec_center))
    ra_offsets_corrected = ra_offsets / max(cos_dec, 0.1)

    n_fields = len(ra_offsets_corrected) * len(dec_offsets)
    n_total = n_fields * len(surveys)

    print(f"\n{'=' * 60}")
    print(f"  SURVEY IMAGE DOWNLOAD - SEARCH GRID")
    print(f"  Center: RA={ra_center:.3f} deg, Dec={dec_center:.3f} deg")
    print(f"  Grid: {grid_size_deg} x {grid_size_deg} deg "
          f"(step={grid_step_deg} deg, {n_fields} fields)")
    print(f"  Surveys: {surveys}")
    print(f"  Total downloads: {n_total}")
    print(f"{'=' * 60}\n")

    catalog_rows = []
    count = 0

    for dec_off in dec_offsets:
        for ra_off in ra_offsets_corrected:
            ra = ra_center + ra_off
            dec = dec_center + dec_off
            count += 1

            print(f"[{count}/{n_fields}] RA={ra:.3f} deg, Dec={dec:.3f} deg")

            paths = download_field(
                ra_deg=ra,
                dec_deg=dec,
                surveys=surveys,
                radius_arcmin=radius_arcmin,
                pixels=pixels,
                output_dir=output_dir,
                rate_limit_sec=rate_limit_sec,
            )

            for i, survey in enumerate(surveys):
                filepath = paths[i] if i < len(paths) else None
                catalog_rows.append({
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "survey": survey,
                    "filepath": str(filepath) if filepath else None,
                    "status": "ok" if filepath else "failed",
                })

    catalog_df = pd.DataFrame(catalog_rows)

    # Save catalog
    catalog_path = output_dir / "catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)
    print(f"\nCatalog saved: {catalog_path}")
    print(f"  Total: {len(catalog_df)} entries "
          f"({(catalog_df['status'] == 'ok').sum()} ok, "
          f"{(catalog_df['status'] == 'failed').sum()} failed)")

    return catalog_df
