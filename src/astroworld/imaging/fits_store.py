"""
FITS image catalog and storage utilities.

Provides functions to:
  - Scan a directory of FITS files and build a catalog DataFrame
  - Load individual FITS images with their headers
  - Query the catalog by survey, coordinates, or epoch
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits


def load_fits_image(path: Path) -> tuple[np.ndarray, dict]:
    """
    Load a FITS image and return the data array and header as a dict.

    Parameters
    ----------
    path : Path to FITS file

    Returns
    -------
    (image_data, header_dict) tuple
        image_data : 2D numpy array (float64)
        header_dict : dict of FITS header keywords
    """
    path = Path(path)
    with fits.open(str(path)) as hdul:
        data = hdul[0].data
        header = dict(hdul[0].header)

    if data is not None:
        data = data.astype(np.float64)

    return data, header


def build_catalog(image_dir: Path) -> pd.DataFrame:
    """
    Scan a directory tree for FITS files and build a catalog DataFrame.

    Extracts metadata from FITS headers including coordinates, survey info,
    observation epoch, and image dimensions.

    Parameters
    ----------
    image_dir : Root directory to scan for .fits files

    Returns
    -------
    DataFrame with columns:
        filepath, filename, survey_dir, naxis1, naxis2,
        crval1 (RA center), crval2 (Dec center),
        date_obs, object_name
    """
    image_dir = Path(image_dir)
    rows = []

    for fits_path in sorted(image_dir.rglob("*.fits")):
        row = {
            "filepath": str(fits_path),
            "filename": fits_path.name,
            "survey_dir": fits_path.parent.name,
        }

        try:
            with fits.open(str(fits_path)) as hdul:
                header = hdul[0].header

                # Image dimensions
                row["naxis1"] = header.get("NAXIS1", None)
                row["naxis2"] = header.get("NAXIS2", None)

                # WCS center coordinates
                row["crval1"] = header.get("CRVAL1", None)  # RA center (deg)
                row["crval2"] = header.get("CRVAL2", None)  # Dec center (deg)

                # Observation info
                row["date_obs"] = header.get("DATE-OBS", None)
                row["object_name"] = header.get("OBJECT", None)

                # Pixel scale (if available)
                row["cdelt1"] = header.get("CDELT1", None)
                row["cdelt2"] = header.get("CDELT2", None)

        except Exception as e:
            row["error"] = str(e)

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        print(f"Catalog: {len(df)} FITS files found in {image_dir}")
        print(f"  Surveys: {df['survey_dir'].unique().tolist()}")
        if "crval1" in df.columns and df["crval1"].notna().any():
            print(f"  RA range: {df['crval1'].min():.3f} - "
                  f"{df['crval1'].max():.3f} deg")
            print(f"  Dec range: {df['crval2'].min():.3f} - "
                  f"{df['crval2'].max():.3f} deg")
    else:
        print(f"No FITS files found in {image_dir}")

    return df


def query_catalog(
    catalog: pd.DataFrame,
    survey: Optional[str] = None,
    ra_range: Optional[tuple[float, float]] = None,
    dec_range: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Filter a catalog DataFrame by survey and/or coordinate range.

    Parameters
    ----------
    catalog : Catalog DataFrame from build_catalog()
    survey : Filter by survey directory name (e.g., "dss2_red")
    ra_range : (ra_min, ra_max) in degrees
    dec_range : (dec_min, dec_max) in degrees

    Returns
    -------
    Filtered DataFrame
    """
    mask = pd.Series(True, index=catalog.index)

    if survey is not None:
        mask &= catalog["survey_dir"] == survey

    if ra_range is not None and "crval1" in catalog.columns:
        mask &= (catalog["crval1"] >= ra_range[0]) & (
            catalog["crval1"] <= ra_range[1]
        )

    if dec_range is not None and "crval2" in catalog.columns:
        mask &= (catalog["crval2"] >= dec_range[0]) & (
            catalog["crval2"] <= dec_range[1]
        )

    return catalog[mask].copy()
