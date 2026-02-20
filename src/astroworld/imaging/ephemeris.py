"""
JPL Horizons sky-plane ephemeris client.

Queries JPL Horizons for observable quantities (RA, Dec, magnitude,
angular rates) of any solar system body. Complements the existing
data/horizons.py module which fetches state vectors (x, y, z).

For known objects like Sedna, Horizons ephemerides are strictly more
accurate than Keplerian propagation because they include perturbations
from all major planets and relativistic corrections.
"""

import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons


# Sedna JPL small-body designation
SEDNA_ID = "90377"


def fetch_sky_ephemeris(
    body_id: str | int,
    epochs: list[float] | dict,
    location: str = "500",
) -> pd.DataFrame:
    """
    Query JPL Horizons for sky-plane ephemeris (RA, Dec, magnitude).

    Parameters
    ----------
    body_id : str or int
        JPL body identifier. Use "90377" for Sedna, planet IDs (e.g. 5
        for Jupiter), or small-body designations (e.g. "2012 VP113").
    epochs : list of float or dict
        Julian dates as a list, or a dict with 'start', 'stop', 'step'
        keys (e.g. {"start": "2000-01-01", "stop": "2020-01-01",
        "step": "1y"}).
    location : str
        Observer location code. "500" = geocentric (matches survey
        observations). "@sun" = heliocentric.

    Returns
    -------
    pd.DataFrame
        Columns: datetime_jd, ra_deg, dec_deg, V_mag, delta_au,
        r_au, ra_rate_arcsec_hr, dec_rate_arcsec_hr
    """
    # Determine id_type: numbered small bodies use "smallbody"
    id_type = _infer_id_type(body_id)

    obj = Horizons(
        id=str(body_id),
        location=location,
        epochs=epochs,
        id_type=id_type,
    )

    eph = obj.ephemerides()

    # Extract columns — Horizons returns RA/DEC in degrees, rates in
    # arcsec/hr, and V magnitude when available
    df = pd.DataFrame(
        {
            "datetime_jd": np.array(eph["datetime_jd"], dtype=np.float64),
            "ra_deg": np.array(eph["RA"], dtype=np.float64),
            "dec_deg": np.array(eph["DEC"], dtype=np.float64),
            "V_mag": np.array(
                eph["V"] if "V" in eph.colnames else eph.get("Tmag", [np.nan] * len(eph)),
                dtype=np.float64,
            ),
            "delta_au": np.array(eph["delta"], dtype=np.float64),
            "r_au": np.array(eph["r"], dtype=np.float64),
            "ra_rate_arcsec_hr": np.array(
                eph["RA_rate"], dtype=np.float64,
            ),
            "dec_rate_arcsec_hr": np.array(
                eph["DEC_rate"], dtype=np.float64,
            ),
        }
    )
    return df


def fetch_sedna_position(epoch_jd: float) -> dict:
    """
    Get Sedna's sky position at a single epoch.

    Convenience wrapper around fetch_sky_ephemeris for Sedna (90377).

    Parameters
    ----------
    epoch_jd : float
        Julian date of the observation epoch.

    Returns
    -------
    dict
        Keys: ra_deg, dec_deg, V_mag, delta_au, r_au,
        ra_rate_arcsec_hr, dec_rate_arcsec_hr
    """
    df = fetch_sky_ephemeris(SEDNA_ID, epochs=[epoch_jd])
    row = df.iloc[0]
    return {
        "ra_deg": float(row["ra_deg"]),
        "dec_deg": float(row["dec_deg"]),
        "V_mag": float(row["V_mag"]),
        "delta_au": float(row["delta_au"]),
        "r_au": float(row["r_au"]),
        "ra_rate_arcsec_hr": float(row["ra_rate_arcsec_hr"]),
        "dec_rate_arcsec_hr": float(row["dec_rate_arcsec_hr"]),
    }


def compute_inter_epoch_motion(
    body_id: str | int,
    epoch1_jd: float,
    epoch2_jd: float,
    pixel_scale_arcsec: float = 1.7,
) -> dict:
    """
    Compute a body's sky motion between two epochs.

    Parameters
    ----------
    body_id : str or int
        JPL body identifier (e.g. "90377" for Sedna).
    epoch1_jd, epoch2_jd : float
        Julian dates of the two epochs.
    pixel_scale_arcsec : float
        Pixel scale for computing motion in pixels.

    Returns
    -------
    dict
        Keys: delta_ra_arcsec, delta_dec_arcsec, separation_arcsec,
        separation_pixels, pa_deg (position angle, N through E)
    """
    df = fetch_sky_ephemeris(body_id, epochs=[epoch1_jd, epoch2_jd])

    ra1, dec1 = df.iloc[0]["ra_deg"], df.iloc[0]["dec_deg"]
    ra2, dec2 = df.iloc[1]["ra_deg"], df.iloc[1]["dec_deg"]

    # Angular separation in arcseconds
    delta_ra_arcsec = (ra2 - ra1) * 3600.0 * np.cos(np.radians((dec1 + dec2) / 2))
    delta_dec_arcsec = (dec2 - dec1) * 3600.0

    separation_arcsec = np.sqrt(delta_ra_arcsec**2 + delta_dec_arcsec**2)
    separation_pixels = separation_arcsec / pixel_scale_arcsec

    # Position angle (N through E)
    pa_deg = np.degrees(np.arctan2(delta_ra_arcsec, delta_dec_arcsec)) % 360

    return {
        "ra1_deg": float(ra1),
        "dec1_deg": float(dec1),
        "ra2_deg": float(ra2),
        "dec2_deg": float(dec2),
        "delta_ra_arcsec": float(delta_ra_arcsec),
        "delta_dec_arcsec": float(delta_dec_arcsec),
        "separation_arcsec": float(separation_arcsec),
        "separation_pixels": float(separation_pixels),
        "pa_deg": float(pa_deg),
    }


def _infer_id_type(body_id: str | int) -> str | None:
    """
    Infer JPL Horizons id_type from the body identifier.

    - Integer planet IDs (1-9, 199, 299, etc.) → None (default)
    - String numbers like "90377" → "smallbody"
    - Named bodies like "Sedna" → "smallbody"
    """
    if isinstance(body_id, int):
        return None  # Standard planet/satellite ID

    body_str = str(body_id).strip()

    # Major bodies have simple integer IDs < 1000 or specific ranges
    # Small bodies like TNOs have 5+ digit designations
    try:
        num = int(body_str)
        if num >= 10000:
            return "smallbody"
        return None
    except ValueError:
        # Named body (e.g. "Sedna", "2012 VP113")
        return "smallbody"
