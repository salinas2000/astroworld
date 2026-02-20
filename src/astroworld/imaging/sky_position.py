"""
Planet 9 sky position prediction.

Converts Planet 9 Keplerian orbital elements to observable sky coordinates
(RA, Dec) using astropy coordinate transformations. Also estimates apparent
magnitude and angular velocity for survey planning.

The ecliptic Cartesian positions from keplerian_to_cartesian() are in the
heliocentric ecliptic frame (J2000). We transform to ICRS (equatorial) via
astropy to obtain RA/Dec.

Reference frame chain:
  Keplerian elements → ecliptic (x,y,z) → HeliocentricMeanEcliptic → ICRS
"""

import numpy as np
import pandas as pd
from astropy.coordinates import (
    SkyCoord,
    HeliocentricMeanEcliptic,
)
import astropy.units as u

from astroworld.data.keplerian import keplerian_to_cartesian
from astroworld.catalogs.outer_solar_system import GM_STAR
from astroworld.constants import AU, DAY


# --- Default best-fit P9 parameters (from grid search + phase scan) ---
DEFAULT_P9 = dict(
    mass_earth=10.0,
    distance_au=400.0,
    eccentricity=0.20,
    inclination_deg=16.0,
    omega_node_deg=100.0,
    omega_peri_deg=150.0,
    mean_anomaly_deg=180.0,
)

# Absolute magnitude estimate for a Neptune-class ice giant
# H ~ 3-5 for a 10 M_Earth body at T_eff ~ 40-50 K
# Using Linder & Mordasini (2016) models for cold super-Earths
_H_ABSOLUTE = 4.0  # visual absolute magnitude (conservative)

# J2000 epoch in Julian Date
_J2000_JD = 2451545.0


def predict_p9_sky_position(
    epoch_jd: float = _J2000_JD,
    mass_earth: float = DEFAULT_P9["mass_earth"],
    distance_au: float = DEFAULT_P9["distance_au"],
    eccentricity: float = DEFAULT_P9["eccentricity"],
    inclination_deg: float = DEFAULT_P9["inclination_deg"],
    omega_node_deg: float = DEFAULT_P9["omega_node_deg"],
    omega_peri_deg: float = DEFAULT_P9["omega_peri_deg"],
    mean_anomaly_deg: float = DEFAULT_P9["mean_anomaly_deg"],
) -> dict:
    """
    Predict Planet 9's sky position at a given epoch.

    Parameters
    ----------
    epoch_jd : Julian Date of observation (default: J2000.0)
    mass_earth : Planet 9 mass in Earth masses (for magnitude only)
    distance_au : Semi-major axis in AU
    eccentricity : Orbital eccentricity
    inclination_deg : Inclination in degrees
    omega_node_deg : Longitude of ascending node in degrees
    omega_peri_deg : Argument of perihelion in degrees
    mean_anomaly_deg : Mean anomaly at J2000 in degrees

    Returns
    -------
    dict with keys:
        ra_deg : Right ascension (degrees, ICRS)
        dec_deg : Declination (degrees, ICRS)
        distance_au : Heliocentric distance (AU)
        distance_geo_au : Geocentric distance (AU, approx = heliocentric)
        v_angular_arcsec_hr : Angular velocity (arcsec/hour)
        mag_estimate : Estimated apparent magnitude
        x_ecl, y_ecl, z_ecl : Ecliptic Cartesian position (AU)
    """
    # Propagate mean anomaly from J2000 to epoch
    dt_days = epoch_jd - _J2000_JD
    n = np.sqrt(GM_STAR / distance_au**3)  # mean motion (rad/day)
    M_epoch = np.radians(mean_anomaly_deg) + n * dt_days

    # Convert orbital elements to ecliptic Cartesian
    pos_ecl, vel_ecl = keplerian_to_cartesian(
        a=distance_au,
        e=eccentricity,
        i=np.radians(inclination_deg),
        omega_node=np.radians(omega_node_deg),
        omega_peri=np.radians(omega_peri_deg),
        mean_anomaly=M_epoch,
        gm_star=GM_STAR,
    )

    # Heliocentric distance
    r_helio = np.linalg.norm(pos_ecl)

    # Transform ecliptic → equatorial (ICRS) via astropy
    # keplerian_to_cartesian returns heliocentric ecliptic J2000
    coord_ecl = SkyCoord(
        x=pos_ecl[0] * u.AU,
        y=pos_ecl[1] * u.AU,
        z=pos_ecl[2] * u.AU,
        frame=HeliocentricMeanEcliptic,
        representation_type="cartesian",
    )
    coord_icrs = coord_ecl.icrs

    ra_deg = coord_icrs.ra.deg
    dec_deg = coord_icrs.dec.deg

    # Geocentric distance (for magnitude)
    # At 400+ AU, heliocentric ~= geocentric (Earth is 1 AU from Sun)
    delta = r_helio  # Simplified; exact would subtract Earth position

    # Apparent magnitude: m = H + 5*log10(r * delta)
    # where r = heliocentric dist, delta = geocentric dist (both in AU)
    mag_estimate = _H_ABSOLUTE + 5.0 * np.log10(r_helio * delta)

    # Angular velocity on sky (arcsec/hour)
    # Project velocity onto sky plane perpendicular to line of sight
    r_hat = pos_ecl / r_helio
    v_sky = vel_ecl - np.dot(vel_ecl, r_hat) * r_hat  # tangential component
    v_sky_mag = np.linalg.norm(v_sky)  # AU/day

    # Convert to arcsec/hour
    # angular_velocity = v_sky / r  (rad/day)
    omega_rad_day = v_sky_mag / r_helio
    omega_arcsec_hr = np.degrees(omega_rad_day) * 3600.0 / 24.0

    return {
        "ra_deg": float(ra_deg),
        "dec_deg": float(dec_deg),
        "distance_au": float(r_helio),
        "distance_geo_au": float(delta),
        "v_angular_arcsec_hr": float(omega_arcsec_hr),
        "mag_estimate": float(mag_estimate),
        "x_ecl": float(pos_ecl[0]),
        "y_ecl": float(pos_ecl[1]),
        "z_ecl": float(pos_ecl[2]),
    }


def predict_p9_track(
    start_jd: float = _J2000_JD,
    end_jd: float = _J2000_JD + 365.25 * 50,  # 50 years from J2000
    n_points: int = 100,
    **p9_params,
) -> pd.DataFrame:
    """
    Predict Planet 9's sky track over a time range.

    Parameters
    ----------
    start_jd : Start Julian Date
    end_jd : End Julian Date
    n_points : Number of track points
    **p9_params : Passed to predict_p9_sky_position (mass_earth, distance_au, etc.)

    Returns
    -------
    DataFrame with columns:
        jd, year, ra_deg, dec_deg, distance_au, v_angular_arcsec_hr, mag_estimate
    """
    # Merge with defaults
    params = {**DEFAULT_P9, **p9_params}

    epochs = np.linspace(start_jd, end_jd, n_points)
    rows = []

    for jd in epochs:
        result = predict_p9_sky_position(epoch_jd=jd, **params)
        result["jd"] = jd
        result["year"] = 2000.0 + (jd - _J2000_JD) / 365.25
        rows.append(result)

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["jd", "year", "ra_deg", "dec_deg", "distance_au",
            "v_angular_arcsec_hr", "mag_estimate",
            "x_ecl", "y_ecl", "z_ecl", "distance_geo_au"]
    return df[cols]


def get_p9_search_region(
    epoch_jd: float = _J2000_JD + 365.25 * 25,  # ~2025
    margin_deg: float = 5.0,
    **p9_params,
) -> dict:
    """
    Get the recommended search region for Planet 9 surveys.

    Returns the predicted position plus a margin to account for
    orbital uncertainty.

    Parameters
    ----------
    epoch_jd : Epoch for prediction (default: ~2025.0)
    margin_deg : Search margin around predicted position (degrees)
    **p9_params : Planet 9 orbital parameters

    Returns
    -------
    dict with keys:
        ra_center, dec_center : Center of search region (degrees)
        ra_min, ra_max, dec_min, dec_max : Bounding box (degrees)
        constellation : Approximate constellation name
        position : Full prediction dict from predict_p9_sky_position
    """
    params = {**DEFAULT_P9, **p9_params}
    pos = predict_p9_sky_position(epoch_jd=epoch_jd, **params)

    ra_c = pos["ra_deg"]
    dec_c = pos["dec_deg"]

    # RA margin needs cos(dec) correction for spherical geometry
    cos_dec = np.cos(np.radians(dec_c))
    ra_margin = margin_deg / max(cos_dec, 0.1)

    return {
        "ra_center": ra_c,
        "dec_center": dec_c,
        "ra_min": ra_c - ra_margin,
        "ra_max": ra_c + ra_margin,
        "dec_min": dec_c - margin_deg,
        "dec_max": dec_c + margin_deg,
        "position": pos,
    }
