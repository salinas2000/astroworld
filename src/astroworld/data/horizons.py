"""
JPL Horizons data acquisition client.

Downloads heliocentric state vectors (x, y, z, vx, vy, vz) for
Solar System bodies using the astroquery.jplhorizons interface.
All returned data is in AU and AU/day (simulation units).
"""

import pandas as pd
import numpy as np
from astroquery.jplhorizons import Horizons
from typing import List, Optional, Dict

from astroworld.constants import BODIES


def fetch_state_vectors(
    body_id: int,
    epoch_start: str = "2000-01-01",
    epoch_stop: str = "2020-01-01",
    step: str = "1d",
    center: str = "@sun",
) -> pd.DataFrame:
    """
    Download heliocentric state vectors for a body over a time range.

    Returns DataFrame with columns:
        datetime_jd, x, y, z, vx, vy, vz
    Positions in AU, velocities in AU/day.
    """
    obj = Horizons(
        id=body_id,
        location=center,
        epochs={"start": epoch_start, "stop": epoch_stop, "step": step},
    )
    vectors = obj.vectors(refplane="ecliptic")

    df = pd.DataFrame(
        {
            "datetime_jd": np.array(vectors["datetime_jd"], dtype=np.float64),
            "x": np.array(vectors["x"], dtype=np.float64),
            "y": np.array(vectors["y"], dtype=np.float64),
            "z": np.array(vectors["z"], dtype=np.float64),
            "vx": np.array(vectors["vx"], dtype=np.float64),
            "vy": np.array(vectors["vy"], dtype=np.float64),
            "vz": np.array(vectors["vz"], dtype=np.float64),
        }
    )
    return df


def fetch_initial_conditions(
    body_ids: List[int],
    epoch_jd: float = 2451545.0,
    center: str = "@0",  # Solar System Barycenter (inertial frame)
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Get position and velocity vectors at a single epoch for multiple bodies.

    Default center is @0 (Solar System Barycenter), which is an inertial
    frame. This is more accurate for N-body simulation than heliocentric.

    Returns dict mapping body_id -> {"r": array([x,y,z]), "v": array([vx,vy,vz])}
    Units: AU and AU/day.
    """
    result = {}
    for body_id in body_ids:
        obj = Horizons(
            id=body_id,
            location=center,
            epochs=[epoch_jd],
        )
        vectors = obj.vectors(refplane="ecliptic")

        result[body_id] = {
            "r": np.array(
                [float(vectors["x"][0]), float(vectors["y"][0]), float(vectors["z"][0])]
            ),
            "v": np.array(
                [
                    float(vectors["vx"][0]),
                    float(vectors["vy"][0]),
                    float(vectors["vz"][0]),
                ]
            ),
        }
    return result


def fetch_all_planets(
    epoch_start: str = "2000-01-01",
    epoch_stop: str = "2020-01-01",
    step: str = "1d",
    bodies: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download state vectors for multiple planets.

    Parameters
    ----------
    bodies : list of str or None
        Body names from constants.BODIES. None = all planets (excluding Sun).
    """
    if bodies is None:
        bodies = [name for name in BODIES if name != "sun"]

    result = {}
    for name in bodies:
        body = BODIES[name]
        print(f"Downloading {body.name} (ID={body.jpl_id})...")
        result[name] = fetch_state_vectors(
            body_id=body.jpl_id,
            epoch_start=epoch_start,
            epoch_stop=epoch_stop,
            step=step,
        )
        print(f"  -> {len(result[name])} data points")

    return result
