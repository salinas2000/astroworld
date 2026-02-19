"""
Instantaneous Keplerian orbital element extraction from Cartesian state vectors.

Shared utility used by run_scenario.py, run_p9_investigations.py, and
any analysis that needs osculating orbital elements from simulation output.

Units: AU, AU/day, degrees for angles.
"""

import numpy as np

from astroworld.engine.simulation import SimulationResult
from astroworld.data.state import SystemState


def compute_orbital_elements(
    pos: np.ndarray,
    vel: np.ndarray,
    gm_central: float,
) -> dict:
    """
    Compute instantaneous Keplerian elements from Cartesian state.

    Parameters
    ----------
    pos : ndarray (3,) — position relative to central body (AU).
    vel : ndarray (3,) — velocity relative to central body (AU/day).
    gm_central : float — GM of the central body (AU^3/day^2).

    Returns
    -------
    dict with keys:
        a : float — semi-major axis (AU)
        e : float — eccentricity
        i : float — inclination (degrees)
        omega_node : float — longitude of ascending node Ω (degrees)
        omega_peri : float — argument of perihelion ω (degrees)
        longitude_peri : float — longitude of perihelion ϖ = Ω + ω (degrees)
    """
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)

    # Specific angular momentum h = r × v
    h_vec = np.cross(pos, vel)
    h = np.linalg.norm(h_vec)

    # Node vector: n = z_hat × h
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector: e_vec = (v × h)/GM - r_hat
    e_vec = np.cross(vel, h_vec) / gm_central - pos / r
    e = np.linalg.norm(e_vec)

    # Semi-major axis (vis-viva)
    energy = 0.5 * v**2 - gm_central / r
    if abs(energy) > 1e-30:
        a = -gm_central / (2 * energy)
    else:
        a = float("inf")

    # Inclination
    i = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1, 1)))

    # Longitude of ascending node
    if n > 1e-30:
        Omega = np.degrees(np.arctan2(n_vec[1], n_vec[0])) % 360
    else:
        Omega = 0.0

    # Argument of perihelion
    if n > 1e-30 and e > 1e-10:
        cos_omega = np.dot(n_vec, e_vec) / (n * e)
        omega = np.degrees(np.arccos(np.clip(cos_omega, -1, 1)))
        if e_vec[2] < 0:
            omega = 360 - omega
    else:
        omega = 0.0

    # Longitude of perihelion (ϖ = Ω + ω)
    longitude_peri = (Omega + omega) % 360

    return {
        "a": a, "e": e, "i": i,
        "omega_node": Omega, "omega_peri": omega,
        "longitude_peri": longitude_peri,
    }


def extract_body_elements_timeseries(
    result: SimulationResult,
    state: SystemState,
    body_name: str,
    gm_central: float,
    n_samples: int = 2000,
) -> dict:
    """
    Extract time-series of orbital elements for a single body.

    Parameters
    ----------
    result : SimulationResult from NBodySimulation.run().
    state : SystemState with star_index.
    body_name : Name of the body to extract elements for.
    gm_central : GM of the central body (AU^3/day^2).
    n_samples : Number of evenly-spaced samples to extract (default 2000).

    Returns
    -------
    dict with keys:
        a : ndarray — semi-major axis (AU)
        e : ndarray — eccentricity
        omega_peri : ndarray — argument of perihelion (degrees, unwrapped)
        longitude_peri : ndarray — longitude of perihelion (degrees, unwrapped)
        t_years : ndarray — time in years
    """
    body_idx = result.body_names.index(body_name)
    star_idx = state.star_index

    n_samples = min(n_samples, len(result.times))
    sample_idx = np.unique(
        np.linspace(0, len(result.times) - 1, n_samples, dtype=int)
    )
    t_years = result.times[sample_idx] / 365.25

    a_list, e_list, omega_list, varpi_list = [], [], [], []

    for k in sample_idx:
        pos = result.positions[k, body_idx, :] - result.positions[k, star_idx, :]
        vel = result.velocities[k, body_idx, :] - result.velocities[k, star_idx, :]
        elems = compute_orbital_elements(pos, vel, gm_central)
        a_list.append(elems["a"])
        e_list.append(elems["e"])
        omega_list.append(elems["omega_peri"])
        varpi_list.append(elems["longitude_peri"])

    omega_arr = np.array(omega_list)
    varpi_arr = np.array(varpi_list)

    return {
        "a": np.array(a_list),
        "e": np.array(e_list),
        "omega_peri": np.degrees(np.unwrap(np.radians(omega_arr))),
        "longitude_peri": np.degrees(np.unwrap(np.radians(varpi_arr))),
        "t_years": t_years,
    }
