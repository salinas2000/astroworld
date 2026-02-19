"""
Keplerian orbital elements to Cartesian state vector conversion.

For systems where no ephemeris service exists (exoplanets, hypothetical
systems), initial conditions are specified as Keplerian elements and
converted to position/velocity vectors for the N-body integrator.

Reference: Bate, Mueller & White, "Fundamentals of Astrodynamics" (1971)
"""

import numpy as np


def keplerian_to_cartesian(
    a: float,
    e: float,
    i: float,
    omega_node: float,
    omega_peri: float,
    mean_anomaly: float,
    gm_star: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian orbital elements to Cartesian state vectors.

    Parameters
    ----------
    a : Semi-major axis (AU)
    e : Eccentricity (0 ≤ e < 1 for elliptical orbits)
    i : Inclination (radians)
    omega_node : Longitude of ascending node (radians), Ω
    omega_peri : Argument of perihelion (radians), ω
    mean_anomaly : Mean anomaly at epoch (radians), M
    gm_star : GM of central body (AU³/day²)

    Returns
    -------
    pos : (3,) position vector in AU [x, y, z]
    vel : (3,) velocity vector in AU/day [vx, vy, vz]
    """
    # Solve Kepler's equation: M = E - e*sin(E) for eccentric anomaly E
    E = _solve_kepler(mean_anomaly, e)

    # True anomaly from eccentric anomaly
    nu = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2),
    )

    # Distance from focus
    r = a * (1 - e * np.cos(E))

    # Position in orbital plane (perifocal frame: P toward perihelion, Q 90° ahead)
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Velocity in orbital plane
    # v = sqrt(GM/p) where p = a(1-e²) is semi-latus rectum
    p = a * (1 - e**2)
    h = np.sqrt(gm_star * p)  # specific angular momentum
    vx_orb = -gm_star / h * np.sin(nu)
    vy_orb = gm_star / h * (e + np.cos(nu))

    # Rotation matrices: orbital plane → inertial frame
    # R = Rz(-Ω) · Rx(-i) · Rz(-ω)
    cos_o = np.cos(omega_peri)
    sin_o = np.sin(omega_peri)
    cos_O = np.cos(omega_node)
    sin_O = np.sin(omega_node)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    # Combined rotation matrix elements (perifocal → inertial)
    Px = cos_O * cos_o - sin_O * sin_o * cos_i
    Py = sin_O * cos_o + cos_O * sin_o * cos_i
    Pz = sin_o * sin_i
    Qx = -cos_O * sin_o - sin_O * cos_o * cos_i
    Qy = -sin_O * sin_o + cos_O * cos_o * cos_i
    Qz = cos_o * sin_i

    # Transform to inertial frame
    pos = np.array([
        Px * x_orb + Qx * y_orb,
        Py * x_orb + Qy * y_orb,
        Pz * x_orb + Qz * y_orb,
    ])

    vel = np.array([
        Px * vx_orb + Qx * vy_orb,
        Py * vx_orb + Qy * vy_orb,
        Pz * vx_orb + Qz * vy_orb,
    ])

    return pos, vel


def _solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.

    Uses Newton-Raphson iteration. Converges in ~3-5 iterations for e < 0.9.
    """
    # Initial guess
    E = M + e * np.sin(M) if e < 0.8 else np.pi

    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break

    return E
