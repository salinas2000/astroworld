"""
Post-Newtonian relativistic corrections for N-body dynamics.

Implements the 1PN (first post-Newtonian) correction from General
Relativity, dominated by the Sun's gravitational field. This is the
same approach used by JPL in their DE440 ephemeris integrator.

The correction accounts for:
- Spacetime curvature near the Sun
- Velocity-dependent terms from special relativity
- The famous 43 arcsec/century precession of Mercury's perihelion

Reference: Will, C.M. "Theory and Experiment in Gravitational Physics"
           Eq. 9.1 (isotropic PPN formulation with beta=gamma=1 for GR)
"""

import numpy as np
from astroworld.constants import C_LIGHT, GM_SUN


def compute_gr_correction(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm_sun: float = GM_SUN,
    sun_index: int = 0,
) -> np.ndarray:
    """
    Compute 1PN General Relativity correction to accelerations.

    Uses the Sun-centered post-Newtonian approximation (dominant term).
    For body i with position r_i and velocity v_i relative to the Sun:

        a_GR_i = (GM / (c^2 * r^3)) * {
            [4*GM/r - v^2] * r_vec + 4*(v . r_vec) * v_vec
        }

    This is the isotropic PPN formula with beta=gamma=1 (pure GR).

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Position vectors in AU (barycentric or heliocentric).
    velocities : ndarray, shape (N, 3)
        Velocity vectors in AU/day.
    gm_sun : float
        GM of the Sun in AU^3/day^2.
    sun_index : int
        Index of the Sun in the arrays.

    Returns
    -------
    ndarray, shape (N, 3)
        GR acceleration correction in AU/day^2.
    """
    n = len(positions)
    c2 = C_LIGHT**2
    acc_gr = np.zeros_like(positions)

    pos_sun = positions[sun_index]
    vel_sun = velocities[sun_index]

    for i in range(n):
        if i == sun_index:
            continue

        # Position and velocity relative to Sun
        r_vec = positions[i] - pos_sun
        v_vec = velocities[i] - vel_sun

        r = np.linalg.norm(r_vec)
        v_sq = np.dot(v_vec, v_vec)
        v_dot_r = np.dot(v_vec, r_vec)

        # 1PN correction (beta=gamma=1 for GR)
        # a_GR = (GM / (c^2 * r^3)) * [(4*GM/r - v^2)*r_vec + 4*(v.r_hat)*v_vec]
        # Note: v.r_hat = v.r / r, so 4*(v.r_hat)*v = 4*(v.r/r)*v
        prefactor = gm_sun / (c2 * r**3)

        term1 = (4.0 * gm_sun / r - v_sq) * r_vec
        term2 = 4.0 * v_dot_r * v_vec

        acc_gr[i] = prefactor * (term1 + term2)

    return acc_gr
