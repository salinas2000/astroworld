"""
Newtonian gravitational acceleration computation.

Computes pairwise gravitational forces/accelerations for N bodies
using vectorized NumPy operations.

Units: AU, AU/day, AU/day^2
"""

import numpy as np

from astroworld.engine.relativity import compute_gr_correction
from astroworld.engine.kernels import (
    HAS_NUMBA,
    _compute_accelerations_jit,
    _compute_total_energy_jit,
)


def compute_accelerations(
    positions: np.ndarray,
    gm_values: np.ndarray,
    softening: float = 0.0,
) -> np.ndarray:
    """
    Compute gravitational accelerations on all bodies.

    a_i = sum_{j != i} GM_j * (r_j - r_i) / |r_j - r_i|^3

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Position vectors in AU.
    gm_values : ndarray, shape (N,)
        GM values in AU^3/day^2.
    softening : float
        Softening length in AU to prevent singularity at r=0.

    Returns
    -------
    ndarray, shape (N, 3)
        Acceleration vectors in AU/day^2.
    """
    n = len(positions)

    # Displacement tensor: dx[i,j] = positions[j] - positions[i]
    dx = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N, N, 3)

    # Squared distances: r_sq[i,j] = |r_j - r_i|^2
    r_sq = np.sum(dx**2, axis=2) + softening**2  # (N, N)

    # Avoid division by zero on diagonal (self-interaction)
    np.fill_diagonal(r_sq, 1.0)

    # 1 / |r|^3
    inv_r3 = r_sq ** (-1.5)  # (N, N)
    np.fill_diagonal(inv_r3, 0.0)  # No self-interaction

    # a_i = sum_j GM_j * dx[i,j] / |dx[i,j]|^3
    acc = np.einsum("ij,ijk,j->ik", inv_r3, dx, gm_values)  # (N, 3)

    return acc


def compute_accelerations_with_gr(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm_values: np.ndarray,
    gm_sun: float,
    sun_index: int = 0,
) -> np.ndarray:
    """
    Compute Newtonian accelerations + 1PN General Relativity correction.

    Parameters
    ----------
    positions : ndarray, shape (N, 3) in AU.
    velocities : ndarray, shape (N, 3) in AU/day.
    gm_values : ndarray, shape (N,) in AU^3/day^2.
    gm_sun : float, GM of the Sun.
    sun_index : int, index of the Sun in the arrays.

    Returns ndarray, shape (N, 3) in AU/day^2.
    """
    acc_newton = compute_accelerations(positions, gm_values)
    acc_gr = compute_gr_correction(positions, velocities, gm_sun, sun_index)
    return acc_newton + acc_gr


def compute_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm_values: np.ndarray,
    masses: np.ndarray,
) -> float:
    """
    Compute total mechanical energy (kinetic + potential).

    KE = 0.5 * sum(m_i * |v_i|^2)
    PE = -sum_{i<j} GM_i * m_j / |r_j - r_i|
       = -sum_{i<j} G * m_i * m_j / |r_j - r_i|

    Since GM_i = G * m_i, we use: PE = -sum_{i<j} gm_i * m_j / |r_ij|
    """
    # Kinetic energy
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)

    # Potential energy (upper triangle only to avoid double counting)
    n = len(positions)
    pe = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[j] - positions[i])
            pe -= gm_values[i] * masses[j] / r

    return ke + pe


def compute_accelerations_fast(
    positions: np.ndarray,
    gm_values: np.ndarray,
    softening: float = 0.0,
) -> np.ndarray:
    """
    JIT-accelerated compute_accelerations (when Numba is available).

    Falls back to the NumPy vectorized version otherwise.
    The JIT path does not support softening (always 0.0).
    """
    if HAS_NUMBA and softening == 0.0:
        return _compute_accelerations_jit(positions, gm_values, len(positions))
    return compute_accelerations(positions, gm_values, softening)


def compute_total_energy_fast(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm_values: np.ndarray,
    masses: np.ndarray,
) -> float:
    """JIT-accelerated compute_total_energy (when Numba is available)."""
    if HAS_NUMBA:
        return _compute_total_energy_jit(positions, velocities, gm_values, masses, len(positions))
    return compute_total_energy(positions, velocities, gm_values, masses)


def compute_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Compute total angular momentum vector L = sum(m_i * r_i x v_i).

    Returns ndarray, shape (3,).
    """
    cross = np.cross(positions, velocities)  # (N, 3)
    return np.sum(masses[:, np.newaxis] * cross, axis=0)
