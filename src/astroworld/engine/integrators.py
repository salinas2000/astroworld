"""
Numerical integrators for the N-body problem.

Two integrators:
  - RK4: High per-step accuracy (4th order), 4 force evaluations/step.
         Not symplectic -- energy drifts over long runs.
  - Velocity Verlet: 2nd-order symplectic. Conserves energy over
         arbitrarily long runs. 1 force evaluation/step.
"""

import numpy as np
from typing import Callable, Tuple

AccelFn = Callable[[np.ndarray], np.ndarray]


def rk4_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    accel_fn: AccelFn,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advance state by one RK4 time step.

    Solves the coupled ODEs:
        dr/dt = v
        dv/dt = a(r)

    Returns (new_positions, new_velocities).
    """
    # Stage 1
    kr1 = velocities
    kv1 = accel_fn(positions)

    # Stage 2
    kr2 = velocities + 0.5 * dt * kv1
    kv2 = accel_fn(positions + 0.5 * dt * kr1)

    # Stage 3
    kr3 = velocities + 0.5 * dt * kv2
    kv3 = accel_fn(positions + 0.5 * dt * kr2)

    # Stage 4
    kr4 = velocities + dt * kv3
    kv4 = accel_fn(positions + dt * kr3)

    new_positions = positions + (dt / 6.0) * (kr1 + 2 * kr2 + 2 * kr3 + kr4)
    new_velocities = velocities + (dt / 6.0) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)

    return new_positions, new_velocities


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    accel_fn: AccelFn,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Advance state by one Velocity Verlet step (symplectic).

    Algorithm:
        r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        a(t+dt) = accel_fn(r(t+dt))
        v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

    Returns (new_positions, new_velocities, new_accelerations).
    The returned accelerations should be passed to the next step.
    """
    new_positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    new_accelerations = accel_fn(new_positions)
    new_velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt

    return new_positions, new_velocities, new_accelerations
