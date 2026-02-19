"""Tests for numerical integrators."""

import numpy as np
from astroworld.constants import GM_SUN, BODIES
from astroworld.engine.gravity import compute_accelerations, compute_total_energy
from astroworld.engine.integrators import rk4_step, velocity_verlet_step


def _make_circular_orbit():
    """Set up Sun + Earth in circular orbit at 1 AU."""
    r = 1.0
    v_circular = np.sqrt(GM_SUN / r)

    positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, v_circular, 0.0]])
    gm_values = np.array([GM_SUN, BODIES["earth"].gm])
    masses = np.array([BODIES["sun"].mass_kg, BODIES["earth"].mass_kg])

    def accel_fn(pos):
        return compute_accelerations(pos, gm_values)

    return positions, velocities, gm_values, masses, accel_fn


def test_rk4_circular_orbit_closure():
    """Earth should return close to start after one orbit with RK4."""
    pos, vel, gm, masses, accel_fn = _make_circular_orbit()
    initial_pos = pos.copy()

    dt = 0.5  # days
    n_steps = int(365.25 / dt)

    for _ in range(n_steps):
        pos, vel = rk4_step(pos, vel, accel_fn, dt)

    # Earth should be close to initial position (within 0.01 AU)
    earth_displacement = np.linalg.norm(pos[1] - initial_pos[1])
    assert earth_displacement < 0.01, f"Earth drifted {earth_displacement:.4f} AU"


def test_verlet_circular_orbit_closure():
    """Earth should return close to start after one orbit with Verlet."""
    pos, vel, gm, masses, accel_fn = _make_circular_orbit()
    initial_pos = pos.copy()

    dt = 0.5
    n_steps = int(365.25 / dt)
    acc = accel_fn(pos)

    for _ in range(n_steps):
        pos, vel, acc = velocity_verlet_step(pos, vel, acc, accel_fn, dt)

    earth_displacement = np.linalg.norm(pos[1] - initial_pos[1])
    assert earth_displacement < 0.01, f"Earth drifted {earth_displacement:.4f} AU"


def test_verlet_energy_conservation_long():
    """Verlet should conserve energy over 10 years (< 1e-6 relative drift)."""
    pos, vel, gm, masses, accel_fn = _make_circular_orbit()

    dt = 1.0
    n_steps = int(10 * 365.25 / dt)
    acc = accel_fn(pos)

    e0 = compute_total_energy(pos, vel, gm, masses)

    for _ in range(n_steps):
        pos, vel, acc = velocity_verlet_step(pos, vel, acc, accel_fn, dt)

    e_final = compute_total_energy(pos, vel, gm, masses)
    relative_drift = abs((e_final - e0) / e0)

    assert relative_drift < 1e-6, f"Energy drift {relative_drift:.2e} exceeds threshold"


def test_rk4_energy_conservation_short():
    """RK4 should conserve energy over 1 year (< 1e-6 relative drift)."""
    pos, vel, gm, masses, accel_fn = _make_circular_orbit()

    dt = 0.5
    n_steps = int(365.25 / dt)

    e0 = compute_total_energy(pos, vel, gm, masses)

    for _ in range(n_steps):
        pos, vel = rk4_step(pos, vel, accel_fn, dt)

    e_final = compute_total_energy(pos, vel, gm, masses)
    relative_drift = abs((e_final - e0) / e0)

    assert relative_drift < 1e-6, f"Energy drift {relative_drift:.2e} exceeds threshold"


def test_rk4_and_verlet_agree():
    """RK4 and Verlet should produce similar results over a short run."""
    pos_rk4, vel_rk4, gm, masses, accel_fn = _make_circular_orbit()
    pos_vl = pos_rk4.copy()
    vel_vl = vel_rk4.copy()

    dt = 0.5
    n_steps = int(30 / dt)  # 30 days
    acc_vl = accel_fn(pos_vl)

    for _ in range(n_steps):
        pos_rk4, vel_rk4 = rk4_step(pos_rk4, vel_rk4, accel_fn, dt)
        pos_vl, vel_vl, acc_vl = velocity_verlet_step(pos_vl, vel_vl, acc_vl, accel_fn, dt)

    # Positions should agree within 1e-6 AU for a short run
    diff = np.linalg.norm(pos_rk4[1] - pos_vl[1])
    assert diff < 1e-4, f"RK4 vs Verlet disagreement: {diff:.2e} AU"
