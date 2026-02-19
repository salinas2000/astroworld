"""Tests for gravitational acceleration computation."""

import numpy as np
from astroworld.constants import GM_SUN, BODIES
from astroworld.engine.gravity import (
    compute_accelerations,
    compute_total_energy,
    compute_angular_momentum,
)


def test_two_body_acceleration_magnitude():
    """Sun + Earth at 1 AU: acceleration on Earth should match GM_sun / r^2."""
    positions = np.array([
        [0.0, 0.0, 0.0],  # Sun
        [1.0, 0.0, 0.0],  # Earth at 1 AU
    ])
    gm_values = np.array([GM_SUN, BODIES["earth"].gm])

    acc = compute_accelerations(positions, gm_values)

    # Acceleration on Earth (index 1) should point toward Sun (negative x)
    expected_mag = GM_SUN / 1.0**2  # GM_sun / r^2
    actual_mag = np.linalg.norm(acc[1])

    assert abs(actual_mag - expected_mag) / expected_mag < 1e-10
    assert acc[1, 0] < 0  # Points toward Sun (negative x direction)


def test_acceleration_antisymmetry():
    """Acceleration vectors on two bodies should point toward each other."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    gm_values = np.array([GM_SUN, BODIES["earth"].gm])

    acc = compute_accelerations(positions, gm_values)

    # Body 0 (Sun) should be accelerated in +x (toward Earth)
    assert acc[0, 0] > 0
    # Body 1 (Earth) should be accelerated in -x (toward Sun)
    assert acc[1, 0] < 0

    # Magnitudes: a_sun = GM_earth / r^2, a_earth = GM_sun / r^2
    r = 2.0
    expected_sun = BODIES["earth"].gm / r**2
    expected_earth = GM_SUN / r**2
    np.testing.assert_allclose(np.linalg.norm(acc[0]), expected_sun, rtol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(acc[1]), expected_earth, rtol=1e-10)


def test_three_body_no_nan():
    """Three-body computation should produce no NaN or Inf values."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.5, 0.0],
    ])
    gm_values = np.array([GM_SUN, BODIES["earth"].gm, BODIES["mars"].gm])

    acc = compute_accelerations(positions, gm_values)
    assert not np.any(np.isnan(acc))
    assert not np.any(np.isinf(acc))


def test_circular_orbit_energy():
    """For a circular orbit, E = -GM*m / (2*a), where a = radius."""
    r = 1.0  # 1 AU
    v_circular = np.sqrt(GM_SUN / r)  # Circular velocity

    positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, v_circular, 0.0]])
    gm_values = np.array([GM_SUN, BODIES["earth"].gm])
    masses = np.array([BODIES["sun"].mass_kg, BODIES["earth"].mass_kg])

    energy = compute_total_energy(positions, velocities, gm_values, masses)

    # Total energy should be negative (bound orbit)
    assert energy < 0
