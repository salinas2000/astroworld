"""Tests for General Relativity corrections."""

import pytest
import numpy as np
from astroworld.constants import GM_SUN, BODIES, C_LIGHT
from astroworld.engine.relativity import compute_gr_correction
from astroworld.engine.gravity import compute_accelerations, compute_accelerations_with_gr


def test_gr_correction_zero_for_sun():
    """GR correction on the Sun itself should be zero."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.017, 0.0]])

    acc_gr = compute_gr_correction(positions, velocities, GM_SUN, sun_index=0)
    np.testing.assert_array_equal(acc_gr[0], [0.0, 0.0, 0.0])


def test_gr_correction_small_vs_newton():
    """GR correction should be ~1e-8 of Newtonian for Earth at 1 AU."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.0172, 0.0]])
    gm_values = np.array([GM_SUN, BODIES["earth"].gm])

    acc_newton = compute_accelerations(positions, gm_values)
    acc_gr = compute_gr_correction(positions, velocities, GM_SUN, sun_index=0)

    ratio = np.linalg.norm(acc_gr[1]) / np.linalg.norm(acc_newton[1])
    # GR/Newton ~ GM/(c^2 * r) ~ 2.96e-4 / (173^2 * 1) ~ 1e-8
    assert 1e-9 < ratio < 1e-7, f"GR/Newton ratio: {ratio:.2e}"


def test_gr_correction_larger_for_mercury():
    """GR correction should be larger for Mercury (closer to Sun)."""
    # Mercury at ~0.387 AU
    r_mercury = 0.387
    v_mercury = np.sqrt(GM_SUN / r_mercury)

    positions = np.array([[0.0, 0.0, 0.0], [r_mercury, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, v_mercury, 0.0]])

    acc_gr = compute_gr_correction(positions, velocities, GM_SUN, sun_index=0)

    # Also compute for Earth at 1 AU
    positions_e = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v_earth = np.sqrt(GM_SUN / 1.0)
    velocities_e = np.array([[0.0, 0.0, 0.0], [0.0, v_earth, 0.0]])
    acc_gr_e = compute_gr_correction(positions_e, velocities_e, GM_SUN, sun_index=0)

    # Mercury's GR correction should be much larger
    assert np.linalg.norm(acc_gr[1]) > 5 * np.linalg.norm(acc_gr_e[1])


def test_gr_with_newton_wrapper():
    """compute_accelerations_with_gr should return Newton + GR."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.017, 0.0]])
    gm_values = np.array([GM_SUN, BODIES["earth"].gm])

    acc_newton = compute_accelerations(positions, gm_values)
    acc_gr = compute_gr_correction(positions, velocities, GM_SUN, sun_index=0)
    acc_total = compute_accelerations_with_gr(
        positions, velocities, gm_values, GM_SUN, sun_index=0
    )

    np.testing.assert_allclose(acc_total, acc_newton + acc_gr, rtol=1e-14)


def _compute_lrl_precession(result, body_index=1, sun_index=0):
    """
    Compute perihelion precession rate using the Laplace-Runge-Lenz vector.

    The LRL vector always points toward perihelion. In a pure Keplerian
    orbit it's conserved; GR (or numerical errors) cause it to rotate.
    Averaging over each orbit and fitting a line gives the precession rate.
    """
    # Heliocentric positions and velocities
    pos = result.positions[:, body_index, :] - result.positions[:, sun_index, :]
    vel = result.velocities[:, body_index, :] - result.velocities[:, sun_index, :]
    t = result.times

    # Segment into orbits (~880 points per Mercury orbit at dt=0.01, record=10)
    pts_per_orbit = int(87.969 / 0.1)
    n_orbits = len(t) // pts_per_orbit

    omega_per_orbit = []
    t_mid = []
    for k in range(n_orbits):
        i0 = k * pts_per_orbit
        i1 = (k + 1) * pts_per_orbit
        r_seg = pos[i0:i1]
        v_seg = vel[i0:i1]

        # h = r x v (angular momentum)
        h = np.cross(r_seg, v_seg)
        # LRL: e = (v x h) / GM - r_hat
        r_mag = np.linalg.norm(r_seg, axis=1, keepdims=True)
        e_vec = np.cross(v_seg, h) / GM_SUN - r_seg / r_mag

        # Average over the orbit to get a clean direction
        e_mean = np.mean(e_vec, axis=0)
        omega_per_orbit.append(np.arctan2(e_mean[1], e_mean[0]))
        t_mid.append(0.5 * (t[i0] + t[i1 - 1]))

    omega_arr = np.unwrap(np.array(omega_per_orbit))
    t_arr = np.array(t_mid)

    # Linear fit: omega = rate * t + offset
    coeffs = np.polyfit(t_arr, omega_arr, 1)
    return coeffs[0]  # rad/day


@pytest.mark.slow
def test_mercury_perihelion_precession():
    """
    Simulate Mercury for ~240 years and verify GR-induced perihelion
    precession is approximately 43 arcsec/century.

    Uses the Laplace-Runge-Lenz vector to track the perihelion direction
    continuously. Subtracts a Newtonian-only run to isolate the GR
    contribution and cancel numerical integrator artifacts.
    """
    from astroworld.engine.simulation import NBodySimulation

    # Mercury orbital parameters
    a = 0.3871  # AU
    e = 0.2056
    r_peri = a * (1 - e)
    v_peri = np.sqrt(GM_SUN * (2.0 / r_peri - 1.0 / a))  # vis-viva

    positions = np.array([
        [0.0, 0.0, 0.0],
        [r_peri, 0.0, 0.0],
    ])
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, v_peri, 0.0],
    ])

    # Simulate 1000 orbits (~240 years) for statistical precision
    duration = 1000 * 87.969

    # Run WITH GR
    sim_gr = NBodySimulation(body_names=["sun", "mercury"], relativistic=True)
    sim_gr.set_initial_conditions(positions.copy(), velocities.copy())
    result_gr = sim_gr.run(duration_days=duration, dt=0.01, integrator="verlet", record_interval=10)

    # Run WITHOUT GR (same IC)
    sim_nr = NBodySimulation(body_names=["sun", "mercury"], relativistic=False)
    sim_nr.set_initial_conditions(positions.copy(), velocities.copy())
    result_nr = sim_nr.run(duration_days=duration, dt=0.01, integrator="verlet", record_interval=10)

    # Compute precession rates via Runge-Lenz vector
    rate_gr = _compute_lrl_precession(result_gr)
    rate_nr = _compute_lrl_precession(result_nr)

    # GR-only precession = difference (cancels integrator numerical precession)
    gr_only_rad_per_day = rate_gr - rate_nr
    gr_arcsec_per_century = gr_only_rad_per_day * 206265.0 * 36525.0

    # The GR-only precession should be ~43 arcsec/century
    assert 35 < gr_arcsec_per_century < 55, (
        f"Mercury GR precession: {gr_arcsec_per_century:.1f} arcsec/century "
        f"(expected ~43)"
    )
