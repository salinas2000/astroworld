"""Tests for the simulation orchestrator (offline, no Horizons)."""

import numpy as np
from astroworld.constants import GM_SUN, BODIES
from astroworld.engine.simulation import NBodySimulation


def _setup_sim_with_circular_orbits():
    """Create a simulation with manual circular orbit initial conditions."""
    sim = NBodySimulation(body_names=["sun", "earth", "mars"])

    # Sun at origin, not moving
    # Earth at 1 AU, circular velocity
    # Mars at 1.524 AU, circular velocity
    r_earth = 1.0
    r_mars = 1.524
    v_earth = np.sqrt(GM_SUN / r_earth)
    v_mars = np.sqrt(GM_SUN / r_mars)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [r_earth, 0.0, 0.0],
        [r_mars, 0.0, 0.0],
    ])
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, v_earth, 0.0],
        [0.0, v_mars, 0.0],
    ])

    sim.set_initial_conditions(positions, velocities)
    return sim


def test_simulation_no_nan():
    """Simulation should produce no NaN or Inf values."""
    sim = _setup_sim_with_circular_orbits()
    result = sim.run(duration_days=365.25, dt=1.0, integrator="verlet")

    assert not np.any(np.isnan(result.positions))
    assert not np.any(np.isinf(result.positions))
    assert not np.any(np.isnan(result.velocities))
    assert not np.any(np.isnan(result.energies))


def test_simulation_earth_stays_bound():
    """Earth should stay within 2 AU of the Sun over 1 year."""
    sim = _setup_sim_with_circular_orbits()
    result = sim.run(duration_days=365.25, dt=1.0, integrator="verlet")

    earth_distances = np.linalg.norm(result.positions[:, 1, :], axis=1)
    assert np.all(earth_distances < 2.0), "Earth escaped!"
    assert np.all(earth_distances > 0.5), "Earth fell into the Sun!"


def test_simulation_to_dataframe():
    """to_dataframe should return correct shape and columns."""
    sim = _setup_sim_with_circular_orbits()
    result = sim.run(duration_days=30, dt=1.0, integrator="verlet")

    df = result.to_dataframe("earth")
    assert "time_days" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns
    assert "z" in df.columns
    assert len(df) == len(result.times)


def test_simulation_energy_conservation():
    """Energy should be approximately conserved with Verlet."""
    sim = _setup_sim_with_circular_orbits()
    result = sim.run(duration_days=365.25 * 2, dt=0.5, integrator="verlet")

    e0 = result.energies[0]
    relative_drift = np.max(np.abs((result.energies - e0) / e0))
    assert relative_drift < 1e-5, f"Max energy drift: {relative_drift:.2e}"


def test_simulation_both_integrators_run():
    """Both RK4 and Verlet should complete without errors."""
    for integrator in ["rk4", "verlet"]:
        sim = _setup_sim_with_circular_orbits()
        result = sim.run(duration_days=30, dt=1.0, integrator=integrator)
        assert len(result.times) > 1
