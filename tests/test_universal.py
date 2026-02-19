"""
Tests for the universal architecture: SystemState, Keplerian converter,
TRAPPIST-1 catalog, and the universal simulation mode.

These verify that the new abstraction layer works correctly independently
of the legacy Solar System code path.
"""

import numpy as np
import pytest

from astroworld.data.state import SystemState
from astroworld.data.keplerian import keplerian_to_cartesian, _solve_kepler
from astroworld.constants import GM_SUN
from astroworld.engine.simulation import NBodySimulation


# ═══════════════════════════════════════════════════════════════════
# SystemState tests
# ═══════════════════════════════════════════════════════════════════

class TestSystemState:

    def test_basic_creation(self):
        """SystemState can be created with valid inputs."""
        state = SystemState(
            names=["star", "planet"],
            gm_values=np.array([1.0, 0.001]),
            positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
            velocities=np.array([[0, 0, 0], [0, 0.01, 0]], dtype=float),
        )
        assert state.n == 2
        assert state.names == ["star", "planet"]
        assert state.star_index == 0

    def test_n_property(self):
        """The n property returns the number of bodies."""
        state = SystemState(
            names=["a", "b", "c"],
            gm_values=np.ones(3),
            positions=np.zeros((3, 3)),
            velocities=np.zeros((3, 3)),
        )
        assert state.n == 3

    def test_shape_validation_gm(self):
        """SystemState rejects mismatched gm_values shape."""
        with pytest.raises(AssertionError, match="gm_values"):
            SystemState(
                names=["a", "b"],
                gm_values=np.ones(3),  # Wrong: 3 instead of 2
                positions=np.zeros((2, 3)),
                velocities=np.zeros((2, 3)),
            )

    def test_shape_validation_positions(self):
        """SystemState rejects mismatched positions shape."""
        with pytest.raises(AssertionError, match="positions"):
            SystemState(
                names=["a", "b"],
                gm_values=np.ones(2),
                positions=np.zeros((3, 3)),  # Wrong: 3 instead of 2
                velocities=np.zeros((2, 3)),
            )

    def test_shape_validation_velocities(self):
        """SystemState rejects mismatched velocities shape."""
        with pytest.raises(AssertionError, match="velocities"):
            SystemState(
                names=["a", "b"],
                gm_values=np.ones(2),
                positions=np.zeros((2, 3)),
                velocities=np.zeros((2, 2)),  # Wrong: (2,2) instead of (2,3)
            )

    def test_shape_validation_masses(self):
        """SystemState rejects mismatched masses_kg shape."""
        with pytest.raises(AssertionError, match="masses_kg"):
            SystemState(
                names=["a", "b"],
                gm_values=np.ones(2),
                positions=np.zeros((2, 3)),
                velocities=np.zeros((2, 3)),
                masses_kg=np.ones(3),  # Wrong: 3 instead of 2
            )

    def test_optional_fields(self):
        """masses_kg and colors are optional."""
        state = SystemState(
            names=["star"],
            gm_values=np.array([1.0]),
            positions=np.zeros((1, 3)),
            velocities=np.zeros((1, 3)),
        )
        assert state.masses_kg is None
        assert state.colors is None

    def test_star_index_custom(self):
        """star_index can be set to a non-zero value."""
        state = SystemState(
            names=["planet", "star"],
            gm_values=np.array([0.001, 1.0]),
            positions=np.zeros((2, 3)),
            velocities=np.zeros((2, 3)),
            star_index=1,
        )
        assert state.star_index == 1


# ═══════════════════════════════════════════════════════════════════
# Kepler equation solver tests
# ═══════════════════════════════════════════════════════════════════

class TestKeplerSolver:

    def test_circular_orbit(self):
        """For e=0, eccentric anomaly E = mean anomaly M."""
        for M in [0.0, 0.5, 1.0, np.pi, 2 * np.pi - 0.1]:
            E = _solve_kepler(M, e=0.0)
            assert abs(E - M) < 1e-10, f"Failed for M={M}"

    def test_known_solution(self):
        """Test against known solution: M = E - e*sin(E)."""
        # For E=1.0, e=0.5: M = 1.0 - 0.5*sin(1.0) = 0.57943...
        E_true = 1.0
        e = 0.5
        M = E_true - e * np.sin(E_true)
        E_solved = _solve_kepler(M, e)
        assert abs(E_solved - E_true) < 1e-10

    def test_high_eccentricity(self):
        """Solver works for high eccentricity (e=0.9)."""
        E_true = 2.0
        e = 0.9
        M = E_true - e * np.sin(E_true)
        E_solved = _solve_kepler(M, e)
        assert abs(E_solved - E_true) < 1e-10

    def test_zero_mean_anomaly(self):
        """M=0 gives E=0 for any eccentricity."""
        for e in [0.0, 0.1, 0.5, 0.9]:
            E = _solve_kepler(0.0, e)
            assert abs(E) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# Keplerian-to-Cartesian conversion tests
# ═══════════════════════════════════════════════════════════════════

class TestKeplerianToCartesian:

    def test_circular_orbit_distance(self):
        """Circular orbit at a=1 AU should have |r| = 1 AU."""
        pos, vel = keplerian_to_cartesian(
            a=1.0, e=0.0, i=0.0,
            omega_node=0.0, omega_peri=0.0,
            mean_anomaly=0.0, gm_star=GM_SUN,
        )
        r = np.linalg.norm(pos)
        assert abs(r - 1.0) < 1e-10, f"|r| = {r}, expected 1.0"

    def test_circular_orbit_speed(self):
        """Circular orbit speed should be sqrt(GM/a)."""
        a = 1.0
        pos, vel = keplerian_to_cartesian(
            a=a, e=0.0, i=0.0,
            omega_node=0.0, omega_peri=0.0,
            mean_anomaly=0.0, gm_star=GM_SUN,
        )
        v = np.linalg.norm(vel)
        v_expected = np.sqrt(GM_SUN / a)
        assert abs(v - v_expected) / v_expected < 1e-10

    def test_perihelion_at_M0(self):
        """At M=0 (perihelion), distance should be a(1-e)."""
        a, e = 1.0, 0.2
        pos, vel = keplerian_to_cartesian(
            a=a, e=e, i=0.0,
            omega_node=0.0, omega_peri=0.0,
            mean_anomaly=0.0, gm_star=GM_SUN,
        )
        r = np.linalg.norm(pos)
        r_peri = a * (1 - e)
        assert abs(r - r_peri) < 1e-10, f"|r| = {r}, expected r_peri = {r_peri}"

    def test_velocity_perpendicular_at_perihelion(self):
        """At perihelion, velocity should be perpendicular to position."""
        pos, vel = keplerian_to_cartesian(
            a=1.0, e=0.3, i=0.0,
            omega_node=0.0, omega_peri=0.0,
            mean_anomaly=0.0, gm_star=GM_SUN,
        )
        # Dot product should be zero (perpendicular)
        dot = np.dot(pos, vel)
        assert abs(dot) < 1e-10, f"dot(r, v) = {dot}, expected 0"

    def test_coplanar_orbit_z_zero(self):
        """Coplanar orbit (i=0) should have z=0."""
        for M in [0.0, 1.0, 3.0, 5.5]:
            pos, vel = keplerian_to_cartesian(
                a=1.0, e=0.1, i=0.0,
                omega_node=0.0, omega_peri=0.0,
                mean_anomaly=M, gm_star=GM_SUN,
            )
            assert abs(pos[2]) < 1e-14, f"z = {pos[2]} for M={M}"
            assert abs(vel[2]) < 1e-14

    def test_inclined_orbit_has_z(self):
        """Inclined orbit should have non-zero z component."""
        pos, vel = keplerian_to_cartesian(
            a=1.0, e=0.1, i=np.radians(30),
            omega_node=0.0, omega_peri=0.0,
            mean_anomaly=1.0, gm_star=GM_SUN,
        )
        assert abs(pos[2]) > 0.01, "Inclined orbit should have z != 0"

    def test_vis_viva_energy(self):
        """Verify vis-viva equation: v^2 = GM(2/r - 1/a)."""
        a, e = 2.0, 0.4
        for M in [0.0, 1.0, 2.5, 4.0]:
            pos, vel = keplerian_to_cartesian(
                a=a, e=e, i=0.1,
                omega_node=0.3, omega_peri=0.7,
                mean_anomaly=M, gm_star=GM_SUN,
            )
            r = np.linalg.norm(pos)
            v2 = np.dot(vel, vel)
            v2_visviva = GM_SUN * (2.0 / r - 1.0 / a)
            assert abs(v2 - v2_visviva) / v2_visviva < 1e-10, \
                f"vis-viva mismatch at M={M}: {v2} vs {v2_visviva}"

    def test_angular_momentum_conservation(self):
        """All points on the same orbit should have the same |L|."""
        a, e = 1.5, 0.3
        h_values = []
        for M in np.linspace(0, 2 * np.pi, 20, endpoint=False):
            pos, vel = keplerian_to_cartesian(
                a=a, e=e, i=0.2,
                omega_node=0.5, omega_peri=1.0,
                mean_anomaly=M, gm_star=GM_SUN,
            )
            L = np.cross(pos, vel)
            h_values.append(np.linalg.norm(L))

        h_arr = np.array(h_values)
        # All |L| should be equal (h = sqrt(GM * a * (1-e^2)))
        assert np.std(h_arr) / np.mean(h_arr) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# TRAPPIST-1 catalog tests
# ═══════════════════════════════════════════════════════════════════

class TestTrappist1:

    def test_state_creation(self):
        """make_trappist1_state creates a valid SystemState."""
        from astroworld.catalogs.trappist_1 import make_trappist1_state
        state = make_trappist1_state()
        assert state.n == 8  # star + 7 planets
        assert state.names[0] == "trappist_1"
        assert state.star_index == 0

    def test_state_subset(self):
        """Can create state with a subset of planets."""
        from astroworld.catalogs.trappist_1 import make_trappist1_state
        state = make_trappist1_state(include_planets=["b", "c", "d"])
        assert state.n == 4  # star + 3 planets
        assert "trappist_1_b" in state.names
        assert "trappist_1_d" in state.names
        assert "trappist_1_e" not in state.names

    def test_star_at_origin(self):
        """Star should be at origin with zero velocity."""
        from astroworld.catalogs.trappist_1 import make_trappist1_state
        state = make_trappist1_state()
        np.testing.assert_array_equal(state.positions[0], [0, 0, 0])
        np.testing.assert_array_equal(state.velocities[0], [0, 0, 0])

    def test_planet_distances_reasonable(self):
        """Planet distances should be in range 0.01 - 0.07 AU."""
        from astroworld.catalogs.trappist_1 import make_trappist1_state
        state = make_trappist1_state()
        for i in range(1, state.n):
            r = np.linalg.norm(state.positions[i])
            assert 0.008 < r < 0.08, \
                f"{state.names[i]}: r = {r} AU outside expected range"

    def test_orbital_periods_in_range(self):
        """Theoretical periods should be 1-20 days."""
        from astroworld.catalogs.trappist_1 import get_orbital_periods
        periods = get_orbital_periods()
        assert len(periods) == 7
        for letter, T in periods.items():
            assert 1.0 < T < 20.0, f"Planet {letter}: T = {T} days outside range"

    def test_periods_ordered(self):
        """Orbital periods should increase b < c < d < e < f < g < h."""
        from astroworld.catalogs.trappist_1 import get_orbital_periods
        periods = get_orbital_periods()
        letters = ["b", "c", "d", "e", "f", "g", "h"]
        for i in range(len(letters) - 1):
            assert periods[letters[i]] < periods[letters[i + 1]], \
                f"T({letters[i]}) >= T({letters[i+1]})"

    def test_gm_values_positive(self):
        """All GM values should be positive."""
        from astroworld.catalogs.trappist_1 import make_trappist1_state
        state = make_trappist1_state()
        assert np.all(state.gm_values > 0)

    def test_colors_provided(self):
        """Colors should be provided for all bodies."""
        from astroworld.catalogs.trappist_1 import make_trappist1_state
        state = make_trappist1_state()
        assert state.colors is not None
        assert len(state.colors) == state.n


# ═══════════════════════════════════════════════════════════════════
# Universal simulation mode tests
# ═══════════════════════════════════════════════════════════════════

class TestUniversalSimulation:

    def _make_two_body_state(self):
        """Create a simple Sun + circular-orbit planet SystemState."""
        a = 1.0  # 1 AU
        v_circ = np.sqrt(GM_SUN / a)
        return SystemState(
            names=["star", "planet"],
            gm_values=np.array([GM_SUN, GM_SUN * 3e-6]),  # ~Earth mass
            positions=np.array([[0, 0, 0], [a, 0, 0]], dtype=float),
            velocities=np.array([[0, 0, 0], [0, v_circ, 0]], dtype=float),
            masses_kg=np.array([1.989e30, 5.972e24]),
            star_index=0,
        )

    def test_universal_mode_runs(self):
        """Simulation runs successfully via set_initial_state."""
        state = self._make_two_body_state()
        sim = NBodySimulation(relativistic=False)
        sim.set_initial_state(state)
        result = sim.run(duration_days=10, dt=0.1, integrator="verlet")
        assert len(result.times) > 0
        assert not np.any(np.isnan(result.positions))

    def test_universal_mode_energy_conservation(self):
        """Universal mode conserves energy."""
        state = self._make_two_body_state()
        sim = NBodySimulation(relativistic=False)
        sim.set_initial_state(state)
        result = sim.run(duration_days=365.25, dt=0.5, integrator="verlet")

        e0 = result.energies[0]
        drift = np.max(np.abs(result.energies - e0)) / abs(e0)
        assert drift < 1e-6, f"Energy drift {drift} > 1e-6"

    def test_universal_mode_circular_orbit(self):
        """Planet on circular orbit stays at ~constant distance."""
        state = self._make_two_body_state()
        sim = NBodySimulation(relativistic=False)
        sim.set_initial_state(state)
        result = sim.run(duration_days=365.25, dt=0.5, integrator="verlet")

        # Distance from star
        dr = result.positions[:, 1, :] - result.positions[:, 0, :]
        dist = np.linalg.norm(dr, axis=1)
        assert abs(dist.min() - 1.0) < 0.01, f"min distance = {dist.min()}"
        assert abs(dist.max() - 1.0) < 0.01, f"max distance = {dist.max()}"

    def test_to_dataframe_star_relative(self):
        """to_dataframe_star_relative returns correct relative positions."""
        state = self._make_two_body_state()
        sim = NBodySimulation(relativistic=False)
        sim.set_initial_state(state)
        result = sim.run(duration_days=10, dt=1.0, integrator="verlet")

        df = result.to_dataframe_star_relative("planet", star_index=0)
        assert "x" in df.columns
        assert "vx" in df.columns
        # Initial position should be ~(1, 0, 0) relative to star
        assert abs(df["x"].iloc[0] - 1.0) < 1e-10

    def test_universal_with_relativistic(self):
        """Universal mode works with relativistic corrections enabled."""
        state = self._make_two_body_state()
        sim = NBodySimulation(relativistic=True)
        sim.set_initial_state(state)
        result = sim.run(duration_days=10, dt=0.1, integrator="verlet")
        assert not np.any(np.isnan(result.positions))

    def test_trappist1_short_simulation(self):
        """TRAPPIST-1 can be simulated for a few orbital periods."""
        from astroworld.catalogs.trappist_1 import make_trappist1_state

        state = make_trappist1_state(include_planets=["b", "c"])
        sim = NBodySimulation(relativistic=False)
        sim.set_initial_state(state)
        result = sim.run(duration_days=5.0, dt=0.001, integrator="verlet")

        # Should have no NaN
        assert not np.any(np.isnan(result.positions))
        assert not np.any(np.isnan(result.velocities))

        # Energy should be conserved
        e0 = result.energies[0]
        drift = np.max(np.abs(result.energies - e0)) / abs(e0)
        assert drift < 1e-4, f"TRAPPIST-1 energy drift {drift} > 1e-4"

        # Planets should stay bound (distance < 0.1 AU)
        for i in range(1, state.n):
            dr = result.positions[:, i, :] - result.positions[:, 0, :]
            dist = np.linalg.norm(dr, axis=1)
            assert dist.max() < 0.1, f"{state.names[i]} escaped: max dist = {dist.max()}"
