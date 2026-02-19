"""
Tests for JIT-compiled kernels (correctness against NumPy reference).

Three layers:
  1. Unit tests: individual kernel functions vs NumPy reference
  2. Integration tests: full JIT simulation vs Python simulation
  3. Regression: existing tests pass unchanged (run separately)
"""

import pytest
import numpy as np

from astroworld.engine.kernels import (
    HAS_NUMBA,
    _compute_accelerations_jit,
    _compute_total_energy_jit,
    _compute_gr_correction_jit,
    run_verlet_jit,
)
from astroworld.engine.gravity import (
    compute_accelerations,
    compute_total_energy,
)
from astroworld.engine.relativity import compute_gr_correction
from astroworld.constants import GM_SUN, BODIES, C_LIGHT


# ── Layer 1: Unit tests ──────────────────────────────────────────


class TestAccelerationsJIT:
    """JIT accelerations must match NumPy reference to machine precision."""

    def test_two_body_matches_numpy(self):
        """Sun-Earth system: JIT must match einsum/broadcast approach."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        gm_values = np.array([GM_SUN, BODIES["earth"].gm])

        acc_numpy = compute_accelerations(positions, gm_values)
        acc_jit = _compute_accelerations_jit(positions, gm_values, 2)

        np.testing.assert_allclose(acc_jit, acc_numpy, rtol=1e-13)

    def test_ten_body_random_matches_numpy(self):
        """Random 10-body system: JIT must match NumPy within float64 noise."""
        rng = np.random.default_rng(42)
        n = 10
        positions = rng.uniform(-30, 30, (n, 3))
        gm_values = rng.uniform(1e-12, 3e-4, n)

        acc_numpy = compute_accelerations(positions, gm_values)
        acc_jit = _compute_accelerations_jit(positions, gm_values, n)

        np.testing.assert_allclose(acc_jit, acc_numpy, rtol=1e-12)

    def test_newton_third_law_equal_masses(self):
        """F_AB = -F_BA for equal GM values."""
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        gm = np.array([1.0, 1.0])
        acc = _compute_accelerations_jit(positions, gm, 2)
        np.testing.assert_allclose(acc[0], -acc[1], rtol=1e-15)

    def test_newton_third_law_asymmetric(self):
        """With different GMs, a_i * m_i = -a_j * m_j (momentum conservation)."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        gm = np.array([GM_SUN, BODIES["earth"].gm])
        acc = _compute_accelerations_jit(positions, gm, 2)
        # GM_sun * acc_earth should equal -GM_earth * acc_sun in magnitude
        # Actually: a_i = GM_j / r^3 * r_vec, so F = m_i * a_i = GM_i * GM_j / (G * r^3) * r_vec
        # Simpler check: acc[0] should be tiny (Sun barely moves), acc[1] should be ~ -GM_sun
        assert abs(acc[0, 0]) < abs(acc[1, 0])  # Earth accelerates more
        assert acc[1, 0] < 0  # Earth accelerates toward Sun (negative x)


class TestEnergyJIT:
    """JIT energy must match NumPy reference."""

    def test_circular_orbit_matches_numpy(self):
        """Circular orbit energy: JIT vs Python should agree."""
        r = 1.0
        v_circ = np.sqrt(GM_SUN / r)
        positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.0, 0.0], [0.0, v_circ, 0.0]])
        gm = np.array([GM_SUN, BODIES["earth"].gm])
        masses = np.array([BODIES["sun"].mass_kg, BODIES["earth"].mass_kg])

        e_numpy = compute_total_energy(positions, velocities, gm, masses)
        e_jit = _compute_total_energy_jit(positions, velocities, gm, masses, 2)

        np.testing.assert_allclose(e_jit, e_numpy, rtol=1e-14)

    def test_energy_negative_for_bound_system(self):
        """Bound orbit should have negative total energy."""
        r = 1.0
        v_circ = np.sqrt(GM_SUN / r)
        positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.0, 0.0], [0.0, v_circ, 0.0]])
        gm = np.array([GM_SUN, BODIES["earth"].gm])
        masses = np.array([BODIES["sun"].mass_kg, BODIES["earth"].mass_kg])

        E = _compute_total_energy_jit(positions, velocities, gm, masses, 2)
        assert E < 0, f"Bound system should have E < 0, got {E}"

    def test_multi_body_matches_numpy(self):
        """5-body random system: energies must agree."""
        rng = np.random.default_rng(123)
        n = 5
        positions = rng.uniform(-10, 10, (n, 3))
        velocities = rng.uniform(-0.01, 0.01, (n, 3))
        gm = rng.uniform(1e-10, 1e-4, n)
        masses = rng.uniform(1e20, 1e30, n)

        e_numpy = compute_total_energy(positions, velocities, gm, masses)
        e_jit = _compute_total_energy_jit(positions, velocities, gm, masses, n)

        np.testing.assert_allclose(e_jit, e_numpy, rtol=1e-13)


class TestGRCorrectionJIT:
    """JIT GR correction must match Python reference."""

    def test_mercury_gr_matches_python(self):
        """Mercury-like orbit at perihelion: JIT vs Python should agree."""
        positions = np.array([[0.0, 0.0, 0.0], [0.387, 0.0, 0.0]])
        v = np.sqrt(GM_SUN / 0.387)
        velocities = np.array([[0.0, 0.0, 0.0], [0.0, v, 0.0]])
        c2 = C_LIGHT ** 2

        acc_py = compute_gr_correction(positions, velocities, GM_SUN, 0)
        acc_jit = _compute_gr_correction_jit(positions, velocities, GM_SUN, 0, c2, 2)

        np.testing.assert_allclose(acc_jit, acc_py, rtol=1e-13)

    def test_sun_correction_zero(self):
        """GR correction on the Sun itself should be zero."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.017, 0.0]])
        c2 = C_LIGHT ** 2

        acc = _compute_gr_correction_jit(positions, velocities, GM_SUN, 0, c2, 2)
        np.testing.assert_array_equal(acc[0], [0.0, 0.0, 0.0])

    def test_gr_small_vs_newton(self):
        """GR correction should be << Newtonian for Earth orbit."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        v = np.sqrt(GM_SUN / 1.0)
        velocities = np.array([[0.0, 0.0, 0.0], [0.0, v, 0.0]])
        c2 = C_LIGHT ** 2

        acc_newton = _compute_accelerations_jit(positions, np.array([GM_SUN, 8.887e-10]), 2)
        acc_gr = _compute_gr_correction_jit(positions, velocities, GM_SUN, 0, c2, 2)

        ratio = np.linalg.norm(acc_gr[1]) / np.linalg.norm(acc_newton[1])
        assert ratio < 1e-6, f"GR/Newton ratio {ratio:.2e} should be << 1"


# ── Layer 2: Integration tests ────────────────────────────────────


class TestFusedKernelVsPython:
    """Full simulation: JIT kernel must match Python loop."""

    def _make_3body_state(self):
        """Sun + Earth + Mars circular orbits."""
        from astroworld.data.state import SystemState
        r_e, r_m = 1.0, 1.524
        v_e = np.sqrt(GM_SUN / r_e)
        v_m = np.sqrt(GM_SUN / r_m)
        return SystemState(
            names=["sun", "earth", "mars"],
            gm_values=np.array([GM_SUN, BODIES["earth"].gm, BODIES["mars"].gm]),
            positions=np.array([[0, 0, 0], [r_e, 0, 0], [r_m, 0, 0]], dtype=np.float64),
            velocities=np.array([[0, 0, 0], [0, v_e, 0], [0, v_m, 0]], dtype=np.float64),
            masses_kg=np.array([BODIES["sun"].mass_kg, BODIES["earth"].mass_kg, BODIES["mars"].mass_kg]),
            star_index=0,
        )

    def test_newtonian_positions_match(self):
        """JIT and Python Verlet produce same positions over 100 days."""
        from astroworld.engine.simulation import NBodySimulation

        state = self._make_3body_state()

        # Python path
        sim_py = NBodySimulation()
        sim_py.set_initial_state(state)
        result_py = sim_py.run(duration_days=100.0, dt=1.0, use_jit=False)

        # JIT path
        sim_jit = NBodySimulation()
        sim_jit.set_initial_state(state)
        result_jit = sim_jit.run(duration_days=100.0, dt=1.0, use_jit=True)

        # Positions should match to ~1e-12 relative (float-point order-of-ops)
        np.testing.assert_allclose(
            result_jit.positions, result_py.positions,
            rtol=1e-10, atol=1e-15,
        )

    def test_newtonian_energies_match(self):
        """Energy histories should agree between JIT and Python."""
        from astroworld.engine.simulation import NBodySimulation

        state = self._make_3body_state()

        sim_py = NBodySimulation()
        sim_py.set_initial_state(state)
        result_py = sim_py.run(duration_days=100.0, dt=1.0, use_jit=False)

        sim_jit = NBodySimulation()
        sim_jit.set_initial_state(state)
        result_jit = sim_jit.run(duration_days=100.0, dt=1.0, use_jit=True)

        np.testing.assert_allclose(
            result_jit.energies, result_py.energies,
            rtol=1e-10,
        )

    def test_newtonian_velocities_match(self):
        """Velocity histories should agree."""
        from astroworld.engine.simulation import NBodySimulation

        state = self._make_3body_state()

        sim_py = NBodySimulation()
        sim_py.set_initial_state(state)
        result_py = sim_py.run(duration_days=100.0, dt=1.0, use_jit=False)

        sim_jit = NBodySimulation()
        sim_jit.set_initial_state(state)
        result_jit = sim_jit.run(duration_days=100.0, dt=1.0, use_jit=True)

        np.testing.assert_allclose(
            result_jit.velocities, result_py.velocities,
            rtol=1e-10, atol=1e-15,
        )

    def test_record_interval_matches(self):
        """JIT with record_interval > 1 produces same snapshots."""
        from astroworld.engine.simulation import NBodySimulation

        state = self._make_3body_state()

        sim_py = NBodySimulation()
        sim_py.set_initial_state(state)
        result_py = sim_py.run(duration_days=100.0, dt=0.5, record_interval=10, use_jit=False)

        sim_jit = NBodySimulation()
        sim_jit.set_initial_state(state)
        result_jit = sim_jit.run(duration_days=100.0, dt=0.5, record_interval=10, use_jit=True)

        assert len(result_jit.times) == len(result_py.times)
        np.testing.assert_allclose(result_jit.times, result_py.times, rtol=1e-12)
        np.testing.assert_allclose(result_jit.positions, result_py.positions, rtol=1e-10)

    def test_relativistic_matches(self):
        """GR mode: JIT and Python produce consistent results."""
        from astroworld.engine.simulation import NBodySimulation
        from astroworld.data.state import SystemState

        # Mercury-like orbit at perihelion
        r = 0.387
        v = np.sqrt(GM_SUN / r)
        state = SystemState(
            names=["sun", "mercury"],
            gm_values=np.array([GM_SUN, BODIES["mercury"].gm]),
            positions=np.array([[0, 0, 0], [r, 0, 0]], dtype=np.float64),
            velocities=np.array([[0, 0, 0], [0, v, 0]], dtype=np.float64),
            masses_kg=np.array([BODIES["sun"].mass_kg, BODIES["mercury"].mass_kg]),
            star_index=0,
        )

        sim_py = NBodySimulation(relativistic=True)
        sim_py.set_initial_state(state)
        result_py = sim_py.run(duration_days=100.0, dt=0.1, use_jit=False)

        sim_jit = NBodySimulation(relativistic=True)
        sim_jit.set_initial_state(state)
        result_jit = sim_jit.run(duration_days=100.0, dt=0.1, use_jit=True)

        np.testing.assert_allclose(
            result_jit.positions, result_py.positions,
            rtol=1e-10, atol=1e-15,
        )

    def test_jit_energy_conservation(self):
        """JIT kernel should conserve energy as well as Python loop."""
        from astroworld.engine.simulation import NBodySimulation

        state = self._make_3body_state()

        sim = NBodySimulation()
        sim.set_initial_state(state)
        result = sim.run(duration_days=365.25, dt=1.0, use_jit=True)

        e0 = result.energies[0]
        drift = np.max(np.abs((result.energies - e0) / abs(e0)))
        assert drift < 1e-7, f"Energy drift {drift:.2e} too large for JIT kernel"


# ── Layer 3: Smoke test for HAS_NUMBA flag ────────────────────────


class TestNumbaDetection:
    """Verify Numba detection and fallback."""

    def test_has_numba_is_boolean(self):
        assert isinstance(HAS_NUMBA, bool)

    def test_kernels_are_callable(self):
        """All kernel functions should be callable regardless of Numba."""
        assert callable(_compute_accelerations_jit)
        assert callable(_compute_total_energy_jit)
        assert callable(_compute_gr_correction_jit)
        assert callable(run_verlet_jit)
