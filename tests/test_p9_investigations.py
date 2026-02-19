"""
Tests for Planet 9 investigation catalog functions and analysis helpers.

Covers:
  - make_outer_system_custom_p9(): parametric P9 factory
  - make_debris_disk(): distributed mass ring catalog
  - compute_orbital_elements(): shared element extraction
  - Smoke test: disk simulation runs without NaN
"""

import pytest
import numpy as np

from astroworld.catalogs.outer_solar_system import (
    make_outer_system_without_p9,
    make_outer_system_with_p9,
    make_outer_system_custom_p9,
    make_debris_disk,
    GM_STAR,
    _M_EARTH_KG,
)
from astroworld.analysis.orbital_elements import (
    compute_orbital_elements,
    extract_body_elements_timeseries,
)


# ── Custom P9 catalog ─────────────────────────────────────────────


class TestCustomP9Catalog:
    """Tests for make_outer_system_custom_p9."""

    def test_default_body_count(self):
        """Default params: Sun + 4 giants + 7 TNOs + P9 = 13."""
        state = make_outer_system_custom_p9()
        assert state.n == 13

    def test_planet9_present(self):
        """Planet 9 should be in the body list."""
        state = make_outer_system_custom_p9()
        assert "planet_9" in state.names

    def test_custom_mass_scales_gm(self):
        """Doubling mass should double GM."""
        state_3 = make_outer_system_custom_p9(mass_earth=3.0)
        state_9 = make_outer_system_custom_p9(mass_earth=9.0)
        p9_idx_3 = state_3.names.index("planet_9")
        p9_idx_9 = state_9.names.index("planet_9")
        ratio = state_9.gm_values[p9_idx_9] / state_3.gm_values[p9_idx_3]
        assert abs(ratio - 3.0) < 0.01

    def test_custom_distance_places_body(self):
        """P9 at distance_au=500 should have |r| near 500 AU (± eccentricity)."""
        state = make_outer_system_custom_p9(distance_au=500.0, eccentricity=0.1)
        p9_idx = state.names.index("planet_9")
        r = np.linalg.norm(state.positions[p9_idx])
        # With e=0.1: r in [a(1-e), a(1+e)] = [450, 550]
        assert 400 < r < 600, f"Planet 9: r={r:.1f} AU"

    def test_no_nan(self):
        """No NaN in positions/velocities for various parameters."""
        for mass, dist in [(2, 300), (10, 700), (6.2, 380)]:
            state = make_outer_system_custom_p9(mass_earth=mass, distance_au=dist)
            assert not np.any(np.isnan(state.positions)), f"NaN at mass={mass}, dist={dist}"
            assert not np.any(np.isnan(state.velocities)), f"NaN at mass={mass}, dist={dist}"

    def test_bound_orbit(self):
        """P9 should be on a bound orbit (v < v_escape)."""
        state = make_outer_system_custom_p9()
        p9_idx = state.names.index("planet_9")
        r = np.linalg.norm(state.positions[p9_idx])
        v = np.linalg.norm(state.velocities[p9_idx])
        v_esc = np.sqrt(2 * GM_STAR / r)
        assert v < v_esc, f"v={v:.6f} > v_esc={v_esc:.6f}"

    def test_tno_subset(self):
        """TNO subset selection works with custom P9."""
        state = make_outer_system_custom_p9(tno_names=["sedna"])
        assert state.n == 7  # Sun + 4 giants + sedna + P9
        assert "sedna" in state.names
        assert "2012_vp113" not in state.names
        assert "2004_vn112" not in state.names


# ── Debris disk catalog ────────────────────────────────────────────


class TestDebrisDisk:
    """Tests for make_debris_disk."""

    def test_body_count_100(self):
        """100 disk + Sun + 4 giants + 7 TNOs = 112."""
        state = make_debris_disk(n_particles=100)
        assert state.n == 112

    def test_body_count_30(self):
        """30 disk + Sun + 4 giants + 7 TNOs = 42."""
        state = make_debris_disk(n_particles=30)
        assert state.n == 42

    def test_total_mass_conservation(self):
        """Total disk GM should equal single P9 GM (same total mass)."""
        state_disk = make_debris_disk(n_particles=50, total_mass_earth=6.2)
        state_p9 = make_outer_system_with_p9()

        # Sum disk body GMs
        disk_gm = sum(
            state_disk.gm_values[i]
            for i, name in enumerate(state_disk.names)
            if name.startswith("disk_")
        )
        p9_gm = state_p9.gm_values[state_p9.names.index("planet_9")]

        assert abs(disk_gm - p9_gm) / p9_gm < 0.01, (
            f"disk_gm={disk_gm:.2e}, p9_gm={p9_gm:.2e}"
        )

    def test_disk_particles_at_ring_radius(self):
        """Disk particles should be near ring_radius_au (circular orbit)."""
        state = make_debris_disk(n_particles=10, ring_radius_au=400.0)
        for i, name in enumerate(state.names):
            if name.startswith("disk_"):
                r = np.linalg.norm(state.positions[i])
                # Circular orbit at 400 AU, r should equal a
                assert 380 < r < 420, f"{name}: r={r:.1f}"

    def test_disk_particles_evenly_spaced(self):
        """Disk particles should be spread around the ring (not clustered)."""
        state = make_debris_disk(n_particles=20, ring_radius_au=400.0)
        angles = []
        for i, name in enumerate(state.names):
            if name.startswith("disk_"):
                pos = state.positions[i]
                angles.append(np.arctan2(pos[1], pos[0]))
        angles = np.sort(angles)
        # 20 particles at 18° apart; angular separations should be roughly uniform
        diffs = np.diff(angles)
        # Allow for wrap-around: append the last-to-first gap
        diffs = np.append(diffs, 2 * np.pi + angles[0] - angles[-1])
        mean_sep = np.mean(diffs)
        # Coefficient of variation should be small
        cv = np.std(diffs) / mean_sep if mean_sep > 0 else 0
        assert cv < 0.5, f"Angular spacing CV={cv:.2f}, too clustered"

    def test_no_nan(self):
        """No NaN in positions/velocities."""
        state = make_debris_disk(n_particles=50)
        assert not np.any(np.isnan(state.positions))
        assert not np.any(np.isnan(state.velocities))

    def test_tnos_present(self):
        """All 7 TNOs should still be present in disk scenario."""
        state = make_debris_disk(n_particles=10)
        assert "sedna" in state.names
        assert "2012_vp113" in state.names
        assert "leleakuhonua" in state.names
        assert "2004_vn112" in state.names
        assert "2007_tg422" in state.names
        assert "2013_rf98" in state.names
        assert "2010_gb174" in state.names

    def test_star_at_origin(self):
        """Sun should be at the origin."""
        state = make_debris_disk(n_particles=10)
        np.testing.assert_array_equal(state.positions[0], [0, 0, 0])

    def test_ring_width(self):
        """With ring_width > 0, particles should span a range of distances."""
        state = make_debris_disk(
            n_particles=20, ring_radius_au=400.0, ring_width_au=100.0,
        )
        radii = []
        for i, name in enumerate(state.names):
            if name.startswith("disk_"):
                radii.append(np.linalg.norm(state.positions[i]))
        radii = np.array(radii)
        spread = radii.max() - radii.min()
        # Should have spread near 100 AU (from 350 to 450)
        assert spread > 50, f"Radial spread={spread:.1f}, expected > 50 AU"


# ── Orbital elements extraction ────────────────────────────────────


class TestOrbitalElements:
    """Tests for the shared compute_orbital_elements function."""

    def test_circular_orbit_eccentricity(self):
        """Circular orbit should have e near 0."""
        r = 1.0
        v = np.sqrt(GM_STAR / r)
        pos = np.array([r, 0, 0])
        vel = np.array([0, v, 0])
        elems = compute_orbital_elements(pos, vel, GM_STAR)
        assert elems["e"] < 0.01

    def test_circular_orbit_sma(self):
        """Circular orbit should have a = r."""
        r = 5.2
        v = np.sqrt(GM_STAR / r)
        pos = np.array([r, 0, 0])
        vel = np.array([0, v, 0])
        elems = compute_orbital_elements(pos, vel, GM_STAR)
        assert abs(elems["a"] - r) / r < 0.001

    def test_eccentric_orbit_at_perihelion(self):
        """At perihelion, recovered e and a should match input."""
        a, e = 2.0, 0.3
        r_peri = a * (1 - e)
        v_peri = np.sqrt(GM_STAR * (2 / r_peri - 1 / a))
        pos = np.array([r_peri, 0, 0])
        vel = np.array([0, v_peri, 0])
        elems = compute_orbital_elements(pos, vel, GM_STAR)
        assert abs(elems["e"] - e) < 0.001, f"e={elems['e']:.4f}, expected {e}"
        assert abs(elems["a"] - a) / a < 0.001, f"a={elems['a']:.4f}, expected {a}"

    def test_inclined_orbit(self):
        """Orbit in xz-plane should have i near 90 degrees."""
        r = 1.0
        v = np.sqrt(GM_STAR / r)
        pos = np.array([r, 0, 0])
        vel = np.array([0, 0, v])  # velocity in z-direction
        elems = compute_orbital_elements(pos, vel, GM_STAR)
        assert abs(elems["i"] - 90) < 1, f"i={elems['i']:.1f}, expected ~90"


# ── Smoke test: disk simulation ────────────────────────────────────


class TestDiskSimulationSmoke:
    """Quick smoke test: can the debris disk actually be simulated?"""

    @pytest.mark.slow
    def test_disk_short_simulation(self):
        """Debris disk (N=10) simulation runs 100 years without NaN."""
        from astroworld.engine.simulation import NBodySimulation

        state = make_debris_disk(n_particles=10)
        sim = NBodySimulation(relativistic=False)
        sim.set_initial_state(state)
        result = sim.run(
            duration_days=36525,  # 100 years
            dt=10.0, integrator="verlet", record_interval=100,
        )
        assert not np.any(np.isnan(result.positions))
        assert not np.any(np.isnan(result.velocities))
        e0 = result.energies[0]
        drift = abs((result.energies[-1] - e0) / abs(e0))
        assert drift < 1e-4, f"Energy drift {drift:.2e} too large"
