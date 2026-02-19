"""Tests for the outer Solar System catalog (Planet 9 hunt)."""

import pytest
import numpy as np

from astroworld.catalogs.outer_solar_system import (
    make_outer_system_without_p9,
    make_outer_system_with_p9,
    get_theoretical_periods,
    GM_STAR,
    _GAS_GIANTS,
    _TNOS,
    _PLANET_9,
)
from astroworld.constants import GM_SUN


class TestOuterSystemCatalog:
    """Basic catalog structure tests."""

    def test_without_p9_body_count(self):
        """Without P9: Sun + 4 giants + 7 TNOs = 12."""
        state = make_outer_system_without_p9()
        assert state.n == 12

    def test_with_p9_body_count(self):
        """With P9: Sun + 4 giants + 7 TNOs + P9 = 13."""
        state = make_outer_system_with_p9()
        assert state.n == 13

    def test_planet9_is_last_body(self):
        """Planet 9 should be the last body in the list."""
        state = make_outer_system_with_p9()
        assert state.names[-1] == "planet_9"

    def test_star_is_first(self):
        """Sun should be at index 0."""
        state = make_outer_system_without_p9()
        assert state.names[0] == "sun"
        assert state.star_index == 0

    def test_star_gm_includes_inner_planets(self):
        """GM_STAR should be larger than GM_SUN (inner masses absorbed)."""
        assert GM_STAR > GM_SUN
        # Inner planets are ~0.0006% of the Sun's mass
        ratio = GM_STAR / GM_SUN
        assert 1.0 < ratio < 1.001

    def test_no_nan_positions(self):
        """No NaN in any body's initial position."""
        state = make_outer_system_with_p9()
        assert not np.any(np.isnan(state.positions))

    def test_no_nan_velocities(self):
        """No NaN in any body's initial velocity."""
        state = make_outer_system_with_p9()
        assert not np.any(np.isnan(state.velocities))

    def test_star_at_origin(self):
        """Sun should be at the origin."""
        state = make_outer_system_without_p9()
        np.testing.assert_array_equal(state.positions[0], [0, 0, 0])
        np.testing.assert_array_equal(state.velocities[0], [0, 0, 0])

    def test_colors_match_body_count(self):
        """Colors list length should match number of bodies."""
        for state in [make_outer_system_without_p9(), make_outer_system_with_p9()]:
            assert len(state.colors) == state.n


class TestOuterSystemPhysics:
    """Physical validity of initial conditions."""

    def test_gas_giant_distances(self):
        """Gas giants should be at approximately their semi-major axis distances."""
        state = make_outer_system_without_p9()
        expected_a = {"jupiter": 5.2, "saturn": 9.6, "uranus": 19.2, "neptune": 30.1}
        for i, name in enumerate(state.names):
            if name in expected_a:
                r = np.linalg.norm(state.positions[i])
                a = expected_a[name]
                # Distance can vary from a due to eccentricity: r ∈ [a(1-e), a(1+e)]
                assert 0.5 * a < r < 1.5 * a, (
                    f"{name}: r={r:.2f}, expected near a={a} AU"
                )

    def test_tno_distances_are_large(self):
        """All TNOs should be at > 30 AU from the Sun (beyond Neptune)."""
        state = make_outer_system_without_p9()
        tno_names = {"sedna", "2012_vp113", "leleakuhonua",
                     "2004_vn112", "2007_tg422", "2013_rf98", "2010_gb174"}
        for i, name in enumerate(state.names):
            if name in tno_names:
                r = np.linalg.norm(state.positions[i])
                assert r > 30, f"{name}: r={r:.1f} AU, should be > 30 AU"

    def test_planet9_is_distant(self):
        """Planet 9 should be far from the Sun (> 200 AU)."""
        state = make_outer_system_with_p9()
        p9_idx = state.names.index("planet_9")
        r = np.linalg.norm(state.positions[p9_idx])
        # P9 at a=380, e=0.2 → r ∈ [304, 456] AU
        assert r > 200, f"Planet 9: r={r:.1f} AU, should be > 200 AU"

    def test_planet9_gm_roughly_6_earth_masses(self):
        """Planet 9 GM should be ~ 6.2 Earth masses × G."""
        state = make_outer_system_with_p9()
        p9_idx = state.names.index("planet_9")
        gm_p9 = state.gm_values[p9_idx]
        # Earth GM ~ 8.887e-10 AU³/day², P9 ~ 6.2x that
        gm_earth_sim = 6.6743e-11 * 5.97237e24 * (86400**2) / (1.496e11**3)
        ratio = gm_p9 / gm_earth_sim
        assert 4 < ratio < 10, f"P9 GM / Earth GM = {ratio:.1f}, expected ~6.2"

    def test_velocities_are_bound(self):
        """All bodies should have velocities below escape speed."""
        state = make_outer_system_with_p9()
        for i, name in enumerate(state.names):
            if i == 0:
                continue
            r = np.linalg.norm(state.positions[i])
            v = np.linalg.norm(state.velocities[i])
            v_escape = np.sqrt(2 * GM_STAR / r)
            assert v < v_escape, (
                f"{name}: v={v:.6f} > v_escape={v_escape:.6f} AU/day"
            )

    def test_jupiter_velocity_is_circular_ish(self):
        """Jupiter's velocity should be close to circular: v_circ = sqrt(GM/r)."""
        state = make_outer_system_without_p9()
        jup_idx = state.names.index("jupiter")
        r = np.linalg.norm(state.positions[jup_idx])
        v = np.linalg.norm(state.velocities[jup_idx])
        v_circ = np.sqrt(GM_STAR / r)
        # Jupiter e=0.049, so v/v_circ should be ~1 ± 5%
        assert 0.9 < v / v_circ < 1.1, (
            f"v/v_circ = {v/v_circ:.4f} for Jupiter"
        )


class TestOuterSystemSubsets:
    """Test TNO subset selection."""

    def test_single_tno(self):
        """Selecting a single TNO should give Sun + 4 giants + 1 TNO = 6."""
        state = make_outer_system_without_p9(tno_names=["sedna"])
        assert state.n == 6
        assert "2004_vn112" not in state.names
        assert "sedna" in state.names
        assert "2012_vp113" not in state.names

    def test_two_tnos(self):
        """Selecting two TNOs."""
        state = make_outer_system_without_p9(tno_names=["sedna", "leleakuhonua"])
        assert state.n == 7
        assert "sedna" in state.names
        assert "leleakuhonua" in state.names
        assert "2012_vp113" not in state.names

    def test_p9_with_subset(self):
        """P9 + subset of TNOs."""
        state = make_outer_system_with_p9(tno_names=["sedna"])
        assert state.n == 7  # Sun + 4 giants + sedna + P9
        assert "planet_9" in state.names
        assert "sedna" in state.names


class TestTheoreticalPeriods:
    """Test Keplerian period calculations."""

    def test_jupiter_period(self):
        """Jupiter period should be ~4332 days (~11.86 years)."""
        periods = get_theoretical_periods()
        T_jup = periods["jupiter"]
        assert 4200 < T_jup < 4500, f"Jupiter period: {T_jup:.0f} days"

    def test_neptune_period(self):
        """Neptune period should be ~60190 days (~164.8 years)."""
        periods = get_theoretical_periods()
        T_nep = periods["neptune"]
        assert 59000 < T_nep < 62000, f"Neptune period: {T_nep:.0f} days"

    def test_sedna_period(self):
        """Sedna period should be ~11,000+ years."""
        periods = get_theoretical_periods()
        T_sedna_years = periods["sedna"] / 365.25
        assert 10000 < T_sedna_years < 13000, (
            f"Sedna period: {T_sedna_years:.0f} years"
        )

    def test_planet9_period(self):
        """Planet 9 period should be ~7400 years (a=380 AU)."""
        periods = get_theoretical_periods()
        T_p9_years = periods["planet_9"] / 365.25
        assert 6000 < T_p9_years < 9000, (
            f"Planet 9 period: {T_p9_years:.0f} years"
        )

    def test_all_bodies_have_periods(self):
        """All 12 bodies should have computed periods."""
        periods = get_theoretical_periods()
        expected = {"jupiter", "saturn", "uranus", "neptune",
                    "sedna", "2012_vp113", "leleakuhonua",
                    "2004_vn112", "2007_tg422", "2013_rf98", "2010_gb174",
                    "planet_9"}
        assert set(periods.keys()) == expected

    def test_periods_increase_with_distance(self):
        """Kepler's 3rd law: period increases with semi-major axis."""
        periods = get_theoretical_periods()
        giants = ["jupiter", "saturn", "uranus", "neptune"]
        for i in range(len(giants) - 1):
            assert periods[giants[i]] < periods[giants[i + 1]], (
                f"{giants[i]} period should be < {giants[i+1]}"
            )
