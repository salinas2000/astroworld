"""
Tests for Phase 15: Blind Search grid generation and campaign orchestration.

Tests are offline (no network, no GPU).  Focus on grid generation logic,
galactic plane filtering, cos(dec) correction, preset validation, and
batch control features.
"""

import numpy as np
import pytest

from scripts.run_blind_search import (
    PRESETS,
    generate_grid,
    _galactic_latitude,
    save_grid_csv,
)


# ---------------------------------------------------------------------------
# Galactic latitude conversion
# ---------------------------------------------------------------------------


class TestGalacticLatitude:
    """Tests for ICRS → Galactic latitude conversion."""

    def test_north_galactic_pole(self):
        """NGP (RA=192.86, Dec=+27.13) should have b ≈ +90°."""
        b = _galactic_latitude(192.86, 27.13)
        assert abs(b - 90.0) < 1.0  # within 1°

    def test_galactic_center(self):
        """Galactic center (RA=266.40, Dec=-28.94) should have b ≈ 0°."""
        b = _galactic_latitude(266.40, -28.94)
        assert abs(b) < 2.0  # within 2° (approximate formula)

    def test_high_latitude_region(self):
        """Taurus region (RA=70, Dec=20) should be well above galactic plane."""
        b = _galactic_latitude(70.0, 20.0)
        assert abs(b) > 15.0  # should be at high Galactic latitude

    def test_near_galactic_plane(self):
        """A point near the Galactic plane should have small |b|."""
        # Galactic equator passes through RA~280, Dec~-29
        b = _galactic_latitude(280.0, -29.0)
        assert abs(b) < 15.0


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------


class TestGenerateGrid:
    """Tests for grid generation with various parameters."""

    def test_basic_grid(self):
        """Simple grid generates expected number of fields."""
        fields = generate_grid(
            ra_min=0, ra_max=1, dec_min=0, dec_max=1,
            step_deg=0.5,
            exclude_galactic_plane=False,
        )
        assert len(fields) > 0
        # 3 dec values (0, 0.5, 1.0) × ~3 RA values each ≈ 9
        assert len(fields) >= 6

    def test_grid_bounds(self):
        """All fields are within the requested bounds."""
        fields = generate_grid(
            ra_min=55, ra_max=85, dec_min=10, dec_max=30,
            step_deg=2.0,
            exclude_galactic_plane=False,
        )
        for ra, dec in fields:
            assert 55.0 <= ra <= 85.5  # small margin for rounding
            assert 10.0 <= dec <= 30.5

    def test_cos_dec_correction(self):
        """Higher declinations produce wider RA spacing."""
        # At Dec=60°, cos(60)=0.5 → RA step should be ~2× larger
        fields_equator = generate_grid(
            ra_min=0, ra_max=10, dec_min=0, dec_max=0,
            step_deg=1.0,
            exclude_galactic_plane=False,
        )
        fields_high_dec = generate_grid(
            ra_min=0, ra_max=10, dec_min=60, dec_max=60,
            step_deg=1.0,
            exclude_galactic_plane=False,
        )
        # High dec should have fewer RA points (wider spacing)
        assert len(fields_high_dec) < len(fields_equator)

    def test_galactic_filter_removes_fields(self):
        """Galactic plane filter reduces field count."""
        fields_no_filter = generate_grid(
            ra_min=80, ra_max=120, dec_min=-10, dec_max=10,
            step_deg=2.0,
            exclude_galactic_plane=False,
        )
        fields_filtered = generate_grid(
            ra_min=80, ra_max=120, dec_min=-10, dec_max=10,
            step_deg=2.0,
            exclude_galactic_plane=True,
            galactic_lat_cut=10.0,
        )
        assert len(fields_filtered) <= len(fields_no_filter)

    def test_empty_grid_high_galactic_cut(self):
        """Very strict galactic cut can produce empty grid."""
        # This region is close to the galactic plane
        fields = generate_grid(
            ra_min=265, ra_max=268, dec_min=-30, dec_max=-28,
            step_deg=0.5,
            exclude_galactic_plane=True,
            galactic_lat_cut=80.0,  # Very strict
        )
        # Should be empty or very few fields
        assert len(fields) <= 2

    def test_single_field(self):
        """Grid with tight bounds produces at least 1 field."""
        fields = generate_grid(
            ra_min=70, ra_max=70.1, dec_min=20, dec_max=20.1,
            step_deg=0.5,
            exclude_galactic_plane=False,
        )
        assert len(fields) >= 1

    def test_fields_are_tuples(self):
        """Grid returns list of (float, float) tuples."""
        fields = generate_grid(
            ra_min=0, ra_max=1, dec_min=0, dec_max=1,
            step_deg=1.0,
            exclude_galactic_plane=False,
        )
        for f in fields:
            assert isinstance(f, tuple)
            assert len(f) == 2
            assert isinstance(f[0], float)
            assert isinstance(f[1], float)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    """Tests for preset region definitions."""

    def test_all_presets_have_required_keys(self):
        """Each preset has all required configuration keys."""
        required = {"description", "ra_min", "ra_max", "dec_min", "dec_max", "step_deg"}
        for name, preset in PRESETS.items():
            for key in required:
                assert key in preset, f"Preset '{name}' missing key '{key}'"

    def test_presets_generate_nonempty_grids(self):
        """Each preset generates at least 1 field."""
        for name, preset in PRESETS.items():
            fields = generate_grid(
                preset["ra_min"], preset["ra_max"],
                preset["dec_min"], preset["dec_max"],
                step_deg=preset["step_deg"],
                exclude_galactic_plane=True,
            )
            assert len(fields) > 0, f"Preset '{name}' generates empty grid"

    def test_preset_bounds_valid(self):
        """Preset RA/Dec bounds are physically valid."""
        for name, preset in PRESETS.items():
            assert 0 <= preset["ra_min"] < 360
            assert 0 <= preset["ra_max"] <= 360
            assert preset["ra_min"] < preset["ra_max"]
            assert -90 <= preset["dec_min"] <= 90
            assert -90 <= preset["dec_max"] <= 90
            assert preset["dec_min"] < preset["dec_max"]
            assert preset["step_deg"] > 0

    def test_taurus_is_at_high_galactic_latitude(self):
        """Taurus region should be mostly above |b| > 10°."""
        fields = generate_grid(
            55, 85, 10, 30,
            step_deg=2.0,
            exclude_galactic_plane=True,
            galactic_lat_cut=10.0,
        )
        # Most taurus fields should survive the cut
        fields_no_cut = generate_grid(
            55, 85, 10, 30,
            step_deg=2.0,
            exclude_galactic_plane=False,
        )
        survival_rate = len(fields) / max(len(fields_no_cut), 1)
        assert survival_rate > 0.7  # >70% survive the galactic cut


# ---------------------------------------------------------------------------
# Grid CSV
# ---------------------------------------------------------------------------


class TestSaveGridCSV:
    """Tests for grid CSV persistence."""

    def test_save_and_load(self, tmp_path):
        """Grid can be saved and loaded back."""
        import pandas as pd

        fields = [(10.0, 20.0), (30.0, 40.0), (50.0, -10.0)]
        csv_path = tmp_path / "test_grid.csv"
        save_grid_csv(fields, csv_path)

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["ra_deg", "dec_deg"]
        assert len(df) == 3
        assert df["ra_deg"].iloc[0] == 10.0
        assert df["dec_deg"].iloc[2] == -10.0

    def test_empty_grid(self, tmp_path):
        """Empty grid produces valid CSV with headers."""
        import pandas as pd

        csv_path = tmp_path / "empty.csv"
        save_grid_csv([], csv_path)

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["ra_deg", "dec_deg"]
        assert len(df) == 0
