"""Tests for catalog cross-matching (SIMBAD + Gaia DR3).

All external API calls are mocked to avoid network dependencies
and to respect server rate limits during CI/testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from astroworld.ml.crossmatch import (
    CONTAMINANT_TYPES,
    GAIA_PM_THRESHOLD_MAS_YR,
    CatalogMatcher,
    MatchResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def matcher():
    """CatalogMatcher with fast rate limit for testing."""
    return CatalogMatcher(
        search_radius_arcsec=10.0,
        rate_limit_s=0.0,  # No delay in tests
        pm_threshold=5.0,
    )


def _make_simbad_table(name: str, otype: str, ra: float, dec: float) -> Table:
    """Create a mock SIMBAD result table."""
    return Table(
        {
            "main_id": [name],
            "ra": [ra],
            "dec": [dec],
            "coo_err_maj": [0.5],
            "coo_err_min": [0.5],
            "coo_err_angle": [0],
            "coo_wavelength": ["O"],
            "coo_bibcode": ["2020A&A...000..000X"],
            "otype": [otype],
        }
    )


def _make_gaia_table(
    designation: str,
    ra: float, dec: float,
    pmra: float, pmdec: float,
    parallax: float,
) -> Table:
    """Create a mock Gaia DR3 result table."""
    return Table(
        {
            "designation": [designation],
            "ra": [ra],
            "dec": [dec],
            "pmra": [pmra],
            "pmdec": [pmdec],
            "parallax": [parallax],
            "source_id": [1234567890],
        }
    )


def _sample_candidates_df(n: int = 5) -> pd.DataFrame:
    """Create a small sample DataFrame mimicking pipeline output."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "field_id": [f"field_{i}" for i in range(n)],
        "ra_deg": rng.uniform(69.5, 73.0, n),
        "dec_deg": rng.uniform(12.5, 16.5, n),
        "probability": rng.uniform(0.8, 1.0, n),
        "temperature_k": rng.uniform(15, 40, n),
        "planck_class": ["p9_candidate"] * n,
    })


# ---------------------------------------------------------------------------
# MatchResult tests
# ---------------------------------------------------------------------------

class TestMatchResult:
    def test_defaults(self):
        m = MatchResult(source="none")
        assert m.source == "none"
        assert m.name is None
        assert m.otype is None
        assert m.is_contaminant is False
        assert m.separation_arcsec == 0.0

    def test_simbad_contaminant(self):
        m = MatchResult(
            source="SIMBAD", name="NGC 1234", otype="G",
            separation_arcsec=3.2, is_contaminant=True,
        )
        assert m.is_contaminant is True
        assert m.otype == "G"

    def test_gaia_with_pm(self):
        m = MatchResult(
            source="Gaia", name="Gaia DR3 123",
            total_pm=15.3, pmra=10.0, pmdec=11.5,
            is_contaminant=True,
        )
        assert m.total_pm == 15.3
        assert m.is_contaminant is True


# ---------------------------------------------------------------------------
# Contaminant type set
# ---------------------------------------------------------------------------

class TestContaminantTypes:
    def test_galaxies_included(self):
        for t in ["G", "AGN", "QSO", "Sy1", "Sy2", "SyG"]:
            assert t in CONTAMINANT_TYPES, f"{t} should be a contaminant"

    def test_stars_included(self):
        for t in ["Star", "V*", "PM*", "WD*", "RR*"]:
            assert t in CONTAMINANT_TYPES, f"{t} should be a contaminant"

    def test_non_contaminant_not_included(self):
        # Made-up types that should NOT be in the set
        for t in ["Planet9", "Asteroid", "Comet", "Unknown"]:
            assert t not in CONTAMINANT_TYPES


# ---------------------------------------------------------------------------
# SIMBAD query tests (mocked)
# ---------------------------------------------------------------------------

class TestCheckSimbad:
    def test_galaxy_match(self, matcher):
        """SIMBAD returns a galaxy → contaminant."""
        table = _make_simbad_table("NGC 1234", "G", 69.83, 12.96)
        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = table

        result = matcher.check_simbad(69.83, 12.96)
        assert result.source == "SIMBAD"
        assert result.otype == "G"
        assert result.is_contaminant is True
        assert result.name == "NGC 1234"

    def test_agn_match(self, matcher):
        """SIMBAD returns an AGN → contaminant."""
        table = _make_simbad_table("QSO J0459+1234", "QSO", 69.83, 12.96)
        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = table

        result = matcher.check_simbad(69.83, 12.96)
        assert result.is_contaminant is True
        assert result.otype == "QSO"

    def test_no_match(self, matcher):
        """SIMBAD returns nothing → no contaminant."""
        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = None

        result = matcher.check_simbad(69.83, 12.96)
        assert result.source == "none"
        assert result.is_contaminant is False

    def test_empty_table(self, matcher):
        """SIMBAD returns empty table → no contaminant."""
        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = Table()

        result = matcher.check_simbad(69.83, 12.96)
        assert result.source == "none"

    def test_network_error(self, matcher):
        """SIMBAD network error → graceful error."""
        matcher.simbad = MagicMock()
        matcher.simbad.query_region.side_effect = ConnectionError("timeout")

        result = matcher.check_simbad(69.83, 12.96)
        assert result.source == "error"
        assert "timeout" in result.details.get("error", "")

    def test_non_contaminant_type(self, matcher):
        """SIMBAD returns unknown type → NOT contaminant but still matched."""
        table = _make_simbad_table("IRC+10216", "C*", 69.83, 12.96)
        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = table

        result = matcher.check_simbad(69.83, 12.96)
        assert result.source == "SIMBAD"
        assert result.is_contaminant is False  # C* not in CONTAMINANT_TYPES


# ---------------------------------------------------------------------------
# Gaia query tests (mocked)
# ---------------------------------------------------------------------------

class TestCheckGaia:
    @patch("astroworld.ml.crossmatch.Gaia")
    def test_high_pm_star(self, MockGaia, matcher):
        """Gaia star with high proper motion → contaminant."""
        table = _make_gaia_table(
            "Gaia DR3 123", 69.83, 12.96,
            pmra=20.0, pmdec=30.0, parallax=5.0,
        )
        mock_job = MagicMock()
        mock_job.get_results.return_value = table
        MockGaia.cone_search_async.return_value = mock_job

        result = matcher.check_gaia(69.83, 12.96)
        assert result.source == "Gaia"
        assert result.is_contaminant is True
        assert result.total_pm > 5.0

    @patch("astroworld.ml.crossmatch.Gaia")
    def test_low_pm_star(self, MockGaia, matcher):
        """Gaia star with low proper motion → NOT contaminant."""
        table = _make_gaia_table(
            "Gaia DR3 456", 69.83, 12.96,
            pmra=1.0, pmdec=1.0, parallax=0.5,
        )
        mock_job = MagicMock()
        mock_job.get_results.return_value = table
        MockGaia.cone_search_async.return_value = mock_job

        result = matcher.check_gaia(69.83, 12.96)
        assert result.source == "Gaia"
        assert result.is_contaminant is False
        assert result.total_pm < 5.0

    @patch("astroworld.ml.crossmatch.Gaia")
    def test_no_gaia_match(self, MockGaia, matcher):
        """No Gaia sources in cone → no match."""
        empty = Table(
            names=["designation", "ra", "dec", "pmra", "pmdec",
                   "parallax", "source_id"],
            dtype=[str, float, float, float, float, float, int],
        )
        mock_job = MagicMock()
        mock_job.get_results.return_value = empty
        MockGaia.cone_search_async.return_value = mock_job

        result = matcher.check_gaia(69.83, 12.96)
        assert result.source == "none"

    @patch("astroworld.ml.crossmatch.Gaia")
    def test_nan_proper_motions(self, MockGaia, matcher):
        """Gaia source with NaN proper motions → no contaminant."""
        table = _make_gaia_table(
            "Gaia DR3 789", 69.83, 12.96,
            pmra=float("nan"), pmdec=float("nan"), parallax=0.5,
        )
        mock_job = MagicMock()
        mock_job.get_results.return_value = table
        MockGaia.cone_search_async.return_value = mock_job

        result = matcher.check_gaia(69.83, 12.96)
        assert result.source == "none"

    @patch("astroworld.ml.crossmatch.Gaia")
    def test_gaia_network_error(self, MockGaia, matcher):
        """Gaia network error → graceful handling."""
        MockGaia.cone_search_async.side_effect = ConnectionError("timeout")

        result = matcher.check_gaia(69.83, 12.96)
        assert result.source == "error"


# ---------------------------------------------------------------------------
# Combined candidate check
# ---------------------------------------------------------------------------

class TestCheckCandidate:
    def test_simbad_contaminant_skips_gaia(self, matcher):
        """If SIMBAD finds a contaminant, Gaia is not queried."""
        table = _make_simbad_table("NGC 1234", "G", 69.83, 12.96)
        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = table

        with patch("astroworld.ml.crossmatch.Gaia") as MockGaia:
            result = matcher.check_candidate(69.83, 12.96)
            MockGaia.cone_search_async.assert_not_called()

        assert result.source == "SIMBAD"
        assert result.is_contaminant is True

    def test_no_simbad_falls_through_to_gaia(self, matcher):
        """If SIMBAD finds nothing, Gaia is checked."""
        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = None

        with patch("astroworld.ml.crossmatch.Gaia") as MockGaia:
            table = _make_gaia_table(
                "Gaia DR3 123", 69.83, 12.96,
                pmra=50.0, pmdec=50.0, parallax=10.0,
            )
            mock_job = MagicMock()
            mock_job.get_results.return_value = table
            MockGaia.cone_search_async.return_value = mock_job

            result = matcher.check_candidate(69.83, 12.96)

        assert result.source == "Gaia"
        assert result.is_contaminant is True

    def test_no_match_anywhere(self, matcher):
        """Nothing in SIMBAD or Gaia → source="none"."""
        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = None

        with patch("astroworld.ml.crossmatch.Gaia") as MockGaia:
            empty = Table(
                names=["designation", "ra", "dec", "pmra", "pmdec",
                       "parallax", "source_id"],
                dtype=[str, float, float, float, float, float, int],
            )
            mock_job = MagicMock()
            mock_job.get_results.return_value = empty
            MockGaia.cone_search_async.return_value = mock_job

            result = matcher.check_candidate(69.83, 12.96)

        assert result.source == "none"
        assert result.is_contaminant is False


# ---------------------------------------------------------------------------
# DataFrame filtering
# ---------------------------------------------------------------------------

class TestFilterCandidates:
    def test_mixed_results(self, matcher):
        """Test filter with mix of known and unknown candidates."""
        df = _sample_candidates_df(4)

        # Mock: candidate 0 = galaxy, 1 = nothing, 2 = star, 3 = nothing
        simbad_results = [
            _make_simbad_table("NGC 1", "G", df.iloc[0]["ra_deg"], df.iloc[0]["dec_deg"]),
            None,
            _make_simbad_table("HD 12345", "Star", df.iloc[2]["ra_deg"], df.iloc[2]["dec_deg"]),
            None,
        ]
        call_count = {"idx": 0}

        def mock_query_region(coord, radius):
            idx = call_count["idx"]
            call_count["idx"] += 1
            return simbad_results[idx % len(simbad_results)]

        matcher.simbad = MagicMock()
        matcher.simbad.query_region.side_effect = mock_query_region

        with patch("astroworld.ml.crossmatch.Gaia") as MockGaia:
            empty = Table(
                names=["designation", "ra", "dec", "pmra", "pmdec",
                       "parallax", "source_id"],
                dtype=[str, float, float, float, float, float, int],
            )
            mock_job = MagicMock()
            mock_job.get_results.return_value = empty
            MockGaia.cone_search_async.return_value = mock_job

            df_known, df_unknown = matcher.filter_candidates(df, verbose=False)

        # Galaxy + Star are contaminants
        assert len(df_known) == 2
        assert len(df_unknown) == 2
        assert "catalog_source" in df_known.columns
        assert "catalog_otype" in df_known.columns
        assert "is_known_object" in df_known.columns

    def test_all_unknown(self, matcher):
        """No matches → all candidates are unknown."""
        df = _sample_candidates_df(3)

        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = None

        with patch("astroworld.ml.crossmatch.Gaia") as MockGaia:
            empty = Table(
                names=["designation", "ra", "dec", "pmra", "pmdec",
                       "parallax", "source_id"],
                dtype=[str, float, float, float, float, float, int],
            )
            mock_job = MagicMock()
            mock_job.get_results.return_value = empty
            MockGaia.cone_search_async.return_value = mock_job

            df_known, df_unknown = matcher.filter_candidates(df, verbose=False)

        assert len(df_known) == 0
        assert len(df_unknown) == 3

    def test_output_columns(self, matcher):
        """Verify output DataFrames have all expected columns."""
        df = _sample_candidates_df(2)

        matcher.simbad = MagicMock()
        matcher.simbad.query_region.return_value = None

        with patch("astroworld.ml.crossmatch.Gaia") as MockGaia:
            empty = Table(
                names=["designation", "ra", "dec", "pmra", "pmdec",
                       "parallax", "source_id"],
                dtype=[str, float, float, float, float, float, int],
            )
            mock_job = MagicMock()
            mock_job.get_results.return_value = empty
            MockGaia.cone_search_async.return_value = mock_job

            _, df_unknown = matcher.filter_candidates(df, verbose=False)

        expected_cols = {
            "catalog_source", "catalog_name", "catalog_otype",
            "catalog_sep_arcsec", "is_known_object", "gaia_pm_mas_yr",
        }
        assert expected_cols.issubset(set(df_unknown.columns))

    def test_empty_input(self, matcher):
        """Empty DataFrame → empty outputs."""
        df = pd.DataFrame(columns=["ra_deg", "dec_deg", "probability"])

        df_known, df_unknown = matcher.filter_candidates(df, verbose=False)
        assert len(df_known) == 0
        assert len(df_unknown) == 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_default_radius(self):
        m = CatalogMatcher()
        assert m.radius.value == 10.0

    def test_custom_radius(self):
        m = CatalogMatcher(search_radius_arcsec=15.0)
        assert m.radius.value == 15.0

    def test_pm_threshold(self):
        m = CatalogMatcher(pm_threshold=10.0)
        assert m.pm_threshold == 10.0

    def test_default_pm_threshold(self):
        assert GAIA_PM_THRESHOLD_MAS_YR == 5.0
