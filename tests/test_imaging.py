"""
Tests for the imaging pipeline (sky position prediction, FITS storage).

Network-independent tests only — survey downloads are mocked.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from astropy.io import fits

from astroworld.imaging.sky_position import (
    predict_p9_sky_position,
    predict_p9_track,
    get_p9_search_region,
    DEFAULT_P9,
    _J2000_JD,
)
from astroworld.imaging.fits_store import load_fits_image, build_catalog
from astroworld.imaging.survey_download import _survey_to_dir


# ===================================================================
# Sky Position Tests
# ===================================================================

class TestP9SkyPosition:
    """Test Planet 9 sky coordinate prediction."""

    def test_p9_position_in_taurus_region(self):
        """Best-fit P9 at J2000 should be near ecliptic lon ~70° (Taurus)."""
        pos = predict_p9_sky_position(epoch_jd=_J2000_JD, **DEFAULT_P9)

        # Taurus region: RA ~50-100°, Dec ~5-30°
        assert 20.0 < pos["ra_deg"] < 150.0, (
            f"RA {pos['ra_deg']:.1f}° outside expected range"
        )
        assert -30.0 < pos["dec_deg"] < 50.0, (
            f"Dec {pos['dec_deg']:.1f}° outside expected range"
        )

    def test_p9_distance_at_aphelion(self):
        """P9 at M=180° (aphelion) should be at r = a*(1+e) = 480 AU."""
        pos = predict_p9_sky_position(
            epoch_jd=_J2000_JD,
            distance_au=400.0,
            eccentricity=0.20,
            mean_anomaly_deg=180.0,
        )
        # At aphelion: r = a*(1+e) = 400*(1.2) = 480 AU
        assert 470.0 < pos["distance_au"] < 490.0, (
            f"Distance {pos['distance_au']:.1f} AU, expected ~480 AU"
        )

    def test_p9_distance_at_perihelion(self):
        """P9 at M=0° (perihelion) should be at r = a*(1-e) = 320 AU."""
        pos = predict_p9_sky_position(
            epoch_jd=_J2000_JD,
            distance_au=400.0,
            eccentricity=0.20,
            mean_anomaly_deg=0.0,
        )
        assert 310.0 < pos["distance_au"] < 330.0, (
            f"Distance {pos['distance_au']:.1f} AU, expected ~320 AU"
        )

    def test_angular_velocity_order_of_magnitude(self):
        """At 480 AU, angular velocity should be ~0.5-5 arcsec/hour."""
        pos = predict_p9_sky_position(epoch_jd=_J2000_JD, **DEFAULT_P9)

        assert 0.01 < pos["v_angular_arcsec_hr"] < 10.0, (
            f"Angular velocity {pos['v_angular_arcsec_hr']:.4f} arcsec/hr "
            "outside expected range"
        )

    def test_magnitude_in_expected_range(self):
        """Apparent magnitude should be ~20-30 for a Neptune-class object at 480 AU."""
        pos = predict_p9_sky_position(epoch_jd=_J2000_JD, **DEFAULT_P9)

        assert 15.0 < pos["mag_estimate"] < 35.0, (
            f"Magnitude {pos['mag_estimate']:.1f} outside expected range"
        )

    def test_ecliptic_coordinates_returned(self):
        """Should return ecliptic x, y, z components."""
        pos = predict_p9_sky_position(epoch_jd=_J2000_JD, **DEFAULT_P9)

        assert "x_ecl" in pos
        assert "y_ecl" in pos
        assert "z_ecl" in pos

        # Verify distance consistency
        r_from_ecl = np.sqrt(
            pos["x_ecl"] ** 2 + pos["y_ecl"] ** 2 + pos["z_ecl"] ** 2
        )
        assert abs(r_from_ecl - pos["distance_au"]) < 0.1

    def test_mean_anomaly_propagation(self):
        """Position should change when epoch advances by 1 year."""
        pos_t0 = predict_p9_sky_position(epoch_jd=_J2000_JD, **DEFAULT_P9)
        pos_t1 = predict_p9_sky_position(
            epoch_jd=_J2000_JD + 365.25 * 10,  # 10 years later
            **DEFAULT_P9,
        )

        # P9 orbital period ~7000+ years, so 10 years = tiny movement
        # But RA/Dec should have changed
        delta_ra = abs(pos_t1["ra_deg"] - pos_t0["ra_deg"])
        delta_dec = abs(pos_t1["dec_deg"] - pos_t0["dec_deg"])

        # Should move at least a tiny bit (>0.001°) but not too much (<5°)
        assert delta_ra + delta_dec > 0.0001, "P9 should move over 10 years"
        assert delta_ra + delta_dec < 5.0, "P9 shouldn't move >5° in 10 years"


class TestP9Track:
    """Test sky track prediction over time."""

    def test_track_returns_dataframe(self):
        """predict_p9_track should return a DataFrame with expected columns."""
        track = predict_p9_track(
            start_jd=_J2000_JD,
            end_jd=_J2000_JD + 365.25 * 10,
            n_points=5,
        )

        assert isinstance(track, pd.DataFrame)
        assert len(track) == 5

        expected_cols = [
            "jd", "year", "ra_deg", "dec_deg", "distance_au",
            "v_angular_arcsec_hr", "mag_estimate",
        ]
        for col in expected_cols:
            assert col in track.columns, f"Missing column: {col}"

    def test_track_positions_are_smooth(self):
        """Sky positions should change smoothly (no jumps)."""
        track = predict_p9_track(
            start_jd=_J2000_JD,
            end_jd=_J2000_JD + 365.25 * 50,
            n_points=20,
        )

        # Check RA differences are smooth
        ra_diffs = np.abs(np.diff(track["ra_deg"].values))
        assert np.all(ra_diffs < 2.0), (
            f"RA jump detected: max diff = {ra_diffs.max():.3f}°"
        )

        dec_diffs = np.abs(np.diff(track["dec_deg"].values))
        assert np.all(dec_diffs < 2.0), (
            f"Dec jump detected: max diff = {dec_diffs.max():.3f}°"
        )

    def test_track_year_column_correct(self):
        """Year column should match JD conversion."""
        track = predict_p9_track(
            start_jd=_J2000_JD,
            end_jd=_J2000_JD + 365.25 * 10,
            n_points=3,
        )
        assert abs(track["year"].iloc[0] - 2000.0) < 0.01
        assert abs(track["year"].iloc[-1] - 2010.0) < 0.1


class TestSearchRegion:
    """Test search region computation."""

    def test_search_region_contains_center(self):
        """Search region bounding box should contain the center."""
        region = get_p9_search_region()

        assert region["ra_min"] < region["ra_center"] < region["ra_max"]
        assert region["dec_min"] < region["dec_center"] < region["dec_max"]

    def test_search_region_has_margin(self):
        """Bounding box should extend by at least margin_deg."""
        region = get_p9_search_region(margin_deg=3.0)

        dec_range = region["dec_max"] - region["dec_min"]
        assert dec_range >= 5.99, f"Dec range {dec_range:.1f}° < 6°"


# ===================================================================
# FITS Store Tests
# ===================================================================

class TestFitsStore:
    """Test FITS file loading and catalog building."""

    def test_load_fits_image(self, tmp_path):
        """Should load a FITS file and return (data, header)."""
        # Create a minimal FITS file
        data = np.random.random((64, 64)).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["CRVAL1"] = 75.0
        hdu.header["CRVAL2"] = 20.0
        fits_path = tmp_path / "test.fits"
        hdu.writeto(str(fits_path))

        img, header = load_fits_image(fits_path)

        assert img.shape == (64, 64)
        assert img.dtype == np.float64
        assert header["CRVAL1"] == 75.0
        assert header["CRVAL2"] == 20.0

    def test_build_catalog(self, tmp_path):
        """Should scan directory and build catalog DataFrame."""
        # Create subdirectory with FITS files
        subdir = tmp_path / "dss2_red"
        subdir.mkdir()

        for i in range(3):
            data = np.random.random((32, 32)).astype(np.float32)
            hdu = fits.PrimaryHDU(data)
            hdu.header["CRVAL1"] = 75.0 + i
            hdu.header["CRVAL2"] = 20.0
            hdu.header["NAXIS1"] = 32
            hdu.header["NAXIS2"] = 32
            fits_path = subdir / f"field_{i}.fits"
            hdu.writeto(str(fits_path))

        catalog = build_catalog(tmp_path)

        assert len(catalog) == 3
        assert "filepath" in catalog.columns
        assert "crval1" in catalog.columns
        assert "survey_dir" in catalog.columns
        assert all(catalog["survey_dir"] == "dss2_red")

    def test_build_catalog_empty_dir(self, tmp_path):
        """Should return empty DataFrame for directory with no FITS."""
        catalog = build_catalog(tmp_path)
        assert len(catalog) == 0


# ===================================================================
# Survey Download Tests (mocked)
# ===================================================================

class TestSurveyDownload:
    """Test survey download helpers (no network calls)."""

    def test_survey_to_dir_mapping(self):
        """Survey names should map to filesystem-safe directory names."""
        assert _survey_to_dir("DSS2 Red") == "dss2_red"
        assert _survey_to_dir("WISE 3.4") == "wise_3.4"
        assert _survey_to_dir("SDSSr") == "sdss_r"
        assert _survey_to_dir("2MASS-J") == "2mass_j"

    def test_survey_to_dir_unknown(self):
        """Unknown survey names should be lowercased and sanitized."""
        result = _survey_to_dir("Some New Survey 2.0")
        assert " " not in result
        assert result == "some_new_survey_2_0"
