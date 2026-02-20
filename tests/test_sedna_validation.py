"""
Tests for Sedna positive-control validation pipeline.

Tests the ephemeris query, WCS reprojection, and validation logic.
Network-dependent tests are marked with @pytest.mark.network and
skipped by default. Unit tests use mocks.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from astropy.io import fits


# ---------------------------------------------------------------------------
# Mock data for JPL Horizons responses
# ---------------------------------------------------------------------------

def _mock_ephemerides_table():
    """Create a mock astroquery Table matching Horizons .ephemerides()."""
    from astropy.table import Table

    return Table({
        "datetime_jd": [2449200.0, 2453553.0],
        "RA": [56.123, 56.456],
        "DEC": [7.234, 7.567],
        "V": [20.8, 20.9],
        "delta": [85.1, 84.2],
        "r": [86.3, 85.5],
        "RA_rate": [-0.012, -0.011],
        "DEC_rate": [0.003, 0.004],
    })


def _make_wcs_header(
    ra_center: float = 56.0,
    dec_center: float = 7.0,
    pixel_scale_arcsec: float = 1.7,
    nx: int = 512,
    ny: int = 512,
) -> dict:
    """Create a minimal WCS header dict."""
    scale_deg = pixel_scale_arcsec / 3600.0
    return {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": ra_center,
        "CRVAL2": dec_center,
        "CRPIX1": (nx + 1) / 2.0,
        "CRPIX2": (ny + 1) / 2.0,
        "CDELT1": -scale_deg,
        "CDELT2": scale_deg,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "RADESYS": "FK5",
        "EQUINOX": 2000.0,
    }


# ===========================================================================
# Tests: Ephemeris Module
# ===========================================================================


class TestFetchSkyEphemeris:
    """Test the JPL Horizons sky-plane ephemeris client."""

    @patch("astroworld.imaging.ephemeris.Horizons")
    def test_returns_expected_columns(self, MockHorizons):
        """fetch_sky_ephemeris returns DataFrame with required columns."""
        mock_obj = MagicMock()
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()
        MockHorizons.return_value = mock_obj

        from astroworld.imaging.ephemeris import fetch_sky_ephemeris

        df = fetch_sky_ephemeris("90377", epochs=[2449200.0])

        expected_cols = {
            "datetime_jd", "ra_deg", "dec_deg", "V_mag",
            "delta_au", "r_au", "ra_rate_arcsec_hr",
            "dec_rate_arcsec_hr",
        }
        assert expected_cols.issubset(set(df.columns))

    @patch("astroworld.imaging.ephemeris.Horizons")
    def test_ra_dec_values_reasonable(self, MockHorizons):
        """RA and Dec values are within valid ranges."""
        mock_obj = MagicMock()
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()
        MockHorizons.return_value = mock_obj

        from astroworld.imaging.ephemeris import fetch_sky_ephemeris

        df = fetch_sky_ephemeris("90377", epochs=[2449200.0])

        assert all(0 <= df["ra_deg"]) and all(df["ra_deg"] <= 360)
        assert all(-90 <= df["dec_deg"]) and all(df["dec_deg"] <= 90)

    @patch("astroworld.imaging.ephemeris.Horizons")
    def test_smallbody_id_type_for_sedna(self, MockHorizons):
        """Sedna (90377) should use id_type='smallbody'."""
        mock_obj = MagicMock()
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()
        MockHorizons.return_value = mock_obj

        from astroworld.imaging.ephemeris import fetch_sky_ephemeris

        fetch_sky_ephemeris("90377", epochs=[2449200.0])

        MockHorizons.assert_called_once()
        call_kwargs = MockHorizons.call_args
        assert call_kwargs.kwargs.get("id_type") == "smallbody" or \
            call_kwargs[1].get("id_type") == "smallbody"

    @patch("astroworld.imaging.ephemeris.Horizons")
    def test_geocentric_location_default(self, MockHorizons):
        """Default location should be geocentric ('500')."""
        mock_obj = MagicMock()
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()
        MockHorizons.return_value = mock_obj

        from astroworld.imaging.ephemeris import fetch_sky_ephemeris

        fetch_sky_ephemeris("90377", epochs=[2449200.0])

        call_kwargs = MockHorizons.call_args
        assert call_kwargs.kwargs.get("location") == "500" or \
            call_kwargs[1].get("location") == "500"


class TestFetchSednaPosition:
    """Test the Sedna convenience wrapper."""

    @patch("astroworld.imaging.ephemeris.Horizons")
    def test_returns_expected_keys(self, MockHorizons):
        """fetch_sedna_position returns dict with all required keys."""
        mock_obj = MagicMock()
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()
        MockHorizons.return_value = mock_obj

        from astroworld.imaging.ephemeris import fetch_sedna_position

        pos = fetch_sedna_position(2449200.0)

        expected_keys = {
            "ra_deg", "dec_deg", "V_mag", "delta_au",
            "r_au", "ra_rate_arcsec_hr", "dec_rate_arcsec_hr",
        }
        assert expected_keys == set(pos.keys())

    @patch("astroworld.imaging.ephemeris.Horizons")
    def test_values_are_floats(self, MockHorizons):
        """All returned values should be Python floats."""
        mock_obj = MagicMock()
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()
        MockHorizons.return_value = mock_obj

        from astroworld.imaging.ephemeris import fetch_sedna_position

        pos = fetch_sedna_position(2449200.0)

        for key, val in pos.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"


class TestComputeMotion:
    """Test inter-epoch motion computation."""

    @patch("astroworld.imaging.ephemeris.Horizons")
    def test_positive_separation(self, MockHorizons):
        """Different epochs should give positive separation."""
        mock_obj = MagicMock()
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()
        MockHorizons.return_value = mock_obj

        from astroworld.imaging.ephemeris import compute_inter_epoch_motion

        motion = compute_inter_epoch_motion(
            "90377", 2449200.0, 2453553.0,
        )

        assert motion["separation_arcsec"] > 0
        assert motion["separation_pixels"] > 0

    @patch("astroworld.imaging.ephemeris.Horizons")
    def test_returns_expected_keys(self, MockHorizons):
        """Motion dict has all expected keys."""
        mock_obj = MagicMock()
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()
        MockHorizons.return_value = mock_obj

        from astroworld.imaging.ephemeris import compute_inter_epoch_motion

        motion = compute_inter_epoch_motion(
            "90377", 2449200.0, 2453553.0,
        )

        expected = {
            "ra1_deg", "dec1_deg", "ra2_deg", "dec2_deg",
            "delta_ra_arcsec", "delta_dec_arcsec",
            "separation_arcsec", "separation_pixels", "pa_deg",
        }
        assert expected == set(motion.keys())

    @patch("astroworld.imaging.ephemeris.Horizons")
    def test_pixel_scale_affects_pixels(self, MockHorizons):
        """Larger pixel scale = fewer pixel displacement."""
        mock_obj = MagicMock()
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()
        MockHorizons.return_value = mock_obj

        from astroworld.imaging.ephemeris import compute_inter_epoch_motion

        motion_fine = compute_inter_epoch_motion(
            "90377", 2449200.0, 2453553.0, pixel_scale_arcsec=0.25,
        )
        # Re-mock for second call
        mock_obj.ephemerides.return_value = _mock_ephemerides_table()

        motion_coarse = compute_inter_epoch_motion(
            "90377", 2449200.0, 2453553.0, pixel_scale_arcsec=1.7,
        )

        assert motion_fine["separation_pixels"] > motion_coarse["separation_pixels"]
        # Arcsec should be the same
        assert abs(
            motion_fine["separation_arcsec"]
            - motion_coarse["separation_arcsec"]
        ) < 0.01


class TestIdTypeInference:
    """Test _infer_id_type helper."""

    def test_planet_id_returns_none(self):
        from astroworld.imaging.ephemeris import _infer_id_type
        assert _infer_id_type(5) is None  # Jupiter
        assert _infer_id_type(399) is None  # Earth

    def test_sedna_returns_smallbody(self):
        from astroworld.imaging.ephemeris import _infer_id_type
        assert _infer_id_type("90377") == "smallbody"

    def test_named_body_returns_smallbody(self):
        from astroworld.imaging.ephemeris import _infer_id_type
        assert _infer_id_type("Sedna") == "smallbody"
        assert _infer_id_type("2012 VP113") == "smallbody"


# ===========================================================================
# Tests: Reprojection Module
# ===========================================================================


class TestBuildCommonWCS:
    """Test WCS header construction."""

    def test_header_has_wcs_keys(self):
        from astroworld.imaging.reprojection import build_common_wcs

        header = build_common_wcs(
            ra_center=56.0, dec_center=7.0,
            pixel_scale_arcsec=1.7,
            shape=(512, 512),
        )

        assert header["CTYPE1"] == "RA---TAN"
        assert header["CTYPE2"] == "DEC--TAN"
        assert header["NAXIS1"] == 512
        assert header["NAXIS2"] == 512
        assert abs(header["CRVAL1"] - 56.0) < 1e-6
        assert abs(header["CRVAL2"] - 7.0) < 1e-6

    def test_pixel_scale_correct(self):
        from astroworld.imaging.reprojection import build_common_wcs

        header = build_common_wcs(
            ra_center=56.0, dec_center=7.0,
            pixel_scale_arcsec=1.0,
            shape=(256, 256),
        )

        expected_deg = 1.0 / 3600.0
        assert abs(abs(header["CDELT1"]) - expected_deg) < 1e-10
        assert abs(header["CDELT2"] - expected_deg) < 1e-10

    def test_shape_matches_request(self):
        from astroworld.imaging.reprojection import build_common_wcs

        header = build_common_wcs(
            ra_center=0.0, dec_center=0.0,
            pixel_scale_arcsec=2.0,
            shape=(128, 256),
        )

        assert header["NAXIS1"] == 256
        assert header["NAXIS2"] == 128


class TestReprojectToCommonFrame:
    """Test cross-survey WCS alignment."""

    def test_output_shapes_match(self):
        """Both reprojected images have identical shape."""
        from astroworld.imaging.reprojection import reproject_to_common_frame

        img_1 = np.random.RandomState(42).normal(100, 10, (128, 128))
        hdr_1 = _make_wcs_header(nx=128, ny=128, pixel_scale_arcsec=1.7)

        img_2 = np.random.RandomState(43).normal(200, 20, (64, 64))
        hdr_2 = _make_wcs_header(
            ra_center=56.01, dec_center=7.01,
            nx=64, ny=64, pixel_scale_arcsec=0.4,
        )

        r1, r2, out_hdr = reproject_to_common_frame(
            img_1, hdr_1, img_2, hdr_2,
            target_shape=(128, 128),
        )

        assert r1.shape == r2.shape == (128, 128)

    def test_same_wcs_is_near_identity(self):
        """Reprojecting onto the same WCS returns similar values."""
        from astroworld.imaging.reprojection import reproject_to_common_frame

        rng = np.random.RandomState(42)
        img = rng.normal(100, 10, (64, 64)).astype(np.float64)
        hdr = _make_wcs_header(nx=64, ny=64)

        r1, r2, _ = reproject_to_common_frame(
            img, hdr, img, hdr,
            target_shape=(64, 64),
        )

        # Central region should be close to original
        center = slice(16, 48)
        corr = np.corrcoef(r1[center, center].ravel(),
                           img[center, center].ravel())[0, 1]
        assert corr > 0.9, f"Correlation too low: {corr}"

    def test_no_nan_in_output(self):
        """NaN values should be replaced with 0."""
        from astroworld.imaging.reprojection import reproject_to_common_frame

        img = np.ones((64, 64))
        hdr = _make_wcs_header(nx=64, ny=64)

        r1, r2, _ = reproject_to_common_frame(
            img, hdr, img, hdr,
            target_shape=(64, 64),
        )

        assert not np.any(np.isnan(r1))
        assert not np.any(np.isnan(r2))

    def test_custom_pixel_scale(self):
        """Can override target pixel scale."""
        from astroworld.imaging.reprojection import reproject_to_common_frame

        img = np.ones((64, 64))
        hdr = _make_wcs_header(nx=64, ny=64, pixel_scale_arcsec=1.7)

        r1, r2, out_hdr = reproject_to_common_frame(
            img, hdr, img, hdr,
            target_shape=(128, 128),
            target_pixel_scale_arcsec=0.5,
        )

        assert r1.shape == (128, 128)
        expected_deg = 0.5 / 3600.0
        assert abs(abs(out_hdr["CDELT1"]) - expected_deg) < 1e-10


class TestPixelScaleExtraction:
    """Test pixel scale extraction from headers."""

    def test_cdelt_format(self):
        from astroworld.imaging.reprojection import _get_pixel_scale

        hdr = {"CDELT1": -0.000472222, "CDELT2": 0.000472222}
        scale = _get_pixel_scale(hdr)
        assert abs(scale - 1.7) < 0.01

    def test_cd_matrix_format(self):
        from astroworld.imaging.reprojection import _get_pixel_scale

        scale_deg = 0.4 / 3600.0  # SDSS-like
        hdr = {"CD1_1": -scale_deg, "CD1_2": 0.0}
        scale = _get_pixel_scale(hdr)
        assert abs(scale - 0.4) < 0.01

    def test_fallback_for_missing(self):
        from astroworld.imaging.reprojection import _get_pixel_scale

        scale = _get_pixel_scale({})
        assert abs(scale - 1.7) < 0.01  # DSS2 default


# ===========================================================================
# Tests: Validation Pipeline Logic
# ===========================================================================


class TestTier1Logic:
    """Test the Tier 1 cutout validation logic."""

    def test_asymmetric_source_detection(self):
        """Siamese should detect a source present in one image but not other."""
        pytest.importorskip("torch")
        from astroworld.ml.model import P9SiameseDetector
        from astroworld.ml.inference import P9Searcher

        model = P9SiameseDetector()
        model.eval()

        searcher = P9Searcher(model, patch_size=64, overlap=8, threshold=0.3)

        rng = np.random.RandomState(42)
        # Reference: uniform noise
        img_ref = rng.normal(100, 10, (128, 128)).astype(np.float32)
        # Target: same noise + bright source at center
        img_target = img_ref.copy()
        y, x = np.ogrid[-64:64, -64:64]
        gaussian = 500 * np.exp(-(x**2 + y**2) / (2 * 3**2))
        img_target += gaussian.astype(np.float32)

        result = searcher.search_from_arrays(img_ref, img_target)

        # At least one patch should have elevated probability
        max_prob = result.heatmap.max()
        assert max_prob > 0.1, f"Expected signal, got max_prob={max_prob:.4f}"

    def test_symmetric_no_false_positive(self):
        """Identical images should produce low probabilities."""
        pytest.importorskip("torch")
        from astroworld.ml.model import P9SiameseDetector
        from astroworld.ml.inference import P9Searcher

        model = P9SiameseDetector()
        model.eval()

        searcher = P9Searcher(model, patch_size=64, overlap=8, threshold=0.95)

        rng = np.random.RandomState(42)
        img = rng.normal(100, 10, (128, 128)).astype(np.float32)

        result = searcher.search_from_arrays(img, img)

        # Self-pair should produce no detections above threshold
        assert len(result.patch_detections) == 0
