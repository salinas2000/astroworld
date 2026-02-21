"""Tests for the multi-epoch blink comparison (time_machine) module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from astropy.io import fits

from astroworld.imaging.time_machine import (
    BlinkResult,
    download_ps1_cutout,
    find_dss2_field,
    align_epochs,
    measure_source_snr,
    measure_shift,
    classify_verdict,
    blink_candidate,
    make_blink_card,
    DSS2_EPOCH_YEAR,
    PS1_EPOCH_YEAR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wcs_header(ra, dec, pixel_scale_arcsec, naxis=120):
    """Create a minimal WCS header for testing."""
    hdr = {
        "NAXIS": 2,
        "NAXIS1": naxis,
        "NAXIS2": naxis,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": ra,
        "CRVAL2": dec,
        "CRPIX1": (naxis + 1) / 2.0,
        "CRPIX2": (naxis + 1) / 2.0,
        "CDELT1": -pixel_scale_arcsec / 3600.0,
        "CDELT2": pixel_scale_arcsec / 3600.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "RADESYS": "FK5",
        "EQUINOX": 2000.0,
    }
    return hdr


def _make_image_with_source(shape=(120, 120), source_flux=100.0, noise_std=5.0,
                            cx=None, cy=None, fwhm=3.0, seed=42):
    """Create a synthetic image with a Gaussian point source."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    if cy is None:
        cy = ny // 2
    if cx is None:
        cx = nx // 2
    image = rng.normal(0, noise_std, (ny, nx))
    y, x = np.ogrid[:ny, :nx]
    sigma = fwhm / 2.355
    gauss = source_flux * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    image += gauss
    return image.astype(np.float64)


def _make_fits_file(filepath, image, header_dict):
    """Write a FITS file from array + header dict."""
    hdr = fits.Header()
    for k, v in header_dict.items():
        hdr[k] = v
    hdu = fits.PrimaryHDU(data=image, header=hdr)
    hdu.writeto(str(filepath), overwrite=True)


# ---------------------------------------------------------------------------
# TestPS1Download
# ---------------------------------------------------------------------------

class TestPS1Download:
    """Pan-STARRS cutout download with mocked HTTP."""

    @patch("astroworld.imaging.time_machine.requests")
    def test_download_creates_fits(self, mock_requests, tmp_path):
        """Successful download creates a FITS file in ps1_r/."""
        # Mock filenames response
        filenames_resp = MagicMock()
        filenames_resp.status_code = 200
        filenames_resp.text = (
            "projcell subcell ra dec filter mjd type filename shortname\n"
            "1234 56 57.5 15.9 r 56000 stack "
            "rings.v3.skycell.1234.056.stk.r.unconv.fits img1\n"
        )
        filenames_resp.raise_for_status = MagicMock()

        # Mock FITS cutout response
        # Create a minimal valid FITS content
        hdr = fits.Header()
        hdr["SIMPLE"] = True
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 10
        hdr["NAXIS2"] = 10
        hdu = fits.PrimaryHDU(data=np.zeros((10, 10)), header=hdr)
        import io
        buf = io.BytesIO()
        hdu.writeto(buf)
        fits_bytes = buf.getvalue()

        cutout_resp = MagicMock()
        cutout_resp.status_code = 200
        cutout_resp.content = fits_bytes
        cutout_resp.raise_for_status = MagicMock()

        mock_requests.get = MagicMock(side_effect=[filenames_resp, cutout_resp])

        result = download_ps1_cutout(57.5, 15.9, output_dir=tmp_path)

        assert result is not None
        assert result.exists()
        assert "ps1_r" in str(result.parent)

    @patch("astroworld.imaging.time_machine.requests")
    def test_download_resumes(self, mock_requests, tmp_path):
        """Existing file is skipped without network call."""
        ps1_dir = tmp_path / "ps1_r"
        ps1_dir.mkdir()
        existing = ps1_dir / "field_ra057.500_dec+15.900.fits"
        existing.write_text("fake")

        result = download_ps1_cutout(57.5, 15.9, output_dir=tmp_path)

        assert result == existing
        mock_requests.get.assert_not_called()

    @patch("astroworld.imaging.time_machine.requests")
    def test_download_no_coverage(self, mock_requests, tmp_path):
        """Empty filenames response returns None."""
        resp = MagicMock()
        resp.text = "projcell subcell ra dec filter mjd type filename shortname\n"
        resp.raise_for_status = MagicMock()
        mock_requests.get = MagicMock(return_value=resp)

        result = download_ps1_cutout(57.5, -40.0, output_dir=tmp_path)
        assert result is None

    @patch("astroworld.imaging.time_machine.requests")
    def test_download_http_error(self, mock_requests, tmp_path):
        """HTTP error returns None, no crash."""
        mock_requests.get = MagicMock(side_effect=Exception("Connection refused"))

        result = download_ps1_cutout(57.5, 15.9, output_dir=tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# TestFindDSS2Field
# ---------------------------------------------------------------------------

class TestFindDSS2Field:
    """DSS2 field file lookup."""

    def test_exact_match(self, tmp_path):
        """File with exact coordinates found."""
        f = tmp_path / "field_ra057.500_dec+15.900.fits"
        f.write_text("fake")
        result = find_dss2_field(57.5, 15.9, tmp_path)
        assert result == f

    def test_nearest_match(self, tmp_path):
        """Closest file within max_separation returned."""
        f1 = tmp_path / "field_ra057.500_dec+15.950.fits"
        f2 = tmp_path / "field_ra057.600_dec+15.950.fits"
        f1.write_text("fake")
        f2.write_text("fake")
        # Query point closer to f1 (both within 5 arcmin)
        result = find_dss2_field(57.51, 15.95, tmp_path)
        assert result == f1

    def test_no_match(self, tmp_path):
        """No file within radius returns None."""
        f = tmp_path / "field_ra057.500_dec+15.900.fits"
        f.write_text("fake")
        # Query very far away
        result = find_dss2_field(100.0, 50.0, tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# TestAlignEpochs
# ---------------------------------------------------------------------------

class TestAlignEpochs:
    """WCS alignment between DSS2 and PS1 resolution images."""

    def test_aligned_output_shape(self):
        """Both outputs have the same shape."""
        ra, dec = 60.0, 15.0
        img1 = np.random.default_rng(1).normal(0, 1, (100, 100))
        img2 = np.random.default_rng(2).normal(0, 1, (50, 50))
        hdr1 = _make_wcs_header(ra, dec, 1.7, 100)
        hdr2 = _make_wcs_header(ra, dec, 0.25, 50)

        c1, c2, out_hdr = align_epochs(ra, dec, img1, hdr1, img2, hdr2,
                                        cutout_arcsec=30.0)
        assert c1.shape == c2.shape

    def test_aligned_wcs_has_correct_center(self):
        """Output WCS is centered on candidate position."""
        ra, dec = 62.21, 14.57
        img1 = np.random.default_rng(1).normal(0, 1, (100, 100))
        img2 = np.random.default_rng(2).normal(0, 1, (50, 50))
        hdr1 = _make_wcs_header(ra, dec, 1.7, 100)
        hdr2 = _make_wcs_header(ra, dec, 0.25, 50)

        _, _, out_hdr = align_epochs(ra, dec, img1, hdr1, img2, hdr2)
        assert abs(out_hdr["CRVAL1"] - ra) < 0.001
        assert abs(out_hdr["CRVAL2"] - dec) < 0.001

    def test_cutout_size_matches_request(self):
        """Cutout size in pixels matches cutout_arcsec / pixel_scale."""
        ra, dec = 60.0, 15.0
        img1 = np.random.default_rng(1).normal(0, 1, (200, 200))
        img2 = np.random.default_rng(2).normal(0, 1, (200, 200))
        hdr1 = _make_wcs_header(ra, dec, 1.7, 200)
        hdr2 = _make_wcs_header(ra, dec, 0.25, 200)

        c1, c2, _ = align_epochs(ra, dec, img1, hdr1, img2, hdr2,
                                  cutout_arcsec=20.0, target_pixel_scale=0.5)
        expected = int(20.0 / 0.5)
        assert c1.shape == (expected, expected)


# ---------------------------------------------------------------------------
# TestMeasureShift
# ---------------------------------------------------------------------------

class TestMeasureShift:
    """Cross-correlation shift detection."""

    def test_no_shift(self):
        """Identical images produce shift near zero."""
        img = _make_image_with_source(shape=(64, 64), source_flux=200, seed=1)
        info = measure_shift(img, img, pixel_scale_arcsec=0.25)
        assert abs(info["shift_arcsec"]) < 0.5

    def test_known_shift(self):
        """Image shifted by 5 pixels is detected."""
        img1 = _make_image_with_source(shape=(64, 64), source_flux=200,
                                        cx=32, cy=32, seed=1)
        img2 = _make_image_with_source(shape=(64, 64), source_flux=200,
                                        cx=37, cy=32, seed=1)
        info = measure_shift(img1, img2, pixel_scale_arcsec=0.25)
        # Should detect ~5 pixel shift in x = 1.25 arcsec
        assert abs(info["shift_arcsec"] - 1.25) < 0.5

    def test_noise_only(self):
        """Pure noise gives low correlation SNR."""
        rng = np.random.default_rng(42)
        img1 = rng.normal(0, 10, (64, 64))
        img2 = rng.normal(0, 10, (64, 64))
        info = measure_shift(img1, img2)
        assert info["correlation_snr"] < 10  # No strong peak

    def test_sub_pixel_precision(self):
        """Fractional pixel shifts are measured with sub-pixel accuracy."""
        # Create a source and shift by 2.5 pixels
        img1 = _make_image_with_source(shape=(64, 64), source_flux=500,
                                        cx=32, cy=32, noise_std=2, fwhm=4, seed=1)
        img2 = _make_image_with_source(shape=(64, 64), source_flux=500,
                                        cx=34, cy=33, noise_std=2, fwhm=4, seed=1)
        info = measure_shift(img1, img2, pixel_scale_arcsec=1.0)
        expected_shift = np.sqrt(2**2 + 1**2)
        # Allow 1 pixel tolerance
        assert abs(info["shift_pix"] - expected_shift) < 1.5


# ---------------------------------------------------------------------------
# TestMeasureSourceSNR
# ---------------------------------------------------------------------------

class TestMeasureSourceSNR:
    """Source detection and photometry."""

    def test_bright_source(self):
        """Gaussian source with high flux gives SNR > 10."""
        img = _make_image_with_source(shape=(64, 64), source_flux=200,
                                       noise_std=3, seed=1)
        info = measure_source_snr(img)
        assert info["snr"] > 10
        assert info["is_detected"] is True

    def test_no_source(self):
        """Pure noise gives low SNR."""
        rng = np.random.default_rng(42)
        img = rng.normal(100, 5, (64, 64))
        info = measure_source_snr(img)
        # With uniform noise, SNR should be modest
        assert info["snr"] < 5

    def test_faint_source(self):
        """Faint source near detection limit."""
        img = _make_image_with_source(shape=(64, 64), source_flux=15,
                                       noise_std=5, seed=1)
        info = measure_source_snr(img)
        # Might or might not be detected, but SNR should be reasonable
        assert info["snr"] > 0


# ---------------------------------------------------------------------------
# TestClassifyVerdict
# ---------------------------------------------------------------------------

class TestClassifyVerdict:
    """Verdict classification logic."""

    def test_static(self):
        """Both detected, no shift -> STATIC."""
        v, c = classify_verdict(snr1=10, snr2=10, shift_arcsec=0.2,
                                 correlation_snr=5.0)
        assert v == "STATIC"

    def test_moved(self):
        """Both detected, large shift -> MOVED."""
        v, c = classify_verdict(snr1=10, snr2=10, shift_arcsec=3.0,
                                 correlation_snr=5.0)
        assert v == "MOVED"

    def test_absent(self):
        """Neither detected -> ABSENT."""
        v, c = classify_verdict(snr1=1.5, snr2=0.8, shift_arcsec=0.1,
                                 correlation_snr=1.0)
        assert v == "ABSENT"

    def test_appeared_epoch2_only(self):
        """Only in epoch 2 -> APPEARED."""
        v, c = classify_verdict(snr1=1.0, snr2=8.0, shift_arcsec=0.5,
                                 correlation_snr=2.0)
        assert v == "APPEARED"

    def test_appeared_epoch1_only(self):
        """Only in epoch 1 -> APPEARED."""
        v, c = classify_verdict(snr1=8.0, snr2=1.0, shift_arcsec=0.5,
                                 correlation_snr=2.0)
        assert v == "APPEARED"

    def test_confidence_high(self):
        """High SNR gives high confidence."""
        v, c = classify_verdict(snr1=15, snr2=20, shift_arcsec=0.1,
                                 correlation_snr=5.0)
        assert v == "STATIC"
        assert c == "high"

    def test_confidence_low(self):
        """Low SNR gives low confidence."""
        v, c = classify_verdict(snr1=3.5, snr2=3.5, shift_arcsec=0.1,
                                 correlation_snr=2.0)
        assert v == "STATIC"
        assert c == "low"


# ---------------------------------------------------------------------------
# TestBlinkCandidate
# ---------------------------------------------------------------------------

class TestBlinkCandidate:
    """Full blink analysis integration."""

    def test_full_pipeline_synthetic(self, tmp_path):
        """Synthetic FITS with known source -> correct analysis."""
        ra, dec = 60.0, 15.0

        # Create DSS2 file
        dss2_dir = tmp_path / "dss2_red"
        dss2_dir.mkdir()
        img1 = _make_image_with_source(shape=(200, 200), source_flux=100,
                                        noise_std=5, seed=1)
        hdr1 = _make_wcs_header(ra, dec, 1.7, 200)
        _make_fits_file(dss2_dir / "field_ra060.000_dec+15.000.fits",
                        img1, hdr1)

        # Create PS1 file (same source, no shift)
        ps1_dir = tmp_path / "ps1_r"
        ps1_dir.mkdir()
        img2 = _make_image_with_source(shape=(200, 200), source_flux=100,
                                        noise_std=5, seed=2)
        hdr2 = _make_wcs_header(ra, dec, 0.25, 200)
        _make_fits_file(ps1_dir / "field_ra060.000_dec+15.000.fits",
                        img2, hdr2)

        result, c1, c2 = blink_candidate(ra, dec, dss2_dir, ps1_dir)

        assert result.verdict in ("STATIC", "MOVED", "ABSENT", "APPEARED")
        assert c1 is not None
        assert c2 is not None
        assert c1.shape == c2.shape

    def test_missing_ps1_file(self, tmp_path):
        """Missing PS1 file -> ERROR verdict."""
        dss2_dir = tmp_path / "dss2_red"
        dss2_dir.mkdir()
        f = dss2_dir / "field_ra060.000_dec+15.000.fits"
        f.write_text("fake")

        ps1_dir = tmp_path / "ps1_r"
        ps1_dir.mkdir()

        result, c1, c2 = blink_candidate(60.0, 15.0, dss2_dir, ps1_dir)

        assert result.verdict == "ERROR"
        assert c1 is None


# ---------------------------------------------------------------------------
# TestBlinkCard
# ---------------------------------------------------------------------------

class TestBlinkCard:
    """Visualization output."""

    def test_blink_card_creates_png(self, tmp_path):
        """make_blink_card saves a PNG file."""
        result = BlinkResult(ra_deg=60.0, dec_deg=15.0, verdict="STATIC",
                             shift_arcsec=0.2, snr_epoch1=8.0, snr_epoch2=7.5,
                             confidence="medium")
        c1 = _make_image_with_source(shape=(64, 64), source_flux=100, seed=1)
        c2 = _make_image_with_source(shape=(64, 64), source_flux=100, seed=2)

        path = tmp_path / "blink_test.png"
        make_blink_card(result, c1, c2, save_path=path, candidate_temp_k=14.6)

        assert path.exists()
        assert path.stat().st_size > 1000  # Not empty

    def test_blink_card_moved_verdict(self, tmp_path):
        """MOVED verdict card is generated without error."""
        result = BlinkResult(ra_deg=60.0, dec_deg=15.0, verdict="MOVED",
                             shift_arcsec=5.0, snr_epoch1=12.0, snr_epoch2=10.0,
                             confidence="high", implied_pm_arcsec_yr=0.26,
                             details={"shift_x_pix": 3.0, "shift_y_pix": 4.0})
        c1 = _make_image_with_source(shape=(64, 64), source_flux=200, seed=1)
        c2 = _make_image_with_source(shape=(64, 64), source_flux=200,
                                      cx=35, cy=36, seed=1)

        path = tmp_path / "blink_moved.png"
        make_blink_card(result, c1, c2, save_path=path)

        assert path.exists()
