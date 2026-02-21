"""Tests for the Dust Piercer (IR morphology + NEOWISE PM) module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from astropy.table import Table

from astroworld.imaging.dust_piercer import (
    MorphologyResult,
    NEOWISEDetection,
    ProperMotionResult,
    DustPiercerResult,
    fit_gaussian_2d,
    classify_morphology,
    query_neowise_tap,
    group_by_epoch,
    fit_proper_motion,
    classify_dust_piercer_verdict,
    analyze_candidate,
    make_dust_piercer_card,
    WISE_W2_PSF_FWHM_PIX,
    WISE_W2_PIXEL_SCALE_ARCSEC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wise_header(ra, dec, naxis=64, pixel_scale_arcsec=2.75):
    """Create a minimal WISE W2 WCS header."""
    return {
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


def _make_point_source_cutout(
    shape=(15, 15), flux=100.0, noise_std=5.0,
    fwhm_pix=None, cx=None, cy=None, seed=42,
):
    """Create a cutout with a Gaussian point source (WISE W2 PSF-like)."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    if cy is None:
        cy = ny // 2
    if cx is None:
        cx = nx // 2
    if fwhm_pix is None:
        fwhm_pix = WISE_W2_PSF_FWHM_PIX

    sigma = fwhm_pix / 2.355
    y, x = np.ogrid[:ny, :nx]
    gauss = flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    image = rng.normal(0, noise_std, (ny, nx)) + gauss
    return image.astype(np.float64)


def _make_diffuse_cutout(
    shape=(15, 15), flux=50.0, noise_std=5.0,
    scale_pix=8.0, seed=42,
):
    """Create a cutout with diffuse extended emission (much wider than PSF)."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    y, x = np.ogrid[:ny, :nx]
    sigma = scale_pix
    blob = flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    image = rng.normal(0, noise_std, (ny, nx)) + blob
    return image.astype(np.float64)


def _make_wise_image_with_source(
    ra, dec, naxis=64, flux=100.0, noise_std=5.0, fwhm_pix=None, seed=42,
):
    """Create a full WISE W2 image with a point source at the center."""
    hdr = _make_wise_header(ra, dec, naxis)
    image = _make_point_source_cutout(
        shape=(naxis, naxis), flux=flux, noise_std=noise_std,
        fwhm_pix=fwhm_pix, cx=naxis // 2, cy=naxis // 2, seed=seed,
    )
    return image, hdr


def _make_neowise_detections(
    n_epochs=10, ra0=60.0, dec0=15.0,
    mu_ra_arcsec_yr=0.0, mu_dec_arcsec_yr=0.0,
    scatter_arcsec=0.5, mjd_start=57000.0, epoch_spacing_days=182.0,
    frames_per_epoch=5, seed=42,
):
    """Create synthetic NEOWISE detections with optional proper motion."""
    rng = np.random.default_rng(seed)
    detections = []
    cos_dec = np.cos(np.radians(dec0))

    for i in range(n_epochs):
        mjd_epoch = mjd_start + i * epoch_spacing_days
        t_yr = (mjd_epoch - mjd_start) / 365.25

        # True position at this epoch
        ra_true = ra0 + (mu_ra_arcsec_yr * t_yr / 3600.0) / cos_dec
        dec_true = dec0 + mu_dec_arcsec_yr * t_yr / 3600.0

        for j in range(frames_per_epoch):
            det = NEOWISEDetection(
                mjd=mjd_epoch + rng.uniform(-0.5, 0.5),
                ra_deg=ra_true + rng.normal(0, scatter_arcsec / 3600.0) / cos_dec,
                dec_deg=dec_true + rng.normal(0, scatter_arcsec / 3600.0),
                w2mpro=14.5 + rng.normal(0, 0.1),
                w2sigmpro=0.1,
                qual_frame=10,
                qi_fact=1,
            )
            detections.append(det)

    return detections


# ---------------------------------------------------------------------------
# TestGaussian2DFit
# ---------------------------------------------------------------------------

class TestGaussian2DFit:
    """2D Gaussian fitting on cutouts."""

    def test_fit_point_source(self):
        """Gaussian point source: recovers amplitude and sigma."""
        cutout = _make_point_source_cutout(flux=200, noise_std=3, seed=1)
        result = fit_gaussian_2d(cutout)
        assert result is not None
        assert result["amplitude"] > 50
        # sigma should be near WISE PSF sigma
        expected_sigma = WISE_W2_PSF_FWHM_PIX / 2.355
        assert abs(result["sigma_x"] - expected_sigma) < 1.5
        assert abs(result["sigma_y"] - expected_sigma) < 1.5

    def test_fit_offset_source(self):
        """Source not at center: recovers offset position."""
        cutout = _make_point_source_cutout(flux=200, noise_std=3, cx=9, cy=10, seed=2)
        result = fit_gaussian_2d(cutout)
        assert result is not None
        assert abs(result["x0"] - 9) < 2
        assert abs(result["y0"] - 10) < 2

    def test_fit_noise_only(self):
        """Pure noise: returns None."""
        rng = np.random.default_rng(42)
        cutout = rng.normal(100, 5, (15, 15))
        result = fit_gaussian_2d(cutout)
        # Should return None (no significant source above background)
        # or fit with very small amplitude
        if result is not None:
            assert result["amplitude"] < 20


# ---------------------------------------------------------------------------
# TestClassifyMorphology
# ---------------------------------------------------------------------------

class TestClassifyMorphology:
    """Morphological classification in WISE W2."""

    def test_bright_point_source(self):
        """Clear Gaussian point source is classified as point-like."""
        image, hdr = _make_wise_image_with_source(60.0, 15.0, flux=200, noise_std=3)
        result = classify_morphology(60.0, 15.0, image, hdr)
        assert result.snr > 5
        assert result.is_point_source is True
        # Pointiness near 1.0 (fitted FWHM ~ PSF FWHM)
        assert result.pointiness > 0.5

    def test_diffuse_emission(self):
        """Wide flat emission is classified as diffuse."""
        hdr = _make_wise_header(60.0, 15.0, 200)
        # Create diffuse image — wide blob in large image so edges are clean
        rng = np.random.default_rng(42)
        ny, nx = 200, 200
        cy, cx = 100, 100
        y, x = np.ogrid[:ny, :nx]
        # Wide blob (much larger than PSF): sigma=8 px ~ 22" vs PSF 6.4"
        sigma = 8.0
        blob = 300 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        image = rng.normal(0, 3, (ny, nx)) + blob

        result = classify_morphology(60.0, 15.0, image, hdr)
        assert result.snr > 3
        assert result.is_point_source is False
        # Pointiness should be low (FWHM >> PSF)
        assert result.pointiness < 0.5

    def test_low_snr_source(self):
        """SNR below threshold -> not classified as point source."""
        hdr = _make_wise_header(60.0, 15.0, 64)
        rng = np.random.default_rng(42)
        # Very faint source buried in noise
        image = rng.normal(100, 10, (64, 64))
        image[32, 32] += 5  # barely above noise

        result = classify_morphology(60.0, 15.0, image, hdr)
        assert result.snr < 3
        assert result.is_point_source is False

    def test_elliptical_source(self):
        """Elongated source: ellipticity is computed."""
        hdr = _make_wise_header(60.0, 15.0, 64)
        rng = np.random.default_rng(42)
        ny, nx = 64, 64
        cy, cx = 32, 32
        y, x = np.ogrid[:ny, :nx]
        sigma_x, sigma_y = 1.0, 3.0  # elongated
        blob = 200 * np.exp(-((x - cx) ** 2 / (2 * sigma_x ** 2) +
                               (y - cy) ** 2 / (2 * sigma_y ** 2)))
        image = rng.normal(0, 3, (ny, nx)) + blob

        result = classify_morphology(60.0, 15.0, image, hdr)
        assert result.snr > 3
        assert result.ellipticity > 0.1

    def test_empty_cutout(self):
        """All-noise image handled gracefully."""
        hdr = _make_wise_header(60.0, 15.0, 64)
        rng = np.random.default_rng(42)
        image = rng.normal(100, 5, (64, 64))

        result = classify_morphology(60.0, 15.0, image, hdr)
        assert result.is_point_source is False
        assert result.ra_deg == 60.0


# ---------------------------------------------------------------------------
# TestNEOWISEQuery
# ---------------------------------------------------------------------------

class TestNEOWISEQuery:
    """NEOWISE TAP queries (all mocked)."""

    @patch("astroworld.imaging.dust_piercer._query_neowise_astroquery")
    def test_query_success(self, mock_aq):
        """Mock astroquery returns detections."""
        mock_aq.return_value = [
            NEOWISEDetection(mjd=57000, ra_deg=60.0, dec_deg=15.0,
                             w2mpro=14.5, w2sigmpro=0.1, qual_frame=10, qi_fact=1),
            NEOWISEDetection(mjd=57182, ra_deg=60.001, dec_deg=15.001,
                             w2mpro=14.4, w2sigmpro=0.1, qual_frame=10, qi_fact=1),
        ]
        result = query_neowise_tap(60.0, 15.0, rate_limit_sec=0.0)
        assert result is not None
        assert len(result) == 2

    @patch("astroworld.imaging.dust_piercer._query_neowise_astroquery")
    def test_query_no_results(self, mock_aq):
        """Empty table returns empty list."""
        mock_aq.return_value = []
        result = query_neowise_tap(60.0, 15.0, rate_limit_sec=0.0)
        assert result == []

    @patch("astroworld.imaging.dust_piercer._query_neowise_astroquery")
    @patch("astroworld.imaging.dust_piercer._query_neowise_raw_tap")
    def test_query_network_error(self, mock_raw, mock_aq):
        """Both backends fail returns None."""
        mock_aq.return_value = None
        mock_raw.return_value = None
        result = query_neowise_tap(60.0, 15.0, rate_limit_sec=0.0)
        assert result is None


# ---------------------------------------------------------------------------
# TestGroupByEpoch
# ---------------------------------------------------------------------------

class TestGroupByEpoch:
    """Epoch grouping of NEOWISE detections."""

    def test_single_epoch(self):
        """All detections in one window -> one epoch."""
        dets = [
            NEOWISEDetection(mjd=57000.1, ra_deg=60.0, dec_deg=15.0,
                             w2mpro=14.5, w2sigmpro=0.1),
            NEOWISEDetection(mjd=57000.5, ra_deg=60.001, dec_deg=15.001,
                             w2mpro=14.4, w2sigmpro=0.1),
            NEOWISEDetection(mjd=57000.9, ra_deg=59.999, dec_deg=14.999,
                             w2mpro=14.6, w2sigmpro=0.1),
        ]
        epochs = group_by_epoch(dets)
        assert len(epochs) == 1
        assert epochs[0]["n_frames"] == 3

    def test_multiple_epochs(self):
        """Detections 6 months apart -> distinct epochs."""
        dets = _make_neowise_detections(n_epochs=5, frames_per_epoch=3)
        epochs = group_by_epoch(dets)
        assert len(epochs) == 5
        for e in epochs:
            assert e["n_frames"] == 3

    def test_empty_input(self):
        """Empty list returns empty list."""
        assert group_by_epoch([]) == []


# ---------------------------------------------------------------------------
# TestFitProperMotion
# ---------------------------------------------------------------------------

class TestFitProperMotion:
    """Proper motion fitting from multi-epoch data."""

    def test_stationary_source(self):
        """Zero PM input gives near-zero total PM."""
        dets = _make_neowise_detections(
            n_epochs=10, mu_ra_arcsec_yr=0.0, mu_dec_arcsec_yr=0.0,
            scatter_arcsec=0.3,
        )
        epochs = group_by_epoch(dets)
        result = fit_proper_motion(epochs, 60.0, 15.0)
        assert result.n_epochs == 10
        # PM should be near zero (within noise)
        assert result.mu_total_arcsec_yr < 0.3
        assert result.pm_significance < 3.0

    def test_known_proper_motion(self):
        """Inject 0.5"/yr PM and recover it."""
        dets = _make_neowise_detections(
            n_epochs=15, mu_ra_arcsec_yr=0.4, mu_dec_arcsec_yr=0.3,
            scatter_arcsec=0.3, seed=1,
        )
        epochs = group_by_epoch(dets)
        result = fit_proper_motion(epochs, 60.0, 15.0)
        expected_total = np.sqrt(0.4**2 + 0.3**2)
        # Should recover PM within 0.15"/yr
        assert abs(result.mu_total_arcsec_yr - expected_total) < 0.15
        # High significance
        assert result.pm_significance > 3.0

    def test_insufficient_epochs(self):
        """Fewer than 3 epochs -> no fit."""
        dets = _make_neowise_detections(n_epochs=2, frames_per_epoch=3)
        epochs = group_by_epoch(dets)
        result = fit_proper_motion(epochs, 60.0, 15.0)
        assert result.n_epochs == 2
        assert result.mu_total_arcsec_yr == 0.0

    def test_significance_calculation(self):
        """PM significance = mu_total / mu_total_err."""
        dets = _make_neowise_detections(
            n_epochs=20, mu_ra_arcsec_yr=0.5, mu_dec_arcsec_yr=0.0,
            scatter_arcsec=0.2, seed=10,
        )
        epochs = group_by_epoch(dets)
        result = fit_proper_motion(epochs, 60.0, 15.0)
        if result.mu_total_err > 0:
            expected_sig = result.mu_total_arcsec_yr / result.mu_total_err
            assert abs(result.pm_significance - expected_sig) < 0.01


# ---------------------------------------------------------------------------
# TestClassifyVerdict
# ---------------------------------------------------------------------------

class TestClassifyVerdict:
    """Verdict classification logic."""

    def test_point_moving(self):
        """Point source with significant PM -> POINT_MOVING."""
        morph = MorphologyResult(ra_deg=60, dec_deg=15, is_point_source=True,
                                  snr=10, pointiness=0.5)
        pm = ProperMotionResult(ra_deg=60, dec_deg=15, n_epochs=10,
                                 mu_total_arcsec_yr=0.5, mu_total_err=0.05,
                                 pm_significance=10.0)
        v, c = classify_dust_piercer_verdict(morph, pm)
        assert v == "POINT_MOVING"
        assert c == "high"

    def test_point_static(self):
        """Point source, no PM -> POINT_STATIC."""
        morph = MorphologyResult(ra_deg=60, dec_deg=15, is_point_source=True,
                                  snr=10, pointiness=0.5)
        pm = ProperMotionResult(ra_deg=60, dec_deg=15, n_epochs=10,
                                 mu_total_arcsec_yr=0.05, mu_total_err=0.03,
                                 pm_significance=1.5)
        v, c = classify_dust_piercer_verdict(morph, pm)
        assert v == "POINT_STATIC"

    def test_diffuse(self):
        """Diffuse emission -> DIFFUSE."""
        morph = MorphologyResult(ra_deg=60, dec_deg=15, is_point_source=False,
                                  snr=8, pointiness=0.1)
        v, c = classify_dust_piercer_verdict(morph)
        assert v == "DIFFUSE"

    def test_no_source(self):
        """SNR below threshold -> NO_SOURCE."""
        morph = MorphologyResult(ra_deg=60, dec_deg=15, snr=1.5)
        v, c = classify_dust_piercer_verdict(morph)
        assert v == "NO_SOURCE"


# ---------------------------------------------------------------------------
# TestAnalyzeCandidate
# ---------------------------------------------------------------------------

class TestAnalyzeCandidate:
    """Full pipeline integration test."""

    @patch("astroworld.imaging.dust_piercer.query_neowise_tap")
    def test_full_pipeline_point_source(self, mock_neowise):
        """Synthetic point source -> correct analysis pipeline."""
        ra, dec = 60.0, 15.0
        image, hdr = _make_wise_image_with_source(ra, dec, flux=200, noise_std=3)

        # Mock NEOWISE with stationary source
        mock_neowise.return_value = []  # No detections

        result = analyze_candidate(ra, dec, image, hdr, rate_limit_sec=0.0)

        assert result.morphology is not None
        assert result.morphology.is_point_source is True
        assert result.verdict in ("POINT_STATIC", "POINT_MOVING", "ERROR")


# ---------------------------------------------------------------------------
# TestDustPiercerCard
# ---------------------------------------------------------------------------

class TestDustPiercerCard:
    """Visualization output."""

    def test_card_creates_png(self, tmp_path):
        """make_dust_piercer_card saves a PNG file."""
        morph = MorphologyResult(ra_deg=60, dec_deg=15, snr=8,
                                  pointiness=0.2, is_point_source=False)
        result = DustPiercerResult(
            ra_deg=60, dec_deg=15, morphology=morph,
            verdict="DIFFUSE", confidence="high",
        )
        cutout = _make_diffuse_cutout()
        path = tmp_path / "card_test.png"
        make_dust_piercer_card(result, cutout, save_path=path, candidate_temp_k=14.6)
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_card_point_moving(self, tmp_path):
        """POINT_MOVING card with PM data."""
        morph = MorphologyResult(ra_deg=60, dec_deg=15, snr=12,
                                  pointiness=0.5, is_point_source=True)
        pm = ProperMotionResult(
            ra_deg=60, dec_deg=15, n_epochs=10, baseline_years=5.0,
            mu_ra_arcsec_yr=0.4, mu_dec_arcsec_yr=0.3,
            mu_total_arcsec_yr=0.5, mu_ra_err=0.05, mu_dec_err=0.05,
            mu_total_err=0.04, pm_significance=12.5, mean_w2mag=14.5,
            details={"t_mean_mjd": 58000},
        )
        result = DustPiercerResult(
            ra_deg=60, dec_deg=15, morphology=morph, proper_motion=pm,
            verdict="POINT_MOVING", confidence="high",
        )
        cutout = _make_point_source_cutout(flux=200, noise_std=3)
        path = tmp_path / "card_moving.png"
        make_dust_piercer_card(result, cutout, save_path=path)
        assert path.exists()
        assert path.stat().st_size > 1000
