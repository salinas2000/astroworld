"""
Tests for the synthetic source injection pipeline (Phase 7).

Tests PSF extraction, source injection, and training data generation.
All tests use synthetic FITS images (no network access required).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from astropy.io import fits

from astroworld.imaging.psf_extraction import (
    estimate_background,
    detect_stars,
    extract_psf,
    make_gaussian_psf,
    PSFResult,
    HAS_PHOTUTILS,
)
from astroworld.imaging.synthetic_injection import (
    radec_to_pixel,
    pixel_to_radec,
    estimate_zero_point,
    magnitude_to_flux,
    inject_source,
    inject_p9_source,
    InjectionResult,
    DEFAULT_ZERO_POINT,
)
from astroworld.imaging.training_generator import (
    generate_single_example,
    generate_training_set,
    generate_multi_epoch_pair,
    TrainingExample,
)

requires_photutils = pytest.mark.skipif(
    not HAS_PHOTUTILS,
    reason="photutils not installed",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_fits(
    path: Path,
    nx: int = 512,
    ny: int = 512,
    bkg_mean: float = 1000.0,
    bkg_std: float = 50.0,
    n_stars: int = 20,
    star_flux_range: tuple = (5000.0, 50000.0),
    star_sigma_range: tuple = (1.5, 2.5),
    seed: int = 42,
) -> tuple[Path, list[tuple]]:
    """Create a mock FITS image with WCS, background noise, and fake stars."""
    rng = np.random.default_rng(seed)
    image = rng.normal(bkg_mean, bkg_std, (ny, nx))

    star_positions = []
    for i in range(n_stars):
        sx = int(rng.integers(30, nx - 30))
        sy = int(rng.integers(30, ny - 30))
        flux = float(rng.uniform(star_flux_range[0], star_flux_range[1]))
        sigma = float(rng.uniform(star_sigma_range[0], star_sigma_range[1]))
        yy, xx = np.mgrid[
            max(0, sy - 10) : min(ny, sy + 11),
            max(0, sx - 10) : min(nx, sx + 11),
        ]
        gaussian = flux * np.exp(
            -0.5 * ((xx - sx) ** 2 + (yy - sy) ** 2) / sigma ** 2
        )
        image[
            max(0, sy - 10) : min(ny, sy + 11),
            max(0, sx - 10) : min(nx, sx + 11),
        ] += gaussian
        star_positions.append((sx, sy, flux))

    hdu = fits.PrimaryHDU(image.astype(np.float32))
    hdu.header["CRVAL1"] = 71.3
    hdu.header["CRVAL2"] = 14.5
    hdu.header["CRPIX1"] = 256.5
    hdu.header["CRPIX2"] = 256.5
    hdu.header["CDELT1"] = -0.000326
    hdu.header["CDELT2"] = 0.000326
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["NAXIS1"] = nx
    hdu.header["NAXIS2"] = ny
    hdu.writeto(str(path), overwrite=True)

    return path, star_positions


@pytest.fixture
def mock_fits(tmp_path):
    """Create a single mock FITS image."""
    path = tmp_path / "dss2_red" / "mock_field.fits"
    path.parent.mkdir(parents=True, exist_ok=True)
    return _make_mock_fits(path)


@pytest.fixture
def mock_fits_dir(tmp_path):
    """Create a directory with 3 mock FITS images."""
    survey_dir = tmp_path / "dss2_red"
    survey_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(3):
        path = survey_dir / f"field_{i:02d}.fits"
        _make_mock_fits(path, seed=42 + i)
        paths.append(path)
    return tmp_path


# ---------------------------------------------------------------------------
# TestBackgroundEstimation
# ---------------------------------------------------------------------------

class TestBackgroundEstimation:
    def test_sigma_clip_on_uniform_noise(self):
        """Background estimate should be close to true value."""
        rng = np.random.default_rng(42)
        image = rng.normal(1000.0, 50.0, (512, 512))
        mean, std = estimate_background(image)
        assert abs(mean - 1000.0) < 5.0
        assert abs(std - 50.0) < 10.0

    def test_background_ignores_bright_stars(self):
        """Bright stars should not bias the background estimate."""
        rng = np.random.default_rng(42)
        image = rng.normal(1000.0, 50.0, (512, 512))
        # Add 10 very bright stars
        for _ in range(10):
            x, y = rng.integers(30, 480, 2)
            image[y - 5 : y + 6, x - 5 : x + 6] += 100000.0
        mean, std = estimate_background(image)
        assert abs(mean - 1000.0) < 20.0

    def test_returns_positive_values(self):
        """Background mean and std should both be positive."""
        rng = np.random.default_rng(42)
        image = rng.normal(500.0, 30.0, (256, 256))
        mean, std = estimate_background(image)
        assert mean > 0
        assert std > 0


# ---------------------------------------------------------------------------
# TestStarDetection
# ---------------------------------------------------------------------------

class TestStarDetection:
    @requires_photutils
    def test_detects_known_stars(self, mock_fits):
        """Should detect most injected stars."""
        path, true_stars = mock_fits
        image, header = fits.open(str(path))[0].data, dict(fits.open(str(path))[0].header)
        image = image.astype(np.float64)
        positions, fluxes = detect_stars(image, fwhm_estimate=2.5)
        # Should detect at least half the injected stars
        assert len(positions) >= len(true_stars) // 2

    def test_detection_count_reasonable(self, mock_fits):
        """Detection count should be in reasonable range."""
        path, true_stars = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        positions, fluxes = detect_stars(image, fwhm_estimate=2.5)
        # Should detect between 5 and 200 sources (mock has 20 stars)
        assert 5 <= len(positions) <= 200

    def test_no_stars_in_empty_field(self):
        """Pure noise image should return few or no detections."""
        rng = np.random.default_rng(42)
        image = rng.normal(1000.0, 50.0, (256, 256))
        positions, fluxes = detect_stars(
            image, fwhm_estimate=2.5, threshold_sigma=10.0
        )
        assert len(positions) <= 5  # Very few false detections at 10-sigma

    def test_returns_positions_and_fluxes(self, mock_fits):
        """Output should be correctly shaped arrays."""
        path, _ = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        positions, fluxes = detect_stars(image)
        assert positions.ndim == 2
        if len(positions) > 0:
            assert positions.shape[1] == 2
        assert fluxes.ndim == 1
        assert len(positions) == len(fluxes)


# ---------------------------------------------------------------------------
# TestPSFExtraction
# ---------------------------------------------------------------------------

class TestPSFExtraction:
    def test_extract_psf_returns_result(self, mock_fits):
        """Should return a PSFResult dataclass."""
        path, _ = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        result = extract_psf(image, psf_size=25)
        assert isinstance(result, PSFResult)

    def test_psf_shape_odd_and_correct(self, mock_fits):
        """PSF should have correct odd dimensions."""
        path, _ = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        result = extract_psf(image, psf_size=25)
        assert result.psf_image.shape == (25, 25)
        assert result.psf_size == 25

    def test_psf_normalized_to_unit_peak(self, mock_fits):
        """PSF peak should be 1.0."""
        path, _ = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        result = extract_psf(image, psf_size=25)
        assert abs(result.psf_image.max() - 1.0) < 1e-10

    def test_psf_fwhm_reasonable(self, mock_fits):
        """Measured FWHM should be between 1 and 8 pixels."""
        path, _ = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        result = extract_psf(image, psf_size=25)
        assert 1.0 <= result.fwhm_pixels <= 8.0

    def test_gaussian_fallback_when_few_stars(self):
        """Should use Gaussian fallback when too few stars found."""
        rng = np.random.default_rng(42)
        # Image with only 2 faint stars
        image = rng.normal(1000.0, 50.0, (256, 256))
        for i in range(2):
            x, y = 100 + i * 50, 128
            sigma = 2.0
            yy, xx = np.mgrid[y - 5 : y + 6, x - 5 : x + 6]
            image[y - 5 : y + 6, x - 5 : x + 6] += 2000 * np.exp(
                -0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sigma ** 2
            )
        result = extract_psf(image, psf_size=25, min_stars=10)
        assert result.method == "gaussian_fallback"


# ---------------------------------------------------------------------------
# TestGaussianPSF
# ---------------------------------------------------------------------------

class TestGaussianPSF:
    def test_gaussian_psf_shape(self):
        """Should return correct shape."""
        psf = make_gaussian_psf(3.0, 25)
        assert psf.shape == (25, 25)

    def test_gaussian_psf_peak_normalized(self):
        """Max value should be 1.0."""
        psf = make_gaussian_psf(3.0, 25)
        assert abs(psf.max() - 1.0) < 1e-10

    def test_gaussian_psf_symmetry(self):
        """PSF should be rotationally symmetric (4-fold)."""
        psf = make_gaussian_psf(3.0, 25)
        np.testing.assert_allclose(psf, psf[::-1, :], atol=1e-10)
        np.testing.assert_allclose(psf, psf[:, ::-1], atol=1e-10)


# ---------------------------------------------------------------------------
# TestWCSConversion
# ---------------------------------------------------------------------------

class TestWCSConversion:
    def test_radec_to_pixel_center(self, mock_fits):
        """Image center RA/Dec should map near center pixel."""
        path, _ = mock_fits
        header = dict(fits.open(str(path))[0].header)
        px, py = radec_to_pixel(71.3, 14.5, header)
        # WCS CRPIX is 1-indexed (256.5) but all_world2pix with origin=0
        # returns 0-indexed (~255.5). Allow 2 px tolerance.
        assert abs(px - 255.5) < 2.0
        assert abs(py - 255.5) < 2.0

    def test_radec_to_pixel_offset(self, mock_fits):
        """A 1-CDELT offset should shift by ~1 pixel."""
        path, _ = mock_fits
        header = dict(fits.open(str(path))[0].header)
        cdelt = abs(header["CDELT1"])
        px_center, py_center = radec_to_pixel(71.3, 14.5, header)
        px_off, py_off = radec_to_pixel(71.3, 14.5 + cdelt, header)
        # Dec offset should move ~1 pixel in Y
        assert abs(abs(py_off - py_center) - 1.0) < 0.2

    def test_out_of_bounds_detected(self, mock_fits):
        """Very far-away position should map outside image or give NaN."""
        path, _ = mock_fits
        header = dict(fits.open(str(path))[0].header)
        # A position far from the image center should either raise
        # ValueError, map to pixels far outside, or return NaN
        try:
            px, py = radec_to_pixel(180.0, -60.0, header)
            # WCS may return NaN or wildly out-of-bounds pixels
            out_of_bounds = (
                np.isnan(px) or np.isnan(py)
                or px < -100 or px > 600
                or py < -100 or py > 600
            )
            assert out_of_bounds
        except ValueError:
            pass  # Expected behavior

    def test_pixel_to_radec_roundtrip(self, mock_fits):
        """pixel -> radec -> pixel should roundtrip."""
        path, _ = mock_fits
        header = dict(fits.open(str(path))[0].header)
        ra, dec = pixel_to_radec(200.0, 300.0, header)
        px, py = radec_to_pixel(ra, dec, header)
        assert abs(px - 200.0) < 0.01
        assert abs(py - 300.0) < 0.01


# ---------------------------------------------------------------------------
# TestZeroPoint
# ---------------------------------------------------------------------------

class TestZeroPoint:
    def test_zero_point_positive(self, mock_fits):
        """Estimated zero point should be positive."""
        path, _ = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        header = dict(fits.open(str(path))[0].header)
        zp = estimate_zero_point(image, header)
        assert zp > 0

    def test_magnitude_flux_roundtrip(self):
        """mag -> flux -> mag should roundtrip."""
        zp = 25.0
        mag = 22.0
        flux = magnitude_to_flux(mag, zp)
        # Back: mag = zp - 2.5*log10(flux)
        mag_back = zp - 2.5 * np.log10(flux)
        assert abs(mag_back - mag) < 1e-10

    def test_brighter_magnitude_gives_more_flux(self):
        """Brighter (lower) magnitude should give more flux."""
        zp = 25.0
        flux_bright = magnitude_to_flux(18.0, zp)
        flux_faint = magnitude_to_flux(22.0, zp)
        assert flux_bright > flux_faint

    def test_default_zp_used_when_no_stars(self):
        """Should fall back to DEFAULT_ZERO_POINT for empty images."""
        rng = np.random.default_rng(42)
        image = rng.normal(0.0, 0.1, (64, 64))  # Very low signal
        header = {}
        zp = estimate_zero_point(image, header)
        # Should be a positive finite number
        assert np.isfinite(zp)
        assert zp > 0


# ---------------------------------------------------------------------------
# TestSourceInjection
# ---------------------------------------------------------------------------

class TestSourceInjection:
    def test_inject_adds_flux(self):
        """Injected image sum should be greater than original."""
        rng = np.random.default_rng(42)
        image = rng.normal(1000.0, 50.0, (256, 256))
        psf = make_gaussian_psf(2.5, 25)
        result = inject_source(
            image, psf, 128.0, 128.0, 10000.0,
            background_std=0.0, add_noise=False,
        )
        assert result.sum() > image.sum()

    def test_inject_preserves_original(self):
        """Original image array should be unchanged."""
        rng = np.random.default_rng(42)
        image = rng.normal(1000.0, 50.0, (256, 256))
        original_sum = image.sum()
        psf = make_gaussian_psf(2.5, 25)
        _ = inject_source(image, psf, 128.0, 128.0, 10000.0, add_noise=False)
        assert abs(image.sum() - original_sum) < 1e-6

    def test_inject_at_known_position(self):
        """Pixel at injection position should have higher value."""
        rng = np.random.default_rng(42)
        image = np.ones((256, 256)) * 1000.0
        psf = make_gaussian_psf(2.5, 25)
        result = inject_source(
            image, psf, 100.0, 150.0, 50000.0, add_noise=False
        )
        assert result[150, 100] > image[150, 100]

    def test_inject_p9_source_returns_result(self, mock_fits):
        """Full pipeline should return InjectionResult."""
        path, _ = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        header = dict(fits.open(str(path))[0].header)
        psf_result = extract_psf(image, psf_size=25)
        result = inject_p9_source(
            image, header, psf_result,
            magnitude=22.0,
            random_position=True,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, InjectionResult)
        assert result.image_injected.shape == image.shape
        assert np.isfinite(result.flux_adu)
        assert np.isfinite(result.snr_estimate)

    def test_inject_very_faint_source(self):
        """Very faint source (mag 24) should still complete without error."""
        rng = np.random.default_rng(42)
        image = rng.normal(1000.0, 50.0, (256, 256))
        psf = make_gaussian_psf(2.5, 25)
        # Use a very small flux (faint source)
        result = inject_source(
            image, psf, 128.0, 128.0, 0.01,
            background_std=50.0, add_noise=True, rng=rng,
        )
        assert result.shape == image.shape
        assert np.all(np.isfinite(result))

    def test_subpixel_placement(self):
        """Inject at fractional position should spread flux across pixels."""
        image = np.zeros((256, 256))
        psf = make_gaussian_psf(2.0, 11)
        result = inject_source(
            image, psf, 100.3, 100.7, 10000.0, add_noise=False
        )
        # Peak should not be exactly at (100, 101) -- flux should be spread
        assert result[101, 100] > 0
        assert result[100, 100] > 0  # Flux spread to neighboring pixel


# ---------------------------------------------------------------------------
# TestTrainingGenerator
# ---------------------------------------------------------------------------

class TestTrainingGenerator:
    def test_generate_single_positive(self, mock_fits, tmp_path):
        """Should create a FITS file with injection metadata."""
        path, _ = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        psf_result = extract_psf(image, psf_size=25)
        output_dir = tmp_path / "train_out"

        ex = generate_single_example(
            fits_path=path,
            output_dir=output_dir,
            psf_result=psf_result,
            magnitude=22.0,
            inject=True,
            rng=np.random.default_rng(42),
            example_id="pos_0001",
        )
        assert ex.source_present is True
        assert Path(ex.fits_path).exists()
        assert np.isfinite(ex.magnitude)

    def test_generate_single_negative(self, mock_fits, tmp_path):
        """Should create a clean FITS file."""
        path, _ = mock_fits
        image = fits.open(str(path))[0].data.astype(np.float64)
        psf_result = extract_psf(image, psf_size=25)
        output_dir = tmp_path / "train_out"

        ex = generate_single_example(
            fits_path=path,
            output_dir=output_dir,
            psf_result=psf_result,
            inject=False,
            rng=np.random.default_rng(42),
            example_id="neg_0001",
        )
        assert ex.source_present is False
        assert Path(ex.fits_path).exists()

    def test_generate_training_set(self, mock_fits_dir, tmp_path):
        """Should create output directory with FITS + CSV catalog."""
        output_dir = tmp_path / "training"
        catalog = generate_training_set(
            image_dir=mock_fits_dir,
            output_dir=output_dir,
            n_positive=5,
            n_negative=3,
            magnitude_range=(20.0, 23.0),
            seed=42,
        )
        assert isinstance(catalog, pd.DataFrame)
        assert len(catalog) == 8  # 5 positive + 3 negative
        assert (output_dir / "training_catalog.csv").exists()
        assert (output_dir / "positive").is_dir()
        assert (output_dir / "negative").is_dir()

    def test_catalog_has_expected_columns(self, mock_fits_dir, tmp_path):
        """CSV should have all TrainingExample fields."""
        output_dir = tmp_path / "training"
        catalog = generate_training_set(
            image_dir=mock_fits_dir,
            output_dir=output_dir,
            n_positive=3,
            n_negative=2,
            seed=42,
        )
        expected_cols = [
            "example_id", "fits_path", "source_present",
            "ra_deg", "dec_deg", "pixel_x", "pixel_y",
            "magnitude", "flux_adu", "snr",
            "source_fits", "survey", "psf_fwhm", "zero_point",
        ]
        for col in expected_cols:
            assert col in catalog.columns, f"Missing column: {col}"

    def test_reproducible_with_seed(self, mock_fits_dir, tmp_path):
        """Two runs with same seed should produce identical catalogs."""
        output_1 = tmp_path / "run1"
        output_2 = tmp_path / "run2"

        cat1 = generate_training_set(
            image_dir=mock_fits_dir, output_dir=output_1,
            n_positive=3, n_negative=2, seed=123,
        )
        cat2 = generate_training_set(
            image_dir=mock_fits_dir, output_dir=output_2,
            n_positive=3, n_negative=2, seed=123,
        )

        # Magnitudes and positions should match
        np.testing.assert_allclose(
            cat1["magnitude"].values, cat2["magnitude"].values, rtol=1e-6
        )


# ---------------------------------------------------------------------------
# TestMultiEpoch
# ---------------------------------------------------------------------------

class TestMultiEpoch:
    def test_multi_epoch_pair_positions_differ(self, mock_fits_dir, tmp_path):
        """The two injected positions should differ by expected motion."""
        survey_dir = mock_fits_dir / "dss2_red"
        fits_files = sorted(survey_dir.glob("*.fits"))
        assert len(fits_files) >= 2

        image = fits.open(str(fits_files[0]))[0].data.astype(np.float64)
        psf_result = extract_psf(image, psf_size=25)
        output_dir = tmp_path / "multi"

        ex1, ex2 = generate_multi_epoch_pair(
            fits_path_1=fits_files[0],
            fits_path_2=fits_files[1],
            output_dir=output_dir,
            psf_result=psf_result,
            magnitude=22.0,
            dt_days=30.0,
            rng=np.random.default_rng(42),
        )

        # Positions should differ
        assert ex1.pixel_x != ex2.pixel_x or ex1.pixel_y != ex2.pixel_y

    def test_multi_epoch_same_magnitude(self, mock_fits_dir, tmp_path):
        """Both epochs should have the same magnitude."""
        survey_dir = mock_fits_dir / "dss2_red"
        fits_files = sorted(survey_dir.glob("*.fits"))

        image = fits.open(str(fits_files[0]))[0].data.astype(np.float64)
        psf_result = extract_psf(image, psf_size=25)
        output_dir = tmp_path / "multi"

        ex1, ex2 = generate_multi_epoch_pair(
            fits_path_1=fits_files[0],
            fits_path_2=fits_files[1],
            output_dir=output_dir,
            psf_result=psf_result,
            magnitude=21.5,
            dt_days=30.0,
            rng=np.random.default_rng(42),
        )

        assert abs(ex1.magnitude - ex2.magnitude) < 0.01

    def test_multi_epoch_motion_magnitude(self, mock_fits_dir, tmp_path):
        """Pixel displacement should match expected P9 motion."""
        survey_dir = mock_fits_dir / "dss2_red"
        fits_files = sorted(survey_dir.glob("*.fits"))

        image = fits.open(str(fits_files[0]))[0].data.astype(np.float64)
        psf_result = extract_psf(image, psf_size=25)
        output_dir = tmp_path / "multi"

        dt_days = 30.0
        ex1, ex2 = generate_multi_epoch_pair(
            fits_path_1=fits_files[0],
            fits_path_2=fits_files[1],
            output_dir=output_dir,
            psf_result=psf_result,
            magnitude=22.0,
            dt_days=dt_days,
            rng=np.random.default_rng(42),
        )

        # Expected displacement: 0.013 arcsec/hr * 24 * 30 = 9.36 arcsec
        # Pixel scale ~1.17 arcsec/px -> ~8 pixels
        dx = ex2.pixel_x - ex1.pixel_x
        dy = ex2.pixel_y - ex1.pixel_y
        displacement_px = np.sqrt(dx ** 2 + dy ** 2)
        assert 3.0 < displacement_px < 20.0  # Reasonable range
