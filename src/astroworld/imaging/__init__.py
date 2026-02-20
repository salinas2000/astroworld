"""
Imaging pipeline for Planet 9 sky survey operations.

Modules:
  - sky_position: Convert P9 orbital elements to sky coordinates (RA, Dec)
  - survey_download: Download FITS cutouts from astronomical sky surveys
  - fits_store: Organize and catalog downloaded FITS files
  - psf_extraction: Extract empirical PSF from stars in FITS images
  - synthetic_injection: Inject synthetic point sources into FITS images
  - training_generator: Batch generate labeled training data for ML
"""

from astroworld.imaging.sky_position import (
    predict_p9_sky_position,
    predict_p9_track,
)
from astroworld.imaging.survey_download import (
    download_field,
    download_p9_search_grid,
)
from astroworld.imaging.fits_store import (
    build_catalog,
    load_fits_image,
)
from astroworld.imaging.psf_extraction import (
    extract_psf,
    make_gaussian_psf,
    detect_stars,
    estimate_background,
    PSFResult,
)
from astroworld.imaging.synthetic_injection import (
    inject_source,
    inject_p9_source,
    radec_to_pixel,
    magnitude_to_flux,
    estimate_zero_point,
    InjectionResult,
)
from astroworld.imaging.training_generator import (
    generate_training_set,
    generate_single_example,
    generate_multi_epoch_pair,
    TrainingExample,
)

__all__ = [
    # Sky position
    "predict_p9_sky_position",
    "predict_p9_track",
    # Survey download
    "download_field",
    "download_p9_search_grid",
    # FITS store
    "build_catalog",
    "load_fits_image",
    # PSF extraction
    "extract_psf",
    "make_gaussian_psf",
    "detect_stars",
    "estimate_background",
    "PSFResult",
    # Synthetic injection
    "inject_source",
    "inject_p9_source",
    "radec_to_pixel",
    "magnitude_to_flux",
    "estimate_zero_point",
    "InjectionResult",
    # Training generator
    "generate_training_set",
    "generate_single_example",
    "generate_multi_epoch_pair",
    "TrainingExample",
]
