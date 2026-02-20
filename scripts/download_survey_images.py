#!/usr/bin/env python3
"""
Download sky survey images for Planet 9 search.

Downloads FITS cutouts from astronomical surveys (DSS, WISE, SDSS, 2MASS)
around the predicted Planet 9 sky position. Uses Astroworld's orbital
mechanics to compute the precise RA/Dec prediction from the best-fit
parameters (10 M_Earth, 400 AU, M=180 deg aphelion).

Usage:
  # Download at P9's predicted position (single field, default surveys)
  uv run python scripts/download_survey_images.py --target p9

  # Download a search grid (5x5 degree area)
  uv run python scripts/download_survey_images.py --target p9 --grid

  # Custom coordinates
  uv run python scripts/download_survey_images.py --ra 75.0 --dec 20.0

  # Choose specific surveys
  uv run python scripts/download_survey_images.py --target p9 --surveys "DSS2 Red" "WISE 3.4" "SDSSr"

  # Predict P9 sky track over 50 years (no download)
  uv run python scripts/download_survey_images.py --target p9 --track-only
"""

import argparse
from pathlib import Path

import numpy as np

from astroworld.imaging.sky_position import (
    predict_p9_sky_position,
    predict_p9_track,
    get_p9_search_region,
    DEFAULT_P9,
    _J2000_JD,
)
from astroworld.imaging.survey_download import (
    download_field,
    download_p9_search_grid,
    DEFAULT_SURVEYS,
)
from astroworld.imaging.fits_store import build_catalog


def main():
    parser = argparse.ArgumentParser(
        description="Download sky survey images for Planet 9 search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --target p9                       Single field at P9 position
  %(prog)s --target p9 --grid                5x5 degree search grid
  %(prog)s --target p9 --grid --grid-size 3  3x3 degree grid
  %(prog)s --ra 75.0 --dec 20.0              Custom coordinates
  %(prog)s --target p9 --track-only          Print P9 sky track (no download)
  %(prog)s --target p9 --surveys "WISE 3.4"  WISE infrared only
        """,
    )

    # Target selection
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target", choices=["p9"],
        help="Use predicted Planet 9 position",
    )
    target_group.add_argument(
        "--ra", type=float,
        help="Right ascension in degrees (requires --dec)",
    )

    parser.add_argument(
        "--dec", type=float,
        help="Declination in degrees (requires --ra)",
    )

    # Grid options
    parser.add_argument(
        "--grid", action="store_true",
        help="Download a grid of fields around the target",
    )
    parser.add_argument(
        "--grid-size", type=float, default=5.0,
        help="Grid size in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--grid-step", type=float, default=1.0,
        help="Grid step in degrees (default: 1.0)",
    )

    # Survey options
    parser.add_argument(
        "--surveys", nargs="+", default=None,
        help=f"Survey names (default: {DEFAULT_SURVEYS})",
    )
    parser.add_argument(
        "--radius", type=float, default=5.0,
        help="Field of view radius in arcminutes (default: 5.0)",
    )
    parser.add_argument(
        "--pixels", type=int, default=512,
        help="Image size in pixels (default: 512)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/survey_images"),
        help="Output directory (default: data/survey_images)",
    )

    # Track-only mode (no download)
    parser.add_argument(
        "--track-only", action="store_true",
        help="Only compute and display P9 sky track (no download)",
    )

    # Epoch
    parser.add_argument(
        "--epoch-year", type=float, default=2026.0,
        help="Epoch year for P9 position prediction (default: 2026.0)",
    )

    args = parser.parse_args()

    # Validate --ra requires --dec
    if args.ra is not None and args.dec is None:
        parser.error("--ra requires --dec")

    # Compute epoch JD
    epoch_jd = _J2000_JD + (args.epoch_year - 2000.0) * 365.25

    # --- Determine target coordinates ---
    if args.target == "p9":
        print("\n" + "=" * 60)
        print("  ASTROWORLD - PLANET 9 SKY SURVEY PIPELINE")
        print("=" * 60)

        # Predict P9 position
        pos = predict_p9_sky_position(epoch_jd=epoch_jd, **DEFAULT_P9)

        print(f"\n  Best-fit P9: {DEFAULT_P9['mass_earth']} M_Earth "
              f"at {DEFAULT_P9['distance_au']} AU")
        print(f"  Epoch: {args.epoch_year:.1f}")
        print(f"\n  Predicted sky position:")
        print(f"    RA  = {pos['ra_deg']:.4f} deg "
              f"({pos['ra_deg']/15:.4f} h)")
        print(f"    Dec = {pos['dec_deg']:+.4f} deg")
        print(f"    Distance = {pos['distance_au']:.1f} AU")
        print(f"    Angular velocity = {pos['v_angular_arcsec_hr']:.3f} "
              f"arcsec/hr")
        print(f"    Estimated magnitude = {pos['mag_estimate']:.1f}")

        ra_center = pos["ra_deg"]
        dec_center = pos["dec_deg"]

        # Show track if requested
        if args.track_only:
            print("\n  P9 Sky Track (2000-2050):")
            print("  " + "-" * 56)
            track = predict_p9_track(
                start_jd=_J2000_JD,
                end_jd=_J2000_JD + 365.25 * 50,
                n_points=11,
                **DEFAULT_P9,
            )
            for _, row in track.iterrows():
                print(f"    {row['year']:.0f}: "
                      f"RA={row['ra_deg']:.3f} deg "
                      f"Dec={row['dec_deg']:+.3f} deg "
                      f"r={row['distance_au']:.1f} AU "
                      f"mag~{row['mag_estimate']:.1f}")

            # Save track CSV
            args.output_dir.mkdir(parents=True, exist_ok=True)
            track_path = args.output_dir / "p9_track.csv"
            track.to_csv(track_path, index=False, float_format="%.6f")
            print(f"\n  Track saved: {track_path}")
            return

    else:
        ra_center = args.ra
        dec_center = args.dec
        print(f"\n  Target: RA={ra_center:.3f} deg, Dec={dec_center:.3f} deg")

    # --- Download ---
    print(f"\n  Output: {args.output_dir}")
    print(f"  Surveys: {args.surveys or DEFAULT_SURVEYS}")
    print()

    if args.grid:
        catalog = download_p9_search_grid(
            ra_center=ra_center,
            dec_center=dec_center,
            grid_size_deg=args.grid_size,
            grid_step_deg=args.grid_step,
            surveys=args.surveys,
            radius_arcmin=args.radius,
            pixels=args.pixels,
            output_dir=args.output_dir,
        )
    else:
        paths = download_field(
            ra_deg=ra_center,
            dec_deg=dec_center,
            surveys=args.surveys,
            radius_arcmin=args.radius,
            pixels=args.pixels,
            output_dir=args.output_dir,
        )
        print(f"\nDownloaded {len(paths)} files")

    # Build and show catalog
    print("\n--- Catalog Summary ---")
    catalog = build_catalog(args.output_dir)

    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
