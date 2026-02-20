#!/usr/bin/env python3
"""
Run the Observatory Pipeline for Planet 9 detection.

Processes sky fields through a cascading multi-stage detector:
  1. (Optional) Fast optical pre-scan with P9Searcher
  2. Targeted IR download (WISE W1/W2 from SkyView)
  3. Spectral validation (SSTF + MC Dropout + PlanckFilter)
  4. Kinematic validation
  5. Report generation (CSV, heatmaps, summary)

Usage examples
--------------
Single field::

    uv run python scripts/run_observatory_pipeline.py \\
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \\
        --ra 71.3 --dec 14.5

Multiple fields from CSV::

    uv run python scripts/run_observatory_pipeline.py \\
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \\
        --field-file targets.csv

Dual-optical mode with both checkpoints::

    uv run python scripts/run_observatory_pipeline.py \\
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \\
        --optical-checkpoint checkpoints/siamese/best_model.pt \\
        --mode dual_optical \\
        --ra 71.3 --dec 14.5 \\
        --output-dir results/observatory_dual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Observatory Pipeline for Planet 9 detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model checkpoints
    parser.add_argument(
        "--spectral-checkpoint",
        type=Path,
        required=True,
        help="Path to SpectralSiameseNet checkpoint (best_model.pt)",
    )
    parser.add_argument(
        "--optical-checkpoint",
        type=Path,
        default=None,
        help="Path to P9SiameseDetector checkpoint (for dual_optical mode)",
    )

    # Input coordinates
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--ra",
        type=float,
        default=None,
        help="Right ascension in degrees (single field, requires --dec)",
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=None,
        help="Declination in degrees (single field, requires --ra)",
    )
    input_group.add_argument(
        "--field-file",
        type=Path,
        default=None,
        help="CSV file with ra_deg,dec_deg columns (batch mode)",
    )

    # Pipeline configuration
    parser.add_argument(
        "--mode",
        type=str,
        default="spectral_only",
        choices=["spectral_only", "dual_optical"],
        help="Pipeline mode (default: spectral_only)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/observatory"),
        help="Output directory for reports (default: results/observatory)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/survey_images"),
        help="Directory for downloaded survey images (default: data/survey_images)",
    )

    # Thresholds
    parser.add_argument(
        "--spectral-threshold",
        type=float,
        default=0.80,
        help="Spectral detection probability threshold (default: 0.80)",
    )
    parser.add_argument(
        "--optical-threshold",
        type=float,
        default=0.95,
        help="Optical detection threshold for dual_optical mode (default: 0.95)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10,
        help="MC Dropout samples for uncertainty (default: 10)",
    )

    # Filters
    parser.add_argument(
        "--planck-strict",
        action="store_true",
        help="Only keep 'p9_candidate' (exclude cold_object)",
    )
    parser.add_argument(
        "--no-kinematic",
        action="store_true",
        help="Disable kinematic filter",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)",
    )

    # Download settings
    parser.add_argument(
        "--download-radius",
        type=float,
        default=5.0,
        help="Field of view radius in arcminutes (default: 5.0)",
    )
    parser.add_argument(
        "--download-pixels",
        type=int,
        default=512,
        help="Image pixel size (default: 512)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Patch size in pixels (default: 64)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=8,
        help="Patch overlap in pixels (default: 8)",
    )

    return parser.parse_args()


def load_fields(args: argparse.Namespace) -> list[tuple[float, float]]:
    """Load field coordinates from arguments or file.

    Returns
    -------
    List of (ra_deg, dec_deg) tuples.
    """
    if args.ra is not None:
        if args.dec is None:
            print("ERROR: --ra requires --dec", file=sys.stderr)
            sys.exit(1)
        return [(args.ra, args.dec)]

    if args.field_file is not None:
        if not args.field_file.exists():
            print(
                f"ERROR: Field file not found: {args.field_file}",
                file=sys.stderr,
            )
            sys.exit(1)

        df = pd.read_csv(args.field_file)

        if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
            print(
                "ERROR: Field file must have 'ra_deg' and 'dec_deg' columns",
                file=sys.stderr,
            )
            sys.exit(1)

        return list(zip(df["ra_deg"].tolist(), df["dec_deg"].tolist()))

    print(
        "ERROR: Provide either --ra/--dec or --field-file",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    """Run the Observatory Pipeline."""
    args = parse_args()

    # Load field coordinates
    fields = load_fields(args)

    # Build pipeline configuration
    from astroworld.ml.pipeline import (
        ObservatoryPipeline,
        PipelineConfig,
    )

    planck_keep = ["p9_candidate"]
    if not args.planck_strict:
        planck_keep.append("cold_object")

    config = PipelineConfig(
        mode=args.mode,
        optical_threshold=args.optical_threshold,
        spectral_threshold=args.spectral_threshold,
        mc_samples=args.mc_samples,
        planck_classes_keep=planck_keep,
        kinematic_enabled=not args.no_kinematic,
        patch_size=args.patch_size,
        overlap=args.overlap,
        device=args.device,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        download_radius_arcmin=args.download_radius,
        download_pixels=args.download_pixels,
    )

    # Build pipeline
    print(f"Loading spectral model from: {args.spectral_checkpoint}")
    pipeline = ObservatoryPipeline.from_checkpoints(
        spectral_checkpoint=args.spectral_checkpoint,
        optical_checkpoint=args.optical_checkpoint,
        config=config,
    )

    # Run pipeline
    summary = pipeline.run(fields=fields)

    # Exit code: 0 if any candidates found, 1 otherwise
    if summary.total_candidates > 0:
        print(
            f"\n  {summary.total_candidates} candidate(s) found! "
            f"See: {args.output_dir / 'observatory_candidates.csv'}"
        )
    else:
        print("\n  No candidates found.")


if __name__ == "__main__":
    main()
