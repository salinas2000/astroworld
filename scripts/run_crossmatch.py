#!/usr/bin/env python3
"""Cross-match observatory pipeline candidates against SIMBAD & Gaia DR3.

Reads a candidate CSV (from the Observatory Pipeline), queries astronomical
databases to identify known objects, and splits into:
  - candidates_known.csv     (galaxies, stars, AGN — contaminants)
  - candidates_unknown.csv   (no catalog match — possible P9!)

Usage
-----
    uv run python scripts/run_crossmatch.py \
        --input results/observatory_calibrated/observatory_candidates.csv \
        --radius 10.0 \
        --output-dir results/crossmatched

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from astroworld.ml.crossmatch import CatalogMatcher


def main():
    parser = argparse.ArgumentParser(
        description="Cross-match P9 candidates against SIMBAD & Gaia DR3",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to candidate CSV (from Observatory Pipeline)",
    )
    parser.add_argument(
        "--radius", type=float, default=10.0,
        help="Search radius in arcseconds (default: 10.0)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/crossmatched"),
        help="Output directory for filtered CSVs",
    )
    parser.add_argument(
        "--pm-threshold", type=float, default=5.0,
        help="Gaia proper motion threshold in mas/yr (default: 5.0)",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.2,
        help="Pause between API queries in seconds (default: 0.2)",
    )
    args = parser.parse_args()

    # Load candidates
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    n = len(df)

    print("=" * 60)
    print("  CATALOG CROSS-MATCHER — SIMBAD + Gaia DR3")
    print("=" * 60)
    print(f"  Input:        {args.input}")
    print(f"  Candidates:   {n}")
    print(f"  Radius:       {args.radius} arcsec")
    print(f"  PM threshold: {args.pm_threshold} mas/yr")
    print(f"  Rate limit:   {args.rate_limit}s")
    print(f"  Output:       {args.output_dir}")
    print("=" * 60)
    print()

    if n == 0:
        print("No candidates to process.")
        return

    # Run cross-matching
    matcher = CatalogMatcher(
        search_radius_arcsec=args.radius,
        rate_limit_s=args.rate_limit,
        pm_threshold=args.pm_threshold,
    )

    t0 = time.time()
    df_known, df_unknown = matcher.filter_candidates(df, verbose=True)
    elapsed = time.time() - t0

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    known_path = args.output_dir / "candidates_known.csv"
    unknown_path = args.output_dir / "candidates_unknown.csv"

    df_known.to_csv(known_path, index=False)
    df_unknown.to_csv(unknown_path, index=False)

    # Summary
    print()
    print("=" * 60)
    print("  CROSS-MATCH SUMMARY")
    print("=" * 60)
    print(f"  Total candidates:  {n}")
    print(f"  Known objects:     {len(df_known)} (contaminants)")
    print(f"  Unknown (P9?):     {len(df_unknown)}")
    print(f"  Rejection rate:    {len(df_known)/n*100:.1f}%")
    print(f"  Time:              {elapsed:.1f}s ({elapsed/n:.1f}s/candidate)")
    print("-" * 60)

    if len(df_known) > 0:
        print(f"\n  Known object types:")
        otype_counts = df_known["catalog_otype"].value_counts()
        for otype, count in otype_counts.items():
            print(f"    {otype:>20}: {count}")

    print(f"\n  Output files:")
    print(f"    Known:    {known_path}")
    print(f"    Unknown:  {unknown_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
