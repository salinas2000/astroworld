#!/usr/bin/env python3
"""The Dust Piercer — IR morphological filter + NEOWISE proper motion search.

Stage 1: Morphological pre-filter in WISE W2 (reject diffuse nebulosity)
Stage 2: NEOWISE multi-epoch proper motion search (2014-2024)

Usage
-----
    # From time_machine results (filter ABSENT candidates):
    uv run python scripts/dust_piercer.py \
        --input results/time_machine/time_machine_results.csv \
        --output-dir results/dust_piercer

    # Single candidate:
    uv run python scripts/dust_piercer.py \
        --ra 57.5437 --dec 15.9455 \
        --output-dir results/dust_piercer

    # Morphology-only mode (skip NEOWISE query):
    uv run python scripts/dust_piercer.py \
        --input results/time_machine/time_machine_results.csv \
        --morphology-only

    # Download WISE W2 first, then analyze:
    uv run python scripts/dust_piercer.py \
        --input results/time_machine/time_machine_results.csv \
        --download-w2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from astroworld.imaging.dust_piercer import (
    analyze_candidates,
    make_dust_piercer_card,
    DustPiercerResult,
    MIN_SNR_MORPHOLOGY,
)


def main():
    parser = argparse.ArgumentParser(
        description="The Dust Piercer: IR morphology + NEOWISE motion search",
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=Path,
        help="Path to candidate CSV (with ra_deg, dec_deg columns)",
    )
    input_group.add_argument(
        "--ra", type=float,
        help="Single candidate RA (degrees). Must use with --dec",
    )
    parser.add_argument(
        "--dec", type=float,
        help="Single candidate Dec (degrees). Must use with --ra",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/dust_piercer"),
        help="Output directory (default: results/dust_piercer)",
    )

    # Filters
    parser.add_argument(
        "--verdict-filter", type=str, default=None,
        help="Only analyze candidates with this verdict (e.g., ABSENT)",
    )
    parser.add_argument(
        "--temp-max", type=float, default=None,
        help="Only analyze candidates with temperature_k <= this value",
    )

    # Directories
    parser.add_argument(
        "--w2-dir", type=Path,
        default=Path("data/survey_images/wise_4.6"),
        help="Directory with WISE W2 FITS files",
    )

    # Modes
    parser.add_argument(
        "--morphology-only", action="store_true",
        help="Skip NEOWISE query (Stage 2), only run morphological filter",
    )
    parser.add_argument(
        "--download-w2", action="store_true",
        help="Download WISE W2 images before analysis",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=1.0,
        help="Delay between NEOWISE queries (default: 1.0s)",
    )
    parser.add_argument(
        "--no-cards", action="store_true",
        help="Skip generating analysis card PNGs",
    )

    # Thresholds
    parser.add_argument(
        "--pointiness-threshold", type=float, default=0.3,
        help="Pointiness threshold for point source classification (default: 0.3)",
    )
    parser.add_argument(
        "--pm-min", type=float, default=0.3,
        help="Minimum PM in arcsec/yr to flag as MOVED (default: 0.3)",
    )

    args = parser.parse_args()

    # Validate
    if args.ra is not None and args.dec is None:
        parser.error("--ra requires --dec")

    # Build candidate DataFrame
    if args.input:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        df = pd.read_csv(args.input)
    else:
        df = pd.DataFrame([{"ra_deg": args.ra, "dec_deg": args.dec}])

    # Apply filters
    n_original = len(df)

    if args.verdict_filter and "verdict" in df.columns:
        df = df[df["verdict"] == args.verdict_filter].copy()

    if args.temp_max is not None and "temperature_k" in df.columns:
        df = df[df["temperature_k"] <= args.temp_max].copy()

    n_filtered = len(df)

    # Header
    print()
    print("=" * 64)
    print("  THE DUST PIERCER - IR Morphology + NEOWISE Motion Search")
    print("  Phase 19: Piercing the Taurus Molecular Cloud")
    print("=" * 64)
    print(f"  Candidates:       {n_filtered}", end="")
    if n_filtered != n_original:
        filters = []
        if args.verdict_filter:
            filters.append(f"verdict={args.verdict_filter}")
        if args.temp_max:
            filters.append(f"T<={args.temp_max}K")
        print(f" (filtered from {n_original}: {', '.join(filters)})")
    else:
        print()
    print(f"  WISE W2 dir:      {args.w2_dir}")
    print(f"  Output:           {args.output_dir}")
    print(f"  Pointiness:       >= {args.pointiness_threshold}")
    print(f"  PM threshold:     >= {args.pm_min} arcsec/yr (3-sigma)")
    print(f"  Rate limit:       {args.rate_limit}s")
    if args.morphology_only:
        print(f"  Mode:             MORPHOLOGY ONLY (Stage 1)")
    print("=" * 64)
    print()

    if n_filtered == 0:
        print("No candidates to process.")
        return

    # ---------------------------------------------------------------
    # Step 0 (optional): Download WISE W2 images
    # ---------------------------------------------------------------
    if args.download_w2:
        from astroworld.imaging.survey_download import download_field

        print("STEP 0: Downloading WISE W2 images...")
        print("-" * 64)
        t0 = time.time()
        for i, (_, row) in enumerate(df.iterrows()):
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
            print(f"  [{i+1:3d}/{n_filtered}] RA={ra:.4f} Dec={dec:+.4f}", end=" ")
            paths = download_field(
                ra, dec, surveys=["WISE 4.6"],
                output_dir=args.w2_dir.parent,
                rate_limit_sec=args.rate_limit,
            )
            if paths:
                print(f"-> OK")
            else:
                print(f"-> FAILED")
        print(f"\n  Download complete ({time.time() - t0:.1f}s)\n")

    # ---------------------------------------------------------------
    # Step 1+2: Run Dust Piercer analysis
    # ---------------------------------------------------------------
    stage_name = "Stage 1 (Morphology)" if args.morphology_only else "Stage 1+2 (Morphology + NEOWISE)"
    print(f"Running {stage_name}...")
    print("-" * 64)

    t0_analysis = time.time()
    results = analyze_candidates(
        df,
        w2_dir=args.w2_dir,
        rate_limit_sec=args.rate_limit,
        skip_neowise=args.morphology_only,
        verbose=True,
    )
    t_analysis = time.time() - t0_analysis

    print(f"\n  Analysis complete ({t_analysis:.1f}s)\n")

    # ---------------------------------------------------------------
    # Step 3: Generate cards
    # ---------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir = args.output_dir / "dust_piercer_cards"

    if not args.no_cards:
        print("Generating analysis cards...")
        print("-" * 64)
        cards_dir.mkdir(parents=True, exist_ok=True)

        n_cards = 0
        for i, (res, (_, row)) in enumerate(zip(results, df.iterrows())):
            # Get W2 cutout for visualization
            w2_cutout = None
            if res.morphology and res.morphology.snr >= MIN_SNR_MORPHOLOGY:
                # Try to extract cutout for card
                try:
                    from astroworld.imaging.fits_store import load_fits_image
                    from astroworld.imaging.time_machine import find_dss2_field
                    from astroworld.imaging.reprojection import _sanitize_header
                    from astropy.wcs import WCS
                    import numpy as np

                    w2_path = find_dss2_field(
                        float(row["ra_deg"]), float(row["dec_deg"]),
                        args.w2_dir, max_separation_arcmin=7.0,
                    )
                    if w2_path:
                        img, hdr = load_fits_image(w2_path)
                        clean_hdr = _sanitize_header(hdr)
                        wcs = WCS(clean_hdr)
                        px, py = wcs.world_to_pixel_values(
                            float(row["ra_deg"]), float(row["dec_deg"])
                        )
                        px, py = int(round(float(px))), int(round(float(py)))
                        r = 7
                        ny, nx = img.shape
                        if r <= px < nx - r and r <= py < ny - r:
                            w2_cutout = img[py-r:py+r+1, px-r:px+r+1].astype(np.float64)
                except Exception:
                    pass

            temp_k = float(row["temperature_k"]) if "temperature_k" in row.index else None
            card_name = (
                f"dp_{i+1:03d}_ra{res.ra_deg:.3f}_"
                f"dec{res.dec_deg:+.3f}_{res.verdict}.png"
            )
            card_path = cards_dir / card_name
            make_dust_piercer_card(
                res, w2_cutout, save_path=card_path, candidate_temp_k=temp_k,
            )
            n_cards += 1
            print(f"  [{i+1:3d}] {card_name}")

        print(f"\n  Generated {n_cards} cards in {cards_dir}\n")

    # ---------------------------------------------------------------
    # Step 4: Save results CSV
    # ---------------------------------------------------------------
    rows = []
    for i, (res, (_, row)) in enumerate(zip(results, df.iterrows())):
        r = {
            "candidate_id": i + 1,
            "ra_deg": res.ra_deg,
            "dec_deg": res.dec_deg,
            "verdict": res.verdict,
            "confidence": res.confidence,
            "passed_morphology": res.passed_morphology,
            "has_significant_pm": res.has_significant_pm,
        }
        # Morphology fields
        if res.morphology:
            r["morph_snr"] = round(res.morphology.snr, 2)
            r["morph_pointiness"] = round(res.morphology.pointiness, 3)
            r["morph_fwhm_x"] = round(res.morphology.fwhm_x_arcsec, 2)
            r["morph_fwhm_y"] = round(res.morphology.fwhm_y_arcsec, 2)
            r["morph_ellipticity"] = round(res.morphology.ellipticity, 3)
            r["morph_is_point"] = res.morphology.is_point_source
        # PM fields
        if res.proper_motion and res.proper_motion.n_epochs >= 3:
            pm = res.proper_motion
            r["pm_n_detections"] = pm.n_detections
            r["pm_n_epochs"] = pm.n_epochs
            r["pm_baseline_yr"] = round(pm.baseline_years, 2)
            r["pm_ra_arcsec_yr"] = round(pm.mu_ra_arcsec_yr, 4)
            r["pm_dec_arcsec_yr"] = round(pm.mu_dec_arcsec_yr, 4)
            r["pm_total_arcsec_yr"] = round(pm.mu_total_arcsec_yr, 4)
            r["pm_significance"] = round(pm.pm_significance, 2)
            r["pm_mean_w2mag"] = round(pm.mean_w2mag, 2)
        # Original columns
        if "temperature_k" in row.index:
            r["temperature_k"] = row["temperature_k"]
        if "temperature_std_k" in row.index:
            r["temperature_std_k"] = row["temperature_std_k"]
        if "probability" in row.index:
            r["probability"] = row["probability"]
        rows.append(r)

    results_df = pd.DataFrame(rows)
    csv_path = args.output_dir / "dust_piercer_results.csv"
    results_df.to_csv(csv_path, index=False)

    # ---------------------------------------------------------------
    # Step 5: Summary
    # ---------------------------------------------------------------
    print("=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)

    verdicts = results_df["verdict"].value_counts()
    for verdict, count in verdicts.items():
        pct = count / len(results_df) * 100
        marker = ""
        if verdict == "POINT_MOVING":
            marker = " *** P9 CANDIDATE ***"
        elif verdict == "POINT_STATIC":
            marker = " (star/compact galaxy)"
        elif verdict == "DIFFUSE":
            marker = " (ISM/nebulosity)"
        elif verdict == "NO_SOURCE":
            marker = " (noise)"
        print(f"  {verdict:15s}: {count:3d} ({pct:5.1f}%){marker}")

    print(f"\n  Total: {len(results_df)} candidates analyzed")
    print(f"  Time:  {t_analysis:.1f}s")

    # Highlight POINT_MOVING
    moving = results_df[results_df["verdict"] == "POINT_MOVING"]
    if len(moving) > 0:
        print()
        print("!" * 64)
        print("  PROPER MOTION DETECTIONS IN INFRARED:")
        print("!" * 64)
        for _, m in moving.iterrows():
            temp_str = f"T={m['temperature_k']:.1f}K" if "temperature_k" in m.index and pd.notna(m.get("temperature_k")) else ""
            print(
                f"  RA={m['ra_deg']:.4f} Dec={m['dec_deg']:+.4f}  "
                f"PM={m.get('pm_total_arcsec_yr', 0):.3f}\"/yr  "
                f"({m.get('pm_significance', 0):.1f}sigma)  "
                f"W2={m.get('pm_mean_w2mag', 0):.1f}  "
                f"{temp_str}  "
                f"conf={m['confidence']}"
            )

    # Highlight POINT_STATIC
    static = results_df[results_df["verdict"] == "POINT_STATIC"]
    if len(static) > 0:
        print(f"\n  Point sources (static): {len(static)}")
        for _, s in static.iterrows():
            temp_str = f"T={s['temperature_k']:.1f}K" if "temperature_k" in s.index and pd.notna(s.get("temperature_k")) else ""
            print(
                f"    RA={s['ra_deg']:.4f} Dec={s['dec_deg']:+.4f}  "
                f"P={s.get('morph_pointiness', 0):.2f}  "
                f"FWHM={s.get('morph_fwhm_x', 0):.1f}\"  "
                f"{temp_str}"
            )

    print()
    print(f"  Output files:")
    print(f"    Results CSV: {csv_path}")
    if not args.no_cards:
        print(f"    Cards:       {cards_dir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
