#!/usr/bin/env python3
"""Multi-epoch blink analysis for Planet 9 candidate verification.

Downloads Pan-STARRS r-band images (~2012 epoch) and compares against
DSS2 Red images (~1993 epoch) to detect proper motion in candidates
from the Taurus Hunt (or any candidate list).

Verdicts:
  STATIC:   Source at same position in both epochs -> ISM / background galaxy
  MOVED:    Source shifted between epochs -> possible real object!
  ABSENT:   No source in either epoch -> pipeline noise
  APPEARED: Source in only one epoch -> transient or moved out of field

Usage
-----
    uv run python scripts/time_machine.py \
        --input results/taurus_crossmatched/candidates_unknown.csv \
        --temp-max 20.0 \
        --output-dir results/time_machine

    # Single candidate mode:
    uv run python scripts/time_machine.py \
        --ra 57.5437 --dec 15.9455 \
        --output-dir results/time_machine

    # Download-only mode (no analysis):
    uv run python scripts/time_machine.py \
        --input results/taurus_crossmatched/candidates_unknown.csv \
        --download-only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from astroworld.imaging.time_machine import (
    download_ps1_cutout,
    blink_candidate,
    blink_candidates,
    make_blink_card,
    BlinkResult,
)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-epoch blink analysis for P9 candidate verification",
    )

    # Input: CSV or single coordinate
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=Path,
        help="Path to candidate CSV (with ra_deg, dec_deg columns)",
    )
    input_group.add_argument(
        "--ra", type=float,
        help="Single candidate RA (degrees). Must be used with --dec",
    )
    parser.add_argument(
        "--dec", type=float,
        help="Single candidate Dec (degrees). Must be used with --ra",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/time_machine"),
        help="Output directory for results (default: results/time_machine)",
    )

    # Filters
    parser.add_argument(
        "--temp-max", type=float, default=None,
        help="Only analyze candidates with temperature_k <= this value",
    )

    # Analysis parameters
    parser.add_argument(
        "--cutout-arcsec", type=float, default=30.0,
        help="Cutout size in arcseconds (default: 30.0)",
    )
    parser.add_argument(
        "--shift-threshold", type=float, default=1.0,
        help="Minimum shift in arcsec to classify as MOVED (default: 1.0)",
    )
    parser.add_argument(
        "--dss2-dir", type=Path,
        default=Path("data/survey_images/dss2_red"),
        help="Directory with DSS2 Red FITS files",
    )
    parser.add_argument(
        "--ps1-dir", type=Path,
        default=Path("data/survey_images/ps1_r"),
        help="Directory for Pan-STARRS FITS files",
    )

    # Modes
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download PS1 cutouts, skip blink analysis",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=1.0,
        help="Delay between PS1 download requests (default: 1.0s)",
    )
    parser.add_argument(
        "--no-cards", action="store_true",
        help="Skip generating blink card PNGs",
    )

    args = parser.parse_args()

    # Validate single-coordinate mode
    if args.ra is not None and args.dec is None:
        parser.error("--ra requires --dec")

    # Build candidate DataFrame
    if args.input:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        df = pd.read_csv(args.input)
    else:
        df = pd.DataFrame([{"ra_deg": args.ra, "dec_deg": args.dec}])

    # Apply temperature filter
    n_original = len(df)
    if args.temp_max is not None and "temperature_k" in df.columns:
        df = df[df["temperature_k"] <= args.temp_max].copy()

    n_filtered = len(df)

    # Header
    print()
    print("=" * 64)
    print("  LA MAQUINA DEL TIEMPO - Multi-Epoch Blink Analysis")
    print("  Phase 18: Planet 9 Candidate Verification")
    print("=" * 64)
    print(f"  Candidates:       {n_filtered}", end="")
    if args.temp_max is not None:
        print(f" (filtered from {n_original}, T <= {args.temp_max} K)")
    else:
        print()
    print(f"  DSS2 directory:   {args.dss2_dir}")
    print(f"  PS1 directory:    {args.ps1_dir}")
    print(f"  Output:           {args.output_dir}")
    print(f"  Cutout size:      {args.cutout_arcsec} arcsec")
    print(f"  Shift threshold:  {args.shift_threshold} arcsec")
    print(f"  Rate limit:       {args.rate_limit}s")
    if args.download_only:
        print(f"  Mode:             DOWNLOAD ONLY")
    print("=" * 64)
    print()

    if n_filtered == 0:
        print("No candidates to process.")
        return

    # ---------------------------------------------------------------
    # Step 1: Download PS1 cutouts
    # ---------------------------------------------------------------
    print("STEP 1: Downloading Pan-STARRS r-band cutouts...")
    print("-" * 64)

    t0_download = time.time()
    downloaded = 0
    skipped = 0
    failed = 0

    for i, (_, row) in enumerate(df.iterrows()):
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])
        print(f"  [{i+1:3d}/{n_filtered}] RA={ra:.4f} Dec={dec:+.4f}", end=" ")

        result = download_ps1_cutout(
            ra, dec,
            output_dir=args.ps1_dir.parent,
            rate_limit_sec=args.rate_limit,
        )

        if result is None:
            print("-> FAILED (no coverage or error)")
            failed += 1
        elif result.stat().st_size < 100:
            # Tiny file = cached resume marker or error
            print(f"-> SKIP (cached: {result.name})")
            skipped += 1
        else:
            print(f"-> OK ({result.name})")
            downloaded += 1

    t_download = time.time() - t0_download

    print()
    print(f"  Download complete: {downloaded} new, {skipped} cached, "
          f"{failed} failed ({t_download:.1f}s)")
    print()

    if args.download_only:
        print("Download-only mode. Exiting.")
        return

    # ---------------------------------------------------------------
    # Step 2: Blink analysis
    # ---------------------------------------------------------------
    print("STEP 2: Running blink comparison (DSS2 ~1993 vs PS1 ~2012)...")
    print("-" * 64)

    t0_blink = time.time()
    results_list = blink_candidates(
        df,
        dss2_dir=args.dss2_dir,
        ps1_dir=args.ps1_dir,
        cutout_arcsec=args.cutout_arcsec,
        shift_threshold_arcsec=args.shift_threshold,
        verbose=True,
    )
    t_blink = time.time() - t0_blink

    print()
    print(f"  Blink analysis complete ({t_blink:.1f}s)")
    print()

    # ---------------------------------------------------------------
    # Step 3: Generate blink cards
    # ---------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir = args.output_dir / "blink_cards"

    if not args.no_cards:
        print("STEP 3: Generating blink cards...")
        print("-" * 64)
        cards_dir.mkdir(parents=True, exist_ok=True)

        n_cards = 0
        for i, ((result, c1, c2), (_, row)) in enumerate(
            zip(results_list, df.iterrows())
        ):
            if c1 is None or c2 is None:
                continue

            temp_k = float(row["temperature_k"]) if "temperature_k" in row.index else None
            card_name = (
                f"blink_{i+1:03d}_ra{result.ra_deg:.3f}_"
                f"dec{result.dec_deg:+.3f}_{result.verdict}.png"
            )
            card_path = cards_dir / card_name
            make_blink_card(result, c1, c2, save_path=card_path,
                            candidate_temp_k=temp_k)
            n_cards += 1
            print(f"  [{i+1:3d}] {card_name}")

        print(f"\n  Generated {n_cards} blink cards in {cards_dir}")
        print()

    # ---------------------------------------------------------------
    # Step 4: Save results CSV
    # ---------------------------------------------------------------
    rows = []
    for i, ((result, _, _), (_, row)) in enumerate(
        zip(results_list, df.iterrows())
    ):
        r = {
            "candidate_id": i + 1,
            "ra_deg": result.ra_deg,
            "dec_deg": result.dec_deg,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "shift_arcsec": round(result.shift_arcsec, 3),
            "shift_ra_arcsec": round(result.shift_ra_arcsec, 3),
            "shift_dec_arcsec": round(result.shift_dec_arcsec, 3),
            "snr_epoch1": round(result.snr_epoch1, 2),
            "snr_epoch2": round(result.snr_epoch2, 2),
            "flux_ratio": round(result.flux_ratio, 3),
            "epoch1_year": result.epoch1_year,
            "epoch2_year": result.epoch2_year,
            "baseline_years": result.baseline_years,
            "implied_pm_arcsec_yr": round(result.implied_pm_arcsec_yr, 4),
        }
        # Carry over original columns
        if "temperature_k" in row.index:
            r["temperature_k"] = row["temperature_k"]
        if "temperature_std_k" in row.index:
            r["temperature_std_k"] = row["temperature_std_k"]
        if "probability" in row.index:
            r["probability"] = row["probability"]
        rows.append(r)

    results_df = pd.DataFrame(rows)
    csv_path = args.output_dir / "time_machine_results.csv"
    results_df.to_csv(csv_path, index=False)

    # ---------------------------------------------------------------
    # Step 5: Summary
    # ---------------------------------------------------------------
    print("=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)

    # Verdict distribution
    verdicts = results_df["verdict"].value_counts()
    for verdict, count in verdicts.items():
        pct = count / len(results_df) * 100
        marker = ""
        if verdict == "MOVED":
            marker = " *** ALERT ***"
        elif verdict == "STATIC":
            marker = " (ISM/background)"
        elif verdict == "ABSENT":
            marker = " (noise)"
        print(f"  {verdict:10s}: {count:3d} ({pct:5.1f}%){marker}")

    print(f"\n  Total: {len(results_df)} candidates analyzed")
    print(f"  Time:  {t_download + t_blink:.1f}s "
          f"(download: {t_download:.1f}s, blink: {t_blink:.1f}s)")

    # Highlight MOVED candidates
    moved = results_df[results_df["verdict"] == "MOVED"]
    if len(moved) > 0:
        print()
        print("!" * 64)
        print("  PROPER MOTION DETECTIONS:")
        print("!" * 64)
        for _, m in moved.iterrows():
            temp_str = f"T={m['temperature_k']:.1f}K" if "temperature_k" in m.index else ""
            print(f"  RA={m['ra_deg']:.4f} Dec={m['dec_deg']:+.4f}  "
                  f"shift={m['shift_arcsec']:.2f}\"  "
                  f"PM={m['implied_pm_arcsec_yr']:.3f}\"/yr  "
                  f"{temp_str}  "
                  f"conf={m['confidence']}")

    # Highlight APPEARED candidates
    appeared = results_df[results_df["verdict"] == "APPEARED"]
    if len(appeared) > 0:
        print()
        print("  APPEARED (transient/moving objects):")
        for _, a in appeared.iterrows():
            temp_str = f"T={a['temperature_k']:.1f}K" if "temperature_k" in a.index else ""
            print(f"  RA={a['ra_deg']:.4f} Dec={a['dec_deg']:+.4f}  "
                  f"SNR1={a['snr_epoch1']:.1f} SNR2={a['snr_epoch2']:.1f}  "
                  f"{temp_str}  "
                  f"conf={a['confidence']}")

    print()
    print(f"  Output files:")
    print(f"    Results CSV: {csv_path}")
    if not args.no_cards:
        print(f"    Blink cards: {cards_dir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
