#!/usr/bin/env python3
"""
Phase 15: Blind Search Campaign for Planet 9.

Systematic scan of the sky regions where Batygin & Brown (2016, 2021)
predict Planet 9 resides, using the Observatory Pipeline with per-patch
IR quality filtering.

Search regions
--------------
The primary search targets the anti-Galactic-center corridor where P9's
aphelion should lie (~380 AU, M ≈ 180°).  Three preset regions:

  **taurus**   — Taurus Molecular Cloud region (RA 55-85°, Dec +10-30°)
                 Covers Sedna's neighborhood + P9 aphelion zone.
                 ~100 fields, ~4 min with cached images.

  **ecliptic** — Full ecliptic corridor (RA 20-120°, Dec -30 to +30°)
                 Complete Batygin/Brown (2021) aphelion region.
                 ~500 fields, ~20 min with cached images.

  **galactic_gap** — Galactic plane gap (RA 80-120°, Dec -10 to +10°)
                 Often excluded by surveys due to stellar confusion.
                 ~50 fields, ~2 min.

Grid geometry
-------------
Each field covers ~10'×10' (FoV = 2 × download_radius_arcmin).
Default spacing = 8' (slight overlap for edge detections).
For 512×512 px images at DSS2 scale (~1.7"/px), this gives ~0.5°
overlap between adjacent fields.

Usage
-----
    # Quick test (4 fields):
    uv run python scripts/run_blind_search.py \\
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \\
        --region taurus --quick

    # Full Taurus campaign:
    uv run python scripts/run_blind_search.py \\
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \\
        --region taurus

    # Custom region:
    uv run python scripts/run_blind_search.py \\
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \\
        --ra-min 40 --ra-max 80 --dec-min 0 --dec-max 20 --step 0.15

    # Full ecliptic campaign (long!):
    uv run python scripts/run_blind_search.py \\
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \\
        --region ecliptic --planck-strict
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── Preset regions ────────────────────────────────────────────────

PRESETS = {
    "taurus": {
        "description": "Taurus Molecular Cloud + P9 aphelion zone",
        "ra_min": 55.0,
        "ra_max": 85.0,
        "dec_min": 10.0,
        "dec_max": 30.0,
        "step_deg": 0.50,  # ~30 arcmin: coarse survey pass
    },
    "ecliptic": {
        "description": "Full Batygin/Brown (2021) ecliptic corridor",
        "ra_min": 20.0,
        "ra_max": 120.0,
        "dec_min": -30.0,
        "dec_max": 30.0,
        "step_deg": 0.75,  # ~45 arcmin: wide-field survey
    },
    "galactic_gap": {
        "description": "Galactic plane gap (survey exclusion zone)",
        "ra_min": 80.0,
        "ra_max": 120.0,
        "dec_min": -10.0,
        "dec_max": 10.0,
        "step_deg": 0.50,
    },
    "sedna": {
        "description": "Sedna orbital neighborhood (validation grid)",
        "ra_min": 69.0,
        "ra_max": 73.5,
        "dec_min": 13.0,
        "dec_max": 16.5,
        "step_deg": 0.25,  # ~15 arcmin: fine grid for known region
    },
    # --- New regions: P9 full-orbit coverage ---
    "perihelion": {
        "description": "P9 perihelion corridor (Ophiuchus/Libra, ~320 AU, brightest)",
        "ra_min": 235.0,
        "ra_max": 258.0,
        "dec_min": -17.0,
        "dec_max": -1.0,
        "step_deg": 0.50,
    },
    "orion_fringe": {
        "description": "Orion/Gemini fringe above galactic plane (|b|~16-25 deg)",
        "ra_min": 108.0,
        "ra_max": 120.0,
        "dec_min": 23.0,
        "dec_max": 27.0,
        "step_deg": 0.25,
    },
    "cetus": {
        "description": "Cetus/Aquarius cleanest sky (|b|>67 deg, ecliptic quadrature)",
        "ra_min": 350.0,
        "ra_max": 380.0,  # wraps: 350-360 + 0-20
        "dec_min": -20.0,
        "dec_max": -8.0,
        "step_deg": 0.50,
    },
    "south_ecliptic": {
        "description": "South ecliptic descent (Capricornus, max southern dec)",
        "ra_min": 310.0,
        "ra_max": 345.0,
        "dec_min": -27.0,
        "dec_max": -20.0,
        "step_deg": 0.75,
    },
}


def generate_grid(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    step_deg: float = 0.15,
    exclude_galactic_plane: bool = True,
    galactic_lat_cut: float = 10.0,
) -> list[tuple[float, float]]:
    """Generate a grid of (RA, Dec) field centers.

    Parameters
    ----------
    ra_min, ra_max : RA bounds in degrees.
    dec_min, dec_max : Dec bounds in degrees.
    step_deg : Grid spacing in degrees.
    exclude_galactic_plane : If True, skip fields with |b| < galactic_lat_cut.
    galactic_lat_cut : Galactic latitude threshold (degrees).

    Returns
    -------
    List of (ra_deg, dec_deg) tuples.
    """
    ra_values = np.arange(ra_min, ra_max + step_deg / 2, step_deg)
    dec_values = np.arange(dec_min, dec_max + step_deg / 2, step_deg)

    fields = []
    for dec in dec_values:
        # Adjust RA spacing for cos(dec) projection
        cos_dec = max(np.cos(np.radians(dec)), 0.1)
        ra_step = step_deg / cos_dec
        ra_vals = np.arange(ra_min, ra_max + ra_step / 2, ra_step)

        for ra in ra_vals:
            if ra > ra_max:
                continue
            # Wrap RA > 360° back into [0, 360) range
            ra_wrapped = float(ra) % 360.0

            if exclude_galactic_plane:
                b = _galactic_latitude(ra_wrapped, dec)
                if abs(b) < galactic_lat_cut:
                    continue

            fields.append((round(ra_wrapped, 4), round(float(dec), 4)))

    return fields


def _galactic_latitude(ra_deg: float, dec_deg: float) -> float:
    """Approximate Galactic latitude b from ICRS (RA, Dec).

    Uses the standard rotation: the North Galactic Pole is at
    (RA=192.86°, Dec=+27.13°) and the Galactic center at
    (RA=266.40°, Dec=-28.94°).
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    # North Galactic Pole (J2000)
    ra_ngp = np.radians(192.85948)
    dec_ngp = np.radians(27.12825)

    sin_b = (
        np.sin(dec) * np.sin(dec_ngp)
        + np.cos(dec) * np.cos(dec_ngp) * np.cos(ra - ra_ngp)
    )
    return float(np.degrees(np.arcsin(np.clip(sin_b, -1, 1))))


def save_grid_csv(
    fields: list[tuple[float, float]],
    path: Path,
) -> None:
    """Save field grid to CSV."""
    df = pd.DataFrame(fields, columns=["ra_deg", "dec_deg"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 15: Blind Search Campaign for Planet 9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--spectral-checkpoint",
        type=Path,
        required=True,
        help="Path to SpectralSiameseNet checkpoint",
    )
    parser.add_argument(
        "--optical-checkpoint",
        type=Path,
        default=None,
        help="Path to P9SiameseDetector checkpoint (dual_optical mode)",
    )

    # Region selection
    region_group = parser.add_mutually_exclusive_group()
    region_group.add_argument(
        "--region",
        type=str,
        choices=list(PRESETS.keys()),
        default=None,
        help="Preset search region",
    )
    parser.add_argument(
        "--ra-min", type=float, default=None,
        help="Custom RA minimum (degrees)",
    )
    parser.add_argument(
        "--ra-max", type=float, default=None,
        help="Custom RA maximum (degrees)",
    )
    parser.add_argument(
        "--dec-min", type=float, default=None,
        help="Custom Dec minimum (degrees)",
    )
    parser.add_argument(
        "--dec-max", type=float, default=None,
        help="Custom Dec maximum (degrees)",
    )
    parser.add_argument(
        "--step", type=float, default=None,
        help="Grid step in degrees (default: from preset or 0.15)",
    )

    # Galactic filter
    parser.add_argument(
        "--no-galactic-filter",
        action="store_true",
        help="Don't exclude fields near the Galactic plane",
    )
    parser.add_argument(
        "--galactic-cut",
        type=float,
        default=10.0,
        help="Galactic latitude cut (degrees, default: 10)",
    )

    # Pipeline options
    parser.add_argument(
        "--planck-strict",
        action="store_true",
        help="Only keep 'p9_candidate' (exclude cold_object)",
    )
    parser.add_argument(
        "--mc-samples", type=int, default=10,
        help="MC Dropout samples (default: 10)",
    )
    parser.add_argument(
        "--spectral-threshold", type=float, default=0.80,
        help="Detection threshold (default: 0.80)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: results/blind_search_{region})",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/survey_images"),
        help="Survey image cache directory",
    )

    # Batch control
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Fields per batch (saves intermediate results, default: 50)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 2×2 grid subsample for testing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate grid and show stats without running pipeline",
    )
    parser.add_argument(
        "--resume-from", type=int, default=0,
        help="Resume from field N (skip first N fields)",
    )
    parser.add_argument(
        "--ir-only", action="store_true",
        help="IR-only mode: skip optical, use WISE W1+W2 + Dust Piercer. "
             "For targets invisible in optical (P9 at ~25 mag).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Force UTF-8 on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # ── Determine region bounds ───────────────────────────────────
    if args.region:
        preset = PRESETS[args.region]
        ra_min = preset["ra_min"]
        ra_max = preset["ra_max"]
        dec_min = preset["dec_min"]
        dec_max = preset["dec_max"]
        step = args.step or preset["step_deg"]
        region_name = args.region
        description = preset["description"]
    elif all(v is not None for v in [args.ra_min, args.ra_max, args.dec_min, args.dec_max]):
        ra_min = args.ra_min
        ra_max = args.ra_max
        dec_min = args.dec_min
        dec_max = args.dec_max
        step = args.step or 0.15
        region_name = "custom"
        description = f"Custom: RA [{ra_min:.1f}, {ra_max:.1f}], Dec [{dec_min:.1f}, {dec_max:.1f}]"
    else:
        print(
            "ERROR: Provide --region or all of --ra-min/--ra-max/--dec-min/--dec-max",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = args.output_dir or Path(f"results/blind_search_{region_name}")

    # ── Generate grid ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 15 — BLIND SEARCH CAMPAIGN")
    print("=" * 65)
    print(f"  Region:    {region_name} — {description}")
    print(f"  RA range:  [{ra_min:.1f}°, {ra_max:.1f}°]")
    print(f"  Dec range: [{dec_min:.1f}°, {dec_max:.1f}°]")
    print(f"  Step:      {step:.3f}° ({step * 60:.1f}')")
    print(f"  Gal. cut:  {'OFF' if args.no_galactic_filter else f'|b| > {args.galactic_cut:.0f}°'}")

    fields = generate_grid(
        ra_min, ra_max, dec_min, dec_max,
        step_deg=step,
        exclude_galactic_plane=not args.no_galactic_filter,
        galactic_lat_cut=args.galactic_cut,
    )

    if args.quick:
        # Subsample: take 4 evenly spaced fields
        n = len(fields)
        indices = [0, n // 3, 2 * n // 3, n - 1] if n >= 4 else list(range(n))
        fields = [fields[i] for i in indices]
        print(f"  Quick mode: subsampled to {len(fields)} fields")

    if args.resume_from > 0:
        fields = fields[args.resume_from:]
        print(f"  Resuming from field {args.resume_from}, {len(fields)} remaining")

    print(f"  Total fields: {len(fields)}")

    # Estimate time
    est_sec = len(fields) * 2.5  # ~2.5s/field with cached images
    est_min = est_sec / 60
    print(f"  Est. time: {est_min:.0f} min ({est_sec:.0f}s)")
    print("-" * 65)

    # Save grid CSV
    grid_csv = output_dir / f"grid_{region_name}.csv"
    grid_csv.parent.mkdir(parents=True, exist_ok=True)
    save_grid_csv(fields, grid_csv)
    print(f"  Grid saved: {grid_csv}")

    if args.dry_run:
        print("\n  [DRY RUN] Grid generated. Exiting without running pipeline.")
        _print_grid_stats(fields)
        return

    # ── Load pipeline ─────────────────────────────────────────────
    from astroworld.ml.pipeline import (
        ObservatoryPipeline,
        PipelineConfig,
    )

    planck_keep = ["p9_candidate"]
    if not args.planck_strict:
        planck_keep.append("cold_object")

    pipeline_mode = "ir_only" if getattr(args, "ir_only", False) else "spectral_only"
    config = PipelineConfig(
        mode=pipeline_mode,
        spectral_threshold=args.spectral_threshold,
        mc_samples=args.mc_samples,
        planck_classes_keep=planck_keep,
        kinematic_enabled=True,
        device=args.device,
        output_dir=output_dir,
        data_dir=args.data_dir,
    )

    print(f"\n  Loading model: {args.spectral_checkpoint}")
    pipeline = ObservatoryPipeline.from_checkpoints(
        spectral_checkpoint=args.spectral_checkpoint,
        optical_checkpoint=args.optical_checkpoint,
        config=config,
    )

    # ── Run in batches ────────────────────────────────────────────
    batch_size = args.batch_size
    n_batches = (len(fields) + batch_size - 1) // batch_size
    all_results = []
    t_campaign = time.time()

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(fields))
        batch_fields = fields[start:end]

        print(f"\n{'─' * 65}")
        print(
            f"  BATCH {batch_idx + 1}/{n_batches} "
            f"(fields {start + 1}–{end} of {len(fields)})"
        )
        print(f"{'─' * 65}")

        summary = pipeline.run(fields=batch_fields)
        all_results.append(summary)

        # Save intermediate results after each batch
        _save_intermediate(all_results, output_dir, region_name)

    # ── Final aggregation ─────────────────────────────────────────
    campaign_time = time.time() - t_campaign
    _print_campaign_summary(all_results, campaign_time, region_name)
    _save_final_report(all_results, output_dir, region_name, campaign_time)


def _save_intermediate(
    summaries: list,
    output_dir: Path,
    region_name: str,
) -> None:
    """Save accumulated results after each batch."""
    all_dfs = [s.candidate_table for s in summaries if len(s.candidate_table) > 0]
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        path = output_dir / f"candidates_{region_name}_partial.csv"
        combined.to_csv(path, index=False)


def _save_final_report(
    summaries: list,
    output_dir: Path,
    region_name: str,
    campaign_time: float,
) -> None:
    """Save final aggregated results."""
    all_dfs = [s.candidate_table for s in summaries if len(s.candidate_table) > 0]

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=[
                "field_id", "field_ra", "field_dec",
                "pixel_x", "pixel_y", "ra_deg", "dec_deg",
                "probability", "temperature_k", "temperature_std_k",
                "planck_class", "kinematic_valid",
            ]
        )

    final_csv = output_dir / f"candidates_{region_name}_final.csv"
    combined.to_csv(final_csv, index=False)

    # Summary text
    total_fields = sum(s.n_fields_total for s in summaries)
    total_candidates = len(combined)
    total_errors = sum(s.n_fields_errored for s in summaries)

    lines = [
        "=" * 60,
        "PHASE 15 — BLIND SEARCH CAMPAIGN REPORT",
        "=" * 60,
        f"Region:          {region_name}",
        f"Fields scanned:  {total_fields}",
        f"Fields errored:  {total_errors}",
        f"Total candidates: {total_candidates}",
        f"Campaign time:   {campaign_time:.1f}s ({campaign_time / 60:.1f} min)",
        f"Avg per field:   {campaign_time / max(total_fields, 1):.2f}s",
        "",
    ]

    if total_candidates > 0:
        lines.extend([
            "Top candidates (by probability):",
            "-" * 60,
        ])
        top = combined.nlargest(20, "probability")
        for _, row in top.iterrows():
            lines.append(
                f"  RA={row['ra_deg']:.4f}° Dec={row['dec_deg']:+.4f}° "
                f"p={row['probability']:.3f} T={row['temperature_k']:.1f}K "
                f"[{row['planck_class']}]"
            )

    report_path = output_dir / f"campaign_report_{region_name}.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report: {report_path}")
    print(f"  Candidates CSV: {final_csv}")


def _print_campaign_summary(
    summaries: list,
    campaign_time: float,
    region_name: str,
) -> None:
    """Print final campaign summary."""
    total_fields = sum(s.n_fields_total for s in summaries)
    total_stage3 = sum(s.n_stage3_detections for s in summaries)
    total_planck = sum(s.n_stage3_planck_filtered for s in summaries)
    total_candidates = sum(s.total_candidates for s in summaries)
    total_errors = sum(s.n_fields_errored for s in summaries)

    print(f"\n{'=' * 65}")
    print(f"  CAMPAIGN COMPLETE — {region_name}")
    print(f"{'=' * 65}")
    print(f"  Fields scanned:    {total_fields}")
    if total_errors > 0:
        print(f"  Fields errored:    {total_errors}")
    print(f"  Raw detections:    {total_stage3}")
    print(f"  Planck filtered:   {total_planck}")
    print(f"  FINAL CANDIDATES:  {total_candidates}")
    print(f"{'─' * 65}")
    print(f"  Campaign time:     {campaign_time:.1f}s ({campaign_time / 60:.1f} min)")
    print(f"  Avg per field:     {campaign_time / max(total_fields, 1):.2f}s")

    if total_candidates > 0:
        fp_rate = total_candidates / max(total_fields, 1)
        print(f"  FP rate:           {fp_rate:.2f} candidates/field")
    print(f"{'=' * 65}\n")


def _print_grid_stats(fields: list[tuple[float, float]]) -> None:
    """Print grid statistics for dry-run mode."""
    if not fields:
        print("  Grid is empty!")
        return

    ras = [f[0] for f in fields]
    decs = [f[1] for f in fields]

    print(f"\n  Grid statistics:")
    print(f"    RA:  {min(ras):.2f}° — {max(ras):.2f}° ({len(set(ras))} unique)")
    print(f"    Dec: {min(decs):.2f}° — {max(decs):.2f}° ({len(set(decs))} unique)")
    print(f"    Total fields: {len(fields)}")
    print(f"    Sky area: ~{(max(ras) - min(ras)) * (max(decs) - min(decs)):.0f} sq deg")


if __name__ == "__main__":
    main()
