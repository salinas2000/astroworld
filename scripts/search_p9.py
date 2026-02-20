#!/usr/bin/env python3
"""
Search sky survey images for Planet 9 candidates.

Uses a trained P9SiameseDetector with a sliding-window approach to
scan FITS images for faint moving sources.

Modes:
  - synthetic: Inject a source into each field and search the
    (original, injected) pair.  Validates the full pipeline.
  - self-pair: Pair each image with itself (null test, measures
    false positive rate).

Usage:
  # Synthetic validation (default — tests if pipeline finds injected sources)
  uv run python scripts/search_p9.py \\
      --checkpoint checkpoints/siamese_sanity_64/best_model.pt \\
      --image-dir data/survey_images/dss2_red

  # Self-pair null test (characterize false positive rate)
  uv run python scripts/search_p9.py \\
      --checkpoint checkpoints/siamese_sanity_64/best_model.pt \\
      --image-dir data/survey_images/dss2_red \\
      --mode self-pair

  # Custom threshold
  uv run python scripts/search_p9.py \\
      --checkpoint checkpoints/siamese_sanity_64/best_model.pt \\
      --image-dir data/survey_images/dss2_red \\
      --threshold 0.80 --inject-mag 20.0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Search sky survey images for Planet 9 candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--image-dir", type=Path, required=True,
        help="Directory containing FITS survey images",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/p9_search"),
        help="Output directory for results (default: results/p9_search)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="Detection probability threshold (default: 0.95)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=64,
        help="Patch size in pixels (default: 64)",
    )
    parser.add_argument(
        "--overlap", type=int, default=8,
        help="Patch overlap in pixels (default: 8)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--mode", choices=["synthetic", "self-pair"], default="synthetic",
        help="Pairing mode: synthetic (inject + search) or self-pair (null test)",
    )
    parser.add_argument(
        "--inject-mag", type=float, default=20.0,
        help="Injection magnitude for synthetic mode (default: 20.0)",
    )
    parser.add_argument(
        "--inject-snr", type=float, default=10.0,
        help="Injection S/N for synthetic mode (default: 10.0)",
    )
    parser.add_argument(
        "--motion-px", type=float, default=8.0,
        help="Source motion in pixels between epochs (default: 8.0)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ---- Check PyTorch ----
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        print("Install with: uv add --optional ml 'torch>=2.0'")
        sys.exit(1)

    from astroworld.imaging.fits_store import load_fits_image
    from astroworld.ml.inference import P9Searcher
    from astroworld.ml.post_processing import (
        detections_to_candidates,
        generate_report,
        plot_heatmap,
        save_diagnostic_cutouts,
    )

    # ---- Validate inputs ----
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not args.image_dir.exists():
        print(f"ERROR: Image directory not found: {args.image_dir}")
        sys.exit(1)

    fits_files = sorted(args.image_dir.glob("*.fits"))
    if not fits_files:
        print(f"ERROR: No FITS files found in {args.image_dir}")
        sys.exit(1)

    # ---- Header ----
    print("=" * 60)
    print("  ASTROWORLD — PLANET 9 CANDIDATE SEARCH")
    print("=" * 60)
    print()
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Images:      {len(fits_files)} FITS files in {args.image_dir}")
    print(f"  Mode:        {args.mode}")
    print(f"  Threshold:   {args.threshold}")
    print(f"  Patch:       {args.patch_size}×{args.patch_size}, overlap={args.overlap}")
    if args.mode == "synthetic":
        print(f"  Injection:   S/N={args.inject_snr}, motion={args.motion_px} px")
    print(f"  Output:      {args.output_dir}")
    print()

    # ---- Load model ----
    print("Loading model from checkpoint...")
    searcher = P9Searcher.from_checkpoint(
        args.checkpoint,
        patch_size=args.patch_size,
        overlap=args.overlap,
        threshold=args.threshold,
        device=args.device,
    )
    print(f"  Device: {searcher.device}")
    print(f"  Model: P9SiameseDetector ({searcher.model.num_parameters:,} params)")
    print()

    # ---- Prepare output directories ----
    output_dir = args.output_dir
    heatmap_dir = output_dir / "heatmaps"
    cutout_dir = output_dir / "cutouts"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run search ----
    rng = np.random.default_rng(args.seed)
    all_results = []
    total_candidates = 0
    t_total = time.time()

    print("-" * 60)

    for i, fits_path in enumerate(fits_files):
        field_name = fits_path.stem

        # Load original image
        image_orig, header = load_fits_image(fits_path)

        if args.mode == "synthetic":
            # Create synthetic epoch pair with injected moving source
            image_1 = image_orig.copy()
            image_2 = image_orig.copy()

            # Inject source at random position
            h, w = image_orig.shape
            margin = args.patch_size
            inject_x = rng.uniform(margin, w - margin)
            inject_y = rng.uniform(margin, h - margin)

            # Motion direction
            angle = rng.uniform(0, 2 * np.pi)
            inject_x2 = inject_x + args.motion_px * np.cos(angle)
            inject_y2 = inject_y + args.motion_px * np.sin(angle)

            # Inject using simple Gaussian PSF
            flux = args.inject_snr * _estimate_bg_std(image_orig) * 6.0
            image_1 = _inject_gaussian(image_1, inject_x, inject_y, flux)
            image_2 = _inject_gaussian(image_2, inject_x2, inject_y2, flux)

            label = f"synthetic (S/N={args.inject_snr})"

        elif args.mode == "self-pair":
            image_1 = image_orig
            image_2 = image_orig
            inject_x = inject_y = float("nan")
            label = "self-pair (null test)"

        # Search
        result = searcher.search_from_arrays(
            image_1, image_2,
            image_path_1=str(fits_path),
            image_path_2=str(fits_path),
        )

        # Post-processing: NMS + coordinate conversion
        candidates = detections_to_candidates(
            result.patch_detections, header, result.grid_shape,
        )
        result.candidates = candidates

        n_raw = len(result.patch_detections)
        n_cands = len(candidates)
        total_candidates += n_cands

        # Status
        inject_info = ""
        if args.mode == "synthetic":
            inject_info = f" | injected at ({inject_x:.0f},{inject_y:.0f})"
        print(
            f"  [{i + 1:2d}/{len(fits_files)}] {field_name} | "
            f"{n_raw} raw, {n_cands} candidates | "
            f"{result.inference_time_seconds:.1f}s{inject_info}"
        )

        # Save heatmap
        heatmap_path = heatmap_dir / f"{field_name}_heatmap.png"
        plot_heatmap(
            result, image_1, heatmap_path,
            title=f"{field_name} | {label}",
        )

        # Save cutouts
        if n_cands > 0:
            save_diagnostic_cutouts(
                result, image_1, image_2, cutout_dir,
                pair_label=field_name,
            )

        all_results.append(result)

    # ---- Generate report ----
    print()
    print("-" * 60)

    report_path = output_dir / "detection_report.csv"
    df = generate_report(all_results, report_path)

    elapsed_total = time.time() - t_total

    # ---- Summary ----
    print()
    print("=" * 60)
    print("  SEARCH COMPLETE")
    print("=" * 60)
    print(f"  Images scanned:   {len(fits_files)}")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Report:           {report_path}")
    print(f"  Heatmaps:         {heatmap_dir}")
    print(f"  Cutouts:          {cutout_dir}")
    print(f"  Total time:       {elapsed_total:.1f} seconds")
    print()

    if total_candidates > 0 and args.mode == "synthetic":
        detected = sum(1 for r in all_results if len(r.candidates) > 0)
        print(f"  Detection rate:   {detected}/{len(fits_files)} "
              f"({100 * detected / len(fits_files):.0f}%)")
        print()

    if total_candidates == 0 and args.mode == "self-pair":
        print("  [OK] Zero false positives in self-pair null test")
        print()

    # Save summary
    summary_path = output_dir / "search_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"P9 Candidate Search Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Image dir:  {args.image_dir}\n")
        f.write(f"Mode:       {args.mode}\n")
        f.write(f"Threshold:  {args.threshold}\n")
        f.write(f"Patch size: {args.patch_size}\n")
        f.write(f"Overlap:    {args.overlap}\n")
        f.write(f"Images:     {len(fits_files)}\n")
        f.write(f"Candidates: {total_candidates}\n")
        f.write(f"Time:       {elapsed_total:.1f}s\n")

    print(f"  Summary saved:    {summary_path}")


# ---------------------------------------------------------------------------
# Helpers for synthetic injection
# ---------------------------------------------------------------------------

def _estimate_bg_std(image: np.ndarray) -> float:
    """Quick background std estimate (3-sigma clipping)."""
    arr = image.flatten().astype(np.float64)
    mask = np.ones(len(arr), dtype=bool)
    for _ in range(3):
        masked = arr[mask]
        if len(masked) == 0:
            return 1.0
        mean = masked.mean()
        std = masked.std()
        if std < 1e-10:
            return 1.0
        mask = np.abs(arr - mean) < 3.0 * std
    return max(float(arr[mask].std()), 1.0)


def _inject_gaussian(
    image: np.ndarray,
    cx: float,
    cy: float,
    flux: float,
    fwhm: float = 2.5,
    stamp_size: int = 11,
) -> np.ndarray:
    """Inject a simple Gaussian point source into an image."""
    result = image.copy()
    h, w = image.shape
    sigma = fwhm / 2.3548

    ix, iy = int(round(cx)), int(round(cy))
    half = stamp_size // 2

    for j in range(-half, half + 1):
        for i in range(-half, half + 1):
            yy = iy + j
            xx = ix + i
            if 0 <= yy < h and 0 <= xx < w:
                r2 = (xx - cx) ** 2 + (yy - cy) ** 2
                g = np.exp(-r2 / (2 * sigma ** 2))
                # Normalize so total flux matches
                result[yy, xx] += flux * g / (2 * np.pi * sigma ** 2)

    return result


if __name__ == "__main__":
    main()
