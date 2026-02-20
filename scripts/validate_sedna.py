#!/usr/bin/env python3
"""
Sedna Positive-Control Validation for P9 Siamese Detector.

Proves the detection pipeline works on a KNOWN object (Sedna, V~20.8)
before scaling to 10,000+ images for Planet 9 hunting.

Two validation tiers:

  Tier 1 (cutout): Single survey image at Sedna's position. Download a
    field centered on Sedna (target) and an offset field (reference).
    The Siamese detects that a source exists in the target but not in
    the reference.

  Tier 2 (cross-survey): Two survey images from different epochs,
    reprojected to common WCS. Tests the full sliding-window pipeline
    on real multi-epoch data with real proper motion.

Usage:
  # Tier 1: Cutout-level validation (fast, no reprojection)
  uv run python scripts/validate_sedna.py \\
      --checkpoint checkpoints/hard_negatives/best_model.pt \\
      --tier 1 --survey "DSS2 Red"

  # Tier 2: Full cross-survey validation
  uv run python scripts/validate_sedna.py \\
      --checkpoint checkpoints/hard_negatives/best_model.pt \\
      --tier 2 --survey1 "DSS2 Red" --survey2 "SDSSr"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Approximate DSS2 Red observation epoch in JD (mid-1993)
_DSS2_EPOCH_JD = 2449200.0  # ~1993.7

# Approximate SDSS epoch in JD (mid-2005)
_SDSS_EPOCH_JD = 2453553.0  # ~2005.5


def main():
    parser = argparse.ArgumentParser(
        description="Sedna positive-control validation for P9 detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--tier", type=int, default=1, choices=[1, 2],
        help="Validation tier: 1 (cutout) or 2 (cross-survey)",
    )
    parser.add_argument(
        "--survey", default="DSS2 Red",
        help="Survey for Tier 1 (default: DSS2 Red)",
    )
    parser.add_argument(
        "--survey1", default="DSS2 Red",
        help="First survey for Tier 2 (default: DSS2 Red)",
    )
    parser.add_argument(
        "--survey2", default="SDSSr",
        help="Second survey for Tier 2 (default: SDSSr)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Detection probability threshold (default: 0.80)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/sedna_validation"),
    )
    parser.add_argument(
        "--radius-arcmin", type=float, default=5.0,
        help="Field of view radius in arcminutes (default: 5.0)",
    )
    parser.add_argument(
        "--pixels", type=int, default=512,
        help="Image size in pixels (default: 512)",
    )
    parser.add_argument(
        "--offset-arcmin", type=float, default=2.0,
        help="Offset for reference field in Tier 1 (default: 2.0)",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--patch-size", type=int, default=64,
    )
    parser.add_argument(
        "--overlap", type=int, default=8,
    )

    args = parser.parse_args()

    # ---- Check dependencies ----
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        sys.exit(1)

    from astroworld.imaging.ephemeris import (
        fetch_sedna_position,
        compute_inter_epoch_motion,
    )
    from astroworld.imaging.survey_download import download_field
    from astroworld.imaging.fits_store import load_fits_image
    from astroworld.imaging.synthetic_injection import radec_to_pixel
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

    # ---- Header ----
    print("=" * 60)
    print("  ASTROWORLD -- SEDNA POSITIVE-CONTROL VALIDATION")
    print("=" * 60)
    print()
    print(f"  Tier:        {args.tier}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Threshold:   {args.threshold}")
    print(f"  Patch:       {args.patch_size}x{args.patch_size}")
    print(f"  Output:      {args.output_dir}")
    print()

    # ---- Setup output ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir = args.output_dir / "cutouts"
    cutout_dir.mkdir(exist_ok=True)

    t_start = time.time()

    if args.tier == 1:
        _run_tier1(args, fetch_sedna_position, download_field,
                   load_fits_image, radec_to_pixel, P9Searcher,
                   detections_to_candidates, generate_report,
                   plot_heatmap, save_diagnostic_cutouts)
    elif args.tier == 2:
        _run_tier2(args, fetch_sedna_position, compute_inter_epoch_motion,
                   download_field, load_fits_image, radec_to_pixel,
                   P9Searcher, detections_to_candidates, generate_report,
                   plot_heatmap, save_diagnostic_cutouts)

    elapsed = time.time() - t_start
    print()
    print(f"  Total time: {elapsed:.1f} seconds")


def _run_tier1(args, fetch_sedna_position, download_field,
               load_fits_image, radec_to_pixel, P9Searcher,
               detections_to_candidates, generate_report,
               plot_heatmap, save_diagnostic_cutouts):
    """Tier 1: Single-survey cutout validation."""

    # 1. Get Sedna's position at the survey epoch
    print("[1/6] Querying JPL Horizons for Sedna's position...")
    sedna = fetch_sedna_position(_DSS2_EPOCH_JD)
    ra_sedna = sedna["ra_deg"]
    dec_sedna = sedna["dec_deg"]
    v_mag = sedna["V_mag"]
    r_au = sedna["r_au"]
    delta_au = sedna["delta_au"]

    print(f"  Sedna @ JD {_DSS2_EPOCH_JD:.1f} (~1993.7):")
    print(f"    RA  = {ra_sedna:.4f} deg")
    print(f"    Dec = {dec_sedna:.4f} deg")
    print(f"    V   = {v_mag:.1f} mag")
    print(f"    r   = {r_au:.1f} AU (heliocentric)")
    print(f"    d   = {delta_au:.1f} AU (geocentric)")
    print()

    # Save ephemeris
    eph_df = pd.DataFrame([sedna])
    eph_path = args.output_dir / "sedna_ephemeris.csv"
    eph_df.to_csv(eph_path, index=False)

    # 2. Download target field (centered on Sedna)
    print(f"[2/6] Downloading {args.survey} field at Sedna's position...")
    target_dir = args.output_dir / "fits_target"
    target_paths = download_field(
        ra_deg=ra_sedna,
        dec_deg=dec_sedna,
        surveys=[args.survey],
        radius_arcmin=args.radius_arcmin,
        pixels=args.pixels,
        output_dir=target_dir,
    )

    if not target_paths:
        print("  ERROR: Failed to download target field.")
        print("  Sedna may be outside the survey coverage area.")
        sys.exit(1)

    print(f"  Downloaded: {target_paths[0]}")

    # 3. Download reference field (offset, no Sedna)
    print(f"[3/6] Downloading reference field (offset {args.offset_arcmin}')...")
    ref_dir = args.output_dir / "fits_reference"
    ref_ra = ra_sedna
    ref_dec = dec_sedna + args.offset_arcmin / 60.0

    ref_paths = download_field(
        ra_deg=ref_ra,
        dec_deg=ref_dec,
        surveys=[args.survey],
        radius_arcmin=args.radius_arcmin,
        pixels=args.pixels,
        output_dir=ref_dir,
    )

    if not ref_paths:
        print("  ERROR: Failed to download reference field.")
        sys.exit(1)

    print(f"  Downloaded: {ref_paths[0]}")
    print()

    # 4. Load images and locate Sedna
    print("[4/6] Loading images and locating Sedna...")
    img_target, hdr_target = load_fits_image(target_paths[0])
    img_ref, hdr_ref = load_fits_image(ref_paths[0])

    try:
        px_sedna, py_sedna = radec_to_pixel(ra_sedna, dec_sedna, hdr_target)
        print(f"  Sedna pixel position in target: ({px_sedna:.1f}, {py_sedna:.1f})")
    except ValueError:
        print("  WARNING: Sedna position is outside the target image bounds.")
        print("  Trying with a larger field or different survey may help.")
        px_sedna, py_sedna = float("nan"), float("nan")

    # Check image dimensions match (required by Siamese)
    if img_target.shape != img_ref.shape:
        print(f"  WARNING: Image shapes differ: target={img_target.shape}, "
              f"ref={img_ref.shape}")
        # Crop to common size
        min_h = min(img_target.shape[0], img_ref.shape[0])
        min_w = min(img_target.shape[1], img_ref.shape[1])
        img_target = img_target[:min_h, :min_w]
        img_ref = img_ref[:min_h, :min_w]
        print(f"  Cropped both to: {img_target.shape}")

    print()

    # 5. Run Siamese search
    print("[5/6] Running Siamese detector...")
    searcher = P9Searcher.from_checkpoint(
        args.checkpoint,
        patch_size=args.patch_size,
        overlap=args.overlap,
        threshold=args.threshold,
        device=args.device,
    )
    print(f"  Device: {searcher.device}")
    print(f"  Model: P9SiameseDetector ({searcher.model.num_parameters:,} params)")

    # Search: reference vs target (Sedna only in target)
    result = searcher.search_from_arrays(
        img_ref, img_target,
        image_path_1="reference_field",
        image_path_2="sedna_target_field",
    )

    # Post-processing
    candidates = detections_to_candidates(
        result.patch_detections, hdr_target, result.grid_shape,
    )
    result.candidates = candidates

    n_raw = len(result.patch_detections)
    n_cands = len(candidates)
    print(f"  Raw detections:  {n_raw}")
    print(f"  After NMS:       {n_cands} candidates")
    print()

    # 6. Evaluate: check if any candidate is near Sedna
    print("[6/6] Evaluating detection at Sedna's position...")
    print("-" * 60)

    sedna_detected = False
    best_distance_px = float("inf")
    best_prob = 0.0

    if not np.isnan(px_sedna):
        for cand in candidates:
            dist = np.sqrt(
                (cand.pixel_x - px_sedna) ** 2
                + (cand.pixel_y - py_sedna) ** 2
            )
            if dist < best_distance_px:
                best_distance_px = dist
                best_prob = cand.probability

            if dist < 32:  # Within half a patch
                sedna_detected = True

    # Also check heatmap at Sedna's position for sub-threshold signal
    heatmap_at_sedna = float("nan")
    if not np.isnan(px_sedna) and result.heatmap is not None:
        stride = args.patch_size - args.overlap
        grid_row = int(py_sedna / stride)
        grid_col = int(px_sedna / stride)
        h_rows, h_cols = result.heatmap.shape
        if 0 <= grid_row < h_rows and 0 <= grid_col < h_cols:
            heatmap_at_sedna = float(result.heatmap[grid_row, grid_col])

    # Print results
    print()
    print("=" * 60)
    print("  SEDNA VALIDATION RESULTS")
    print("=" * 60)
    print()
    print(f"  Sedna RA/Dec:     {ra_sedna:.4f}, {dec_sedna:.4f}")
    print(f"  Sedna pixel:      ({px_sedna:.1f}, {py_sedna:.1f})")
    print(f"  Sedna V mag:      {v_mag:.1f}")
    print(f"  Survey:           {args.survey}")
    print(f"  Threshold:        {args.threshold}")
    print()
    print(f"  Total candidates:     {n_cands}")
    print(f"  Heatmap at Sedna:     {heatmap_at_sedna:.4f}")
    print(f"  Nearest candidate:    {best_distance_px:.1f} px (p={best_prob:.3f})")
    print()

    if sedna_detected or heatmap_at_sedna > args.threshold:
        print("  >>> SEDNA DETECTED <<<")
        print(f"  Heatmap confidence: {heatmap_at_sedna:.4f}")
        if best_distance_px < float("inf"):
            print(f"  Nearest NMS candidate: {best_distance_px:.1f} px (p={best_prob:.3f})")
        if n_cands > 5:
            print(f"  NOTE: {n_cands} total candidates (offset fields differ broadly).")
            print(f"  Sedna's signal is confirmed by heatmap at its known pixel position.")
    elif heatmap_at_sedna > 0.5:
        print("  >>> SEDNA WEAKLY DETECTED <<<")
        print(f"  Heatmap signal: {heatmap_at_sedna:.3f}")
    else:
        print("  >>> SEDNA NOT DETECTED <<<")
        print(f"  Heatmap signal at Sedna's position: {heatmap_at_sedna:.4f}")
        print("  This may indicate:")
        print("    - Sedna is below detection limit in this survey")
        print("    - Model sensitivity needs improvement for real data")
        print("    - Offset field may share some Sedna light")

    print()

    # Save diagnostics
    heatmap_path = args.output_dir / "heatmap_tier1.png"
    plot_heatmap(
        result, img_target, heatmap_path,
        title=f"Sedna Validation Tier 1 | {args.survey}",
    )
    print(f"  Heatmap saved: {heatmap_path}")

    if n_cands > 0:
        save_diagnostic_cutouts(
            result, img_ref, img_target,
            args.output_dir / "cutouts",
            pair_label="sedna_tier1",
        )

    # Save detection report
    report_path = args.output_dir / "detection_report.csv"
    generate_report([result], report_path)
    print(f"  Report saved:  {report_path}")

    # Save summary
    summary_path = args.output_dir / "sedna_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Sedna Positive-Control Validation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Tier: 1 (single-survey cutout)\n")
        f.write(f"Survey: {args.survey}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        f.write(f"Sedna Ephemeris (JD {_DSS2_EPOCH_JD:.1f}):\n")
        f.write(f"  RA:  {ra_sedna:.6f} deg\n")
        f.write(f"  Dec: {dec_sedna:.6f} deg\n")
        f.write(f"  V:   {v_mag:.1f} mag\n")
        f.write(f"  r:   {r_au:.1f} AU\n")
        f.write(f"  d:   {delta_au:.1f} AU\n\n")
        f.write(f"Detection Results:\n")
        f.write(f"  Candidates: {n_cands}\n")
        f.write(f"  Heatmap at Sedna: {heatmap_at_sedna:.4f}\n")
        f.write(f"  Nearest candidate: {best_distance_px:.1f} px\n")
        detected = sedna_detected or heatmap_at_sedna > args.threshold
        f.write(f"  Detected: {'YES' if detected else 'NO'}\n")

    print(f"  Summary saved: {summary_path}")


def _run_tier2(args, fetch_sedna_position, compute_inter_epoch_motion,
               download_field, load_fits_image, radec_to_pixel,
               P9Searcher, detections_to_candidates, generate_report,
               plot_heatmap, save_diagnostic_cutouts):
    """Tier 2: Cross-survey validation with reprojection."""

    from astroworld.imaging.reprojection import reproject_to_common_frame

    # 1. Get Sedna's position at both epochs
    print("[1/7] Querying JPL Horizons for Sedna at both epochs...")

    sedna_1 = fetch_sedna_position(_DSS2_EPOCH_JD)
    sedna_2 = fetch_sedna_position(_SDSS_EPOCH_JD)

    print(f"  Epoch 1 ({args.survey1}, ~1993):")
    print(f"    RA={sedna_1['ra_deg']:.4f}, Dec={sedna_1['dec_deg']:.4f}, "
          f"V={sedna_1['V_mag']:.1f}")
    print(f"  Epoch 2 ({args.survey2}, ~2005):")
    print(f"    RA={sedna_2['ra_deg']:.4f}, Dec={sedna_2['dec_deg']:.4f}, "
          f"V={sedna_2['V_mag']:.1f}")

    # Motion between epochs
    motion = compute_inter_epoch_motion(
        "90377", _DSS2_EPOCH_JD, _SDSS_EPOCH_JD,
    )
    print(f"  Inter-epoch motion: {motion['separation_arcsec']:.1f} arcsec "
          f"({motion['separation_pixels']:.1f} px @ 1.7\"/px)")
    print()

    # Save ephemeris
    eph_df = pd.DataFrame([
        {**sedna_1, "epoch": "survey1", "jd": _DSS2_EPOCH_JD},
        {**sedna_2, "epoch": "survey2", "jd": _SDSS_EPOCH_JD},
    ])
    eph_df.to_csv(args.output_dir / "sedna_ephemeris.csv", index=False)

    # 2. Download both surveys (centered on mean position)
    ra_mean = (sedna_1["ra_deg"] + sedna_2["ra_deg"]) / 2.0
    dec_mean = (sedna_1["dec_deg"] + sedna_2["dec_deg"]) / 2.0

    print(f"[2/7] Downloading {args.survey1} at mean position...")
    dir_1 = args.output_dir / "fits_epoch1"
    paths_1 = download_field(
        ra_deg=ra_mean, dec_deg=dec_mean,
        surveys=[args.survey1],
        radius_arcmin=args.radius_arcmin,
        pixels=args.pixels,
        output_dir=dir_1,
    )

    print(f"[3/7] Downloading {args.survey2} at mean position...")
    dir_2 = args.output_dir / "fits_epoch2"
    paths_2 = download_field(
        ra_deg=ra_mean, dec_deg=dec_mean,
        surveys=[args.survey2],
        radius_arcmin=args.radius_arcmin,
        pixels=args.pixels,
        output_dir=dir_2,
    )

    if not paths_1 or not paths_2:
        print("  ERROR: Failed to download one or both surveys.")
        sys.exit(1)

    print(f"  Epoch 1: {paths_1[0]}")
    print(f"  Epoch 2: {paths_2[0]}")
    print()

    # 3. Load and reproject
    print("[4/7] Reprojecting to common WCS frame...")
    img_1, hdr_1 = load_fits_image(paths_1[0])
    img_2, hdr_2 = load_fits_image(paths_2[0])

    reproj_1, reproj_2, common_hdr = reproject_to_common_frame(
        img_1, hdr_1,
        img_2, hdr_2,
        target_shape=(args.pixels, args.pixels),
    )
    print(f"  Output shape: {reproj_1.shape}")
    print(f"  Non-zero epoch 1: {np.count_nonzero(reproj_1)} px")
    print(f"  Non-zero epoch 2: {np.count_nonzero(reproj_2)} px")
    print()

    # 4. Locate Sedna in common frame
    print("[5/7] Locating Sedna in common frame...")
    try:
        px1, py1 = radec_to_pixel(sedna_1["ra_deg"], sedna_1["dec_deg"], common_hdr)
        px2, py2 = radec_to_pixel(sedna_2["ra_deg"], sedna_2["dec_deg"], common_hdr)
        print(f"  Sedna @ epoch 1: pixel ({px1:.1f}, {py1:.1f})")
        print(f"  Sedna @ epoch 2: pixel ({px2:.1f}, {py2:.1f})")
        print(f"  Pixel motion: {np.sqrt((px2-px1)**2 + (py2-py1)**2):.1f} px")
    except ValueError:
        print("  WARNING: Sedna outside common frame bounds.")
        px1 = py1 = px2 = py2 = float("nan")
    print()

    # 5. Run search
    print("[6/7] Running Siamese detector on reprojected pair...")
    searcher = P9Searcher.from_checkpoint(
        args.checkpoint,
        patch_size=args.patch_size,
        overlap=args.overlap,
        threshold=args.threshold,
        device=args.device,
    )

    result = searcher.search_from_arrays(
        reproj_1, reproj_2,
        image_path_1=f"reproj_{args.survey1}",
        image_path_2=f"reproj_{args.survey2}",
    )

    candidates = detections_to_candidates(
        result.patch_detections, common_hdr, result.grid_shape,
    )
    result.candidates = candidates

    n_cands = len(candidates)
    print(f"  Candidates: {n_cands}")
    print()

    # 6. Evaluate
    print("[7/7] Evaluating Tier 2 results...")
    print("-" * 60)

    sedna_detected = False
    best_dist = float("inf")
    best_prob = 0.0

    for cand in candidates:
        for sx, sy in [(px1, py1), (px2, py2)]:
            if np.isnan(sx):
                continue
            dist = np.sqrt((cand.pixel_x - sx)**2 + (cand.pixel_y - sy)**2)
            if dist < best_dist:
                best_dist = dist
                best_prob = cand.probability
            if dist < 32:
                sedna_detected = True

    print()
    print("=" * 60)
    print("  TIER 2 CROSS-SURVEY VALIDATION RESULTS")
    print("=" * 60)
    print()
    print(f"  Surveys:     {args.survey1} vs {args.survey2}")
    print(f"  Motion:      {motion['separation_arcsec']:.1f} arcsec")
    print(f"  Candidates:  {n_cands}")
    print(f"  Nearest to Sedna: {best_dist:.1f} px (p={best_prob:.3f})")
    print()

    if sedna_detected:
        print("  >>> SEDNA DETECTED IN CROSS-SURVEY MODE <<<")
    else:
        print("  >>> SEDNA NOT DETECTED IN CROSS-SURVEY <<<")
        print("  Cross-survey detection is challenging due to:")
        print("    - PSF differences between surveys")
        print("    - Background noise level differences")
        print("    - Large inter-epoch motion exceeding patch size")

    print()

    # Save outputs
    heatmap_path = args.output_dir / "heatmap_tier2.png"
    plot_heatmap(
        result, reproj_2, heatmap_path,
        title=f"Sedna Tier 2 | {args.survey1} vs {args.survey2}",
    )
    print(f"  Heatmap saved: {heatmap_path}")

    report_path = args.output_dir / "detection_report.csv"
    generate_report([result], report_path)
    print(f"  Report saved:  {report_path}")


if __name__ == "__main__":
    main()
