#!/usr/bin/env python
"""
Sedna positive-control validation with the SSTF spectral model.

Downloads optical (DSS2 Red) + infrared (WISE W1, W2) images for Sedna's
known position, builds 5-channel spectral cubes, and runs the
SpectralSearcher to verify:
  1. Motion detection at Sedna's position
  2. Temperature estimation consistent with Sedna's distance (~93 AU, T~29K)
  3. Planck classification = "p9_candidate"

Usage
-----
    uv run python scripts/validate_sedna_spectral.py \\
        --checkpoint checkpoints/spectral/best_model.pt \\
        --threshold 0.70 --mc-samples 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from astroworld.imaging.ephemeris import fetch_sedna_position
from astroworld.imaging.survey_download import download_field
from astroworld.imaging.fits_store import load_fits_image
from astroworld.imaging.spectral_cube import (
    build_spectral_cube,
    compute_temporal_gradient,
    estimate_color_temperature,
    BAND_TEMPORAL,
)
from astroworld.imaging.synthetic_injection import radec_to_pixel
from astroworld.ml.spectral_model import PlanckFilter
from astroworld.ml.spectral_inference import SpectralSearcher


# Epoch constants
_DSS2_EPOCH_JD = 2449200.0     # ~1993.7 (DSS2 Red mean epoch)
_WISE_EPOCH_JD = 2455400.0     # ~2010.7 (WISE all-sky survey)

# WISE surveys via SkyView
_OPTICAL_SURVEY = "DSS2 Red"
_W1_SURVEY = "WISE 3.4"
_W2_SURVEY = "WISE 4.6"


def main():
    parser = argparse.ArgumentParser(
        description="Sedna spectral validation for SSTF model"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to SpectralSiameseNet checkpoint (.pt)"
    )
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--offset-arcmin", type=float, default=2.0)
    parser.add_argument("--radius-arcmin", type=float, default=5.0)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/sedna_spectral"))

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SEDNA SPECTRAL VALIDATION — SSTF Model")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Step 1: Get Sedna's position
    # -----------------------------------------------------------------
    print("\n[1/6] Querying JPL Horizons for Sedna position...")
    sedna_info = fetch_sedna_position(_DSS2_EPOCH_JD)
    ra_sedna = sedna_info["ra_deg"]
    dec_sedna = sedna_info["dec_deg"]
    v_mag = sedna_info.get("V_mag", 21.0)
    delta_au = sedna_info.get("delta_au", 93.0)

    print(f"  Sedna at epoch JD={_DSS2_EPOCH_JD:.1f}:")
    print(f"    RA  = {ra_sedna:.6f} deg")
    print(f"    Dec = {dec_sedna:.6f} deg")
    print(f"    V   = {v_mag:.1f} mag")
    print(f"    d   = {delta_au:.1f} AU")

    # Expected temperature
    t_expected = PlanckFilter.expected_p9_temperature(delta_au)
    print(f"    T_expected = {t_expected:.1f} K (radiative equilibrium)")

    # Save ephemeris
    eph_df = pd.DataFrame([{
        "epoch_jd": _DSS2_EPOCH_JD,
        "ra_deg": ra_sedna, "dec_deg": dec_sedna,
        "V_mag": v_mag, "delta_au": delta_au,
        "T_expected_K": t_expected,
    }])
    eph_df.to_csv(args.output_dir / "sedna_ephemeris.csv", index=False)

    # -----------------------------------------------------------------
    # Step 2: Download 3 surveys (target field)
    # -----------------------------------------------------------------
    print("\n[2/6] Downloading survey images (target field)...")
    surveys = [_OPTICAL_SURVEY, _W1_SURVEY, _W2_SURVEY]
    target_paths = download_field(
        ra_deg=ra_sedna, dec_deg=dec_sedna,
        surveys=surveys,
        radius_arcmin=args.radius_arcmin,
        pixels=args.pixels,
    )

    if len(target_paths) < 1:
        print("ERROR: Could not download optical image for Sedna.")
        sys.exit(1)

    # Download reference field (offset, no Sedna)
    print("\n[3/6] Downloading survey images (reference field)...")
    ra_ref = ra_sedna
    dec_ref = dec_sedna + args.offset_arcmin / 60.0
    ref_paths = download_field(
        ra_deg=ra_ref, dec_deg=dec_ref,
        surveys=surveys,
        radius_arcmin=args.radius_arcmin,
        pixels=args.pixels,
    )

    if len(ref_paths) < 1:
        print("ERROR: Could not download reference field.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Step 3: Build spectral cubes
    # -----------------------------------------------------------------
    print("\n[4/6] Building 5-channel spectral cubes...")

    target_cube, target_header = _build_cube_from_paths(target_paths, surveys)
    ref_cube, ref_header = _build_cube_from_paths(ref_paths, surveys)

    # Fill temporal gradient (C4): target vs reference
    delta_t = abs(_WISE_EPOCH_JD - _DSS2_EPOCH_JD) * 365.25  # ~17 years in days
    if delta_t > 0:
        tgrad = compute_temporal_gradient(ref_cube, target_cube, delta_t)
        ref_cube[BAND_TEMPORAL] = -tgrad
        target_cube[BAND_TEMPORAL] = tgrad

    print(f"  Target cube shape: {target_cube.shape}")
    print(f"  Reference cube shape: {ref_cube.shape}")

    # -----------------------------------------------------------------
    # Step 4: Load model and run spectral search
    # -----------------------------------------------------------------
    print("\n[5/6] Loading SSTF model and running spectral search...")
    searcher = SpectralSearcher.from_checkpoint(
        args.checkpoint,
        patch_size=args.patch_size,
        overlap=args.overlap,
        threshold=args.threshold,
        mc_samples=args.mc_samples,
        device=args.device,
    )

    result = searcher.search_from_cubes(
        ref_cube, target_cube,
        label_1="reference_field",
        label_2="sedna_target_field",
    )

    print(f"  Inference time: {result.inference_time_seconds:.2f} s")
    print(f"  MC Dropout samples: {args.mc_samples}")
    print(f"  Detections above threshold: {len(result.patch_detections)}")

    # -----------------------------------------------------------------
    # Step 5: Check detection at Sedna's position
    # -----------------------------------------------------------------
    print("\n[6/6] Analyzing results...")

    # Locate Sedna in pixel coords
    try:
        sedna_px, sedna_py = radec_to_pixel(
            ra_sedna, dec_sedna, target_header
        )
        print(f"  Sedna pixel position: ({sedna_px:.1f}, {sedna_py:.1f})")
    except Exception:
        sedna_px = args.pixels / 2.0
        sedna_py = args.pixels / 2.0
        print(f"  WCS conversion failed, using image center")

    # Check heatmap at Sedna's position
    stride = args.patch_size - args.overlap
    patch_row = int(sedna_py / stride)
    patch_col = int(sedna_px / stride)
    n_rows, n_cols = result.grid_shape
    patch_row = min(patch_row, n_rows - 1)
    patch_col = min(patch_col, n_cols - 1)

    heatmap_at_sedna = float(result.heatmap[patch_row, patch_col])
    motion_std = float(result.motion_std_map[patch_row, patch_col])
    temp_at_sedna = float(result.temperature_map[patch_row, patch_col])
    temp_std = float(result.temperature_std_map[patch_row, patch_col])

    # Planck classification
    planck_class = PlanckFilter.classify(temp_at_sedna, temp_std)

    print(f"\n  === SEDNA SPECTRAL ANALYSIS ===")
    print(f"  Motion probability:  {heatmap_at_sedna:.4f} +/- {motion_std:.4f}")
    print(f"  Temperature:         {temp_at_sedna:.1f} +/- {temp_std:.1f} K")
    print(f"  Expected temp:       {t_expected:.1f} K")
    print(f"  Planck class:        {planck_class}")

    sedna_detected = heatmap_at_sedna > args.threshold
    if sedna_detected:
        print(f"\n  >>> SEDNA DETECTED <<<")
        print(f"  Spectral confidence: {heatmap_at_sedna:.4f}")
        print(f"  Temperature match: {'YES' if abs(temp_at_sedna - t_expected) < 20 else 'MARGINAL'}")
    else:
        print(f"\n  Sedna NOT detected at threshold {args.threshold}")
        print(f"  Heatmap value: {heatmap_at_sedna:.4f}")

    # -----------------------------------------------------------------
    # Generate report
    # -----------------------------------------------------------------
    report_lines = [
        "SEDNA SPECTRAL VALIDATION REPORT",
        "=" * 50,
        f"Model checkpoint: {args.checkpoint}",
        f"Threshold: {args.threshold}",
        f"MC Dropout samples: {args.mc_samples}",
        "",
        "SEDNA POSITION",
        f"  RA: {ra_sedna:.6f} deg",
        f"  Dec: {dec_sedna:.6f} deg",
        f"  V mag: {v_mag:.1f}",
        f"  Distance: {delta_au:.1f} AU",
        "",
        "SPECTRAL RESULTS",
        f"  Motion probability: {heatmap_at_sedna:.4f} +/- {motion_std:.4f}",
        f"  Temperature: {temp_at_sedna:.1f} +/- {temp_std:.1f} K",
        f"  Expected temperature: {t_expected:.1f} K",
        f"  Planck classification: {planck_class}",
        f"  Detected: {'YES' if sedna_detected else 'NO'}",
        "",
        f"Total detections: {len(result.patch_detections)}",
        f"Inference time: {result.inference_time_seconds:.2f} s",
    ]

    report_path = args.output_dir / "spectral_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n  Report saved: {report_path}")

    # Save heatmap visualization
    try:
        _save_spectral_heatmap(result, args.output_dir, sedna_px, sedna_py)
    except Exception as e:
        print(f"  [warn] Could not save heatmap: {e}")

    print("\n" + "=" * 60)
    print("  Spectral Validation Complete")
    print("=" * 60)


def _build_cube_from_paths(
    paths: list[Path],
    surveys: list[str],
) -> tuple[np.ndarray, dict]:
    """Build spectral cube from downloaded FITS paths."""
    optical_image = None
    optical_header = None
    w1_image, w1_header = None, None
    w2_image, w2_header = None, None

    for i, (path, survey) in enumerate(zip(paths, surveys)):
        try:
            img, hdr = load_fits_image(path)
            if survey == _OPTICAL_SURVEY:
                optical_image = img
                optical_header = dict(hdr)
            elif survey == _W1_SURVEY:
                w1_image = img
                w1_header = dict(hdr)
            elif survey == _W2_SURVEY:
                w2_image = img
                w2_header = dict(hdr)
        except Exception as e:
            print(f"  [warn] Could not load {survey}: {e}")

    if optical_image is None:
        raise RuntimeError("Optical image is required")

    cube = build_spectral_cube(
        optical_image, optical_header,
        w1_image, w1_header,
        w2_image, w2_header,
    )

    return cube, optical_header


def _save_spectral_heatmap(
    result,
    output_dir: Path,
    sedna_px: float,
    sedna_py: float,
):
    """Save spectral heatmap + temperature map visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Motion heatmap
    im0 = axes[0].imshow(
        result.heatmap, origin="lower", cmap="hot",
        vmin=0.0, vmax=1.0, aspect="equal",
    )
    axes[0].set_title("Motion Probability")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Temperature map
    im1 = axes[1].imshow(
        result.temperature_map, origin="lower", cmap="coolwarm",
        aspect="equal",
    )
    axes[1].set_title("Temperature (K)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Uncertainty map
    im2 = axes[2].imshow(
        result.motion_std_map, origin="lower", cmap="viridis",
        aspect="equal",
    )
    axes[2].set_title("Motion Uncertainty (std)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # Mark Sedna position
    stride = result.stride
    for ax in axes:
        sx = sedna_px / stride
        sy = sedna_py / stride
        ax.plot(sx, sy, "c*", markersize=15, markeredgecolor="white",
                markeredgewidth=1)

    fig.suptitle("Sedna Spectral Validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_dir / "spectral_heatmap.png"), dpi=150)
    plt.close(fig)
    print(f"  Heatmap saved: {output_dir / 'spectral_heatmap.png'}")


if __name__ == "__main__":
    main()
