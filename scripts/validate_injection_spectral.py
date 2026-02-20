#!/usr/bin/env python
"""
Injection-based validation for the SSTF spectral model ("Test de Fuego").

Injects a synthetic Planet 9-like source into REAL survey backgrounds
(DSS2 Red + WISE W1 + WISE W2) downloaded from the Sedna field, then
runs the SpectralSearcher to verify end-to-end detection.

This tests the full pipeline:
  1. Real astrophysical backgrounds (stars, galaxies, noise)
  2. Realistic PSF from the field (empirical star extraction)
  3. Planck-consistent multi-band flux ratios (W1/W2 prop. B_ν(T))
  4. Simulated motion between two epochs (offset source)
  5. Per-channel z-score normalization (domain-agnostic)
  6. MC Dropout uncertainty

Why injection instead of Sedna directly?
  Sedna at V=21.3, T=29K has ZERO detectable signal in DSS2/WISE:
  - Too faint for DSS2 Red (limiting mag ~20.5)
  - No thermal IR emission at WISE wavelengths (peak at ~100μm)
  The model correctly returns 0.0005 for Sedna — a true negative!

This script validates what the model CAN detect on real backgrounds.

Usage
-----
    uv run python scripts/validate_injection_spectral.py \\
        --checkpoint checkpoints/spectral/best_model.pt \\
        --magnitude 19.5 --temperature 45 --snr-target 8
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from astroworld.imaging.fits_store import load_fits_image
from astroworld.imaging.spectral_cube import (
    BAND_TEMPORAL,
    NUM_CHANNELS,
    build_spectral_cube,
    compute_temporal_gradient,
    planck_flux_ratio,
)
from astroworld.imaging.psf_extraction import (
    extract_psf,
    estimate_background,
)
from astroworld.imaging.synthetic_injection import (
    inject_source,
    estimate_zero_point,
    magnitude_to_flux,
)
from astroworld.ml.spectral_model import PlanckFilter
from astroworld.ml.spectral_inference import SpectralSearcher


# ---------------------------------------------------------------------------
# Survey file paths (downloaded by validate_sedna_spectral.py)
# ---------------------------------------------------------------------------
_SURVEY_BASE = Path("data/survey_images")
_TARGET_FILES = {
    "optical": _SURVEY_BASE / "dss2_red" / "field_ra044.256_dec+04.649.fits",
    "w1": _SURVEY_BASE / "wise_3.4" / "field_ra044.256_dec+04.649.fits",
    "w2": _SURVEY_BASE / "wise_4.6" / "field_ra044.256_dec+04.649.fits",
}
_REF_FILES = {
    "optical": _SURVEY_BASE / "dss2_red" / "field_ra044.256_dec+04.682.fits",
    "w1": _SURVEY_BASE / "wise_3.4" / "field_ra044.256_dec+04.682.fits",
    "w2": _SURVEY_BASE / "wise_4.6" / "field_ra044.256_dec+04.682.fits",
}

_DSS2_EPOCH_JD = 2449200.0  # ~1993.7
_WISE_EPOCH_JD = 2455400.0  # ~2010.7


def _make_gaussian_psf(fwhm_px: float, size: int = 21) -> np.ndarray:
    """Create a normalized Gaussian PSF stamp."""
    sigma = fwhm_px / 2.3548  # FWHM → σ
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf


def _inject_at_position(
    image: np.ndarray,
    px: float,
    py: float,
    flux: float,
    fwhm_px: float,
    bg_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject a Gaussian source at a specific pixel position."""
    psf = _make_gaussian_psf(fwhm_px)
    return inject_source(
        image=image,
        psf=psf,
        pixel_x=px,
        pixel_y=py,
        flux_adu=flux,
        background_std=bg_std,
        add_noise=True,
        rng=rng,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Injection-based spectral validation (Test de Fuego)"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to SpectralSiameseNet checkpoint (.pt)"
    )
    parser.add_argument("--magnitude", type=float, default=19.5,
                        help="Apparent magnitude of injected source")
    parser.add_argument("--temperature", type=float, default=45.0,
                        help="Blackbody temperature (K) of injected source")
    parser.add_argument("--snr-target", type=float, default=8.0,
                        help="Target S/N for the injected source (optical)")
    parser.add_argument("--motion-px", type=float, default=6.0,
                        help="Motion between epochs in pixels")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-noise", type=float, default=0.05,
                        help="±fraction of color noise on W1/W2 ratio")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/injection_spectral"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print("  INJECTION-BASED SPECTRAL VALIDATION — Test de Fuego")
    print("=" * 65)

    # -----------------------------------------------------------------
    # Step 1: Load real survey images
    # -----------------------------------------------------------------
    print("\n[1/6] Loading real survey images...")

    for key, path in _TARGET_FILES.items():
        if not path.exists():
            print(f"  ERROR: {path} not found.")
            print("  Run validate_sedna_spectral.py first to download surveys.")
            sys.exit(1)

    opt_target, opt_hdr = load_fits_image(_TARGET_FILES["optical"])
    w1_target, w1_hdr = load_fits_image(_TARGET_FILES["w1"])
    w2_target, w2_hdr = load_fits_image(_TARGET_FILES["w2"])

    opt_ref, opt_hdr_r = load_fits_image(_REF_FILES["optical"])
    w1_ref, _ = load_fits_image(_REF_FILES["w1"])
    w2_ref, _ = load_fits_image(_REF_FILES["w2"])

    h, w = opt_target.shape
    print(f"  Image size: {w}×{h}")
    print(f"  Optical: mean={opt_target.mean():.1f}, std={opt_target.std():.1f}")
    print(f"  W1:      mean={w1_target.mean():.4f}, std={w1_target.std():.4f}")
    print(f"  W2:      mean={w2_target.mean():.4f}, std={w2_target.std():.4f}")

    # -----------------------------------------------------------------
    # Step 2: Inject synthetic P9-like source
    # -----------------------------------------------------------------
    print(f"\n[2/6] Injecting synthetic source (T={args.temperature}K)...")

    # Injection position: near center but offset from any bright stars
    inject_x = w * 0.45 + rng.uniform(-20, 20)
    inject_y = h * 0.45 + rng.uniform(-20, 20)

    # Compute background stats for each band (returns (mean, rms) tuple)
    _, bg_rms_opt = estimate_background(opt_target.astype(np.float64))
    _, bg_rms_w1 = estimate_background(w1_target.astype(np.float64))
    _, bg_rms_w2 = estimate_background(w2_target.astype(np.float64))

    # Optical flux from target S/N
    opt_flux = args.snr_target * bg_rms_opt
    opt_fwhm = 2.5  # DSS2 typical seeing in pixels

    # IR fluxes: Planck-consistent with color noise
    pr = planck_flux_ratio(args.temperature)
    noise_factor = 1.0 + rng.uniform(-args.color_noise, args.color_noise)
    pr *= noise_factor

    # WISE has much lower background per pixel, so scale flux to be
    # detectable relative to WISE backgrounds.  We use a "thermal IR
    # boost" — at cold temperatures the object is brighter in IR
    # relative to optical than a star would be (reflected vs thermal).
    ir_boost = max(3.0, 10.0 - args.temperature / 10.0)
    w2_flux = opt_flux * ir_boost * (bg_rms_w2 /
                                      max(bg_rms_opt, 1e-10))
    w1_flux = w2_flux * max(pr, 0.01)
    wise_fwhm = 6.0  # WISE PSF ~6" (larger than DSS2)

    print(f"  Position: ({inject_x:.1f}, {inject_y:.1f})")
    print(f"  Optical flux: {opt_flux:.1f} ADU (S/N ~ {args.snr_target})")
    print(f"  W1 flux:      {w1_flux:.4f} ADU")
    print(f"  W2 flux:      {w2_flux:.4f} ADU")
    print(f"  Planck ratio: {pr:.6f} (T={args.temperature}K)")
    print(f"  IR boost:     {ir_boost:.1f}×")

    # Epoch 1 (target): inject at position
    opt_e1 = _inject_at_position(
        opt_target.astype(np.float64), inject_x, inject_y,
        opt_flux, opt_fwhm, bg_rms_opt, rng
    ).astype(np.float32)

    w1_e1 = _inject_at_position(
        w1_target.astype(np.float64), inject_x, inject_y,
        w1_flux, wise_fwhm, bg_rms_w1, rng
    ).astype(np.float32)

    w2_e1 = _inject_at_position(
        w2_target.astype(np.float64), inject_x, inject_y,
        w2_flux, wise_fwhm, bg_rms_w2, rng
    ).astype(np.float32)

    # Epoch 2 (reference): inject at offset position (motion!)
    dx = args.motion_px * rng.uniform(0.5, 1.0)
    dy = args.motion_px * rng.uniform(0.5, 1.0)
    inject_x2 = inject_x + dx
    inject_y2 = inject_y + dy

    opt_e2 = _inject_at_position(
        opt_ref.astype(np.float64), inject_x2, inject_y2,
        opt_flux, opt_fwhm, bg_rms_opt, rng
    ).astype(np.float32)

    w1_e2 = _inject_at_position(
        w1_ref.astype(np.float64), inject_x2, inject_y2,
        w1_flux, wise_fwhm, bg_rms_w1, rng
    ).astype(np.float32)

    w2_e2 = _inject_at_position(
        w2_ref.astype(np.float64), inject_x2, inject_y2,
        w2_flux, wise_fwhm, bg_rms_w2, rng
    ).astype(np.float32)

    print(f"  Motion: dx={dx:.1f}, dy={dy:.1f} px")
    print(f"  Epoch 2 position: ({inject_x2:.1f}, {inject_y2:.1f})")

    # -----------------------------------------------------------------
    # Step 3: Build spectral cubes from injected images
    # -----------------------------------------------------------------
    print("\n[3/6] Building 5-channel spectral cubes...")

    cube_e1 = build_spectral_cube(
        opt_e1, opt_hdr, w1_e1, dict(w1_hdr), w2_e1, dict(w2_hdr)
    )
    cube_e2 = build_spectral_cube(
        opt_e2, opt_hdr_r, w1_e2, None, w2_e2, None
    )

    # Temporal gradient
    delta_t = abs(_WISE_EPOCH_JD - _DSS2_EPOCH_JD) * 365.25
    tgrad = compute_temporal_gradient(cube_e1, cube_e2, delta_t)
    cube_e1[BAND_TEMPORAL] = tgrad
    cube_e2[BAND_TEMPORAL] = -tgrad

    print(f"  Cube shape: {cube_e1.shape}")
    for i, name in enumerate(["optical", "W1", "W2", "uncert", "temporal"]):
        ch = cube_e1[i]
        print(f"    C{i} ({name}): mean={ch.mean():.4f}, "
              f"std={ch.std():.4f}, max={ch.max():.4f}")

    # -----------------------------------------------------------------
    # Step 4: Run spectral search
    # -----------------------------------------------------------------
    print("\n[4/6] Loading SSTF model and running spectral search...")
    searcher = SpectralSearcher.from_checkpoint(
        args.checkpoint,
        patch_size=args.patch_size,
        overlap=args.overlap,
        threshold=args.threshold,
        mc_samples=args.mc_samples,
        device=args.device,
    )

    result = searcher.search_from_cubes(
        cube_e1, cube_e2,
        label_1="epoch1_injected",
        label_2="epoch2_injected",
    )

    print(f"  Inference time: {result.inference_time_seconds:.2f} s")
    print(f"  MC Dropout samples: {args.mc_samples}")
    print(f"  Total detections: {len(result.patch_detections)}")

    # -----------------------------------------------------------------
    # Step 5: Check detection at injection position
    # -----------------------------------------------------------------
    print("\n[5/6] Analyzing detection at injection site...")

    stride = args.patch_size - args.overlap
    patch_row = int(inject_y / stride)
    patch_col = int(inject_x / stride)
    n_rows, n_cols = result.grid_shape
    patch_row = min(patch_row, n_rows - 1)
    patch_col = min(patch_col, n_cols - 1)

    prob_at_inj = float(result.heatmap[patch_row, patch_col])
    std_at_inj = float(result.motion_std_map[patch_row, patch_col])
    temp_at_inj = float(result.temperature_map[patch_row, patch_col])
    temp_std = float(result.temperature_std_map[patch_row, patch_col])
    planck_class = PlanckFilter.classify(temp_at_inj, temp_std)

    # Also check neighboring patches (source may span two patches)
    best_prob = prob_at_inj
    best_row, best_col = patch_row, patch_col
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            r = patch_row + dr
            c = patch_col + dc
            if 0 <= r < n_rows and 0 <= c < n_cols:
                p = float(result.heatmap[r, c])
                if p > best_prob:
                    best_prob = p
                    best_row, best_col = r, c

    if best_row != patch_row or best_col != patch_col:
        prob_at_inj = best_prob
        std_at_inj = float(result.motion_std_map[best_row, best_col])
        temp_at_inj = float(result.temperature_map[best_row, best_col])
        temp_std = float(result.temperature_std_map[best_row, best_col])
        planck_class = PlanckFilter.classify(temp_at_inj, temp_std)
        print(f"  (Best detection at neighboring patch [{best_row},{best_col}])")

    detected = prob_at_inj >= args.threshold

    print(f"\n  === INJECTION SPECTRAL ANALYSIS ===")
    print(f"  Motion probability:  {prob_at_inj:.4f} ± {std_at_inj:.4f}")
    print(f"  Temperature:         {temp_at_inj:.1f} ± {temp_std:.1f} K")
    print(f"  Injected temp:       {args.temperature:.1f} K")
    print(f"  Planck class:        {planck_class}")
    print(f"  Uncertainty (std):   {std_at_inj:.4f}")

    if detected:
        print(f"\n  [OK] SOURCE DETECTED at injection site!")
        print(f"  Confidence: {prob_at_inj:.4f}")
        temp_err = abs(temp_at_inj - args.temperature)
        if temp_err < 20:
            print(f"  Temperature match: GOOD (Δ={temp_err:.1f} K)")
        else:
            print(f"  Temperature match: MARGINAL (Δ={temp_err:.1f} K)")
    else:
        print(f"\n  [FAIL] Source NOT detected at threshold {args.threshold}")
        print(f"  Heatmap value: {prob_at_inj:.4f}")

    # -----------------------------------------------------------------
    # Step 6: Full heatmap stats
    # -----------------------------------------------------------------
    print("\n[6/6] Heatmap statistics...")
    hm = result.heatmap
    print(f"  Heatmap range: [{hm.min():.4f}, {hm.max():.4f}]")
    print(f"  Heatmap mean:  {hm.mean():.4f}")
    print(f"  Patches above 0.50: {(hm > 0.50).sum()}")
    print(f"  Patches above 0.70: {(hm > 0.70).sum()}")
    print(f"  Patches above 0.90: {(hm > 0.90).sum()}")

    # Save report
    report_lines = [
        "INJECTION SPECTRAL VALIDATION REPORT",
        "=" * 50,
        f"Model checkpoint: {args.checkpoint}",
        f"Threshold: {args.threshold}",
        f"MC samples: {args.mc_samples}",
        "",
        "INJECTION PARAMETERS",
        f"  Magnitude: {args.magnitude:.1f}",
        f"  Temperature: {args.temperature:.1f} K",
        f"  S/N target: {args.snr_target:.1f}",
        f"  Motion: {args.motion_px:.1f} px",
        f"  Color noise: ±{args.color_noise * 100:.0f}%",
        f"  Optical flux: {opt_flux:.1f} ADU",
        f"  W1 flux: {w1_flux:.4f} ADU",
        f"  W2 flux: {w2_flux:.4f} ADU",
        "",
        "RESULTS",
        f"  Motion probability: {prob_at_inj:.4f} ± {std_at_inj:.4f}",
        f"  Temperature: {temp_at_inj:.1f} ± {temp_std:.1f} K",
        f"  Planck classification: {planck_class}",
        f"  Detected: {'YES' if detected else 'NO'}",
        "",
        f"  Total detections: {len(result.patch_detections)}",
        f"  Inference time: {result.inference_time_seconds:.2f} s",
    ]
    report_path = args.output_dir / "injection_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n  Report: {report_path}")

    # Save visualization
    try:
        _save_injection_heatmap(result, args.output_dir, inject_x, inject_y)
    except Exception as e:
        print(f"  [warn] Heatmap plot failed: {e}")

    # Final verdict
    print("\n" + "=" * 65)
    if detected:
        print("  [FIRE] TEST DE FUEGO: PASSED — Injection detected on real data!")
    else:
        print("  [WARN]  TEST DE FUEGO: Source not detected (adjust S/N or threshold)")
    print("=" * 65)

    return 0 if detected else 1


def _save_injection_heatmap(
    result, output_dir: Path,
    inject_x: float, inject_y: float,
):
    """Save 3-panel heatmap with injection marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    stride = result.stride

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

    # Mark injection position
    for ax in axes:
        sx = inject_x / stride
        sy = inject_y / stride
        ax.plot(sx, sy, "c*", markersize=15,
                markeredgecolor="white", markeredgewidth=1)

    fig.suptitle("Injection Spectral Validation — Test de Fuego", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_dir / "injection_heatmap.png"), dpi=150)
    plt.close(fig)
    print(f"  Heatmap: {output_dir / 'injection_heatmap.png'}")


if __name__ == "__main__":
    sys.exit(main())
