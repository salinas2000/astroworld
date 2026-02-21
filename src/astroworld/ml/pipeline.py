"""
Observatory Pipeline for Planet 9 detection.

Cascading multi-stage detector unifying all ML models (Phases 11-13):

  Stage 1: Fast optical scan (optional, P9Searcher on dual optical epochs)
  Stage 2: Intelligent IR extraction (download WISE W1/W2 from SkyView)
  Stage 3: Spectral validation (SpectralSearcher + MC Dropout + PlanckFilter)
  Stage 4: Kinematic filter (orbital mechanics plausibility check)
  Stage 5: Report generation (CSV, heatmaps, summary text)

The default ``spectral_only`` mode skips Stage 1 and processes each field
through Stages 2-5.  The SSTF model detects P9 candidates via spectral
temperature (W1/W2 ratio) even from single-epoch cubes, where the temporal
gradient channel is zero.

Usage::

    pipeline = ObservatoryPipeline.from_checkpoints(
        spectral_checkpoint="checkpoints/spectral_adapted/best_model.pt",
    )
    summary = pipeline.run(fields=[(71.3, 14.5), (180.0, -30.0)])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from astroworld.imaging.fits_store import load_fits_image
from astroworld.imaging.spectral_cube import (
    BAND_W1,
    BAND_W2,
    build_spectral_cube,
)
from astroworld.imaging.survey_download import download_field
from astroworld.ml.inference import (
    P9Searcher,
    PatchDetection,
    SearchResult,
)
from astroworld.ml.kinematics import filter_detections_kinematic
from astroworld.ml.post_processing import (
    detections_to_candidates,
    plot_heatmap,
)
from astroworld.ml.spectral_inference import (
    SpectralSearcher,
    SpectralSearchResult,
)
from astroworld.ml.spectral_model import PlanckFilter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for all pipeline thresholds and behavior.

    Parameters
    ----------
    mode : Pipeline mode.
        ``"spectral_only"`` (default) — skip optical pre-scan, use SSTF
        directly on downloaded survey images.
        ``"dual_optical"`` — download two optical surveys, run P9Searcher
        as a fast pre-filter before spectral analysis.
        ``"ir_only"`` — download only WISE W1+W2, build IR-anchored
        spectral cube, integrate Dust Piercer as primary validator.
        No optical dependency.  For objects invisible in optical
        (e.g. Planet 9 at ~25 mag, beyond DSS2 limit of ~21 mag).
        ``"dp_direct"`` — bypass SSTF ML model entirely.  Uses classic
        5-sigma peak finding on WISE W2 and validates directly with
        Dust Piercer morphology + NEOWISE proper motion.  Fastest mode,
        no neural network, pure algorithmic detection.
    """

    # --- Mode ---
    mode: str = "spectral_only"

    # --- Stage 1: Optical scan (dual_optical only) ---
    optical_threshold: float = 0.95
    optical_surveys: list[str] = field(
        default_factory=lambda: ["DSS2 Red", "DSS2 Blue"]
    )

    # --- Stage 2: IR download ---
    ir_surveys: list[str] = field(
        default_factory=lambda: ["WISE 3.4", "WISE 4.6"]
    )
    download_radius_arcmin: float = 5.0
    download_pixels: int = 512
    rate_limit_sec: float = 1.0

    # --- Stage 3: Spectral validation ---
    spectral_threshold: float = 0.80  # Sweet spot: ~4 FP/field at single-epoch
    mc_samples: int = 10
    planck_classes_keep: list[str] = field(
        default_factory=lambda: ["p9_candidate", "cold_object"]
    )
    ir_min_valid_fraction: float = 0.90  # Per-patch: reject if <90% IR pixels > 0

    # --- SNR photometric filter (noise rejection) ---
    snr_min: float = 3.0            # Minimum optical SNR to keep a candidate
    snr_aperture_radius: int = 3    # Aperture half-size for peak flux (pixels)
    snr_annulus_inner: int = 20     # Inner radius of background annulus (pixels)
    snr_annulus_outer: int = 30     # Outer radius of background annulus (pixels)

    # --- IR-only / dp_direct mode ---
    ir_snr_band: str = "W2"  # Band for SNR filter in ir_only mode (W1 fallback)
    dp_pointiness_threshold: float = 0.20  # Relaxed for small (fallback) images
    dp_peak_sigma: float = 5.0   # Peak detection threshold (n-sigma above background)
    dp_peak_neighborhood: int = 5  # Local maximum filter radius (pixels)
    dp_max_peaks: int = 50       # Cap on raw peaks per field (brightest kept)

    # --- Stage 4: Kinematic filter ---
    kinematic_enabled: bool = True
    kinematic_min_px: float = 0.5
    kinematic_max_px: float = 30.0

    # --- General ---
    patch_size: int = 64
    overlap: int = 8
    device: str = "auto"
    output_dir: Path = field(default_factory=lambda: Path("results/observatory"))
    data_dir: Path = field(default_factory=lambda: Path("data/survey_images"))
    verbose: bool = True


# ---------------------------------------------------------------------------
# Per-field result
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    """Result for a single sky field processed through the pipeline.

    Each candidate is a dict with keys: ``pixel_x``, ``pixel_y``,
    ``ra_deg``, ``dec_deg``, ``probability``, ``temperature_k``,
    ``temperature_std_k``, ``planck_class``, ``kinematic_valid``.
    """

    field_id: str
    ra_deg: float
    dec_deg: float

    # Stage 1: Optical pre-scan
    stage1_detections: int = 0
    stage1_passed: bool = True  # True if stage skipped or passed

    # Stage 2: Download
    ir_downloaded: bool = False
    has_w1: bool = False  # True if W1 image loaded with real data
    has_w2: bool = False  # True if W2 image loaded with real data
    optical_path: str | None = None
    w1_path: str | None = None
    w2_path: str | None = None

    # Stage 3: Spectral validation
    stage3_detections: int = 0
    stage3_candidates: list[dict] = field(default_factory=list)
    spectral_result: SpectralSearchResult | None = field(
        default=None, repr=False
    )

    # Stage 4: Kinematic filter
    stage4_survivors: int = 0

    # Final
    final_candidates: list[dict] = field(default_factory=list)

    # Stage 3b: Dust Piercer (ir_only mode)
    stage3b_dp_analyzed: int = 0
    stage3b_dp_verdicts: dict[str, int] = field(default_factory=dict)

    # Timing
    download_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    total_time_sec: float = 0.0

    # Errors
    error: str | None = None


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

@dataclass
class ObservatorySummary:
    """Aggregate results across all processed fields."""

    n_fields_total: int
    n_fields_processed: int
    n_fields_with_candidates: int
    n_fields_errored: int

    # Per-stage funnel counts
    n_stage1_passed: int
    n_stage3_detections: int
    n_stage3_planck_filtered: int
    n_stage4_survivors: int

    # Final
    total_candidates: int
    candidate_table: pd.DataFrame

    # Timing
    total_time_sec: float
    avg_time_per_field_sec: float
    download_time_sec: float
    inference_time_sec: float

    # Config & details
    config: PipelineConfig = field(repr=False)
    field_results: list[FieldResult] = field(repr=False)


# ---------------------------------------------------------------------------
# Observatory Pipeline
# ---------------------------------------------------------------------------

class ObservatoryPipeline:
    """End-to-end observatory pipeline for Planet 9 detection.

    Cascades through multiple detection stages, each acting as a filter.
    Only surviving candidates propagate to the next stage, minimizing
    expensive IR downloads and spectral inference.

    Parameters
    ----------
    config : PipelineConfig
    spectral_searcher : SpectralSearcher (loaded model)
    optical_searcher : P9Searcher or None (for dual_optical mode)
    """

    def __init__(
        self,
        config: PipelineConfig,
        spectral_searcher: SpectralSearcher | None = None,
        optical_searcher: P9Searcher | None = None,
    ):
        self.config = config
        self.spectral_searcher = spectral_searcher
        self.optical_searcher = optical_searcher

        if config.mode == "dual_optical" and optical_searcher is None:
            raise ValueError(
                "dual_optical mode requires optical_searcher. "
                "Provide optical_checkpoint to from_checkpoints()."
            )

        if config.mode != "dp_direct" and spectral_searcher is None:
            raise ValueError(
                f"mode={config.mode!r} requires spectral_searcher. "
                "Use dp_direct mode to bypass ML model, or provide "
                "spectral_checkpoint to from_checkpoints()."
            )

    @classmethod
    def from_checkpoints(
        cls,
        spectral_checkpoint: Path | str | None = None,
        optical_checkpoint: Path | str | None = None,
        config: PipelineConfig | None = None,
    ) -> ObservatoryPipeline:
        """Build pipeline from model checkpoint paths.

        Parameters
        ----------
        spectral_checkpoint : Path to SpectralSiameseNet ``best_model.pt``.
            Not required for ``dp_direct`` mode (no ML model used).
        optical_checkpoint : Path to P9SiameseDetector ``best_model.pt``.
            Required only for ``dual_optical`` mode.
        config : Pipeline configuration.  Defaults to ``spectral_only``.

        Returns
        -------
        Configured ObservatoryPipeline ready for inference.
        """
        if config is None:
            config = PipelineConfig()

        # dp_direct mode: no ML model needed
        spectral_searcher = None
        if config.mode != "dp_direct" and spectral_checkpoint is not None:
            spectral_searcher = SpectralSearcher.from_checkpoint(
                spectral_checkpoint,
                patch_size=config.patch_size,
                overlap=config.overlap,
                threshold=config.spectral_threshold,
                mc_samples=config.mc_samples,
                device=config.device,
            )

        optical_searcher = None
        if optical_checkpoint is not None:
            optical_searcher = P9Searcher.from_checkpoint(
                optical_checkpoint,
                patch_size=config.patch_size,
                overlap=config.overlap,
                threshold=config.optical_threshold,
                device=config.device,
            )

        return cls(config, spectral_searcher, optical_searcher)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        fields: list[tuple[float, float]],
    ) -> ObservatorySummary:
        """Process a list of sky fields through the full pipeline.

        Parameters
        ----------
        fields : List of ``(ra_deg, dec_deg)`` coordinate tuples.

        Returns
        -------
        ObservatorySummary with per-field results and aggregate metrics.
        """
        t_total = time.time()

        if self.config.verbose:
            self._print_header(len(fields))

        field_results: list[FieldResult] = []
        for i, (ra, dec) in enumerate(fields):
            if self.config.verbose:
                print(
                    f"  [{i + 1}/{len(fields)}] "
                    f"RA={ra:.3f}, Dec={dec:+.3f}"
                )

            fr = self.process_field(ra, dec)
            field_results.append(fr)

            if self.config.verbose:
                n_final = len(fr.final_candidates)
                if fr.error:
                    status = f"ERROR: {fr.error}"
                else:
                    status = f"{n_final} candidate(s)"
                print(f"    -> {status} ({fr.total_time_sec:.1f}s)")

        total_time = time.time() - t_total
        summary = self._build_summary(field_results, total_time)

        # Stage 5: Report
        self._stage5_report(summary)

        return summary

    def process_field(
        self,
        ra_deg: float,
        dec_deg: float,
    ) -> FieldResult:
        """Process a single field through all pipeline stages.

        Parameters
        ----------
        ra_deg, dec_deg : Sky coordinates (ICRS, degrees).

        Returns
        -------
        FieldResult with detections and metadata from each stage.
        """
        t0 = time.time()
        dec_str = f"{dec_deg:+07.3f}"
        field_id = f"field_ra{ra_deg:07.3f}_dec{dec_str}"
        result = FieldResult(
            field_id=field_id, ra_deg=ra_deg, dec_deg=dec_deg
        )

        try:
            # Stage 1: Optional optical pre-scan (dual_optical only)
            if self.config.mode == "dual_optical":
                result = self._stage1_optical_scan(result)
                if not result.stage1_passed:
                    result.total_time_sec = time.time() - t0
                    return result

            # Stage 2: Download images
            if self.config.mode in ("ir_only", "dp_direct"):
                result = self._stage2_download_ir_only(result)
            else:
                result = self._stage2_download(result)

            # Stage 3: Detection + validation
            if self.config.mode == "dp_direct":
                # DP-Direct: classic peak finding, no ML model
                if result.w2_path is None and result.w1_path is None:
                    result.error = "No IR image available"
                    result.total_time_sec = time.time() - t0
                    return result
                result = self._stage3_dp_direct(result)
            elif self.config.mode == "ir_only":
                # IR-only: require W1, SSTF on IR cube
                if result.w1_path is None:
                    result.error = "No W1 image available"
                    result.total_time_sec = time.time() - t0
                    return result
                result = self._stage3_ir_spectral(result)
                if result.stage3_detections == 0:
                    result.total_time_sec = time.time() - t0
                    return result
                # Stage 3b: Integrated Dust Piercer validation
                result = self._stage3b_dust_piercer(result)
            else:
                # Optical modes: require optical image
                if result.optical_path is None:
                    result.error = "No optical image available"
                    result.total_time_sec = time.time() - t0
                    return result
                result = self._stage3_spectral(result)
                if result.stage3_detections == 0:
                    result.total_time_sec = time.time() - t0
                    return result

            # Stage 4: Kinematic filter (same for all modes)
            result = self._stage4_kinematic(result)

        except Exception as e:
            result.error = str(e)
            if self.config.verbose:
                print(f"    [ERROR] {field_id}: {e}")

        result.total_time_sec = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # SNR photometric filter (noise rejection)
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_snr(
        image: np.ndarray,
        px: float,
        py: float,
        aperture_r: int = 3,
        annulus_inner: int = 20,
        annulus_outer: int = 30,
    ) -> float:
        """Compute robust aperture SNR for a candidate position.

        Uses MAD (Median Absolute Deviation) for background noise
        estimation — robust to cosmic rays and bright neighbours.

        Parameters
        ----------
        image : 2D array (full field image).
        px, py : Candidate center position in pixels.
        aperture_r : Half-size of square aperture for peak flux.
        annulus_inner, annulus_outer : Background annulus radii (pixels).

        Returns
        -------
        snr : float.  Peak pixel minus median background, divided by
              MAD-estimated sigma.  Returns 0.0 on error.
        """
        h, w = image.shape[-2], image.shape[-1]
        if image.ndim == 3:
            image = image[0]

        ix, iy = int(round(px)), int(round(py))

        # Aperture: small box around center
        ay0 = max(iy - aperture_r, 0)
        ay1 = min(iy + aperture_r + 1, h)
        ax0 = max(ix - aperture_r, 0)
        ax1 = min(ix + aperture_r + 1, w)
        aperture = image[ay0:ay1, ax0:ax1]

        if aperture.size == 0:
            return 0.0

        peak = float(np.max(aperture))

        # Background annulus: ring between inner and outer radius
        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - ix) ** 2 + (yy - iy) ** 2
        annulus_mask = (dist2 >= annulus_inner**2) & (dist2 <= annulus_outer**2)

        bg_pixels = image[annulus_mask]
        if bg_pixels.size < 10:
            # Fallback: use entire patch edges
            patch_size = 64
            py0 = max(iy - patch_size // 2, 0)
            px0 = max(ix - patch_size // 2, 0)
            py1 = min(py0 + patch_size, h)
            px1 = min(px0 + patch_size, w)
            patch = image[py0:py1, px0:px1]
            # Use border pixels (outer 5 rows/cols)
            border = np.concatenate([
                patch[:5].ravel(), patch[-5:].ravel(),
                patch[:, :5].ravel(), patch[:, -5:].ravel(),
            ])
            bg_pixels = border

        if bg_pixels.size == 0:
            return 0.0

        bg_median = float(np.median(bg_pixels))
        # MAD-based sigma: robust to outliers
        mad = float(np.median(np.abs(bg_pixels - bg_median)))
        sigma = 1.4826 * mad  # MAD to Gaussian sigma conversion

        if sigma < 1e-6:
            return 0.0

        return (peak - bg_median) / sigma

    # ------------------------------------------------------------------
    # Stage 1: Fast optical pre-scan (dual_optical mode only)
    # ------------------------------------------------------------------

    def _stage1_optical_scan(self, result: FieldResult) -> FieldResult:
        """Fast optical pre-filter using dual-survey images.

        Downloads two optical surveys and runs P9Searcher.  If no
        detections are found, the field is skipped (stage1_passed=False).
        """
        t0 = time.time()

        survey1, survey2 = self.config.optical_surveys[:2]
        paths = download_field(
            result.ra_deg,
            result.dec_deg,
            surveys=[survey1, survey2],
            radius_arcmin=self.config.download_radius_arcmin,
            pixels=self.config.download_pixels,
            output_dir=self.config.data_dir,
            rate_limit_sec=self.config.rate_limit_sec,
        )

        if len(paths) < 2:
            # Insufficient optical coverage — skip pre-filter, proceed
            result.stage1_passed = True
            result.download_time_sec += time.time() - t0
            return result

        image_1, _h1 = load_fits_image(paths[0])
        image_2, _h2 = load_fits_image(paths[1])
        result.optical_path = str(paths[0])

        search_result = self.optical_searcher.search_from_arrays(
            image_1,
            image_2,
            image_path_1=str(paths[0]),
            image_path_2=str(paths[1]),
        )

        result.stage1_detections = len(search_result.patch_detections)
        result.stage1_passed = result.stage1_detections > 0
        result.download_time_sec += time.time() - t0

        return result

    # ------------------------------------------------------------------
    # Stage 2: Download images
    # ------------------------------------------------------------------

    def _stage2_download(self, result: FieldResult) -> FieldResult:
        """Download optical + IR images from SkyView.

        Skips surveys already downloaded in Stage 1.
        """
        t0 = time.time()

        # Determine which surveys to download
        surveys: list[str] = []
        if result.optical_path is None:
            surveys.append("DSS2 Red")
        surveys.extend(self.config.ir_surveys)

        paths = download_field(
            result.ra_deg,
            result.dec_deg,
            surveys=surveys,
            radius_arcmin=self.config.download_radius_arcmin,
            pixels=self.config.download_pixels,
            output_dir=self.config.data_dir,
            rate_limit_sec=self.config.rate_limit_sec,
        )

        # Map paths to bands by directory/filename convention
        for path in paths:
            path_lower = str(path).lower()
            if "dss2_red" in path_lower:
                result.optical_path = str(path)
            elif "wise_3.4" in path_lower or "wise_3_4" in path_lower:
                result.w1_path = str(path)
            elif "wise_4.6" in path_lower or "wise_4_6" in path_lower:
                result.w2_path = str(path)

        result.ir_downloaded = (
            result.w1_path is not None or result.w2_path is not None
        )
        result.download_time_sec += time.time() - t0

        return result

    # ------------------------------------------------------------------
    # Stage 2b: IR-only download (no optical)
    # ------------------------------------------------------------------

    def _stage2_download_ir_only(self, result: FieldResult) -> FieldResult:
        """Download WISE W1/W2 images only — no optical survey.

        Used in ``ir_only`` mode where DSS2 is not required (targets
        invisible in optical, e.g. Planet 9 at ~25 mag).
        """
        t0 = time.time()

        surveys = list(self.config.ir_surveys)  # ["WISE 3.4", "WISE 4.6"]

        paths = download_field(
            result.ra_deg,
            result.dec_deg,
            surveys=surveys,
            radius_arcmin=self.config.download_radius_arcmin,
            pixels=self.config.download_pixels,
            output_dir=self.config.data_dir,
            rate_limit_sec=self.config.rate_limit_sec,
        )

        # Map paths to W1/W2 (no optical mapping)
        for path in paths:
            path_lower = str(path).lower()
            if "wise_3.4" in path_lower or "wise_3_4" in path_lower:
                result.w1_path = str(path)
            elif "wise_4.6" in path_lower or "wise_4_6" in path_lower:
                result.w2_path = str(path)

        result.ir_downloaded = (
            result.w1_path is not None or result.w2_path is not None
        )
        result.download_time_sec += time.time() - t0

        return result

    # ------------------------------------------------------------------
    # Stage 3: Spectral validation
    # ------------------------------------------------------------------

    def _stage3_spectral(self, result: FieldResult) -> FieldResult:
        """Build spectral cubes and run SSTF inference with PlanckFilter."""
        t0 = time.time()

        # Load images
        optical, optical_hdr = load_fits_image(Path(result.optical_path))

        w1, w1_hdr = None, None
        w2, w2_hdr = None, None

        if result.w1_path:
            try:
                w1, w1_hdr = load_fits_image(Path(result.w1_path))
                result.has_w1 = True
            except Exception:
                pass  # Missing W1 → zeros in cube

        if result.w2_path:
            try:
                w2, w2_hdr = load_fits_image(Path(result.w2_path))
                result.has_w2 = True
            except Exception:
                pass  # Missing W2 → zeros in cube

        # Build spectral cube (single-epoch: cube2 = cube1.copy())
        target_shape = optical.shape
        cube1 = build_spectral_cube(
            optical,
            dict(optical_hdr) if hasattr(optical_hdr, "items") else optical_hdr,
            w1,
            dict(w1_hdr) if w1_hdr is not None and hasattr(w1_hdr, "items") else w1_hdr,
            w2,
            dict(w2_hdr) if w2_hdr is not None and hasattr(w2_hdr, "items") else w2_hdr,
            target_shape=target_shape,
        )
        cube2 = cube1.copy()
        # Temporal gradient = 0 for single-epoch (already zeros in cube)

        # Run SpectralSearcher
        spectral_result = self.spectral_searcher.search_from_cubes(
            cube1,
            cube2,
            label_1=result.optical_path or result.field_id,
            label_2=f"{result.field_id}_epoch2",
        )
        result.spectral_result = spectral_result

        # Apply NMS + WCS conversion
        candidates = detections_to_candidates(
            spectral_result.patch_detections,
            dict(optical_hdr) if hasattr(optical_hdr, "items") else optical_hdr,
            spectral_result.grid_shape,
        )
        spectral_result.candidates = candidates

        # Enrich detections with temperature + Planck classification
        enriched: list[dict] = []
        for i, det in enumerate(spectral_result.patch_detections):
            r, c = det.patch_row, det.patch_col

            temp_k = float(spectral_result.temperature_map[r, c])
            temp_std = float(spectral_result.temperature_std_map[r, c])
            planck_class = (
                spectral_result.planck_classifications[i]
                if i < len(spectral_result.planck_classifications)
                else "uncertain"
            )

            # Coordinate conversion
            ra, dec = float("nan"), float("nan")
            try:
                from astroworld.imaging.synthetic_injection import (
                    pixel_to_radec,
                )

                hdr = (
                    dict(optical_hdr)
                    if hasattr(optical_hdr, "items")
                    else optical_hdr
                )
                ra, dec = pixel_to_radec(det.pixel_x, det.pixel_y, hdr)
            except Exception:
                pass

            enriched.append(
                {
                    "pixel_x": det.pixel_x,
                    "pixel_y": det.pixel_y,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "probability": det.probability,
                    "temperature_k": temp_k,
                    "temperature_std_k": temp_std,
                    "planck_class": planck_class,
                }
            )

        # ---- SNR photometric filter (noise rejection) --------------------
        # Reject candidates whose optical patch is pure background noise.
        # Uses robust MAD-based sigma to avoid cosmic ray contamination.
        snr_threshold = self.config.snr_min
        n_before_snr = len(enriched)
        if snr_threshold > 0:
            snr_filtered = []
            for c in enriched:
                snr = self._patch_snr(
                    optical,
                    c["pixel_x"],
                    c["pixel_y"],
                    aperture_r=self.config.snr_aperture_radius,
                    annulus_inner=self.config.snr_annulus_inner,
                    annulus_outer=self.config.snr_annulus_outer,
                )
                c["optical_snr"] = round(snr, 2)
                if snr >= snr_threshold:
                    snr_filtered.append(c)

            n_rejected_snr = n_before_snr - len(snr_filtered)
            if n_rejected_snr > 0 and self.config.verbose:
                print(
                    f"    [SNR filter] {n_rejected_snr}/{n_before_snr} "
                    f"rejected (optical SNR < {snr_threshold:.1f})"
                )
            enriched = snr_filtered

        # ---- Per-patch IR quality filter --------------------------------
        # Even when a WISE image exists for the whole field, the patch's
        # 64×64 region may fall on a tile boundary filled with zeros.
        # Check that at least ``ir_min_valid_fraction`` of the patch's
        # W1/W2 pixels are non-zero before trusting the temperature head.
        ps = self.config.patch_size
        _, cube_h, cube_w = cube1.shape
        min_frac = self.config.ir_min_valid_fraction
        n_rejected_patch = 0

        for c in enriched:
            # Determine the patch bounding box (same layout as searcher)
            cx, cy = c["pixel_x"], c["pixel_y"]
            y0 = max(int(cy - ps / 2), 0)
            x0 = max(int(cx - ps / 2), 0)
            y1 = min(y0 + ps, cube_h)
            x1 = min(x0 + ps, cube_w)

            w1_patch = cube1[BAND_W1, y0:y1, x0:x1]
            w2_patch = cube1[BAND_W2, y0:y1, x0:x1]
            n_pixels = max(w1_patch.size, 1)

            w1_valid = float(np.count_nonzero(w1_patch)) / n_pixels
            w2_valid = float(np.count_nonzero(w2_patch)) / n_pixels

            # Both bands must meet the coverage threshold
            if w1_valid < min_frac and w2_valid < min_frac:
                c["ir_quality"] = "insufficient_ir_patch"
                n_rejected_patch += 1
            elif w1_valid < min_frac or w2_valid < min_frac:
                c["ir_quality"] = "partial_patch"
            else:
                c["ir_quality"] = "full"

        if n_rejected_patch > 0 and self.config.verbose:
            print(
                f"    [IR patch filter] {n_rejected_patch}/{len(enriched)} "
                f"patches rejected (<{min_frac:.0%} valid IR pixels)"
            )

        # ---- Planck filter: keep only desired classes -------------------
        filtered = [
            c
            for c in enriched
            if (
                c["planck_class"] in self.config.planck_classes_keep
                and c.get("ir_quality") != "insufficient_ir_patch"
            )
        ]

        # ---- Field-level IR quality filter (legacy) --------------------
        # Reject ALL candidates when BOTH W1 and W2 are completely
        # missing (HTTP 404 / failed download).
        if not (result.has_w1 or result.has_w2):
            for c in filtered:
                c["planck_class"] = "insufficient_ir"
            filtered = []
            if self.config.verbose:
                print(
                    f"    [IR filter] No WISE data — "
                    f"rejected {len(enriched)} detection(s)"
                )
        elif not (result.has_w1 and result.has_w2):
            # Flag partial field-level coverage (1/2 bands available)
            for c in filtered:
                if c.get("ir_quality") == "full":
                    c["ir_quality"] = "partial"
            if self.config.verbose:
                n_bands = int(result.has_w1) + int(result.has_w2)
                print(
                    f"    [IR quality] {n_bands}/2 WISE bands — "
                    f"partial IR coverage"
                )

        result.stage3_detections = len(spectral_result.patch_detections)
        result.stage3_candidates = filtered
        result.inference_time_sec += time.time() - t0

        return result

    # ------------------------------------------------------------------
    # Stage 3 IR: Spectral validation (ir_only mode)
    # ------------------------------------------------------------------

    def _stage3_ir_spectral(self, result: FieldResult) -> FieldResult:
        """Build IR-only spectral cube and run SSTF inference.

        Uses WISE W1 as the spatial reference frame instead of optical.
        SNR filter uses W2 (or W1 fallback) instead of optical.
        """
        from astroworld.imaging.spectral_cube import build_ir_cube

        t0 = time.time()

        # Load W1 (mandatory in ir_only mode)
        w1, w1_hdr = load_fits_image(Path(result.w1_path))
        result.has_w1 = True

        w2, w2_hdr = None, None
        if result.w2_path:
            try:
                w2, w2_hdr = load_fits_image(Path(result.w2_path))
                result.has_w2 = True
            except Exception:
                pass  # Missing W2 → zeros in cube

        # Build IR cube (W1 as reference)
        target_shape = w1.shape
        w1_hdr_dict = (
            dict(w1_hdr) if hasattr(w1_hdr, "items") else w1_hdr
        )
        w2_hdr_dict = (
            dict(w2_hdr)
            if w2_hdr is not None and hasattr(w2_hdr, "items")
            else w2_hdr
        )

        cube1 = build_ir_cube(
            w1, w1_hdr_dict,
            w2, w2_hdr_dict,
            target_shape=target_shape,
        )
        cube2 = cube1.copy()
        # Temporal gradient = 0 for single-epoch IR (already zeros)

        # Run SpectralSearcher
        spectral_result = self.spectral_searcher.search_from_cubes(
            cube1,
            cube2,
            label_1=result.w1_path or result.field_id,
            label_2=f"{result.field_id}_ir_epoch2",
        )
        result.spectral_result = spectral_result

        # Apply NMS + WCS conversion (using W1 header, not optical)
        candidates = detections_to_candidates(
            spectral_result.patch_detections,
            w1_hdr_dict,
            spectral_result.grid_shape,
        )
        spectral_result.candidates = candidates

        # Enrich detections with temperature + Planck classification
        enriched: list[dict] = []
        for i, det in enumerate(spectral_result.patch_detections):
            r, c = det.patch_row, det.patch_col

            temp_k = float(spectral_result.temperature_map[r, c])
            temp_std = float(spectral_result.temperature_std_map[r, c])
            planck_class = (
                spectral_result.planck_classifications[i]
                if i < len(spectral_result.planck_classifications)
                else "uncertain"
            )

            # Coordinate conversion using W1 WCS
            ra, dec = float("nan"), float("nan")
            try:
                from astroworld.imaging.synthetic_injection import (
                    pixel_to_radec,
                )
                ra, dec = pixel_to_radec(
                    det.pixel_x, det.pixel_y, w1_hdr_dict,
                )
            except Exception:
                pass

            enriched.append(
                {
                    "pixel_x": det.pixel_x,
                    "pixel_y": det.pixel_y,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "probability": det.probability,
                    "temperature_k": temp_k,
                    "temperature_std_k": temp_std,
                    "planck_class": planck_class,
                }
            )

        # ---- SNR photometric filter (IR band) ----------------------------
        # In ir_only mode, measure SNR on W2 (or W1 fallback) instead of
        # optical.  P9 at ~40 K is brightest in W2 (4.6 µm).
        snr_image = w2 if w2 is not None else w1
        snr_band_label = "W2" if w2 is not None else "W1"
        snr_threshold = self.config.snr_min
        n_before_snr = len(enriched)
        if snr_threshold > 0:
            snr_filtered = []
            for cand in enriched:
                snr = self._patch_snr(
                    snr_image,
                    cand["pixel_x"],
                    cand["pixel_y"],
                    aperture_r=self.config.snr_aperture_radius,
                    annulus_inner=self.config.snr_annulus_inner,
                    annulus_outer=self.config.snr_annulus_outer,
                )
                cand["ir_snr"] = round(snr, 2)
                if snr >= snr_threshold:
                    snr_filtered.append(cand)

            n_rejected_snr = n_before_snr - len(snr_filtered)
            if n_rejected_snr > 0 and self.config.verbose:
                print(
                    f"    [SNR filter] {n_rejected_snr}/{n_before_snr} "
                    f"rejected ({snr_band_label} SNR < {snr_threshold:.1f})"
                )
            enriched = snr_filtered

        # ---- Per-patch IR quality filter ----------------------------------
        ps = self.config.patch_size
        _, cube_h, cube_w = cube1.shape
        min_frac = self.config.ir_min_valid_fraction
        n_rejected_patch = 0

        for cand in enriched:
            cx, cy = cand["pixel_x"], cand["pixel_y"]
            y0 = max(int(cy - ps / 2), 0)
            x0 = max(int(cx - ps / 2), 0)
            y1 = min(y0 + ps, cube_h)
            x1 = min(x0 + ps, cube_w)

            w1_patch = cube1[BAND_W1, y0:y1, x0:x1]
            w2_patch = cube1[BAND_W2, y0:y1, x0:x1]
            n_pixels = max(w1_patch.size, 1)

            w1_valid = float(np.count_nonzero(w1_patch)) / n_pixels
            w2_valid = float(np.count_nonzero(w2_patch)) / n_pixels

            if w1_valid < min_frac and w2_valid < min_frac:
                cand["ir_quality"] = "insufficient_ir_patch"
                n_rejected_patch += 1
            elif w1_valid < min_frac or w2_valid < min_frac:
                cand["ir_quality"] = "partial_patch"
            else:
                cand["ir_quality"] = "full"

        if n_rejected_patch > 0 and self.config.verbose:
            print(
                f"    [IR patch filter] {n_rejected_patch}/{len(enriched)} "
                f"patches rejected (<{min_frac:.0%} valid IR pixels)"
            )

        # ---- Planck filter -----------------------------------------------
        filtered = [
            cand
            for cand in enriched
            if (
                cand["planck_class"] in self.config.planck_classes_keep
                and cand.get("ir_quality") != "insufficient_ir_patch"
            )
        ]

        # Field-level IR quality (same logic as _stage3_spectral)
        if not (result.has_w1 or result.has_w2):
            for cand in filtered:
                cand["planck_class"] = "insufficient_ir"
            filtered = []
            if self.config.verbose:
                print(
                    f"    [IR filter] No WISE data — "
                    f"rejected {len(enriched)} detection(s)"
                )

        result.stage3_detections = len(spectral_result.patch_detections)
        result.stage3_candidates = filtered
        result.inference_time_sec += time.time() - t0

        return result

    # ------------------------------------------------------------------
    # Stage 3b: Dust Piercer (ir_only mode)
    # ------------------------------------------------------------------

    def _stage3b_dust_piercer(self, result: FieldResult) -> FieldResult:
        """Run Dust Piercer as integrated validation for ir_only mode.

        For each surviving candidate, runs WISE W2 morphology analysis
        and NEOWISE multi-epoch proper motion search.  Retains only
        POINT_MOVING and POINT_STATIC verdicts (rejects DIFFUSE, NO_SOURCE).

        Relaxes the pointiness threshold for small (fallback) images
        where the PSF may appear artificially broad.
        """
        from astroworld.imaging.dust_piercer import (
            analyze_candidate,
            classify_dust_piercer_verdict,
        )

        if not result.w2_path and not result.w1_path:
            return result

        t0 = time.time()

        # Load W2 for morphology (preferred) or W1 as fallback
        w2_path = result.w2_path or result.w1_path
        w2_img, w2_hdr = load_fits_image(Path(w2_path))

        # Detect small (fallback) images where PSF analysis is degraded
        is_small_image = min(w2_img.shape) < 300

        enriched: list[dict] = []
        verdicts: dict[str, int] = {}

        for cand in result.stage3_candidates:
            try:
                dp_result = analyze_candidate(
                    cand["ra_deg"],
                    cand["dec_deg"],
                    w2_img,
                    dict(w2_hdr) if hasattr(w2_hdr, "items") else w2_hdr,
                    rate_limit_sec=self.config.rate_limit_sec,
                    skip_neowise=False,
                )

                verdict = dp_result.verdict
                cand["dp_verdict"] = verdict
                cand["dp_confidence"] = dp_result.confidence

                if dp_result.morphology:
                    cand["dp_pointiness"] = dp_result.morphology.pointiness
                    cand["dp_snr"] = dp_result.morphology.snr

                if dp_result.proper_motion:
                    cand["dp_pm_arcsec_yr"] = (
                        dp_result.proper_motion.mu_total_arcsec_yr
                    )
                    cand["dp_pm_significance"] = (
                        dp_result.proper_motion.pm_significance
                    )

                # Relaxed pointiness for small (fallback) images:
                # Override DIFFUSE → POINT_STATIC if pointiness exceeds
                # the relaxed threshold and image is small
                if (
                    is_small_image
                    and verdict == "DIFFUSE"
                    and dp_result.morphology
                    and dp_result.morphology.pointiness
                    >= self.config.dp_pointiness_threshold
                ):
                    verdict = "POINT_STATIC"
                    cand["dp_verdict"] = verdict
                    cand["dp_verdict_note"] = "relaxed_small_image"

                verdicts[verdict] = verdicts.get(verdict, 0) + 1

                # Keep only point sources (reject DIFFUSE, NO_SOURCE)
                if verdict in ("POINT_MOVING", "POINT_STATIC"):
                    enriched.append(cand)

            except Exception as e:
                cand["dp_verdict"] = "ERROR"
                cand["dp_error"] = str(e)
                verdicts["ERROR"] = verdicts.get("ERROR", 0) + 1
                enriched.append(cand)  # Keep on error — manual review

        n_rejected = len(result.stage3_candidates) - len(enriched)
        if self.config.verbose and n_rejected > 0:
            print(
                f"    [Dust Piercer] {n_rejected}/"
                f"{len(result.stage3_candidates)} rejected: "
                f"{verdicts}"
            )

        result.stage3b_dp_analyzed = len(result.stage3_candidates)
        result.stage3b_dp_verdicts = verdicts
        result.stage3_candidates = enriched
        result.inference_time_sec += time.time() - t0

        return result

    # ------------------------------------------------------------------
    # Stage 3 DP-Direct: Classic peak finding + Dust Piercer
    # ------------------------------------------------------------------

    def _stage3_dp_direct(self, result: FieldResult) -> FieldResult:
        """Bypass SSTF entirely — classic 5-sigma peak finding on W2.

        Uses ``sigma_clipped_stats`` for robust background estimation and
        ``maximum_filter`` for local peak detection.  Each peak is passed
        directly to Dust Piercer for morphology + NEOWISE PM validation.

        No neural network, no spectral cube.  Pure algorithmic detection.
        """
        from astropy.stats import sigma_clipped_stats
        from astropy.wcs import WCS
        from scipy.ndimage import maximum_filter

        from astroworld.imaging.dust_piercer import analyze_candidate

        t0 = time.time()

        # Prefer W2 (P9 brightest at 4.6 um), fallback to W1
        ir_path = result.w2_path or result.w1_path
        ir_img, ir_hdr = load_fits_image(Path(ir_path))
        ir_hdr_dict = (
            dict(ir_hdr) if hasattr(ir_hdr, "items") else ir_hdr
        )
        ir_band = "W2" if result.w2_path else "W1"

        # 1. Background estimation (sigma-clipped)
        mean, median, std = sigma_clipped_stats(ir_img, sigma=3.0)
        threshold = median + (self.config.dp_peak_sigma * std)

        # 2. Local maxima above threshold
        nbr = self.config.dp_peak_neighborhood
        local_max = maximum_filter(ir_img, size=2 * nbr + 1) == ir_img
        above_threshold = ir_img > threshold
        peaks = local_max & above_threshold

        y_coords, x_coords = np.where(peaks)

        if self.config.verbose:
            print(
                f"    [DP-Direct] {len(x_coords)} raw "
                f"{self.config.dp_peak_sigma:.0f}-sigma peaks in {ir_band} "
                f"(bkg={median:.1f}, thr={threshold:.1f})"
            )

        # Cap to brightest N to prevent NEOWISE API overload
        if len(x_coords) > self.config.dp_max_peaks:
            intensities = ir_img[y_coords, x_coords]
            top_idx = np.argsort(intensities)[-self.config.dp_max_peaks :]
            x_coords = x_coords[top_idx]
            y_coords = y_coords[top_idx]
            if self.config.verbose:
                print(
                    f"    [DP-Direct] Capped to top "
                    f"{self.config.dp_max_peaks} brightest peaks"
                )

        # 3. Pixel → RA/Dec via WCS
        try:
            wcs = WCS(ir_hdr_dict)
            # pixel_to_world_values expects (x, y) order
            ra_arr, dec_arr = wcs.pixel_to_world_values(
                x_coords.astype(float), y_coords.astype(float)
            )
        except Exception:
            # Fallback: use field center for all peaks
            ra_arr = np.full(len(x_coords), result.ra_deg)
            dec_arr = np.full(len(x_coords), result.dec_deg)

        # 4. Load W2 for Dust Piercer (need image + header)
        w2_img, w2_hdr = None, None
        if result.w2_path:
            w2_img, w2_hdr = load_fits_image(Path(result.w2_path))
            w2_hdr = (
                dict(w2_hdr) if hasattr(w2_hdr, "items") else w2_hdr
            )
        else:
            # Use whatever IR we have
            w2_img = ir_img
            w2_hdr = ir_hdr_dict

        is_small_image = min(w2_img.shape) < 300

        # 5. Run Dust Piercer on each peak
        enriched: list[dict] = []
        verdicts: dict[str, int] = {}

        for i in range(len(x_coords)):
            ra_i = float(ra_arr[i])
            dec_i = float(dec_arr[i])
            px_i = float(x_coords[i])
            py_i = float(y_coords[i])
            peak_flux = float(ir_img[y_coords[i], x_coords[i]])

            try:
                dp_result = analyze_candidate(
                    ra_i, dec_i,
                    w2_img, w2_hdr,
                    rate_limit_sec=self.config.rate_limit_sec,
                    skip_neowise=False,
                )

                verdict = dp_result.verdict

                # Relaxed pointiness for small fallback images
                if (
                    is_small_image
                    and verdict == "DIFFUSE"
                    and dp_result.morphology
                    and dp_result.morphology.pointiness
                    >= self.config.dp_pointiness_threshold
                ):
                    verdict = "POINT_STATIC"

                verdicts[verdict] = verdicts.get(verdict, 0) + 1

                if verdict in ("POINT_MOVING", "POINT_STATIC"):
                    cand = {
                        "pixel_x": px_i,
                        "pixel_y": py_i,
                        "ra_deg": ra_i,
                        "dec_deg": dec_i,
                        "probability": 1.0,  # Classic detection, no ML score
                        "temperature_k": 0.0,  # Not estimated in dp_direct
                        "temperature_std_k": 0.0,
                        "planck_class": "dp_direct",
                        "peak_flux": peak_flux,
                        "peak_sigma": (peak_flux - median) / max(std, 1e-10),
                        "dp_verdict": verdict,
                        "dp_confidence": dp_result.confidence,
                    }
                    if dp_result.morphology:
                        cand["dp_pointiness"] = (
                            dp_result.morphology.pointiness
                        )
                        cand["dp_snr"] = dp_result.morphology.snr
                    if dp_result.proper_motion:
                        cand["dp_pm_arcsec_yr"] = (
                            dp_result.proper_motion.mu_total_arcsec_yr
                        )
                        cand["dp_pm_significance"] = (
                            dp_result.proper_motion.pm_significance
                        )
                    enriched.append(cand)

                    if verdict == "POINT_MOVING" and self.config.verbose:
                        pm = dp_result.proper_motion
                        pm_str = (
                            f"{pm.mu_total_arcsec_yr:.2f}\"/yr "
                            f"({pm.pm_significance:.1f}σ)"
                            if pm else "?"
                        )
                        print(
                            f"    *** POINT_MOVING at "
                            f"RA={ra_i:.4f} Dec={dec_i:.4f} "
                            f"PM={pm_str} ***"
                        )

            except Exception as e:
                verdicts["ERROR"] = verdicts.get("ERROR", 0) + 1
                if self.config.verbose:
                    print(f"    [DP error] peak {i}: {e}")

        if self.config.verbose:
            n_total = len(x_coords)
            n_kept = len(enriched)
            print(
                f"    [DP-Direct] {n_kept}/{n_total} peaks survived: "
                f"{verdicts}"
            )

        result.stage3_detections = len(x_coords)
        result.stage3_candidates = enriched
        result.stage3b_dp_analyzed = len(x_coords)
        result.stage3b_dp_verdicts = verdicts
        result.inference_time_sec += time.time() - t0

        return result

    # ------------------------------------------------------------------
    # Stage 4: Kinematic filter
    # ------------------------------------------------------------------

    def _stage4_kinematic(self, result: FieldResult) -> FieldResult:
        """Apply kinematic validation to surviving candidates.

        In single-epoch mode (no displacement measurements), operates in
        conservative mode: all candidates pass.
        """
        if not self.config.kinematic_enabled:
            result.stage4_survivors = len(result.stage3_candidates)
            result.final_candidates = [
                {**c, "kinematic_valid": True}
                for c in result.stage3_candidates
            ]
            return result

        # Convert enriched candidates to PatchDetection for the filter
        patch_dets = [
            PatchDetection(
                patch_row=0,
                patch_col=0,
                pixel_x=c["pixel_x"],
                pixel_y=c["pixel_y"],
                probability=c["probability"],
            )
            for c in result.stage3_candidates
        ]

        # Conservative mode: no displacement estimates available for
        # single-epoch data → filter returns all detections
        filtered = filter_detections_kinematic(
            patch_dets,
            min_displacement_px=self.config.kinematic_min_px,
            max_displacement_px=self.config.kinematic_max_px,
            displacement_estimates=None,
        )

        # Map back to enriched candidates
        surviving_pixels = {
            (d.pixel_x, d.pixel_y) for d in filtered
        }
        result.final_candidates = [
            {**c, "kinematic_valid": True}
            for c in result.stage3_candidates
            if (c["pixel_x"], c["pixel_y"]) in surviving_pixels
        ]
        result.stage4_survivors = len(result.final_candidates)

        return result

    # ------------------------------------------------------------------
    # Stage 5: Report generation
    # ------------------------------------------------------------------

    def _stage5_report(self, summary: ObservatorySummary) -> None:
        """Generate CSV report, heatmap PNGs, and summary text."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save candidate CSV
        csv_path = output_dir / "observatory_candidates.csv"
        summary.candidate_table.to_csv(csv_path, index=False)

        # Save heatmaps for fields with detections
        heatmap_dir = output_dir / "heatmaps"

        for fr in summary.field_results:
            # Use optical as background, or W1 in ir_only mode
            bg_path = fr.optical_path or fr.w1_path
            if (
                fr.spectral_result is not None
                and len(fr.final_candidates) > 0
                and bg_path is not None
            ):
                heatmap_dir.mkdir(parents=True, exist_ok=True)
                try:
                    bg_img, _ = load_fits_image(Path(bg_path))
                    heatmap_path = (
                        heatmap_dir / f"{fr.field_id}_heatmap.png"
                    )
                    plot_heatmap(
                        fr.spectral_result,
                        bg_img,
                        heatmap_path,
                        title=(
                            f"{fr.field_id} | "
                            f"{len(fr.final_candidates)} candidate(s)"
                        ),
                    )
                except Exception:
                    pass  # Non-critical visualization failure

        # Save summary text
        summary_path = output_dir / "observatory_summary.txt"
        self._write_summary_text(summary, summary_path)

        if self.config.verbose:
            self._print_summary(summary)

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        field_results: list[FieldResult],
        total_time: float,
    ) -> ObservatorySummary:
        """Build aggregate summary from per-field results."""
        records: list[dict] = []
        for fr in field_results:
            for cand in fr.final_candidates:
                records.append(
                    {
                        "field_id": fr.field_id,
                        "field_ra": fr.ra_deg,
                        "field_dec": fr.dec_deg,
                        "pixel_x": cand["pixel_x"],
                        "pixel_y": cand["pixel_y"],
                        "ra_deg": cand["ra_deg"],
                        "dec_deg": cand["dec_deg"],
                        "probability": cand["probability"],
                        "temperature_k": cand["temperature_k"],
                        "temperature_std_k": cand["temperature_std_k"],
                        "planck_class": cand["planck_class"],
                        "kinematic_valid": cand.get("kinematic_valid", True),
                    }
                )

        # Ensure consistent schema even with zero candidates
        if records:
            candidate_table = pd.DataFrame(records)
        else:
            candidate_table = pd.DataFrame(
                columns=[
                    "field_id", "field_ra", "field_dec",
                    "pixel_x", "pixel_y", "ra_deg", "dec_deg",
                    "probability", "temperature_k", "temperature_std_k",
                    "planck_class", "kinematic_valid",
                ]
            )

        n_processed = sum(1 for fr in field_results if fr.error is None)
        n_errored = sum(1 for fr in field_results if fr.error is not None)
        n_with_candidates = sum(
            1 for fr in field_results if len(fr.final_candidates) > 0
        )
        n_fields = len(field_results)

        return ObservatorySummary(
            n_fields_total=n_fields,
            n_fields_processed=n_processed,
            n_fields_with_candidates=n_with_candidates,
            n_fields_errored=n_errored,
            n_stage1_passed=sum(
                1 for fr in field_results if fr.stage1_passed
            ),
            n_stage3_detections=sum(
                fr.stage3_detections for fr in field_results
            ),
            n_stage3_planck_filtered=sum(
                len(fr.stage3_candidates) for fr in field_results
            ),
            n_stage4_survivors=sum(
                fr.stage4_survivors for fr in field_results
            ),
            total_candidates=len(candidate_table),
            candidate_table=candidate_table,
            total_time_sec=total_time,
            avg_time_per_field_sec=(
                total_time / max(n_fields, 1)
            ),
            download_time_sec=sum(
                fr.download_time_sec for fr in field_results
            ),
            inference_time_sec=sum(
                fr.inference_time_sec for fr in field_results
            ),
            config=self.config,
            field_results=field_results,
        )

    # ------------------------------------------------------------------
    # Print / output helpers
    # ------------------------------------------------------------------

    def _print_header(self, n_fields: int) -> None:
        """Print pipeline header banner."""
        print("\n" + "=" * 60)
        print("  OBSERVATORY PIPELINE — Planet 9 Cascading Detector")
        print("=" * 60)
        print(f"  Mode:       {self.config.mode}")
        print(f"  Fields:     {n_fields}")
        print(f"  Threshold:  {self.config.spectral_threshold:.2f} (spectral)")
        print(f"  MC samples: {self.config.mc_samples}")
        print(f"  Planck:     {', '.join(self.config.planck_classes_keep)}")
        print(f"  Kinematic:  {'ON' if self.config.kinematic_enabled else 'OFF'}")
        print(f"  Device:     {self.config.device}")
        print("-" * 60)

    def _print_summary(self, summary: ObservatorySummary) -> None:
        """Print final summary to console."""
        print("\n" + "=" * 60)
        print("  OBSERVATORY SUMMARY")
        print("=" * 60)
        print(f"  Fields processed:   {summary.n_fields_processed}/{summary.n_fields_total}")
        if summary.n_fields_errored > 0:
            print(f"  Fields errored:     {summary.n_fields_errored}")
        print(f"  Stage 1 passed:     {summary.n_stage1_passed}")
        print(f"  Stage 3 detections: {summary.n_stage3_detections}")
        print(f"  Stage 3 (Planck):   {summary.n_stage3_planck_filtered}")
        print(f"  Stage 4 survivors:  {summary.n_stage4_survivors}")
        print(f"  TOTAL CANDIDATES:   {summary.total_candidates}")
        print("-" * 60)
        print(f"  Total time:     {summary.total_time_sec:.1f}s")
        print(f"  Download time:  {summary.download_time_sec:.1f}s")
        print(f"  Inference time: {summary.inference_time_sec:.1f}s")
        print(f"  Avg per field:  {summary.avg_time_per_field_sec:.1f}s")
        print(f"  Output:         {self.config.output_dir}")
        print("=" * 60 + "\n")

    def _write_summary_text(
        self,
        summary: ObservatorySummary,
        path: Path,
    ) -> None:
        """Write a human-readable summary text file."""
        lines = [
            "Observatory Pipeline — Summary Report",
            "=" * 40,
            f"Mode:                {summary.config.mode}",
            f"Fields total:        {summary.n_fields_total}",
            f"Fields processed:    {summary.n_fields_processed}",
            f"Fields errored:      {summary.n_fields_errored}",
            "",
            "Detection Funnel",
            "-" * 40,
            f"Stage 1 passed:      {summary.n_stage1_passed}",
            f"Stage 3 detections:  {summary.n_stage3_detections}",
            f"Stage 3 Planck:      {summary.n_stage3_planck_filtered}",
            f"Stage 4 kinematic:   {summary.n_stage4_survivors}",
            f"TOTAL CANDIDATES:    {summary.total_candidates}",
            "",
            "Timing",
            "-" * 40,
            f"Total:       {summary.total_time_sec:.1f}s",
            f"Download:    {summary.download_time_sec:.1f}s",
            f"Inference:   {summary.inference_time_sec:.1f}s",
            f"Avg/field:   {summary.avg_time_per_field_sec:.1f}s",
            "",
            "Configuration",
            "-" * 40,
            f"Spectral threshold:  {summary.config.spectral_threshold}",
            f"MC samples:          {summary.config.mc_samples}",
            f"Planck keep:         {summary.config.planck_classes_keep}",
            f"Kinematic:           {summary.config.kinematic_enabled}",
            f"Patch size:          {summary.config.patch_size}",
            f"Device:              {summary.config.device}",
        ]

        # Per-field details
        lines.append("")
        lines.append("Per-Field Results")
        lines.append("-" * 40)
        for fr in summary.field_results:
            status = "OK" if fr.error is None else f"ERROR: {fr.error}"
            lines.append(
                f"  {fr.field_id}: "
                f"{len(fr.final_candidates)} candidates, "
                f"{fr.total_time_sec:.1f}s [{status}]"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
