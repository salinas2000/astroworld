#!/usr/bin/env python3
"""
Phase 20: P9 Survey Orchestrator — Automated Pipeline.

Chains the full Planet 9 detection pipeline with smart resume:
  Stage 1 (SCAN)        — Blind search with SSTF spectral inference
  Stage 2 (CROSSMATCH)  — Reject known Gaia/SIMBAD objects
  Stage 3 (BLINK)       — Multi-epoch optical blink (DSS2 vs Pan-STARRS)
  Stage 4 (DUST_PIERCER)— IR morphology + NEOWISE proper motion

Usage
-----
    # Full pipeline with auto-resume:
    uv run python scripts/p9_survey.py \
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \
        --region taurus

    # Dry run (show progress without executing):
    uv run python scripts/p9_survey.py \
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \
        --region taurus --dry-run

    # Downstream only (crossmatch + blink + dust_piercer on existing scan):
    uv run python scripts/p9_survey.py \
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \
        --region taurus --downstream-only

    # Scan only (no downstream analysis):
    uv run python scripts/p9_survey.py \
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \
        --region taurus --scan-only

    # Custom region (northern Taurus):
    uv run python scripts/p9_survey.py \
        --spectral-checkpoint checkpoints/spectral_adapted/best_model.pt \
        --ra-min 55 --ra-max 85 --dec-min 16 --dec-max 30 --step 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging (dual: console + file)
# ---------------------------------------------------------------------------

logger = logging.getLogger("p9_survey")


def _setup_logging(output_dir: Path) -> None:
    """Configure dual logging to console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "p9_survey.log"

    logger.setLevel(logging.DEBUG)

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    # File handler (DEBUG+)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Preset regions (shared with run_blind_search.py)
# ---------------------------------------------------------------------------

PRESETS = {
    "taurus": {
        "description": "Taurus Molecular Cloud + P9 aphelion zone",
        "ra_min": 55.0, "ra_max": 85.0,
        "dec_min": 10.0, "dec_max": 30.0,
        "step_deg": 0.50,
    },
    "ecliptic": {
        "description": "Full Batygin/Brown (2021) ecliptic corridor",
        "ra_min": 20.0, "ra_max": 120.0,
        "dec_min": -30.0, "dec_max": 30.0,
        "step_deg": 0.75,
    },
    "galactic_gap": {
        "description": "Galactic plane gap (survey exclusion zone)",
        "ra_min": 80.0, "ra_max": 120.0,
        "dec_min": -10.0, "dec_max": 10.0,
        "step_deg": 0.50,
    },
    "sedna": {
        "description": "Sedna orbital neighborhood (validation grid)",
        "ra_min": 69.0, "ra_max": 73.5,
        "dec_min": 13.0, "dec_max": 16.5,
        "step_deg": 0.25,
    },
    # --- New regions: P9 full-orbit coverage ---
    "perihelion": {
        "description": "P9 perihelion corridor (Ophiuchus/Libra, ~320 AU, brightest)",
        "ra_min": 235.0, "ra_max": 258.0,
        "dec_min": -17.0, "dec_max": -1.0,
        "step_deg": 0.50,
    },
    "orion_fringe": {
        "description": "Orion/Gemini fringe above galactic plane (|b|~16-25 deg)",
        "ra_min": 108.0, "ra_max": 120.0,
        "dec_min": 23.0, "dec_max": 27.0,
        "step_deg": 0.25,
    },
    "cetus": {
        "description": "Cetus/Aquarius cleanest sky (|b|>67 deg, ecliptic quadrature)",
        "ra_min": 350.0, "ra_max": 380.0,  # wraps: 350-360 + 0-20
        "dec_min": -20.0, "dec_max": -8.0,
        "step_deg": 0.50,
    },
    "south_ecliptic": {
        "description": "South ecliptic descent (Capricornus, max southern dec)",
        "ra_min": 310.0, "ra_max": 345.0,
        "dec_min": -27.0, "dec_max": -20.0,
        "step_deg": 0.75,
    },
}

ALL_STAGES = ["scan", "crossmatch", "blink", "dust_piercer"]


# ---------------------------------------------------------------------------
# Grid generation (from run_blind_search.py)
# ---------------------------------------------------------------------------

def _galactic_latitude(ra_deg: float, dec_deg: float) -> float:
    """Approximate Galactic latitude b from ICRS (RA, Dec)."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    ra_ngp = np.radians(192.85948)
    dec_ngp = np.radians(27.12825)
    sin_b = (
        np.sin(dec) * np.sin(dec_ngp)
        + np.cos(dec) * np.cos(dec_ngp) * np.cos(ra - ra_ngp)
    )
    return float(np.degrees(np.arcsin(np.clip(sin_b, -1, 1))))


def generate_grid(
    ra_min: float, ra_max: float, dec_min: float, dec_max: float,
    step_deg: float = 0.50,
    exclude_galactic_plane: bool = True,
    galactic_lat_cut: float = 10.0,
) -> list[tuple[float, float]]:
    """Generate a grid of (RA, Dec) field centers."""
    dec_values = np.arange(dec_min, dec_max + step_deg / 2, step_deg)
    fields = []
    for dec in dec_values:
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


# ---------------------------------------------------------------------------
# Survey state (checkpoint persistence)
# ---------------------------------------------------------------------------

class SurveyState:
    """Persistent checkpoint with atomic JSON writes."""

    def __init__(self, path: Path, region: str, grid_params: dict):
        self.path = path
        self.region = region
        self.grid_params = grid_params
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = now
        self.updated_at = now

        # Stage states
        self.scan_fields_total = 0
        self.scan_fields_completed = 0
        self.scan_candidates_found = 0
        self.scanned_field_keys: set[str] = set()

        self.crossmatch_input = 0
        self.crossmatch_known = 0
        self.crossmatch_unknown = 0
        self.crossmatch_completed = False

        self.blink_input = 0
        self.blink_processed = 0
        self.blink_verdicts: dict[str, int] = {}
        self.blink_completed = False

        self.dp_input = 0
        self.dp_processed = 0
        self.dp_verdicts: dict[str, int] = {}
        self.dp_completed = False

    def save(self) -> None:
        """Atomic write: write to .tmp then os.replace."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        data = {
            "region": self.region,
            "grid_params": self.grid_params,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "scan": {
                "fields_total": self.scan_fields_total,
                "fields_completed": self.scan_fields_completed,
                "candidates_found": self.scan_candidates_found,
                "scanned_field_keys": sorted(self.scanned_field_keys),
            },
            "crossmatch": {
                "candidates_input": self.crossmatch_input,
                "candidates_known": self.crossmatch_known,
                "candidates_unknown": self.crossmatch_unknown,
                "completed": self.crossmatch_completed,
            },
            "blink": {
                "candidates_input": self.blink_input,
                "candidates_processed": self.blink_processed,
                "verdicts": self.blink_verdicts,
                "completed": self.blink_completed,
            },
            "dust_piercer": {
                "candidates_input": self.dp_input,
                "candidates_processed": self.dp_processed,
                "verdicts": self.dp_verdicts,
                "completed": self.dp_completed,
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(str(tmp_path), str(self.path))

    @classmethod
    def load(cls, path: Path) -> SurveyState:
        """Load checkpoint from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        gp = data["grid_params"]
        state = cls(path, data["region"], gp)
        state.created_at = data.get("created_at", state.created_at)

        s = data.get("scan", {})
        state.scan_fields_total = s.get("fields_total", 0)
        state.scan_fields_completed = s.get("fields_completed", 0)
        state.scan_candidates_found = s.get("candidates_found", 0)
        state.scanned_field_keys = set(s.get("scanned_field_keys", []))

        cm = data.get("crossmatch", {})
        state.crossmatch_input = cm.get("candidates_input", 0)
        state.crossmatch_known = cm.get("candidates_known", 0)
        state.crossmatch_unknown = cm.get("candidates_unknown", 0)
        state.crossmatch_completed = cm.get("completed", False)

        bk = data.get("blink", {})
        state.blink_input = bk.get("candidates_input", 0)
        state.blink_processed = bk.get("candidates_processed", 0)
        state.blink_verdicts = bk.get("verdicts", {})
        state.blink_completed = bk.get("completed", False)

        dp = data.get("dust_piercer", {})
        state.dp_input = dp.get("candidates_input", 0)
        state.dp_processed = dp.get("candidates_processed", 0)
        state.dp_verdicts = dp.get("verdicts", {})
        state.dp_completed = dp.get("completed", False)

        return state

    def bootstrap_from_existing(
        self,
        results_dir: Path,
        data_dir: Path,
        full_grid: list[tuple[float, float]],
    ) -> None:
        """First-run bootstrap: discover already-scanned fields from disk."""
        self.scan_fields_total = len(full_grid)
        grid_keys = {_field_key(ra, dec) for ra, dec in full_grid}

        # 1. Parse candidate CSVs for field coordinates
        csv_paths = [
            results_dir / "blind_search_taurus" / "candidates_taurus_partial.csv",
            results_dir / "blind_search_taurus" / "candidates_taurus_final.csv",
            results_dir / "blind_search_taurus" / "observatory_candidates.csv",
        ]
        all_candidates = []
        scanned_from_csvs: set[str] = set()
        for csv_path in csv_paths:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    all_candidates.append(df)
                    if "field_ra" in df.columns and "field_dec" in df.columns:
                        for _, row in df.iterrows():
                            key = _field_key(
                                round(float(row["field_ra"]), 3),
                                round(float(row["field_dec"]), 3),
                            )
                            scanned_from_csvs.add(key)
                except Exception as e:
                    logger.warning("Failed to read %s: %s", csv_path, e)

        # 2. Scan downloaded DSS2 images (catches 0-candidate fields)
        dss2_dir = data_dir / "dss2_red"
        scanned_from_fits: set[str] = set()
        if dss2_dir.exists():
            field_re = re.compile(
                r"field_ra(\d+\.\d+)_dec([+-]?\d+\.\d+)\.fits"
            )
            for fpath in dss2_dir.glob("field_ra*.fits"):
                m = field_re.match(fpath.name)
                if m:
                    key = _field_key(
                        round(float(m.group(1)), 3),
                        round(float(m.group(2)), 3),
                    )
                    scanned_from_fits.add(key)

        # 3. Intersect with actual grid (ignore stale downloads)
        all_scanned = (scanned_from_csvs | scanned_from_fits) & grid_keys
        self.scanned_field_keys = all_scanned
        self.scan_fields_completed = len(all_scanned)

        # 4. Merge candidates
        if all_candidates:
            combined = pd.concat(all_candidates, ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["ra_deg", "dec_deg"], keep="first",
            )
            self.scan_candidates_found = len(combined)
        else:
            self.scan_candidates_found = 0

        logger.info(
            "  Bootstrap: %d/%d fields scanned, %d candidates found",
            self.scan_fields_completed, self.scan_fields_total,
            self.scan_candidates_found,
        )

    def get_remaining_fields(
        self, full_grid: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Return unscanned fields in grid order."""
        return [
            (ra, dec) for ra, dec in full_grid
            if _field_key(ra, dec) not in self.scanned_field_keys
        ]


def _field_key(ra: float, dec: float) -> str:
    """Canonical field key matching FITS filename pattern (3 decimal places)."""
    return f"{ra:07.3f}{dec:+07.3f}"


# ---------------------------------------------------------------------------
# P9 Survey orchestrator
# ---------------------------------------------------------------------------

class P9Survey:
    """Main pipeline orchestrator with checkpoint resume."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pipeline = None  # Lazy-loaded

        # Resolve region
        if args.region:
            preset = PRESETS[args.region]
            self.ra_min = preset["ra_min"]
            self.ra_max = preset["ra_max"]
            self.dec_min = preset["dec_min"]
            self.dec_max = preset["dec_max"]
            self.step = args.step or preset["step_deg"]
            self.region_name = args.region
            self.description = preset["description"]
        else:
            self.ra_min = args.ra_min
            self.ra_max = args.ra_max
            self.dec_min = args.dec_min
            self.dec_max = args.dec_max
            self.step = args.step or 0.50
            self.region_name = "custom"
            self.description = (
                f"Custom: RA [{self.ra_min:.1f}, {self.ra_max:.1f}], "
                f"Dec [{self.dec_min:.1f}, {self.dec_max:.1f}]"
            )

        self.grid_params = {
            "ra_min": self.ra_min, "ra_max": self.ra_max,
            "dec_min": self.dec_min, "dec_max": self.dec_max,
            "step": self.step,
        }

        self.output_dir = args.output_dir or Path(
            f"results/survey/{self.region_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        _setup_logging(self.output_dir)

        # Load or create checkpoint
        self.ckpt_path = self.output_dir / "checkpoint.json"
        if self.ckpt_path.exists():
            self.state = SurveyState.load(self.ckpt_path)
            logger.info("  Loaded checkpoint: %s", self.ckpt_path)
        else:
            self.state = SurveyState(
                self.ckpt_path, self.region_name, self.grid_params,
            )

        # Signal handler for graceful shutdown
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Save checkpoint on Ctrl+C."""
        if self._interrupted:
            # Second Ctrl+C: hard exit
            logger.warning("\n  FORCED EXIT (checkpoint already saved)")
            sys.exit(1)
        self._interrupted = True
        logger.warning(
            "\n  INTERRUPTED — saving checkpoint before exit..."
        )
        self.state.save()
        logger.info("  Checkpoint saved: %s", self.ckpt_path)
        sys.exit(0)

    # ---------------------------------------------------------------
    # Main entry
    # ---------------------------------------------------------------

    def run(self) -> None:
        """Execute the survey pipeline."""
        stages = self._resolve_stages()
        full_grid = generate_grid(
            self.ra_min, self.ra_max, self.dec_min, self.dec_max,
            step_deg=self.step,
        )

        # Bootstrap if first run
        if self.state.scan_fields_total == 0:
            logger.info("  First run — bootstrapping from existing results...")
            self.state.bootstrap_from_existing(
                Path("results"), self.args.data_dir, full_grid,
            )
            self.state.save()

        remaining_fields = self.state.get_remaining_fields(full_grid)
        self._print_banner(full_grid, remaining_fields, stages)

        if self.args.dry_run:
            self._print_dry_run_stats(full_grid, remaining_fields)
            return

        # ---- STAGE 1: SCAN ----
        if "scan" in stages:
            if remaining_fields:
                self._run_scan(remaining_fields)
            else:
                logger.info("\n  [SCAN] All %d fields already scanned.", len(full_grid))

        # ---- Load all scan candidates for downstream ----
        scan_csv = self.output_dir / "scan_candidates.csv"
        if scan_csv.exists():
            all_candidates = pd.read_csv(scan_csv)
        else:
            all_candidates = self._merge_all_scan_candidates()
            if len(all_candidates) > 0:
                all_candidates.to_csv(scan_csv, index=False)

        if len(all_candidates) == 0:
            logger.info("\n  No candidates found. Survey complete.")
            self._print_funnel()
            return

        # ---- STAGE 2: CROSSMATCH ----
        unknown_csv = self.output_dir / "crossmatch_unknown.csv"
        if "crossmatch" in stages:
            unknown_df = self._run_crossmatch(all_candidates)
        elif unknown_csv.exists():
            unknown_df = pd.read_csv(unknown_csv)
        else:
            unknown_df = all_candidates

        if len(unknown_df) == 0:
            logger.info("\n  All candidates matched to known objects.")
            self._print_funnel()
            return

        # ---- STAGE 3: BLINK ----
        blink_csv = self.output_dir / "blink_results.csv"
        if "blink" in stages:
            blink_df = self._run_blink(unknown_df)
        elif blink_csv.exists():
            blink_df = pd.read_csv(blink_csv)
        else:
            blink_df = None

        if blink_df is None or len(blink_df) == 0:
            logger.info("\n  No blink results available.")
            self._print_funnel()
            return

        # ---- STAGE 4: DUST PIERCER ----
        if "dust_piercer" in stages:
            self._run_dust_piercer(blink_df)

        # ---- FINAL ----
        self._print_funnel()
        self._save_report()
        self.state.save()

    # ---------------------------------------------------------------
    # Stage 1: SCAN
    # ---------------------------------------------------------------

    def _run_scan(self, remaining: list[tuple[float, float]]) -> None:
        """Resume blind search from where it left off."""
        from astroworld.ml.pipeline import (
            ObservatoryPipeline,
            PipelineConfig,
        )

        logger.info("")
        logger.info("=" * 65)
        logger.info("  STAGE 1: BLIND SEARCH SCAN")
        logger.info("=" * 65)
        logger.info(
            "  Remaining: %d fields (%d already done)",
            len(remaining), self.state.scan_fields_completed,
        )

        est_min = len(remaining) * 37.5 / 60
        logger.info("  ETA: ~%.0f min @ 37.5s/field", est_min)
        logger.info("-" * 65)

        # Load pipeline (lazy)
        if self.pipeline is None:
            logger.info("  Loading spectral model...")
            config = PipelineConfig(
                mode="spectral_only",
                spectral_threshold=self.args.spectral_threshold,
                mc_samples=self.args.mc_samples,
                planck_classes_keep=["p9_candidate"],
                kinematic_enabled=True,
                device=self.args.device,
                output_dir=self.output_dir,
                data_dir=self.args.data_dir,
            )
            self.pipeline = ObservatoryPipeline.from_checkpoints(
                spectral_checkpoint=self.args.spectral_checkpoint,
                config=config,
            )

        # Load existing candidates
        scan_csv = self.output_dir / "scan_candidates.csv"
        if scan_csv.exists():
            accumulated = pd.read_csv(scan_csv)
        else:
            accumulated = self._merge_all_scan_candidates()

        batch_size = self.args.batch_size
        n_total = len(remaining)
        n_batches = (n_total + batch_size - 1) // batch_size
        t_start = time.time()

        for batch_idx in range(n_batches):
            if self._interrupted:
                break

            start = batch_idx * batch_size
            end = min(start + batch_size, n_total)
            batch = remaining[start:end]

            logger.info(
                "\n  BATCH %d/%d (fields %d-%d of %d)",
                batch_idx + 1, n_batches, start + 1, end, n_total,
            )

            summary = self.pipeline.run(fields=batch)

            # Accumulate candidates
            if len(summary.candidate_table) > 0:
                accumulated = pd.concat(
                    [accumulated, summary.candidate_table], ignore_index=True,
                )

            # Mark fields as scanned
            for ra, dec in batch:
                self.state.scanned_field_keys.add(_field_key(ra, dec))
            self.state.scan_fields_completed += len(batch)
            self.state.scan_candidates_found = len(accumulated)

            # Save checkpoint + CSV
            accumulated.to_csv(scan_csv, index=False)
            self.state.save()

            # Progress
            elapsed = time.time() - t_start
            done = end
            rate = elapsed / done
            eta = rate * (n_total - done)
            new_in_batch = len(summary.candidate_table)
            logger.info(
                "  [%d/%d] +%d candidates (total %d) | "
                "%.1fs/field | ETA %.0f min",
                done, n_total, new_in_batch, len(accumulated),
                rate, eta / 60,
            )

        logger.info("\n  SCAN complete: %d candidates total", len(accumulated))

    # ---------------------------------------------------------------
    # Stage 2: CROSSMATCH
    # ---------------------------------------------------------------

    def _run_crossmatch(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Cross-match against SIMBAD + Gaia DR3."""
        from astroworld.ml.crossmatch import CatalogMatcher

        logger.info("")
        logger.info("=" * 65)
        logger.info("  STAGE 2: CATALOG CROSS-MATCH")
        logger.info("=" * 65)

        known_csv = self.output_dir / "crossmatch_known.csv"
        unknown_csv = self.output_dir / "crossmatch_unknown.csv"

        # Check for already-processed candidates
        already_processed_coords: set[tuple[float, float]] = set()
        prev_known = pd.DataFrame()
        prev_unknown = pd.DataFrame()

        if known_csv.exists():
            prev_known = pd.read_csv(known_csv)
            for _, row in prev_known.iterrows():
                already_processed_coords.add(
                    (round(row["ra_deg"], 5), round(row["dec_deg"], 5))
                )
        if unknown_csv.exists():
            prev_unknown = pd.read_csv(unknown_csv)
            for _, row in prev_unknown.iterrows():
                already_processed_coords.add(
                    (round(row["ra_deg"], 5), round(row["dec_deg"], 5))
                )

        # Find new candidates not yet crossmatched
        new_mask = candidates.apply(
            lambda r: (round(r["ra_deg"], 5), round(r["dec_deg"], 5))
            not in already_processed_coords,
            axis=1,
        )
        new_candidates = candidates[new_mask].copy()

        if len(new_candidates) == 0:
            logger.info(
                "  All %d candidates already crossmatched.", len(candidates),
            )
            self.state.crossmatch_completed = True
            self.state.save()
            if unknown_csv.exists():
                return pd.read_csv(unknown_csv)
            return candidates

        logger.info(
            "  %d new candidates to crossmatch (%d already done)",
            len(new_candidates), len(already_processed_coords),
        )
        logger.info("-" * 65)

        matcher = CatalogMatcher(
            search_radius_arcsec=self.args.crossmatch_radius,
            rate_limit_s=0.2,
        )
        df_known_new, df_unknown_new = matcher.filter_candidates(
            new_candidates, verbose=True,
        )

        # Merge with previous results
        all_known = pd.concat(
            [prev_known, df_known_new], ignore_index=True,
        ) if len(df_known_new) > 0 else prev_known

        all_unknown = pd.concat(
            [prev_unknown, df_unknown_new], ignore_index=True,
        ) if len(df_unknown_new) > 0 else prev_unknown

        # Save
        if len(all_known) > 0:
            all_known.to_csv(known_csv, index=False)
        if len(all_unknown) > 0:
            all_unknown.to_csv(unknown_csv, index=False)

        # Update state
        self.state.crossmatch_input = len(candidates)
        self.state.crossmatch_known = len(all_known)
        self.state.crossmatch_unknown = len(all_unknown)
        self.state.crossmatch_completed = True
        self.state.save()

        logger.info(
            "\n  CROSSMATCH complete: %d known, %d unknown",
            len(all_known), len(all_unknown),
        )

        return all_unknown

    # ---------------------------------------------------------------
    # Stage 3: BLINK (Time Machine)
    # ---------------------------------------------------------------

    def _run_blink(self, unknown_df: pd.DataFrame) -> pd.DataFrame:
        """Multi-epoch optical blink analysis."""
        from astroworld.imaging.time_machine import (
            blink_candidates,
            make_blink_card,
        )

        logger.info("")
        logger.info("=" * 65)
        logger.info("  STAGE 3: TIME MACHINE (Multi-Epoch Blink)")
        logger.info("=" * 65)

        # Apply temperature filter
        to_blink = unknown_df.copy()
        if self.args.temp_max and "temperature_k" in to_blink.columns:
            to_blink = to_blink[
                to_blink["temperature_k"] <= self.args.temp_max
            ].copy()
            logger.info(
                "  Temperature filter (T <= %.0fK): %d -> %d candidates",
                self.args.temp_max, len(unknown_df), len(to_blink),
            )

        # Check for already-processed
        blink_csv = self.output_dir / "blink_results.csv"
        prev_blink = pd.DataFrame()
        already_blinked: set[tuple[float, float]] = set()

        if blink_csv.exists():
            prev_blink = pd.read_csv(blink_csv)
            for _, row in prev_blink.iterrows():
                already_blinked.add(
                    (round(row["ra_deg"], 5), round(row["dec_deg"], 5))
                )

        new_mask = to_blink.apply(
            lambda r: (round(r["ra_deg"], 5), round(r["dec_deg"], 5))
            not in already_blinked,
            axis=1,
        )
        new_to_blink = to_blink[new_mask].copy()

        if len(new_to_blink) == 0:
            logger.info(
                "  All %d candidates already blinked.", len(to_blink),
            )
            self.state.blink_completed = True
            self.state.save()
            return prev_blink if len(prev_blink) > 0 else pd.DataFrame()

        logger.info(
            "  %d new candidates to blink (%d already done)",
            len(new_to_blink), len(already_blinked),
        )
        logger.info("-" * 65)

        # Run blink analysis
        results_list = blink_candidates(
            new_to_blink,
            dss2_dir=self.args.data_dir / "dss2_red",
            ps1_dir=self.args.data_dir / "ps1_r",
            verbose=True,
        )

        # Generate cards
        cards_dir = self.output_dir / "blink_cards"
        if not self.args.no_cards:
            cards_dir.mkdir(parents=True, exist_ok=True)

        # Build result rows
        new_rows = []
        for i, ((res, c1, c2), (_, orig_row)) in enumerate(
            zip(results_list, new_to_blink.iterrows())
        ):
            row = {
                "ra_deg": res.ra_deg,
                "dec_deg": res.dec_deg,
                "verdict": res.verdict,
                "confidence": res.confidence,
                "shift_arcsec": res.shift_arcsec,
                "shift_ra_arcsec": res.shift_ra_arcsec,
                "shift_dec_arcsec": res.shift_dec_arcsec,
                "snr_epoch1": res.snr_epoch1,
                "snr_epoch2": res.snr_epoch2,
                "flux_ratio": res.flux_ratio,
                "epoch1_year": res.epoch1_year,
                "epoch2_year": res.epoch2_year,
                "baseline_years": res.baseline_years,
                "implied_pm_arcsec_yr": res.implied_pm_arcsec_yr,
            }
            # Carry forward original columns
            for col in ["temperature_k", "temperature_std_k", "probability"]:
                if col in orig_row.index:
                    row[col] = orig_row[col]
            new_rows.append(row)

            # Card
            if not self.args.no_cards and c1 is not None and c2 is not None:
                card_name = (
                    f"blink_{len(already_blinked) + i + 1:03d}_"
                    f"ra{res.ra_deg:.3f}_dec{res.dec_deg:+.3f}_"
                    f"{res.verdict}.png"
                )
                temp_k = (
                    float(orig_row["temperature_k"])
                    if "temperature_k" in orig_row.index else None
                )
                make_blink_card(
                    res, c1, c2,
                    save_path=cards_dir / card_name,
                    candidate_temp_k=temp_k,
                )

            # Alert on MOVED
            if res.verdict == "MOVED":
                self._alert_moved(res)

        # Merge with previous
        new_blink_df = pd.DataFrame(new_rows)
        all_blink = pd.concat(
            [prev_blink, new_blink_df], ignore_index=True,
        ) if len(prev_blink) > 0 else new_blink_df

        all_blink.to_csv(blink_csv, index=False)

        # Update state
        self.state.blink_input = len(to_blink)
        self.state.blink_processed = len(all_blink)
        verdicts = all_blink["verdict"].value_counts().to_dict()
        self.state.blink_verdicts = {k: int(v) for k, v in verdicts.items()}
        self.state.blink_completed = True
        self.state.save()

        logger.info(
            "\n  BLINK complete: %d analyzed", len(all_blink),
        )
        for v, c in verdicts.items():
            marker = " *** CHECK ***" if v == "MOVED" else ""
            logger.info("    %s: %d%s", v, c, marker)

        return all_blink

    # ---------------------------------------------------------------
    # Stage 4: DUST PIERCER
    # ---------------------------------------------------------------

    def _run_dust_piercer(self, blink_df: pd.DataFrame) -> None:
        """IR morphology + NEOWISE proper motion search."""
        from astroworld.imaging.dust_piercer import (
            analyze_candidates,
            make_dust_piercer_card,
            DustPiercerResult,
            MIN_SNR_MORPHOLOGY,
        )

        logger.info("")
        logger.info("=" * 65)
        logger.info("  STAGE 4: DUST PIERCER (IR Morphology + NEOWISE PM)")
        logger.info("=" * 65)

        # Filter: typically run on ABSENT candidates (no optical source)
        to_analyze = blink_df.copy()

        # Check already processed
        dp_csv = self.output_dir / "dust_piercer_results.csv"
        prev_dp = pd.DataFrame()
        already_dp: set[tuple[float, float]] = set()

        if dp_csv.exists():
            prev_dp = pd.read_csv(dp_csv)
            for _, row in prev_dp.iterrows():
                already_dp.add(
                    (round(row["ra_deg"], 5), round(row["dec_deg"], 5))
                )

        new_mask = to_analyze.apply(
            lambda r: (round(r["ra_deg"], 5), round(r["dec_deg"], 5))
            not in already_dp,
            axis=1,
        )
        new_to_dp = to_analyze[new_mask].copy()

        if len(new_to_dp) == 0:
            logger.info(
                "  All %d candidates already analyzed.", len(to_analyze),
            )
            self.state.dp_completed = True
            self.state.save()
            return

        logger.info(
            "  %d new candidates for Dust Piercer (%d already done)",
            len(new_to_dp), len(already_dp),
        )
        logger.info("-" * 65)

        # Run analysis
        results = analyze_candidates(
            new_to_dp,
            w2_dir=self.args.data_dir / "wise_4.6",
            rate_limit_sec=1.0,
            skip_neowise=False,
            verbose=True,
        )

        # Generate cards
        cards_dir = self.output_dir / "dust_piercer_cards"
        if not self.args.no_cards:
            cards_dir.mkdir(parents=True, exist_ok=True)

        # Build result rows
        new_rows = []
        for i, (res, (_, orig_row)) in enumerate(
            zip(results, new_to_dp.iterrows())
        ):
            row = {
                "candidate_id": len(already_dp) + i + 1,
                "ra_deg": res.ra_deg,
                "dec_deg": res.dec_deg,
                "verdict": res.verdict,
                "confidence": res.confidence,
                "passed_morphology": res.passed_morphology,
                "has_significant_pm": res.has_significant_pm,
            }
            if res.morphology:
                row["morph_snr"] = round(res.morphology.snr, 2)
                row["morph_pointiness"] = round(res.morphology.pointiness, 3)
                row["morph_fwhm_x"] = round(res.morphology.fwhm_x_arcsec, 2)
                row["morph_fwhm_y"] = round(res.morphology.fwhm_y_arcsec, 2)
                row["morph_ellipticity"] = round(res.morphology.ellipticity, 3)
                row["morph_is_point"] = res.morphology.is_point_source
            if res.proper_motion and res.proper_motion.n_epochs >= 3:
                pm = res.proper_motion
                row["pm_n_detections"] = pm.n_detections
                row["pm_n_epochs"] = pm.n_epochs
                row["pm_baseline_yr"] = round(pm.baseline_years, 2)
                row["pm_ra_arcsec_yr"] = round(pm.mu_ra_arcsec_yr, 4)
                row["pm_dec_arcsec_yr"] = round(pm.mu_dec_arcsec_yr, 4)
                row["pm_total_arcsec_yr"] = round(pm.mu_total_arcsec_yr, 4)
                row["pm_significance"] = round(pm.pm_significance, 2)
                row["pm_mean_w2mag"] = round(pm.mean_w2mag, 2)
            for col in ["temperature_k", "temperature_std_k", "probability"]:
                if col in orig_row.index:
                    row[col] = orig_row[col]
            new_rows.append(row)

            # Card
            if not self.args.no_cards:
                card_name = (
                    f"dp_{len(already_dp) + i + 1:03d}_"
                    f"ra{res.ra_deg:.3f}_dec{res.dec_deg:+.3f}_"
                    f"{res.verdict}.png"
                )
                make_dust_piercer_card(
                    res, None,
                    save_path=cards_dir / card_name,
                    candidate_temp_k=(
                        float(orig_row["temperature_k"])
                        if "temperature_k" in orig_row.index
                        and pd.notna(orig_row.get("temperature_k"))
                        else None
                    ),
                )

            # *** CRITICAL ALERT: POINT_MOVING ***
            if res.verdict == "POINT_MOVING":
                self._alert_point_moving(res, orig_row)

        # Merge with previous
        new_dp_df = pd.DataFrame(new_rows)
        all_dp = pd.concat(
            [prev_dp, new_dp_df], ignore_index=True,
        ) if len(prev_dp) > 0 else new_dp_df

        all_dp.to_csv(dp_csv, index=False)

        # Update state
        self.state.dp_input = len(to_analyze)
        self.state.dp_processed = len(all_dp)
        verdicts = all_dp["verdict"].value_counts().to_dict()
        self.state.dp_verdicts = {k: int(v) for k, v in verdicts.items()}
        self.state.dp_completed = True
        self.state.save()

        logger.info("\n  DUST PIERCER complete: %d analyzed", len(all_dp))
        for v, c in verdicts.items():
            marker = " *** P9 CANDIDATE ***" if v == "POINT_MOVING" else ""
            logger.info("    %s: %d%s", v, c, marker)

    # ---------------------------------------------------------------
    # Alerts
    # ---------------------------------------------------------------

    def _alert_moved(self, res) -> None:
        """Alert for MOVED verdict in blink analysis."""
        logger.info("")
        logger.info("!" * 65)
        logger.info("  MOVED DETECTION IN OPTICAL:")
        logger.info(
            "  RA=%.4f Dec=%+.4f  shift=%.2f arcsec  PM=%.3f\"/yr",
            res.ra_deg, res.dec_deg, res.shift_arcsec,
            res.implied_pm_arcsec_yr,
        )
        logger.info("!" * 65)

    def _alert_point_moving(self, res, orig_row) -> None:
        """Critical alert for POINT_MOVING in Dust Piercer."""
        temp_str = ""
        if "temperature_k" in orig_row.index and pd.notna(
            orig_row.get("temperature_k")
        ):
            temp_str = f"\n  T   = {orig_row['temperature_k']:.1f} K"

        logger.info("")
        logger.info("!" * 70)
        logger.info("!" * 70)
        logger.info("  *** POINT_MOVING DETECTION — POSSIBLE PLANET 9 ***")
        logger.info("  RA  = %.6f", res.ra_deg)
        logger.info("  Dec = %+.6f", res.dec_deg)
        if res.proper_motion:
            pm = res.proper_motion
            logger.info(
                "  PM  = %.4f arcsec/yr (%.1f sigma)",
                pm.mu_total_arcsec_yr, pm.pm_significance,
            )
            logger.info("  W2  = %.1f mag", pm.mean_w2mag)
        if res.morphology:
            logger.info("  Pointiness = %.3f", res.morphology.pointiness)
        if temp_str:
            logger.info("  %s", temp_str.strip())
        logger.info("!" * 70)
        logger.info("!" * 70)

    # ---------------------------------------------------------------
    # Display
    # ---------------------------------------------------------------

    def _resolve_stages(self) -> list[str]:
        """Determine which stages to run."""
        if self.args.scan_only:
            return ["scan"]
        if self.args.downstream_only:
            return ["crossmatch", "blink", "dust_piercer"]
        if self.args.stages:
            requested = [s.strip() for s in self.args.stages.split(",")]
            for s in requested:
                if s not in ALL_STAGES:
                    logger.error("Unknown stage: %s", s)
                    sys.exit(1)
            return requested
        return list(ALL_STAGES)

    def _print_banner(
        self,
        full_grid: list,
        remaining: list,
        stages: list[str],
    ) -> None:
        """Print survey header."""
        logger.info("")
        logger.info("=" * 65)
        logger.info("  P9 SURVEY ORCHESTRATOR — Phase 20")
        logger.info("  Automated Pipeline: Scan -> Crossmatch -> Blink -> Dust Piercer")
        logger.info("=" * 65)
        logger.info("  Region:      %s — %s", self.region_name, self.description)
        logger.info(
            "  RA range:    [%.1f, %.1f] deg", self.ra_min, self.ra_max,
        )
        logger.info(
            "  Dec range:   [%.1f, %.1f] deg", self.dec_min, self.dec_max,
        )
        logger.info("  Step:        %.3f deg (%.1f')", self.step, self.step * 60)
        logger.info(
            "  Grid:        %d fields total", len(full_grid),
        )
        logger.info(
            "  Progress:    %d/%d scanned (%.1f%%)",
            self.state.scan_fields_completed,
            len(full_grid),
            100 * self.state.scan_fields_completed / max(len(full_grid), 1),
        )
        logger.info("  Remaining:   %d fields", len(remaining))
        logger.info("  Candidates:  %d found so far", self.state.scan_candidates_found)
        logger.info("  Stages:      %s", " -> ".join(stages))
        logger.info("  Output:      %s", self.output_dir)
        logger.info("  Checkpoint:  %s", self.ckpt_path)
        logger.info("=" * 65)

    def _print_dry_run_stats(
        self, full_grid: list, remaining: list,
    ) -> None:
        """Show what would happen without executing."""
        logger.info("")
        logger.info("  [DRY RUN] No actions taken.")
        logger.info("")
        logger.info("  Scan estimate:")
        logger.info("    Fields to scan: %d", len(remaining))
        est_sec = len(remaining) * 37.5
        logger.info("    Est. time:      %.0f min (%.1f hr)", est_sec / 60, est_sec / 3600)
        logger.info("")
        logger.info("  Downstream estimate:")
        logger.info("    Candidates:     %d", self.state.scan_candidates_found)
        logger.info(
            "    Crossmatch:     %s",
            "DONE" if self.state.crossmatch_completed else "PENDING",
        )
        logger.info(
            "    Blink:          %s",
            "DONE" if self.state.blink_completed else "PENDING",
        )
        logger.info(
            "    Dust Piercer:   %s",
            "DONE" if self.state.dp_completed else "PENDING",
        )

    def _print_funnel(self) -> None:
        """Print the candidate survival funnel."""
        s = self.state
        logger.info("")
        logger.info("=" * 65)
        logger.info("  P9 SURVEY — CANDIDATE FUNNEL")
        logger.info("=" * 65)
        logger.info(
            "  Fields scanned:      %d / %d (%.1f%%)",
            s.scan_fields_completed, s.scan_fields_total,
            100 * s.scan_fields_completed / max(s.scan_fields_total, 1),
        )
        logger.info("  Raw candidates:      %d", s.scan_candidates_found)
        if s.crossmatch_completed:
            logger.info(
                "  After crossmatch:    %d unknown (%d known rejected)",
                s.crossmatch_unknown, s.crossmatch_known,
            )
        if s.blink_verdicts:
            logger.info("  After blink:")
            for verdict, count in sorted(s.blink_verdicts.items()):
                marker = " *** CHECK ***" if verdict == "MOVED" else ""
                logger.info("    %-12s: %d%s", verdict, count, marker)
        if s.dp_verdicts:
            logger.info("  After dust piercer:")
            for verdict, count in sorted(s.dp_verdicts.items()):
                marker = " *** P9 CANDIDATE ***" if verdict == "POINT_MOVING" else ""
                logger.info("    %-15s: %d%s", verdict, count, marker)
        logger.info("=" * 65)

    def _save_report(self) -> None:
        """Save human-readable survey report."""
        s = self.state
        lines = [
            "=" * 65,
            "  P9 SURVEY REPORT",
            "=" * 65,
            f"  Region:        {self.region_name}",
            f"  Generated:     {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"  Fields total:  {s.scan_fields_total}",
            f"  Fields done:   {s.scan_fields_completed}",
            f"  Candidates:    {s.scan_candidates_found}",
            "",
        ]
        if s.crossmatch_completed:
            lines.extend([
                f"  Crossmatch known:    {s.crossmatch_known}",
                f"  Crossmatch unknown:  {s.crossmatch_unknown}",
            ])
        if s.blink_verdicts:
            lines.append("  Blink verdicts:")
            for v, c in sorted(s.blink_verdicts.items()):
                lines.append(f"    {v}: {c}")
        if s.dp_verdicts:
            lines.append("  Dust Piercer verdicts:")
            for v, c in sorted(s.dp_verdicts.items()):
                lines.append(f"    {v}: {c}")
        lines.append("=" * 65)

        report_path = self.output_dir / "survey_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("  Report saved: %s", report_path)

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _merge_all_scan_candidates(self) -> pd.DataFrame:
        """Merge candidates from all existing blind search runs."""
        csv_paths = [
            Path("results/blind_search_taurus/candidates_taurus_partial.csv"),
            Path("results/blind_search_taurus/candidates_taurus_final.csv"),
            Path("results/blind_search_taurus/observatory_candidates.csv"),
        ]
        dfs = []
        for p in csv_paths:
            if p.exists():
                try:
                    dfs.append(pd.read_csv(p))
                except Exception:
                    pass
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["ra_deg", "dec_deg"], keep="first",
            )
            return combined
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 20: P9 Survey Orchestrator — Automated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--spectral-checkpoint", type=Path, required=True,
        help="Path to SpectralSiameseNet checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    # Region
    region_group = parser.add_mutually_exclusive_group()
    region_group.add_argument(
        "--region", type=str, choices=list(PRESETS.keys()),
        help="Preset search region",
    )
    parser.add_argument("--ra-min", type=float, default=None)
    parser.add_argument("--ra-max", type=float, default=None)
    parser.add_argument("--dec-min", type=float, default=None)
    parser.add_argument("--dec-max", type=float, default=None)
    parser.add_argument("--step", type=float, default=None)

    # Mode
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Only scan, skip downstream stages",
    )
    parser.add_argument(
        "--downstream-only", action="store_true",
        help="Only run downstream stages on existing candidates",
    )
    parser.add_argument(
        "--stages", type=str, default=None,
        help="Comma-separated stages: scan,crossmatch,blink,dust_piercer",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show stats without executing",
    )

    # Pipeline
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--spectral-threshold", type=float, default=0.80)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--crossmatch-radius", type=float, default=10.0)
    parser.add_argument(
        "--temp-max", type=float, default=None,
        help="Only blink candidates with T <= this (K)",
    )
    parser.add_argument("--no-cards", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/survey_images"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate region
    if not args.region and not all(
        v is not None
        for v in [args.ra_min, args.ra_max, args.dec_min, args.dec_max]
    ):
        print(
            "ERROR: Provide --region or all of --ra-min/--ra-max/"
            "--dec-min/--dec-max",
            file=sys.stderr,
        )
        sys.exit(1)

    # Force UTF-8 on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    survey = P9Survey(args)
    survey.run()


if __name__ == "__main__":
    main()
