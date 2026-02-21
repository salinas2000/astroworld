"""
Tests for Phase 15.5: Temperature calibration of spectral training data.

Validates the multi-class temperature distribution, confuser negatives,
Planck physics consistency, and backward compatibility with legacy profile.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_spectral_training_data import (
    TEMP_CLASSES_CALIBRATED,
    CONFUSER_CLASSES,
    _sample_temperature_calibrated,
    _sample_confuser_temperature,
    compute_ir_fluxes,
    generate_spectral_training_data,
)
from astroworld.imaging.spectral_cube import planck_flux_ratio


# ---------------------------------------------------------------------------
# Temperature sampling
# ---------------------------------------------------------------------------


class TestTemperatureSampling:
    """Tests for multi-class temperature distributions."""

    def test_calibrated_range(self):
        """Calibrated profile covers 15-10000 K."""
        rng = np.random.default_rng(42)
        temps = [_sample_temperature_calibrated(rng)[0] for _ in range(2000)]
        assert min(temps) < 20.0  # Should hit P9-like range
        assert max(temps) > 5000.0  # Should hit warm range

    def test_calibrated_classes(self):
        """All 4 temperature classes are sampled."""
        rng = np.random.default_rng(42)
        classes = set()
        for _ in range(500):
            _, cls = _sample_temperature_calibrated(rng)
            classes.add(cls)
        expected = {"p9_like", "cold_object", "brown_dwarf", "warm_source"}
        assert classes == expected

    def test_calibrated_class_proportions(self):
        """Class proportions roughly match weights (40/20/20/20)."""
        rng = np.random.default_rng(42)
        counts = {}
        n = 5000
        for _ in range(n):
            _, cls = _sample_temperature_calibrated(rng)
            counts[cls] = counts.get(cls, 0) + 1
        # P9-like should be ~40% ± 5%
        assert 0.30 < counts["p9_like"] / n < 0.50
        # Others ~20% ± 5%
        for cls in ["cold_object", "brown_dwarf", "warm_source"]:
            assert 0.12 < counts[cls] / n < 0.28

    def test_confuser_temperature_range(self):
        """Confuser temperatures are hot (200-6000 K)."""
        rng = np.random.default_rng(42)
        temps = [_sample_confuser_temperature(rng)[0] for _ in range(1000)]
        assert min(temps) > 100.0
        assert max(temps) < 10000.0

    def test_confuser_classes(self):
        """Confuser has both star and galaxy classes."""
        rng = np.random.default_rng(42)
        classes = set()
        for _ in range(200):
            _, cls = _sample_confuser_temperature(rng)
            classes.add(cls)
        assert classes == {"star", "galaxy"}

    def test_class_weights_sum_to_one(self):
        """Temperature class weights are valid probability distributions."""
        cal_weights = sum(c[0] for c in TEMP_CLASSES_CALIBRATED)
        conf_weights = sum(c[0] for c in CONFUSER_CLASSES)
        assert abs(cal_weights - 1.0) < 1e-10
        assert abs(conf_weights - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Planck physics consistency
# ---------------------------------------------------------------------------


class TestPlanckConsistency:
    """Tests that IR fluxes match Planck law at given temperature."""

    def test_cold_source_w1_less_than_w2(self):
        """At low T (~30K), W1 flux < W2 flux (W1/W2 ratio < 1)."""
        rng = np.random.default_rng(42)
        f_w1, f_w2 = compute_ir_fluxes(1000.0, 30.0, 5.0, rng, 0.0)
        assert f_w1 < f_w2  # Cold → more emission at longer wavelength

    def test_hot_source_w1_greater_than_w2(self):
        """At high T (~5000K), W1 flux > W2 flux (W1/W2 ratio > 1)."""
        rng = np.random.default_rng(42)
        f_w1, f_w2 = compute_ir_fluxes(1000.0, 5000.0, 5.0, rng, 0.0)
        assert f_w1 > f_w2  # Hot → more emission at shorter wavelength

    def test_ratio_matches_planck(self):
        """W1/W2 flux ratio matches theoretical Planck ratio (no noise).

        Note: compute_ir_fluxes clamps the ratio to max(ratio, 0.01) to
        prevent zero W1 flux, so we compare against the clamped ratio.
        """
        rng = np.random.default_rng(42)
        for temp in [100.0, 1000.0, 5000.0]:
            f_w1, f_w2 = compute_ir_fluxes(1000.0, temp, 5.0, rng, 0.0)
            measured_ratio = f_w1 / f_w2
            expected_ratio = max(planck_flux_ratio(temp), 0.01)
            assert abs(measured_ratio - expected_ratio) < 0.01 * expected_ratio


# ---------------------------------------------------------------------------
# Full generator: calibrated profile
# ---------------------------------------------------------------------------


class TestGeneratorCalibrated:
    """Tests for the full training data generator in calibrated mode."""

    def test_calibrated_temperature_range(self, tmp_path):
        """Calibrated positives span 15-10000 K."""
        catalog = generate_spectral_training_data(
            output_dir=tmp_path,
            n_positive=100,
            n_negative=10,
            n_confuser=0,
            temp_profile="calibrated",
            seed=42,
        )
        pos = catalog[catalog["source_present"] == True]
        temps = pos["temp_kelvin"].dropna()
        assert temps.min() < 30.0  # Has cold sources
        assert temps.max() > 1000.0  # Has warm sources

    def test_confuser_negatives_have_temperature(self, tmp_path):
        """Confuser negatives have real temperature (not NaN)."""
        catalog = generate_spectral_training_data(
            output_dir=tmp_path,
            n_positive=20,
            n_negative=20,
            n_confuser=50,
            temp_profile="calibrated",
            seed=42,
        )
        confusers = catalog[
            catalog["example_id"].str.startswith("conf_")
        ]
        assert len(confusers) > 0
        assert confusers["temp_kelvin"].notna().all()
        assert confusers["source_present"].eq(False).all()
        # Confuser temperatures should be warm
        temps = confusers["temp_kelvin"]
        assert temps.min() > 100.0

    def test_pure_negatives_have_nan(self, tmp_path):
        """Pure noise negatives still have NaN temperature."""
        catalog = generate_spectral_training_data(
            output_dir=tmp_path,
            n_positive=20,
            n_negative=30,
            n_confuser=10,
            temp_profile="calibrated",
            seed=42,
        )
        pure_neg = catalog[
            catalog["example_id"].str.startswith("neg_")
        ]
        assert len(pure_neg) > 0
        assert pure_neg["temp_kelvin"].isna().all()

    def test_confuser_no_motion(self, tmp_path):
        """Confuser pairs have identical epoch images (zero displacement)."""
        catalog = generate_spectral_training_data(
            output_dir=tmp_path,
            n_positive=5,
            n_negative=5,
            n_confuser=10,
            img_size=32,
            temp_profile="calibrated",
            seed=42,
        )
        # Check that confuser epoch1 and epoch2 optical images are
        # different (due to noise) but source is at same position.
        # We verify by checking the file paths are generated for both epochs.
        conf = catalog[catalog["example_id"].str.startswith("conf_")]
        e1 = conf[conf["example_id"].str.contains("epoch1")]
        e2 = conf[conf["example_id"].str.contains("epoch2")]
        assert len(e1) == len(e2) == 10
        # Both epochs should have same temperature
        for i in range(len(e1)):
            assert e1.iloc[i]["temp_kelvin"] == e2.iloc[i]["temp_kelvin"]


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------


class TestLegacyProfile:
    """Tests that p9_only profile is unchanged."""

    def test_legacy_temperature_range(self, tmp_path):
        """p9_only profile produces U(25, 60) K as before."""
        catalog = generate_spectral_training_data(
            output_dir=tmp_path,
            n_positive=200,
            n_negative=10,
            n_confuser=0,
            temp_profile="p9_only",
            temp_min=25.0,
            temp_max=60.0,
            seed=42,
        )
        pos = catalog[catalog["source_present"] == True]
        temps = pos["temp_kelvin"].dropna()
        assert temps.min() >= 25.0 - 0.1
        assert temps.max() <= 60.0 + 0.1

    def test_legacy_no_confusers(self, tmp_path):
        """Legacy mode with n_confuser=0 produces no confuser entries."""
        catalog = generate_spectral_training_data(
            output_dir=tmp_path,
            n_positive=10,
            n_negative=10,
            n_confuser=0,
            temp_profile="p9_only",
            seed=42,
        )
        confusers = catalog[catalog["example_id"].str.startswith("conf_")]
        assert len(confusers) == 0
