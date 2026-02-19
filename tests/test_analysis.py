"""Tests for Phase 4 residual analysis functions."""

import numpy as np
import pandas as pd

from astroworld.analysis.compare import (
    decompose_rtn,
    compute_frequency_spectrum,
    separate_secular_periodic,
)


def _make_circular_orbit(period_days=365.25, n_points=1000, radius=1.0):
    """Generate a simple circular orbit for testing."""
    t = np.linspace(0, period_days * 2, n_points)
    omega = 2 * np.pi / period_days
    x = radius * np.cos(omega * t)
    y = radius * np.sin(omega * t)
    z = np.zeros_like(t)
    vx = -radius * omega * np.sin(omega * t)
    vy = radius * omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return t, x, y, z, vx, vy, vz


def test_rtn_radial_error():
    """A purely radial offset should appear only in the R component."""
    t, x, y, z, vx, vy, vz = _make_circular_orbit()
    jd = t + 2451545.0

    eph_df = pd.DataFrame({
        "datetime_jd": jd, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
    })

    # Simulation has the orbit shifted radially outward by 0.001 AU
    r = np.sqrt(x**2 + y**2)
    scale = 1 + 0.001 / r
    sim_df = pd.DataFrame({
        "time_days": t, "x": x * scale, "y": y * scale, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
    })

    rtn = decompose_rtn(sim_df, eph_df)

    # The radial component should dominate
    assert np.abs(rtn["radial_km"]).mean() > 100  # ~150 km for 0.001 AU
    # Transverse and normal should be near zero
    assert np.abs(rtn["transverse_km"]).mean() < 1
    assert np.abs(rtn["normal_km"]).mean() < 1


def test_rtn_normal_error():
    """An out-of-plane offset should appear only in the N component."""
    t, x, y, z, vx, vy, vz = _make_circular_orbit()
    jd = t + 2451545.0

    eph_df = pd.DataFrame({
        "datetime_jd": jd, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
    })

    # Simulation is shifted 0.001 AU in Z (out of plane)
    sim_df = pd.DataFrame({
        "time_days": t, "x": x, "y": y, "z": z + 0.001,
        "vx": vx, "vy": vy, "vz": vz,
    })

    rtn = decompose_rtn(sim_df, eph_df)

    # Normal component should dominate
    assert np.abs(rtn["normal_km"]).mean() > 100
    # Radial and transverse should be near zero
    assert np.abs(rtn["radial_km"]).mean() < 1
    assert np.abs(rtn["transverse_km"]).mean() < 1


def test_fft_detects_known_period():
    """FFT should find a known periodic signal injected into the data."""
    n = 1000
    t = np.linspace(0, 365.25 * 2, n)
    known_period = 27.3  # days (like the Moon!)
    signal = 500 * np.sin(2 * np.pi * t / known_period)

    spectrum = compute_frequency_spectrum(t, signal, n_peaks=3)

    # The dominant peak should be near 27.3 days
    assert len(spectrum["peaks"]) >= 1
    best_period = spectrum["peaks"][0][0]
    assert abs(best_period - known_period) < 1.0, (
        f"Expected ~{known_period} days, got {best_period:.1f} days"
    )


def test_fft_multiple_periods():
    """FFT should find two distinct periodicities."""
    n = 2000
    t = np.linspace(0, 365.25 * 4, n)
    period_1 = 27.3  # Moon-like
    period_2 = 88.0  # Mercury-like
    signal = (
        300 * np.sin(2 * np.pi * t / period_1)
        + 200 * np.sin(2 * np.pi * t / period_2)
    )

    spectrum = compute_frequency_spectrum(t, signal, n_peaks=5)
    peak_periods = sorted([p for p, _ in spectrum["peaks"]])

    # Both periods should appear in the peaks
    found_27 = any(abs(p - period_1) < 2.0 for p in peak_periods)
    found_88 = any(abs(p - period_2) < 3.0 for p in peak_periods)
    assert found_27, f"Did not find {period_1}d period in peaks: {peak_periods}"
    assert found_88, f"Did not find {period_2}d period in peaks: {peak_periods}"


def test_secular_periodic_separation():
    """Should separate a linear trend from an oscillation."""
    t = np.linspace(0, 365.25 * 2, 1000)
    drift_rate = 100  # km/year -> ~0.274 km/day
    periodic_amp = 50
    period = 30.0

    signal = (drift_rate / 365.25) * t + periodic_amp * np.sin(2 * np.pi * t / period)

    result = separate_secular_periodic(t, signal, poly_order=1)

    # The secular rate should be ~100 km/year
    assert abs(result["secular_rate_km_per_year"] - drift_rate) < 5

    # The periodic part should have amplitude ~50
    assert abs(np.max(result["periodic"]) - periodic_amp) < 10


def test_secular_quadratic():
    """Quadratic secular fit should capture accelerating drift."""
    t = np.linspace(0, 365.25, 500)
    # Signal: 0.01 * t^2 (accelerating)
    signal = 0.01 * t**2

    result = separate_secular_periodic(t, signal, poly_order=2)

    # Residual (periodic part) should be near zero
    assert np.max(np.abs(result["periodic"])) < 1
    # Quadratic coefficient should be ~0.01
    assert abs(result["poly_coeffs"][0] - 0.01) < 0.001
