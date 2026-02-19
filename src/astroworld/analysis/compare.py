"""
Comparison of simulated trajectories against JPL Horizons ephemeris.

Includes:
  - Position error computation (XYZ and RTN decomposition)
  - Spectral analysis (FFT) of residual time series
  - Secular vs periodic signal separation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from astroworld.constants import AU


def compute_position_error(
    sim_df: pd.DataFrame,
    eph_df: pd.DataFrame,
    sim_epoch_jd: float = 2451545.0,
) -> pd.DataFrame:
    """
    Compute position error between simulation and ephemeris.

    Parameters
    ----------
    sim_df : DataFrame with columns: time_days, x, y, z
    eph_df : DataFrame with columns: datetime_jd, x, y, z
    sim_epoch_jd : Julian Date of simulation t=0 (default: J2000.0)

    Returns DataFrame with columns: time_days, error_au, error_km,
        error_x, error_y, error_z
    """
    # Convert ephemeris JD to days from simulation epoch
    eph_time = (eph_df["datetime_jd"] - sim_epoch_jd).values

    # Only compare where both datasets overlap
    sim_t_min = sim_df["time_days"].iloc[0]
    sim_t_max = sim_df["time_days"].iloc[-1]
    mask = (eph_time >= sim_t_min) & (eph_time <= sim_t_max)
    eph_time = eph_time[mask]

    # Interpolate simulation onto ephemeris time points
    sim_x = np.interp(eph_time, sim_df["time_days"], sim_df["x"])
    sim_y = np.interp(eph_time, sim_df["time_days"], sim_df["y"])
    sim_z = np.interp(eph_time, sim_df["time_days"], sim_df["z"])

    eph_x = eph_df["x"].values[mask]
    eph_y = eph_df["y"].values[mask]
    eph_z = eph_df["z"].values[mask]

    dx = eph_x - sim_x
    dy = eph_y - sim_y
    dz = eph_z - sim_z

    error_au = np.sqrt(dx**2 + dy**2 + dz**2)
    error_km = error_au * AU / 1e3  # Convert AU to km

    return pd.DataFrame(
        {
            "time_days": eph_time,
            "error_au": error_au,
            "error_km": error_km,
            "error_x": dx,
            "error_y": dy,
            "error_z": dz,
        }
    )


# ---------------------------------------------------------------------------
# Pilar 1: RTN Decomposition
# ---------------------------------------------------------------------------

def decompose_rtn(
    sim_df: pd.DataFrame,
    eph_df: pd.DataFrame,
    sim_epoch_jd: float = 2451545.0,
) -> pd.DataFrame:
    """
    Decompose position residual into Radial-Transverse-Normal (RTN) frame.

    The RTN frame is defined relative to the ephemeris (true) orbit:
      R (radial):     r_hat = r / |r|  (points away from Sun)
      N (normal):     n_hat = (r x v) / |r x v|  (orbit normal)
      T (transverse): t_hat = n_hat x r_hat  (along-track direction)

    Parameters
    ----------
    sim_df : Simulated trajectory (time_days, x, y, z, vx, vy, vz)
    eph_df : Ephemeris trajectory (datetime_jd, x, y, z, vx, vy, vz)
    sim_epoch_jd : Julian Date of simulation t=0

    Returns DataFrame with columns: time_days, radial_km, transverse_km,
        normal_km, total_km
    """
    eph_time = (eph_df["datetime_jd"] - sim_epoch_jd).values
    sim_t_min = sim_df["time_days"].iloc[0]
    sim_t_max = sim_df["time_days"].iloc[-1]
    mask = (eph_time >= sim_t_min) & (eph_time <= sim_t_max)
    eph_time = eph_time[mask]

    # Interpolate simulation onto ephemeris time grid
    sim_x = np.interp(eph_time, sim_df["time_days"], sim_df["x"])
    sim_y = np.interp(eph_time, sim_df["time_days"], sim_df["y"])
    sim_z = np.interp(eph_time, sim_df["time_days"], sim_df["z"])

    eph_x = eph_df["x"].values[mask]
    eph_y = eph_df["y"].values[mask]
    eph_z = eph_df["z"].values[mask]

    # Error vector (AU)
    dx = eph_x - sim_x
    dy = eph_y - sim_y
    dz = eph_z - sim_z
    delta = np.column_stack([dx, dy, dz])

    # Ephemeris position and velocity vectors
    r_vec = np.column_stack([eph_x, eph_y, eph_z])

    eph_vx = eph_df["vx"].values[mask]
    eph_vy = eph_df["vy"].values[mask]
    eph_vz = eph_df["vz"].values[mask]
    v_vec = np.column_stack([eph_vx, eph_vy, eph_vz])

    # Build RTN basis at each time step
    r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r_hat = r_vec / r_mag  # Radial

    h_vec = np.cross(r_vec, v_vec)  # Angular momentum
    h_mag = np.linalg.norm(h_vec, axis=1, keepdims=True)
    n_hat = h_vec / h_mag  # Normal (orbit pole)

    t_hat = np.cross(n_hat, r_hat)  # Transverse (along-track)

    # Project error onto RTN axes (dot product per row)
    radial_au = np.sum(delta * r_hat, axis=1)
    transverse_au = np.sum(delta * t_hat, axis=1)
    normal_au = np.sum(delta * n_hat, axis=1)

    km_per_au = AU / 1e3

    return pd.DataFrame({
        "time_days": eph_time,
        "radial_km": radial_au * km_per_au,
        "transverse_km": transverse_au * km_per_au,
        "normal_km": normal_au * km_per_au,
        "total_km": np.sqrt(dx**2 + dy**2 + dz**2) * km_per_au,
    })


# ---------------------------------------------------------------------------
# Pilar 2: Spectral Analysis (FFT)
# ---------------------------------------------------------------------------

def compute_frequency_spectrum(
    time_days: np.ndarray,
    signal: np.ndarray,
    n_peaks: int = 5,
) -> Dict:
    """
    Compute frequency spectrum of an evenly-sampled time series using FFT.

    Parameters
    ----------
    time_days : Time array in days (must be evenly spaced).
    signal : Signal values (e.g. transverse residual in km).
    n_peaks : Number of dominant peaks to extract.

    Returns dict with keys:
        frequencies : array of frequencies (1/days)
        periods_days : array of periods (days)
        power : array of power spectral density
        peaks : list of (period_days, power) for top n_peaks
    """
    n = len(signal)
    dt = time_days[1] - time_days[0]

    # Remove linear trend (secular drift) before FFT
    coeffs = np.polyfit(time_days, signal, 1)
    detrended = signal - np.polyval(coeffs, time_days)

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    windowed = detrended * window

    # FFT (real-valued input)
    fft_vals = np.fft.rfft(windowed)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)

    # Skip DC component (index 0)
    freqs = freqs[1:]
    power = power[1:]
    periods = 1.0 / freqs

    # Find dominant peaks: local maxima in power spectrum
    peaks = []
    for i in range(1, len(power) - 1):
        if power[i] > power[i - 1] and power[i] > power[i + 1]:
            peaks.append((periods[i], power[i]))

    # Sort by power descending, take top n_peaks
    peaks.sort(key=lambda x: x[1], reverse=True)
    peaks = peaks[:n_peaks]

    return {
        "frequencies": freqs,
        "periods_days": periods,
        "power": power,
        "peaks": peaks,
    }


# ---------------------------------------------------------------------------
# Pilar 3: Secular vs Periodic Separation
# ---------------------------------------------------------------------------

def separate_secular_periodic(
    time_days: np.ndarray,
    signal: np.ndarray,
    poly_order: int = 2,
) -> Dict:
    """
    Separate a residual signal into secular (polynomial) and periodic parts.

    Parameters
    ----------
    time_days : Time array in days.
    signal : Signal values (e.g. radial residual in km).
    poly_order : Polynomial order for secular fit (1=linear, 2=quadratic).

    Returns dict with keys:
        secular : polynomial fit values (same length as signal)
        periodic : signal minus secular fit
        poly_coeffs : polynomial coefficients (highest order first)
        secular_rate_km_per_year : linear drift rate in km/year
    """
    coeffs = np.polyfit(time_days, signal, poly_order)
    secular = np.polyval(coeffs, time_days)
    periodic = signal - secular

    # Linear drift rate: coefficient of t (second-to-last in polyfit output)
    # For poly_order=2: coeffs = [a, b, c] -> signal ~ a*t^2 + b*t + c
    # The instantaneous rate at t=0 is b; average rate over span is from linear fit
    linear_coeffs = np.polyfit(time_days, signal, 1)
    rate_km_per_day = linear_coeffs[0]
    rate_km_per_year = rate_km_per_day * 365.25

    return {
        "secular": secular,
        "periodic": periodic,
        "poly_coeffs": coeffs,
        "secular_rate_km_per_year": rate_km_per_year,
    }


def compute_orbital_period(trajectory_df: pd.DataFrame) -> float:
    """
    Estimate orbital period using polar angle accumulation.

    Tracks the cumulative angle traversed and detects when
    a full 2*pi revolution is completed. Robust regardless
    of initial orbital position.

    Returns period in days.
    """
    x = trajectory_df["x"].values
    y = trajectory_df["y"].values
    t = trajectory_df["time_days"].values

    # Compute polar angle and unwrap to make it continuous
    theta = np.unwrap(np.arctan2(y, x))

    # Find when cumulative rotation crosses multiples of 2*pi
    theta_from_start = theta - theta[0]
    crossings = []
    revolution = 1
    for i in range(1, len(theta_from_start)):
        target = revolution * 2 * np.pi
        if theta_from_start[i - 1] < target <= theta_from_start[i]:
            frac = (target - theta_from_start[i - 1]) / (
                theta_from_start[i] - theta_from_start[i - 1]
            )
            t_cross = t[i - 1] + frac * (t[i] - t[i - 1])
            crossings.append(t_cross)
            revolution += 1

    if len(crossings) < 1:
        return float("nan")

    if len(crossings) == 1:
        return crossings[0] - t[0]

    periods = np.diff(crossings)
    return float(np.mean(periods))


def summary_statistics(errors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate summary statistics for position errors across bodies.

    Returns DataFrame with rows=bodies, columns=mean/max/rms/final error in km.
    """
    rows = []
    for name, err_df in errors.items():
        e = err_df["error_km"].values
        rows.append(
            {
                "body": name,
                "mean_error_km": np.mean(e),
                "max_error_km": np.max(e),
                "rms_error_km": np.sqrt(np.mean(e**2)),
                "final_error_km": e[-1],
            }
        )
    return pd.DataFrame(rows).set_index("body")
