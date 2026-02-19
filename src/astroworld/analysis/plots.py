"""
Matplotlib visualization for orbital simulation validation.

Includes:
  - Basic orbit and error plots (Phase 2)
  - Residual analysis dashboard with RTN + FFT (Phase 4)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from astroworld.constants import BODIES


def plot_orbits_2d(
    trajectories: Dict[str, pd.DataFrame],
    plane: str = "xy",
    title: str = "Orbital Trajectories",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 2D projection of orbital trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax1, ax2 = plane[0], plane[1]

    for name, df in trajectories.items():
        color = BODIES[name].color if name in BODIES else None
        ax.plot(df[ax1], df[ax2], color=color, label=name.capitalize(), linewidth=0.8)

    # Sun at origin
    ax.plot(0, 0, "o", color=BODIES["sun"].color, markersize=12, label="Sun")

    ax.set_xlabel(f"{ax1.upper()} (AU)")
    ax.set_ylabel(f"{ax2.upper()} (AU)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_orbit_comparison(
    sim_trajectory: pd.DataFrame,
    eph_trajectory: pd.DataFrame,
    body_name: str,
    plane: str = "xy",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlay simulated orbit on JPL Horizons ephemeris."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax1, ax2 = plane[0], plane[1]

    ax.plot(
        eph_trajectory[ax1],
        eph_trajectory[ax2],
        "b-",
        label="JPL Horizons",
        linewidth=1.0,
        alpha=0.7,
    )
    ax.plot(
        sim_trajectory[ax1],
        sim_trajectory[ax2],
        "r--",
        label="Simulation",
        linewidth=1.0,
        alpha=0.7,
    )

    ax.plot(0, 0, "o", color=BODIES["sun"].color, markersize=12)
    ax.set_xlabel(f"{ax1.upper()} (AU)")
    ax.set_ylabel(f"{ax2.upper()} (AU)")
    ax.set_title(f"{body_name.capitalize()}: Simulation vs Ephemeris")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_position_error(
    errors: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot position error over time for each body."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for name, err_df in errors.items():
        color = BODIES[name].color if name in BODIES else None
        ax.plot(
            err_df["time_days"] / 365.25,
            err_df["error_km"],
            color=color,
            label=name.capitalize(),
            linewidth=1.0,
        )

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Position Error (km)")
    ax.set_title("Simulation Position Error vs JPL Horizons")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_energy_conservation(
    times: np.ndarray,
    energies: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot relative energy drift: (E - E0) / |E0| over time."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    e0 = energies[0]
    relative_drift = (energies - e0) / abs(e0)

    ax.plot(times / 365.25, relative_drift, "b-", linewidth=0.8)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Relative Energy Drift (E - E0) / |E0|")
    ax.set_title("Energy Conservation")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Phase 4: Residual Analysis Dashboard
# ---------------------------------------------------------------------------

def plot_residual_dashboard(
    rtn_df: pd.DataFrame,
    spectrum: Dict,
    secular_periodic: Dict,
    body_name: str,
    component: str = "transverse",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Three-panel residual analysis dashboard for a single body.

    Top:    Total residual |dr| over time (with secular trend line)
    Middle: RTN decomposition (Radial, Transverse, Normal) over time
    Bottom: Frequency spectrum (FFT) with annotated dominant peaks

    Parameters
    ----------
    rtn_df : DataFrame from decompose_rtn (time_days, radial_km, etc.)
    spectrum : Dict from compute_frequency_spectrum
    secular_periodic : Dict from separate_secular_periodic
    body_name : Planet name for titles
    component : Which RTN component was used for FFT ("radial" or "transverse")
    save_path : Optional path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[1, 1.2, 1])
    fig.suptitle(
        f"Residual Analysis: {body_name.capitalize()}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    t_years = rtn_df["time_days"].values / 365.25

    # --- Panel 1: Total residual + secular trend ---
    ax1 = axes[0]
    ax1.plot(t_years, rtn_df["total_km"], color="#2E86C1", linewidth=0.8, alpha=0.8,
             label="Total |$\\Delta r$|")
    ax1.plot(t_years, np.abs(secular_periodic["secular"]), color="#E74C3C",
             linewidth=2, linestyle="--",
             label=f"Secular trend ({secular_periodic['secular_rate_km_per_year']:.1f} km/yr)")
    ax1.set_ylabel("Position Error (km)")
    ax1.set_title("Total Residual Magnitude")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: RTN components ---
    ax2 = axes[1]
    ax2.plot(t_years, rtn_df["radial_km"], color="#E74C3C", linewidth=0.7,
             alpha=0.8, label="Radial (R)")
    ax2.plot(t_years, rtn_df["transverse_km"], color="#27AE60", linewidth=0.7,
             alpha=0.8, label="Transverse (T)")
    ax2.plot(t_years, rtn_df["normal_km"], color="#8E44AD", linewidth=0.7,
             alpha=0.8, label="Normal (N)")
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.set_ylabel("Error (km)")
    ax2.set_title("RTN Decomposition")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Frequency spectrum ---
    ax3 = axes[2]
    periods = spectrum["periods_days"]
    power = spectrum["power"]

    # Normalize power for display
    power_norm = power / np.max(power) if np.max(power) > 0 else power

    ax3.plot(periods, power_norm, color="#2C3E50", linewidth=0.8)
    ax3.set_xscale("log")
    ax3.set_xlabel("Period (days)")
    ax3.set_ylabel("Normalized Power")
    ax3.set_title(f"Frequency Spectrum ({component.capitalize()} residual)")
    ax3.grid(True, alpha=0.3, which="both")

    # Set reasonable x-limits
    duration_days = rtn_df["time_days"].iloc[-1] - rtn_df["time_days"].iloc[0]
    dt = rtn_df["time_days"].iloc[1] - rtn_df["time_days"].iloc[0]
    ax3.set_xlim(2 * dt, duration_days / 2)

    # Annotate peaks
    colors_peak = ["#E74C3C", "#E67E22", "#F1C40F", "#27AE60", "#2980B9"]
    for i, (period, pwr) in enumerate(spectrum["peaks"]):
        pwr_norm = pwr / np.max(power) if np.max(power) > 0 else pwr
        color = colors_peak[i % len(colors_peak)]
        ax3.axvline(x=period, color=color, linestyle="--", alpha=0.7, linewidth=1)
        ax3.annotate(
            f"{period:.1f} d",
            xy=(period, pwr_norm),
            xytext=(10, 10 + i * 15),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_spectral_comparison(
    spectra: Dict[str, Dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Overlay frequency spectra for all bodies in a single plot.

    Parameters
    ----------
    spectra : Dict mapping body_name -> spectrum dict from compute_frequency_spectrum
    save_path : Optional save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    for name, spectrum in spectra.items():
        periods = spectrum["periods_days"]
        power = spectrum["power"]
        power_norm = power / np.max(power) if np.max(power) > 0 else power
        color = BODIES[name].color if name in BODIES else None
        ax.plot(periods, power_norm, color=color, label=name.capitalize(),
                linewidth=1.0, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Normalized Power")
    ax.set_title("Frequency Spectra of Transverse Residuals")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
