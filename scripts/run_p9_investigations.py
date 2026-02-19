"""
Planet 9 investigation experiments.

Five doctoral-level experiments to probe the Planet 9 hypothesis
using Astroworld's JIT-accelerated N-body engine:

  Experiment 1: Debris disk vs single Planet 9 (FFT signature analysis)
  Experiment 2: Global consensus grid search (7-TNO angular dispersion heatmap)
  Experiment 3: 100 Myr dynamical stability test
  Experiment 4: Grand Tour — best-fit P9 validation (petal diagram)
  Experiment 5: Phase scan — pinpointing P9's orbital location

Usage:
    uv run python scripts/run_p9_investigations.py --experiment disk_vs_planet
    uv run python scripts/run_p9_investigations.py --experiment grid_search
    uv run python scripts/run_p9_investigations.py --experiment stability_100myr
    uv run python scripts/run_p9_investigations.py --experiment grand_tour
    uv run python scripts/run_p9_investigations.py --experiment phase_scan
    uv run python scripts/run_p9_investigations.py --experiment all

    # Quick mode (shorter durations, fewer particles):
    uv run python scripts/run_p9_investigations.py --experiment phase_scan --quick
"""

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astroworld.engine.simulation import NBodySimulation, SimulationResult
from astroworld.data.state import SystemState
from astroworld.analysis.compare import compute_frequency_spectrum
from astroworld.analysis.orbital_elements import compute_orbital_elements


# ═══════════════════════════════════════════════════════════════════
#  Experiment 1: Debris Disk vs Planet 9 (FFT Signature Analysis)
# ═══════════════════════════════════════════════════════════════════

def _extract_radial_distance(
    result: SimulationResult,
    body_name: str,
    star_index: int = 0,
) -> tuple:
    """Extract heliocentric radial distance time series for a body.

    Returns (times_days, distances_au).
    """
    idx = result.body_names.index(body_name)
    dr = result.positions[:, idx, :] - result.positions[:, star_index, :]
    dist = np.linalg.norm(dr, axis=1)
    return result.times, dist


def _plot_disk_vs_planet_fft(spectra: dict, output_dir: Path) -> None:
    """3-panel FFT comparison: no-P9 vs P9 vs disk."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    titles = {
        "no_p9": "Baseline: No Planet 9",
        "with_p9": "Hypothesis: Single Planet 9 (6.2 $M_\\oplus$, 380 AU)",
        "debris_disk": "Alternative: Distributed Debris Disk (same total mass)",
    }
    colors = {"no_p9": "#3498DB", "with_p9": "#00FF88", "debris_disk": "#E67E22"}

    for ax, (label, spectrum) in zip(axes, spectra.items()):
        periods_yr = spectrum["periods_days"] / 365.25
        power = spectrum["power"]
        max_power = np.max(power) if np.max(power) > 0 else 1
        power_norm = power / max_power

        ax.plot(periods_yr, power_norm, color=colors[label], linewidth=0.8)
        ax.set_ylabel("Normalized Power")
        ax.set_title(titles[label], fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both")

        # Annotate top 5 peaks
        for i, (period_d, pwr) in enumerate(spectrum["peaks"][:5]):
            period_yr = period_d / 365.25
            pwr_norm = pwr / max_power
            if pwr_norm > 0.05:  # Only annotate visible peaks
                ax.axvline(x=period_yr, color="red", linestyle="--", alpha=0.4)
                ax.annotate(
                    f"{period_yr:.0f} yr", xy=(period_yr, pwr_norm),
                    fontsize=8, color="red",
                    xytext=(5, 5), textcoords="offset points",
                )

    axes[-1].set_xlabel("Period (years)", fontsize=11)
    fig.suptitle(
        "Experiment 1: Debris Disk vs Planet 9\n"
        "FFT of Sedna's Heliocentric Radial Distance",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / "exp1_disk_vs_planet_fft.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def run_disk_vs_planet(
    output_dir: Path,
    duration_years: float = 10_000,
    dt: float = 10.0,
    n_disk: int = 100,
    record_interval: int = 10,
) -> None:
    """
    Experiment 1: Compare FFT signatures of Sedna's radial distance
    under three scenarios: no P9, single P9, distributed disk.

    Scientific question: Does a debris disk produce the same resonant
    fingerprint as a single massive perturber?
    """
    from astroworld.catalogs.outer_solar_system import (
        make_outer_system_without_p9,
        make_outer_system_with_p9,
        make_debris_disk,
    )

    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: DEBRIS DISK vs PLANET 9")
    print("  FFT Signature Analysis of Sedna's Radial Distance")
    print("=" * 60)

    duration_days = duration_years * 365.25
    scenarios = {
        "no_p9": ("No P9 (baseline)", make_outer_system_without_p9()),
        "with_p9": ("With P9 (6.2 M_Earth)", make_outer_system_with_p9()),
        "debris_disk": (f"Debris Disk ({n_disk} particles)", make_debris_disk(n_particles=n_disk)),
    }

    spectra = {}
    for label, (desc, state) in scenarios.items():
        print(f"\n  --- {desc} ({state.n} bodies) ---")
        t0 = time.perf_counter()
        sim = NBodySimulation(relativistic=False)
        sim.set_initial_state(state)
        result = sim.run(
            duration_days=duration_days, dt=dt,
            integrator="verlet", record_interval=record_interval,
        )
        elapsed = time.perf_counter() - t0
        print(f"    {len(result.times)} records, {elapsed:.1f}s")

        # Energy check
        e0 = result.energies[0]
        drift = abs((result.energies[-1] - e0) / abs(e0))
        print(f"    Energy drift: {drift:.2e}")

        # Extract Sedna's radial distance
        times, dist = _extract_radial_distance(result, "sedna")
        print(f"    Sedna distance: {dist.min():.1f} - {dist.max():.1f} AU")

        # FFT on radial distance
        spectrum = compute_frequency_spectrum(times, dist, n_peaks=10)
        spectra[label] = spectrum

        # Report top peaks
        if spectrum["peaks"]:
            print(f"    Top FFT peaks:")
            for period_d, pwr in spectrum["peaks"][:3]:
                print(f"      Period = {period_d/365.25:.0f} years, power = {pwr:.2e}")

    # Generate comparison figure
    print("\n  Generating FFT comparison figure...")
    _plot_disk_vs_planet_fft(spectra, output_dir)

    print("\n  Experiment 1 complete!")


# ═══════════════════════════════════════════════════════════════════
#  Experiment 2: Global Consensus Grid Search (Mean Angular Spread)
# ═══════════════════════════════════════════════════════════════════

# Names of all 7 eTNO witnesses
_TNO_WITNESSES = [
    "sedna", "2012_vp113", "leleakuhonua",
    "2004_vn112", "2007_tg422", "2013_rf98", "2010_gb174",
]


def _circular_std_deg(angles_deg: np.ndarray) -> float:
    """Compute circular standard deviation of angles in degrees.

    Uses the mean resultant length R:
        R = |mean(exp(i*θ))|
        σ_circ = sqrt(-2 * ln(R))  in radians, then convert to degrees.

    For perfectly clustered angles σ→0; for uniform σ→large.
    """
    theta = np.radians(angles_deg)
    mean_cos = np.mean(np.cos(theta))
    mean_sin = np.mean(np.sin(theta))
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    if R > 1 - 1e-12:
        return 0.0  # perfectly clustered
    return np.degrees(np.sqrt(-2 * np.log(R)))


def _plot_grid_search_heatmap(
    masses: list, distances: list,
    dispersion: np.ndarray, output_dir: Path,
    duration_years: float,
) -> None:
    """Mass vs distance heatmap of TNO ω_peri angular dispersion."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use reversed colormap: low dispersion = hot = good clustering
    im = ax.imshow(
        dispersion, cmap="inferno_r", aspect="auto", origin="lower",
        extent=[
            distances[0] - (distances[1] - distances[0]) / 2,
            distances[-1] + (distances[1] - distances[0]) / 2,
            masses[0] - (masses[1] - masses[0]) / 2,
            masses[-1] + (masses[1] - masses[0]) / 2,
        ],
    )

    # Overlay values
    vmin, vmax = np.min(dispersion), np.max(dispersion)
    for i in range(len(masses)):
        for j in range(len(distances)):
            val = dispersion[i, j]
            # Dark text on bright cells, white text on dark cells
            frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = "white" if frac > 0.5 else "black"
            ax.text(
                distances[j], masses[i], f"{val:.1f}°",
                ha="center", va="center", color=text_color,
                fontsize=9, fontweight="bold",
            )

    # Mark minimum dispersion (best clustering)
    min_idx = np.unravel_index(np.argmin(dispersion), dispersion.shape)
    best_mass = masses[min_idx[0]]
    best_dist = distances[min_idx[1]]
    best_val = dispersion[min_idx]
    ax.plot(
        best_dist, best_mass, "c*", markersize=24, markeredgecolor="white",
        markeredgewidth=2, label=f"Min dispersion: {best_val:.1f}° ({best_mass} $M_\\oplus$, {best_dist} AU)",
        zorder=10,
    )

    # Mark Batygin-Brown nominal
    ax.plot(
        380, 6.2, "r*", markersize=18, markeredgecolor="white",
        markeredgewidth=1.5, label="Batygin-Brown (6.2 $M_\\oplus$, 380 AU)",
        zorder=10,
    )

    ax.set_xlabel("Planet 9 Distance (AU)", fontsize=12)
    ax.set_ylabel("Planet 9 Mass ($M_\\oplus$)", fontsize=12)
    ax.set_title(
        "Experiment 2: Global Consensus — TNO $\\omega_{peri}$ Angular Dispersion\n"
        f"7 eTNO witnesses, {int(duration_years):,}-year N-body integration (Velocity Verlet, JIT)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(distances)
    ax.set_yticks(masses)
    ax.legend(loc="upper right", fontsize=10)

    fig.colorbar(
        im, ax=ax,
        label="Circular std of $\\omega_{peri}$ across 7 TNOs (degrees) — lower = more clustered",
        shrink=0.85,
    )

    save_path = output_dir / "exp2_grid_search_dispersion.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def run_grid_search(
    output_dir: Path,
    duration_years: float = 100_000,
    dt: float = 10.0,
    record_interval: int = 100,
) -> None:
    """
    Experiment 2: Global Consensus Grid Search.

    For each (mass, distance) combination, simulates the outer Solar System
    with a custom Planet 9 and measures the Mean Angular Spread of ω_peri
    across ALL 7 eTNO witnesses at the end of the integration.

    Metric: circular standard deviation of the final ω_peri values.
    Lower dispersion = stronger clustering = better P9 candidate.
    """
    from astroworld.catalogs.outer_solar_system import (
        make_outer_system_custom_p9, GM_STAR,
    )

    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: GLOBAL CONSENSUS GRID SEARCH")
    print(f"  Metric: Circular std(ω_peri) across 7 eTNO witnesses")
    print(f"  Duration: {int(duration_years):,} years per run")
    print("=" * 60)

    masses = [2, 4, 6, 8, 10]           # M_Earth
    distances = [300, 400, 500, 600, 700]  # AU
    duration_days = duration_years * 365.25

    # Results matrix: angular dispersion (lower = better clustering)
    dispersion = np.zeros((len(masses), len(distances)))

    total = len(masses) * len(distances)
    run_count = 0
    t_total_start = time.perf_counter()

    for i, mass in enumerate(masses):
        for j, dist in enumerate(distances):
            run_count += 1
            print(f"\n  [{run_count:2d}/{total}] mass={mass} M_Earth, dist={dist} AU")

            state = make_outer_system_custom_p9(
                mass_earth=mass, distance_au=dist,
            )
            sim = NBodySimulation(relativistic=False)
            sim.set_initial_state(state)

            t0 = time.perf_counter()
            result = sim.run(
                duration_days=duration_days, dt=dt,
                integrator="verlet", record_interval=record_interval,
            )
            elapsed = time.perf_counter() - t0

            # Extract final ω_peri for all TNO witnesses
            star_idx = 0
            omega_finals = []
            for tno_name in _TNO_WITNESSES:
                if tno_name not in result.body_names:
                    continue
                tno_idx = result.body_names.index(tno_name)
                pos_final = result.positions[-1, tno_idx, :] - result.positions[-1, star_idx, :]
                vel_final = result.velocities[-1, tno_idx, :] - result.velocities[-1, star_idx, :]
                elems_final = compute_orbital_elements(pos_final, vel_final, GM_STAR)
                omega_finals.append(elems_final["omega_peri"])

            omega_arr = np.array(omega_finals)
            circ_std = _circular_std_deg(omega_arr)
            dispersion[i, j] = circ_std

            # Report individual TNO values
            tno_str = ", ".join(f"{w:.0f}°" for w in omega_arr)
            print(f"    ω_peri final: [{tno_str}]")
            print(f"    Circular std = {circ_std:.2f}° ({elapsed:.1f}s)")

    t_total = time.perf_counter() - t_total_start
    print(f"\n  Grid search complete: {total} runs in {t_total:.0f}s")

    # Print summary table
    print(f"\n  Angular Dispersion (circular std of ω_peri, degrees):")
    print(f"  {'':>8}", end="")
    for d in distances:
        print(f" {d:>7} AU", end="")
    print()
    print(f"  {'':>8} {'-'*len(distances)*10}")
    for i, m in enumerate(masses):
        print(f"  {m:>5} ME |", end="")
        for j in range(len(distances)):
            print(f" {dispersion[i,j]:>8.2f}", end="")
        print()

    # Identify minimum
    min_idx = np.unravel_index(np.argmin(dispersion), dispersion.shape)
    best_mass = masses[min_idx[0]]
    best_dist = distances[min_idx[1]]
    best_disp = dispersion[min_idx]
    print(f"\n  ★ MINIMUM DISPERSION: {best_disp:.2f}° at mass={best_mass} M⊕, dist={best_dist} AU")

    # Generate heatmap
    print("\n  Generating dispersion heatmap...")
    _plot_grid_search_heatmap(masses, distances, dispersion, output_dir, duration_years)

    print("\n  Experiment 2 complete!")


# ═══════════════════════════════════════════════════════════════════
#  Experiment 3: 100 Million Year Stability Test
# ═══════════════════════════════════════════════════════════════════

def _plot_stability_eccentricity(
    ecc_data: dict, energy_drift: float, output_dir: Path,
    duration_myr: float,
) -> None:
    """Eccentricity evolution over geological timescales."""
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        "sedna": "#E74C3C", "2012_vp113": "#E67E22",
        "leleakuhonua": "#9B59B6", "2004_vn112": "#1ABC9C",
        "2007_tg422": "#F39C12", "2013_rf98": "#8E44AD",
        "2010_gb174": "#2ECC71", "planet_9": "#00FF88",
    }

    for body_name, data in ecc_data.items():
        ax.plot(
            data["t_myr"], data["ecc"],
            color=colors.get(body_name, "white"),
            linewidth=0.6, alpha=0.9, label=body_name,
        )

    # Ejection threshold
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2,
               alpha=0.7, label="Ejection threshold (e=1)")

    ax.set_xlabel("Time (Myr)", fontsize=12)
    ax.set_ylabel("Eccentricity", fontsize=12)
    ax.set_title(
        f"Experiment 3: {int(duration_myr)} Myr Stability Test — Eccentricity Evolution\n"
        f"Energy drift: {energy_drift:.2e}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    max_ecc = max(d["ecc"].max() for d in ecc_data.values())
    ax.set_ylim(0, max(1.1, max_ecc * 1.1))

    save_path = output_dir / "exp3_stability_eccentricity.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_stability_energy(result: SimulationResult, output_dir: Path) -> None:
    """Energy conservation over geological timescales."""
    fig, ax = plt.subplots(figsize=(14, 5))
    e0 = result.energies[0]
    relative = (result.energies - e0) / abs(e0)
    t_myr = result.times / (365.25 * 1e6)

    ax.plot(t_myr, relative, "b-", linewidth=0.5)
    ax.set_xlabel("Time (Myr)", fontsize=11)
    ax.set_ylabel("Relative energy drift $(E - E_0) / |E_0|$", fontsize=11)
    ax.set_title("Energy Conservation over Geological Time", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    save_path = output_dir / "exp3_stability_energy.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def run_stability_100myr(
    output_dir: Path,
    duration_myr: float = 100.0,
    dt: float = 100.0,
    record_interval: int = 1000,
) -> None:
    """
    Experiment 3: Long-term dynamical stability test with Planet 9.

    Checks whether the P9 hypothesis is dynamically stable on geological
    timescales. Tracks eccentricity evolution of all TNOs and Planet 9.

    Parameters
    ----------
    duration_myr : Duration in millions of years (default 100).
    dt : Time step in days (100 gives 43 steps per Jupiter orbit).
    record_interval : Record every N steps (1000 keeps memory manageable).
    """
    from astroworld.catalogs.outer_solar_system import (
        make_outer_system_with_p9, GM_STAR,
    )

    duration_days = duration_myr * 1e6 * 365.25
    n_steps = int(duration_days / dt)
    n_records = n_steps // record_interval + 1

    print("\n" + "=" * 60)
    print(f"  EXPERIMENT 3: {int(duration_myr)} MILLION YEAR STABILITY TEST")
    print("=" * 60)
    print(f"  Duration: {duration_myr} Myr = {duration_days:.2e} days")
    print(f"  dt = {dt} days, n_steps = {n_steps:,}")
    print(f"  record_interval = {record_interval}, est. records = {n_records:,}")
    print(f"  Est. memory: ~{n_records * 13 * 6 * 8 / 1e6:.0f} MB")

    state = make_outer_system_with_p9()
    sim = NBodySimulation(relativistic=False)
    sim.set_initial_state(state)

    print(f"\n  Running {state.n}-body simulation...")
    t0 = time.perf_counter()
    result = sim.run(
        duration_days=duration_days, dt=dt,
        integrator="verlet", record_interval=record_interval,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.1f}s ({len(result.times):,} records)")

    # Energy conservation
    e0 = result.energies[0]
    e_final = result.energies[-1]
    drift = abs((e_final - e0) / abs(e0))
    print(f"  Energy drift: {drift:.2e}")

    # Extract eccentricity evolution for TNOs + Planet 9
    track_bodies = [
        "sedna", "2012_vp113", "leleakuhonua",
        "2004_vn112", "2007_tg422", "2013_rf98", "2010_gb174",
        "planet_9",
    ]
    star_idx = state.star_index

    ecc_data = {}
    print(f"\n  Extracting orbital elements...")
    for body_name in track_bodies:
        if body_name not in result.body_names:
            continue
        body_idx = result.body_names.index(body_name)

        # Sample for orbital element computation
        n_samples = min(5000, len(result.times))
        sample_idx = np.unique(
            np.linspace(0, len(result.times) - 1, n_samples, dtype=int)
        )
        t_myr = result.times[sample_idx] / (365.25 * 1e6)

        ecc_list = []
        for k in sample_idx:
            pos = result.positions[k, body_idx, :] - result.positions[k, star_idx, :]
            vel = result.velocities[k, body_idx, :] - result.velocities[k, star_idx, :]
            elems = compute_orbital_elements(pos, vel, GM_STAR)
            ecc_list.append(elems["e"])

        ecc_arr = np.array(ecc_list)
        ecc_data[body_name] = {"t_myr": t_myr, "ecc": ecc_arr}

        # Report
        ejected = np.any(ecc_arr >= 1.0)
        status = "*** EJECTED ***" if ejected else "STABLE"
        print(f"  {body_name:18s}: e = [{ecc_arr.min():.4f}, {ecc_arr.max():.4f}] — {status}")

    # Summary
    print(f"\n  {'='*50}")
    print(f"  STABILITY VERDICT:")
    all_stable = all(
        not np.any(d["ecc"] >= 1.0) for d in ecc_data.values()
    )
    if all_stable:
        print(f"  ✓ All bodies survived {int(duration_myr)} Myr without ejection")
        print(f"  ✓ Planet 9 hypothesis is dynamically stable")
    else:
        ejected_bodies = [
            name for name, d in ecc_data.items()
            if np.any(d["ecc"] >= 1.0)
        ]
        print(f"  ✗ Ejected bodies: {', '.join(ejected_bodies)}")
        print(f"  ✗ The proposed P9 orbit may be dynamically unstable")
    print(f"  Energy drift: {drift:.2e}")
    print(f"  {'='*50}")

    # Generate plots
    print(f"\n  Generating plots...")
    _plot_stability_eccentricity(ecc_data, drift, output_dir, duration_myr)
    _plot_stability_energy(result, output_dir)

    print("\n  Experiment 3 complete!")


# ═══════════════════════════════════════════════════════════════════
#  Experiment 4: Grand Tour — Best-Fit P9 Validation (Petal Diagram)
# ═══════════════════════════════════════════════════════════════════

def _plot_petal_diagram(
    omega_data_p9: dict, omega_data_baseline: dict,
    dispersion_p9: np.ndarray, dispersion_baseline: np.ndarray,
    t_myr_p9: np.ndarray, t_myr_baseline: np.ndarray,
    output_dir: Path,
    duration_myr: float,
    best_mass: float,
    best_dist: float,
) -> None:
    """Generate the Grand Tour 3-panel figure.

    Top-left:    ω_peri(t) for all 7 TNOs WITH best-fit P9 (petal diagram)
    Top-right:   ω_peri(t) for all 7 TNOs WITHOUT P9 (baseline)
    Bottom:      Dispersion(t) comparison — P9 vs baseline
    """
    colors = {
        "sedna": "#E74C3C", "2012_vp113": "#E67E22",
        "leleakuhonua": "#9B59B6", "2004_vn112": "#1ABC9C",
        "2007_tg422": "#F39C12", "2013_rf98": "#8E44AD",
        "2010_gb174": "#2ECC71",
    }

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.30, wspace=0.25)

    # ── Top-left: With best-fit P9 (petal diagram) ──
    ax1 = fig.add_subplot(gs[0, 0])
    for body_name, omega_arr in omega_data_p9.items():
        ax1.plot(
            t_myr_p9, omega_arr,
            color=colors.get(body_name, "gray"),
            linewidth=0.7, alpha=0.9, label=body_name,
        )
    ax1.set_xlabel("Time (Myr)", fontsize=11)
    ax1.set_ylabel("$\\omega_{peri}$ (degrees)", fontsize=11)
    ax1.set_title(
        f"With Planet 9 ({best_mass} $M_\\oplus$, {best_dist} AU)\n"
        "\"The Shepherd Keeps the Flock Together\"",
        fontsize=12, fontweight="bold", color="#00FF88",
    )
    ax1.legend(fontsize=8, ncol=2, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Top-right: Without P9 (baseline) ──
    ax2 = fig.add_subplot(gs[0, 1])
    for body_name, omega_arr in omega_data_baseline.items():
        ax2.plot(
            t_myr_baseline, omega_arr,
            color=colors.get(body_name, "gray"),
            linewidth=0.7, alpha=0.9, label=body_name,
        )
    ax2.set_xlabel("Time (Myr)", fontsize=11)
    ax2.set_ylabel("$\\omega_{peri}$ (degrees)", fontsize=11)
    ax2.set_title(
        "Without Planet 9 (Null Hypothesis)\n"
        "\"The Flock Scatters Without a Shepherd\"",
        fontsize=12, fontweight="bold", color="#E74C3C",
    )
    ax2.legend(fontsize=8, ncol=2, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── Bottom: Dispersion comparison ──
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(
        t_myr_p9, dispersion_p9,
        color="#00FF88", linewidth=1.5, alpha=0.9,
        label=f"With P9 ({best_mass} $M_\\oplus$, {best_dist} AU)",
    )
    ax3.plot(
        t_myr_baseline, dispersion_baseline,
        color="#E74C3C", linewidth=1.5, alpha=0.9,
        label="Without P9 (baseline)",
    )
    ax3.set_xlabel("Time (Myr)", fontsize=11)
    ax3.set_ylabel("Angular Dispersion σ(ω$_{peri}$) (degrees)", fontsize=11)
    ax3.set_title(
        "Clustering Evolution: Does the Shepherd Maintain Order?",
        fontsize=12, fontweight="bold",
    )
    ax3.legend(fontsize=10, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Annotate final values
    if len(dispersion_p9) > 0 and len(dispersion_baseline) > 0:
        final_p9 = dispersion_p9[-1]
        final_base = dispersion_baseline[-1]
        ax3.annotate(
            f"σ = {final_p9:.1f}°",
            xy=(t_myr_p9[-1], final_p9),
            fontsize=10, color="#00FF88", fontweight="bold",
            xytext=(-80, 10), textcoords="offset points",
        )
        ax3.annotate(
            f"σ = {final_base:.1f}°",
            xy=(t_myr_baseline[-1], final_base),
            fontsize=10, color="#E74C3C", fontweight="bold",
            xytext=(-80, -15), textcoords="offset points",
        )

    fig.suptitle(
        f"Experiment 4: Grand Tour — {int(duration_myr)} Myr Validation\n"
        f"Best-fit Planet 9 ({best_mass} $M_\\oplus$, {best_dist} AU) vs Null Hypothesis",
        fontsize=14, fontweight="bold", y=0.98,
    )

    save_path = output_dir / "exp4_grand_tour.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _generate_polar_clock_html(
    omega_data_p9: dict, omega_data_baseline: dict,
    dispersion_p9: np.ndarray, dispersion_baseline: np.ndarray,
    t_myr_p9: np.ndarray, t_myr_baseline: np.ndarray,
    output_dir: Path,
    duration_myr: float,
    best_mass: float,
    best_dist: float,
    n_frames: int = 200,
) -> None:
    """Generate an animated polar clock HTML visualization using Plotly.

    Two polar subplots side by side:
      - Left: With Planet 9 (TNOs stay clustered)
      - Right: Without Planet 9 (TNOs scatter)

    Each TNO is a colored point on the polar dial at angle = ω_peri.
    Animation shows their evolution over time with Play/Pause controls.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors = {
        "sedna": "#E74C3C", "2012_vp113": "#E67E22",
        "leleakuhonua": "#9B59B6", "2004_vn112": "#1ABC9C",
        "2007_tg422": "#F39C12", "2013_rf98": "#8E44AD",
        "2010_gb174": "#2ECC71",
    }

    tno_names = list(omega_data_p9.keys())
    n_points = len(t_myr_p9)
    frame_step = max(1, n_points // n_frames)
    frame_indices = list(range(0, n_points, frame_step))
    if frame_indices[-1] != n_points - 1:
        frame_indices.append(n_points - 1)

    # Create figure with two polar subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=[
            f"<b style='color:#00FF88'>With Planet 9 ({best_mass} M⊕, {best_dist} AU)</b>",
            "<b style='color:#E74C3C'>Without Planet 9 (Null Hypothesis)</b>",
        ],
        horizontal_spacing=0.08,
    )

    # ── Initial traces (frame 0) ──
    # Each TNO gets 2 traces: one for P9 subplot, one for baseline subplot
    # Plus trail traces showing the path
    for tno_name in tno_names:
        color = colors.get(tno_name, "#FFFFFF")
        omega_p9_0 = omega_data_p9[tno_name][0]
        omega_base_0 = omega_data_baseline[tno_name][0]

        # Short display name
        display_name = tno_name.replace("_", " ").replace("2012 vp113", "VP113").replace(
            "2004 vn112", "VN112").replace("2007 tg422", "TG422").replace(
            "2013 rf98", "RF98").replace("2010 gb174", "GB174")

        # P9 subplot - markers (all TNOs at fixed r=0.8 for visual clarity)
        fig.add_trace(
            go.Scatterpolar(
                r=[0.8], theta=[omega_p9_0], mode="markers+text",
                marker=dict(size=14, color=color, symbol="circle",
                            line=dict(width=1, color="white")),
                text=[display_name], textposition="top center",
                textfont=dict(color=color, size=9),
                name=tno_name,
                showlegend=(True),
                legendgroup=tno_name,
            ),
            row=1, col=1,
        )
        # P9 subplot - trails
        fig.add_trace(
            go.Scatterpolar(
                r=[0.8], theta=[omega_p9_0], mode="lines",
                line=dict(color=color, width=1),
                opacity=0.3,
                showlegend=False,
                legendgroup=tno_name,
            ),
            row=1, col=1,
        )

        # Baseline subplot - markers
        fig.add_trace(
            go.Scatterpolar(
                r=[0.8], theta=[omega_base_0], mode="markers+text",
                marker=dict(size=14, color=color, symbol="circle",
                            line=dict(width=1, color="white")),
                text=[display_name], textposition="top center",
                textfont=dict(color=color, size=9),
                name=tno_name,
                showlegend=False,
                legendgroup=tno_name,
            ),
            row=1, col=2,
        )
        # Baseline subplot - trails
        fig.add_trace(
            go.Scatterpolar(
                r=[0.8], theta=[omega_base_0], mode="lines",
                line=dict(color=color, width=1),
                opacity=0.3,
                showlegend=False,
                legendgroup=tno_name,
            ),
            row=1, col=2,
        )

    # ── Build animation frames ──
    frames = []
    slider_steps = []
    n_traces_per_tno = 4  # marker_p9, trail_p9, marker_base, trail_base
    n_total_traces = len(tno_names) * n_traces_per_tno

    for f_i, idx in enumerate(frame_indices):
        frame_data = []
        for t_i, tno_name in enumerate(tno_names):
            omega_p9_val = omega_data_p9[tno_name][idx]
            omega_base_val = omega_data_baseline[tno_name][idx]

            # Trail: all points up to current time
            trail_end = idx + 1
            trail_step = max(1, trail_end // 100)  # limit trail points
            trail_indices = list(range(0, trail_end, trail_step))
            trail_omegas_p9 = [omega_data_p9[tno_name][k] for k in trail_indices]
            trail_omegas_base = [omega_data_baseline[tno_name][k] for k in trail_indices]
            trail_rs = [0.8] * len(trail_indices)

            # P9 marker
            frame_data.append(go.Scatterpolar(
                r=[0.8], theta=[omega_p9_val],
            ))
            # P9 trail
            frame_data.append(go.Scatterpolar(
                r=trail_rs, theta=trail_omegas_p9,
            ))
            # Baseline marker
            frame_data.append(go.Scatterpolar(
                r=[0.8], theta=[omega_base_val],
            ))
            # Baseline trail
            frame_data.append(go.Scatterpolar(
                r=trail_rs, theta=trail_omegas_base,
            ))

        t_val = t_myr_p9[idx]
        disp_p9_val = dispersion_p9[idx]
        disp_base_val = dispersion_baseline[idx]

        frames.append(go.Frame(
            data=frame_data,
            traces=list(range(n_total_traces)),
            name=str(f_i),
        ))
        slider_steps.append(dict(
            method="animate",
            args=[[str(f_i)], dict(
                mode="immediate",
                frame=dict(duration=50, redraw=True),
                transition=dict(duration=0),
            )],
            label=f"{t_val:.2f}",
        ))

    fig.frames = frames

    # ── Layout ──
    polar_common = dict(
        bgcolor="black",
        angularaxis=dict(
            direction="clockwise",
            tickfont=dict(color="gray", size=10),
            gridcolor="rgba(60, 60, 80, 0.4)",
            linecolor="rgba(80, 80, 100, 0.5)",
        ),
        radialaxis=dict(
            visible=False,
            range=[0, 1.1],
        ),
    )

    fig.update_layout(
        polar=polar_common,
        polar2=polar_common,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        title=dict(
            text=(
                f"<b>Grand Tour: The Shepherd's Dance — {int(duration_myr)} Myr Animation</b><br>"
                f"<span style='font-size:13px; color:#aaa'>"
                f"7 eTNO witnesses on the ω<sub>peri</sub> polar clock | "
                f"P9: {best_mass} M⊕ at {best_dist} AU</span>"
            ),
            font=dict(size=16),
            x=0.5,
        ),
        legend=dict(
            font=dict(size=11, color="white"),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="rgba(100,100,100,0.5)",
            borderwidth=1,
            x=0.5, y=-0.05,
            xanchor="center",
            orientation="h",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=-0.08, x=0.05, xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=50, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        )],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        )],
                    ),
                ],
                font=dict(color="white"),
                bgcolor="rgba(40, 40, 60, 0.8)",
            ),
        ],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.05, len=0.9,
            y=-0.02,
            currentvalue=dict(
                prefix="Time: ",
                suffix=" Myr",
                font=dict(size=13, color="white"),
            ),
            font=dict(color="gray", size=9),
            bgcolor="rgba(40, 40, 60, 0.5)",
            activebgcolor="#00FF88",
            bordercolor="rgba(100,100,100,0.3)",
            ticklen=3,
        )],
        height=650,
        width=1200,
        margin=dict(t=100, b=120, l=40, r=40),
    )

    # Add annotation showing dispersion values
    fig.add_annotation(
        text=(
            f"<b>Initial σ₀ = {dispersion_p9[0]:.1f}°</b><br>"
            f"<span style='color:#00FF88'>Final σ (P9) = {dispersion_p9[-1]:.1f}°</span><br>"
            f"<span style='color:#E74C3C'>Final σ (No P9) = {dispersion_baseline[-1]:.1f}°</span>"
        ),
        xref="paper", yref="paper",
        x=0.5, y=1.01,
        showarrow=False,
        font=dict(size=12, color="white"),
        align="center",
    )

    save_path = output_dir / "exp4_polar_clock.html"
    fig.write_html(str(save_path), include_plotlyjs=True)
    print(f"  Saved: {save_path.name} ({save_path.stat().st_size / 1e6:.1f} MB)")


def _run_and_extract_omega(
    state: "SystemState",
    duration_days: float,
    dt: float,
    record_interval: int,
    gm_central: float,
    n_samples: int = 3000,
    label: str = "",
) -> tuple:
    """Run simulation and extract ω_peri time series for all TNO witnesses.

    Returns (omega_data, dispersion_timeseries, t_myr).
    """
    sim = NBodySimulation(relativistic=False)
    sim.set_initial_state(state)

    t0 = time.perf_counter()
    result = sim.run(
        duration_days=duration_days, dt=dt,
        integrator="verlet", record_interval=record_interval,
    )
    elapsed = time.perf_counter() - t0
    print(f"    {label}: {len(result.times):,} records, {elapsed:.1f}s")

    # Energy check
    e0 = result.energies[0]
    drift = abs((result.energies[-1] - e0) / abs(e0))
    print(f"    Energy drift: {drift:.2e}")

    star_idx = 0
    n_samples = min(n_samples, len(result.times))
    sample_idx = np.unique(
        np.linspace(0, len(result.times) - 1, n_samples, dtype=int)
    )
    t_myr = result.times[sample_idx] / (365.25 * 1e6)

    # Extract ω_peri for each TNO at each sample
    omega_data = {}  # body_name → array of ω_peri
    for tno_name in _TNO_WITNESSES:
        if tno_name not in result.body_names:
            continue
        tno_idx = result.body_names.index(tno_name)
        omega_list = []
        for k in sample_idx:
            pos = result.positions[k, tno_idx, :] - result.positions[k, star_idx, :]
            vel = result.velocities[k, tno_idx, :] - result.velocities[k, star_idx, :]
            elems = compute_orbital_elements(pos, vel, gm_central)
            omega_list.append(elems["omega_peri"])
        omega_data[tno_name] = np.array(omega_list)

    # Compute dispersion at each time step
    dispersion = np.zeros(len(sample_idx))
    for s_i in range(len(sample_idx)):
        omegas_at_t = np.array([omega_data[name][s_i] for name in omega_data])
        dispersion[s_i] = _circular_std_deg(omegas_at_t)

    # Report
    for tno_name, omega_arr in omega_data.items():
        delta = omega_arr[-1] - omega_arr[0]
        print(f"    {tno_name:18s}: ω₀={omega_arr[0]:.1f}° → ω_f={omega_arr[-1]:.1f}° (Δ={delta:+.2f}°)")
    print(f"    Dispersion: σ₀={dispersion[0]:.2f}° → σ_f={dispersion[-1]:.2f}°")

    return omega_data, dispersion, t_myr


def run_grand_tour(
    output_dir: Path,
    duration_myr: float = 10.0,
    dt: float = 50.0,
    record_interval: int = 500,
    best_mass: float = 10.0,
    best_dist: float = 400.0,
) -> None:
    """
    Experiment 4: Grand Tour — Validate best-fit Planet 9 parameters.

    Runs two parallel simulations (with and without best-fit Planet 9)
    over geological timescales. Generates a petal diagram showing whether
    the shepherd (Planet 9) keeps the ω_peri of all 7 eTNO witnesses
    clustered, while without P9 they scatter.

    Parameters
    ----------
    duration_myr : Duration in millions of years (default 10).
    dt : Time step in days (50 → 87 steps/Jupiter orbit, adequate for secular).
    record_interval : Record every N steps.
    best_mass : Best-fit P9 mass from grid search (M_Earth).
    best_dist : Best-fit P9 distance from grid search (AU).
    """
    from astroworld.catalogs.outer_solar_system import (
        make_outer_system_custom_p9,
        make_outer_system_without_p9,
        GM_STAR,
    )

    duration_days = duration_myr * 1e6 * 365.25
    n_steps = int(duration_days / dt)

    print("\n" + "=" * 60)
    print(f"  EXPERIMENT 4: GRAND TOUR — {int(duration_myr)} Myr VALIDATION")
    print(f"  Best-fit P9: {best_mass} M⊕ at {best_dist} AU")
    print("=" * 60)
    print(f"  Duration: {duration_myr} Myr = {duration_days:.2e} days")
    print(f"  dt = {dt} days, n_steps = {n_steps:,}")
    print(f"  record_interval = {record_interval}")

    # ── Run 1: With best-fit Planet 9 ──
    print(f"\n  ── Scenario A: WITH Planet 9 ({best_mass} M⊕, {best_dist} AU) ──")
    state_p9 = make_outer_system_custom_p9(
        mass_earth=best_mass, distance_au=best_dist,
    )
    omega_p9, disp_p9, t_myr_p9 = _run_and_extract_omega(
        state_p9, duration_days, dt, record_interval, GM_STAR,
        label=f"With P9 ({state_p9.n} bodies)",
    )

    # ── Run 2: Without Planet 9 (baseline) ──
    print(f"\n  ── Scenario B: WITHOUT Planet 9 (null hypothesis) ──")
    state_base = make_outer_system_without_p9()
    omega_base, disp_base, t_myr_base = _run_and_extract_omega(
        state_base, duration_days, dt, record_interval, GM_STAR,
        label=f"No P9 ({state_base.n} bodies)",
    )

    # ── Summary ──
    print(f"\n  {'='*55}")
    print(f"  GRAND TOUR VERDICT:")
    print(f"  With P9:    σ₀ = {disp_p9[0]:.2f}° → σ_f = {disp_p9[-1]:.2f}° (Δσ = {disp_p9[-1]-disp_p9[0]:+.2f}°)")
    print(f"  Without P9: σ₀ = {disp_base[0]:.2f}° → σ_f = {disp_base[-1]:.2f}° (Δσ = {disp_base[-1]-disp_base[0]:+.2f}°)")
    if disp_p9[-1] < disp_base[-1]:
        diff = disp_base[-1] - disp_p9[-1]
        print(f"  ✓ P9 reduces final dispersion by {diff:.2f}° — the shepherd works!")
    else:
        print(f"  ✗ P9 does not reduce dispersion at this timescale")
    print(f"  {'='*55}")

    # ── Generate figures ──
    print(f"\n  Generating Grand Tour petal diagram...")
    _plot_petal_diagram(
        omega_p9, omega_base,
        disp_p9, disp_base,
        t_myr_p9, t_myr_base,
        output_dir, duration_myr,
        best_mass, best_dist,
    )

    print(f"\n  Generating animated polar clock (HTML)...")
    _generate_polar_clock_html(
        omega_p9, omega_base,
        disp_p9, disp_base,
        t_myr_p9, t_myr_base,
        output_dir, duration_myr,
        best_mass, best_dist,
    )

    print("\n  Experiment 4 complete!")


# ═══════════════════════════════════════════════════════════════════
#  Experiment 5: Phase Scan — Pinpointing P9's Orbital Location
# ═══════════════════════════════════════════════════════════════════

def _plot_phase_sensitivity(
    phases_deg: np.ndarray,
    dispersions: np.ndarray,
    output_dir: Path,
    duration_myr: float,
    best_mass: float,
    best_dist: float,
    baseline_dispersion: float,
) -> None:
    """Phase sensitivity polar + cartesian dual plot.

    Left:  Polar plot — dispersion as radial distance at each phase angle.
    Right: Cartesian — dispersion vs phase with annotated minimum.
    """
    fig, (ax_polar, ax_cart) = plt.subplots(
        1, 2, figsize=(18, 8),
        subplot_kw={"projection": None},  # override for individual axes
    )
    # Recreate as mixed figure
    fig.delaxes(ax_polar)
    fig.delaxes(ax_cart)
    ax_polar = fig.add_subplot(121, projection="polar")
    ax_cart = fig.add_subplot(122)

    # ── Polar plot ──
    theta = np.radians(phases_deg)
    # Close the curve
    theta_closed = np.append(theta, theta[0])
    disp_closed = np.append(dispersions, dispersions[0])

    ax_polar.plot(theta_closed, disp_closed, "o-", color="#00FF88",
                  linewidth=2, markersize=8, markeredgecolor="white",
                  markeredgewidth=1)
    ax_polar.fill(theta_closed, disp_closed, alpha=0.15, color="#00FF88")

    # Mark minimum
    min_idx = np.argmin(dispersions)
    ax_polar.plot(theta[min_idx], dispersions[min_idx], "*",
                  color="cyan", markersize=20, markeredgecolor="white",
                  markeredgewidth=1.5, zorder=10)

    # Mark maximum
    max_idx = np.argmax(dispersions)
    ax_polar.plot(theta[max_idx], dispersions[max_idx], "*",
                  color="#E74C3C", markersize=16, markeredgecolor="white",
                  markeredgewidth=1, zorder=10)

    ax_polar.set_facecolor("#0a0a14")
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)  # clockwise
    ax_polar.tick_params(colors="gray")
    ax_polar.grid(color=(0.24, 0.24, 0.31, 0.4))
    ax_polar.set_title(
        "Phase Sensitivity Map\n(radial = dispersion σ)",
        fontsize=12, fontweight="bold", color="#333333", pad=15,
    )

    # ── Cartesian plot ──
    ax_cart.plot(phases_deg, dispersions, "o-", color="#00FF88",
                 linewidth=2, markersize=10, markeredgecolor="white",
                 markeredgewidth=1.5, label="With P9")

    # Baseline (no P9)
    ax_cart.axhline(y=baseline_dispersion, color="#E74C3C", linestyle="--",
                    linewidth=2, alpha=0.8,
                    label=f"No P9 baseline (σ = {baseline_dispersion:.2f}°)")

    # Mark minimum
    best_phase = phases_deg[min_idx]
    best_disp = dispersions[min_idx]
    ax_cart.plot(best_phase, best_disp, "c*", markersize=24,
                 markeredgecolor="white", markeredgewidth=2, zorder=10)
    ax_cart.annotate(
        f"★ MINIMUM\nM = {best_phase:.0f}°\nσ = {best_disp:.2f}°",
        xy=(best_phase, best_disp),
        xytext=(best_phase + 25, best_disp - 0.8),
        fontsize=11, fontweight="bold", color="cyan",
        arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5),
    )

    # Mark maximum
    worst_phase = phases_deg[max_idx]
    worst_disp = dispersions[max_idx]
    ax_cart.plot(worst_phase, worst_disp, "r*", markersize=18,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=10)
    ax_cart.annotate(
        f"MAX\nM = {worst_phase:.0f}°\nσ = {worst_disp:.2f}°",
        xy=(worst_phase, worst_disp),
        xytext=(worst_phase + 25, worst_disp + 0.3),
        fontsize=10, color="#E74C3C",
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2),
    )

    ax_cart.set_xlabel("Planet 9 Initial Mean Anomaly M (degrees)", fontsize=12)
    ax_cart.set_ylabel("Final TNO Dispersion σ (degrees)", fontsize=12)
    ax_cart.set_title(
        "Where Is the Shepherd?\n"
        f"Dispersion vs P9 orbital phase ({best_mass} $M_\\oplus$, {best_dist} AU)",
        fontsize=12, fontweight="bold",
    )
    ax_cart.set_xticks(np.arange(0, 361, 30))
    ax_cart.set_xlim(-10, 370)
    ax_cart.legend(fontsize=11, loc="upper right")
    ax_cart.grid(True, alpha=0.3)

    # Shade region below baseline
    ax_cart.fill_between(
        phases_deg, dispersions, baseline_dispersion,
        where=dispersions < baseline_dispersion,
        alpha=0.15, color="#00FF88",
        label="_nolegend_",
    )

    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Experiment 5: Phase Scan — Pinpointing Planet 9's Orbital Location\n"
        f"{int(duration_myr)} Myr integration, 7 eTNO witnesses, "
        f"P9 = {best_mass} $M_\\oplus$ at {best_dist} AU",
        fontsize=14, fontweight="bold", y=1.02,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = output_dir / "exp5_phase_scan.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def run_phase_scan(
    output_dir: Path,
    duration_myr: float = 1.0,
    dt: float = 50.0,
    record_interval: int = 500,
    best_mass: float = 10.0,
    best_dist: float = 400.0,
    n_phases: int = 12,
) -> None:
    """
    Experiment 5: Phase Scan — Orbital Location Sensitivity.

    With mass and distance fixed at best-fit values (from grid search),
    scans over Planet 9's initial mean anomaly M to determine which
    orbital phase produces the tightest TNO clustering.

    This answers: "WHERE in its orbit is Planet 9 right now?"

    The minimum of the dispersion curve predicts the most probable
    current orbital phase of Planet 9 — a falsifiable prediction.

    Parameters
    ----------
    duration_myr : Duration in millions of years per run (default 1).
    dt : Time step in days (default 50).
    record_interval : Record every N steps (default 500).
    best_mass : P9 mass from grid search (M_Earth, default 10).
    best_dist : P9 distance from grid search (AU, default 400).
    n_phases : Number of phase angles to scan (default 12 → 30° steps).
    """
    from astroworld.catalogs.outer_solar_system import (
        make_outer_system_custom_p9,
        make_outer_system_without_p9,
        GM_STAR,
    )

    print("\n" + "=" * 60)
    print("  EXPERIMENT 5: PHASE SCAN — PINPOINTING P9'S LOCATION")
    print(f"  Fixed: {best_mass} M⊕ at {best_dist} AU")
    print(f"  Scanning {n_phases} orbital phases (0° to {360 - 360/n_phases:.0f}°)")
    print(f"  Duration: {duration_myr} Myr per run")
    print("=" * 60)

    phases_deg = np.linspace(0, 360 - 360 / n_phases, n_phases)
    duration_days = duration_myr * 1e6 * 365.25
    dispersions = np.zeros(n_phases)

    t_total_start = time.perf_counter()

    # ── First: baseline (no P9) for reference ──
    print(f"\n  ── Baseline: No Planet 9 ──")
    state_base = make_outer_system_without_p9()
    sim_base = NBodySimulation(relativistic=False)
    sim_base.set_initial_state(state_base)
    t0 = time.perf_counter()
    result_base = sim_base.run(
        duration_days=duration_days, dt=dt,
        integrator="verlet", record_interval=record_interval,
    )
    elapsed = time.perf_counter() - t0
    print(f"    {len(result_base.times):,} records, {elapsed:.1f}s")

    # Extract baseline dispersion
    star_idx = 0
    omega_base_final = []
    for tno_name in _TNO_WITNESSES:
        if tno_name not in result_base.body_names:
            continue
        tno_idx = result_base.body_names.index(tno_name)
        pos = result_base.positions[-1, tno_idx, :] - result_base.positions[-1, star_idx, :]
        vel = result_base.velocities[-1, tno_idx, :] - result_base.velocities[-1, star_idx, :]
        elems = compute_orbital_elements(pos, vel, GM_STAR)
        omega_base_final.append(elems["omega_peri"])
    baseline_dispersion = _circular_std_deg(np.array(omega_base_final))
    print(f"    Baseline σ_f = {baseline_dispersion:.2f}°")

    # ── Scan over P9 phases ──
    for i, phase in enumerate(phases_deg):
        print(f"\n  [{i+1:2d}/{n_phases}] M = {phase:.0f}°")

        state = make_outer_system_custom_p9(
            mass_earth=best_mass,
            distance_au=best_dist,
            mean_anomaly_deg=phase,
        )
        sim = NBodySimulation(relativistic=False)
        sim.set_initial_state(state)

        t0 = time.perf_counter()
        result = sim.run(
            duration_days=duration_days, dt=dt,
            integrator="verlet", record_interval=record_interval,
        )
        elapsed = time.perf_counter() - t0

        # Extract final ω_peri for all witnesses
        omega_finals = []
        for tno_name in _TNO_WITNESSES:
            if tno_name not in result.body_names:
                continue
            tno_idx = result.body_names.index(tno_name)
            pos = result.positions[-1, tno_idx, :] - result.positions[-1, star_idx, :]
            vel = result.velocities[-1, tno_idx, :] - result.velocities[-1, star_idx, :]
            elems = compute_orbital_elements(pos, vel, GM_STAR)
            omega_finals.append(elems["omega_peri"])

        sigma = _circular_std_deg(np.array(omega_finals))
        dispersions[i] = sigma

        # P9's actual position for this run
        p9_idx = result.body_names.index("planet_9")
        p9_r = np.linalg.norm(
            result.positions[0, p9_idx, :] - result.positions[0, star_idx, :]
        )

        print(f"    σ_f = {sigma:.2f}° | r₀(P9) = {p9_r:.1f} AU | {elapsed:.1f}s")

    t_total = time.perf_counter() - t_total_start

    # ── Summary ──
    min_idx = np.argmin(dispersions)
    max_idx = np.argmax(dispersions)
    best_phase = phases_deg[min_idx]
    best_sigma = dispersions[min_idx]
    worst_phase = phases_deg[max_idx]
    worst_sigma = dispersions[max_idx]

    print(f"\n  {'='*60}")
    print(f"  PHASE SCAN RESULTS ({n_phases} phases, {t_total:.0f}s total)")
    print(f"  {'='*60}")
    print(f"  Phase scan (M°):  σ_f (°)")
    print(f"  {'-'*35}")
    for i, phase in enumerate(phases_deg):
        marker = " ★" if i == min_idx else ""
        print(f"    M = {phase:>5.0f}°  →  σ = {dispersions[i]:.2f}°{marker}")
    print(f"  {'-'*35}")
    print(f"  ★ BEST PHASE:  M = {best_phase:.0f}° → σ = {best_sigma:.2f}°")
    print(f"    WORST PHASE: M = {worst_phase:.0f}° → σ = {worst_sigma:.2f}°")
    print(f"    BASELINE (no P9):       σ = {baseline_dispersion:.2f}°")
    print(f"    Sensitivity range: {worst_sigma - best_sigma:.2f}°")
    print(f"  {'='*60}")

    if best_sigma < baseline_dispersion:
        improvement = baseline_dispersion - best_sigma
        print(f"\n  ✓ Optimal P9 at M={best_phase:.0f}° reduces dispersion by {improvement:.2f}° vs no P9")
        print(f"  → Prediction: Planet 9 is most likely near mean anomaly M ≈ {best_phase:.0f}°")
        # Convert to rough sky region
        # P9's ω_peri = 150°, Ω = 100° → longitude of perihelion ≈ 250°
        # True anomaly at M depends on eccentricity, but M ≈ f for small e
        lon_peri = 150 + 100  # ω + Ω
        predicted_lon = (lon_peri + best_phase) % 360
        print(f"  → Ecliptic longitude estimate: λ ≈ {predicted_lon:.0f}° (very rough)")
    else:
        print(f"\n  ✗ No phase produces tighter clustering than baseline")

    # ── Generate figure ──
    print(f"\n  Generating phase sensitivity plot...")
    _plot_phase_sensitivity(
        phases_deg, dispersions, output_dir, duration_myr,
        best_mass, best_dist, baseline_dispersion,
    )

    print("\n  Experiment 5 complete!")


# ═══════════════════════════════════════════════════════════════════
#  Main CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Planet 9 Investigation Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (all experiments, short durations):
  uv run python scripts/run_p9_investigations.py --experiment all --quick

  # Full Experiment 1 (disk vs planet, 10K years, 100 disk particles):
  uv run python scripts/run_p9_investigations.py --experiment disk_vs_planet

  # Full Experiment 2 (grid search, 100K years, 25 runs):
  uv run python scripts/run_p9_investigations.py --experiment grid_search

  # Full Experiment 3 (100 Myr stability, ~45-90 min):
  uv run python scripts/run_p9_investigations.py --experiment stability_100myr

  # Experiment 4 (Grand Tour, best-fit P9, 10 Myr):
  uv run python scripts/run_p9_investigations.py --experiment grand_tour

  # Experiment 5 (Phase Scan, pinpoint P9's orbital location):
  uv run python scripts/run_p9_investigations.py --experiment phase_scan
        """,
    )
    parser.add_argument(
        "--experiment",
        choices=["disk_vs_planet", "grid_search", "stability_100myr", "grand_tour", "phase_scan", "all"],
        required=True,
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/p9_investigations",
        help="Output directory for plots (default: output/p9_investigations)",
    )
    parser.add_argument(
        "--n-disk", type=int, default=100,
        help="Experiment 1: number of disk particles (default 100)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: shorter durations, fewer particles, smaller grids",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ASTROWORLD — PLANET 9 INVESTIGATIONS")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    experiments = {
        "disk_vs_planet": lambda: run_disk_vs_planet(
            output_dir,
            n_disk=30 if args.quick else args.n_disk,
            duration_years=1_000 if args.quick else 10_000,
            dt=10.0,
            record_interval=1 if args.quick else 10,
        ),
        "grid_search": lambda: run_grid_search(
            output_dir,
            duration_years=10_000 if args.quick else 100_000,
            dt=10.0,
            record_interval=10 if args.quick else 100,
        ),
        "stability_100myr": lambda: run_stability_100myr(
            output_dir,
            duration_myr=1.0 if args.quick else 100.0,
            dt=100.0,
            record_interval=100 if args.quick else 1000,
        ),
        "grand_tour": lambda: run_grand_tour(
            output_dir,
            duration_myr=1.0 if args.quick else 10.0,
            dt=50.0,
            record_interval=50 if args.quick else 500,
            best_mass=10.0,
            best_dist=400.0,
        ),
        "phase_scan": lambda: run_phase_scan(
            output_dir,
            duration_myr=0.5 if args.quick else 1.0,
            dt=50.0,
            record_interval=100 if args.quick else 500,
            best_mass=10.0,
            best_dist=400.0,
            n_phases=6 if args.quick else 12,
        ),
    }

    if args.experiment == "all":
        for name, func in experiments.items():
            func()
    else:
        experiments[args.experiment]()

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE!")
    print(f"  Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
