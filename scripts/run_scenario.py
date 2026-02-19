"""
Universal scenario runner for any gravitational system.

This is the new unified entry point. It accepts any system catalog
(Solar System, TRAPPIST-1, or custom) via SystemState, making the
physics engine completely agnostic to the origin of the data.

Usage:
    # TRAPPIST-1 (from Keplerian elements, no internet needed)
    uv run python scripts/run_scenario.py --scenario trappist_1

    # TRAPPIST-1 subset
    uv run python scripts/run_scenario.py --scenario trappist_1 --planets b c d

    # Solar System (fetches from JPL Horizons)
    uv run python scripts/run_scenario.py --scenario solar_system --relativistic

    # Solar System with Earth-Moon split
    uv run python scripts/run_scenario.py --scenario solar_system --split-earth-moon --dt 0.01

    # Outer Solar System: Planet 9 hunt (no internet needed)
    uv run python scripts/run_scenario.py --scenario outer_system
    uv run python scripts/run_scenario.py --scenario outer_system --with-planet9 --plot-3d

    # Custom duration / timestep
    uv run python scripts/run_scenario.py --scenario trappist_1 --duration 100 --dt 0.001
"""

import argparse
import sys
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 encoding errors)
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


# ── Scenario loaders ──────────────────────────────────────────────

def _load_trappist1(args) -> SystemState:
    """Load TRAPPIST-1 system from Keplerian catalog."""
    from astroworld.catalogs.trappist_1 import make_trappist1_state
    include = args.planets if args.planets else None
    return make_trappist1_state(include_planets=include)


def _load_solar_system(args) -> SystemState:
    """Load Solar System from JPL Horizons."""
    from astroworld.catalogs.solar_system import make_solar_system_state, DEFAULT_BODIES, DEFAULT_BODIES_SPLIT

    if args.bodies:
        body_names = args.bodies
    elif args.split_earth_moon:
        body_names = DEFAULT_BODIES_SPLIT
    else:
        body_names = DEFAULT_BODIES

    print(f"Fetching Solar System state from JPL Horizons (JD={args.epoch})...")
    return make_solar_system_state(body_names=body_names, epoch_jd=args.epoch)


def _load_outer_system(args) -> SystemState:
    """Load outer Solar System for Planet 9 hunting."""
    from astroworld.catalogs.outer_solar_system import (
        make_outer_system_with_p9,
        make_outer_system_without_p9,
    )
    tno_names = args.tnos if args.tnos else None
    if args.with_planet9:
        return make_outer_system_with_p9(tno_names=tno_names)
    return make_outer_system_without_p9(tno_names=tno_names)


SCENARIOS = {
    "trappist_1": _load_trappist1,
    "solar_system": _load_solar_system,
    "outer_system": _load_outer_system,
}


# ── Plotting helpers (system-agnostic) ────────────────────────────

def plot_orbits(result: SimulationResult, state: SystemState,
                save_path: Path) -> None:
    """Plot 2D orbits relative to the central star."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    star_idx = state.star_index
    colors = state.colors or [None] * state.n

    for i, name in enumerate(result.body_names):
        if i == star_idx:
            continue
        x = result.positions[:, i, 0] - result.positions[:, star_idx, 0]
        y = result.positions[:, i, 1] - result.positions[:, star_idx, 1]
        ax.plot(x, y, label=name, color=colors[i], linewidth=0.5, alpha=0.8)

    ax.plot(0, 0, "*", color=colors[star_idx] if colors[star_idx] else "gold",
            markersize=15, label=result.body_names[star_idx])
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Orbital Trajectories ({len(result.body_names)} bodies)")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy(result: SimulationResult, save_path: Path) -> None:
    """Plot energy conservation over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    e0 = result.energies[0]
    relative = (result.energies - e0) / abs(e0)
    ax.plot(result.times, relative, "b-", linewidth=0.5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative energy drift (E - E0) / |E0|")
    ax.set_title("Energy Conservation")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distances(result: SimulationResult, state: SystemState,
                   save_path: Path) -> None:
    """Plot distance from star vs time for all bodies."""
    fig, ax = plt.subplots(figsize=(12, 6))
    star_idx = state.star_index
    colors = state.colors or [None] * state.n

    for i, name in enumerate(result.body_names):
        if i == star_idx:
            continue
        dr = result.positions[:, i, :] - result.positions[:, star_idx, :]
        dist = np.linalg.norm(dr, axis=1)
        ax.plot(result.times, dist, label=name, color=colors[i],
                linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Distance from star (AU)")
    ax.set_title("Orbital Distances")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_orbital_periods(result: SimulationResult, state: SystemState) -> dict:
    """
    Estimate orbital periods from simulation via radial distance oscillation.

    Uses zero-crossing detection on the detrended radial distance.
    Returns dict of {body_name: period_days}.
    """
    star_idx = state.star_index
    periods = {}

    for i, name in enumerate(result.body_names):
        if i == star_idx:
            continue
        dr = result.positions[:, i, :] - result.positions[:, star_idx, :]
        dist = np.linalg.norm(dr, axis=1)
        t = result.times

        # Remove mean and find zero-crossings of distance oscillation
        dist_centered = dist - np.mean(dist)
        crossings = []
        for j in range(1, len(dist_centered)):
            if dist_centered[j - 1] < 0 and dist_centered[j] >= 0:
                # Linear interpolation for precise crossing
                frac = -dist_centered[j - 1] / (dist_centered[j] - dist_centered[j - 1])
                crossings.append(t[j - 1] + frac * (t[j] - t[j - 1]))

        if len(crossings) >= 2:
            diffs = np.diff(crossings)
            periods[name] = float(np.median(diffs))
        else:
            periods[name] = float("nan")

    return periods


# ── Outer System / Planet 9 analysis ─────────────────────────────

# Use shared orbital elements module (extracted from this file)
from astroworld.analysis.orbital_elements import compute_orbital_elements as _compute_orbital_elements


def _analyze_tno_clustering(result: SimulationResult, state: SystemState,
                            output_dir: Path) -> None:
    """
    Track how TNO arguments of perihelion (ω) evolve over time.

    In the P9 hypothesis, ω values should remain clustered near ~0°
    (anti-aligned with P9's perihelion). Without P9, gas giant
    perturbations should cause them to disperse.
    """
    from astroworld.catalogs.outer_solar_system import GM_STAR

    star_idx = state.star_index
    tno_names = ["sedna", "2012_vp113", "leleakuhonua"]

    # Find TNO indices
    tno_indices = {}
    for i, name in enumerate(result.body_names):
        if name in tno_names:
            tno_indices[name] = i

    if not tno_indices:
        return

    print(f"\nTNO orbital element evolution ({len(tno_indices)} bodies):")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    colors_tno = {"sedna": "#E74C3C", "2012_vp113": "#E67E22", "leleakuhonua": "#9B59B6"}

    # Sample every N-th point for element computation (expensive)
    n_samples = min(2000, len(result.times))
    sample_idx = np.unique(np.linspace(0, len(result.times) - 1, n_samples, dtype=int))
    t_years = result.times[sample_idx] / 365.25

    for tno_name, body_idx in tno_indices.items():
        omega_peri_list = []
        longitude_peri_list = []
        ecc_list = []

        for k in sample_idx:
            pos = result.positions[k, body_idx, :] - result.positions[k, star_idx, :]
            vel = result.velocities[k, body_idx, :] - result.velocities[k, star_idx, :]
            elems = _compute_orbital_elements(pos, vel, GM_STAR)
            omega_peri_list.append(elems["omega_peri"])
            longitude_peri_list.append(elems["longitude_peri"])
            ecc_list.append(elems["e"])

        omega_arr = np.array(omega_peri_list)
        lp_arr = np.array(longitude_peri_list)
        e_arr = np.array(ecc_list)

        color = colors_tno.get(tno_name, "white")

        # Unwrap angles for smooth plotting
        omega_unwrap = np.degrees(np.unwrap(np.radians(omega_arr)))
        lp_unwrap = np.degrees(np.unwrap(np.radians(lp_arr)))

        axes[0].plot(t_years, omega_unwrap, color=color, linewidth=0.8,
                     label=tno_name, alpha=0.9)
        axes[1].plot(t_years, lp_unwrap, color=color, linewidth=0.8,
                     label=tno_name, alpha=0.9)
        axes[2].plot(t_years, e_arr, color=color, linewidth=0.8,
                     label=tno_name, alpha=0.9)

        # Summary stats
        delta_omega = omega_unwrap[-1] - omega_unwrap[0]
        T_years = t_years[-1] - t_years[0]
        rate = delta_omega / T_years if T_years > 0 else 0
        print(f"  {tno_name}:")
        print(f"    omega initial: {omega_arr[0]:.1f} deg, final: {omega_arr[-1]:.1f} deg")
        print(f"    delta-omega total: {delta_omega:.1f} deg over {T_years:.0f} years "
              f"({rate:.4f} deg/year)")
        print(f"    e range: {e_arr.min():.4f} - {e_arr.max():.4f}")

    axes[0].set_ylabel(r"$\omega$ (argument of perihelion) [$\degree$]")
    axes[0].set_title("TNO Orbital Element Evolution")
    axes[1].set_ylabel(r"$\varpi$ (longitude of perihelion) [$\degree$]")
    axes[2].set_ylabel("Eccentricity")
    axes[2].set_xlabel("Time (years)")

    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "tno_elements.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: tno_elements.png")


def _extract_tno_elements(result: SimulationResult, state: SystemState,
                           gm_star: float) -> dict:
    """
    Extract time-series of orbital elements for all TNOs in a simulation result.

    Returns dict of {tno_name: {omega: array, varpi: array, ecc: array, t_years: array}}.
    """
    tno_names = ["sedna", "2012_vp113", "leleakuhonua"]
    star_idx = state.star_index

    tno_indices = {}
    for i, name in enumerate(result.body_names):
        if name in tno_names:
            tno_indices[name] = i

    # Sample every N-th point for element computation
    n_samples = min(2000, len(result.times))
    sample_idx = np.unique(np.linspace(0, len(result.times) - 1, n_samples, dtype=int))
    t_years = result.times[sample_idx] / 365.25

    elements = {}
    for tno_name, body_idx in tno_indices.items():
        omega_list, varpi_list, ecc_list = [], [], []
        for k in sample_idx:
            pos = result.positions[k, body_idx, :] - result.positions[k, star_idx, :]
            vel = result.velocities[k, body_idx, :] - result.velocities[k, star_idx, :]
            elems = _compute_orbital_elements(pos, vel, gm_star)
            omega_list.append(elems["omega_peri"])
            varpi_list.append(elems["longitude_peri"])
            ecc_list.append(elems["e"])

        omega_arr = np.array(omega_list)
        varpi_arr = np.array(varpi_list)
        # Unwrap for smooth plotting
        elements[tno_name] = {
            "omega": np.degrees(np.unwrap(np.radians(omega_arr))),
            "varpi": np.degrees(np.unwrap(np.radians(varpi_arr))),
            "ecc": np.array(ecc_list),
            "t_years": t_years,
        }

    return elements


def _plot_p9_comparison(elems_no: dict, elems_p9: dict, output_dir: Path) -> None:
    """
    Generate the key science figure: side-by-side TNO element evolution
    WITHOUT vs WITH Planet 9, plus differential panel.

    Layout: 3 rows × 3 columns
      Col 0: Without P9     Col 1: With P9     Col 2: Difference (Δ)
      Row 0: ω (argument of perihelion)
      Row 1: ϖ (longitude of perihelion)
      Row 2: e (eccentricity)
    """
    colors_tno = {"sedna": "#E74C3C", "2012_vp113": "#E67E22", "leleakuhonua": "#9B59B6"}
    row_labels = [
        (r"$\omega$ (°)", "omega"),
        (r"$\varpi$ (°)", "varpi"),
        ("Eccentricity", "ecc"),
    ]
    col_titles = ["Without Planet 9 (baseline)", "With Planet 9 (hypothesis)", "Difference (P9 – baseline)"]

    fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharex=True)

    for row, (ylabel, key) in enumerate(row_labels):
        for tno_name in ["sedna", "2012_vp113", "leleakuhonua"]:
            if tno_name not in elems_no or tno_name not in elems_p9:
                continue
            color = colors_tno[tno_name]
            t = elems_no[tno_name]["t_years"]

            val_no = elems_no[tno_name][key]
            val_p9 = elems_p9[tno_name][key]

            # Col 0: Without P9
            axes[row, 0].plot(t, val_no, color=color, linewidth=0.8,
                              label=tno_name, alpha=0.9)
            # Col 1: With P9
            axes[row, 1].plot(t, val_p9, color=color, linewidth=0.8,
                              label=tno_name, alpha=0.9)
            # Col 2: Difference
            # Ensure arrays are same length (they should be, same dt/duration)
            n_min = min(len(val_no), len(val_p9))
            diff = val_p9[:n_min] - val_no[:n_min]
            axes[row, 2].plot(t[:n_min], diff, color=color, linewidth=0.8,
                              label=tno_name, alpha=0.9)

        # Labels
        axes[row, 0].set_ylabel(ylabel, fontsize=11)
        for col in range(3):
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=8)

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    # X-axis label
    for col in range(3):
        axes[2, col].set_xlabel("Time (years)", fontsize=11)

    fig.suptitle("Planet 9 Hypothesis: TNO Orbital Element Evolution\n"
                 f"10,000-year N-body integration (Velocity Verlet, dt=10 days)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / "p9_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _run_p9_comparison(args, output_dir: Path) -> None:
    """
    Run both P9 and no-P9 simulations, then compare TNO element evolution.

    This is the key scientific result: the differential signal reveals
    whether Planet 9 maintains the observed clustering.
    """
    from astroworld.catalogs.outer_solar_system import (
        make_outer_system_with_p9,
        make_outer_system_without_p9,
        GM_STAR,
    )

    print("\n" + "=" * 60)
    print("PLANET 9 COMPARISON: Running paired simulations")
    print("=" * 60)

    tno_names = args.tnos if args.tnos else None

    # Always run BOTH simulations for a proper comparison
    print("\n--- Baseline: WITHOUT Planet 9 ---")
    state_no = make_outer_system_without_p9(tno_names=tno_names)
    sim_no = NBodySimulation(relativistic=False)
    sim_no.set_initial_state(state_no)
    result_no = sim_no.run(
        duration_days=args.duration, dt=args.dt,
        integrator=args.integrator, record_interval=args.record_interval,
    )
    print(f"  -> {len(result_no.times)} steps")
    e0 = result_no.energies[0]
    drift_no = abs((result_no.energies[-1] - e0) / abs(e0))
    print(f"  Energy drift: {drift_no:.2e}")

    print("\n--- Hypothesis: WITH Planet 9 ---")
    state_p9 = make_outer_system_with_p9(tno_names=tno_names)
    sim_p9 = NBodySimulation(relativistic=False)
    sim_p9.set_initial_state(state_p9)
    result_p9 = sim_p9.run(
        duration_days=args.duration, dt=args.dt,
        integrator=args.integrator, record_interval=args.record_interval,
    )
    print(f"  -> {len(result_p9.times)} steps")
    e0 = result_p9.energies[0]
    drift_p9 = abs((result_p9.energies[-1] - e0) / abs(e0))
    print(f"  Energy drift: {drift_p9:.2e}")

    # Extract TNO orbital elements from both runs
    print("\n--- Extracting TNO orbital elements ---")
    elems_no = _extract_tno_elements(result_no, state_no, GM_STAR)
    elems_p9 = _extract_tno_elements(result_p9, state_p9, GM_STAR)

    # Print comparative summary
    print("\n" + "=" * 60)
    print("COMPARATIVE RESULTS: TNO omega drift over simulation")
    print("=" * 60)
    print(f"  {'TNO':<18} {'No P9 (deg)':>14} {'With P9 (deg)':>14} {'Diff (deg)':>12}")
    print(f"  {'-'*58}")
    for tno_name in ["sedna", "2012_vp113", "leleakuhonua"]:
        if tno_name in elems_no and tno_name in elems_p9:
            d_no = elems_no[tno_name]["omega"][-1] - elems_no[tno_name]["omega"][0]
            d_p9 = elems_p9[tno_name]["omega"][-1] - elems_p9[tno_name]["omega"][0]
            print(f"  {tno_name:<18} {d_no:>+14.4f} {d_p9:>+14.4f} {d_p9-d_no:>+12.4f}")

    # Generate side-by-side comparison plot (key science figure)
    _plot_p9_comparison(elems_no, elems_p9, output_dir)

    # Generate individual TNO element plots for each scenario
    dir_no = output_dir / "no_p9"
    dir_p9 = output_dir / "with_p9"
    dir_no.mkdir(parents=True, exist_ok=True)
    dir_p9.mkdir(parents=True, exist_ok=True)

    _analyze_tno_clustering(result_no, state_no, dir_no)
    _analyze_tno_clustering(result_p9, state_p9, dir_p9)

    # Generate standard plots for both scenarios
    for label, result, state, sub_dir in [
        ("no_p9", result_no, state_no, dir_no),
        ("with_p9", result_p9, state_p9, dir_p9),
    ]:
        plot_orbits(result, state, sub_dir / "orbits.png")
        plot_energy(result, sub_dir / "energy.png")
        plot_distances(result, state, sub_dir / "distances.png")
        print(f"  Saved: {label}/ orbits, energy, distances")

    # Generate 3D visualizations for both if requested
    if args.plot_3d:
        from astroworld.analysis.visualization_3d import plot_system_3d

        html_no = dir_no / "outer_no_p9_3d.html"
        plot_system_3d(result_no, state_no, html_no, animate=args.animate)
        print(f"  Saved: {html_no.name}")

        html_p9 = dir_p9 / "outer_with_p9_3d.html"
        plot_system_3d(result_p9, state_p9, html_p9, animate=args.animate)
        print(f"  Saved: {html_p9.name}")

    print("\n  Comparison complete! Key output: p9_comparison.png")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal N-body scenario runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        required=True,
        help="Which system to simulate (trappist_1, solar_system, outer_system)",
    )
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration in days (default: auto-detect from system)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Time step in days (default: auto-detect)")
    parser.add_argument("--integrator", choices=["rk4", "verlet"], default="verlet")
    parser.add_argument("--relativistic", action="store_true",
                        help="Enable 1PN GR corrections")
    parser.add_argument("--record-interval", type=int, default=1,
                        help="Record state every N steps")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory for plots")

    # Solar System specific
    parser.add_argument("--bodies", nargs="+", default=None,
                        help="Solar System: specific body names")
    parser.add_argument("--split-earth-moon", action="store_true",
                        help="Solar System: separate Earth and Moon")
    parser.add_argument("--epoch", type=float, default=2451545.0,
                        help="Solar System: epoch JD (default: J2000)")

    # TRAPPIST-1 specific
    parser.add_argument("--planets", nargs="+", default=None,
                        help="TRAPPIST-1: planet letters to include (e.g. b c d)")

    # Outer system / Planet 9 specific
    parser.add_argument("--with-planet9", action="store_true",
                        help="Outer system: include hypothetical Planet 9")
    parser.add_argument("--tnos", nargs="+", default=None,
                        help="Outer system: specific TNO names (e.g. sedna 2012_vp113)")
    parser.add_argument("--compare-p9", action="store_true",
                        help="Outer system: run both WITH and WITHOUT P9 and compare")

    # 3D visualization
    parser.add_argument("--plot-3d", action="store_true",
                        help="Generate interactive 3D Plotly visualization (.html)")
    parser.add_argument("--animate", action="store_true",
                        help="Add animation controls to 3D plot (Play/Pause + slider)")

    args = parser.parse_args()

    # ── Defaults per scenario ──
    defaults = {
        "trappist_1": {"duration": 50.0, "dt": 0.001, "record_interval": 1},
        "solar_system": {"duration": 365.25, "dt": 0.5, "record_interval": 1},
        # 10,000 years, dt=10 days (Jupiter well-resolved: 4332/10 = 433 steps/orbit)
        # Record every 10 steps to keep memory manageable (~365K steps → ~36K records)
        "outer_system": {"duration": 3652500.0, "dt": 10.0, "record_interval": 10},
    }
    scenario_defaults = defaults[args.scenario]
    if args.duration is None:
        args.duration = scenario_defaults["duration"]
    if args.dt is None:
        args.dt = scenario_defaults["dt"]
    if args.record_interval == 1 and "record_interval" in scenario_defaults:
        args.record_interval = scenario_defaults["record_interval"]

    # ── Load system state ──
    print(f"=== Scenario: {args.scenario} ===")
    state = SCENARIOS[args.scenario](args)
    print(f"System: {state.n} bodies - {', '.join(state.names)}")

    # ── Setup and run simulation ──
    sim = NBodySimulation(relativistic=args.relativistic)
    sim.set_initial_state(state)

    mode = "relativistic (1PN)" if args.relativistic else "Newtonian"
    print(f"Running {mode} simulation: {args.duration} days, dt={args.dt}, "
          f"integrator={args.integrator}")

    result = sim.run(
        duration_days=args.duration,
        dt=args.dt,
        integrator=args.integrator,
        record_interval=args.record_interval,
    )
    print(f"  -> {len(result.times)} recorded steps, "
          f"final time = {result.times[-1]:.2f} days")

    # ── Energy conservation ──
    e0 = result.energies[0]
    e_final = result.energies[-1]
    drift = abs((e_final - e0) / abs(e0))
    print(f"\nEnergy conservation: drift = {drift:.2e}")

    # ── Orbital periods ──
    periods = analyze_orbital_periods(result, state)
    print("\nOrbital periods (from simulation):")
    for name, period in periods.items():
        if np.isnan(period):
            print(f"  {name}: could not detect (too few orbits?)")
        else:
            print(f"  {name}: {period:.4f} days")

    # ── TRAPPIST-1: compare with theoretical Keplerian periods ──
    if args.scenario == "trappist_1":
        from astroworld.catalogs.trappist_1 import get_orbital_periods
        theoretical = get_orbital_periods()
        print("\nComparison with Keplerian theory:")
        print(f"  {'Planet':<20} {'Simulated':>12} {'Keplerian':>12} {'Diff %':>10}")
        print(f"  {'-'*54}")
        for name, sim_period in periods.items():
            letter = name.split("_")[-1]
            if letter in theoretical and not np.isnan(sim_period):
                kep = theoretical[letter]
                diff_pct = (sim_period - kep) / kep * 100
                print(f"  {name:<20} {sim_period:>12.4f} {kep:>12.4f} {diff_pct:>+10.4f}%")

    # ── Generate plots ──
    output_dir = Path(args.output_dir) / args.scenario
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in {output_dir}/")

    plot_orbits(result, state, output_dir / "orbits.png")
    print(f"  Saved: orbits.png")

    plot_energy(result, output_dir / "energy.png")
    print(f"  Saved: energy.png")

    plot_distances(result, state, output_dir / "distances.png")
    print(f"  Saved: distances.png")

    # ── Stability check (for compact systems) ──
    if args.scenario == "trappist_1":
        star_idx = state.star_index
        print("\nStability check (min/max distance from star):")
        for i, name in enumerate(result.body_names):
            if i == star_idx:
                continue
            dr = result.positions[:, i, :] - result.positions[:, star_idx, :]
            dist = np.linalg.norm(dr, axis=1)
            print(f"  {name}: {dist.min():.6f} - {dist.max():.6f} AU "
                  f"(ratio max/min = {dist.max()/dist.min():.4f})")

    # ── Outer system: theoretical periods + TNO analysis ──
    if args.scenario == "outer_system":
        from astroworld.catalogs.outer_solar_system import get_theoretical_periods
        theoretical = get_theoretical_periods()
        print("\nTheoretical Keplerian periods:")
        for name, T_days in theoretical.items():
            T_years = T_days / 365.25
            print(f"  {name}: {T_days:.1f} days ({T_years:.1f} years)")

        # Analyze TNO argument of perihelion clustering
        _analyze_tno_clustering(result, state, output_dir)

    # ── Outer system: compare P9 vs no-P9 ──
    if args.scenario == "outer_system" and args.compare_p9:
        _run_p9_comparison(args, output_dir)

    # ── 3D interactive visualization ──
    if args.plot_3d:
        from astroworld.analysis.visualization_3d import plot_system_3d
        html_path = output_dir / f"{args.scenario}_3d.html"
        plot_system_3d(
            result, state, html_path,
            animate=args.animate,
        )
        mode_str = "animated" if args.animate else "static"
        print(f"  Saved: {html_path.name} ({mode_str} 3D)")

    print("\nDone!")


if __name__ == "__main__":
    main()
