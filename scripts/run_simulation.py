"""
Run N-body simulation and compare against JPL Horizons data.

Usage:
    uv run python scripts/run_simulation.py
    uv run python scripts/run_simulation.py --duration 730 --dt 0.5 --integrator verlet
    uv run python scripts/run_simulation.py --duration 365.25 --dt 0.1 --relativistic --analyze
    uv run python scripts/run_simulation.py --split-earth-moon --relativistic --dt 0.1 --analyze
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astroworld.engine.simulation import NBodySimulation
from astroworld.data.store import load_all
from astroworld.analysis.compare import (
    compute_position_error,
    compute_orbital_period,
    summary_statistics,
    decompose_rtn,
    compute_frequency_spectrum,
    separate_secular_periodic,
)
from astroworld.analysis.plots import (
    plot_orbits_2d,
    plot_orbit_comparison,
    plot_position_error,
    plot_energy_conservation,
    plot_residual_dashboard,
    plot_spectral_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="Run N-body simulation")
    from astroworld.constants import DEFAULT_BODIES, DEFAULT_BODIES_SPLIT
    parser.add_argument(
        "--bodies",
        nargs="+",
        default=None,
        help="Bodies to simulate (default: full Solar System)",
    )
    parser.add_argument(
        "--split-earth-moon",
        action="store_true",
        help="Simulate Earth and Moon as separate bodies instead of EMB",
    )
    parser.add_argument("--duration", type=float, default=365.25, help="Duration in days")
    parser.add_argument("--dt", type=float, default=0.5, help="Time step in days")
    parser.add_argument(
        "--integrator", choices=["rk4", "verlet"], default="verlet", help="Integrator"
    )
    parser.add_argument("--relativistic", action="store_true", help="Enable GR corrections")
    parser.add_argument("--epoch", type=float, default=2451545.0, help="Epoch JD (default: J2000)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for plots")
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run deep residual analysis (RTN decomposition + FFT + secular/periodic)",
    )
    args = parser.parse_args()

    # Determine body list
    if args.bodies is not None:
        bodies = args.bodies
    elif args.split_earth_moon:
        bodies = DEFAULT_BODIES_SPLIT
    else:
        bodies = DEFAULT_BODIES

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- Run simulation ---
    mode = "relativistic" if args.relativistic else "Newtonian"
    split_info = " [Earth+Moon split]" if args.split_earth_moon else ""
    print(f"Setting up {len(bodies)}-body {mode} simulation{split_info}: {', '.join(bodies)}")
    sim = NBodySimulation(body_names=bodies, relativistic=args.relativistic)

    print(f"Fetching initial conditions from JPL Horizons (JD={args.epoch})...")
    sim.set_initial_conditions_from_horizons(epoch_jd=args.epoch)

    print(f"Running simulation: {args.duration} days, dt={args.dt}, integrator={args.integrator}")
    result = sim.run(
        duration_days=args.duration,
        dt=args.dt,
        integrator=args.integrator,
    )
    print(f"  -> {len(result.times)} recorded steps")

    # --- Extract trajectories (heliocentric for comparison with Horizons) ---
    sim_trajectories = result.to_dataframes(heliocentric=True)

    # --- Orbital periods ---
    print("\nOrbital periods (estimated from simulation):")
    for name in bodies:
        if name == "sun":
            continue
        period = compute_orbital_period(sim_trajectories[name])
        print(f"  {name.capitalize()}: {period:.2f} days")

    # --- Plot orbits ---
    planets_only = {k: v for k, v in sim_trajectories.items() if k != "sun"}
    fig = plot_orbits_2d(planets_only, title="Simulated Orbits (Heliocentric)", save_path=output_dir / "orbits.png")
    print(f"\nSaved: {output_dir / 'orbits.png'}")

    # --- Energy conservation ---
    fig = plot_energy_conservation(
        result.times,
        result.energies,
        save_path=output_dir / "energy.png",
    )
    e0 = result.energies[0]
    e_final = result.energies[-1]
    drift = abs((e_final - e0) / e0)
    print(f"Saved: {output_dir / 'energy.png'}")
    print(f"Energy drift: {drift:.2e}")

    # --- Compare with ephemeris (if available) ---
    compare_bodies = [b for b in bodies if b != "sun"]
    try:
        eph_data = load_all(compare_bodies)
    except FileNotFoundError:
        print("\nNo ephemeris data found. Run download_ephemeris.py first to enable comparison.")
        return

    errors = {}
    for name in compare_bodies:
        err = compute_position_error(sim_trajectories[name], eph_data[name])
        errors[name] = err

        fig = plot_orbit_comparison(
            sim_trajectories[name],
            eph_data[name],
            name,
            save_path=output_dir / f"{name}_comparison.png",
        )
        print(f"Saved: {output_dir / f'{name}_comparison.png'}")

    fig = plot_position_error(errors, save_path=output_dir / "errors.png")
    print(f"Saved: {output_dir / 'errors.png'}")

    stats = summary_statistics(errors)
    print("\nPosition Error Summary:")
    print(stats.to_string())

    # --- Deep residual analysis (Phase 4) ---
    if not args.analyze:
        return

    print("\n" + "=" * 60)
    print("DEEP RESIDUAL ANALYSIS")
    print("=" * 60)

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    spectra = {}

    for name in compare_bodies:
        print(f"\n--- {name.capitalize()} ---")

        # RTN decomposition
        rtn = decompose_rtn(sim_trajectories[name], eph_data[name])

        print(f"  RTN final errors: R={rtn['radial_km'].iloc[-1]:.1f} km, "
              f"T={rtn['transverse_km'].iloc[-1]:.1f} km, "
              f"N={rtn['normal_km'].iloc[-1]:.1f} km")

        # Spectral analysis on transverse component (most sensitive to missing bodies)
        spectrum = compute_frequency_spectrum(
            rtn["time_days"].values,
            rtn["transverse_km"].values,
            n_peaks=5,
        )
        spectra[name] = spectrum

        # Secular vs periodic separation on total residual
        sec_per = separate_secular_periodic(
            rtn["time_days"].values,
            rtn["total_km"].values,
        )

        print(f"  Secular drift: {sec_per['secular_rate_km_per_year']:.1f} km/year")

        if spectrum["peaks"]:
            print(f"  Dominant spectral peaks:")
            for period, power in spectrum["peaks"][:3]:
                print(f"    Period = {period:.1f} days (power = {power:.2e})")

        # Dashboard plot
        fig = plot_residual_dashboard(
            rtn, spectrum, sec_per, name,
            component="transverse",
            save_path=analysis_dir / f"{name}_dashboard.png",
        )
        plt.close(fig)
        print(f"  Saved: {analysis_dir / f'{name}_dashboard.png'}")

    # Comparative spectral plot
    fig = plot_spectral_comparison(spectra, save_path=analysis_dir / "spectra_comparison.png")
    plt.close(fig)
    print(f"\nSaved: {analysis_dir / 'spectra_comparison.png'}")


if __name__ == "__main__":
    main()
