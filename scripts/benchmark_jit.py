"""
Benchmark JIT kernels vs Python loop.

Measures wall-clock time for several scenarios to quantify
the speedup from Numba JIT compilation.

Usage:
    uv run python scripts/benchmark_jit.py
"""

import sys
import time
import numpy as np

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from astroworld.engine.simulation import NBodySimulation
from astroworld.engine.kernels import HAS_NUMBA, run_verlet_jit
from astroworld.data.state import SystemState
from astroworld.constants import GM_SUN, BODIES, C_LIGHT


def _make_solar_system_ic():
    """Sun + 4 inner planets circular orbits (5 bodies)."""
    names = ["sun", "mercury", "venus", "earth", "mars"]
    bodies = [BODIES[n] for n in names]
    n = len(names)

    a_vals = [0.0, 0.387, 0.723, 1.0, 1.524]
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    gm_values = np.array([b.gm for b in bodies])
    masses = np.array([b.mass_kg for b in bodies])

    for i in range(1, n):
        positions[i, 0] = a_vals[i]
        velocities[i, 1] = np.sqrt(GM_SUN / a_vals[i])

    return SystemState(
        names=names,
        gm_values=gm_values,
        positions=positions,
        velocities=velocities,
        masses_kg=masses,
        star_index=0,
    )


def _make_outer_system_ic():
    """Outer solar system: 9 bodies (with P9)."""
    from astroworld.catalogs.outer_solar_system import make_outer_system_with_p9
    return make_outer_system_with_p9()


def benchmark_scenario(name, state, duration_days, dt, record_interval=1,
                       relativistic=False, n_warmup=1):
    """Run scenario with both JIT and Python, measure wall-clock times."""
    n_steps = int(duration_days / dt)

    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"  Bodies: {state.n}, Steps: {n_steps:,}, dt={dt}, record_interval={record_interval}")
    print(f"{'─' * 60}")

    # ── Warm up JIT (first call triggers compilation, ~2-5 sec) ──
    if HAS_NUMBA:
        sim_warmup = NBodySimulation(relativistic=relativistic)
        sim_warmup.set_initial_state(state)
        _ = sim_warmup.run(duration_days=10 * dt, dt=dt, use_jit=True, record_interval=1)

    # ── JIT benchmark ──
    if HAS_NUMBA:
        sim_jit = NBodySimulation(relativistic=relativistic)
        sim_jit.set_initial_state(state)
        t0 = time.perf_counter()
        result_jit = sim_jit.run(
            duration_days=duration_days, dt=dt,
            use_jit=True, record_interval=record_interval,
        )
        t_jit = time.perf_counter() - t0
    else:
        t_jit = None
        result_jit = None

    # ── Python benchmark ──
    sim_py = NBodySimulation(relativistic=relativistic)
    sim_py.set_initial_state(state)
    t0 = time.perf_counter()
    result_py = sim_py.run(
        duration_days=duration_days, dt=dt,
        use_jit=False, record_interval=record_interval,
    )
    t_py = time.perf_counter() - t0

    # ── Report ──
    print(f"  Python loop:  {t_py:.3f} sec")
    if t_jit is not None:
        speedup = t_py / t_jit if t_jit > 0 else float("inf")
        print(f"  JIT kernel:   {t_jit:.3f} sec")
        print(f"  Speedup:      {speedup:.1f}x")

        # Verify results match
        if result_jit is not None:
            pos_diff = np.max(np.abs(result_jit.positions - result_py.positions))
            e_diff = np.max(np.abs(
                (result_jit.energies - result_py.energies) / abs(result_py.energies[0])
            ))
            print(f"  Max pos diff: {pos_diff:.2e} AU")
            print(f"  Max E diff:   {e_diff:.2e} (relative)")
    else:
        print(f"  JIT kernel:   N/A (Numba not installed)")

    # Energy conservation
    e0 = result_py.energies[0]
    drift_py = abs((result_py.energies[-1] - e0) / abs(e0))
    print(f"  Energy drift: {drift_py:.2e}")

    return t_py, t_jit


def main():
    print("=" * 60)
    print("  ASTROWORLD JIT BENCHMARK")
    print(f"  Numba available: {HAS_NUMBA}")
    print("=" * 60)

    results = []

    # Scenario 1: 5-body, 10K steps (quick test)
    state1 = _make_solar_system_ic()
    t_py, t_jit = benchmark_scenario(
        "5-body inner solar system, 10K steps",
        state1, duration_days=10000.0, dt=1.0, record_interval=10,
    )
    results.append(("5-body 10K steps", t_py, t_jit))

    # Scenario 2: 5-body, 100K steps
    t_py, t_jit = benchmark_scenario(
        "5-body inner solar system, 100K steps",
        state1, duration_days=100000.0, dt=1.0, record_interval=100,
    )
    results.append(("5-body 100K steps", t_py, t_jit))

    # Scenario 3: 9-body outer system, 36.5K steps (standard P9 run)
    state3 = _make_outer_system_ic()
    t_py, t_jit = benchmark_scenario(
        "9-body outer system, 36.5K steps (10,000 years, dt=10)",
        state3, duration_days=3652500.0, dt=10.0, record_interval=10,
    )
    results.append(("9-body 36.5K steps", t_py, t_jit))

    # Scenario 4: 9-body outer system, 365K steps (100K years)
    t_py, t_jit = benchmark_scenario(
        "9-body outer system, 365K steps (100,000 years, dt=10)",
        state3, duration_days=36525000.0, dt=10.0, record_interval=100,
    )
    results.append(("9-body 365K steps", t_py, t_jit))

    # Scenario 5: 2-body relativistic (Sun + Mercury), 10K steps
    state5 = SystemState(
        names=["sun", "mercury"],
        gm_values=np.array([GM_SUN, BODIES["mercury"].gm]),
        positions=np.array([[0, 0, 0], [0.387, 0, 0]], dtype=np.float64),
        velocities=np.array([[0, 0, 0], [0, np.sqrt(GM_SUN / 0.387), 0]], dtype=np.float64),
        masses_kg=np.array([BODIES["sun"].mass_kg, BODIES["mercury"].mass_kg]),
        star_index=0,
    )
    t_py, t_jit = benchmark_scenario(
        "2-body GR (Sun+Mercury), 100K steps, dt=0.01",
        state5, duration_days=1000.0, dt=0.01, record_interval=100,
        relativistic=True,
    )
    results.append(("2-body GR 100K steps", t_py, t_jit))

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Scenario':<30} {'Python':>8} {'JIT':>8} {'Speedup':>8}")
    print(f"  {'-' * 54}")
    for name, t_py, t_jit in results:
        if t_jit is not None:
            speedup = f"{t_py / t_jit:.1f}x"
            t_jit_str = f"{t_jit:.3f}s"
        else:
            speedup = "N/A"
            t_jit_str = "N/A"
        print(f"  {name:<30} {t_py:>7.3f}s {t_jit_str:>8} {speedup:>8}")


if __name__ == "__main__":
    main()
