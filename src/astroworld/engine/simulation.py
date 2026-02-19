"""
N-body simulation orchestrator.

Sets up initial conditions, runs the integration loop, and records
trajectory history for analysis.

Supports two initialization modes:
  1. SystemState (universal): accepts any gravitational system
  2. body_names (legacy): Solar System bodies from constants.BODIES
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass

from astroworld.data.state import SystemState
from astroworld.engine.gravity import (
    compute_accelerations,
    compute_accelerations_with_gr,
    compute_total_energy,
)
from astroworld.engine.integrators import rk4_step, velocity_verlet_step
from astroworld.engine.kernels import HAS_NUMBA, run_verlet_jit


@dataclass
class SimulationResult:
    """Complete trajectory history from a simulation run."""

    times: np.ndarray  # (T,) days from epoch
    positions: np.ndarray  # (T, N, 3) AU
    velocities: np.ndarray  # (T, N, 3) AU/day
    energies: np.ndarray  # (T,) total energy
    body_names: List[str]

    def to_dataframe(self, body_name: str, heliocentric: bool = False) -> pd.DataFrame:
        """
        Extract trajectory for one body as a DataFrame.

        If heliocentric=True, positions and velocities are given relative
        to body at index 0 (the star/Sun). This is needed for comparison
        with heliocentric ephemeris data.
        """
        idx = self.body_names.index(body_name)
        pos = self.positions[:, idx, :]
        vel = self.velocities[:, idx, :]

        if heliocentric and "sun" in self.body_names:
            sun_idx = self.body_names.index("sun")
            pos = pos - self.positions[:, sun_idx, :]
            vel = vel - self.velocities[:, sun_idx, :]

        return pd.DataFrame(
            {
                "time_days": self.times,
                "x": pos[:, 0],
                "y": pos[:, 1],
                "z": pos[:, 2],
                "vx": vel[:, 0],
                "vy": vel[:, 1],
                "vz": vel[:, 2],
            }
        )

    def to_dataframe_star_relative(self, body_name: str, star_index: int = 0) -> pd.DataFrame:
        """
        Extract trajectory relative to the central star (universal version).

        Works for any system (Solar System, exoplanets, etc.)
        """
        idx = self.body_names.index(body_name)
        pos = self.positions[:, idx, :] - self.positions[:, star_index, :]
        vel = self.velocities[:, idx, :] - self.velocities[:, star_index, :]

        return pd.DataFrame(
            {
                "time_days": self.times,
                "x": pos[:, 0],
                "y": pos[:, 1],
                "z": pos[:, 2],
                "vx": vel[:, 0],
                "vy": vel[:, 1],
                "vz": vel[:, 2],
            }
        )

    def to_dataframes(self, heliocentric: bool = False) -> Dict[str, pd.DataFrame]:
        """Extract trajectories for all bodies."""
        return {name: self.to_dataframe(name, heliocentric=heliocentric) for name in self.body_names}


class NBodySimulation:
    """
    N-body gravitational simulation.

    Supports two initialization modes:

    1. Universal (SystemState):
        sim = NBodySimulation(relativistic=True)
        sim.set_initial_state(system_state)

    2. Legacy (Solar System body names):
        sim = NBodySimulation(body_names=["sun", "earth", "mars"])
        sim.set_initial_conditions_from_horizons(epoch_jd=2451545.0)
    """

    def __init__(
        self,
        body_names: Optional[List[str]] = None,
        relativistic: bool = False,
    ):
        self.relativistic = relativistic
        self.positions: np.ndarray | None = None
        self.velocities: np.ndarray | None = None
        self._current_velocities: np.ndarray | None = None

        if body_names is not None:
            # Legacy mode: initialize from constants.BODIES
            from astroworld.constants import BODIES
            self.body_names = body_names
            self.n = len(body_names)
            self.bodies = [BODIES[name] for name in body_names]
            self.gm_values = np.array([b.gm for b in self.bodies])
            self.masses = np.array([b.mass_kg for b in self.bodies])
            self.star_index = body_names.index("sun") if "sun" in body_names else 0
        else:
            # Universal mode: will be set via set_initial_state()
            self.body_names = []
            self.n = 0
            self.bodies = []
            self.gm_values = np.array([])
            self.masses = np.array([])
            self.star_index = 0

    def set_initial_state(self, state: SystemState) -> None:
        """
        Initialize simulation from a universal SystemState.

        This is the preferred method for any gravitational system.
        """
        self.body_names = state.names
        self.n = state.n
        self.gm_values = state.gm_values.copy()
        self.masses = state.masses_kg.copy() if state.masses_kg is not None else self.gm_values.copy()
        self.star_index = state.star_index
        self.positions = state.positions.copy()
        self.velocities = state.velocities.copy()
        self.bodies = []  # Not available in universal mode

    def set_initial_conditions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        """Set initial positions (N,3) in AU and velocities (N,3) in AU/day."""
        assert positions.shape == (self.n, 3)
        assert velocities.shape == (self.n, 3)
        self.positions = positions.copy()
        self.velocities = velocities.copy()

    def set_initial_conditions_from_horizons(
        self,
        epoch_jd: float = 2451545.0,
    ) -> None:
        """Fetch initial conditions from JPL Horizons at the given epoch (legacy Solar System mode)."""
        from astroworld.data.horizons import fetch_initial_conditions
        body_ids = [b.jpl_id for b in self.bodies]
        ic = fetch_initial_conditions(body_ids, epoch_jd=epoch_jd)

        positions = np.zeros((self.n, 3))
        velocities = np.zeros((self.n, 3))
        for i, body in enumerate(self.bodies):
            data = ic[body.jpl_id]
            positions[i] = data["r"]
            velocities[i] = data["v"]

        self.positions = positions
        self.velocities = velocities

    def _accel_fn(self, positions: np.ndarray) -> np.ndarray:
        """Acceleration function. If relativistic, includes 1PN GR correction."""
        if self.relativistic and self._current_velocities is not None:
            return compute_accelerations_with_gr(
                positions,
                self._current_velocities,
                self.gm_values,
                gm_sun=self.gm_values[self.star_index],
                sun_index=self.star_index,
            )
        return compute_accelerations(positions, self.gm_values)

    def run(
        self,
        duration_days: float,
        dt: float = 1.0,
        integrator: Literal["rk4", "verlet"] = "verlet",
        record_interval: int = 1,
        use_jit: bool = True,
    ) -> SimulationResult:
        """
        Run the N-body simulation.

        Parameters
        ----------
        duration_days : Total simulation time in days.
        dt : Time step in days.
        integrator : "rk4" or "verlet".
        record_interval : Record state every N steps.
        use_jit : Use Numba JIT-compiled kernels when available (default True).
                  Set to False to force the Python loop (useful for validation).

        Returns SimulationResult with full trajectory history.
        """
        if self.positions is None or self.velocities is None:
            raise RuntimeError("Initial conditions not set. Call set_initial_conditions first.")

        n_steps = int(duration_days / dt)

        # ── JIT dispatch: fused compiled kernel ──
        # Conditions: Numba available + Verlet integrator + user allows
        if use_jit and HAS_NUMBA and integrator == "verlet":
            from astroworld.constants import C_LIGHT
            c2 = C_LIGHT ** 2
            gm_sun = float(self.gm_values[self.star_index]) if self.relativistic else 0.0

            times, pos_hist, vel_hist, energy_hist = run_verlet_jit(
                self.positions, self.velocities,
                self.gm_values, self.masses,
                dt, n_steps,
                record_interval=record_interval,
                relativistic=self.relativistic,
                gm_sun=gm_sun,
                sun_index=self.star_index,
                c2=c2,
            )

            return SimulationResult(
                times=times,
                positions=pos_hist,
                velocities=vel_hist,
                energies=energy_hist,
                body_names=self.body_names,
            )

        # ── Fallback: Python loop (unchanged from original) ──
        n_records = n_steps // record_interval + 1

        # Pre-allocate history arrays
        times = np.zeros(n_records)
        pos_history = np.zeros((n_records, self.n, 3))
        vel_history = np.zeros((n_records, self.n, 3))
        energy_history = np.zeros(n_records)

        # Current state
        pos = self.positions.copy()
        vel = self.velocities.copy()
        t = 0.0

        # Set current velocities for GR correction
        self._current_velocities = vel.copy()

        # For Verlet, compute initial accelerations
        acc = self._accel_fn(pos) if integrator == "verlet" else None

        # Record initial state
        record_idx = 0
        times[0] = t
        pos_history[0] = pos
        vel_history[0] = vel
        energy_history[0] = compute_total_energy(pos, vel, self.gm_values, self.masses)

        # Integration loop
        for step in range(1, n_steps + 1):
            if integrator == "rk4":
                self._current_velocities = vel.copy()
                pos, vel = rk4_step(pos, vel, self._accel_fn, dt)
            else:
                self._current_velocities = vel.copy()
                pos, vel, acc = velocity_verlet_step(pos, vel, acc, self._accel_fn, dt)

            t += dt

            if step % record_interval == 0:
                record_idx += 1
                if record_idx < n_records:
                    times[record_idx] = t
                    pos_history[record_idx] = pos
                    vel_history[record_idx] = vel
                    energy_history[record_idx] = compute_total_energy(
                        pos, vel, self.gm_values, self.masses
                    )

        # Trim if we recorded fewer than expected
        actual_records = record_idx + 1
        return SimulationResult(
            times=times[:actual_records],
            positions=pos_history[:actual_records],
            velocities=vel_history[:actual_records],
            energies=energy_history[:actual_records],
            body_names=self.body_names,
        )
