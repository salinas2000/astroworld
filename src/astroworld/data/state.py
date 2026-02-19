"""
Universal system state: the contract between data providers and the physics engine.

Any gravitational system (Solar System, exoplanets, moons, star clusters) can be
described by this single dataclass. The engine never needs to know WHERE the data
came from — only positions, velocities, and GM values matter.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SystemState:
    """
    Initial conditions for an N-body gravitational simulation.

    This is the universal interface between data sources (JPL Horizons,
    Keplerian elements, manual input) and the physics engine.

    All units are in the simulation system: AU, days, AU³/day².

    Attributes
    ----------
    names : List of body names (e.g. ["star", "planet_b", "planet_c"])
    gm_values : (N,) array of GM values in AU³/day²
    positions : (N, 3) array of positions in AU
    velocities : (N, 3) array of velocities in AU/day
    masses_kg : (N,) optional array of masses in kg (for energy computation)
    star_index : Index of the central star (for GR corrections), default 0
    colors : Optional list of colors for plotting
    """

    names: List[str]
    gm_values: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses_kg: Optional[np.ndarray] = None
    star_index: int = 0
    colors: Optional[List[str]] = None

    def __post_init__(self):
        n = len(self.names)
        assert self.gm_values.shape == (n,), f"gm_values shape {self.gm_values.shape} != ({n},)"
        assert self.positions.shape == (n, 3), f"positions shape {self.positions.shape} != ({n}, 3)"
        assert self.velocities.shape == (n, 3), f"velocities shape {self.velocities.shape} != ({n}, 3)"
        if self.masses_kg is not None:
            assert self.masses_kg.shape == (n,), f"masses_kg shape {self.masses_kg.shape} != ({n},)"

    @property
    def n(self) -> int:
        """Number of bodies."""
        return len(self.names)
