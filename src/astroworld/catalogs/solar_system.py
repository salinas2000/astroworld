"""
Solar System catalog.

Body data sourced from JPL DE440/441. All GM values converted to
simulation units (AU³/day²) via constants._gm_to_sim().

This module re-exports the existing BODIES dict and DEFAULT_BODIES lists
from constants.py for backward compatibility, and adds a function to
create a SystemState from JPL Horizons initial conditions.
"""

import numpy as np
from typing import List, Optional

from astroworld.constants import BODIES, DEFAULT_BODIES, DEFAULT_BODIES_SPLIT, Body
from astroworld.data.state import SystemState
from astroworld.data.horizons import fetch_initial_conditions


def make_solar_system_state(
    body_names: Optional[List[str]] = None,
    epoch_jd: float = 2451545.0,
) -> SystemState:
    """
    Create a SystemState for the Solar System by fetching from JPL Horizons.

    Parameters
    ----------
    body_names : List of body keys from BODIES dict.
                 Default: DEFAULT_BODIES (all planets as EMB).
    epoch_jd : Julian Date for initial conditions (default: J2000.0).

    Returns SystemState ready for simulation.
    """
    if body_names is None:
        body_names = DEFAULT_BODIES

    bodies = [BODIES[name] for name in body_names]
    body_ids = [b.jpl_id for b in bodies]

    ic = fetch_initial_conditions(body_ids, epoch_jd=epoch_jd)

    n = len(bodies)
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    for i, body in enumerate(bodies):
        data = ic[body.jpl_id]
        positions[i] = data["r"]
        velocities[i] = data["v"]

    # Determine star index
    star_index = body_names.index("sun") if "sun" in body_names else 0

    return SystemState(
        names=body_names,
        gm_values=np.array([b.gm for b in bodies]),
        positions=positions,
        velocities=velocities,
        masses_kg=np.array([b.mass_kg for b in bodies]),
        star_index=star_index,
        colors=[b.color for b in bodies],
    )
