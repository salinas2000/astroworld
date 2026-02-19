"""
Physical constants and Solar System body catalog.

All simulation values use AU and days:
  - Distances: AU  (1 AU = 1.495978707e11 m)
  - Time: days     (1 day = 86400 s)
  - GM: AU^3 / day^2

This avoids floating-point issues from extremely large SI values.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

# --- Fundamental Constants ---
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
AU = 1.495978707e11  # meters per AU
DAY = 86400.0  # seconds per day

# Gravitational constant in simulation units: AU^3 / (kg * day^2)
G = G_SI * (DAY**2) / (AU**3)


def _gm_to_sim(gm_si: float) -> float:
    """Convert GM from m^3/s^2 to AU^3/day^2."""
    return gm_si * (DAY**2) / (AU**3)


# --- Standard Gravitational Parameters ---
# Source: JPL DE440/441
GM_SUN_SI = 1.32712440041279419e20  # m^3/s^2
GM_SUN = _gm_to_sim(GM_SUN_SI)


@dataclass(frozen=True)
class Body:
    """Immutable descriptor for a Solar System body."""

    name: str
    jpl_id: int  # JPL Horizons ID
    gm: float  # GM in AU^3/day^2
    mass_kg: float
    color: str
    radius_km: float


BODIES: Dict[str, Body] = {
    "sun": Body(
        "Sun", 10, GM_SUN, 1.98892e30, "#FDB813", 695700.0
    ),
    "mercury": Body(
        "Mercury", 199, _gm_to_sim(2.2031868551e13), 3.30110e23, "#8C8C8C", 2439.7
    ),
    "venus": Body(
        "Venus", 299, _gm_to_sim(3.24858592000e14), 4.86747e24, "#E6C35C", 6051.8
    ),
    # Earth-Moon Barycenter: uses combined GM (Earth+Moon) and Horizons ID=3
    # Since we don't simulate the Moon separately, we treat the system as
    # a single body at the barycenter. This eliminates ~700,000 km/yr error.
    "earth": Body(
        "Earth", 3, _gm_to_sim(3.98600435507e14 + 4.9028695e12), 5.97237e24 + 7.342e22, "#2E86C1", 6371.0
    ),
    "mars": Body(
        "Mars", 499, _gm_to_sim(4.28283146057e13), 6.41710e23, "#C0392B", 3389.5
    ),
    "jupiter": Body(
        "Jupiter", 599, _gm_to_sim(1.26712764800e17), 1.89819e27, "#D4A574", 69911.0
    ),
    "saturn": Body(
        "Saturn", 699, _gm_to_sim(3.79395259000e16), 5.68340e26, "#F4D03F", 58232.0
    ),
    "uranus": Body(
        "Uranus", 799, _gm_to_sim(5.79393658000e15), 8.68130e25, "#85C1E9", 25362.0
    ),
    "neptune": Body(
        "Neptune", 899, _gm_to_sim(6.83509920000e15), 1.02413e26, "#2471A3", 24622.0
    ),
    "pluto": Body(
        "Pluto", 999, _gm_to_sim(8.696e11), 1.303e22, "#C4A882", 1188.3
    ),
    # --- Earth and Moon as separate bodies (for split mode) ---
    # Earth geocenter (ID=399) with only Earth's GM
    "earth_399": Body(
        "Earth", 399, _gm_to_sim(3.98600435507e14), 5.97237e24, "#2E86C1", 6371.0
    ),
    # Moon (ID=301)
    "moon": Body(
        "Moon", 301, _gm_to_sim(4.9028695e12), 7.342e22, "#999999", 1737.4
    ),
}

# Speed of light in simulation units (AU/day)
# c = 299792.458 km/s * 86400 s/day / 149597870.7 km/AU
C_LIGHT = 173.14463267424034

# Default: full Solar System (Earth as EMB, no separate Moon)
DEFAULT_BODIES = [
    "sun", "mercury", "venus", "earth", "mars",
    "jupiter", "saturn", "uranus", "neptune", "pluto",
]

# Split mode: Earth and Moon as separate bodies
DEFAULT_BODIES_SPLIT = [
    "sun", "mercury", "venus", "earth_399", "moon", "mars",
    "jupiter", "saturn", "uranus", "neptune", "pluto",
]
