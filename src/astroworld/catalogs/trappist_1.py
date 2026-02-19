"""
TRAPPIST-1 exoplanetary system catalog.

Seven Earth-sized planets orbiting an ultra-cool red dwarf star.
All seven planets are in a tight resonance chain with periods from
1.5 to 18.8 days, making this the densest known planetary system.

Data sources:
  - Agol et al. (2021), PSJ, 2:1 — transit timing analysis
  - Gillon et al. (2017), Nature, 542:456 — discovery paper
  - Grimm et al. (2018), A&A, 613:A68 — refined masses

Star properties:
  - M* = 0.0898 M_sun
  - R* = 0.1192 R_sun
  - Distance: 12.43 pc (40.5 ly)

Simulation notes:
  - GM_star in AU³/day² (converted from solar units)
  - Orbital elements referenced to J2000 ecliptic (approximate)
  - Inclinations are nearly edge-on (~89-90°) but we use coplanar
    as first approximation for stability studies
"""

import numpy as np
from typing import Optional

from astroworld.constants import GM_SUN, AU
from astroworld.data.state import SystemState
from astroworld.data.keplerian import keplerian_to_cartesian

# TRAPPIST-1 stellar GM (0.0898 solar masses)
GM_TRAPPIST1 = GM_SUN * 0.0898

# Planet data: (name, a_AU, e, mass_Mearth, radius_Rearth, color)
# Semi-major axes and eccentricities from Agol et al. (2021)
# Masses from Grimm et al. (2018) + Agol et al. (2021) refinements
_PLANET_DATA = [
    ("b", 0.01154,  0.00622,  1.374, 1.116, "#E74C3C"),
    ("c", 0.01580,  0.00654,  1.308, 1.097, "#E67E22"),
    ("d", 0.02227,  0.00837,  0.388, 0.788, "#3498DB"),
    ("e", 0.02925,  0.00510,  0.692, 0.920, "#27AE60"),
    ("f", 0.03849,  0.01007,  1.039, 1.045, "#9B59B6"),
    ("g", 0.04683,  0.00208,  1.321, 1.129, "#F39C12"),
    ("h", 0.06189,  0.00567,  0.326, 0.755, "#95A5A6"),
]

# Earth mass in kg and GM
_M_EARTH_KG = 5.97237e24
_GM_EARTH_SI = 3.98600435507e14  # m³/s²
_GM_TO_SIM = (86400.0**2) / (AU**3)  # conversion factor m³/s² → AU³/day²


def make_trappist1_state(
    include_planets: Optional[list[str]] = None,
    mean_anomaly_offset: float = 0.0,
) -> SystemState:
    """
    Create a SystemState for the TRAPPIST-1 system.

    Positions and velocities are generated from Keplerian elements
    (no ephemeris service needed). All planets start at their
    respective mean anomalies (spread around the orbit).

    Parameters
    ----------
    include_planets : List of planet letters to include (e.g. ["b", "c", "d"]).
                     Default: all 7 planets.
    mean_anomaly_offset : Global offset to mean anomaly (radians).
                          Useful for testing different configurations.

    Returns SystemState ready for simulation.
    """
    if include_planets is None:
        include_planets = [p[0] for p in _PLANET_DATA]

    # Filter planets
    planets = [p for p in _PLANET_DATA if p[0] in include_planets]

    n = len(planets) + 1  # +1 for star
    names = ["trappist_1"] + [f"trappist_1_{p[0]}" for p in planets]

    gm_values = np.zeros(n)
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    masses_kg = np.zeros(n)
    colors = ["#FF6B6B"] + [p[5] for p in planets]

    # Star at origin, at rest
    gm_values[0] = GM_TRAPPIST1
    masses_kg[0] = 0.0898 * 1.98892e30

    # Spread planets with initial mean anomalies based on golden angle
    # to avoid starting all at perihelion (unrealistic and numerically bad)
    golden_angle = 2 * np.pi * (1 - 1 / ((1 + np.sqrt(5)) / 2))

    for j, (letter, a, e, mass_me, radius_re, color) in enumerate(planets):
        idx = j + 1

        # GM from mass (M_earth units → SI → sim units)
        gm_si = _GM_EARTH_SI * mass_me  # approximate: GM scales with mass
        gm_values[idx] = gm_si * _GM_TO_SIM
        masses_kg[idx] = mass_me * _M_EARTH_KG

        # Keplerian → Cartesian
        # Coplanar (i=0), Ω=0, ω=0, spread mean anomalies
        M = golden_angle * j + mean_anomaly_offset
        pos, vel = keplerian_to_cartesian(
            a=a, e=e, i=0.0,
            omega_node=0.0, omega_peri=0.0,
            mean_anomaly=M,
            gm_star=GM_TRAPPIST1,
        )
        positions[idx] = pos
        velocities[idx] = vel

    return SystemState(
        names=names,
        gm_values=gm_values,
        positions=positions,
        velocities=velocities,
        masses_kg=masses_kg,
        star_index=0,
        colors=colors,
    )


def get_orbital_periods() -> dict[str, float]:
    """Return theoretical Keplerian periods for all TRAPPIST-1 planets (days)."""
    periods = {}
    for letter, a, e, *_ in _PLANET_DATA:
        T = 2 * np.pi * np.sqrt(a**3 / GM_TRAPPIST1)
        periods[letter] = T
    return periods
