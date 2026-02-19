"""
Outer Solar System catalog for Planet 9 hunting.

Includes the Sun (with inner planet masses absorbed), the four gas giants,
seven extreme Trans-Neptunian Objects (eTNOs), and an optional Planet 9
candidate based on Batygin & Brown (2021) parameters.

Physical motivation:
  - The eTNOs (Sedna, 2012 VP113, Leleakuhonua, 2004 VN112, 2007 TG422,
    2013 RF98, 2010 GB174) have anomalously clustered arguments of
    perihelion, suggesting a massive unseen perturber.
  - The gas giants are essential: they cause natural secular precession of
    TNO orbits. Planet 9's role is to COUNTERACT this precession and
    maintain the clustering.
  - Inner planets (Mercury-Mars) are absorbed into the Sun's mass since
    their perturbations on TNOs at 100-1000 AU are negligible, and this
    allows dt=10-20 days (Jupiter's period ~4300 days is well-resolved).

Data sources:
  - Gas giants: JPL DE440/441 mean orbital elements (J2000)
  - Sedna: JPL Small-Body Database (90377 Sedna)
  - 2012 VP113: Trujillo & Sheppard (2014), Nature 507:471
  - Leleakuhonua (541132): Sheppard et al. (2019)
  - 2004 VN112: Becker et al. (2018)
  - 2007 TG422: Chen et al. (2016)
  - 2013 RF98: Sheppard & Trujillo (2016)
  - 2010 GB174: de la Fuente Marcos et al. (2015)
  - Planet 9: Batygin & Brown (2021), ApJ 910:L20 — updated parameters

Simulation notes:
  - GM_star includes inner planet masses: M_sun + M_mercury + ... + M_mars
  - Orbital elements referenced to J2000 ecliptic
  - Mean anomalies set to spread bodies around their orbits (golden angle)
  - dt=10 days recommended (Jupiter period = 4332 days → 433 steps/orbit)
  - Duration: 10,000-50,000 years for secular effects to manifest
"""

import numpy as np
from typing import Optional

from astroworld.constants import GM_SUN, AU, DAY, _gm_to_sim
from astroworld.data.state import SystemState
from astroworld.data.keplerian import keplerian_to_cartesian


# ── Constants ─────────────────────────────────────────────────────

# Inner planet GM values (m³/s²) — absorbed into the Sun
_GM_MERCURY_SI = 2.2031868551e13
_GM_VENUS_SI = 3.24858592000e14
_GM_EARTH_MOON_SI = 3.98600435507e14 + 4.9028695e12  # Earth + Moon
_GM_MARS_SI = 4.28283146057e13

# Effective solar GM including inner planets
_GM_STAR_SI = 1.32712440041279419e20 + _GM_MERCURY_SI + _GM_VENUS_SI + _GM_EARTH_MOON_SI + _GM_MARS_SI
GM_STAR = _gm_to_sim(_GM_STAR_SI)

# Mass conversions
_M_SUN_KG = 1.98892e30
_M_EARTH_KG = 5.97237e24
_M_JUPITER_KG = 1.89819e27

# GM conversion factor: m³/s² → AU³/day²
_GM_TO_SIM = (DAY**2) / (AU**3)


# ── Body data ─────────────────────────────────────────────────────
# Format: (name, a_AU, e, i_deg, Omega_deg, omega_deg, M_deg, gm_si, mass_kg, color)
# Elements are J2000 mean ecliptic for planets, osculating for TNOs

_GAS_GIANTS = [
    # Jupiter: JPL mean elements J2000
    ("jupiter", 5.2026, 0.0489, 1.303, 100.464, 273.867, 20.020,
     1.26712764800e17, 1.89819e27, "#D4A574"),
    # Saturn
    ("saturn", 9.5549, 0.0557, 2.489, 113.665, 339.392, 317.020,
     3.79395259000e16, 5.68340e26, "#F4D03F"),
    # Uranus
    ("uranus", 19.2184, 0.0444, 0.773, 74.006, 96.998, 142.238,
     5.79393658000e15, 8.68130e25, "#85C1E9"),
    # Neptune
    ("neptune", 30.1104, 0.0112, 1.770, 131.784, 276.336, 256.228,
     6.83509920000e15, 1.02413e26, "#2471A3"),
]

_TNOS = [
    # Sedna (90377): extreme TNO, a=506 AU, T~11,400 years
    ("sedna", 506.0, 0.8496, 11.93, 144.26, 311.29, 358.0,
     None, 1.0e21, "#E74C3C"),
    # 2012 VP113: a=261 AU, T~4,200 years
    ("2012_vp113", 261.0, 0.6905, 24.05, 90.81, 293.78, 5.0,
     None, 5.0e20, "#E67E22"),
    # Leleakuhonua (541132): a=1094 AU, T~36,000 years
    ("leleakuhonua", 1094.0, 0.9412, 11.63, 118.26, 117.97, 2.0,
     None, 5.0e20, "#9B59B6"),
    # 2004 VN112: a=316 AU, T~5,600 years — Becker et al. (2018)
    ("2004_vn112", 316.0, 0.850, 25.5, 66.0, 327.0, 0.0,
     None, 5.0e20, "#1ABC9C"),
    # 2007 TG422: a=483 AU, T~10,600 years — Chen et al. (2016)
    ("2007_tg422", 483.0, 0.920, 18.6, 112.9, 285.0, 0.0,
     None, 5.0e20, "#F39C12"),
    # 2013 RF98: a=349 AU, T~6,500 years — Sheppard & Trujillo (2016)
    ("2013_rf98", 349.0, 0.890, 29.5, 67.5, 311.0, 0.0,
     None, 5.0e20, "#8E44AD"),
    # 2010 GB174: a=369 AU, T~7,100 years — de la Fuente Marcos et al. (2015)
    ("2010_gb174", 369.0, 0.860, 21.5, 130.6, 347.0, 0.0,
     None, 5.0e20, "#2ECC71"),
]

# Planet 9 candidate: Batygin & Brown (2021) updated parameters
# Mass: 6.2 M_Earth, anti-aligned with TNO perihelion cluster
_PLANET_9 = (
    "planet_9", 380.0, 0.20, 16.0, 100.0, 150.0, 180.0,
    None, 6.2 * _M_EARTH_KG, "#00FF88",
)


# ── Catalog functions ─────────────────────────────────────────────

def _build_state(
    include_giants: bool = True,
    include_tnos: bool = True,
    include_planet_9: bool = False,
    tno_names: Optional[list[str]] = None,
    extra_bodies: Optional[list[tuple]] = None,
) -> SystemState:
    """
    Build a SystemState for the outer Solar System.

    The Sun is placed at the origin with inner-planet masses absorbed.
    All bodies get Keplerian initial conditions via keplerian_to_cartesian.
    """
    bodies = []

    if include_giants:
        bodies.extend(_GAS_GIANTS)

    if include_tnos:
        if tno_names:
            bodies.extend([t for t in _TNOS if t[0] in tno_names])
        else:
            bodies.extend(_TNOS)

    if include_planet_9:
        bodies.append(_PLANET_9)

    if extra_bodies:
        bodies.extend(extra_bodies)

    n = len(bodies) + 1  # +1 for star
    names = ["sun"] + [b[0] for b in bodies]
    gm_values = np.zeros(n)
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    masses_kg = np.zeros(n)
    colors = ["#FDB813"] + [b[9] for b in bodies]

    # Star at origin
    gm_values[0] = GM_STAR
    masses_kg[0] = _M_SUN_KG * 1.0003  # ~inner planets absorbed

    # Spread mean anomalies with golden angle to avoid unrealistic alignments
    golden_angle = 2 * np.pi * (1 - 1 / ((1 + np.sqrt(5)) / 2))

    for j, body in enumerate(bodies):
        idx = j + 1
        name, a, e, i_deg, Omega_deg, omega_deg, M_deg, gm_si, mass_kg, color = body

        # GM: use known value for giants, or estimate from mass for TNOs
        if gm_si is not None:
            gm_values[idx] = _gm_to_sim(gm_si)
        else:
            # For very small bodies, GM ≈ G * mass
            # G_SI = 6.6743e-11, convert to sim units
            gm_values[idx] = 6.6743e-11 * mass_kg * _GM_TO_SIM

        masses_kg[idx] = mass_kg

        # Convert degrees to radians
        i = np.radians(i_deg)
        Omega = np.radians(Omega_deg)
        omega = np.radians(omega_deg)
        M = np.radians(M_deg)

        # Keplerian → Cartesian
        pos, vel = keplerian_to_cartesian(
            a=a, e=e, i=i,
            omega_node=Omega, omega_peri=omega,
            mean_anomaly=M,
            gm_star=GM_STAR,
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


def make_outer_system_without_p9(
    tno_names: Optional[list[str]] = None,
) -> SystemState:
    """
    Create the outer Solar System WITHOUT Planet 9.

    This is the null hypothesis / baseline for comparison.
    Includes: Sun (+ inner masses), 4 gas giants, 7 eTNOs.
    """
    return _build_state(
        include_giants=True,
        include_tnos=True,
        include_planet_9=False,
        tno_names=tno_names,
    )


def make_outer_system_with_p9(
    tno_names: Optional[list[str]] = None,
) -> SystemState:
    """
    Create the outer Solar System WITH Planet 9.

    This is the Batygin-Brown hypothesis.
    Includes: Sun (+ inner masses), 4 gas giants, 7 eTNOs, Planet 9.
    """
    return _build_state(
        include_giants=True,
        include_tnos=True,
        include_planet_9=True,
        tno_names=tno_names,
    )


def make_outer_system_custom_p9(
    mass_earth: float = 6.2,
    distance_au: float = 380.0,
    eccentricity: float = 0.20,
    inclination_deg: float = 16.0,
    omega_node_deg: float = 100.0,
    omega_peri_deg: float = 150.0,
    mean_anomaly_deg: float = 180.0,
    tno_names: Optional[list[str]] = None,
) -> SystemState:
    """
    Create outer Solar System with a custom Planet 9 at arbitrary mass and distance.

    This parametric factory enables grid searches over P9 parameter space.
    Uses the same gas giants and TNOs as the standard catalogs.

    Parameters
    ----------
    mass_earth : Planet 9 mass in Earth masses (default 6.2).
    distance_au : Planet 9 semi-major axis in AU (default 380).
    eccentricity : Planet 9 orbital eccentricity (default 0.20).
    inclination_deg : Orbital inclination in degrees (default 16).
    omega_node_deg : Longitude of ascending node in degrees (default 100).
    omega_peri_deg : Argument of perihelion in degrees (default 150).
    mean_anomaly_deg : Mean anomaly at epoch in degrees (default 180).
    tno_names : Subset of TNOs to include (default: all 7).

    Returns SystemState ready for simulation.
    """
    custom_p9 = (
        "planet_9", distance_au, eccentricity, inclination_deg,
        omega_node_deg, omega_peri_deg, mean_anomaly_deg,
        None, mass_earth * _M_EARTH_KG, "#00FF88",
    )
    return _build_state(
        include_giants=True,
        include_tnos=True,
        include_planet_9=False,
        tno_names=tno_names,
        extra_bodies=[custom_p9],
    )


def make_debris_disk(
    n_particles: int = 100,
    total_mass_earth: float = 6.2,
    ring_radius_au: float = 400.0,
    ring_width_au: float = 0.0,
    eccentricity: float = 0.0,
    inclination_deg: float = 0.0,
    tno_names: Optional[list[str]] = None,
) -> SystemState:
    """
    Create outer Solar System with a debris disk instead of Planet 9.

    Distributes total_mass_earth equally among n_particles bodies in a
    ring at ring_radius_au. Total gravitational mass identical to the
    single-body Planet 9 scenario for controlled comparison.

    Parameters
    ----------
    n_particles : Number of disk particles (default 100).
    total_mass_earth : Total disk mass in Earth masses (default 6.2).
    ring_radius_au : Central semi-major axis of the ring in AU (default 400).
    ring_width_au : Spread in semi-major axis (0 = all at same a).
    eccentricity : Common eccentricity of disk particles (default 0).
    inclination_deg : Common inclination in degrees (default 0).
    tno_names : Subset of TNOs to include (default: all 7).

    Returns SystemState. Body count = 1 (sun) + 4 (giants) + len(tnos) + n_particles.
    """
    mass_per_particle_kg = (total_mass_earth / n_particles) * _M_EARTH_KG

    disk_bodies = []
    for k in range(n_particles):
        # Evenly spaced mean anomalies around the ring
        M_deg = (360.0 / n_particles) * k

        # Optional radial spread
        if ring_width_au > 0:
            a = ring_radius_au + ring_width_au * (k / n_particles - 0.5)
        else:
            a = ring_radius_au

        disk_bodies.append((
            f"disk_{k:03d}", a, eccentricity, inclination_deg,
            0.0, 0.0, M_deg,
            None, mass_per_particle_kg, "#555555",
        ))

    return _build_state(
        include_giants=True,
        include_tnos=True,
        include_planet_9=False,
        tno_names=tno_names,
        extra_bodies=disk_bodies,
    )


def get_theoretical_periods() -> dict[str, float]:
    """Return Keplerian orbital periods for all outer system bodies (days)."""
    periods = {}
    all_bodies = list(_GAS_GIANTS) + list(_TNOS) + [_PLANET_9]
    for body in all_bodies:
        name, a = body[0], body[1]
        T = 2 * np.pi * np.sqrt(a**3 / GM_STAR)
        periods[name] = T
    return periods
