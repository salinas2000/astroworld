"""
Kinematic validation for Planet 9 candidate detections.

Provides two complementary filters for SpectralSearchResult.patch_detections:

1. **Patch-level displacement filter** — Rejects detections whose apparent
   motion (in pixels within a patch) is outside the plausible range for a
   P9-like body.  Static artifacts (0 motion) and very fast movers (high
   proper motion stars) are rejected.

2. **Orbital mechanics calculator** — Computes the expected proper motion
   for a body at a given heliocentric distance, useful for contextualizing
   detected displacements.

Practical context
-----------------
For a 64x64 patch-based search on DSS2 (1993) vs WISE (2010) images:

- The model is trained on displacements of ~6-8 pixels.
- The maximum detectable displacement within a patch is ~patch_size/2.
- A P9-like body at 400 AU has orbital proper motion ~162 arcsec/yr,
  which over 17 years is ~2753 arcsec = ~1620 pixels at 1.7 arcsec/px.
  This is far larger than any single-patch field of view.
- However, the SSTF model detects motion *within individual patches*,
  where the apparent shift is much smaller (a few pixels).

The practical kinematic filter therefore operates on **measured patch
displacement** (not orbital velocity), rejecting:
  - Static artifacts: displacement < min_displacement_px
  - Fast-moving stars: displacement > max_displacement_px

Default limits are calibrated to the training data range (2-30 px).
"""

from __future__ import annotations

import math

from astroworld.ml.inference import PatchDetection


# ---------------------------------------------------------------------------
# Physical constants for orbital mechanics
# ---------------------------------------------------------------------------

#: Reference orbital velocity at 1 AU (km/s).
V_CIRC_1AU_KMS = 29.78  # Earth's orbital velocity

#: Conversion: 1 AU in km.
AU_KM = 1.496e8

#: Arcseconds per radian.
ARCSEC_PER_RAD = 206265.0

#: Seconds per year (Julian).
SEC_PER_YEAR = 365.25 * 86400.0


# ---------------------------------------------------------------------------
# Default patch displacement limits
# ---------------------------------------------------------------------------

#: Minimum displacement to reject static artifacts (pixels).
DEFAULT_MIN_DISPLACEMENT_PX = 0.5

#: Maximum displacement within a patch for a plausible detection (pixels).
#: Calibrated to ~3x the training motion range (training uses ~8 px).
#: Stars with proper motion large enough to shift >30 px between DSS2 and
#: WISE epochs are almost certainly foreground objects, not TNOs.
DEFAULT_MAX_DISPLACEMENT_PX = 30.0


# ---------------------------------------------------------------------------
# Orbital mechanics
# ---------------------------------------------------------------------------

def orbital_proper_motion(distance_au: float) -> float:
    """Annual proper motion for a body on a circular orbit.

    Uses the vis-viva approximation for circular orbits:
        v_transverse = 29.78 / sqrt(d)  km/s
        mu = v_t / d  [angular velocity in arcsec/yr]

    Parameters
    ----------
    distance_au : Heliocentric distance in AU.

    Returns
    -------
    Proper motion in arcsec/year.

    Examples
    --------
    >>> orbital_proper_motion(400.0)  # P9 at 400 AU
    161.97...
    >>> orbital_proper_motion(1.0)    # Earth's orbit
    6283847...  # ~6.3 million arcsec/yr (not physical — Earth moves fast!)
    """
    if distance_au <= 0:
        return 0.0

    v_kms = V_CIRC_1AU_KMS / math.sqrt(distance_au)
    d_km = distance_au * AU_KM
    mu_rad_per_sec = (v_kms * 1e3) / (d_km * 1e3)  # m/s / m
    return mu_rad_per_sec * ARCSEC_PER_RAD * SEC_PER_YEAR


def max_displacement_arcsec(
    delta_t_years: float,
    distance_au: float = 400.0,
) -> float:
    """Total angular displacement over *delta_t_years* at *distance_au*.

    Parameters
    ----------
    delta_t_years : Time baseline between epochs (years).
    distance_au : Heliocentric distance (AU).

    Returns
    -------
    Total angular displacement in arcseconds.
    """
    return orbital_proper_motion(distance_au) * delta_t_years


# ---------------------------------------------------------------------------
# Patch-level kinematic filter
# ---------------------------------------------------------------------------

def is_kinematically_valid(
    displacement_px: float,
    min_displacement_px: float = DEFAULT_MIN_DISPLACEMENT_PX,
    max_displacement_px: float = DEFAULT_MAX_DISPLACEMENT_PX,
) -> bool:
    """Check if a patch-level displacement is plausible for P9 detection.

    A detection is valid if::

        min_displacement_px  <=  displacement_px  <=  max_displacement_px

    - Below minimum: static artifact or noise (no real motion).
    - Above maximum: likely a high-proper-motion foreground star,
      not a distant TNO.

    The defaults (0.5 to 30 px) are calibrated for a 64x64 patch
    with training motion of ~8 px on DSS2 (1.7 arcsec/px) images.

    Parameters
    ----------
    displacement_px : Observed pixel displacement between epochs.
    min_displacement_px : Minimum displacement to reject static sources.
    max_displacement_px : Maximum displacement to reject fast stars.

    Returns
    -------
    True if displacement is within the valid range.
    """
    return min_displacement_px <= displacement_px <= max_displacement_px


def filter_detections_kinematic(
    detections: list[PatchDetection],
    min_displacement_px: float = DEFAULT_MIN_DISPLACEMENT_PX,
    max_displacement_px: float = DEFAULT_MAX_DISPLACEMENT_PX,
    displacement_estimates: dict[int, float] | None = None,
) -> list[PatchDetection]:
    """Filter patch detections by kinematic plausibility.

    For each detection, checks whether its estimated displacement is
    within the valid range for a P9-like body.

    Displacement is provided via *displacement_estimates* — a mapping
    from detection index to measured displacement in pixels.  If not
    provided, all detections are returned (conservative: no filter
    applied without displacement data).

    Parameters
    ----------
    detections : PatchDetection list from SpectralSearchResult.
    min_displacement_px : Minimum displacement (reject static).
    max_displacement_px : Maximum displacement (reject fast stars).
    displacement_estimates : Dict mapping detection index to measured
        displacement in pixels.  If None, all detections returned.

    Returns
    -------
    List of kinematically valid detections.
    """
    if displacement_estimates is None:
        return list(detections)

    valid = []
    for idx, det in enumerate(detections):
        disp = displacement_estimates.get(idx)
        if disp is None:
            # No displacement measurement -> keep (conservative)
            valid.append(det)
            continue

        if is_kinematically_valid(
            displacement_px=disp,
            min_displacement_px=min_displacement_px,
            max_displacement_px=max_displacement_px,
        ):
            valid.append(det)

    return valid
