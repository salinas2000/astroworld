"""Tests for physical constants and body catalog."""

import numpy as np
from astroworld.constants import G, GM_SUN, AU, DAY, BODIES, _gm_to_sim


def test_gm_sun_value():
    """GM_SUN in AU^3/day^2 should be approximately 2.9591e-4."""
    assert abs(GM_SUN - 2.9591e-4) / 2.9591e-4 < 0.01  # within 1%


def test_unit_conversion_roundtrip():
    """Converting GM_SI -> sim units -> back to SI should be consistent."""
    gm_si = 3.98600435507e14  # Earth GM in m^3/s^2
    gm_sim = _gm_to_sim(gm_si)
    gm_back = gm_sim * (AU**3) / (DAY**2)
    assert abs(gm_back - gm_si) / gm_si < 1e-10


def test_all_bodies_have_valid_fields():
    """Every body in catalog should have positive GM and mass."""
    for name, body in BODIES.items():
        assert body.gm > 0, f"{name} has non-positive GM"
        assert body.mass_kg > 0, f"{name} has non-positive mass"
        assert body.jpl_id > 0, f"{name} has invalid JPL ID"
        assert body.radius_km > 0, f"{name} has non-positive radius"


def test_gm_ordering():
    """Sun should have largest GM, then Jupiter, Saturn, etc."""
    assert BODIES["sun"].gm > BODIES["jupiter"].gm
    assert BODIES["jupiter"].gm > BODIES["saturn"].gm
    assert BODIES["saturn"].gm > BODIES["neptune"].gm
