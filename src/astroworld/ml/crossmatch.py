"""Catalog cross-matching against SIMBAD and Gaia DR3.

Queries astronomical databases to identify known objects among pipeline
candidates, filtering out galaxies, AGN, variable stars, and other
contaminants that masquerade as Planet 9.

Usage
-----
>>> from astroworld.ml.crossmatch import CatalogMatcher
>>> matcher = CatalogMatcher(search_radius_arcsec=10.0)
>>> match = matcher.check_simbad(69.83, 12.96)
>>> print(match)  # e.g. MatchResult(source='SIMBAD', name='...', otype='Galaxy')
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SIMBAD object types that are definite contaminants for P9 search
# ---------------------------------------------------------------------------
#   Reference: http://simbad.cds.unistra.fr/guide/otypes.htx
CONTAMINANT_TYPES = {
    # Galaxies & AGN
    "G", "GiC", "GiG", "GiP", "GNe", "GrG", "IG", "PaG", "BiC",
    "Sy1", "Sy2", "AGN", "QSO", "Bla", "BLL", "LIN", "SyG", "rG",
    "EmG", "SBG", "HII", "LSB", "bCG",
    # Stars (any type — P9 is not a star)
    "Star", "**", "EB*", "V*", "Psr", "WD*", "PM*", "HB*",
    "RG*", "sg*", "s*r", "s*y", "s*b", "AB*", "LP*", "Mi*",
    "sv*", "pA*", "WR*", "Be*", "Ce*", "RR*", "dS*", "RS*",
    "Cep", "SX*", "gD*", "CV*", "No*", "Su*", "bL*",
    "Y*O", "Ae*", "Em*", "BS*", "LM*", "HS*", "OH*",
    # Planetary nebulae / SNR
    "PN", "SNR",
    # Radio / X-ray sources (likely AGN)
    "Rad", "X", "gam",
}

# Gaia proper motion threshold (mas/yr) — objects moving faster than this
# are likely foreground stars, not Planet 9 at ~400 AU
GAIA_PM_THRESHOLD_MAS_YR = 5.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Result from a catalog cross-match query."""
    source: str              # "SIMBAD", "Gaia", "none"
    name: Optional[str] = None    # Object name (e.g. "NGC 1234")
    otype: Optional[str] = None   # Object type (e.g. "Galaxy", "V*")
    separation_arcsec: float = 0.0
    is_contaminant: bool = False
    # Gaia-specific
    pmra: Optional[float] = None        # mas/yr
    pmdec: Optional[float] = None       # mas/yr
    total_pm: Optional[float] = None    # mas/yr
    parallax: Optional[float] = None    # mas
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CatalogMatcher
# ---------------------------------------------------------------------------

class CatalogMatcher:
    """Cross-match candidate coordinates against SIMBAD and Gaia DR3.

    Parameters
    ----------
    search_radius_arcsec : float
        Cone search radius in arcseconds (default 10″).
    rate_limit_s : float
        Pause between API requests to avoid server bans.
    pm_threshold : float
        Gaia proper motion threshold (mas/yr) above which an object
        is flagged as a high-PM contaminant.
    """

    def __init__(
        self,
        search_radius_arcsec: float = 10.0,
        rate_limit_s: float = 0.2,
        pm_threshold: float = GAIA_PM_THRESHOLD_MAS_YR,
    ):
        self.radius = search_radius_arcsec * u.arcsec
        self.rate_limit_s = rate_limit_s
        self.pm_threshold = pm_threshold
        self._setup_simbad()

    def _setup_simbad(self):
        """Configure SIMBAD client with required fields."""
        self.simbad = Simbad()
        self.simbad.add_votable_fields("otype")

    # ----- SIMBAD --------------------------------------------------------

    def check_simbad(
        self, ra_deg: float, dec_deg: float,
    ) -> MatchResult:
        """Query SIMBAD for known objects near the given coordinates.

        Returns
        -------
        MatchResult
            With ``source="SIMBAD"`` if a match is found, else ``source="none"``.
        """
        coord = SkyCoord(
            ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame="icrs",
        )
        try:
            result = self.simbad.query_region(coord, radius=self.radius)
        except Exception as exc:
            logger.warning("SIMBAD query failed for (%.4f, %.4f): %s",
                           ra_deg, dec_deg, exc)
            return MatchResult(
                source="error",
                details={"error": str(exc)},
            )

        if result is None or len(result) == 0:
            return MatchResult(source="none")

        # Take closest match
        row = result[0]
        name = str(row["main_id"])
        otype = str(row["otype"])

        # Compute separation
        match_coord = SkyCoord(
            ra=float(row["ra"]), dec=float(row["dec"]),
            unit=(u.deg, u.deg), frame="icrs",
        )
        sep = coord.separation(match_coord).arcsec

        is_contaminant = otype in CONTAMINANT_TYPES

        return MatchResult(
            source="SIMBAD",
            name=name,
            otype=otype,
            separation_arcsec=sep,
            is_contaminant=is_contaminant,
        )

    # ----- Gaia ----------------------------------------------------------

    def check_gaia(
        self, ra_deg: float, dec_deg: float,
    ) -> MatchResult:
        """Query Gaia DR3 for high proper-motion stars near coordinates.

        Returns
        -------
        MatchResult
            With ``source="Gaia"`` if a high-PM match is found.
        """
        coord = SkyCoord(
            ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame="icrs",
        )
        try:
            job = Gaia.cone_search_async(
                coord, radius=self.radius,
            )
            result = job.get_results()
        except Exception as exc:
            logger.warning("Gaia query failed for (%.4f, %.4f): %s",
                           ra_deg, dec_deg, exc)
            return MatchResult(
                source="error",
                details={"error": str(exc)},
            )

        if result is None or len(result) == 0:
            return MatchResult(source="none")

        # Find the star with highest proper motion
        pmra_col = np.array(result["pmra"], dtype=float)
        pmdec_col = np.array(result["pmdec"], dtype=float)

        # Handle NaN proper motions
        valid = np.isfinite(pmra_col) & np.isfinite(pmdec_col)
        if not np.any(valid):
            return MatchResult(source="none")

        total_pm = np.sqrt(pmra_col**2 + pmdec_col**2)
        total_pm[~valid] = 0.0

        best_idx = int(np.argmax(total_pm))
        best_pm = float(total_pm[best_idx])
        best_row = result[best_idx]

        is_contaminant = best_pm > self.pm_threshold

        return MatchResult(
            source="Gaia",
            name=str(best_row["designation"]),
            otype="Star",
            separation_arcsec=0.0,  # Gaia doesn't easily give sep in this mode
            is_contaminant=is_contaminant,
            pmra=float(pmra_col[best_idx]),
            pmdec=float(pmdec_col[best_idx]),
            total_pm=best_pm,
            parallax=float(best_row["parallax"]) if np.isfinite(best_row["parallax"]) else None,
        )

    # ----- Combined query ------------------------------------------------

    def check_candidate(
        self, ra_deg: float, dec_deg: float,
    ) -> MatchResult:
        """Run SIMBAD + Gaia queries for a single candidate.

        SIMBAD is checked first. If a contaminant is found, returns immediately.
        Otherwise checks Gaia for high proper-motion stars.
        """
        # SIMBAD first
        simbad_match = self.check_simbad(ra_deg, dec_deg)
        if simbad_match.is_contaminant:
            return simbad_match

        # Gaia second
        time.sleep(self.rate_limit_s)
        gaia_match = self.check_gaia(ra_deg, dec_deg)
        if gaia_match.is_contaminant:
            return gaia_match

        # No contaminant found — return SIMBAD result (may have non-contaminant match)
        if simbad_match.source == "SIMBAD":
            return simbad_match

        return MatchResult(source="none")

    # ----- DataFrame processing ------------------------------------------

    def filter_candidates(
        self, df: pd.DataFrame, verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Cross-match all candidates in a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Must have ``ra_deg`` and ``dec_deg`` columns.
        verbose : bool
            Print progress per candidate.

        Returns
        -------
        (df_known, df_unknown) : tuple of DataFrames
            ``df_known`` — candidates matched to known objects.
            ``df_unknown`` — candidates with NO catalog match (P9 hopefuls).
        """
        n = len(df)
        catalog_source = []
        catalog_name = []
        catalog_otype = []
        catalog_sep = []
        is_contaminant = []
        gaia_pm = []

        for i, (_, row) in enumerate(df.iterrows()):
            ra, dec = float(row["ra_deg"]), float(row["dec_deg"])

            match = self.check_candidate(ra, dec)

            catalog_source.append(match.source)
            catalog_name.append(match.name or "")
            catalog_otype.append(match.otype or "")
            catalog_sep.append(match.separation_arcsec)
            is_contaminant.append(match.is_contaminant)
            gaia_pm.append(match.total_pm)

            if verbose:
                status = (
                    f"Match {match.source}: '{match.otype}' ({match.name})"
                    if match.source not in ("none", "error")
                    else "No match"
                )
                tag = "REJECT" if match.is_contaminant else "KEEP"
                print(
                    f"  [{i+1:>{len(str(n))}}/{n}] "
                    f"RA={ra:.4f} Dec={dec:+.4f} -> {status} [{tag}]"
                )

            # Rate limit
            if i < n - 1:
                time.sleep(self.rate_limit_s)

        # Add columns
        result = df.copy()
        result["catalog_source"] = catalog_source
        result["catalog_name"] = catalog_name
        result["catalog_otype"] = catalog_otype
        result["catalog_sep_arcsec"] = catalog_sep
        result["is_known_object"] = is_contaminant
        result["gaia_pm_mas_yr"] = gaia_pm

        df_known = result[result["is_known_object"]].reset_index(drop=True)
        df_unknown = result[~result["is_known_object"]].reset_index(drop=True)

        return df_known, df_unknown
