"""Reference-catalog access and sky cross-matching.

Reference stars come from either a local CSV (offline / user-supplied / the
synthetic demo's ``reference_catalog.csv``) or an online query to Gaia DR3 /
Pan-STARRS via astroquery. Cross-matching uses astropy ``SkyCoord`` nearest
neighbours. Everything degrades gracefully: no network -> local CSV -> empty.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .logs import get_logger

log = get_logger("catalogs")


# --------------------------------------------------------------------------- #
# loading reference stars
# --------------------------------------------------------------------------- #
def load_local_reference(path: str | Path) -> Optional[pd.DataFrame]:
    """Load a CSV with at least ra, dec (deg) and optional mag columns."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "ra" not in cols or "dec" not in cols:
        log.warning("Local reference %s lacks ra/dec columns", path.name)
        return None
    out = pd.DataFrame({"ra": df[cols["ra"]].astype(float),
                        "dec": df[cols["dec"]].astype(float)})
    out["mag"] = df[cols["mag"]].astype(float) if "mag" in cols else np.nan
    log.info("Loaded %d reference stars from local file %s", len(out), path.name)
    return out


def query_gaia(ra_deg: float, dec_deg: float, radius_deg: float,
               max_mag: float = 20.5) -> Optional[pd.DataFrame]:
    """Cone-search Gaia DR3. Returns ra/dec/mag or None if the query fails."""
    try:
        from astroquery.gaia import Gaia
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        Gaia.ROW_LIMIT = 50000
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        job = Gaia.cone_search_async(coord, radius=radius_deg * u.deg)
        tbl = job.get_results()
        df = pd.DataFrame({
            "ra": np.asarray(tbl["ra"], float),
            "dec": np.asarray(tbl["dec"], float),
            "mag": np.asarray(tbl["phot_g_mean_mag"], float),
        }).dropna(subset=["ra", "dec"])
        df = df[df["mag"] < max_mag]
        log.info("Gaia DR3 returned %d stars within %.3f deg", len(df), radius_deg)
        return df.reset_index(drop=True)
    except Exception as exc:  # network/astroquery unavailable
        log.warning("Gaia query failed (%s); will use local/none", exc)
        return None


def query_panstarrs(ra_deg: float, dec_deg: float, radius_deg: float,
                    max_mag: float = 21.0) -> Optional[pd.DataFrame]:
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        v = Vizier(columns=["RAJ2000", "DEJ2000", "rmag"], row_limit=50000)
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        res = v.query_region(coord, radius=radius_deg * u.deg, catalog="II/349/ps1")
        if not res:
            return None
        tbl = res[0]
        df = pd.DataFrame({
            "ra": np.asarray(tbl["RAJ2000"], float),
            "dec": np.asarray(tbl["DEJ2000"], float),
            "mag": np.asarray(tbl["rmag"], float),
        }).dropna(subset=["ra", "dec"])
        df = df[df["mag"] < max_mag]
        log.info("Pan-STARRS returned %d stars", len(df))
        return df.reset_index(drop=True)
    except Exception as exc:
        log.warning("Pan-STARRS query failed (%s)", exc)
        return None


def get_reference_catalog(source: str, ra_deg: float, dec_deg: float,
                          radius_deg: float, local_path: str | Path | None = None,
                          max_mag: float = 21.0) -> pd.DataFrame:
    """Return a reference catalog DataFrame (ra, dec, mag).

    Resolution order: an explicit local CSV always wins (reproducible/offline);
    otherwise the named online source; otherwise an empty frame.
    """
    if local_path:
        df = load_local_reference(local_path)
        if df is not None:
            return df
    src = (source or "none").lower()
    if src in ("gaia", "auto"):
        df = query_gaia(ra_deg, dec_deg, radius_deg, max_mag)
        if df is not None and len(df):
            return df
    if src in ("panstarrs", "ps1", "auto"):
        df = query_panstarrs(ra_deg, dec_deg, radius_deg, max_mag)
        if df is not None and len(df):
            return df
    log.warning("No reference catalog available (source=%s); unmatched-flagging "
                "and photometric calibration will be limited.", source)
    return pd.DataFrame({"ra": [], "dec": [], "mag": []})


# --------------------------------------------------------------------------- #
# cross-matching
# --------------------------------------------------------------------------- #
def crossmatch(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray,
               radius_arcsec: float):
    """Nearest-neighbour match of set 1 into set 2.

    Returns (idx, sep_arcsec, matched) where ``idx[i]`` is the index in set 2
    of the nearest neighbour to source ``i``, ``sep_arcsec[i]`` its separation,
    and ``matched[i]`` whether that separation is within ``radius_arcsec``.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra1 = np.asarray(ra1, float)
    dec1 = np.asarray(dec1, float)
    ra2 = np.asarray(ra2, float)
    dec2 = np.asarray(dec2, float)
    n1 = len(ra1)
    idx = np.full(n1, -1, int)
    sep_arcsec = np.full(n1, np.inf)
    matched = np.zeros(n1, bool)
    if n1 == 0 or len(ra2) == 0:
        return (idx, sep_arcsec, matched)
    # astropy's match_to_catalog_sky raises on NaN coordinates (e.g. sources
    # without a WCS solution), so match only the finite subsets and map the
    # results back onto full-length arrays (non-finite rows stay unmatched)
    finite1 = np.isfinite(ra1) & np.isfinite(dec1)
    keep2 = np.flatnonzero(np.isfinite(ra2) & np.isfinite(dec2))
    if not finite1.any() or keep2.size == 0:
        return (idx, sep_arcsec, matched)
    c1 = SkyCoord(ra1[finite1] * u.deg, dec1[finite1] * u.deg)
    c2 = SkyCoord(ra2[keep2] * u.deg, dec2[keep2] * u.deg)
    idx_sub, sep2d, _ = c1.match_to_catalog_sky(c2)
    sep_sub = np.asarray(sep2d.arcsec, float)
    idx[finite1] = keep2[np.asarray(idx_sub, int)]   # back to original set-2 indices
    sep_arcsec[finite1] = sep_sub
    matched[finite1] = sep_sub <= radius_arcsec
    return idx, sep_arcsec, matched
