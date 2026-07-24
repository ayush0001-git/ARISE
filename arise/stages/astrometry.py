"""Stage 6 -- astrometric calibration (world coordinate system).

Preference order:
1. a valid WCS already in the header (telescope pointing model), optionally
   *refined* by matching detected stars to a reference catalog and correcting
   the CRVAL offset;
2. a blind/bounded plate solve via astroquery.astrometry_net (needs network +
   API key), used only when no header WCS exists.

A Gaia/reference **residual-RMS sanity gate** is always computed and reported,
so a confident-but-wrong solution is caught rather than trusted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from ..catalogs import crossmatch
from ..config import AstrometryConfig, Instrument
from ..fitsio import FrameMeta
from ..logs import get_logger

log = get_logger("astrometry")


@dataclass
class AstrometryResult:
    wcs: WCS | None
    solved: bool
    method: str
    n_matched: int = 0
    residual_rms_arcsec: float = float("nan")
    refined: bool = False


def _header_has_wcs(header) -> bool:
    try:
        w = WCS(header)
        return bool(w.has_celestial)
    except Exception:
        return False


def solve_astrometry(meta: FrameMeta, catalog: pd.DataFrame, header,
                     inst: Instrument, acfg: AstrometryConfig,
                     reference: pd.DataFrame | None = None) -> AstrometryResult:
    """Return a validated WCS for the frame and its astrometric QA."""
    if not acfg.enabled:
        return AstrometryResult(wcs=None, solved=False, method="disabled")

    wcs = None
    method = "none"
    if acfg.use_header_wcs and _header_has_wcs(header):
        wcs = WCS(header)
        method = "header"
    elif acfg.solver in ("astrometry_net", "auto"):
        wcs = _solve_astrometry_net(meta, catalog, inst, acfg)
        method = "astrometry_net"

    if wcs is None:
        log.warning("No WCS for %s (no header WCS, solver unavailable)", meta.path.name)
        return AstrometryResult(wcs=None, solved=False, method=method)

    res = AstrometryResult(wcs=wcs, solved=True, method=method)

    # --- refine + validate against a reference catalog ------------------- #
    if reference is not None and len(reference) and {"x", "y"}.issubset(catalog.columns):
        res = _refine_and_validate(wcs, catalog, reference, meta, method)
    log.info("Astrometry %s: method=%s matched=%d residual RMS=%.3f arcsec%s",
             meta.path.name, res.method, res.n_matched, res.residual_rms_arcsec,
             " (refined)" if res.refined else "")
    return res


def _refine_and_validate(wcs: WCS, catalog: pd.DataFrame, reference: pd.DataFrame,
                         meta: FrameMeta, method: str) -> AstrometryResult:
    """Correct a bulk CRVAL offset and report the residual RMS vs reference."""
    x = catalog["x"].to_numpy()
    y = catalog["y"].to_numpy()
    ra, dec = wcs.all_pix2world(x, y, 0)

    idx, sep, matched = crossmatch(ra, dec, reference["ra"].to_numpy(),
                                   reference["dec"].to_numpy(), radius_arcsec=3.0)
    n = int(matched.sum())
    rms0 = float(np.sqrt(np.nanmean(sep[matched] ** 2))) if n else float("nan")
    if n < 4:
        return AstrometryResult(wcs=wcs, solved=True, method=method,
                                n_matched=n, residual_rms_arcsec=rms0)

    # median offset in degrees (robust to mismatches), applied to CRVAL;
    # wrap the RA difference into [-180, 180) so fields near RA=0 are safe
    d_ra = np.median(((reference["ra"].to_numpy()[idx[matched]] - ra[matched]
                       + 180.0) % 360.0) - 180.0)
    d_dec = np.median(reference["dec"].to_numpy()[idx[matched]] - dec[matched])
    refined = wcs.deepcopy()
    refined.wcs.crval = refined.wcs.crval + np.array([d_ra, d_dec])

    ra2, dec2 = refined.all_pix2world(x, y, 0)
    idx2, sep2, matched2 = crossmatch(ra2, dec2, reference["ra"].to_numpy(),
                                      reference["dec"].to_numpy(), radius_arcsec=3.0)
    n2 = int(matched2.sum())
    rms = float(np.sqrt(np.nanmean(sep2[matched2] ** 2))) if n2 else float("nan")
    # keep the refinement only if it does not degrade the solution (at least
    # as many matches, finite and no-worse residual RMS); otherwise roll back
    # to the original WCS and report its QA numbers
    if n2 >= n and np.isfinite(rms) and rms <= rms0:
        return AstrometryResult(wcs=refined, solved=True, method=method,
                                n_matched=n2, residual_rms_arcsec=rms, refined=True)
    log.warning("Astrometry %s: CRVAL refinement degraded the solution "
                "(matches %d -> %d, RMS %.3f -> %.3f arcsec); keeping original WCS",
                meta.path.name, n, n2, rms0, rms)
    return AstrometryResult(wcs=wcs, solved=True, method=method,
                            n_matched=n, residual_rms_arcsec=rms0, refined=False)


def _solve_astrometry_net(meta, catalog, inst, acfg):
    """Bounded plate solve from the source list via astroquery (network)."""
    try:
        from astroquery.astrometry_net import AstrometryNet
    except Exception:
        log.warning("astroquery.astrometry_net unavailable; cannot blind-solve")
        return None
    try:
        import os
        ast = AstrometryNet()
        key = os.environ.get("ASTROMETRY_NET_API_KEY", "")
        if key:
            ast.api_key = key
        else:
            log.warning("No ASTROMETRY_NET_API_KEY set (config/keys.yaml); "
                        "blind solve will fail without it")
        cat = catalog.sort_values("flux", ascending=False)
        scale_low = acfg.scale_low or inst.pixel_scale * 0.8
        scale_high = acfg.scale_high or inst.pixel_scale * 1.2
        hdr = ast.solve_from_source_list(
            cat["x"].to_numpy(), cat["y"].to_numpy(),
            meta.naxis1, meta.naxis2,
            scale_units="arcsecperpix", scale_lower=scale_low, scale_upper=scale_high,
            center_ra=meta.ra, center_dec=meta.dec,
            crpix_center=True, solve_timeout=120,
        )
        if isinstance(hdr, dict) and not hdr:
            log.warning("astrometry.net failed to solve %s", meta.path.name)
            return None
        return WCS(hdr)
    except Exception as exc:  # network/key issues
        log.warning("astrometry.net solve error for %s: %s", meta.path.name, exc)
        return None
