"""Stage 7 -- photometric calibration (instrumental -> calibrated magnitudes).

Cross-match extracted sources to a reference catalog, derive a per-frame zero
point from clean (unsaturated, high-SNR, isolated) calibration stars via a
sigma-clipped fit, and write calibrated magnitudes + errors back into the
catalog. The in-frame zero point already absorbs atmospheric extinction at the
frame's airmass; airmass and the filter's extinction prior are recorded for QA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats

from ..catalogs import crossmatch
from ..config import Instrument, PhotometryConfig
from ..fitsio import FrameMeta
from ..logs import get_logger

log = get_logger("photometry")

# first-order atmospheric extinction priors for a good high-altitude site
# (mag / airmass) -- Devasthal/Manora order (cf. Kumar et al. 2022).
DEFAULT_EXTINCTION = {
    "U": 0.55, "B": 0.25, "V": 0.15, "R": 0.09, "I": 0.06,
    "u": 0.55, "g": 0.20, "r": 0.10, "i": 0.05, "z": 0.05,
}


@dataclass
class PhotometryResult:
    zeropoint: float = float("nan")
    zp_scatter: float = float("nan")
    n_calib: int = 0
    extinction: float = 0.0
    airmass: float = 1.0
    limiting_mag: float = float("nan")   # 5-sigma limiting magnitude
    calibrated: bool = False


def calibrate_photometry(catalog: pd.DataFrame, meta: FrameMeta, inst: Instrument,
                         pcfg: PhotometryConfig,
                         reference: pd.DataFrame | None = None) -> tuple[pd.DataFrame, PhotometryResult]:
    """Add instrumental + calibrated magnitudes to ``catalog``; return QA."""
    df = catalog.copy()
    exptime = max(meta.exptime, 1e-3)

    # prefer the stable fixed-aperture flux for magnitudes; fall back to segment
    flux_for_mag = df["flux_aper"] if "flux_aper" in df else df["flux"]
    snr_for_mag = df["snr_aper"] if "snr_aper" in df else df["snr"]
    flux_arr = np.asarray(flux_for_mag, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # non-positive fluxes (possible after local-annulus subtraction) have
        # no defined magnitude -> NaN, so downstream stats exclude them
        df["mag_inst"] = np.where(flux_arr > 0,
                                  -2.5 * np.log10(flux_arr / exptime), np.nan)
        df["mag_inst_err"] = np.where(snr_for_mag > 0, 1.0857 / snr_for_mag, np.nan)

    res = PhotometryResult(airmass=meta.airmass or 1.0)
    if not pcfg.enabled or reference is None or not len(reference) or "mag" not in reference:
        df["mag_calib"] = np.nan
        log.info("Photometry for %s: no reference mags; instrumental only", meta.path.name)
        return df, res

    if reference["mag"].isna().all():
        df["mag_calib"] = np.nan
        return df, res

    # match sources -> reference stars
    idx, sep, matched = crossmatch(df["ra"].to_numpy(), df["dec"].to_numpy(),
                                   reference["ra"].to_numpy(), reference["dec"].to_numpy(),
                                   radius_arcsec=pcfg.match_radius_arcsec)

    # clean calibrators: matched, bright enough, unsaturated, round, not edge,
    # and ISOLATED (a close neighbour biases fixed-aperture flux -> ZP scatter)
    isolated = _isolation_mask(df)
    good = (matched & (df["snr"].to_numpy() > 20)
            & (~df["flag_saturated"].to_numpy()) & (~df["flag_edge"].to_numpy())
            & (df["ellipticity"].to_numpy() < 0.3) & isolated)
    ref_mag = reference["mag"].to_numpy()[idx]
    mag_inst = df["mag_inst"].to_numpy()
    zp_samples = (ref_mag - mag_inst)[good & np.isfinite(ref_mag) & np.isfinite(mag_inst)]

    ext = DEFAULT_EXTINCTION.get(str(meta.filt), 0.0)
    res.extinction = ext

    if len(zp_samples) >= pcfg.min_ref_stars:
        zp_mean, zp_med, zp_std = sigma_clipped_stats(zp_samples, sigma=3.0, maxiters=5)
        res.zeropoint = float(zp_med)
        res.zp_scatter = float(zp_std)
        res.n_calib = int(len(zp_samples))
        res.calibrated = True
        # m_cal = m_inst + ZP. The zero point is fit against catalog magnitudes
        # of stars seen through the same atmosphere as the frame, so it already
        # absorbs extinction -- adding a -k*(X-1) term here would double-correct.
        # k and airmass are still recorded in the QA result for trending.
        df["mag_calib"] = df["mag_inst"] + res.zeropoint
        df["mag_calib_err"] = np.sqrt(df["mag_inst_err"] ** 2 + res.zp_scatter ** 2)
        res.limiting_mag = _limiting_mag(df, res, pcfg)
        log.info("Photometry %s: ZP=%.3f +/- %.3f from %d stars, k=%.2f, "
                 "5-sigma limit ~%.2f mag", meta.path.name, res.zeropoint,
                 res.zp_scatter, res.n_calib, ext, res.limiting_mag)
    else:
        df["mag_calib"] = np.nan
        log.warning("Photometry %s: only %d calibrators (< %d); not calibrated",
                    meta.path.name, len(zp_samples), pcfg.min_ref_stars)
    return df, res


def _isolation_mask(df: pd.DataFrame, min_sep_fwhm: float = 3.0) -> np.ndarray:
    """True where a source has no detected neighbour within min_sep_fwhm*FWHM."""
    n = len(df)
    if n < 2:
        return np.ones(n, bool)
    med_fwhm = float(np.nanmedian(df["fwhm"])) if "fwhm" in df else 4.0
    if not np.isfinite(med_fwhm) or med_fwhm <= 0:
        med_fwhm = 4.0
    thresh = max(10.0, min_sep_fwhm * med_fwhm)
    from scipy.spatial import cKDTree
    xy = np.column_stack([df["x"].to_numpy(), df["y"].to_numpy()])
    dist, _ = cKDTree(xy).query(xy, k=2)   # k=1 is the point itself
    return dist[:, 1] > thresh


def _limiting_mag(df: pd.DataFrame, res: PhotometryResult, pcfg: PhotometryConfig) -> float:
    """Estimate the 5-sigma limiting magnitude from faint calibrated sources."""
    if "mag_calib" not in df or not np.isfinite(res.zeropoint):
        return float("nan")
    near5 = df[(df["snr"] > 3) & (df["snr"] < 8) & np.isfinite(df["mag_calib"])]
    if len(near5) < 3:
        faint = df[np.isfinite(df["mag_calib"])]["mag_calib"]
        return float(np.nanpercentile(faint, 95)) if len(faint) else float("nan")
    return float(np.nanmedian(near5["mag_calib"]))
