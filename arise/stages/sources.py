"""Stage 5 -- source detection, deblending, and extraction.

Matched-filter detection above ``nsigma * background_rms``, watershed
deblending, then :class:`~photutils.segmentation.SourceCatalog` to measure each
source: centroid (pixel + sky), segment & Kron flux with errors, SNR, FWHM,
ellipticity/elongation, peak value and quality flags. Returns a tidy
:class:`pandas.DataFrame` catalog plus the segmentation image (a star mask).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.convolution import convolve
from astropy.wcs import WCS
from photutils.segmentation import SourceCatalog, SourceFinder, make_2dgaussian_kernel

from ..config import DetectConfig, Instrument
from ..fitsio import FrameMeta
from ..logs import get_logger
from .background import BackgroundResult

log = get_logger("sources")


@dataclass
class Extraction:
    catalog: pd.DataFrame
    segment_map: np.ndarray          # labelled segmentation (0 = sky)
    median_fwhm: float               # seeing estimate (pixels)
    n_sources: int


def _col(quantity) -> np.ndarray:
    """Strip astropy units -> plain float array."""
    arr = getattr(quantity, "value", quantity)
    return np.asarray(arr, dtype=float)


def _attr(cat, *names):
    """Fetch the first attribute that exists (handles photutils renames)."""
    for n in names:
        if hasattr(cat, n):
            return getattr(cat, n)
    raise AttributeError(f"SourceCatalog has none of {names}")


def _make_finder(dcfg: DetectConfig) -> "SourceFinder":
    base = dict(deblend=dcfg.deblend, contrast=dcfg.deblend_contrast, progress_bar=False)
    try:  # photutils >= 3.0
        return SourceFinder(n_pixels=dcfg.npixels, n_levels=dcfg.deblend_nlevels, **base)
    except TypeError:  # photutils < 3.0
        return SourceFinder(npixels=dcfg.npixels, nlevels=dcfg.deblend_nlevels, **base)


def extract_sources(bkg: BackgroundResult, variance: np.ndarray, meta: FrameMeta,
                    inst: Instrument, dcfg: DetectConfig,
                    wcs: WCS | None = None) -> Extraction:
    """Detect and measure every source above threshold in a reduced frame."""
    data = bkg.data_sub
    error = np.sqrt(np.clip(variance, 0, None)).astype(np.float32)
    kernel = make_2dgaussian_kernel(dcfg.kernel_fwhm, size=5)
    convolved = convolve(data, kernel, normalize_kernel=True)
    threshold = dcfg.nsigma * bkg.rms

    finder = _make_finder(dcfg)
    segm = finder(convolved, threshold)
    if segm is None:
        log.warning("No sources detected in %s", meta.path.name)
        return Extraction(catalog=_empty_catalog(), segment_map=np.zeros(data.shape, int),
                          median_fwhm=float("nan"), n_sources=0)

    cat = SourceCatalog(data, segm, convolved_data=convolved, error=error,
                        background=bkg.background, wcs=wcs)

    x = _col(_attr(cat, "x_centroid", "xcentroid"))
    y = _col(_attr(cat, "y_centroid", "ycentroid"))
    seg_flux = _col(_attr(cat, "segment_flux"))
    seg_err = _col(_attr(cat, "segment_flux_err", "segment_fluxerr"))
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(seg_err > 0, seg_flux / seg_err, 0.0)
    fwhm = _col(_attr(cat, "fwhm"))
    ellip = _col(_attr(cat, "ellipticity"))
    elong = _col(_attr(cat, "elongation"))
    peak = _col(_attr(cat, "max_value"))
    area = _col(_attr(cat, "area"))
    try:
        kron = _col(_attr(cat, "kron_flux"))
    except Exception:
        kron = seg_flux

    df = pd.DataFrame({
        "id": np.asarray(cat.labels, dtype=int),
        "x": x, "y": y,
        "flux": seg_flux, "flux_err": seg_err, "snr": snr,
        "kron_flux": kron,
        "fwhm": fwhm, "ellipticity": ellip, "elongation": elong,
        "peak": peak, "area": area,
    })

    # world coordinates when a WCS is available
    if wcs is not None and getattr(cat, "sky_centroid", None) is not None:
        sky = cat.sky_centroid
        df["ra"] = sky.ra.deg
        df["dec"] = sky.dec.deg
    else:
        df["ra"] = np.nan
        df["dec"] = np.nan

    # quality flags
    ny, nx = data.shape
    edge = 8
    df["flag_edge"] = ((x < edge) | (y < edge) | (x > nx - edge) | (y > ny - edge))
    # saturation is a property of the pre-sky-subtraction signal: peak is
    # measured on data_sub, so add the local background back before comparing
    # (the -400 approximates the bias/dark pedestal removed in calibration)
    try:
        local_bkg = _col(_attr(cat, "background_mean"))
    except Exception:
        local_bkg = np.full_like(peak, bkg.median)
    local_bkg = np.where(np.isfinite(local_bkg), local_bkg, bkg.median)
    df["flag_saturated"] = (peak + local_bkg) >= 0.95 * (inst.saturation - 400.0)
    df["frame"] = meta.path.name

    # seeing estimate from compact, high-SNR, round sources (i.e. stars)
    starlike = df[(df.snr > 20) & (df.ellipticity < 0.3) & (~df.flag_saturated)
                  & (~df.flag_edge)]
    median_fwhm = float(np.nanmedian(starlike.fwhm)) if len(starlike) else float(np.nanmedian(df.fwhm))
    if not np.isfinite(median_fwhm) or median_fwhm <= 0:
        median_fwhm = 3.0

    # fixed-aperture photometry: threshold-independent, stable frame-to-frame,
    # local sky from an annulus -> reliable light curves. This (not isophotal
    # segment flux) is what feeds magnitudes and variability.
    _add_aperture_photometry(df, data, error, median_fwhm, dcfg.aperture_scale)

    log.info("Extracted %d sources from %s (median stellar FWHM %.2f px)",
             len(df), meta.path.name, median_fwhm)
    return Extraction(catalog=df, segment_map=np.asarray(segm.data, dtype=int),
                      median_fwhm=median_fwhm, n_sources=len(df))


def _add_aperture_photometry(df: pd.DataFrame, data: np.ndarray, error: np.ndarray,
                             median_fwhm: float, aperture_scale: float) -> None:
    """Add fixed circular-aperture flux (local-sky subtracted) to the catalog."""
    from photutils.aperture import (CircularAperture, CircularAnnulus,
                                     ApertureStats, aperture_photometry)
    if len(df) == 0:
        df["flux_aper"] = []
        df["flux_aper_err"] = []
        df["snr_aper"] = []
        return
    r = max(3.0, aperture_scale * median_fwhm)
    positions = np.transpose((df["x"].to_numpy(), df["y"].to_numpy()))
    ap = CircularAperture(positions, r=r)
    ann = CircularAnnulus(positions, r_in=r * 1.8, r_out=r * 2.8)
    # residual local background per pixel (data is already globally sky-subtracted)
    local = ApertureStats(data, ann, error=error).median
    local = np.where(np.isfinite(local), local, 0.0)
    phot = aperture_photometry(data, ap, error=error)
    flux = np.asarray(phot["aperture_sum"], float) - local * ap.area
    ferr = np.asarray(phot["aperture_sum_err"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(ferr > 0, flux / ferr, 0.0)
    df["flux_aper"] = flux
    df["flux_aper_err"] = ferr
    df["snr_aper"] = snr


def _empty_catalog() -> pd.DataFrame:
    cols = ["id", "x", "y", "flux", "flux_err", "snr", "kron_flux", "fwhm",
            "ellipticity", "elongation", "peak", "area", "ra", "dec",
            "flag_edge", "flag_saturated", "frame"]
    return pd.DataFrame({c: [] for c in cols})
