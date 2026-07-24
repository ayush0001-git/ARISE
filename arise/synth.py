"""Synthetic-night generator -- ARISE's self-contained demonstration engine.

It writes a realistic single-night dataset to disk:

* calibration frames -- bias, dark, and per-filter flats carrying a real
  detector signature (bias pedestal, dark current + hot pixels, vignetting,
  dust "donuts", pixel-to-pixel gain) so calibration actually has work to do;
* a science *sequence* of the same star field, each frame with a Moffat PSF
  star field, a sky gradient, Poisson + read noise, and cosmic-ray hits;
* three planted discoveries -- a **moving asteroid** (linear motion across the
  sequence), a **stationary transient** (appears mid-sequence), and a
  **variable star** (sinusoidal brightness) -- none of which are in the
  bundled reference catalog;
* ``truth_sources.csv`` (ground truth for every planted object) and
  ``reference_catalog.csv`` (the "known sky" = static stars only), so the
  discovery stage runs fully offline and its recovery can be scored.

Everything is driven by a seeded RNG, so a night is byte-for-byte reproducible.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from .config import Instrument, get_instrument
from .logs import get_logger

log = get_logger("synth")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class SynthConfig:
    nx: int = 1024
    ny: int = 1024
    n_stars: int = 320
    seeing_fwhm_pix: float = 3.2         # PSF FWHM (Moffat)
    moffat_beta: float = 3.5
    sky_e_per_s: float = 12.0            # sky electrons/pixel/second (mean)
    sky_gradient: float = 0.25           # fractional gradient across the frame
    science_exptime: float = 60.0
    n_science: int = 6                   # frames in the time sequence
    n_bias: int = 5
    n_dark: int = 5
    n_flat: int = 5
    filt: str = "V"
    zeropoint: float = 25.0              # mag giving 1 e-/s
    faint_mag: float = 21.0
    bright_mag: float = 13.0
    bias_level: float = 400.0            # ADU pedestal
    dark_e_per_s: float = 0.05           # dark current e-/pix/s
    n_hot_pixels: int = 60
    n_cosmic_rays: int = 120             # per science frame
    field_ra_deg: float = 132.8000       # arbitrary but fixed field centre
    field_dec_deg: float = 11.6000
    wcs_pointing_err_arcsec: float = 0.4  # simulated telescope pointing error
    seed: int = 20260704

    # planted discoveries -------------------------------------------------- #
    asteroid_mag: float = 18.2
    asteroid_speed_arcsec_per_min: float = 3.0   # sky motion (fast NEO-like mover)
    transient_mag: float = 17.4
    transient_appears_frame: int = 2             # 0-based frame it turns on
    variable_mag: float = 16.0
    variable_amp_mag: float = 0.6                # peak-to-peak variation

    cadence_min: float = 8.0                     # minutes between science frames


# --------------------------------------------------------------------------- #
# PSF rendering
# --------------------------------------------------------------------------- #
def _moffat_alpha(fwhm: float, beta: float) -> float:
    return fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def _render_source(image: np.ndarray, x: float, y: float, flux: float,
                   fwhm: float, beta: float) -> None:
    """Add a normalised Moffat PSF of total ``flux`` (electrons) at (x, y)."""
    ny, nx = image.shape
    alpha = _moffat_alpha(fwhm, beta)
    radius = int(np.ceil(5.0 * fwhm))
    x0, x1 = max(0, int(x) - radius), min(nx, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(ny, int(y) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r2 = (xx - x) ** 2 + (yy - y) ** 2
    prof = (beta - 1.0) / (np.pi * alpha ** 2) * (1.0 + r2 / alpha ** 2) ** (-beta)
    image[y0:y1, x0:x1] += flux * prof


def _mag_to_flux_e(mag: float, exptime: float, zeropoint: float) -> float:
    """Total collected electrons for a source of magnitude ``mag``."""
    return float(10.0 ** (-0.4 * (mag - zeropoint)) * exptime)


# --------------------------------------------------------------------------- #
# detector signature (shared across a night)
# --------------------------------------------------------------------------- #
def _build_flat_response(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative detector response: vignetting * dust donuts * pixel gain."""
    ny, nx = cfg.ny, cfg.nx
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = nx / 2.0, ny / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1.0 - 0.18 * r ** 2                       # smooth radial falloff
    response = vignette.astype(np.float32)

    # a few out-of-focus dust "donuts"
    for _ in range(6):
        dx, dy = rng.uniform(0, nx), rng.uniform(0, ny)
        rad = rng.uniform(18, 42)
        depth = rng.uniform(0.05, 0.14)
        dd = np.sqrt((xx - dx) ** 2 + (yy - dy) ** 2)
        ring = np.exp(-((dd - rad) ** 2) / (2 * (rad * 0.18) ** 2))
        response *= (1.0 - depth * ring).astype(np.float32)

    # pixel-to-pixel gain variation (~1% rms)
    response *= rng.normal(1.0, 0.01, size=(ny, nx)).astype(np.float32)
    return np.clip(response, 0.2, 1.5).astype(np.float32)


def _build_hot_pixels(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    """A dark-current hot-pixel map (extra e-/s at specific pixels)."""
    hot = np.zeros((cfg.ny, cfg.nx), dtype=np.float32)
    for _ in range(cfg.n_hot_pixels):
        i = rng.integers(0, cfg.ny)
        j = rng.integers(0, cfg.nx)
        hot[i, j] = rng.uniform(20, 300)  # e-/s
    return hot


def _empty_position(star_x, star_y, nx, ny, rng, min_sep=20.0, margin=40,
                    xrange=None, yrange=None, tries=800):
    """Pick a location at least ``min_sep`` px from every star (best-effort)."""
    x0, x1 = xrange or (margin, nx - margin)
    y0, y1 = yrange or (margin, ny - margin)
    best = None
    for _ in range(tries):
        x = rng.uniform(x0, x1)
        y = rng.uniform(y0, y1)
        d2 = float(np.min((star_x - x) ** 2 + (star_y - y) ** 2))
        if d2 > min_sep ** 2:
            return x, y
        if best is None or d2 > best[2]:
            best = (x, y, d2)
    return best[0], best[1]


def _add_cosmic_rays(adu: np.ndarray, n: int, sat: float, rng: np.random.Generator) -> None:
    """Add sharp cosmic-ray hits (single pixels + short streaks) in-place."""
    ny, nx = adu.shape
    for _ in range(n):
        i = rng.integers(0, ny)
        j = rng.integers(0, nx)
        amp = rng.uniform(0.4, 1.4) * sat
        if rng.random() < 0.35:  # short streak
            length = rng.integers(2, 6)
            ang = rng.uniform(0, np.pi)
            for k in range(length):
                ii = int(i + k * np.sin(ang))
                jj = int(j + k * np.cos(ang))
                if 0 <= ii < ny and 0 <= jj < nx:
                    adu[ii, jj] = min(sat, adu[ii, jj] + amp)
        else:
            adu[i, j] = min(sat, adu[i, j] + amp)


# --------------------------------------------------------------------------- #
# header / WCS
# --------------------------------------------------------------------------- #
def _make_wcs(cfg: SynthConfig, inst: Instrument, dra: float = 0.0, ddec: float = 0.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [cfg.nx / 2.0, cfg.ny / 2.0]
    w.wcs.crval = [cfg.field_ra_deg + dra, cfg.field_dec_deg + ddec]
    scale_deg = inst.pixel_scale / 3600.0
    w.wcs.cd = np.array([[-scale_deg, 0.0], [0.0, scale_deg]])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _base_header(cfg: SynthConfig, inst: Instrument, imagetyp: str, exptime: float,
                 obj: str, dateobs: str, wcs: WCS | None,
                 seq_minutes: float = 0.0) -> fits.Header:
    hdr = wcs.to_header() if wcs is not None else fits.Header()
    hdr["IMAGETYP"] = (imagetyp, "frame type")
    hdr["EXPTIME"] = (exptime, "[s] exposure time")
    hdr["FILTER"] = (cfg.filt, "filter")
    hdr["OBJECT"] = (obj, "target")
    hdr["DATE-OBS"] = (dateobs, "UTC start of exposure")
    hdr["GAIN"] = (inst.gain, "[e-/ADU] detector gain")
    hdr["RDNOISE"] = (inst.read_noise, "[e-] read noise")
    hdr["TELESCOP"] = (inst.telescope, "telescope")
    hdr["INSTRUME"] = (inst.detector, "detector")
    # airmass climbs slowly and deterministically through the sequence
    airmass = min(2.5, 1.05 + 0.004 * max(0.0, seq_minutes))
    hdr["AIRMASS"] = (round(airmass, 3), "airmass")
    hdr["SYNTH"] = (True, "ARISE synthetic frame")
    return hdr


def _iso_time(base_minutes: float) -> str:
    """Deterministic ISO timestamp offset from a fixed night start (no clock)."""
    start = datetime(2026, 7, 4, 18, 0, 0)  # fixed night start (UTC)
    t = start + timedelta(seconds=round(base_minutes * 60))
    return t.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
# frame builders
# --------------------------------------------------------------------------- #
def _digitize(electrons: np.ndarray, inst: Instrument, cfg: SynthConfig,
              rng: np.random.Generator) -> np.ndarray:
    """electrons -> ADU with read noise, bias pedestal, saturation, integerise."""
    read_adu = rng.normal(0.0, inst.read_noise, size=electrons.shape) / inst.gain
    adu = electrons / inst.gain + cfg.bias_level + read_adu
    np.clip(adu, 0, inst.saturation, out=adu)
    return np.round(adu).astype(np.float32)


def _make_bias(cfg, inst, rng) -> np.ndarray:
    return _digitize(np.zeros((cfg.ny, cfg.nx), dtype=np.float32), inst, cfg, rng)


def _make_dark(cfg, inst, hot, rng) -> np.ndarray:
    dark_e = (cfg.dark_e_per_s + 0.0) * cfg.science_exptime + hot * cfg.science_exptime
    dark_e = rng.poisson(np.clip(dark_e, 0, None)).astype(np.float32)
    return _digitize(dark_e, inst, cfg, rng)


def _make_flat(cfg, inst, flat_resp, rng) -> np.ndarray:
    illum = 30000.0 * inst.gain  # target ~30k ADU well-exposed flat (electrons)
    e = flat_resp * illum
    e = rng.poisson(np.clip(e, 0, None)).astype(np.float32)
    return _digitize(e, inst, cfg, rng)


def _science_electrons(cfg, inst, star_xy, star_mag, extra_sources,
                       flat_resp, hot, rng) -> np.ndarray:
    """Photon-electron image of the sky (before detector digitisation)."""
    ny, nx = cfg.ny, cfg.nx
    img = np.zeros((ny, nx), dtype=np.float32)

    # stars
    for (x, y), mag in zip(star_xy, star_mag, strict=True):
        flux = _mag_to_flux_e(mag, cfg.science_exptime, cfg.zeropoint)
        _render_source(img, x, y, flux, cfg.seeing_fwhm_pix, cfg.moffat_beta)

    # planted discoveries
    for (x, y, mag) in extra_sources:
        flux = _mag_to_flux_e(mag, cfg.science_exptime, cfg.zeropoint)
        _render_source(img, x, y, flux, cfg.seeing_fwhm_pix, cfg.moffat_beta)

    # sky background with a linear gradient
    yy, xx = np.mgrid[0:ny, 0:nx]
    grad = 1.0 + cfg.sky_gradient * ((xx / nx - 0.5) + 0.6 * (yy / ny - 0.5))
    sky = cfg.sky_e_per_s * cfg.science_exptime * grad
    img += sky.astype(np.float32)

    # optics/QE response multiplies incoming light; dark adds regardless
    img *= flat_resp
    img += (cfg.dark_e_per_s * cfg.science_exptime) + hot * cfg.science_exptime

    # shot noise
    img = rng.poisson(np.clip(img, 0, None)).astype(np.float32)
    return img


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def generate_night(outdir: str | Path, instrument: str | Instrument = "dfot_2kx2k",
                   cfg: SynthConfig | None = None) -> dict[str, Any]:
    """Generate a full synthetic observing night under ``outdir``.

    Returns a manifest dict describing everything written (also saved as
    ``manifest.json`` alongside the data).
    """
    cfg = cfg or SynthConfig()
    inst = instrument if isinstance(instrument, Instrument) else get_instrument(instrument)
    outdir = Path(outdir)
    raw = outdir
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    log.info("Synthesising a night for %s (%dx%d, %d science frames)",
             inst.telescope or inst.name, cfg.nx, cfg.ny, cfg.n_science)

    flat_resp = _build_flat_response(cfg, rng)
    hot = _build_hot_pixels(cfg, rng)

    # --- static star field (the "known sky") ------------------------------ #
    margin = 12
    star_x = rng.uniform(margin, cfg.nx - margin, cfg.n_stars)
    star_y = rng.uniform(margin, cfg.ny - margin, cfg.n_stars)
    # magnitude distribution: many faint, few bright (roughly exponential)
    star_mag = cfg.bright_mag + (cfg.faint_mag - cfg.bright_mag) * rng.power(2.5, cfg.n_stars)
    star_xy = list(zip(star_x, star_y, strict=True))

    # one of the static stars is variable: pick an ISOLATED one so its light
    # curve isn't diluted by a neighbour (clean, unambiguous demo).
    isolation = np.empty(cfg.n_stars)
    for i in range(cfg.n_stars):
        d2 = (star_x - star_x[i]) ** 2 + (star_y - star_y[i]) ** 2
        d2[i] = np.inf
        isolation[i] = np.sqrt(d2.min())
    isolated = np.where(isolation > 25.0)[0]
    var_idx = int(isolated[rng.integers(len(isolated))]) if len(isolated) else int(rng.integers(cfg.n_stars))
    star_mag[var_idx] = cfg.variable_mag       # anchor its mean brightness
    var_x, var_y = star_xy[var_idx]

    true_wcs = _make_wcs(cfg, inst)
    ras, decs = true_wcs.wcs_pix2world(star_x, star_y, 0)

    manifest: dict[str, Any] = {
        "instrument": inst.name, "telescope": inst.telescope,
        "nx": cfg.nx, "ny": cfg.ny, "filter": cfg.filt,
        "science_exptime": cfg.science_exptime, "seed": cfg.seed,
        "bias": [], "dark": [], "flat": [], "science": [],
    }

    # --- calibration frames ---------------------------------------------- #
    for n in range(cfg.n_bias):
        p = raw / f"bias_{n:03d}.fits"
        hdr = _base_header(cfg, inst, "bias", 0.0, "BIAS", _iso_time(-40 + n), None)
        fits.PrimaryHDU(_make_bias(cfg, inst, rng), hdr).writeto(p, overwrite=True)
        manifest["bias"].append(p.name)

    for n in range(cfg.n_dark):
        p = raw / f"dark_{n:03d}.fits"
        hdr = _base_header(cfg, inst, "dark", cfg.science_exptime, "DARK", _iso_time(-30 + n), None)
        fits.PrimaryHDU(_make_dark(cfg, inst, hot, rng), hdr).writeto(p, overwrite=True)
        manifest["dark"].append(p.name)

    for n in range(cfg.n_flat):
        p = raw / f"flat_{cfg.filt}_{n:03d}.fits"
        hdr = _base_header(cfg, inst, "flat", 3.0, "FLAT", _iso_time(-20 + n), None)
        fits.PrimaryHDU(_make_flat(cfg, inst, flat_resp, rng), hdr).writeto(p, overwrite=True)
        manifest["flat"].append(p.name)

    # --- planted-object geometry ----------------------------------------- #
    # Place planted objects in *empty* sky so they are cleanly detectable as
    # new sources (not blended with a catalogued star).
    scale = inst.pixel_scale  # arcsec/pix
    ast_speed_pix = cfg.asteroid_speed_arcsec_per_min / scale * cfg.cadence_min
    ast_ang = np.deg2rad(28.0)
    # asteroid: start in the lower-left quadrant so it has room to move up-right
    ast_x0, ast_y0 = _empty_position(
        star_x, star_y, cfg.nx, cfg.ny, rng, min_sep=20.0,
        xrange=(margin, cfg.nx * 0.35), yrange=(margin, cfg.ny * 0.35))
    # transient: a fixed empty position, turns on mid-sequence
    tr_x, tr_y = _empty_position(star_x, star_y, cfg.nx, cfg.ny, rng, min_sep=22.0)

    truth_rows: list[dict[str, Any]] = []

    # --- science sequence ------------------------------------------------ #
    for k in range(cfg.n_science):
        t_min = k * cfg.cadence_min
        extra: list[tuple[float, float, float]] = []

        # asteroid position this frame
        ax = ast_x0 + ast_speed_pix * k * np.cos(ast_ang)
        ay = ast_y0 + ast_speed_pix * k * np.sin(ast_ang)
        if 0 <= ax < cfg.nx and 0 <= ay < cfg.ny:
            extra.append((ax, ay, cfg.asteroid_mag))

        # transient (steps on and stays)
        tr_on = k >= cfg.transient_appears_frame
        if tr_on:
            extra.append((tr_x, tr_y, cfg.transient_mag))

        # variable star: modulate the chosen static star's magnitude
        phase = 2 * np.pi * (k / max(1, cfg.n_science))
        var_mag_k = cfg.variable_mag + 0.5 * cfg.variable_amp_mag * np.sin(phase)
        this_star_mag = np.array(star_mag, dtype=float)
        this_star_mag[var_idx] = var_mag_k

        e = _science_electrons(cfg, inst, star_xy, this_star_mag, extra, flat_resp, hot, rng)
        adu = _digitize(e, inst, cfg, rng)
        _add_cosmic_rays(adu, cfg.n_cosmic_rays, inst.saturation, rng)

        # header WCS carries a small pointing error (refined later by astrometry)
        derr = cfg.wcs_pointing_err_arcsec / 3600.0
        wcs_hdr = _make_wcs(cfg, inst, dra=rng.normal(0, derr), ddec=rng.normal(0, derr))
        iso = _iso_time(t_min)
        hdr = _base_header(cfg, inst, "light", cfg.science_exptime,
                           "ARISE_FIELD", iso, wcs_hdr, seq_minutes=t_min)
        hdr["FRAMENUM"] = (k, "sequence index")
        hdr["MJD-OBS"] = (float(Time(iso, format="isot", scale="utc").mjd), "[d] MJD of DATE-OBS")
        p = raw / f"science_{cfg.filt}_{k:03d}.fits"
        fits.PrimaryHDU(adu, hdr).writeto(p, overwrite=True)
        manifest["science"].append(p.name)

        # record truth for planted movers/transients per frame
        if 0 <= ax < cfg.nx and 0 <= ay < cfg.ny:
            r, d = true_wcs.wcs_pix2world(ax, ay, 0)
            truth_rows.append({"frame": k, "kind": "asteroid", "x": float(ax), "y": float(ay),
                               "ra": float(r), "dec": float(d), "mag": cfg.asteroid_mag})
        if tr_on:
            r, d = true_wcs.wcs_pix2world(tr_x, tr_y, 0)
            truth_rows.append({"frame": k, "kind": "transient", "x": float(tr_x), "y": float(tr_y),
                               "ra": float(r), "dec": float(d), "mag": cfg.transient_mag})
        r, d = true_wcs.wcs_pix2world(var_x, var_y, 0)
        truth_rows.append({"frame": k, "kind": "variable", "x": float(var_x), "y": float(var_y),
                           "ra": float(r), "dec": float(d), "mag": float(var_mag_k)})

    # --- truth + reference catalogs -------------------------------------- #
    truth_path = outdir / "truth_sources.csv"
    with open(truth_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", "kind", "x", "y", "ra", "dec", "mag"])
        writer.writeheader()
        writer.writerows(truth_rows)

    # reference catalog = static stars only (planted objects deliberately absent)
    ref_path = outdir / "reference_catalog.csv"
    with open(ref_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ra", "dec", "mag"])
        for (x, y), ra, dec, mag in zip(star_xy, ras, decs, star_mag, strict=True):
            if int(x) == int(var_x) and int(y) == int(var_y):
                mag = cfg.variable_mag  # catalogued at mean brightness
            writer.writerow([f"{ra:.6f}", f"{dec:.6f}", f"{mag:.3f}"])

    manifest.update({
        "truth_catalog": truth_path.name,
        "reference_catalog": ref_path.name,
        "planted": {
            "asteroid": {"mag": cfg.asteroid_mag,
                         "speed_arcsec_per_min": cfg.asteroid_speed_arcsec_per_min},
            "transient": {"mag": cfg.transient_mag, "appears_frame": cfg.transient_appears_frame},
            "variable": {"mag": cfg.variable_mag, "amp_mag": cfg.variable_amp_mag},
        },
    })

    import json
    with open(outdir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    log.info("Wrote %d bias, %d dark, %d flat, %d science frames to %s",
             cfg.n_bias, cfg.n_dark, cfg.n_flat, cfg.n_science, raw)
    log.info("Planted: moving asteroid (mag %.1f), transient (mag %.1f, frame %d+), "
             "variable star (mag %.1f +/- %.2f)",
             cfg.asteroid_mag, cfg.transient_mag, cfg.transient_appears_frame,
             cfg.variable_mag, cfg.variable_amp_mag)
    return manifest
