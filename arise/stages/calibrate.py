"""Stage 2 -- calibration (instrumental-signature removal).

Builds master bias/dark/flat with sigma-clipped combination, then reduces each
science frame::

    reduced = (raw - bias - dark_scaled) / flat_norm

and propagates a per-pixel variance map (read noise + Poisson + dark shot
noise, carried through the flat division) that later stages use for detection
thresholds and photometric errors. Order-of-operations
and scaling follow standard CCD reduction (cf. ccdproc / observatory pipelines).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.stats import sigma_clip

from ..config import Instrument
from ..fitsio import FrameMeta, read_frame, write_frame
from ..logs import get_logger

log = get_logger("calibrate")


# --------------------------------------------------------------------------- #
# combination
# --------------------------------------------------------------------------- #
def combine(frames: Sequence[np.ndarray], method: str = "median",
            sigma: float = 3.0, maxiters: int = 5) -> np.ndarray:
    """Sigma-clipped stack combine of same-shape frames -> single image."""
    if not frames:
        raise ValueError("combine() got no frames")
    cube = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    if cube.shape[0] == 1:
        return cube[0].copy()
    # median centre + MAD-based deviation: a single bright cosmic ray can't
    # inflate the scatter enough to escape the clip (as plain std allows).
    clipped = sigma_clip(cube, sigma=sigma, maxiters=maxiters, axis=0, masked=True,
                         cenfunc="median", stdfunc="mad_std")
    if method == "mean":
        out = clipped.mean(axis=0)
    else:
        out = np.ma.median(clipped, axis=0)
    # fill any fully-masked pixel with the plain median so there are no NaNs
    filled = np.ma.filled(out, np.nan)
    if np.isnan(filled).any():
        plain = np.median(cube, axis=0)
        filled = np.where(np.isnan(filled), plain, filled)
    return filled.astype(np.float32)


def _load(metas: Sequence[FrameMeta]) -> list[np.ndarray]:
    return [read_frame(m.path)[0] for m in metas]


# --------------------------------------------------------------------------- #
# master frames
# --------------------------------------------------------------------------- #
@dataclass
class Masters:
    bias: np.ndarray | None = None
    dark_rate: np.ndarray | None = None   # bias-subtracted dark current, ADU/sec
    dark_exptime: float = 0.0
    flats: dict[str, np.ndarray] | None = None   # normalised, per filter
    bad_pixel_mask: np.ndarray | None = None

    def flat_for(self, filt: str) -> np.ndarray | None:
        if not self.flats:
            return None
        if filt in self.flats:
            return self.flats[filt]
        # single flat, unknown filter mismatch: fall back to the only one present
        if len(self.flats) == 1:
            only_filt, flat = next(iter(self.flats.items()))
            log.warning(
                "No master flat for filter %r; falling back to the only "
                "available flat (filter %r). Flat-fielding may carry "
                "filter-dependent systematics.", filt, only_filt)
            return flat
        return None


def build_masters(frames, inst: Instrument, out_dir: str | Path | None = None) -> Masters:
    """Construct master bias, dark-rate and per-filter normalised flats."""
    from .ingest import FrameSet  # local import to avoid cycle
    assert isinstance(frames, FrameSet)
    out_dir = Path(out_dir) if out_dir else None
    m = Masters(flats={})

    # ---- master bias ---------------------------------------------------- #
    # sigma-clipped MEAN (not median): averaging N frames beats median, read
    # noise drops by ~sqrt(N); sigma-clipping still rejects outliers/CR hits.
    if frames.bias:
        m.bias = combine(_load(frames.bias), method="mean")
        log.info("Master bias from %d frames (median %.1f ADU)",
                 len(frames.bias), float(np.median(m.bias)))
        if out_dir:
            write_frame(out_dir / "master_bias.fits", m.bias,
                        history=[f"master bias, sigma-clipped mean of {len(frames.bias)} frames"])
    else:
        log.warning("No bias frames; bias subtraction will be skipped.")

    # ---- master dark (as a per-second rate) ----------------------------- #
    if frames.dark:
        darks_by_exp = frames.darks_by_exptime()
        # use the longest-exposure dark group for the best-SNR rate estimate
        exp = max(darks_by_exp)
        master_dark = combine(_load(darks_by_exp[exp]), method="mean")
        if m.bias is not None:
            master_dark = master_dark - m.bias
        m.dark_exptime = exp
        m.dark_rate = master_dark / exp if exp > 0 else master_dark
        log.info("Master dark from %d frames at %.1fs (median rate %.4f ADU/s)",
                 len(darks_by_exp[exp]), exp, float(np.median(m.dark_rate)))
        if out_dir:
            write_frame(out_dir / "master_dark_rate.fits", m.dark_rate,
                        history=[f"master dark rate (ADU/s), from {exp}s darks"])
        # bad-pixel mask: pixels with anomalously high dark current
        med = float(np.median(m.dark_rate))
        std = float(np.std(m.dark_rate))
        m.bad_pixel_mask = m.dark_rate > (med + 8.0 * std)
        log.info("Flagged %d hot/bad pixels", int(m.bad_pixel_mask.sum()))
    else:
        log.warning("No dark frames; dark subtraction will be skipped.")

    # ---- master flats (per filter, normalised to ~1) -------------------- #
    for filt, flist in frames.flats_by_filter().items():
        raws = _load(flist)
        cleaned = []
        for fmeta, f in zip(flist, raws, strict=True):
            g = f.copy()
            if m.bias is not None:
                g = g - m.bias
            if m.dark_rate is not None:
                # scale the dark rate to each flat's own (short) exposure
                g = g - m.dark_rate * fmeta.exptime
            # scale each flat to a common level by its own median so that
            # sky/dome-illumination differences between flats don't bias the mean
            med = float(np.median(g))
            if med > 0:
                g = g / med
            cleaned.append(g)
        master_flat = combine(cleaned, method="mean")
        norm = float(np.median(master_flat))
        if norm <= 0:
            log.warning("Flat for filter %s has non-positive median; skipping.", filt)
            continue
        master_flat = master_flat / norm
        # guard against divide-by-zero in reduction: neutralise low-response
        # pixels, but flag them as bad first so the DQ plane records them
        low = master_flat < 0.05
        if low.any():
            if m.bad_pixel_mask is None:
                m.bad_pixel_mask = np.zeros(master_flat.shape, dtype=bool)
            m.bad_pixel_mask |= low
            log.info("Flagged %d low-response (<5%%) flat pixels in filter %s as bad",
                     int(low.sum()), filt)
        master_flat[low] = 1.0
        m.flats[filt] = master_flat.astype(np.float32)
        log.info("Master flat for filter %s from %d frames (norm %.1f ADU)",
                 filt, len(flist), norm)
        if out_dir:
            write_frame(out_dir / f"master_flat_{filt}.fits", m.flats[filt],
                        history=[f"master flat {filt}, normalised, {len(flist)} frames"])

    if not frames.flats_by_filter():
        log.warning("No flat frames; flat-fielding will be skipped.")
    return m


# --------------------------------------------------------------------------- #
# reduce a science frame
# --------------------------------------------------------------------------- #
@dataclass
class ReducedFrame:
    data: np.ndarray            # calibrated science image (ADU)
    variance: np.ndarray        # per-pixel variance (ADU^2)
    meta: FrameMeta
    bad_pixel_mask: np.ndarray | None = None


def reduce_frame(meta: FrameMeta, masters: Masters, inst: Instrument) -> ReducedFrame:
    """Apply bias, dark, and flat correction to one science frame + variance."""
    raw, _hdr = read_frame(meta.path)
    data = raw.astype(np.float32)

    steps = []
    if masters.bias is not None:
        data = data - masters.bias
        steps.append("bias")
    if masters.dark_rate is not None:
        data = data - masters.dark_rate * meta.exptime
        steps.append("dark")

    flat = masters.flat_for(meta.filt)
    pre_flat = data              # signal before flat division, for the variance
    if flat is not None:
        data = data / flat
        steps.append("flat")

    gain = meta.gain or inst.gain
    rdn = meta.read_noise or inst.read_noise
    # variance in ADU^2: read noise term + Poisson term on the pre-flat signal
    # (N_e/gain^2 = S_ADU/gain) + shot noise of the subtracted dark charge,
    # then divided by flat^2 to propagate the flat-field division
    variance = (rdn / gain) ** 2 + np.clip(pre_flat, 0, None) / gain
    if masters.dark_rate is not None:
        variance = variance + np.clip(masters.dark_rate, 0, None) * meta.exptime / gain
    if flat is not None:
        variance = variance / (flat ** 2)

    log.info("Reduced %s [%s]", meta.path.name, "+".join(steps) or "no-op")
    return ReducedFrame(data=data, variance=variance.astype(np.float32),
                        meta=meta, bad_pixel_mask=masters.bad_pixel_mask)
