"""Stage 3 -- cosmic-ray rejection (and optional, photometry-safe denoise).

Primary method is the L.A.Cosmic Laplacian-edge algorithm (van Dokkum 2001) via
astroscrappy, which distinguishes sharp CR hits from real (PSF-broad) sources.
An optional deepCR backend and an *opt-in* edge-preserving denoise are provided;
denoising is off by default because smoothing biases astrometry and photometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CosmicRayConfig, DenoiseConfig, Instrument
from ..fitsio import FrameMeta
from ..logs import get_logger

log = get_logger("cosmicray")


@dataclass
class CRResult:
    data: np.ndarray            # cleaned image
    mask: np.ndarray            # bool CR mask (True = affected pixel)
    n_flagged: int


def reject_cosmic_rays(data: np.ndarray, meta: FrameMeta, inst: Instrument,
                       cfg: CosmicRayConfig) -> CRResult:
    """Return a CR-cleaned image and the CR mask."""
    if not cfg.enabled or cfg.method == "none":
        return CRResult(data=data, mask=np.zeros(data.shape, bool), n_flagged=0)

    gain = meta.gain or inst.gain
    rdn = meta.read_noise or inst.read_noise

    if cfg.method == "deepcr":
        cleaned, mask = _deepcr(data, gain)
        if cleaned is not None:
            n = int(mask.sum())
            log.info("deepCR flagged %d pixels in %s", n, meta.path.name)
            return CRResult(cleaned, mask, n)
        log.warning("deepCR unavailable; falling back to L.A.Cosmic")

    import astroscrappy

    # astroscrappy expects ADU input; it applies gain internally for the noise
    # model. satlevel, however, is compared against the gain-multiplied image,
    # i.e. it must be given in ELECTRONS -- convert the ADU ceiling.
    satlevel = float(inst.saturation) * gain
    mask, cleaned = astroscrappy.detect_cosmics(
        np.ascontiguousarray(data, dtype=np.float32),
        sigclip=cfg.sigclip,
        sigfrac=cfg.sigfrac,
        objlim=cfg.objlim,
        gain=gain,
        readnoise=rdn,
        satlevel=satlevel,
        niter=cfg.niter,
        sepmed=True,
        cleantype="meanmask",
        fsmode="median",
        verbose=False,
    )
    # astroscrappy uses gain only for its internal noise model and returns the
    # cleaned array in the SAME units as the input (ADU) -- verified empirically.
    # Do NOT divide by gain here (that is the classic double-gain bug).
    cleaned = np.asarray(cleaned, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    log.info("L.A.Cosmic flagged %d cosmic-ray pixels in %s (%.3f%%)",
             n, meta.path.name, 100.0 * n / mask.size)
    return CRResult(cleaned, mask, n)


def denoise(data: np.ndarray, cfg: DenoiseConfig) -> np.ndarray:
    """Optional edge-preserving denoise. Off by default (photometry safety)."""
    if not cfg.enabled or cfg.method == "none":
        return data
    try:
        from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet
    except Exception:  # pragma: no cover
        log.warning("scikit-image restoration unavailable; skipping denoise")
        return data

    lo, hi = np.percentile(data, [1, 99])
    span = max(hi - lo, 1e-6)
    norm = np.clip((data - lo) / span, 0, 1)
    if cfg.method == "bilateral":
        out = denoise_bilateral(norm, sigma_color=0.05 * cfg.strength, sigma_spatial=cfg.strength)
    elif cfg.method == "tv":
        out = denoise_tv_chambolle(norm, weight=0.02 * cfg.strength)
    elif cfg.method == "wavelet":
        out = denoise_wavelet(norm, rescale_sigma=True)
    else:
        log.warning("Unknown denoise method '%s'; skipping", cfg.method)
        return data
    log.info("Applied %s denoise (strength %.2f)", cfg.method, cfg.strength)
    return (out * span + lo).astype(np.float32)


def _deepcr(data: np.ndarray, gain: float):
    """Deep-learning CR rejection; returns (None, None) if deepCR isn't installed."""
    try:
        from deepCR import deepCR
    except Exception:
        return None, None
    try:
        mdl = deepCR(mask="ACS-WFC-F606W-2-32", device="CPU")
        mask_prob, cleaned = mdl.clean(np.asarray(data, np.float32), inpaint=True)
        return np.asarray(cleaned, np.float32), np.asarray(mask_prob > 0.5, bool)
    except Exception as exc:  # pragma: no cover
        log.warning("deepCR run failed: %s", exc)
        return None, None
