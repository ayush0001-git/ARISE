"""Stage 4 -- sky-background estimation & removal.

Uses photutils' :class:`Background2D` with a SExtractor-style mesh estimator and
*iterative source masking*: estimate background, detect sources, mask them, and
re-estimate so bright objects don't bias the sky model. Returns the
background-subtracted image, the 2-D background and its RMS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.convolution import convolve
from astropy.stats import SigmaClip
from photutils.background import (Background2D, MedianBackground,
                                  MMMBackground, SExtractorBackground)
from photutils.segmentation import detect_sources, make_2dgaussian_kernel
from scipy.ndimage import binary_dilation

from ..config import BackgroundConfig, DetectConfig
from ..logs import get_logger

log = get_logger("background")


@dataclass
class BackgroundResult:
    data_sub: np.ndarray        # background-subtracted image
    background: np.ndarray      # 2-D sky model
    rms: np.ndarray             # per-pixel background RMS
    source_mask: np.ndarray     # bool mask of pixels belonging to sources
    median: float
    median_rms: float


def _estimator(name: str):
    return {
        "sextractor": SExtractorBackground(),
        "median": MedianBackground(),
        "mmm": MMMBackground(),
    }.get(name, SExtractorBackground())


def model_background(data: np.ndarray, bcfg: BackgroundConfig, dcfg: DetectConfig,
                     mask: np.ndarray | None = None) -> BackgroundResult:
    """Estimate and subtract the sky background with iterative source masking."""
    sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
    estimator = _estimator(bcfg.estimator)
    kernel = make_2dgaussian_kernel(dcfg.kernel_fwhm, size=5)

    box = (bcfg.box_size, bcfg.box_size)
    filt = (bcfg.filter_size, bcfg.filter_size)

    src_mask = np.zeros(data.shape, bool)
    bkg = None
    for _it in range(max(1, bcfg.mask_sources_iters)):
        combined = src_mask if mask is None else (src_mask | mask)
        bkg = Background2D(
            data, box_size=box, filter_size=filt,
            sigma_clip=sigma_clip, bkg_estimator=estimator,
            mask=combined if combined.any() else None,
            exclude_percentile=90.0,
        )
        sub = data - bkg.background
        threshold = dcfg.nsigma * bkg.background_rms
        convolved = convolve(sub, kernel, normalize_kernel=True)
        # npixels is positional -> compatible across photutils versions
        segm = detect_sources(convolved, threshold, dcfg.npixels)
        if segm is None:
            break
        # dilate the source footprint a little so wings don't leak into the sky
        src_mask = binary_dilation(segm.data > 0, iterations=2)

    assert bkg is not None
    sub = data - bkg.background
    res = BackgroundResult(
        data_sub=sub.astype(np.float32),
        background=np.asarray(bkg.background, np.float32),
        rms=np.asarray(bkg.background_rms, np.float32),
        source_mask=src_mask,
        median=float(bkg.background_median),
        median_rms=float(bkg.background_rms_median),
    )
    log.info("Background: median %.2f ADU, RMS %.2f ADU; masked %.1f%% as sources",
             res.median, res.median_rms, 100.0 * src_mask.mean())
    return res
