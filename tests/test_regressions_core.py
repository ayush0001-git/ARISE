"""Regression tests pinning audit fixes in the core reduction modules.

Each test guards one confirmed-and-fixed defect from the 2026-07 audit
(finding numbers referenced in docstrings). They exercise the code as it is
NOW, so any silent re-introduction of the old behavior fails loudly.

No network access: everything runs on tiny in-memory arrays / tmp_path FITS.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from arise.catalogs import crossmatch
from arise.config import (CosmicRayConfig, DetectConfig, PhotometryConfig,
                          get_instrument)
from arise.fitsio import FrameMeta, _coerce_coord, read_meta
from arise.stages.astrometry import _refine_and_validate
from arise.stages.background import BackgroundResult
from arise.stages.calibrate import Masters, build_masters, reduce_frame
from arise.stages.cosmicray import reject_cosmic_rays
from arise.stages.ingest import FrameSet
from arise.stages.photometry import calibrate_photometry
from arise.stages.sources import extract_sources


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_fits(path, data, **cards):
    hdr = fits.Header()
    for k, v in cards.items():
        hdr[k] = v
    fits.PrimaryHDU(data=np.asarray(data, np.float32), header=hdr).writeto(path)
    return path


def _light_meta(name="sci.fits", **kw):
    return FrameMeta(path=Path(name), ftype="light", **kw)


def _phot_catalog(n=10, zp_true=25.0):
    """Tiny source catalog + matching reference with a known exact zero point."""
    ref_mag = np.linspace(12.0, 19.0, n)
    flux = 10.0 ** (-0.4 * (ref_mag - zp_true))     # exptime=1 -> mag_inst = ref - ZP
    ra = 150.0 + np.arange(n) * 30.0 / 3600.0       # 30 arcsec apart
    dec = np.full(n, 20.0)
    df = pd.DataFrame({
        "id": np.arange(n) + 1,
        "x": 50.0 * np.arange(n) + 10.0,            # 50 px apart -> isolated
        "y": 50.0 * np.arange(n) + 10.0,
        "flux": flux, "flux_err": flux / 100.0, "snr": np.full(n, 100.0),
        "flux_aper": flux, "flux_aper_err": flux / 100.0,
        "snr_aper": np.full(n, 100.0),
        "fwhm": np.full(n, 3.0),
        "ellipticity": np.full(n, 0.1), "elongation": np.full(n, 1.1),
        "peak": flux / 10.0, "area": np.full(n, 20.0),
        "ra": ra, "dec": dec,
        "flag_edge": np.zeros(n, bool), "flag_saturated": np.zeros(n, bool),
        "frame": "sci_0001.fits",
    })
    reference = pd.DataFrame({"ra": ra, "dec": dec, "mag": ref_mag})
    return df, reference, ref_mag


def _tan_wcs(crval_ra, crval_dec, scale_arcsec=1.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [crval_ra, crval_dec]
    w.wcs.crpix = [50.0, 50.0]
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    return w


# --------------------------------------------------------------------------- #
# findings #2 / #8 -- fitsio._coerce_coord: sexagesimal RA is hours (x15)
# --------------------------------------------------------------------------- #
def test_coerce_coord_colon_sexagesimal_ra_is_hours():
    """#2/#8: 'HH:MM:SS' RA strings must convert hours->degrees (x15), not be
    read as degrees (the pre-fix behavior returned 6.75 for this input)."""
    val = _coerce_coord("06:45:08.9", is_ra=True)
    assert val == pytest.approx((6 + 45 / 60 + 8.9 / 3600) * 15.0, abs=1e-9)
    assert val == pytest.approx(101.28708333, abs=1e-6)


def test_coerce_coord_space_sexagesimal_ra_is_hours():
    """#2/#8: MaxIm/SBIG-style 'HH MM SS' OBJCTRA strings get the same x15."""
    val = _coerce_coord("06 45 08.9", is_ra=True)
    assert val == pytest.approx(101.28708333, abs=1e-6)


def test_coerce_coord_decimal_and_numeric_ra_stay_degrees():
    """#2/#8: decimal-degree strings and numeric values (CRVAL1-style) must
    NOT be multiplied by 15 -- they are already degrees."""
    assert _coerce_coord("101.287", is_ra=True) == pytest.approx(101.287)
    assert _coerce_coord(101.287, is_ra=True) == pytest.approx(101.287)
    assert _coerce_coord(120, is_ra=True) == pytest.approx(120.0)
    # sexagesimal with a leading field >= 24 can only be degrees already
    assert _coerce_coord("101:17:13.5", is_ra=True) == pytest.approx(
        101 + 17 / 60 + 13.5 / 3600, abs=1e-9)


def test_coerce_coord_dec_sexagesimal_never_scaled():
    """#2/#8: DEC sexagesimal is degrees; no x15, sign preserved."""
    assert _coerce_coord("-16:42:58") == pytest.approx(-(16 + 42 / 60 + 58 / 3600), abs=1e-9)
    assert _coerce_coord("-16 42 58") == pytest.approx(-16.71611111, abs=1e-6)
    assert _coerce_coord("16:42:58") == pytest.approx(16.71611111, abs=1e-6)


def test_read_meta_wires_is_ra_flag(tmp_path):
    """#2/#8: read_meta must pass is_ra=True for the RA keyword only, so a
    standard sexagesimal RA/DEC header pair lands ~95 deg apart, not ~10."""
    inst = get_instrument("generic")
    p1 = _write_fits(tmp_path / "a.fits", np.zeros((4, 4)), IMAGETYP="object",
                     EXPTIME=60.0, RA="06:45:08.9", DEC="-16:42:58")
    m1 = read_meta(p1, inst)
    assert m1.ra == pytest.approx(101.28708333, abs=1e-5)
    assert m1.dec == pytest.approx(-16.71611111, abs=1e-5)

    # OBJCTRA / OBJCTDEC space-separated variant
    p2 = _write_fits(tmp_path / "b.fits", np.zeros((4, 4)), IMAGETYP="object",
                     EXPTIME=60.0, OBJCTRA="06 45 08.9", OBJCTDEC="-16 42 58")
    m2 = read_meta(p2, inst)
    assert m2.ra == pytest.approx(101.28708333, abs=1e-5)
    assert m2.dec == pytest.approx(-16.71611111, abs=1e-5)

    # numeric CRVAL1/CRVAL2 are already degrees and must stay unscaled
    p3 = _write_fits(tmp_path / "c.fits", np.zeros((4, 4)), IMAGETYP="object",
                     EXPTIME=60.0, CRVAL1=101.287, CRVAL2=-16.716)
    m3 = read_meta(p3, inst)
    assert m3.ra == pytest.approx(101.287, abs=1e-9)
    assert m3.dec == pytest.approx(-16.716, abs=1e-9)


# --------------------------------------------------------------------------- #
# finding #4 -- catalogs.crossmatch: NaN coordinates
# --------------------------------------------------------------------------- #
def test_crossmatch_nan_rows_map_to_original_reference_indices():
    """#4: NaN rows in either set must not raise, and finite rows must match
    back to ORIGINAL set-2 indices (a subset-relative index would silently
    assign the wrong reference magnitudes downstream)."""
    ra1 = np.array([10.0 + 0.2 / 3600, np.nan, 20.0 - 0.1 / 3600])
    dec1 = np.array([0.0, 5.0, 0.0])
    # set 2 has non-finite rows at original indices 0 and 2
    ra2 = np.array([np.nan, 10.0, 300.0, 20.0])
    dec2 = np.array([0.0, 0.0, np.nan, 0.0])
    idx, sep, matched = crossmatch(ra1, dec1, ra2, dec2, radius_arcsec=1.0)
    assert matched[0] and matched[2]
    assert idx[0] == 1, "index must be in the ORIGINAL set-2 frame, not the finite subset"
    assert idx[2] == 3
    # the NaN source row comes back unmatched, not crashed
    assert not matched[1]
    assert idx[1] == -1
    assert np.isinf(sep[1])


def test_crossmatch_all_nan_inputs_return_unmatched():
    """#4: fully non-finite inputs on either side yield the no-match contract
    instead of astropy's ValueError ('cannot contain NaN entries')."""
    nan2 = np.full(2, np.nan)
    finite_ra = np.array([10.0, 20.0])
    finite_dec = np.zeros(2)
    idx, sep, matched = crossmatch(nan2, nan2, finite_ra, finite_dec, 1.0)
    assert (idx == -1).all() and not matched.any() and np.isinf(sep).all()
    idx, sep, matched = crossmatch(finite_ra, finite_dec, nan2, nan2, 1.0)
    assert (idx == -1).all() and not matched.any() and np.isinf(sep).all()


def test_crossmatch_empty_input_contract():
    """#4: the pre-existing empty-input contract must survive the NaN fix."""
    e = np.array([])
    idx, sep, matched = crossmatch(e, e, np.array([1.0]), np.array([1.0]), 1.0)
    assert len(idx) == 0 and len(sep) == 0 and len(matched) == 0
    idx, sep, matched = crossmatch(np.array([1.0]), np.array([1.0]), e, e, 1.0)
    assert (idx == -1).all() and not matched.any()


# --------------------------------------------------------------------------- #
# finding #3 -- photometry: no -k*(X-1) double extinction correction
# --------------------------------------------------------------------------- #
def test_calibrated_mags_have_no_extinction_term_at_high_airmass():
    """#3: the per-frame ZP fit against in-frame catalog stars already absorbs
    extinction, so mag_calib must reproduce the reference magnitudes exactly
    even at airmass 2.0 (the old code subtracted k*(X-1)=0.25 mag in B)."""
    df, reference, ref_mag = _phot_catalog(n=10, zp_true=25.0)
    meta = _light_meta("sci_b.fits", exptime=1.0, filt="B", airmass=2.0)
    out, res = calibrate_photometry(df, meta, get_instrument("generic"),
                                    PhotometryConfig(), reference)
    assert res.calibrated
    assert res.airmass == pytest.approx(2.0)
    assert res.extinction == pytest.approx(0.25)     # recorded for QA only
    assert res.zeropoint == pytest.approx(25.0, abs=1e-6)
    # calibrated mags == catalog mags; any -k*(X-1) term would shift all
    # of these by exactly 0.25 mag and fail this assertion
    assert np.allclose(out["mag_calib"].to_numpy(), ref_mag, atol=1e-6)


# --------------------------------------------------------------------------- #
# finding #39 -- photometry: non-positive flux -> NaN magnitudes
# --------------------------------------------------------------------------- #
def test_nonpositive_flux_yields_nan_mags_and_does_not_poison_zp():
    """#39: negative/zero aperture fluxes must give NaN mag_inst/mag_calib
    (not the old clip-to-1e-9 junk of ~27.5+ZP), be excluded from the ZP
    sample, and leave the limiting-magnitude fallback sane."""
    df, reference, ref_mag = _phot_catalog(n=10, zp_true=25.0)
    # two junk rows that pass every calibrator gate except a defined magnitude,
    # matched to absurdly bright reference stars so any leakage into the ZP
    # sample or the limiting-mag percentile is detectable
    extra = pd.DataFrame({
        "id": [90, 91], "x": [1000.0, 1200.0], "y": [1000.0, 1200.0],
        "flux": [-5.0, 0.0], "flux_err": [1.0, 1.0], "snr": [100.0, 100.0],
        "flux_aper": [-5.0, 0.0], "flux_aper_err": [1.0, 1.0],
        "snr_aper": [50.0, 50.0],
        "fwhm": [3.0, 3.0], "ellipticity": [0.1, 0.1], "elongation": [1.1, 1.1],
        "peak": [1.0, 1.0], "area": [20.0, 20.0],
        "ra": [151.0, 151.1], "dec": [20.0, 20.0],
        "flag_edge": [False, False], "flag_saturated": [False, False],
        "frame": ["sci_0001.fits"] * 2,
    })
    cat = pd.concat([df, extra], ignore_index=True)
    ref = pd.concat([reference, pd.DataFrame(
        {"ra": [151.0, 151.1], "dec": [20.0, 20.0], "mag": [5.0, 5.0]})],
        ignore_index=True)
    meta = _light_meta("sci_v.fits", exptime=1.0, filt="V", airmass=1.0)
    out, res = calibrate_photometry(cat, meta, get_instrument("generic"),
                                    PhotometryConfig(), ref)
    # non-positive fluxes have no magnitude
    assert np.isnan(out.loc[10, "mag_inst"]) and np.isnan(out.loc[10, "mag_calib"])
    assert np.isnan(out.loc[11, "mag_inst"]) and np.isnan(out.loc[11, "mag_calib"])
    # the NaN rows never entered the ZP fit
    assert res.calibrated
    assert res.n_calib == 10
    assert res.zeropoint == pytest.approx(25.0, abs=1e-6)
    # limiting-mag fallback percentile is over real mags, not ~50-mag junk
    assert np.isfinite(res.limiting_mag)
    assert res.limiting_mag < 25.0


# --------------------------------------------------------------------------- #
# finding #11 -- calibrate.reduce_frame: variance propagation
# --------------------------------------------------------------------------- #
def test_variance_carries_inverse_flat_squared(tmp_path):
    """#11: after dividing the signal by flat=0.7, both the read-noise and
    Poisson variance terms must carry the 1/flat^2 factor (the old code
    computed variance from the post-flat signal with no propagation)."""
    p = _write_fits(tmp_path / "raw.fits", np.full((16, 16), 100.0))
    meta = _light_meta(p, exptime=60.0, filt="V", gain=2.0, read_noise=8.0)
    masters = Masters(flats={"V": np.full((16, 16), 0.7, np.float32)})
    red = reduce_frame(meta, masters, get_instrument("generic"))
    assert np.allclose(red.data, 100.0 / 0.7, rtol=1e-5)
    expected = ((8.0 / 2.0) ** 2 + 100.0 / 2.0) / 0.7 ** 2
    assert np.allclose(red.variance, expected, rtol=1e-4), (
        f"variance {red.variance[0, 0]:.3f} != analytic {expected:.3f} "
        f"(missing 1/flat^2 gives {(8.0 / 2.0) ** 2 + (100.0 / 0.7) / 2.0:.3f})")


def test_variance_includes_dark_shot_noise(tmp_path):
    """#11: when a dark is subtracted, the shot noise of the removed dark
    charge (dark_rate*exptime/gain in ADU^2) must appear in the variance."""
    p = _write_fits(tmp_path / "raw_d.fits", np.full((16, 16), 100.0))
    meta = _light_meta(p, exptime=60.0, filt="V", gain=2.0, read_noise=8.0)
    masters = Masters(dark_rate=np.full((16, 16), 0.05, np.float32),
                      dark_exptime=60.0)
    red = reduce_frame(meta, masters, get_instrument("generic"))
    # data = 100 - 0.05*60 = 97 ADU
    assert np.allclose(red.data, 97.0, rtol=1e-5)
    # var = (rdn/g)^2 + S/g + dark*t/g = 16 + 48.5 + 1.5
    assert np.allclose(red.variance, 16.0 + 48.5 + 1.5, rtol=1e-4), \
        "dark shot-noise term (dark_rate*exptime/gain) missing from variance"


# --------------------------------------------------------------------------- #
# finding #37 -- master-bias HISTORY provenance
# --------------------------------------------------------------------------- #
def test_master_bias_history_says_sigma_clipped_mean(tmp_path):
    """#37: the master-bias HISTORY card must describe the actual combine
    (sigma-clipped mean), not the old incorrect 'median of N frames'."""
    rng = np.random.default_rng(7)
    fs = FrameSet()
    for i in range(3):
        p = tmp_path / f"bias_{i}.fits"
        _write_fits(p, rng.normal(400.0, 2.0, (16, 16)))
        fs.bias.append(FrameMeta(path=p, ftype="bias", exptime=0.0))
    out_dir = tmp_path / "master"
    build_masters(fs, get_instrument("generic"), out_dir=out_dir)
    hdr = fits.getheader(out_dir / "master_bias.fits")
    history = " ".join(str(card) for card in hdr.get("HISTORY", []))
    assert "sigma-clipped mean" in history
    assert "median of" not in history


# --------------------------------------------------------------------------- #
# finding #38 -- low-response flat pixels join the bad-pixel mask
# --------------------------------------------------------------------------- #
def test_low_response_flat_pixels_enter_bad_pixel_mask(tmp_path):
    """#38: a dead pixel (flat response < 0.05) must be flagged in the
    bad-pixel mask (created even without darks) before being neutralised to
    1.0 in the master flat -- the old code silently reset it to 1.0 only."""
    rng = np.random.default_rng(11)
    fs = FrameSet()
    for i in range(3):
        data = rng.normal(10000.0, 20.0, (16, 16)).astype(np.float32)
        data[5, 7] = 10.0                       # dead pixel: ~0.001 response
        p = tmp_path / f"flat_{i}.fits"
        _write_fits(p, data)
        fs.flat.append(FrameMeta(path=p, ftype="flat", exptime=1.0, filt="V"))
    m = build_masters(fs, get_instrument("generic"))   # no darks -> mask was None
    assert m.bad_pixel_mask is not None, "mask must be created when no darks exist"
    assert m.bad_pixel_mask[5, 7], "low-response flat pixel not flagged bad"
    assert not m.bad_pixel_mask[0, 0]
    # the flat itself is still neutralised so reduction never divides by ~0
    assert m.flats["V"][5, 7] == pytest.approx(1.0)
    assert abs(m.flats["V"][0, 0] - 1.0) < 0.05


# --------------------------------------------------------------------------- #
# finding #36 -- different-filter flat substitution logs a WARNING
# --------------------------------------------------------------------------- #
def test_flat_filter_fallback_logs_warning(caplog, monkeypatch):
    """#36: applying the only available flat to a science frame of a different
    filter must emit a WARNING naming both filters (was silent pre-fix)."""
    # the arise root logger may have propagate=False if setup_logging ran
    monkeypatch.setattr(logging.getLogger("arise"), "propagate", True)
    flat_v = np.ones((4, 4), np.float32)
    m = Masters(flats={"V": flat_v})
    with caplog.at_level(logging.WARNING, logger="arise.calibrate"):
        flat = m.flat_for("R")
    assert flat is flat_v, "fallback must still return the only flat"
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("'R'" in msg and "'V'" in msg for msg in warnings), \
        f"no WARNING naming both filters; got {warnings!r}"
    # an exact filter match stays silent
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="arise.calibrate"):
        assert m.flat_for("V") is flat_v
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --------------------------------------------------------------------------- #
# finding #12 -- cosmicray: satlevel passed in ELECTRONS
# --------------------------------------------------------------------------- #
def test_detect_cosmics_receives_satlevel_in_electrons(monkeypatch):
    """#12: astroscrappy compares satlevel against the gain-multiplied image,
    so the ADU saturation ceiling must be converted to electrons (x gain);
    the old ADU value masked bright star cores as 'saturated'."""
    import astroscrappy

    captured = {}

    def fake_detect_cosmics(data, **kwargs):
        captured.update(kwargs)
        return np.zeros(data.shape, bool), np.asarray(data, np.float32)

    monkeypatch.setattr(astroscrappy, "detect_cosmics", fake_detect_cosmics)
    inst = get_instrument("generic")            # saturation = 60000 ADU
    meta = _light_meta("cr.fits", gain=2.0, read_noise=6.0, exptime=60.0)
    data = np.full((16, 16), 100.0, np.float32)
    res = reject_cosmic_rays(data, meta, inst, CosmicRayConfig())
    assert captured["gain"] == pytest.approx(2.0)
    assert captured["satlevel"] == pytest.approx(60000.0 * 2.0), \
        "satlevel must be saturation_ADU * gain (electrons), not raw ADU"
    assert res.n_flagged == 0


# --------------------------------------------------------------------------- #
# finding #13 -- sources.flag_saturated accounts for the subtracted sky
# --------------------------------------------------------------------------- #
def test_flag_saturated_on_sky_subtracted_peak():
    """#13: a star saturated in RAW ADU whose peak looks modest after a bright
    sky (4000 ADU) was subtracted must still be flagged; the local background
    has to be added back before comparing to the saturation threshold."""
    inst = get_instrument("generic")            # saturation 60000 ADU
    sky = 4000.0
    ny = nx = 64
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

    def star(x0, y0, amp, sig=1.6):
        return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))

    # star A: raw peak ~59600 ADU (saturated); appears as 55600 after the
    # 4000 ADU sky subtraction -- below the raw threshold 0.95*(60000-400)=56620,
    # so the pre-fix comparison on the subtracted peak missed it.
    data_sub = star(20, 20, 55600.0) + star(45, 45, 1200.0)
    bkg = BackgroundResult(
        data_sub=data_sub.astype(np.float32),
        background=np.full((ny, nx), sky, np.float32),
        rms=np.full((ny, nx), 5.0, np.float32),
        source_mask=np.zeros((ny, nx), bool),
        median=sky, median_rms=5.0,
    )
    variance = np.full((ny, nx), 25.0, np.float32)
    meta = _light_meta("sat.fits", exptime=60.0)
    ext = extract_sources(bkg, variance, meta, inst, DetectConfig())
    df = ext.catalog
    assert len(df) >= 2, "both synthetic stars should be detected"
    row_a = df.iloc[np.argmin((df["x"] - 20) ** 2 + (df["y"] - 20) ** 2)]
    row_b = df.iloc[np.argmin((df["x"] - 45) ** 2 + (df["y"] - 45) ** 2)]
    assert bool(row_a["flag_saturated"]), \
        "raw-saturated star escaped the flag under a bright subtracted sky"
    assert not bool(row_b["flag_saturated"]), "modest star wrongly flagged"


# --------------------------------------------------------------------------- #
# finding #14 -- astrometry._refine_and_validate rollback
# --------------------------------------------------------------------------- #
def test_refinement_kept_when_it_improves():
    """#14: a genuine bulk CRVAL offset must be corrected and the refined WCS
    kept (refined=True, RMS drops)."""
    true_wcs = _tan_wcs(150.0, 20.0)
    offset_wcs = _tan_wcs(150.0, 20.0 + 2.0 / 3600)     # +2 arcsec dec error
    xs, ys = np.meshgrid([20.0, 50.0, 80.0], [20.0, 50.0, 80.0])
    x, y = xs.ravel(), ys.ravel()
    ra, dec = true_wcs.all_pix2world(x, y, 0)
    catalog = pd.DataFrame({"x": x, "y": y})
    reference = pd.DataFrame({"ra": ra, "dec": dec})
    meta = _light_meta("astro.fits")
    res = _refine_and_validate(offset_wcs, catalog, reference, meta, "header")
    assert res.refined is True
    assert res.n_matched == 9
    assert res.residual_rms_arcsec < 0.3
    # the returned WCS was actually shifted back onto the truth
    assert abs(res.wcs.wcs.crval[1] - 20.0) * 3600.0 < 0.3


def test_refinement_rolled_back_when_it_degrades():
    """#14: if the median-offset shift LOSES matches (skewed by inconsistent
    pairings), the ORIGINAL WCS and its QA numbers must come back with
    refined=False -- the old code returned the degraded WCS unconditionally."""
    w = _tan_wcs(150.0, 20.0)
    x = np.array([10.0, 90.0, 10.0, 90.0, 50.0])
    y = np.array([10.0, 10.0, 90.0, 90.0, 50.0])
    ra, dec = w.all_pix2world(x, y, 0)
    # inconsistent per-star offsets: median (+2") helps 3 stars but pushes the
    # other two from 1.5" to 3.5" -- outside the 3" match radius -> n drops 5->3
    delta_arcsec = np.array([2.0, 2.0, 2.0, -1.5, -1.5])
    reference = pd.DataFrame({"ra": ra, "dec": dec + delta_arcsec / 3600.0})
    catalog = pd.DataFrame({"x": x, "y": y})
    meta = _light_meta("astro2.fits")
    res = _refine_and_validate(w, catalog, reference, meta, "header")
    assert res.refined is False, "degrading refinement must be rolled back"
    # original WCS returned untouched
    assert np.allclose(res.wcs.wcs.crval, [150.0, 20.0], atol=1e-9)
    # original QA numbers reported, not the degraded ones
    assert res.n_matched == 5
    rms0 = float(np.sqrt(np.mean(delta_arcsec ** 2)))
    assert res.residual_rms_arcsec == pytest.approx(rms0, abs=0.05)
