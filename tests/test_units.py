"""Unit tests for the lower-level building blocks."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from arise.config import PipelineConfig, get_instrument, HeaderMap
from arise.fitsio import classify_frame, resolve
from arise.stages.calibrate import combine
from arise.catalogs import crossmatch


def test_config_roundtrip(tmp_path):
    cfg = PipelineConfig(instrument="dot_4kx4k")
    cfg.detect.nsigma = 4.2
    cfg.discovery.top_n = 17
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    loaded = PipelineConfig.from_yaml(p)
    assert loaded.instrument == "dot_4kx4k"
    assert loaded.detect.nsigma == 4.2
    assert loaded.discovery.top_n == 17


def test_instrument_registry():
    inst = get_instrument("dfot_2kx2k")
    assert inst.pixel_scale > 0
    assert inst.gain > 0
    with pytest.raises(KeyError):
        get_instrument("no_such_scope")


def test_classify_frame():
    hmap = HeaderMap()
    h = fits.Header()
    h["IMAGETYP"] = "Bias Frame"
    assert classify_frame(h, hmap) == "bias"
    h["IMAGETYP"] = "object"
    assert classify_frame(h, hmap) == "light"
    h["IMAGETYP"] = "Dome Flat"
    assert classify_frame(h, hmap) == "flat"
    # exposure-time fallback: zero-second unlabelled frame -> bias
    h2 = fits.Header()
    h2["EXPTIME"] = 0.0
    assert classify_frame(h2, hmap) == "bias"


def test_resolve_first_present():
    h = fits.Header()
    h["EXPOSURE"] = 30.0
    assert resolve(h, ("EXPTIME", "EXPOSURE"), None) == 30.0
    assert resolve(h, ("NOPE",), "default") == "default"


def test_sigma_clipped_combine_rejects_outliers():
    rng = np.random.default_rng(1)
    stack = [rng.normal(100.0, 2.0, (16, 16)).astype(np.float32) for _ in range(7)]
    stack[3][8, 8] = 50000.0            # a cosmic-ray-like outlier
    out = combine(stack, method="mean", sigma=3.0)
    assert abs(out[8, 8] - 100.0) < 3.0, "outlier not rejected by sigma clipping"


def test_combine_single_frame():
    f = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = combine([f])
    assert np.array_equal(out, f)


def test_crossmatch_basic():
    ra1 = np.array([10.0, 20.0, 30.0])
    dec1 = np.array([0.0, 0.0, 0.0])
    # catalogue: matches for #0 and #2 (within 1"), #1 far away
    ra2 = np.array([10.0 + 0.2 / 3600, 200.0, 30.0 - 0.1 / 3600])
    dec2 = np.array([0.0, 0.0, 0.0])
    idx, sep, matched = crossmatch(ra1, dec1, ra2, dec2, radius_arcsec=1.0)
    assert matched[0] and matched[2] and not matched[1]
    assert idx[0] == 0 and idx[2] == 2


def test_mpc_80col_format():
    """Every MPC data line must be exactly 80 chars and end with the obs code,
    regardless of designation length."""
    from arise.stages.dossier import mpc_80col
    dets = [
        {"frame": "f0", "frame_index": 0, "time_min": 0.0,
         "ra": 132.80, "dec": 11.59, "snr": 90, "mag_calib": 18.2},
        {"frame": "f1", "frame_index": 1, "time_min": 8.0,
         "ra": 132.8009, "dec": -11.5905, "snr": 92, "mag_calib": 18.25},
    ]
    dob = {"f0": "2026-07-04T18:00:00", "f1": "2026-07-04T18:08:00"}
    for desig in ("ARI0001", "ARISE205", "X"):
        out = mpc_80col(dets, dob, designation=desig)
        data_lines = [l for l in out.splitlines() if l.startswith("     ")]
        assert len(data_lines) == 2
        for l in data_lines:
            assert len(l) == 80, f"line not 80 cols ({len(l)}): {l!r}"
            assert l.endswith("XXX"), f"obs code truncated: {l!r}"


def test_motion_fit_prediction():
    from arise.stages.dossier import fit_motion
    # constant velocity: 0.001 deg/frame in RA over 8-minute cadence
    dets = [{"frame": f"f{i}", "frame_index": i, "time_min": 8.0 * i,
             "ra": 132.0 + 0.001 * i, "dec": 11.0, "snr": 50} for i in range(4)]
    m = fit_motion(dets)
    assert m["n_epochs"] == 4
    # +24h prediction continues the straight line
    exp_ra = 132.0 + 0.001 * (3 + 1440.0 / 8.0)
    assert abs(m["predictions"]["+24h"]["ra"] - exp_ra) < 1e-6
    assert m["fit_rms_arcsec"] < 0.01


def test_crossmatch_empty():
    idx, sep, matched = crossmatch(np.array([]), np.array([]),
                                   np.array([1.0]), np.array([1.0]), 1.0)
    assert len(idx) == 0 and not matched.any()
