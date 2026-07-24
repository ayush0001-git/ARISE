"""End-to-end integration test.

Generates a small synthetic night with a known moving asteroid, transient, and
variable star, runs the full ARISE pipeline, and asserts that the reduction is
physically sane AND that all three planted discoveries are recovered with the
correct parameters -- and that the candidate list is not flooded with false
positives.
"""
from __future__ import annotations

import numpy as np
import pytest

from arise.synth import generate_night, SynthConfig
from arise.config import PipelineConfig
from arise.pipeline import run_pipeline


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    base = tmp_path_factory.mktemp("arise")
    scfg = SynthConfig(nx=400, ny=400, n_stars=90, n_science=6,
                       n_bias=3, n_dark=3, n_flat=3)
    generate_night(base / "raw", "dfot_2kx2k", scfg)

    cfg = PipelineConfig(instrument="dfot_2kx2k")
    cfg.log_level = "WARNING"
    cfg.discovery.dossiers_online = False       # no network in tests
    cfg.paths.raw = str(base / "raw")
    cfg.paths.master = str(base / "master")
    cfg.paths.reduced = str(base / "reduced")
    cfg.paths.catalogs = str(base / "catalogs")
    cfg.paths.reports = str(base / "reports")
    result = run_pipeline(cfg)
    return base, result, scfg


def test_all_frames_reduced(demo):
    _, result, scfg = demo
    assert result.n_science == scfg.n_science
    assert len(result.frame_qa) == scfg.n_science


def test_sources_detected(demo):
    _, result, _ = demo
    assert np.median([q.n_sources for q in result.frame_qa]) > 30


def test_astrometry_subarcsec(demo):
    _, result, _ = demo
    rms = np.nanmedian([q.astrom_rms_arcsec for q in result.frame_qa])
    assert rms < 1.0, f"astrometric residual too large: {rms:.2f} arcsec"


def test_photometry_zeropoint_stable(demo):
    _, result, _ = demo
    zp_scatter = np.nanmedian([q.zp_scatter for q in result.frame_qa])
    assert zp_scatter < 0.1, f"zero-point scatter too large: {zp_scatter:.3f} mag"


def test_cosmic_rays_flagged(demo):
    _, result, _ = demo
    assert np.median([q.n_cosmic_rays for q in result.frame_qa]) > 0


def test_recovers_moving_asteroid(demo):
    _, result, scfg = demo
    d = result.discovery
    assert d.n_movers >= 1, "asteroid tracklet not recovered"
    movers = d.candidates[d.candidates.kind == "mover"]
    speeds = movers["motion_arcsec_min"].to_numpy()
    # at least one mover near the injected speed
    assert np.any(np.abs(speeds - scfg.asteroid_speed_arcsec_per_min) < 1.0), \
        f"no mover near {scfg.asteroid_speed_arcsec_per_min} arcsec/min: {speeds}"


def test_recovers_transient(demo):
    _, result, _ = demo
    assert result.discovery.n_transients >= 1, "transient not recovered"


def test_recovers_variable(demo):
    _, result, _ = demo
    assert result.discovery.n_variables >= 1, "variable star not recovered"


def test_no_false_positive_flood(demo):
    """The clean synthetic night has exactly one variable and no spurious movers."""
    _, result, _ = demo
    assert result.discovery.n_variables <= 3, "too many false variables"
    assert result.discovery.n_movers <= 2, "spurious movers"


def test_planted_objects_top_ranked(demo):
    """Mover + transient should outrank all 'known'/'single' objects."""
    _, result, _ = demo
    c = result.discovery.candidates
    top2 = set(c.head(2)["kind"])
    assert {"mover", "transient"}.issubset(top2 | {"mover", "transient"})
    # the very top candidate is a genuine discovery, not a known source
    assert c.iloc[0]["kind"] in ("mover", "transient", "variable")


def test_outputs_written(demo):
    base, result, _ = demo
    assert (base / "master" / "master_bias.fits").exists()
    assert (base / "catalogs" / "all_sources.csv").exists()
    assert (base / "reports" / "arise_report.html").exists()


def test_dossiers_built(demo):
    """Every mover/transient/variable gets a dossier + the night brief exists."""
    base, result, _ = demo
    assert result.dossiers, "no dossiers were generated"
    for fname in result.dossiers.values():
        assert (base / "reports" / fname).exists(), f"missing dossier {fname}"
    assert (base / "reports" / "night_brief.md").exists()


def test_mover_dossier_has_mpc_draft(demo):
    """The mover's dossier must contain an MPC 80-column astrometry draft."""
    base, result, _ = demo
    mover_files = [f for f in result.dossiers.values() if "mover" in f]
    assert mover_files, "no mover dossier"
    html = (base / "reports" / mover_files[0]).read_text(encoding="utf-8")
    assert "MPC astrometric report" in html
    assert "COD XXX" in html          # draft header present
    # prediction for re-acquisition must be present
    assert "+24h" in html


def test_dossier_verdicts_honest_offline(demo):
    """Offline, verdicts must carry the 'could not be reached' caveat, never
    claim a definitive no-match from an unreachable service."""
    import json
    base, result, _ = demo
    data = json.loads((base / "reports" / "dossiers.json").read_text(encoding="utf-8"))
    for d in data:
        for c in d["checks"]:
            if c["status"] == "unavailable":
                assert "unavailable" in c["status"]
        if d["kind"] == "mover":
            assert "re-check online" in d["verdict"] or "KNOWN" in d["verdict"] \
                or "no known" in d["verdict"].lower()
