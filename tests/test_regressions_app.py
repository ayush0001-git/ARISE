"""Regression tests pinning audit fixes in the app-facing layers.

Covers the confirmed-and-fixed defects in:

* ``arise.webapp``  -- upload limits / ZIP-bomb budget (#9), filename
  sanitisation (#27), ZIP sub-folder flattening + zip-slip (#30), SSRF guard on
  /api/kb/add (#31), null JSON body on /api/demo (#55);
* ``arise.rag``     -- credential files never ingested (#1), add_url SSRF guard
  (#34) and download cap (#33), BOM/UTF-16 urls.txt tolerance (#32),
  dot-parent vault ingestion (#56), _stem co-tokenisation (#58);
* config / cli / keys -- header instrument_overrides (#26), unknown-key
  warnings (#53), --instrument override with --config (#35), config log_level
  survival (#57), non-mapping keys.yaml tolerance (#51);
* ``arise.pipeline`` -- one corrupt frame must not kill the night (#7), strict
  JSON qa_summary (#52), numeric MJD/JD DATE-OBS parsing (#24).

No test touches the network: requests / socket.getaddrinfo are monkeypatched
wherever a code path could otherwise reach out.
"""
from __future__ import annotations

import io
import json
import logging
import socket
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from astropy.io import fits


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _force_log_propagation(monkeypatch):
    """setup_logging() sets propagate=False on the 'arise' logger, which breaks
    caplog (whose handler sits on the root logger). Re-enable propagation for
    the duration of a test so warnings are capturable regardless of ordering."""
    monkeypatch.setattr(logging.getLogger("arise"), "propagate", True)


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, payload in entries:
            z.writestr(name, payload)
    return buf.getvalue()


def _boom_get(*args, **kwargs):
    pytest.fail(f"outbound requests.get was reached: args={args!r}")


# --------------------------------------------------------------------------- #
# webapp fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def webapp_env(monkeypatch, tmp_path):
    """Flask test client with WEB_DATA/ROOT sandboxed to tmp_path and the
    background pipeline launcher replaced by a recorder (no real reduction)."""
    from arise import webapp

    web_data = tmp_path / "webdata"
    web_data.mkdir()
    monkeypatch.setattr(webapp, "WEB_DATA", web_data)
    monkeypatch.setattr(webapp, "ROOT", tmp_path)
    monkeypatch.setattr(webapp, "_runs", {})
    launches: list[tuple] = []
    monkeypatch.setattr(
        webapp, "_launch_run",
        lambda run_id, instrument, demo, size=512: launches.append(
            (run_id, instrument, demo)))
    return SimpleNamespace(webapp=webapp, web_data=web_data,
                           client=webapp.app.test_client(),
                           launches=launches, tmp=tmp_path)


# --------------------------------------------------------------------------- #
# webapp #9 -- upload size limits + ZIP-bomb budget
# --------------------------------------------------------------------------- #
def test_max_content_length_configured():
    """#9: MAX_CONTENT_LENGTH must be set (Werkzeug's default None means an
    unbounded request body -> trivial memory/disk DoS)."""
    from arise import webapp
    assert webapp.app.config["MAX_CONTENT_LENGTH"] is not None
    assert webapp.app.config["MAX_CONTENT_LENGTH"] == webapp._MAX_UPLOAD_BYTES
    assert webapp.app.config["MAX_CONTENT_LENGTH"] > 0


def test_oversize_request_gets_json_413(webapp_env, monkeypatch):
    """#9: a request body over MAX_CONTENT_LENGTH is rejected with a JSON 413
    (the registered errorhandler), not an unhandled crash."""
    monkeypatch.setitem(webapp_env.webapp.app.config, "MAX_CONTENT_LENGTH", 500)
    resp = webapp_env.client.post(
        "/api/upload",
        data={"files": (io.BytesIO(b"x" * 5000), "big.fits")},
        content_type="multipart/form-data")
    assert resp.status_code == 413
    assert "error" in resp.get_json()
    assert not webapp_env.launches, "no pipeline run may start for a rejected upload"


def test_zip_bomb_per_entry_budget_rejected(webapp_env, monkeypatch):
    """#9: a ZIP whose entry declares more uncompressed bytes than the
    per-member budget is rejected 4xx and the orphan run dir is removed."""
    wa = webapp_env.webapp
    monkeypatch.setattr(wa, "_MAX_MEMBER_BYTES", 1000)
    monkeypatch.setattr(wa, "_MAX_EXTRACT_BYTES", 10000)
    payload = _zip_bytes([("bomb.fits", b"0" * 5000)])
    resp = webapp_env.client.post(
        "/api/upload",
        data={"files": (io.BytesIO(payload), "night.zip")},
        content_type="multipart/form-data")
    assert 400 <= resp.status_code < 500
    assert "error" in resp.get_json()
    assert not list(webapp_env.web_data.iterdir()), \
        "rejected upload must not leave a run directory behind"
    assert not webapp_env.launches


def test_zip_bomb_total_budget_rejected(webapp_env, monkeypatch):
    """#9: the *cumulative* uncompressed budget is enforced too -- many small
    entries cannot bypass the per-entry cap."""
    wa = webapp_env.webapp
    monkeypatch.setattr(wa, "_MAX_MEMBER_BYTES", 4000)
    monkeypatch.setattr(wa, "_MAX_EXTRACT_BYTES", 3000)
    payload = _zip_bytes([(f"f{i}.fits", b"0" * 1500) for i in range(3)])
    resp = webapp_env.client.post(
        "/api/upload",
        data={"files": (io.BytesIO(payload), "night.zip")},
        content_type="multipart/form-data")
    assert 400 <= resp.status_code < 500
    assert not list(webapp_env.web_data.iterdir()), \
        "partially extracted run dir must be cleaned up on rejection"


# --------------------------------------------------------------------------- #
# webapp #27 -- uploaded filename sanitisation
# --------------------------------------------------------------------------- #
def test_upload_filename_sanitized_on_disk(webapp_env):
    """#27: '../<script>x</script>.fits' must be stored inside the run's raw
    dir under a sanitized name -- no path escape, no raw '<' (the name later
    reaches the HTML report via FrameQA)."""
    resp = webapp_env.client.post(
        "/api/upload",
        data={"files": (io.BytesIO(b"not really fits"),
                        "../<script>x</script>.fits")},
        content_type="multipart/form-data")
    assert resp.status_code == 200
    run_id = resp.get_json()["run_id"]
    raw = webapp_env.web_data / run_id / "raw"
    stored = list(raw.iterdir())
    assert len(stored) == 1
    name = stored[0].name
    assert "<" not in name and ">" not in name
    assert "/" not in name and "\\" not in name and not name.startswith(".")
    assert stored[0].resolve().is_relative_to(raw.resolve())
    # nothing escaped the sandbox: only the webdata dir exists at tmp level
    assert sorted(p.name for p in webapp_env.tmp.iterdir()) == ["webdata"]


def test_upload_dotdot_filename_stays_inside_run_dir(webapp_env):
    """#27: a plain '../../evil.fits' traversal name is flattened to its
    basename inside the run dir."""
    resp = webapp_env.client.post(
        "/api/upload",
        data={"files": (io.BytesIO(b"x"), "../../evil.fits")},
        content_type="multipart/form-data")
    assert resp.status_code == 200
    run_id = resp.get_json()["run_id"]
    raw = webapp_env.web_data / run_id / "raw"
    assert (raw / "evil.fits").exists()
    assert not (webapp_env.tmp / "evil.fits").exists()
    assert not (webapp_env.web_data / "evil.fits").exists()


# --------------------------------------------------------------------------- #
# webapp #30 -- ZIP sub-folder entries kept distinct; zip-slip contained
# --------------------------------------------------------------------------- #
def test_zip_subfolder_entries_do_not_overwrite(webapp_env):
    """#30: bias/frame.fits and dark/frame.fits must become two distinct files
    (previously both flattened to raw/frame.fits, silently overwriting), and a
    '../evil.fits' member must stay inside the run dir."""
    payload = _zip_bytes([
        ("bias/frame.fits", b"BIAS-CONTENT"),
        ("dark/frame.fits", b"DARK-CONTENT"),
        ("../evil.fits", b"EVIL-CONTENT"),
    ])
    resp = webapp_env.client.post(
        "/api/upload",
        data={"files": (io.BytesIO(payload), "night.zip")},
        content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    run_id = body["run_id"]
    raw = webapp_env.web_data / run_id / "raw"
    stored = {p.name: p.read_bytes() for p in raw.iterdir()}
    # two distinct files with both payloads intact
    assert len([n for n in stored if "frame" in n]) == 2
    assert b"BIAS-CONTENT" in set(stored.values())
    assert b"DARK-CONTENT" in set(stored.values())
    # n_fits agrees with what actually landed on disk
    assert body["n_fits"] == len(stored) == 3
    # the traversal member stayed inside the run dir
    for p in raw.iterdir():
        assert p.resolve().is_relative_to(raw.resolve())
    assert not (webapp_env.tmp / "evil.fits").exists()
    assert not (webapp_env.web_data / "evil.fits").exists()


# --------------------------------------------------------------------------- #
# webapp #31 -- /api/kb/add SSRF guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", [
    "ftp://example.com/file.txt",
    "http://127.0.0.1/latest",
    "http://169.254.169.254/latest/meta-data/",
    "http://192.168.1.1/admin",
])
def test_kb_add_rejects_ssrf_targets(webapp_env, monkeypatch, url):
    """#31: /api/kb/add must 400 non-http schemes and loopback/link-local/
    private targets WITHOUT any outbound request (getaddrinfo on IP literals
    is local; requests.get is the actual outbound step and must not run)."""
    wa = webapp_env.webapp
    monkeypatch.setattr(requests, "get", _boom_get)
    monkeypatch.setattr(
        wa, "_get_kb",
        lambda rebuild=False: pytest.fail("KB reached for a blocked URL"))
    resp = webapp_env.client.post("/api/kb/add", json={"url": url})
    assert resp.status_code == 400
    assert "error" in resp.get_json()
    # nothing persisted for a rejected URL
    assert not (webapp_env.tmp / "kb" / "urls.txt").exists()


# --------------------------------------------------------------------------- #
# webapp #55 -- null JSON body on /api/demo
# --------------------------------------------------------------------------- #
def test_demo_null_json_body_does_not_500(webapp_env):
    """#55: POST /api/demo with body 'null' (valid JSON, decodes to None) must
    fall back to the default instrument instead of AttributeError -> 500."""
    resp = webapp_env.client.post("/api/demo", data="null",
                                  content_type="application/json")
    assert resp.status_code == 200
    assert "run_id" in resp.get_json()
    assert webapp_env.launches and webapp_env.launches[0][1] == "dfot_2kx2k"


def test_demo_missing_body_does_not_500(webapp_env):
    """#55 (companion): a POST with no body at all is equally tolerated."""
    resp = webapp_env.client.post("/api/demo")
    assert resp.status_code == 200
    assert "run_id" in resp.get_json()


# --------------------------------------------------------------------------- #
# rag #1 -- credentials never ingested into the knowledge base
# --------------------------------------------------------------------------- #
def test_build_default_kb_never_ingests_key_files(tmp_path, monkeypatch):
    """#1: config/keys.yaml (and secrets.yaml) must be excluded from KB
    ingestion while ordinary config YAMLs are still indexed -- otherwise raw
    API keys become retrievable chunks and get POSTed to third-party LLMs."""
    from arise.rag import build_default_kb

    monkeypatch.setattr(requests, "get", _boom_get)   # belt and braces
    root = tmp_path / "proj"
    (root / "config").mkdir(parents=True)
    secret = "nvapi-SUPER-SECRET-SENTINEL-0123456789"
    (root / "config" / "keys.yaml").write_text(
        "# credentials\nnvidia:\n  - " + secret + "\n" + "filler: x\n" * 30,
        encoding="utf-8")
    (root / "config" / "secrets.yaml").write_text(
        "token: " + secret + "\n" + "pad: y\n" * 30, encoding="utf-8")
    marker = "NORMAL-CONFIG-MARKER-XYZ"
    (root / "config" / "pipeline.yaml").write_text(
        f"instrument: generic  # {marker}\n"
        + "\n".join(f"comment_{i}: some pipeline option value" for i in range(30)),
        encoding="utf-8")

    kb = build_default_kb(root)
    all_text = " ".join(c.text for c in kb.chunks)
    sources = {c.source for c in kb.chunks}
    assert secret not in all_text, "secret leaked into a KB chunk"
    assert "keys.yaml" not in sources and "secrets.yaml" not in sources
    assert marker in all_text, "ordinary config YAML should still be ingested"


# --------------------------------------------------------------------------- #
# rag #34 -- add_url SSRF guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", [
    "ftp://example.com/file.txt",
    "file:///etc/passwd",
    "http://127.0.0.1:8080/internal",
    "http://localhost/internal",
    "http://169.254.169.254/latest/meta-data/",
    "http://192.168.1.1/router",
])
def test_add_url_refuses_private_and_nonhttp(monkeypatch, url):
    """#34: KnowledgeBase.add_url must refuse loopback/private/metadata hosts
    and non-http(s) schemes before any outbound request is made."""
    from arise.rag import KnowledgeBase

    monkeypatch.setattr(requests, "get", _boom_get)
    kb = KnowledgeBase()
    assert kb.add_url(url) == 0
    assert kb.chunks == []


# --------------------------------------------------------------------------- #
# rag #33 -- add_url download size cap
# --------------------------------------------------------------------------- #
class _FakeStreamResponse:
    """Minimal stand-in for a streaming requests.Response."""

    def __init__(self, headers: dict, n_chunks: int = 200,
                 chunk: bytes = b"tok " * 16384):
        self.headers = headers
        self.status_code = 200
        self.is_redirect = False
        self.encoding = "utf-8"
        self.bytes_yielded = 0
        self.iter_called = False
        self._n = n_chunks
        self._chunk = chunk

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=65536):
        self.iter_called = True
        for _ in range(self._n):
            self.bytes_yielded += len(self._chunk)
            yield self._chunk

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_public_dns(monkeypatch):
    """Resolve every hostname to a public address without touching real DNS."""
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))])


def test_add_url_streams_with_hard_byte_cap(monkeypatch):
    """#33: a response bigger than the cap is truncated mid-stream (no full
    buffering of a multi-GB body) and still yields chunks without crashing."""
    from arise.rag import KnowledgeBase, _KB_URL_MAX_BYTES

    _patch_public_dns(monkeypatch)
    fake = _FakeStreamResponse(headers={"content-type": "text/plain"})
    monkeypatch.setattr(requests, "get", lambda *a, **k: fake)
    kb = KnowledgeBase()
    n = kb.add_url("http://example.com/huge.txt")
    assert n > 0, "truncated fetch should still produce chunks"
    assert fake.iter_called
    # the stream stopped at (roughly) the cap, far below the 13 MB on offer
    assert fake.bytes_yielded < _KB_URL_MAX_BYTES + 2 * 65536


def test_add_url_rejects_declared_oversize_body(monkeypatch):
    """#33: a Content-Length above the cap is refused before reading the body."""
    from arise.rag import KnowledgeBase, _KB_URL_MAX_BYTES

    _patch_public_dns(monkeypatch)
    fake = _FakeStreamResponse(
        headers={"content-type": "text/plain",
                 "content-length": str(_KB_URL_MAX_BYTES * 10)})
    monkeypatch.setattr(requests, "get", lambda *a, **k: fake)
    kb = KnowledgeBase()
    assert kb.add_url("http://example.com/huge.bin") == 0
    assert not fake.iter_called, "body must not be read when declared oversize"


# --------------------------------------------------------------------------- #
# rag #32 -- BOM / UTF-16 kb files must not crash KB construction
# --------------------------------------------------------------------------- #
def test_urls_txt_with_utf8_bom_tolerated(tmp_path, monkeypatch):
    """#32: a UTF-8-BOM kb/urls.txt (classic Notepad) must not crash
    build_default_kb, and the BOM must not corrupt the first line (a comment
    here -- previously the BOM glued onto it and it was fetched as a URL)."""
    from arise.rag import build_default_kb

    monkeypatch.setattr(requests, "get", _boom_get)
    root = tmp_path / "proj"
    (root / "kb").mkdir(parents=True)
    (root / "kb" / "urls.txt").write_text("# only a comment, no urls\n",
                                          encoding="utf-8-sig")
    kb = build_default_kb(root)          # must not raise, must not fetch
    assert kb is not None


def test_urls_and_sources_txt_utf16_tolerated(tmp_path, monkeypatch):
    """#32: UTF-16-encoded kb/urls.txt + kb/sources.txt (PowerShell 5.1
    redirection default) must not raise UnicodeDecodeError -- the webapp warms
    the KB at startup, so this crash used to take down the whole console."""
    from arise.rag import build_default_kb

    monkeypatch.setattr(requests, "get", _boom_get)
    root = tmp_path / "proj"
    (root / "kb").mkdir(parents=True)
    (root / "kb" / "urls.txt").write_text("# a comment line\n", encoding="utf-16")
    (root / "kb" / "sources.txt").write_text("# another comment\n", encoding="utf-16")
    kb = build_default_kb(root)          # must not raise
    assert kb is not None


# --------------------------------------------------------------------------- #
# rag #56 -- vault under a dot-directory parent
# --------------------------------------------------------------------------- #
def test_vault_under_dot_parent_is_ingested(tmp_path):
    """#56: a vault living below a dot-prefixed parent (~/.notes/vault) must
    ingest normally; only dot-directories *inside* the vault (.obsidian) are
    skipped. Previously every note was dropped because the absolute path
    contained a dotted component."""
    from arise.rag import KnowledgeBase

    vault = tmp_path / ".notes" / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    note_text = ("Observation planning notes for the DFOT run: " +
                 "photometry calibration details " * 10)
    (vault / "note1.md").write_text(note_text, encoding="utf-8")
    hidden = "OBSIDIAN-INTERNAL-SENTINEL " * 20
    (vault / ".obsidian" / "workspace.md").write_text(hidden, encoding="utf-8")

    kb = KnowledgeBase()
    n = kb.add_folder(vault)
    assert n > 0, "notes under a dot-parent directory were not ingested"
    all_text = " ".join(c.text for c in kb.chunks)
    assert "Observation planning notes" in all_text
    assert "OBSIDIAN-INTERNAL-SENTINEL" not in all_text, \
        ".obsidian/ content must still be skipped"


# --------------------------------------------------------------------------- #
# rag #58 -- _stem co-tokenisation
# --------------------------------------------------------------------------- #
def test_stem_cotokenizes_morphological_families():
    """#58: discovery/discoveries/discovered/discovers must map to one token,
    and source/sources likewise, so queries match KB chunks across forms."""
    from arise.rag import _stem

    discovery_family = {_stem(w) for w in
                        ("discovery", "discoveries", "discovered", "discovers")}
    assert len(discovery_family) == 1, f"split tokens: {discovery_family}"
    source_family = {_stem(w) for w in ("source", "sources")}
    assert len(source_family) == 1, f"split tokens: {source_family}"
    # short words must survive the length guards untouched
    assert _stem("sky") == "sky"


# --------------------------------------------------------------------------- #
# config #26 -- instrument_overrides {'header': {...}} reaches the HeaderMap
# --------------------------------------------------------------------------- #
def test_instrument_header_override_applied():
    """#26: a 'header' block in instrument_overrides must merge into the
    resolved instrument's HeaderMap (previously silently dropped, so custom
    keyword maps for non-ARIES cameras were ignored)."""
    from arise.config import PipelineConfig

    cfg = PipelineConfig(
        instrument="generic",
        instrument_overrides={"gain": 3.3,
                              "header": {"exptime": ["MYEXPT"],
                                         "imagetyp": ["FRAMETYP"]}})
    inst = cfg.resolve_instrument()
    assert inst.header.exptime == ("MYEXPT",)
    assert inst.header.imagetyp == ("FRAMETYP",)
    assert inst.gain == 3.3
    # untouched HeaderMap fields keep their defaults (merge, not replace)
    assert "FILTER" in inst.header.filt
    # the config object itself must not be mutated by resolution
    assert "header" in cfg.instrument_overrides
    # resolving twice gives the same answer (no destructive pop on the config)
    assert cfg.resolve_instrument().header.exptime == ("MYEXPT",)


def test_instrument_header_override_from_dict_roundtrip():
    """#26 (YAML path): the same override arriving via from_dict (as YAML
    lists) is coerced to tuples and honoured."""
    from arise.config import PipelineConfig

    cfg = PipelineConfig.from_dict({
        "instrument": "dfot_2kx2k",
        "instrument_overrides": {"header": {"exptime": ["MYEXPT"]}}})
    inst = cfg.resolve_instrument()
    assert inst.header.exptime == ("MYEXPT",)
    assert inst.name == "dfot_2kx2k"


# --------------------------------------------------------------------------- #
# config #53 -- unknown config keys warn by name
# --------------------------------------------------------------------------- #
def test_unknown_nested_config_key_warns_and_keeps_default(monkeypatch, caplog):
    """#53: a typo'd key inside a section ('n_sigma' vs 'nsigma') must produce
    a warning naming the key instead of silently running with defaults."""
    from arise.config import PipelineConfig

    _force_log_propagation(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="arise"):
        cfg = PipelineConfig.from_dict({"detect": {"n_sigma": 9.9}})
    assert cfg.detect.nsigma == 3.0
    assert any("n_sigma" in r.getMessage() for r in caplog.records), \
        "warning must name the unknown key"


def test_unknown_toplevel_config_key_warns_by_name(monkeypatch, caplog):
    """#53: unknown top-level keys are reported by name too."""
    from arise.config import PipelineConfig

    _force_log_propagation(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="arise"):
        PipelineConfig.from_dict({"phootometry": {"min_ref_stars": 4}})
    assert any("phootometry" in r.getMessage() for r in caplog.records)


def test_non_dict_config_section_warns_and_keeps_dataclass(monkeypatch, caplog):
    """#53: 'detect: 5' must not replace the DetectConfig dataclass with an
    int -- the section is ignored with a warning."""
    from arise.config import PipelineConfig, DetectConfig

    _force_log_propagation(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="arise"):
        cfg = PipelineConfig.from_dict({"detect": 5})
    assert isinstance(cfg.detect, DetectConfig)
    assert any("detect" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# cli #35 / #57 -- --instrument / --log-level interplay with --config
# --------------------------------------------------------------------------- #
class _FakeResult:
    outputs: dict = {}
    discovery = None


def _run_cli(monkeypatch, argv):
    """Drive build_parser + cmd_run with run_pipeline replaced by a recorder."""
    import arise.pipeline
    from arise.cli import build_parser

    captured: dict = {}

    def fake_run(cfg):
        captured["cfg"] = cfg
        return _FakeResult()

    monkeypatch.setattr(arise.pipeline, "run_pipeline", fake_run)
    args = build_parser().parse_args(argv)
    rc = args.func(args)
    assert rc == 0
    return captured["cfg"]


def test_cli_instrument_flag_overrides_config(monkeypatch, tmp_path):
    """#35: 'arise run --config X --instrument Y' must reduce with Y (the
    override used to be a dead branch gated on 'not args.config')."""
    from arise.config import PipelineConfig

    p = tmp_path / "night.yaml"
    PipelineConfig(instrument="dfot_2kx2k").to_yaml(p)
    cfg = _run_cli(monkeypatch,
                   ["run", "--config", str(p), "--instrument", "dot_4kx4k"])
    assert cfg.instrument == "dot_4kx4k"


def test_cli_config_instrument_kept_without_flag(monkeypatch, tmp_path):
    """#35 (companion): without an explicit --instrument the config file's
    instrument must survive (the flag's old default 'generic' used to make
    user intent undetectable)."""
    from arise.cli import build_parser
    from arise.config import PipelineConfig

    args = build_parser().parse_args(["run", "--raw", "x"])
    assert args.instrument is None, "--instrument default must be None"
    p = tmp_path / "night.yaml"
    PipelineConfig(instrument="dfot_2kx2k").to_yaml(p)
    cfg = _run_cli(monkeypatch, ["run", "--config", str(p)])
    assert cfg.instrument == "dfot_2kx2k"


def test_cli_config_log_level_survives_absent_flag(monkeypatch, tmp_path):
    """#57: log_level set in the config YAML must be honoured when --log-level
    is not passed (it used to be clobbered by the flag's 'INFO' default)."""
    from arise.config import PipelineConfig

    p = tmp_path / "night.yaml"
    c = PipelineConfig(instrument="generic")
    c.log_level = "DEBUG"
    c.to_yaml(p)
    cfg = _run_cli(monkeypatch, ["run", "--config", str(p)])
    assert cfg.log_level == "DEBUG"


def test_cli_explicit_log_level_still_wins(monkeypatch, tmp_path):
    """#57 (companion): an explicit --log-level must override the config."""
    from arise.config import PipelineConfig

    p = tmp_path / "night.yaml"
    c = PipelineConfig(instrument="generic")
    c.log_level = "DEBUG"
    c.to_yaml(p)
    cfg = _run_cli(monkeypatch,
                   ["run", "--config", str(p), "--log-level", "WARNING"])
    assert cfg.log_level == "WARNING"


# --------------------------------------------------------------------------- #
# keys #51 -- non-mapping keys.yaml tolerated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("content", [
    "- just-a-list-item\n- another\n",       # parses to a list
    "just a plain string\n",                 # parses to a str
])
def test_keys_yaml_non_mapping_tolerated(tmp_path, monkeypatch, caplog, content):
    """#51: a keys.yaml that parses to a list/string must degrade gracefully
    with a warning, not AttributeError (which used to kill every CLI command)."""
    from arise.keys import load_keys

    for var in ("ASTROMETRY_NET_API_KEY", "NASA_API_KEY",
                "ANTHROPIC_API_KEY", "NVIDIA_API_KEYS"):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "proj"
    (root / "config").mkdir(parents=True)
    (root / "config" / "keys.yaml").write_text(content, encoding="utf-8")

    _force_log_propagation(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="arise"):
        loaded = load_keys(root)             # must not raise
    assert loaded == {}
    assert any("mapping" in r.getMessage() for r in caplog.records)
    assert "NVIDIA_API_KEYS" not in __import__("os").environ


# --------------------------------------------------------------------------- #
# pipeline #24 -- numeric MJD / JD DATE-OBS values
# --------------------------------------------------------------------------- #
def test_frame_time_min_parses_numeric_mjd_and_jd():
    """#24: bare-number MJD-OBS / JD strings must parse to real epoch minutes
    (they used to silently fall back to 'frame index = minutes')."""
    from arise.pipeline import _frame_time_min

    t_mjd = _frame_time_min("60123.456", 3)
    assert t_mjd == pytest.approx(60123.456 * 1440.0, abs=1e-3)
    assert t_mjd != 3.0

    # JD 2460123.956 is exactly MJD 60123.456: both forms must agree
    t_jd = _frame_time_min("2460123.956", 4)
    assert t_jd == pytest.approx(t_mjd, abs=1e-3)
    assert t_jd != 4.0


def test_frame_time_min_relative_minutes_correct():
    """#24: two MJD frames 5 minutes apart must differ by 5.0 epoch-minutes
    (this is what tracklet velocities in arcsec/min are computed from)."""
    from arise.pipeline import _frame_time_min

    t0 = _frame_time_min("60123.456", 0)
    t1 = _frame_time_min(str(60123.456 + 5.0 / 1440.0), 1)
    assert t1 - t0 == pytest.approx(5.0, abs=1e-3)


def test_frame_time_min_iso_and_fallback():
    """#24 (companion): ISO strings still parse, and empty/garbage values fall
    back to the frame index."""
    from astropy.time import Time
    from arise.pipeline import _frame_time_min

    iso = "2026-01-15T00:00:00"
    expect = float(Time(iso, format="isot", scale="utc").mjd * 1440.0)
    assert _frame_time_min(iso, 0) == pytest.approx(expect, abs=1e-6)
    assert _frame_time_min("", 7) == 7.0
    assert _frame_time_min("not a date", 2) == 2.0


# --------------------------------------------------------------------------- #
# pipeline #7 / #52 -- one corrupt frame must not kill the night;
#                      qa_summary.json must be strict JSON
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def corrupt_night(tmp_path_factory):
    """Smallest viable synthetic night (256x256, 4 science frames) with one
    science frame replaced by a shape-corrupted FITS, run through the full
    pipeline. Module-scoped: several tests share the single (slow) run."""
    from arise.synth import generate_night, SynthConfig
    from arise.config import PipelineConfig
    from arise.pipeline import run_pipeline

    mp = pytest.MonkeyPatch()
    for var in ("ASTROMETRY_NET_API_KEY", "NASA_API_KEY",
                "ANTHROPIC_API_KEY", "NVIDIA_API_KEYS"):
        mp.delenv(var, raising=False)
    try:
        base = tmp_path_factory.mktemp("corrupt_night")
        raw = base / "raw"
        scfg = SynthConfig(nx=256, ny=256, n_stars=70, n_science=4,
                           n_bias=3, n_dark=3, n_flat=3)
        generate_night(raw, "dfot_2kx2k", scfg)

        # corrupt one mid-sequence science frame: header intact (so it ingests
        # and classifies as 'light') but the data shape is wrong, which makes
        # calibration fail for exactly this frame inside the per-frame loop
        science = sorted(raw.glob("science_*.fits"))
        assert len(science) == 4
        bad = science[1]
        hdr = fits.getheader(bad)
        fits.PrimaryHDU(np.zeros((16, 16), np.float32),
                        header=hdr).writeto(bad, overwrite=True)

        cfg = PipelineConfig(instrument="dfot_2kx2k")
        cfg.log_level = "WARNING"
        cfg.discovery.dossiers = False
        cfg.discovery.dossiers_online = False        # no network in tests
        cfg.paths.raw = str(raw)
        cfg.paths.master = str(base / "master")
        cfg.paths.reduced = str(base / "reduced")
        cfg.paths.catalogs = str(base / "catalogs")
        cfg.paths.reports = str(base / "reports")
        result = run_pipeline(cfg)                   # must NOT raise (#7)
        yield base, result, bad.name
    finally:
        mp.undo()


def test_corrupt_frame_does_not_abort_night(corrupt_night):
    """#7: the run completes with the bad frame skipped -- the other frames
    are fully reduced instead of the whole night being lost."""
    base, result, bad_name = corrupt_night
    assert result.n_science == 4
    assert len(result.frame_qa) == 4
    failed = [q for q in result.frame_qa if not q.ok]
    assert len(failed) == 1
    assert failed[0].name == bad_name
    assert failed[0].error, "the failure reason must be recorded"
    good = [q for q in result.frame_qa if q.ok]
    assert len(good) == 3
    assert all(q.n_sources > 0 for q in good), "surviving frames must reduce"


def test_corrupt_frame_products_written(corrupt_night):
    """#7: the night-level products (report, QA, catalogs, reduced frames for
    the good frames) exist despite the failed frame."""
    base, result, bad_name = corrupt_night
    assert (base / "reports" / "arise_report.html").exists()
    assert (base / "reports" / "qa_summary.json").exists()
    assert (base / "catalogs" / "all_sources.csv").exists()
    reduced = {p.name for p in (base / "reduced").glob("reduced_*.fits")}
    assert len(reduced) == 3
    assert f"reduced_{bad_name}" not in reduced


def test_qa_summary_is_strict_json(corrupt_night):
    """#52: qa_summary.json must contain no bare NaN/Infinity tokens -- it has
    to parse under a strict-JSON consumer (jq / JSON.parse / Go / Rust). The
    failed frame's all-NaN QA stub is the regression trigger here."""
    base, result, bad_name = corrupt_night
    text = (base / "reports" / "qa_summary.json").read_text(encoding="utf-8")
    data = json.loads(
        text, parse_constant=lambda tok: pytest.fail(
            f"non-strict JSON constant {tok!r} in qa_summary.json"))
    frames = data["frames"]
    assert len(frames) == 4
    bad_rows = [f for f in frames if not f["ok"]]
    assert len(bad_rows) == 1
    assert bad_rows[0]["name"] == bad_name
    assert bad_rows[0]["error"]
