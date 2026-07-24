"""ARISE web console.

A small Flask app that gives ARISE a face:

* **Drop FITS frames** (or a whole night as a ZIP, or even a plain PNG/JPG) on
  the page -> the full pipeline runs -> the HTML report + ranked candidates
  appear inline.
* **Run demo night** button -> synthesises the demo night (hidden asteroid,
  transient, variable) and reduces it, no data needed.
* **Ask ARISE** -- a retrieval assistant (see :mod:`arise.rag`) answering from
  the project docs, the research brief, and the latest run's own outputs.

Start with:  python -m arise.webapp   (default http://127.0.0.1:8770)
"""
from __future__ import annotations

import io
import ipaddress
import re
import shutil
import socket
import threading
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from .config import PipelineConfig, INSTRUMENTS
from .logs import setup_logging, get_logger
from .rag import build_default_kb, KnowledgeBase

log = get_logger("webapp")

ROOT = Path(__file__).resolve().parent.parent          # project root (D:/ARISE)
WEB_DATA = ROOT / "data_web"
ASSETS = Path(__file__).resolve().parent / "assets"

# Upload limits: request bodies beyond _MAX_UPLOAD_BYTES get HTTP 413, and ZIP
# extraction enforces a per-entry and total uncompressed budget (bomb guard).
_MAX_UPLOAD_BYTES = 2 * 1024 ** 3                      # 2 GB request cap
_MAX_MEMBER_BYTES = 2 * 1024 ** 3                      # per ZIP entry, uncompressed
_MAX_EXTRACT_BYTES = 8 * 1024 ** 3                     # whole ZIP, uncompressed
_MAX_REDIRECTS = 5                                     # /api/kb/add redirect cap
_RUN_KEEP = 20                                         # retention: newest N finished runs

app = Flask("arise")
app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_BYTES
_runs: dict[str, dict] = {}                            # run_id -> state
_kb_lock = threading.Lock()
_kb: KnowledgeBase | None = None
# One pipeline at a time: setup_logging() re-points the file handler of the
# single shared "arise" logger, so concurrent runs would write into each
# other's arise.log. Overlapping requests queue on this lock instead.
_run_lock = threading.Lock()

_FITS_EXT = (".fits", ".fit", ".fts", ".fz")
_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# upload hygiene + run retention
# --------------------------------------------------------------------------- #
def _safe_name(name: str, fallback: str = "upload") -> str:
    """Sanitize an externally supplied filename: basename only, characters
    restricted to [A-Za-z0-9._-], no leading dots (the name ends up on disk
    and, via FrameQA, in the HTML report)."""
    base = Path(str(name).replace("\\", "/")).name
    base = _SAFE_NAME_RE.sub("_", base).lstrip(".")
    return base or fallback


def _zip_member_name(member: str) -> str:
    """Flatten a ZIP member path to one safe filename, keeping the directory
    structure in the name (bias/frame001.fits -> bias__frame001.fits) so
    same-named frames in different sub-folders don't collide."""
    parts = [p for p in PurePosixPath(member.replace("\\", "/")).parts
             if p not in ("", ".", "..", "/")]
    return _safe_name("__".join(parts))


def _unique_path(raw: Path, name: str) -> Path:
    """Destination for *name* inside *raw*, uniquified (name_1.fits, ...) so
    colliding uploads never silently overwrite each other."""
    dest = raw / name
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 0
    while dest.exists():
        i += 1
        dest = raw / f"{stem}_{i}{suffix}"
    if i:
        log.warning("upload name collision: %s stored as %s", name, dest.name)
    return dest


def _prune_runs(keep: int = _RUN_KEEP) -> None:
    """Retention policy: keep the newest *keep* finished runs, drop older
    entries from ``_runs`` and delete their WEB_DATA/<run_id> directories.
    Runs still in state "running" are never touched."""
    finished = sorted((rid for rid, st in _runs.items()
                       if st.get("state") in ("done", "error")),
                      key=lambda rid: _runs[rid].get("t0", 0.0))
    for rid in finished[:max(0, len(finished) - keep)]:
        _runs.pop(rid, None)
        shutil.rmtree(WEB_DATA / rid, ignore_errors=True)


def _sweep_web_data(keep: int = _RUN_KEEP) -> None:
    """Startup sweep: run directories left behind by previous server processes
    are not in ``_runs``, so cap those by mtime as well (newest *keep* kept)."""
    if not WEB_DATA.exists():
        return
    try:
        dirs = sorted((p for p in WEB_DATA.iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    for p in dirs[keep:]:
        shutil.rmtree(p, ignore_errors=True)


# --------------------------------------------------------------------------- #
# knowledge base
# --------------------------------------------------------------------------- #
def _get_kb(rebuild: bool = False) -> KnowledgeBase:
    global _kb
    with _kb_lock:
        if _kb is None or rebuild:
            latest = sorted(WEB_DATA.glob("*"), key=lambda p: p.stat().st_mtime,
                            reverse=True) if WEB_DATA.exists() else []
            data_dirs = [latest[0]] if latest else [ROOT / "data"]
            _kb = build_default_kb(ROOT, data_dirs=data_dirs)
        return _kb


# --------------------------------------------------------------------------- #
# pipeline execution (background thread per run)
# --------------------------------------------------------------------------- #
def _image_to_fits(data: bytes, name: str, raw_dir: Path) -> Path:
    """Convert an ordinary image (PNG/JPG/...) to a light-frame FITS."""
    from astropy.io import fits
    import matplotlib.image as mpimg

    arr = mpimg.imread(io.BytesIO(data))
    if arr.ndim == 3:                       # RGB(A) -> luminance
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114])
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() <= 1.5:                    # 0-1 float images -> counts-like scale
        arr = arr * 60000.0
    arr = np.flipud(arr)                    # image row order -> FITS row order
    hdr = fits.Header()
    hdr["IMAGETYP"] = "light"
    hdr["EXPTIME"] = 1.0
    hdr["FILTER"] = "NONE"
    hdr["OBJECT"] = Path(name).stem
    hdr["COMMENT"] = "converted from a plain image by ARISE web console"
    out = raw_dir / (Path(name).stem + ".fits")
    fits.PrimaryHDU(arr, hdr).writeto(out, overwrite=True)
    return out


def _launch_run(run_id: str, instrument: str, demo: bool, size: int = 512) -> None:
    state = _runs[run_id]
    base = WEB_DATA / run_id
    raw = base / "raw"

    def work():
        try:
            # Serialize runs: setup_logging (here and inside run_pipeline)
            # reconfigures the shared "arise" logger, so overlapping runs
            # would clobber each other's per-run log files.
            state["message"] = "Waiting for a free run slot..."
            with _run_lock:
                setup_logging("INFO", log_file=base / "reports" / "arise.log")
                if demo:
                    state["message"] = "Generating synthetic night..."
                    from .synth import generate_night, SynthConfig
                    generate_night(raw, instrument,
                                   SynthConfig(nx=size, ny=size, n_science=6))
                state["message"] = "Reducing frames..."
                cfg = PipelineConfig(instrument=instrument)
                cfg.log_level = "INFO"
                cfg.paths.raw = str(raw)
                cfg.paths.master = str(base / "master")
                cfg.paths.reduced = str(base / "reduced")
                cfg.paths.catalogs = str(base / "catalogs")
                cfg.paths.reports = str(base / "reports")
                from .pipeline import run_pipeline
                result = run_pipeline(cfg)

                d = result.discovery
                state.update({
                    "state": "done",
                    "message": "Complete",
                    "report": f"/runs/{run_id}/reports/arise_report.html",
                    "summary": {
                        "frames": result.n_science,
                        "movers": d.n_movers if d else 0,
                        "transients": d.n_transients if d else 0,
                        "variables": d.n_variables if d else 0,
                        "objects": d.n_objects if d else 0,
                    },
                })
                _get_kb(rebuild=True)      # so Ask-ARISE knows this run
        except Exception as exc:
            log.error("Run %s failed: %s", run_id, exc, exc_info=True)
            state.update({"state": "error", "message": f"{exc}"})

    threading.Thread(target=work, daemon=True).start()


def _tail_log(run_id: str, lines: int = 12) -> list[str]:
    p = WEB_DATA / run_id / "reports" / "arise.log"
    if not p.exists():
        return []
    try:
        content = p.read_text(encoding="utf-8", errors="replace").splitlines()
        out = []
        for line in content[-lines:]:
            # strip "date level logger" prefix for compact display
            out.append(re.sub(r"^[\d\-]+ [\d:]+\s+\w+\s+\S+\s+", "", line))
        return out
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return (ASSETS / "webapp.html").read_text(encoding="utf-8")


@app.get("/api/instruments")
def instruments():
    return jsonify([{"id": k, "label": v.telescope or k} for k, v in INSTRUMENTS.items()])


@app.post("/api/demo")
def demo():
    run_id = uuid.uuid4().hex[:10]
    # get_json(silent=True) tolerates absent/null/malformed JSON bodies
    instrument = (request.get_json(silent=True) or {}).get("instrument", "dfot_2kx2k")
    _prune_runs()
    (WEB_DATA / run_id / "raw").mkdir(parents=True, exist_ok=True)
    _runs[run_id] = {"state": "running", "message": "Starting...", "t0": time.time()}
    _launch_run(run_id, instrument, demo=True)
    return jsonify({"run_id": run_id})


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files received"}), 400
    instrument = request.form.get("instrument", "generic")
    _prune_runs()
    run_id = uuid.uuid4().hex[:10]
    base = WEB_DATA / run_id
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    def _reject(msg: str, code: int = 400):
        shutil.rmtree(base, ignore_errors=True)     # no run: drop the orphan dir
        return jsonify({"error": msg}), code

    n_fits = n_img = 0
    for f in files:
        name = _safe_name(f.filename or "upload")
        low = name.lower()
        if low.endswith(".zip"):
            tmp = base / f"_upload_{uuid.uuid4().hex[:8]}.zip"
            f.save(tmp)                             # stream to disk, not RAM
            extracted = 0
            try:
                with zipfile.ZipFile(tmp) as z:
                    for info in z.infolist():
                        if info.is_dir() or not info.filename.lower().endswith(_FITS_EXT):
                            continue
                        # ZIP-bomb guard: per-entry and total uncompressed budget
                        if (info.file_size > _MAX_MEMBER_BYTES
                                or extracted + info.file_size > _MAX_EXTRACT_BYTES):
                            return _reject(
                                "ZIP decompresses beyond the server limits "
                                f"({_MAX_MEMBER_BYTES // 2 ** 30} GB per file, "
                                f"{_MAX_EXTRACT_BYTES // 2 ** 30} GB total)", 413)
                        dest = _unique_path(raw, _zip_member_name(info.filename))
                        if not dest.resolve().is_relative_to(raw.resolve()):
                            continue                # zip-slip guard (belt and braces)
                        with z.open(info) as src, open(dest, "wb") as out:
                            shutil.copyfileobj(src, out, length=1024 * 1024)
                        extracted += info.file_size
                        n_fits += 1
            except zipfile.BadZipFile:
                return _reject(f"{name} is not a valid ZIP file")
            finally:
                tmp.unlink(missing_ok=True)
        elif low.endswith(_FITS_EXT) or low.endswith(".fits.gz"):
            f.save(_unique_path(raw, name))         # stream to disk, not RAM
            n_fits += 1
        elif low.endswith(_IMG_EXT):
            _image_to_fits(f.read(), name, raw)
            n_img += 1

    if n_fits + n_img == 0:
        return _reject("no FITS / image files found in the drop")

    _runs[run_id] = {"state": "running", "message": "Files received...",
                     "t0": time.time(),
                     "note": (f"{n_img} plain image(s) converted to FITS -- "
                              "extraction only, no sky coordinates" if n_img else "")}
    _launch_run(run_id, instrument, demo=False)
    return jsonify({"run_id": run_id, "n_fits": n_fits, "n_images": n_img})


@app.get("/api/status/<run_id>")
def status(run_id: str):
    state = _runs.get(run_id)
    if state is None:
        return jsonify({"error": "unknown run"}), 404
    out = dict(state)
    out["elapsed"] = round(time.time() - state.get("t0", time.time()), 1)
    out["log"] = _tail_log(run_id)
    return jsonify(out)


@app.get("/runs/<run_id>/<path:subpath>")
def run_files(run_id: str, subpath: str):
    return send_from_directory(WEB_DATA / run_id, subpath)


@app.post("/api/ask")
def ask():
    q = (request.json or {}).get("question", "").strip()
    if not q:
        return jsonify({"error": "empty question"}), 400
    return jsonify(_get_kb().answer(q))


@app.errorhandler(413)
def _too_large(_exc):
    return jsonify({"error": "upload larger than the "
                             f"{_MAX_UPLOAD_BYTES // 2 ** 20} MB request limit"}), 413


def _check_public_http_url(url: str, max_redirects: int = _MAX_REDIRECTS) -> str:
    """SSRF guard for user-supplied KB URLs.

    Rejects non-http(s) schemes and hostnames that resolve to loopback /
    private / link-local / reserved / otherwise non-public addresses (cloud
    metadata endpoints included), follows redirects manually so every hop is
    re-validated, and caps the redirect chain. Returns the final URL to
    fetch; raises ValueError with a user-facing message on rejection.
    """
    import requests

    current = url
    for _ in range(max_redirects + 1):
        parts = urlparse(current)
        if parts.scheme not in ("http", "https"):
            raise ValueError("provide an http(s) URL")
        host = parts.hostname
        if not host:
            raise ValueError("URL has no hostname")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except (OSError, ValueError):
            raise ValueError(f"cannot resolve host {host!r}") from None
        for info in infos:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                    or not ip.is_global):
                raise ValueError("URL resolves to a non-public address")
        try:
            r = requests.get(current, timeout=10, allow_redirects=False,
                             stream=True, headers={"User-Agent": "ARISE-KB/0.1"})
        except requests.RequestException as exc:
            raise ValueError(f"could not reach URL ({exc.__class__.__name__})") from None
        try:
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("location")
                if not loc:
                    raise ValueError("redirect without a Location header")
                current = urljoin(current, loc)
                continue
            return current
        finally:
            r.close()
    raise ValueError(f"too many redirects (> {max_redirects})")


@app.post("/api/kb/add")
def kb_add():
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    try:
        fetch_url = _check_public_http_url(url)    # validates every redirect hop
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    n = _get_kb().add_url(fetch_url)
    # persist so it survives restarts
    kb_dir = ROOT / "kb"
    kb_dir.mkdir(exist_ok=True)
    urls_file = kb_dir / "urls.txt"
    existing = urls_file.read_text(encoding="utf-8") if urls_file.exists() else ""
    if url not in existing:
        with open(urls_file, "a", encoding="utf-8") as fh:
            fh.write(url + "\n")
    return jsonify({"ok": n > 0, "chunks": n})


def main(host: str = "127.0.0.1", port: int = 8770) -> None:
    setup_logging("INFO")
    from .keys import load_keys
    load_keys(ROOT)
    WEB_DATA.mkdir(exist_ok=True)
    _sweep_web_data()                      # prune run dirs from earlier processes
    _get_kb()                              # warm the knowledge base
    log.info("ARISE web console on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
