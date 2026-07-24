"""Stage 9 -- QA metrics + HTML report.

Builds a single self-contained HTML report: a night summary, a per-frame
quality table (seeing, zero point, limiting magnitude, astrometric residual,
sky, cosmic-ray count), the ranked candidate list, and rendered cutout stamps
of the top discoveries (movers shown across epochs so their motion is visible).
All images are embedded as base64 so the report is one portable file.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..logs import get_logger

log = get_logger("report")


def _esc(value: Any) -> str:
    """HTML-escape an externally derived value (filenames, header/config strings)."""
    return html.escape(str(value), quote=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


# --------------------------------------------------------------------------- #
def _reduced_image(reduced_dir: Path, frame_name: str) -> np.ndarray | None:
    from astropy.io import fits
    p = reduced_dir / f"reduced_{frame_name}"
    if not p.exists():
        return None
    try:
        with fits.open(p, memmap=False) as hdul:
            return np.asarray(hdul[0].data, dtype=np.float32)
    except Exception:
        return None


def _stamp_b64(image: np.ndarray, x: float, y: float, size: int = 31,
               marker: bool = True) -> str | None:
    if not _HAVE_MPL or image is None:
        return None
    ny, nx = image.shape
    xi, yi = int(round(x)), int(round(y))
    half = size // 2
    x0, x1 = max(0, xi - half), min(nx, xi + half + 1)
    y0, y1 = max(0, yi - half), min(ny, yi + half + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    stamp = image[y0:y1, x0:x1]
    fig = plt.figure(figsize=(1.5, 1.5), dpi=80)
    ax = fig.add_axes([0, 0, 1, 1])
    try:
        norm = ImageNormalize(stamp, interval=ZScaleInterval(), stretch=AsinhStretch())
    except Exception:
        norm = None
    ax.imshow(stamp, origin="lower", cmap="gray", norm=norm)
    if marker:
        ax.plot(x - x0, y - y0, "o", mfc="none", mec="#2dd4bf", ms=16, mew=1.4)
    ax.set_xticks([]); ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _qa_trend_b64(qa_rows: list) -> str | None:
    if not _HAVE_MPL or not qa_rows:
        return None
    idx = list(range(len(qa_rows)))
    fwhm = [q.fwhm_arcsec for q in qa_rows]
    zp = [q.zeropoint for q in qa_rows]
    lim = [q.limiting_mag for q in qa_rows]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.4), dpi=90)
    for ax, y, ttl, col in zip(axes, [fwhm, zp, lim],
                               ["Seeing (arcsec)", "Zero point (mag)", "Limiting mag"],
                               ["#2563eb", "#059669", "#7c3aed"], strict=True):
        ax.plot(idx, y, "o-", color=col, lw=1.6, ms=5)
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("frame", fontsize=8)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
def build_report(result, frame_results, reference, reports_dir: Path) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir = Path(result.config.paths.reduced)

    disc = result.discovery
    cand = disc.candidates if disc else pd.DataFrame()
    top = cand.head(12) if len(cand) else cand

    # map obj_id -> DIAObject for cutouts
    obj_by_id = {o.obj_id: o for o in (disc.objects if disc else [])}

    # ---- cutouts for the top candidates -------------------------------- #
    # small LRU: never hold more than a few full frames in memory at once
    # (a 4kx4k float32 frame is 64 MiB; unbounded caching can pin gigabytes)
    max_cached = 4
    img_cache: dict[str, np.ndarray | None] = {}
    cand_cards = []
    for _, row in top.iterrows():
        obj = obj_by_id.get(int(row["obj_id"]))
        if obj is None:
            continue
        dets = sorted(obj.detections, key=lambda d: d["frame_index"])
        # for movers show first/mid/last; else the top-SNR detection
        if obj.kind == "mover" and len(dets) >= 2:
            picks = [dets[0], dets[len(dets) // 2], dets[-1]]
        else:
            picks = [max(dets, key=lambda d: d["snr"])]
        stamps = []
        for d in picks:
            name = d["frame"]
            if name in img_cache:
                img_cache[name] = img_cache.pop(name)  # refresh recency
            else:
                if len(img_cache) >= max_cached:
                    img_cache.pop(next(iter(img_cache)))  # evict least recent
                img_cache[name] = _reduced_image(reduced_dir, name)
            b64 = _stamp_b64(img_cache[name], d["x"], d["y"])
            if b64:
                stamps.append((d["frame_index"], b64))
        cand_cards.append((row, obj, stamps))
    img_cache.clear()  # release frame arrays before HTML assembly

    qa_trend = _qa_trend_b64(result.frame_qa)
    page = _render_html(result, top, cand_cards, qa_trend, reference)
    out = reports_dir / "arise_report.html"
    out.write_text(page, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
_KIND_LABEL = {"mover": "Moving object (asteroid/NEO)", "transient": "Transient (new)",
               "variable": "Variable star", "single": "Single unmatched",
               "known": "Known source"}
_KIND_COLOR = {"mover": "#b45309", "transient": "#b91c1c", "variable": "#6d28d9",
               "single": "#64748b", "known": "#475569"}


def _render_html(result, top: pd.DataFrame, cand_cards, qa_trend, reference) -> str:
    inst = result.instrument_name
    qa = result.frame_qa
    disc = result.discovery

    def med(attr):
        vals = [getattr(q, attr) for q in qa]
        vals = [v for v in vals if v == v]  # drop NaN
        return float(np.median(vals)) if vals else float("nan")

    # ---- QA table rows ---
    qa_rows = "".join(
        f"<tr><td>{_esc(q.name)}</td><td>{q.n_sources}</td><td>{q.n_cosmic_rays}</td>"
        f"<td>{q.fwhm_arcsec:.2f}</td><td>{q.sky_adu:.1f}</td>"
        f"<td>{q.astrom_rms_arcsec:.3f}</td><td>{q.zeropoint:.3f}</td>"
        f"<td>{q.zp_scatter:.3f}</td><td>{q.limiting_mag:.2f}</td></tr>"
        for q in qa
    )

    # ---- candidate table rows ---
    dossiers = getattr(result, "dossiers", {}) or {}

    def cand_row(r):
        col = _KIND_COLOR.get(r["kind"], "#334155")
        extra = ""
        if r["kind"] == "mover" and r.get("motion_arcsec_min") == r.get("motion_arcsec_min"):
            extra = f"{r['motion_arcsec_min']:.2f}&Prime;/min @ PA {r['motion_pa_deg']:.0f}&deg;"
        elif r["kind"] == "variable" and r.get("var_amp_mag") == r.get("var_amp_mag"):
            extra = f"&Delta;m={r['var_amp_mag']:.2f}, &chi;&sup2;={r['var_chi2']:.0f}"
        dossier = dossiers.get(int(r["obj_id"]))
        dlink = f"<a href='{_esc(dossier)}' target='_blank'>dossier</a>" if dossier else "&mdash;"
        return (f"<tr><td><span class='pill' style='background:{col}'>"
                f"{_esc(_KIND_LABEL.get(r['kind'], r['kind']))}</span></td>"
                f"<td><b>{r['rank_score']:.3f}</b></td>"
                f"<td>{r['ra']:.5f}</td><td>{r['dec']:.5f}</td>"
                f"<td>{r['n_det']}</td><td>{r['mean_snr']:.0f}</td>"
                f"<td>{r['rb_score']:.2f}</td><td>{extra}</td><td>{dlink}</td></tr>")

    cand_rows = "".join(cand_row(r) for _, r in top.iterrows()) if len(top) else \
        "<tr><td colspan='9'>No candidates above threshold.</td></tr>"

    # ---- candidate cutout cards ---
    cards = ""
    for _row, obj, stamps in cand_cards:
        imgs = "".join(
            f"<figure><img src='data:image/png;base64,{b64}'/>"
            f"<figcaption>frame {fi}</figcaption></figure>" for fi, b64 in stamps
        )
        col = _KIND_COLOR.get(obj.kind, "#334155")
        cards += (
            f"<div class='card'><div class='card-h' style='border-color:{col}'>"
            f"<span class='pill' style='background:{col}'>{_esc(_KIND_LABEL.get(obj.kind, obj.kind))}</span>"
            f"<span class='rank'>rank {obj.rank_score:.3f}</span></div>"
            f"<div class='stamps'>{imgs}</div>"
            f"<div class='meta'>RA {obj.ra:.5f}&deg;, Dec {obj.dec:.5f}&deg; &middot; "
            f"SNR {obj.mean_snr:.0f} &middot; {obj.n_det} epochs</div>"
            f"<div class='note'>{_esc(obj.notes)}</div></div>"
        )
    if not cards:
        cards = "<p class='muted'>No candidate cutouts (matplotlib unavailable or no candidates).</p>"

    trend = (f"<img class='trend' src='data:image/png;base64,{qa_trend}'/>"
             if qa_trend else "")

    n_obj = disc.n_objects if disc else 0
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>ARISE Reduction Report</title>
<style>
:root{{--bg:#0f172a;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#2563eb;}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:#f8fafc;line-height:1.5}}
header{{background:var(--bg);color:#fff;padding:28px 40px}}
header h1{{margin:0;font-size:22px;font-weight:600;letter-spacing:.02em}}
header .sub{{color:#94a3b8;font-size:13px;margin-top:4px}}
main{{max-width:1120px;margin:0 auto;padding:28px 40px 64px}}
section{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin:18px 0}}
h2{{font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:#334155;margin:0 0 14px;
border-bottom:2px solid var(--accent);display:inline-block;padding-bottom:4px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}}
.kpi{{border:1px solid var(--line);border-radius:8px;padding:14px 16px}}
.kpi .v{{font-size:24px;font-weight:600;color:var(--ink)}}
.kpi .l{{font-size:12px;color:var(--muted);margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
th{{color:#475569;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
tbody tr:hover{{background:#f1f5f9}}
.pill{{color:#fff;padding:3px 9px;border-radius:999px;font-size:11px;font-weight:600;white-space:nowrap}}
.rank{{float:right;color:var(--muted);font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}}
.card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff}}
.card-h{{padding:10px 12px;border-left:4px solid;display:flex;justify-content:space-between;align-items:center}}
.stamps{{display:flex;gap:6px;justify-content:center;padding:12px;background:#0b1220}}
.stamps figure{{margin:0;text-align:center}}
.stamps img{{width:96px;height:96px;image-rendering:pixelated;border-radius:4px;display:block}}
.stamps figcaption{{color:#94a3b8;font-size:10px;margin-top:3px}}
.meta{{padding:10px 12px;font-size:12px;color:#334155}}
.note{{padding:0 12px 12px;font-size:12px;color:var(--muted)}}
.trend{{width:100%;max-width:900px;display:block;margin:6px auto}}
.muted{{color:var(--muted)}}
footer{{text-align:center;color:var(--muted);font-size:12px;padding:24px}}
</style></head><body>
<header>
  <h1>ARISE &mdash; Automated Reduction &amp; Intelligent Source Extraction</h1>
  <div class='sub'>Instrument: {_esc(inst)} &nbsp;&middot;&nbsp; {result.n_science} science frames &nbsp;&middot;&nbsp; reference stars: {result.reference_size}</div>
</header>
<main>
  <section>
    <h2>Night summary</h2>
    <div class='kpis'>
      <div class='kpi'><div class='v'>{med('fwhm_arcsec'):.2f}&Prime;</div><div class='l'>median seeing (FWHM)</div></div>
      <div class='kpi'><div class='v'>{med('zeropoint'):.2f}</div><div class='l'>median zero point (mag)</div></div>
      <div class='kpi'><div class='v'>{med('limiting_mag'):.1f}</div><div class='l'>median 5&sigma; limiting mag</div></div>
      <div class='kpi'><div class='v'>{med('astrom_rms_arcsec'):.2f}&Prime;</div><div class='l'>astrometric residual RMS</div></div>
      <div class='kpi'><div class='v'>{n_obj}</div><div class='l'>tracked objects</div></div>
      <div class='kpi'><div class='v'>{(disc.n_movers if disc else 0)}/{(disc.n_transients if disc else 0)}/{(disc.n_variables if disc else 0)}</div><div class='l'>movers / transients / variables</div></div>
    </div>
    {trend}
  </section>

  <section>
    <h2>Ranked discovery candidates</h2>
    <table><thead><tr><th>Classification</th><th>Rank</th><th>RA</th><th>Dec</th>
    <th>Epochs</th><th>SNR</th><th>Real/bogus</th><th>Detail</th><th>Dossier</th></tr></thead>
    <tbody>{cand_rows}</tbody></table>
  </section>

  <section>
    <h2>Top candidate cutouts</h2>
    <div class='cards'>{cards}</div>
  </section>

  <section>
    <h2>Per-frame quality control</h2>
    <table><thead><tr><th>Frame</th><th>Sources</th><th>CR px</th><th>FWHM&Prime;</th>
    <th>Sky ADU</th><th>Astrom RMS&Prime;</th><th>ZP</th><th>ZP&sigma;</th><th>Lim mag</th></tr></thead>
    <tbody>{qa_rows}</tbody></table>
  </section>
</main>
<footer>Generated by ARISE &middot; astropy / photutils / ccdproc / astroscrappy pipeline</footer>
</body></html>"""
