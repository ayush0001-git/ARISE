"""Stage 10 -- Discovery Dossiers: automated per-candidate science vetting.

For each top-ranked candidate this stage does, automatically, what a human
vetting team spends hours-to-days on per object:

* renders the **evidence** -- epoch-by-epoch cutout strip, light curve, and
  (for movers) the fitted motion track;
* runs an **identity check** -- is this a *known* object?
  - movers: live cone-search of the IMCCE **SkyBoT** service (the standard
    "known solar-system objects in this field at this epoch" service);
  - stationary sources: nearest-object lookup in **SIMBAD** (via astroquery);
  - plus the local/Gaia reference match ARISE already computed;
* fits the motion and **predicts where the object will be** in +1 h / +24 h,
  so it can be re-observed tomorrow night;
* drafts an **MPC 80-column astrometric report** for movers -- the exact
  format the Minor Planet Center accepts for new-object submissions;
* writes a plain-language **verdict + recommended action**.

All network checks are best-effort with short timeouts: offline the dossier
still builds and says which checks could not run. Outputs one self-contained
HTML per candidate plus a machine-readable JSON, and a night brief (md + html).
"""
from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..logs import get_logger

log = get_logger("dossier")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class IdentityCheck:
    service: str
    status: str          # match | no_match | unavailable
    detail: str = ""
    sep_arcsec: float = float("nan")


@dataclass
class Dossier:
    obj_id: int
    kind: str
    ra: float
    dec: float
    rank: float
    verdict: str = ""
    action: str = ""
    checks: list[IdentityCheck] = field(default_factory=list)
    motion: dict[str, Any] = field(default_factory=dict)
    html_name: str = ""


# --------------------------------------------------------------------------- #
# identity checks (best-effort, short-timeout)
# --------------------------------------------------------------------------- #
def check_skybot(ra: float, dec: float, epoch_iso: str, radius_arcsec: float = 600.0,
                 match_arcsec: float = 30.0, timeout: int = 15) -> IdentityCheck:
    """Cone-search IMCCE SkyBoT for known solar-system objects at this epoch."""
    if not epoch_iso:
        return IdentityCheck("SkyBoT (IMCCE)", "unavailable",
                             "no observation timestamp (DATE-OBS) -- cannot query at a known epoch")
    try:
        import requests
        from astropy.time import Time
        jd = Time(epoch_iso, format="isot", scale="utc").jd
        r = requests.get(
            "https://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php",
            params={"-ra": f"{ra:.6f}", "-dec": f"{dec:.6f}",
                    "-rd": f"{radius_arcsec / 3600.0:.4f}", "-ep": f"{jd:.6f}",
                    "-mime": "text", "-output": "object", "-loc": "500"},
            timeout=timeout, headers={"User-Agent": "ARISE-dossier/0.1"})
        r.raise_for_status()
        text = r.text.strip()
        if "No solar system object was found" in text or not text:
            return IdentityCheck("SkyBoT (IMCCE)", "no_match",
                                 f"no known solar-system object within {radius_arcsec/60:.0f} arcmin")
        # parse pipe-separated rows: Num | Name | RA(h) | DE(deg) | ...
        best_name, best_sep = None, np.inf
        for line in text.splitlines():
            if line.startswith(("#", "-")) or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            try:
                ra_h = _sexagesimal_to_deg(parts[2], hours=True)
                de_d = _sexagesimal_to_deg(parts[3], hours=False)
                sep = _sep_arcsec(ra, dec, ra_h, de_d)
                name = (parts[0] + " " + parts[1]).strip()
                if sep < best_sep:
                    best_name, best_sep = name, sep
            except (ValueError, IndexError):
                continue
        if best_name is not None and best_sep <= match_arcsec:
            return IdentityCheck("SkyBoT (IMCCE)", "match",
                                 f"KNOWN object {best_name} at {best_sep:.1f} arcsec",
                                 best_sep)
        detail = (f"nearest known object {best_name} is {best_sep:.0f} arcsec away"
                  if best_name else "no parseable known object nearby")
        return IdentityCheck("SkyBoT (IMCCE)", "no_match",
                             f"nothing within {match_arcsec:.0f} arcsec ({detail})")
    except Exception as exc:
        return IdentityCheck("SkyBoT (IMCCE)", "unavailable", f"{type(exc).__name__}: {exc}")


def check_jpl_sbident(ra: float, dec: float, epoch_iso: str,
                      hwidth_arcmin: float = 10.0, match_arcsec: float = 30.0,
                      timeout: int = 20) -> IdentityCheck:
    """Known solar-system objects via JPL's Small-Body Identification API.

    Independent backend to SkyBoT; parsed defensively (any schema surprise
    degrades to 'unavailable' rather than a wrong answer).
    """
    if not epoch_iso:
        return IdentityCheck("JPL SB-Ident", "unavailable",
                             "no observation timestamp (DATE-OBS) -- cannot query at a known epoch")
    try:
        import requests
        from astropy.time import Time
        jd = Time(epoch_iso, format="isot", scale="utc").jd

        def hms(deg, hours=False):
            v = deg / 15.0 if hours else abs(deg)
            sign = "" if hours else ("M" if deg < 0 else "")
            # integer decomposition at output precision so rounding can never
            # yield seconds == 60 (carry propagates into minutes/hours)
            tot = int(round(v * 360000)) % (24 * 360000 if hours else 360 * 360000)
            h, rem = divmod(tot, 360000)
            m, cs = divmod(rem, 6000)
            return f"{sign}{h:02d}-{m:02d}-{cs / 100.0:05.2f}"

        r = requests.get("https://ssd-api.jpl.nasa.gov/sb_ident.api", params={
            "mpc-code": "500", "obs-time": f"{jd:.5f}",
            "fov-ra-center": hms(ra, hours=True), "fov-dec-center": hms(dec),
            "fov-ra-hwidth": f"{hwidth_arcmin:.0f}", "fov-dec-hwidth": f"{hwidth_arcmin:.0f}",
            "two-pass": "true", "mag-required": "false", "req-elem": "false",
        }, timeout=timeout, headers={"User-Agent": "ARISE-dossier/0.1"})
        r.raise_for_status()
        data = r.json()
        rows = data.get("data_second_pass") or data.get("data_first_pass") or []
        fields = data.get("fields_second") or data.get("fields_first") or []
        if not rows:
            if "data_second_pass" not in data and "data_first_pass" not in data:
                return IdentityCheck("JPL SB-Ident", "unavailable",
                                     "unrecognized response schema (no data fields) -- "
                                     "positions could not be compared")
            return IdentityCheck("JPL SB-Ident", "no_match",
                                 f"no known small body within {hwidth_arcmin:.0f} arcmin")
        # locate RA/Dec columns; fall back to name-only reporting
        def col(*names):
            for n in names:
                for i, f in enumerate(fields):
                    if n.lower() in str(f).lower():
                        return i
            return None
        i_ra, i_dec = col("Astrometric RA", "RA"), col("Astrometric Dec", "Dec")
        best_name, best_sep = str(rows[0][0]), np.inf
        if i_ra is not None and i_dec is not None:
            for row in rows:
                try:
                    rra = _sexagesimal_to_deg(str(row[i_ra]).replace("'", " ").replace('"', " "),
                                              hours=True)
                    rde = _sexagesimal_to_deg(str(row[i_dec]).replace("'", " ").replace('"', " "),
                                              hours=False)
                    sep = _sep_arcsec(ra, dec, rra, rde)
                    if sep < best_sep:
                        best_name, best_sep = str(row[0]), sep
                except (ValueError, IndexError, TypeError):
                    continue
        if np.isfinite(best_sep) and best_sep <= match_arcsec:
            return IdentityCheck("JPL SB-Ident", "match",
                                 f"KNOWN object {best_name} at {best_sep:.1f} arcsec", best_sep)
        if not np.isfinite(best_sep):
            # RA/Dec columns not located (or no row parseable): no position was
            # ever compared, so degrade to unavailable rather than claim no_match
            return IdentityCheck("JPL SB-Ident", "unavailable",
                                 f"{len(rows)} known small bodies in the field, but the "
                                 f"response schema was unrecognized -- positions could "
                                 f"not be compared")
        return IdentityCheck("JPL SB-Ident", "no_match",
                             f"nearest known small body {best_name} is "
                             f"{best_sep:.0f} arcsec away")
    except Exception as exc:
        return IdentityCheck("JPL SB-Ident", "unavailable", f"{type(exc).__name__}: {exc}")


def check_neows_context(epoch_iso: str, timeout: int = 15) -> IdentityCheck:
    """NASA NeoWs close-approach context for the observation date.

    NeoWs lists NEOs by Earth close-approach date, not sky position, so this is
    *context* (how busy the near-Earth environment was that night), never an
    identification. Requires NASA_API_KEY.
    """
    import os
    key = os.environ.get("NASA_API_KEY", "")
    if not key:
        return IdentityCheck("NASA NeoWs (context)", "unavailable", "no NASA_API_KEY configured")
    date = (epoch_iso or "")[:10]
    if not date:
        return IdentityCheck("NASA NeoWs (context)", "unavailable",
                             "no observation timestamp (DATE-OBS)")
    try:
        import requests
        r = requests.get("https://api.nasa.gov/neo/rest/v1/feed",
                         params={"start_date": date, "end_date": date, "api_key": key},
                         timeout=timeout)
        r.raise_for_status()
        data = r.json()
        n = int(data.get("element_count", 0))
        neos = [o for objs in data.get("near_earth_objects", {}).values() for o in objs]
        n_haz = sum(bool(o.get("is_potentially_hazardous_asteroid")) for o in neos)
        return IdentityCheck("NASA NeoWs (context)", "no_match",
                             f"{n} known NEO close approaches on {date} "
                             f"({n_haz} flagged potentially hazardous) -- date context only, "
                             f"not a positional match")
    except Exception as exc:
        return IdentityCheck("NASA NeoWs (context)", "unavailable",
                             f"{type(exc).__name__}: {exc}")


def check_known_mover(ra: float, dec: float, epoch_iso: str,
                      online: bool = True) -> list[IdentityCheck]:
    """Known-asteroid vetting: SkyBoT first, JPL SB-Ident as independent
    fallback, plus NeoWs date context when a NASA key is configured."""
    if not online:
        return [IdentityCheck("SkyBoT (IMCCE)", "unavailable", "offline mode")]
    if not epoch_iso:
        return [IdentityCheck("SkyBoT (IMCCE)", "unavailable",
                              "no DATE-OBS in science headers -- known-object check "
                              "impossible at an unknown epoch")]
    checks = [check_skybot(ra, dec, epoch_iso)]
    if checks[0].status == "unavailable":
        checks.append(check_jpl_sbident(ra, dec, epoch_iso))
    checks.append(check_neows_context(epoch_iso))
    return checks


def check_simbad(ra: float, dec: float, match_arcsec: float = 5.0,
                 timeout: int = 12) -> IdentityCheck:
    """Nearest catalogued astronomical object via SIMBAD."""
    try:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        s = Simbad()
        s.TIMEOUT = timeout
        res = s.query_region(SkyCoord(ra, dec, unit="deg"), radius=match_arcsec * u.arcsec)
        if res is None or len(res) == 0:
            return IdentityCheck("SIMBAD", "no_match",
                                 f"no catalogued object within {match_arcsec:.0f} arcsec")
        name = str(res[0]["main_id" if "main_id" in res.colnames else "MAIN_ID"])
        return IdentityCheck("SIMBAD", "match", f"KNOWN object {name}")
    except Exception as exc:
        return IdentityCheck("SIMBAD", "unavailable", f"{type(exc).__name__}: {exc}")


def _sexagesimal_to_deg(s: str, hours: bool) -> float:
    parts = [float(p) for p in s.replace(":", " ").split()]
    if len(parts) == 1:
        deg = parts[0]
    else:
        sign = -1.0 if s.strip().startswith("-") else 1.0
        deg = sign * (abs(parts[0]) + parts[1] / 60.0 + (parts[2] if len(parts) > 2 else 0) / 3600.0)
    return deg * 15.0 if hours else deg


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    return float(SkyCoord(ra1 * u.deg, dec1 * u.deg)
                 .separation(SkyCoord(ra2 * u.deg, dec2 * u.deg)).arcsec)


# --------------------------------------------------------------------------- #
# motion fit + prediction
# --------------------------------------------------------------------------- #
def fit_motion(dets: list[dict[str, Any]]) -> dict[str, Any]:
    """Linear (constant-velocity) fit of RA/Dec vs time; predict +1h / +24h."""
    t = np.array([d["time_min"] for d in dets], float)
    ra = np.array([d["ra"] for d in dets], float)
    dec = np.array([d["dec"] for d in dets], float)
    if len(t) < 2 or np.ptp(t) <= 0:
        return {}
    order = np.argsort(t)
    t, ra, dec = t[order], ra[order], dec[order]
    # unwrap RA along the time-ordered track so tracklets crossing the RA=0/360
    # boundary fit a continuous line (predictions are re-wrapped into [0,360))
    ra = np.unwrap(ra, period=360.0)
    A = np.vstack([t, np.ones_like(t)]).T
    (v_ra, ra0), res_ra, *_ = np.linalg.lstsq(A, ra, rcond=None)
    (v_dec, dec0), res_dec, *_ = np.linalg.lstsq(A, dec, rcond=None)
    cosd = np.cos(np.deg2rad(np.mean(dec)))
    speed = float(np.hypot(v_ra * cosd, v_dec) * 3600.0)      # arcsec/min
    pa = float(np.degrees(np.arctan2(v_ra * cosd, v_dec)) % 360.0)
    # rms of the linear fit (arcsec, on-sky) -- how straight the track is;
    # RA-coordinate residuals are foreshortened by cos(dec) on the sky
    fit_rms = float(np.sqrt((np.sum(res_ra) * cosd ** 2 + np.sum(res_dec))
                            / max(len(t), 1)) * 3600.0) \
        if len(res_ra) and len(res_dec) else 0.0
    t_end = float(t.max())
    pred = {}
    for label, dt in (("+1h", 60.0), ("+24h", 1440.0)):
        pred[label] = {"ra": float((ra0 + v_ra * (t_end + dt)) % 360.0),
                       "dec": float(dec0 + v_dec * (t_end + dt))}
    return {"speed_arcsec_min": speed, "pa_deg": pa, "fit_rms_arcsec": fit_rms,
            "predictions": pred,
            "arc_minutes": float(np.ptp(t)), "n_epochs": int(len(t))}


# --------------------------------------------------------------------------- #
# MPC 80-column astrometric report (draft)
# --------------------------------------------------------------------------- #
def _frame_exptime(reduced_dir: Path, frame_name: str) -> float:
    """Exposure time (s) recovered from the reduced product's header (0 if unknown).

    FrameResult does not carry exptime, but the reduced FITS preserves the
    original science header, so the MPC mid-exposure correction reads it back.
    """
    from astropy.io import fits
    try:
        hdr = fits.getheader(reduced_dir / f"reduced_{frame_name}")
        for key in ("EXPTIME", "EXPOSURE", "ITIME", "TELAPSE"):
            if key in hdr:
                try:
                    return float(hdr[key])
                except (TypeError, ValueError):
                    continue
    except Exception:
        pass
    return 0.0


def mpc_80col(dets: list[dict[str, Any]], dateobs_of: dict[str, str],
              designation: str = "ARISE01", obs_code: str = "XXX",
              band: str = "V", exptime_of: dict[str, float] | None = None) -> str:
    """Draft an MPC 80-column astrometry report for a mover.

    Columns follow the MPC optical-observation format: packed designation,
    date 'YYYY MM DD.ddddd', RA 'HH MM SS.dd', Dec 'sDD MM SS.d', magnitude,
    band, and observatory code. Timestamps follow the MPC mid-exposure
    convention (start + exptime/2) for frames whose exposure time is given in
    ``exptime_of``; otherwise the exposure start time is kept and a COM header
    line says so. Marked DRAFT: replace the observatory code with your
    MPC-assigned code before submitting.
    """
    from astropy.time import Time, TimeDelta
    exptime_of = exptime_of or {}
    lines = [
        "COD XXX  (DRAFT - replace with your MPC observatory code)",
        "OBS ARISE pipeline (automated astrometry)",
        "NET Gaia DR3",
    ]
    has_exp = [float(exptime_of.get(d["frame"], 0.0) or 0.0) > 0.0 for d in dets]
    if has_exp and all(has_exp):
        lines.append("COM times are mid-exposure UTC")
    elif any(has_exp):
        lines.append("COM times are mid-exposure UTC where EXPTIME was known, "
                     "else exposure start")
    else:
        lines.append("COM times are exposure START (exposure time unavailable)")
    for d in sorted(dets, key=lambda x: x["time_min"]):
        iso = dateobs_of.get(d["frame"], "")
        try:
            tt = Time(iso, format="isot", scale="utc")
            exp = float(exptime_of.get(d["frame"], 0.0) or 0.0)
            if exp > 0.0:                 # MPC convention: mid-exposure epoch
                tt = tt + TimeDelta(exp / 2.0, format="sec")
        except Exception:
            continue
        # day fraction at exactly 5 decimals; a round-up to 1.0 carries into
        # the date instead of emitting a 6-digit fraction (81-char line)
        frac = int(round((tt.mjd % 1.0) * 100000))
        if frac >= 100000:
            tt = tt + TimeDelta(1.0, format="jd")
            frac = 0
        date_str = tt.strftime("%Y %m %d") + f".{frac:05d}"
        # sexagesimal via integer decomposition at output precision, so rounding
        # can never print seconds as '60.00'/'60.0' (carry into min/hour/deg)
        tot = int(round((d["ra"] / 15.0) * 360000)) % (24 * 360000)
        rh, rem = divmod(tot, 360000)
        rm, cs = divmod(rem, 6000)
        rs = cs / 100.0
        dd = d["dec"]; sign = "-" if dd < 0 else "+"
        tot = int(round(abs(dd) * 36000))
        dh, rem = divmod(tot, 36000)
        dm, ts = divmod(rem, 600)
        ds = ts / 10.0
        mag = d.get("mag_calib")
        mag_s = f"{mag:4.1f}" if mag is not None and np.isfinite(mag) else "    "
        line = (f"     {designation[:7]:<7}  C{date_str} "
                f"{rh:02d} {rm:02d} {rs:05.2f} "
                f"{sign}{dh:02d} {dm:02d} {ds:04.1f}          "
                f"{mag_s} {band}      {obs_code}")
        if len(line) != 80:               # never silently truncate the obs code
            line = line[:77].ljust(77) + obs_code[:3]
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def _load_reduced(reduced_dir: Path, frame_name: str):
    from astropy.io import fits
    p = reduced_dir / f"reduced_{frame_name}"
    if not p.exists():
        return None
    try:
        with fits.open(p, memmap=False) as hdul:
            return np.asarray(hdul[0].data, np.float32)
    except Exception:
        return None


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=95)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cutout_strip_b64(dets, reduced_dir: Path, size: int = 41) -> str | None:
    if not _HAVE_MPL:
        return None
    dets = sorted(dets, key=lambda d: d["frame_index"])[:8]
    fig, axes = plt.subplots(1, len(dets), figsize=(1.35 * len(dets), 1.6))
    if len(dets) == 1:
        axes = [axes]
    cache: dict[str, Any] = {}
    for ax, d in zip(axes, dets, strict=True):
        if d["frame"] not in cache:
            cache[d["frame"]] = _load_reduced(reduced_dir, d["frame"])
        img = cache[d["frame"]]
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"epoch {d['frame_index']}", fontsize=7)
        if img is None:
            continue
        ny, nx = img.shape
        xi, yi = int(round(d["x"])), int(round(d["y"]))
        h = size // 2
        x0, x1 = max(0, xi - h), min(nx, xi + h + 1)
        y0, y1 = max(0, yi - h), min(ny, yi + h + 1)
        stamp = img[y0:y1, x0:x1]
        try:
            norm = ImageNormalize(stamp, interval=ZScaleInterval(), stretch=AsinhStretch())
        except Exception:
            norm = None
        ax.imshow(stamp, origin="lower", cmap="gray", norm=norm)
        ax.plot(d["x"] - x0, d["y"] - y0, "o", mfc="none", mec="#2dd4bf", ms=13, mew=1.3)
    return _fig_b64(fig)


def _lightcurve_b64(dets) -> str | None:
    if not _HAVE_MPL or len(dets) < 2:
        return None
    dets = sorted(dets, key=lambda d: d["time_min"])
    t = [d["time_min"] for d in dets]
    # never mix photometric systems in one curve: use calibrated mags only when
    # EVERY epoch calibrated, else instrumental for all epochs (and say which)
    mc = np.array([d.get("mag_calib", np.nan) for d in dets], float)
    if np.isfinite(mc).all():
        m, ylabel = mc, "calibrated magnitude"
    else:
        m = np.array([d.get("mag_inst", np.nan) for d in dets], float)
        ylabel = "instrumental magnitude"
    e = [max(1.0857 / max(d["snr"], 1e-3), 0.005) for d in dets]
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    ax.errorbar(t, m, yerr=e, fmt="o-", color="#2563eb", ms=5, lw=1.2, capsize=2)
    ax.invert_yaxis()
    ax.set_xlabel("minutes since first frame", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    return _fig_b64(fig)


def _track_b64(dets, motion) -> str | None:
    if not _HAVE_MPL or len(dets) < 2:
        return None
    dets = sorted(dets, key=lambda d: d["time_min"])
    # unwrap RA so tracks crossing the RA=0/360 boundary plot contiguously
    ra = list(np.unwrap(np.array([d["ra"] for d in dets], float), period=360.0))
    dec = [d["dec"] for d in dets]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot(ra, dec, "o-", color="#b45309", ms=6, lw=1.3)
    for d, ra_i in zip(dets, ra, strict=True):
        ax.annotate(str(d["frame_index"]), (ra_i, d["dec"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color="#64748b")
    p1 = motion.get("predictions", {}).get("+1h")
    if p1:
        # place the (re-wrapped) prediction on the same unwrapped RA axis
        p1ra = ra[-1] + ((p1["ra"] - ra[-1] + 180.0) % 360.0) - 180.0
        ax.plot(p1ra, p1["dec"], "x", color="#b91c1c", ms=9, mew=2)
        ax.annotate("+1h", (p1ra, p1["dec"]), fontsize=7, color="#b91c1c",
                    xytext=(4, -10), textcoords="offset points")
    ax.invert_xaxis()
    ax.set_xlabel("RA (deg)", fontsize=8); ax.set_ylabel("Dec (deg)", fontsize=8)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_b64(fig)


# --------------------------------------------------------------------------- #
# verdict logic
# --------------------------------------------------------------------------- #
def _verdict(obj, checks: list[IdentityCheck], motion: dict) -> tuple[str, str]:
    known = [c for c in checks if c.status == "match"]
    unavailable = [c for c in checks if c.status == "unavailable"]
    caveat = (" (note: " + ", ".join(c.service for c in unavailable) +
              " could not be reached -- re-check online)") if unavailable else ""

    if obj.kind == "mover":
        if known:
            return (f"KNOWN solar-system object: {known[0].detail}.",
                    "No follow-up needed; use for astrometric residual QA.")
        p24 = motion.get("predictions", {}).get("+24h", {})
        where = (f"RA {p24.get('ra', float('nan')):.5f}, Dec {p24.get('dec', float('nan')):.5f}"
                 if p24 else "unavailable")
        return (f"CANDIDATE NEW moving object -- no known solar-system object at this "
                f"position{caveat}. Motion {motion.get('speed_arcsec_min', 0):.2f} arcsec/min "
                f"at PA {motion.get('pa_deg', 0):.0f} deg over {motion.get('n_epochs', 0)} epochs.",
                f"Re-observe tonight/tomorrow near {where} (+24h prediction). If recovered, "
                f"submit the attached MPC draft with your observatory code.")
    if obj.kind == "transient":
        if known:
            return (f"Matches catalogued object: {known[0].detail}.",
                    "Likely a known variable/star brightening; compare archival magnitudes.")
        return (f"CANDIDATE TRANSIENT -- stationary, repeats across {obj.n_det} epochs, "
                f"no catalog counterpart{caveat}.",
                "Obtain a confirmation image + spectrum if it persists; check TNS before "
                "announcing; monitor nightly for the rise/decline rate.")
    if obj.kind == "variable":
        name = f" ({known[0].detail})" if known else ""
        return (f"Variable star{name}: amplitude {obj.var_amplitude:.2f} mag over the run.",
                "Extend the light curve over more nights to measure the period and classify.")
    return ("Low-confidence single detection.", "Ignore unless it repeats.")


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def build_dossiers(discovery, frame_results, reduced_dir: str | Path,
                   reports_dir: str | Path, instrument_name: str = "",
                   online: bool = True, max_dossiers: int = 8) -> dict[int, str]:
    """Build per-candidate dossier HTMLs + night brief. Returns {obj_id: filename}."""
    reports_dir = Path(reports_dir)
    reduced_dir = Path(reduced_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    dateobs_of = {fr.name: fr.dateobs for fr in frame_results}

    interesting = [o for o in (discovery.objects if discovery else [])
                   if o.kind in ("mover", "transient", "variable")][:max_dossiers]
    # exposure times for MPC mid-exposure timestamps (only movers need them)
    exptime_of: dict[str, float] = {}
    if any(o.kind == "mover" for o in interesting):
        exptime_of = {fr.name: _frame_exptime(reduced_dir, fr.name)
                      for fr in frame_results}
    out: dict[int, str] = {}
    dossiers: list[Dossier] = []

    for obj in interesting:
        dets = sorted(obj.detections, key=lambda d: d["time_min"])
        mid = dets[len(dets) // 2]
        epoch_iso = dateobs_of.get(mid["frame"], "")
        if not epoch_iso:
            log.warning("Object %d (%s): no DATE-OBS in science headers -- known-object "
                        "checks and MPC timestamps unavailable", obj.obj_id, obj.kind)

        checks: list[IdentityCheck] = []
        motion: dict[str, Any] = {}
        if obj.kind == "mover":
            motion = fit_motion(dets)
            checks.extend(check_known_mover(obj.ra, obj.dec, epoch_iso, online=online))
        else:
            checks.append(check_simbad(obj.ra, obj.dec) if online else
                          IdentityCheck("SIMBAD", "unavailable", "offline mode"))
        checks.append(IdentityCheck(
            "Local reference catalog",
            "match" if obj.matched_reference else "no_match",
            "present in the run's reference catalog" if obj.matched_reference
            else "absent from the run's reference catalog"))

        verdict, action = _verdict(obj, checks, motion)
        d = Dossier(obj_id=obj.obj_id, kind=obj.kind, ra=obj.ra, dec=obj.dec,
                    rank=obj.rank_score, verdict=verdict, action=action,
                    checks=checks, motion=motion)

        # designation must fit the 7-character MPC field exactly
        mpc_text = mpc_80col(dets, dateobs_of, designation=f"ARI{obj.obj_id % 10000:04d}",
                             exptime_of=exptime_of) \
            if obj.kind == "mover" else ""
        html = _render_dossier_html(obj, d, dets, reduced_dir, mpc_text, instrument_name)
        d.html_name = f"dossier_{obj.obj_id:03d}_{obj.kind}.html"
        (reports_dir / d.html_name).write_text(html, encoding="utf-8")
        out[obj.obj_id] = d.html_name
        dossiers.append(d)
        log.info("Dossier %s: %s -> %s", d.html_name, obj.kind,
                 verdict.split("--")[0].split(".")[0])

    _write_night_brief(dossiers, discovery, reports_dir, instrument_name)
    return out


def _json_sanitize(o):
    """Recursively map NaN/inf floats to None so outputs are strict RFC-8259 JSON."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_sanitize(v) for v in o]
    return o


def _write_night_brief(dossiers: list[Dossier], discovery, reports_dir: Path,
                       instrument_name: str) -> None:
    """Plain-language morning-after brief (markdown; also indexed by Ask-ARISE)."""
    lines = [f"# ARISE night brief -- {instrument_name}", ""]
    if discovery:
        lines.append(f"Tracked {discovery.n_objects} objects: {discovery.n_movers} mover(s), "
                     f"{discovery.n_transients} transient(s), {discovery.n_variables} variable(s).")
        lines.append("")
    if not dossiers:
        lines.append("No candidates above the dossier threshold tonight.")
    for i, d in enumerate(dossiers, 1):
        lines += [f"## {i}. {d.kind.upper()} at RA {d.ra:.5f}, Dec {d.dec:.5f} "
                  f"(rank {d.rank:.3f})",
                  f"- Verdict: {d.verdict}",
                  f"- Action: {d.action}"]
        for c in d.checks:
            lines.append(f"- {c.service}: {c.status} -- {c.detail}")
        if d.motion:
            p = d.motion.get("predictions", {})
            if "+24h" in p:
                lines.append(f"- Predicted position +24h: RA {p['+24h']['ra']:.5f}, "
                             f"Dec {p['+24h']['dec']:.5f}")
        lines.append(f"- Full dossier: {d.html_name}")
        lines.append("")
    (reports_dir / "night_brief.md").write_text("\n".join(lines), encoding="utf-8")
    with open(reports_dir / "dossiers.json", "w", encoding="utf-8") as fh:
        json.dump(_json_sanitize([{**d.__dict__, "checks": [c.__dict__ for c in d.checks]}
                                  for d in dossiers]),
                  fh, indent=1, default=str, allow_nan=False)


# --------------------------------------------------------------------------- #
_KIND_TITLE = {"mover": "Moving object (asteroid / NEO candidate)",
               "transient": "Transient (new source)", "variable": "Variable star"}
_STATUS_COLOR = {"match": "#b45309", "no_match": "#059669", "unavailable": "#64748b"}
_STATUS_LABEL = {"match": "KNOWN", "no_match": "NOT KNOWN", "unavailable": "UNCHECKED"}


def _render_dossier_html(obj, d: Dossier, dets, reduced_dir: Path,
                         mpc_text: str, instrument_name: str) -> str:
    strip = _cutout_strip_b64(dets, reduced_dir)
    lc = _lightcurve_b64(dets)
    track = _track_b64(dets, d.motion) if obj.kind == "mover" else None

    imgs = ""
    if strip:
        imgs += f"<figure><figcaption>Epoch-by-epoch cutouts</figcaption><img class='wide' src='data:image/png;base64,{strip}'></figure>"
    row = ""
    if lc:
        row += f"<figure><figcaption>Light curve</figcaption><img src='data:image/png;base64,{lc}'></figure>"
    if track:
        row += f"<figure><figcaption>Sky track + 1h prediction</figcaption><img src='data:image/png;base64,{track}'></figure>"
    if row:
        imgs += f"<div class='imgrow'>{row}</div>"

    checks_rows = "".join(
        f"<tr><td>{html.escape(str(c.service))}</td>"
        f"<td><span class='tag' style='background:{_STATUS_COLOR[c.status]}'>"
        f"{_STATUS_LABEL[c.status]}</span></td><td>{html.escape(str(c.detail))}</td></tr>"
        for c in d.checks)

    motion_html = ""
    if d.motion:
        p = d.motion.get("predictions", {})
        rows = "".join(f"<tr><td>{k}</td><td>{v['ra']:.5f}</td><td>{v['dec']:.5f}</td></tr>"
                       for k, v in p.items())
        motion_html = f"""
  <h2>Motion solution</h2>
  <p>Speed <b>{d.motion['speed_arcsec_min']:.2f} arcsec/min</b> at position angle
  <b>{d.motion['pa_deg']:.0f} deg</b>; linear-fit RMS {d.motion['fit_rms_arcsec']:.2f} arcsec
  over a {d.motion['arc_minutes']:.0f}-minute arc ({d.motion['n_epochs']} epochs).</p>
  <table><thead><tr><th>Prediction</th><th>RA (deg)</th><th>Dec (deg)</th></tr></thead>
  <tbody>{rows}</tbody></table>"""

    mpc_html = ""
    if mpc_text:
        mpc_html = f"""
  <h2>MPC astrometric report (draft)</h2>
  <p class='muted'>80-column optical format. Replace <code>XXX</code> with your
  MPC observatory code and verify before submission to the Minor Planet Center.</p>
  <pre>{html.escape(mpc_text)}</pre>"""

    epoch_rows = "".join(
        f"<tr><td>{dd['frame_index']}</td><td>{html.escape(str(dd['frame']))}</td>"
        f"<td>{dd['ra']:.6f}</td><td>{dd['dec']:.6f}</td>"
        f"<td>{dd['snr']:.0f}</td>"
        f"<td>{dd.get('mag_calib', float('nan')):.2f}</td></tr>"
        for dd in dets)

    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>ARISE dossier -- {d.kind} {d.obj_id}</title>
<style>
body{{margin:0;font-family:-apple-system,'Segoe UI',Roboto,Arial,sans-serif;background:#f8fafc;color:#0f172a;line-height:1.55}}
header{{background:#0b1220;color:#fff;padding:20px 34px}}
header h1{{font-size:17px;margin:0;font-weight:600}}
header .sub{{color:#94a3b8;font-size:12px;margin-top:3px}}
main{{max-width:900px;margin:0 auto;padding:24px 34px 60px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#334155;
    border-bottom:2px solid #2563eb;display:inline-block;padding-bottom:3px;margin:26px 0 10px}}
.verdict{{background:#fff;border:1px solid #e2e8f0;border-left:4px solid #2563eb;
          border-radius:8px;padding:16px 18px;margin-top:18px}}
.verdict b.v{{display:block;font-size:15px;margin-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff;
       border:1px solid #e2e8f0;border-radius:8px}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid #e2e8f0}}
th{{font-size:11px;text-transform:uppercase;color:#475569;letter-spacing:.04em}}
.tag{{color:#fff;border-radius:999px;padding:2px 9px;font-size:11px;font-weight:600}}
figure{{margin:12px 0;background:#fff;border:1px solid #e2e8f0;border-radius:8px;
        padding:10px;display:inline-block}}
figcaption{{font-size:11px;color:#64748b;margin-bottom:6px}}
img{{max-width:100%;display:block}} img.wide{{width:100%}}
.imgrow{{display:flex;gap:14px;flex-wrap:wrap}}
pre{{background:#0b1220;color:#a5f3fc;font:12px/1.7 Consolas,monospace;border-radius:8px;
     padding:14px 16px;overflow-x:auto}}
.muted{{color:#64748b;font-size:12px}}
code{{background:#e2e8f0;border-radius:4px;padding:1px 5px}}
</style></head><body>
<header><h1>ARISE Discovery Dossier &mdash; {_KIND_TITLE.get(d.kind, d.kind)}</h1>
<div class='sub'>Object {d.obj_id} &middot; RA {d.ra:.6f}&deg; Dec {d.dec:.6f}&deg;
&middot; rank {d.rank:.3f} &middot; {html.escape(str(instrument_name))}</div></header>
<main>
  <div class='verdict'><b class='v'>{html.escape(d.verdict)}</b>
  <span><b>Recommended action:</b> {html.escape(d.action)}</span></div>

  <h2>Evidence</h2>
  {imgs or "<p class='muted'>Rendering unavailable.</p>"}

  <h2>Identity checks</h2>
  <table><thead><tr><th>Service</th><th>Result</th><th>Detail</th></tr></thead>
  <tbody>{checks_rows}</tbody></table>
  {motion_html}
  {mpc_html}

  <h2>Per-epoch measurements</h2>
  <table><thead><tr><th>Epoch</th><th>Frame</th><th>RA</th><th>Dec</th>
  <th>SNR</th><th>Mag</th></tr></thead><tbody>{epoch_rows}</tbody></table>
</main></body></html>"""
