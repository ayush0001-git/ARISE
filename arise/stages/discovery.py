"""Stage 8 -- discovery: novelty detection & candidate ranking.

Aggregates per-frame catalogs into per-sky-position objects (a DIAObject
analog) and surfaces *new* sources through several converging signals:

* **catalog novelty** -- detections with no match in the reference catalog;
* **moving objects** -- orphan detections linked into a straight-line,
  constant-velocity tracklet across epochs (asteroid / NEO candidates);
* **variables** -- catalogued stars whose calibrated light curve varies beyond
  its errors (robust chi^2, IQR, von-Neumann 1/eta);
* **transients** -- unmatched sources that are stationary and repeat.

A real-vs-bogus quality score demotes CR residuals / edge junk, and a composite
rank combines novelty, significance, persistence, and motion/variability so the
most interesting candidates surface first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..catalogs import crossmatch
from ..config import DiscoveryConfig
from ..logs import get_logger

log = get_logger("discovery")


# --------------------------------------------------------------------------- #
# inputs / outputs
# --------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    """One reduced+measured science frame handed to the discovery stage."""
    name: str
    time_min: float                  # minutes from the first frame
    catalog: pd.DataFrame            # ra,dec,x,y,snr,flux,fwhm,ellipticity,flags,mag_*
    median_fwhm: float
    frame_index: int = 0
    dateobs: str = ""                # ISO UTC start of exposure (for reports/MPC)


@dataclass
class DIAObject:
    obj_id: int
    ra: float
    dec: float
    detections: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "known"             # known | transient | variable | mover | single
    matched_reference: bool = False
    n_det: int = 0
    mean_snr: float = 0.0
    rb_score: float = 0.0
    rank_score: float = 0.0
    motion_arcsec_per_min: float = 0.0
    motion_pa_deg: float = float("nan")
    var_chi2: float = float("nan")
    var_amplitude: float = float("nan")
    mean_mag: float = float("nan")
    mag_rms: float = float("nan")
    var_significance: float = float("nan")
    notes: str = ""


@dataclass
class DiscoveryResult:
    candidates: pd.DataFrame
    objects: list[DIAObject]
    n_objects: int
    n_movers: int
    n_transients: int
    n_variables: int


# --------------------------------------------------------------------------- #
# wrap-safe RA arithmetic
# --------------------------------------------------------------------------- #
def _wrap_dra(dra):
    """Wrap an RA difference (deg) into [-180, 180); works on scalars/arrays."""
    return (dra + 180.0) % 360.0 - 180.0


def _median_ra(ras) -> float:
    """Median RA (deg) that is safe across the 0/360 wrap: unwrap about the
    first value, take the median, then normalize back into [0, 360)."""
    ras = np.asarray(ras, dtype=float)
    ra0 = float(ras[0])
    return float((ra0 + np.median(_wrap_dra(ras - ra0))) % 360.0)


# --------------------------------------------------------------------------- #
# real-vs-bogus quality score
# --------------------------------------------------------------------------- #
def _rb_score(dets: list[dict[str, Any]], median_fwhm: float,
              is_mover: bool = False) -> float:
    """Heuristic real/bogus score in [0,1] from detection shape & persistence."""
    snr = np.mean([d["snr"] for d in dets])
    fwhm = np.median([d["fwhm"] for d in dets])
    ellip = np.median([d["ellipticity"] for d in dets])
    edge = any(d.get("flag_edge", False) for d in dets)

    score = 1.0
    # cosmic-ray residuals & hot pixels are far too sharp (fwhm << stellar)
    if median_fwhm and np.isfinite(fwhm):
        ratio = fwhm / median_fwhm
        if ratio < 0.55:
            score *= 0.15            # much sharper than the PSF -> likely CR
        elif ratio < 0.75:
            score *= 0.6
    if ellip > 0.6 and not is_mover:
        score *= 0.5                 # very elongated -> streak/artifact (movers exempt: trailing is expected)
    if snr < 5:
        score *= 0.5
    if edge:
        score *= 0.7
    # persistence across frames is strong evidence of reality
    score *= min(1.0, 0.5 + 0.25 * len(dets))
    return float(np.clip(score, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# clustering detections into sky-position objects (union-find)
# --------------------------------------------------------------------------- #
def _cluster_by_position(dets: list[dict[str, Any]], radius_arcsec: float) -> list[list[int]]:
    n = len(dets)
    if n == 0:
        return []
    from astropy.coordinates import SkyCoord, search_around_sky
    import astropy.units as u

    coords = SkyCoord([d["ra"] for d in dets] * u.deg, [d["dec"] for d in dets] * u.deg)
    i1, i2, sep, _ = search_around_sky(coords, coords, radius_arcsec * u.arcsec)

    parent = list(range(n))
    # frame indices present in each component, tracked at the root: never merge
    # two detections from the same frame into one object -- not even
    # transitively through a bridging detection in another frame
    frames_of: list[set[int]] = [{d["frame_index"]} for d in dets]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra_, rb_ = find(a), find(b)
        if ra_ == rb_ or frames_of[ra_] & frames_of[rb_]:
            return               # same component, or components share an epoch
        parent[rb_] = ra_
        frames_of[ra_] |= frames_of[rb_]

    # merge nearest pairs first, so a same-frame conflict blocks the more
    # distant of two competing merges rather than an arbitrary one
    for k in np.argsort(sep.arcsec):
        a, b = int(i1[k]), int(i2[k])
        if a != b:
            union(a, b)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


# --------------------------------------------------------------------------- #
# moving-object linking (findTracklets / linkTracklets, MOPS-style)
# --------------------------------------------------------------------------- #
def _angular_speed(ra0, dec0, ra1, dec1, dt_min):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if dt_min <= 0:
        return np.inf, float("nan")
    c0 = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg)
    sep = c0.separation(c1).arcsec
    pa = c0.position_angle(c1).deg
    return sep / dt_min, pa


def _link_movers(orphans: list[dict[str, Any]], dcfg: DiscoveryConfig) -> list[list[dict]]:
    """Link orphan detections into constant-velocity tracklets across frames."""
    by_frame: dict[int, list[dict]] = {}
    for d in orphans:
        by_frame.setdefault(d["frame_index"], []).append(d)
    frames = sorted(by_frame)
    if len(frames) < 2:
        return []

    tol_deg = dcfg.catalog_match_radius_arcsec * 2.5 / 3600.0  # prediction tolerance
    used: set[int] = set()
    tracklets: list[list[dict]] = []

    for fi_a in frames[:-1]:
        for da in by_frame[fi_a]:
            if id(da) in used:
                continue
            for fi_b in frames:
                if fi_b <= fi_a:
                    continue
                for db in by_frame[fi_b]:
                    if id(db) in used:
                        continue
                    dt = db["time_min"] - da["time_min"]
                    speed, pa = _angular_speed(da["ra"], da["dec"], db["ra"], db["dec"], dt)
                    if not np.isfinite(speed) or speed > dcfg.max_motion_arcsec_per_min or speed <= 0:
                        continue
                    # constant-velocity model from a->b; gather matches in all frames
                    v_ra = _wrap_dra(db["ra"] - da["ra"]) / dt   # wrap-safe across RA=0/360
                    v_dec = (db["dec"] - da["dec"]) / dt
                    track = [da, db]
                    for fi_c in frames:
                        if fi_c in (fi_a, fi_b):
                            continue
                        for dc in by_frame[fi_c]:
                            if id(dc) in used:
                                continue
                            pred_ra = da["ra"] + v_ra * (dc["time_min"] - da["time_min"])
                            pred_dec = da["dec"] + v_dec * (dc["time_min"] - da["time_min"])
                            # wrap the RA residual and scale by cos(dec) so the
                            # tolerance is the same on-sky size at any declination
                            dra_sky = abs(_wrap_dra(dc["ra"] - pred_ra)) * np.cos(np.radians(dc["dec"]))
                            if (dra_sky < tol_deg
                                    and abs(dc["dec"] - pred_dec) < tol_deg):
                                track.append(dc)
                                break
                    if len(track) >= dcfg.min_tracklet_points:
                        track = sorted(track, key=lambda d: d["frame_index"])
                        for d in track:
                            used.add(id(d))
                        tracklets.append(track)
                        break
                if da is not None and id(da) in used:
                    break
    return tracklets


# --------------------------------------------------------------------------- #
# variability
# --------------------------------------------------------------------------- #
def _select_lightcurve(dets: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, str]:
    """Extract an object's light curve in ONE magnitude system.

    Calibrated and instrumental magnitudes differ by the zero point, so mixing
    them within a light curve (e.g. when only some frames calibrate) fabricates
    huge spurious variability. Use mag_calib only when >= 3 epochs carry a
    finite calibrated mag (dropping the uncalibrated epochs); otherwise fall
    back to mag_inst for ALL epochs. Returns (mags, errs, system) where system
    is "calibrated" or "instrumental".
    """
    def _mag(d, key):
        m = d.get(key)
        return float(m) if m is not None and np.isfinite(m) else float("nan")

    n_calib = sum(np.isfinite(_mag(d, "mag_calib")) for d in dets)
    use_calib = n_calib >= 3
    mags, errs = [], []
    for d in dets:
        m = _mag(d, "mag_calib") if use_calib else _mag(d, "mag_inst")
        if not np.isfinite(m):
            continue
        e = d.get("mag_err")
        if e is None or not np.isfinite(e) or e <= 0:
            e = max(1.0857 / max(d["snr"], 1e-3), 0.005)
        mags.append(m)
        errs.append(float(e))
    return np.array(mags), np.array(errs), ("calibrated" if use_calib else "instrumental")


def _lightcurve_stats(dets: list[dict[str, Any]]) -> tuple[float, float, float, float, str]:
    """Return (reduced chi^2 vs constant, peak-to-peak amp, mean mag, robust RMS, mag system)."""
    mags, errs, system = _select_lightcurve(dets)
    if len(mags) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), system
    w = 1.0 / errs ** 2
    wmean = np.sum(mags * w) / np.sum(w)
    chi2 = np.sum(((mags - wmean) / errs) ** 2) / (len(mags) - 1)
    amp = float(np.max(mags) - np.min(mags))
    mean_mag = float(np.mean(mags))
    rms = float(np.std(mags, ddof=1))
    return float(chi2), amp, mean_mag, rms, system


def _object_lightcurve(o: DIAObject) -> tuple[np.ndarray, np.ndarray]:
    mags, errs, _system = _select_lightcurve(o.detections)
    return mags, errs


def _flag_variables(objects: list[DIAObject], dcfg: DiscoveryConfig) -> None:
    """Flag genuine variables as outliers on the ensemble RMS-vs-magnitude trend.

    Formal photometric errors underestimate real frame-to-frame scatter, so a
    naive chi^2 test flags almost every star. We instead build the empirical
    noise floor RMS(mag) from the whole star ensemble and require, on top of an
    RMS excess, that the variation appear across **multiple epochs** (not a
    single cosmic-ray/blend glitch) -- the robust variability-survey approach.
    At least 2 deviant epochs are required (deviations are measured from the
    object's own median, which pins one point to zero deviation, so demanding
    3 of 3 would be unsatisfiable); sequences of 6+ epochs need n_det // 2.
    """
    if not dcfg.variability:
        return
    cands = [o for o in objects if o.matched_reference and o.n_det >= 3
             and np.isfinite(o.mean_mag) and np.isfinite(o.mag_rms)]
    if not cands:
        return

    # Global systematic noise floor: the residual frame-to-frame scatter that
    # even bright, non-variable stars show (flat/background/CR-cleaning). Robust
    # median of the ensemble RMS, clipped to a sane range. This -- combined with
    # each point's own formal error -- is the per-epoch noise, so a sparse
    # faint-magnitude bin can't hide a real variable behind its own scatter.
    all_rms = np.array([o.mag_rms for o in cands])
    sys_floor = float(np.clip(np.median(all_rms), 0.005, 0.03))

    for o in cands:
        m_i, e_i, mag_system = _select_lightcurve(o.detections)
        if len(m_i) < 3:
            o.kind = "known"
            continue
        med = float(np.median(m_i))
        noise = np.sqrt(e_i ** 2 + sys_floor ** 2)
        dev = np.abs(m_i - med) / noise
        n_sig = int(np.sum(dev > 4.0))                 # epochs deviating > 4 sigma
        red_chi2 = float(np.mean(dev ** 2))
        o.var_chi2 = red_chi2
        o.var_significance = float(np.sqrt(max(red_chi2, 0.0)))
        need = max(2, o.n_det // 2)
        if (n_sig >= need and red_chi2 > 10.0 and o.var_amplitude > 0.1
                and o.mag_rms > 2.5 * sys_floor):
            o.kind = "variable"
            o.notes = (f"variable: rms={o.mag_rms:.3f} mag, {n_sig}/{o.n_det} epochs "
                       f">4 sigma, reduced chi2={red_chi2:.0f}, amp={o.var_amplitude:.2f} mag"
                       f" ({mag_system} mags)")
        else:
            o.kind = "known"


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def run_discovery(frames: list[FrameResult], reference: pd.DataFrame | None,
                  dcfg: DiscoveryConfig) -> DiscoveryResult:
    """Find and rank candidate new sources across a reduced science sequence."""
    if not dcfg.enabled:
        return DiscoveryResult(pd.DataFrame(), [], 0, 0, 0, 0)

    median_fwhm = float(np.nanmedian([f.median_fwhm for f in frames if np.isfinite(f.median_fwhm)]))

    # flatten all significant detections, tagging reference-match per frame
    all_dets: list[dict[str, Any]] = []
    have_ref = reference is not None and len(reference) > 0
    ref_ra = reference["ra"].to_numpy() if have_ref else np.array([])
    ref_dec = reference["dec"].to_numpy() if have_ref else np.array([])

    for fr in frames:
        cat = fr.catalog
        keep = cat[(cat["snr"] >= dcfg.min_snr) & (~cat["flag_saturated"])]
        # discovery works in sky coordinates; frames with no WCS (e.g. plain
        # image uploads) contribute nothing rather than crashing SkyCoord
        keep = keep[np.isfinite(keep["ra"]) & np.isfinite(keep["dec"])]
        if have_ref and len(keep):
            _, _, matched = crossmatch(keep["ra"].to_numpy(), keep["dec"].to_numpy(),
                                       ref_ra, ref_dec, dcfg.catalog_match_radius_arcsec)
        else:
            matched = np.zeros(len(keep), bool)
        for row, m in zip(keep.itertuples(index=False), matched, strict=True):
            r = row._asdict()
            all_dets.append({
                "frame": fr.name, "frame_index": fr.frame_index, "time_min": fr.time_min,
                "ra": float(r["ra"]), "dec": float(r["dec"]),
                "x": float(r["x"]), "y": float(r["y"]),
                "snr": float(r["snr"]), "flux": float(r.get("flux", np.nan)),
                "fwhm": float(r["fwhm"]), "ellipticity": float(r["ellipticity"]),
                "flag_edge": bool(r.get("flag_edge", False)),
                "mag_calib": float(r.get("mag_calib", np.nan)),
                "mag_inst": float(r.get("mag_inst", np.nan)),
                # per-STAR measurement error (NOT mag_calib_err: the zero-point
                # uncertainty is common to all stars in a frame, so it cancels
                # in differential light curves and must not inflate per-point noise)
                "mag_err": float(r.get("mag_inst_err", np.nan)),
                "matched_ref": bool(m),
            })

    log.info("Discovery: %d significant detections across %d frames (median FWHM %.2f px)",
             len(all_dets), len(frames), median_fwhm)

    # ---- cluster stationary detections by sky position ------------------ #
    clusters = _cluster_by_position(all_dets, dcfg.catalog_match_radius_arcsec)
    objects: list[DIAObject] = []
    orphan_unmatched: list[dict[str, Any]] = []
    oid = 0

    for members in clusters:
        dets = [all_dets[i] for i in members]
        n = len(dets)
        matched_ref = any(d["matched_ref"] for d in dets)
        ra = _median_ra([d["ra"] for d in dets])
        dec = float(np.median([d["dec"] for d in dets]))

        # a single unmatched detection at a unique spot -> candidate mover point
        if n == 1 and not matched_ref and dcfg.moving_object_link:
            orphan_unmatched.append(dets[0])
            continue

        obj = DIAObject(obj_id=oid, ra=ra, dec=dec, detections=dets,
                        matched_reference=matched_ref, n_det=n,
                        mean_snr=float(np.mean([d["snr"] for d in dets])))
        obj.rb_score = _rb_score(dets, median_fwhm)

        if matched_ref:
            chi2, amp, mean_mag, rms, mag_system = _lightcurve_stats(dets)
            obj.var_chi2, obj.var_amplitude = chi2, amp
            obj.mean_mag, obj.mag_rms = mean_mag, rms
            obj.kind = "known"   # variability decided by the ensemble pass below
            if mag_system == "instrumental" and np.isfinite(mean_mag):
                obj.notes = "light curve in instrumental mags (calibration unavailable)"
        elif not dcfg.flag_unmatched:
            obj.kind = "known"
            obj.notes = "unmatched source; novelty flagging disabled"
        else:
            if n >= 2:
                obj.kind = "transient"
                obj.notes = f"stationary source with no catalog match, {n} detections"
            else:
                obj.kind = "single"
                obj.notes = "single unmatched detection (low confidence)"
        objects.append(obj)
        oid += 1

    # ---- ensemble variability flagging (RMS-vs-magnitude outliers) ------- #
    _flag_variables(objects, dcfg)

    # ---- link orphan detections into moving-object tracklets ------------ #
    n_movers = 0
    if dcfg.moving_object_link:
        tracklets = _link_movers(orphan_unmatched, dcfg)
        linked_ids = set()
        for track in tracklets:
            for d in track:
                linked_ids.add(id(d))
            ra = _median_ra([d["ra"] for d in track])
            dec = float(np.median([d["dec"] for d in track]))
            speed, pa = _angular_speed(track[0]["ra"], track[0]["dec"],
                                       track[-1]["ra"], track[-1]["dec"],
                                       track[-1]["time_min"] - track[0]["time_min"])
            obj = DIAObject(obj_id=oid, ra=ra, dec=dec, detections=track,
                            matched_reference=False, n_det=len(track),
                            mean_snr=float(np.mean([d["snr"] for d in track])),
                            kind="mover", motion_arcsec_per_min=float(speed),
                            motion_pa_deg=float(pa))
            obj.rb_score = _rb_score(track, median_fwhm, is_mover=True)
            obj.notes = (f"moving object: {speed:.2f} arcsec/min at PA {pa:.0f} deg, "
                         f"{len(track)} epochs linked")
            objects.append(obj)
            oid += 1
            n_movers += 1
        # leftover orphans -> low-confidence singles (unless novelty flagging is off)
        for d in orphan_unmatched:
            if id(d) in linked_ids:
                continue
            if dcfg.flag_unmatched:
                kind, notes = "single", "single unmatched detection"
            else:
                kind, notes = "known", "unmatched source; novelty flagging disabled"
            obj = DIAObject(obj_id=oid, ra=d["ra"], dec=d["dec"], detections=[d],
                            matched_reference=False, n_det=1, mean_snr=d["snr"],
                            kind=kind, notes=notes)
            obj.rb_score = _rb_score([d], median_fwhm)
            objects.append(obj)
            oid += 1

    # ---- composite ranking --------------------------------------------- #
    for obj in objects:
        obj.rank_score = _composite_rank(obj)

    objects.sort(key=lambda o: o.rank_score, reverse=True)
    candidates = _to_frame(objects)
    n_transients = sum(o.kind == "transient" for o in objects)
    n_variables = sum(o.kind == "variable" for o in objects)

    log.info("Discovery found %d objects: %d movers, %d transients, %d variables "
             "(%d flagged as top candidates)", len(objects), n_movers, n_transients,
             n_variables, int((candidates["rank_score"] > 0.5).sum()) if len(candidates) else 0)
    return DiscoveryResult(candidates=candidates, objects=objects, n_objects=len(objects),
                           n_movers=n_movers, n_transients=n_transients, n_variables=n_variables)


def _composite_rank(obj: DIAObject) -> float:
    """Combine novelty, significance, persistence and behaviour into [0,1]."""
    base = {"mover": 0.85, "transient": 0.8, "variable": 0.55,
            "single": 0.2, "known": 0.02}.get(obj.kind, 0.1)
    snr_term = 1.0 - np.exp(-obj.mean_snr / 20.0)          # saturates with SNR
    persist = min(1.0, obj.n_det / 4.0)
    score = base * (0.4 + 0.3 * snr_term + 0.3 * persist)
    score *= (0.4 + 0.6 * obj.rb_score)                    # quality gate
    if obj.kind == "variable" and np.isfinite(obj.var_significance):
        score *= min(1.3, 1.0 + obj.var_significance / 30.0)
    return float(np.clip(score, 0.0, 1.0))


def _to_frame(objects: list[DIAObject]) -> pd.DataFrame:
    rows = []
    for o in objects:
        frames = sorted({d["frame_index"] for d in o.detections})
        rows.append({
            "obj_id": o.obj_id, "kind": o.kind, "rank_score": round(o.rank_score, 4),
            "ra": round(o.ra, 6), "dec": round(o.dec, 6), "n_det": o.n_det,
            "frames": ",".join(map(str, frames)), "mean_snr": round(o.mean_snr, 1),
            "rb_score": round(o.rb_score, 3),
            "motion_arcsec_min": round(o.motion_arcsec_per_min, 3) if o.kind == "mover" else np.nan,
            "motion_pa_deg": round(o.motion_pa_deg, 1) if o.kind == "mover" else np.nan,
            "var_chi2": round(o.var_chi2, 2) if np.isfinite(o.var_chi2) else np.nan,
            "var_amp_mag": round(o.var_amplitude, 3) if np.isfinite(o.var_amplitude) else np.nan,
            "notes": o.notes,
        })
    return pd.DataFrame(rows)
