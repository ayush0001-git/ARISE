"""ARISE pipeline orchestrator.

Wires the stages together end-to-end for a directory of raw frames:

    ingest -> masters -> [per science frame: reduce -> CR -> background ->
    extract -> astrometry -> photometry] -> discovery -> QA -> report

Writes reduced multi-extension FITS, per-frame catalogs, a ranked candidate
list, and a QA summary. Everything is driven by a :class:`PipelineConfig`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from .config import PipelineConfig
from .catalogs import get_reference_catalog, load_local_reference
from .fitsio import read_frame
from .logs import get_logger, setup_logging
from .stages.ingest import ingest
from .stages.calibrate import build_masters, reduce_frame
from .stages.cosmicray import reject_cosmic_rays, denoise
from .stages.background import model_background
from .stages.sources import extract_sources
from .stages.astrometry import solve_astrometry
from .stages.photometry import calibrate_photometry
from .stages.discovery import run_discovery, FrameResult, DiscoveryResult

log = get_logger("pipeline")


@dataclass
class FrameQA:
    name: str
    n_sources: int = 0
    n_cosmic_rays: int = 0
    fwhm_arcsec: float = float("nan")
    fwhm_pix: float = float("nan")
    sky_adu: float = float("nan")
    bkg_rms_adu: float = float("nan")
    astrom_rms_arcsec: float = float("nan")
    n_astrom_matched: int = 0
    zeropoint: float = float("nan")
    zp_scatter: float = float("nan")
    limiting_mag: float = float("nan")
    airmass: float = float("nan")
    ok: bool = True               # False when the frame failed and was skipped
    error: str = ""               # exception text for a failed frame


@dataclass
class PipelineResult:
    config: PipelineConfig
    instrument_name: str
    frame_qa: list[FrameQA] = field(default_factory=list)
    discovery: DiscoveryResult | None = None
    reference_size: int = 0
    n_science: int = 0
    outputs: dict[str, str] = field(default_factory=dict)
    dossiers: dict[int, str] = field(default_factory=dict)   # obj_id -> html file


# --------------------------------------------------------------------------- #
def _parse_time_min(dateobs: str) -> float | None:
    """Epoch time in minutes (MJD * 1440) from a DATE-OBS-like header value.

    Accepts ISO/ISOT strings as well as numeric MJD-OBS (~20000-80000) and
    JD (> 2,000,000) values, which arrive here as bare number strings that
    astropy ``Time`` cannot guess. Returns ``None`` when unparseable.
    """
    if not dateobs:
        return None
    from astropy.time import Time
    try:
        return float(Time(dateobs, format="isot", scale="utc").mjd * 1440.0)
    except Exception:
        pass
    try:
        val = float(dateobs)
    except (TypeError, ValueError):
        try:
            return float(Time(dateobs).mjd * 1440.0)
        except Exception:
            return None
    try:
        if 20000.0 <= val <= 80000.0:       # MJD-OBS (years ~1941-2107)
            return float(Time(val, format="mjd", scale="utc").mjd * 1440.0)
        if val > 2_000_000.0:               # JD
            return float(Time(val, format="jd", scale="utc").mjd * 1440.0)
    except Exception:
        pass
    return None


def _frame_time_min(dateobs: str, fallback_index: int) -> float:
    t = _parse_time_min(dateobs)
    return t if t is not None else float(fallback_index) * 1.0


def _write_reduced_fits(path: Path, sci: np.ndarray, var: np.ndarray,
                        dq: np.ndarray, header) -> None:
    """Write a BANZAI/DRAGONS-style multi-extension FITS: SCI + ERR + DQ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    err = np.sqrt(np.clip(var, 0, None)).astype(np.float32)
    hdr = header.copy() if header is not None else fits.Header()
    hdr["ARISERED"] = (True, "reduced by ARISE")
    hdus = [
        fits.PrimaryHDU(np.asarray(sci, np.float32), header=hdr),
        fits.ImageHDU(err, name="ERR"),
        fits.ImageHDU(np.asarray(dq, np.uint8), name="DQ"),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the full ARISE reduction + discovery pipeline."""
    paths = config.paths
    reports_dir = Path(paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config.log_level, log_file=reports_dir / "arise.log")

    inst = config.resolve_instrument()
    log.info("ARISE pipeline starting | instrument=%s (%s)", inst.name, inst.telescope)

    # ---- ingest + masters ---------------------------------------------- #
    frames = ingest(paths.raw, inst)
    if not frames.light:
        raise RuntimeError(f"No science (light) frames found in {paths.raw}")
    masters = build_masters(frames, inst, out_dir=paths.master)

    # ---- frame times (chronological; index fallback is all-or-nothing) --- #
    lights = frames.lights_sorted()
    times = [_parse_time_min(m.dateobs) for m in lights]
    if any(t is None for t in times):
        n_bad = sum(t is None for t in times)
        log.warning("%d/%d science frames have a missing/unparseable date; using "
                    "the frame index as epoch for ALL frames (motion rates will "
                    "be per-frame, not per-minute)", n_bad, len(lights))
        times = [float(i) for i in range(len(lights))]
    else:
        # headers may mix ISO and numeric MJD/JD date formats, which do not
        # sort correctly as raw strings -- re-order by the parsed times
        order = sorted(range(len(lights)), key=lambda i: (times[i], lights[i].path.name))
        lights = [lights[i] for i in order]
        times = [times[i] for i in order]
    t0 = times[0]

    # ---- pointing groups + reference catalogs (local file wins) ---------- #
    clusters = _cluster_pointings(lights, inst)
    if len(clusters) > 1:
        log.warning("=" * 60)
        log.warning("MULTIPLE FIELDS DETECTED: the %d science frames span %d "
                    "distinct pointings; reference catalogs and discovery are "
                    "run per field.", len(lights), len(clusters))
        for k, idxs in enumerate(clusters):
            log.warning("  field %d: %d frame(s), first %s", k + 1, len(idxs),
                        lights[idxs[0]].path.name)
        log.warning("=" * 60)
    references = [_acquire_reference(config, [lights[i] for i in idxs], inst)
                  for idxs in clusters]
    cluster_of = {i: k for k, idxs in enumerate(clusters) for i in idxs}
    reference_all = references[0] if len(references) == 1 else pd.concat(
        references, ignore_index=True).drop_duplicates(ignore_index=True)

    # ---- per-frame reduction + measurement ------------------------------ #
    frame_results: list[FrameResult] = []
    qa_list: list[FrameQA] = []
    all_catalogs: list[pd.DataFrame] = []

    for i, meta in enumerate(lights):
        log.info("--- Science frame %d/%d: %s ---", i + 1, len(lights), meta.path.name)
        ref = references[cluster_of[i]]
        try:
            raw_data, header = read_frame(meta.path)
            rf = reduce_frame(meta, masters, inst)

            cr = reject_cosmic_rays(rf.data, meta, inst, config.cosmic_ray)
            data = denoise(cr.data, config.denoise)

            dq = np.zeros(data.shape, np.uint8)
            dq[cr.mask] |= 1
            if rf.bad_pixel_mask is not None:
                dq[rf.bad_pixel_mask] |= 2

            bkg = model_background(data, config.background, config.detect,
                                   mask=rf.bad_pixel_mask)

            # extract once with the header WCS, then refine astrometry and recompute
            # sky coordinates with the validated WCS (avoids a second extraction).
            header_wcs = _safe_wcs(header)
            ext = extract_sources(bkg, rf.variance, meta, inst, config.detect, wcs=header_wcs)
            astro = solve_astrometry(meta, ext.catalog, header, inst, config.astrometry,
                                     reference=ref)
            wcs = astro.wcs if astro.wcs is not None else header_wcs
            if wcs is not None and len(ext.catalog):
                ra, dec = wcs.all_pix2world(ext.catalog["x"].to_numpy(),
                                            ext.catalog["y"].to_numpy(), 0)
                ext.catalog["ra"] = ra
                ext.catalog["dec"] = dec

            cat, phot = calibrate_photometry(ext.catalog, meta, inst, config.photometry,
                                             reference=ref)

            fwhm_arcsec = ext.median_fwhm * inst.pixel_scale
            qa = FrameQA(
                name=meta.path.name, n_sources=ext.n_sources, n_cosmic_rays=cr.n_flagged,
                fwhm_arcsec=fwhm_arcsec, fwhm_pix=ext.median_fwhm,
                sky_adu=bkg.median, bkg_rms_adu=bkg.median_rms,
                astrom_rms_arcsec=astro.residual_rms_arcsec, n_astrom_matched=astro.n_matched,
                zeropoint=phot.zeropoint, zp_scatter=phot.zp_scatter,
                limiting_mag=phot.limiting_mag, airmass=meta.airmass or float("nan"),
            )

            # persist reduced frame + catalog
            _write_reduced_fits(Path(paths.reduced) / f"reduced_{meta.path.name}",
                                data, rf.variance, dq, header)
            cat_path = Path(paths.catalogs) / f"{Path(meta.path.name).stem}_catalog.csv"
            cat_path.parent.mkdir(parents=True, exist_ok=True)
            cat.to_csv(cat_path, index=False)
        except Exception as exc:
            # one bad frame must not kill the night: record it in QA and move on
            log.error("Frame %s failed and was skipped: %s", meta.path.name, exc,
                      exc_info=True)
            qa_list.append(FrameQA(name=meta.path.name, ok=False, error=str(exc)))
            continue

        qa_list.append(qa)
        all_catalogs.append(cat.assign(frame_index=i))
        frame_results.append(FrameResult(
            name=meta.path.name, frame_index=i,
            time_min=times[i] - t0,
            catalog=cat, median_fwhm=ext.median_fwhm,
            dateobs=meta.dateobs,
        ))

    if not frame_results:
        raise RuntimeError(f"All {len(lights)} science frames failed")

    # ---- discovery (per field; candidates ranked jointly) ---------------- #
    parts: list[DiscoveryResult] = []
    for k, idxs in enumerate(clusters):
        members = set(idxs)
        cluster_frames = [fr for fr in frame_results if fr.frame_index in members]
        if not cluster_frames:
            continue
        try:
            parts.append(run_discovery(cluster_frames, references[k], config.discovery))
        except Exception as exc:
            # a discovery crash must not destroy an otherwise-good night
            log.error("Discovery failed for field %d/%d: %s", k + 1, len(clusters),
                      exc, exc_info=True)
    discovery = _merge_discoveries(parts)

    # ---- write catalog products + QA ------------------------------------ #
    outputs: dict[str, str] = {}
    if all_catalogs:
        master_cat = pd.concat(all_catalogs, ignore_index=True)
        mc_path = Path(paths.catalogs) / "all_sources.csv"
        master_cat.to_csv(mc_path, index=False)
        outputs["all_sources"] = str(mc_path)
    if len(discovery.candidates):
        cand_path = Path(paths.catalogs) / "candidates.csv"
        top = discovery.candidates.head(config.discovery.top_n)
        top.to_csv(cand_path, index=False)
        outputs["candidates"] = str(cand_path)

    # plain-English summary (also feeds the Ask-ARISE knowledge base)
    try:
        (reports_dir / "run_summary.txt").write_text(
            _run_summary_text(inst, qa_list, discovery, len(lights)), encoding="utf-8")
    except Exception as exc:
        log.warning("run summary not written: %s", exc)

    qa_path = reports_dir / "qa_summary.json"
    with open(qa_path, "w", encoding="utf-8") as fh:
        # bare NaN/Infinity are invalid strict JSON: sanitize to null, and keep
        # allow_nan=False so a regression fails loudly instead of silently
        json.dump(_sanitize_json({"instrument": inst.name, "telescope": inst.telescope,
                                  "frames": [asdict(q) for q in qa_list],
                                  "reference_size": len(reference_all)}),
                  fh, indent=2, allow_nan=False, default=_json_default)
    outputs["qa_summary"] = str(qa_path)

    result = PipelineResult(
        config=config, instrument_name=inst.name, frame_qa=qa_list,
        discovery=discovery, reference_size=len(reference_all),
        n_science=len(lights), outputs=outputs,
    )

    # ---- discovery dossiers (per-candidate science vetting) -------------- #
    if config.discovery.dossiers:
        try:
            from .stages.dossier import build_dossiers
            result.dossiers = build_dossiers(
                discovery, frame_results, paths.reduced, reports_dir,
                instrument_name=f"{inst.name} ({inst.telescope})",
                online=config.discovery.dossiers_online)
            if result.dossiers:
                outputs["night_brief"] = str(reports_dir / "night_brief.md")
        except Exception as exc:
            log.warning("Dossier generation failed: %s", exc, exc_info=True)

    # ---- report --------------------------------------------------------- #
    try:
        from .stages.report import build_report
        report_path = build_report(result, frame_results, reference_all, reports_dir)
        outputs["report"] = str(report_path)
        log.info("HTML report: %s", report_path)
    except Exception as exc:
        log.warning("Report generation failed: %s", exc)

    _log_summary(result)
    return result


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _safe_wcs(header):
    try:
        w = WCS(header)
        return w if w.has_celestial else None
    except Exception:
        return None


def _field_centre(meta) -> tuple[float, float] | None:
    """Best-effort field centre in degrees: header RA/Dec, else WCS CRVAL."""
    if meta.ra is not None and meta.dec is not None:
        return float(meta.ra), float(meta.dec)
    try:
        w = _safe_wcs(read_frame(meta.path)[1])
    except Exception:
        return None
    if w is not None:
        return float(w.wcs.crval[0]), float(w.wcs.crval[1])
    return None


def _delta_ra_deg(ra2: float, ra1: float) -> float:
    """Signed RA difference wrapped into [-180, 180) degrees."""
    return ((ra2 - ra1 + 180.0) % 360.0) - 180.0


def _cluster_pointings(lights, inst) -> list[list[int]]:
    """Group the time-sorted science frames into pointing clusters.

    Frames further than ~half the field of view from a cluster's anchor
    pointing belong to a different field. Frames with no usable pointing
    (no RA/Dec keywords and no WCS) cannot be told apart, so they join the
    first cluster. Returns lists of indices into ``lights``.
    """
    link_deg = max((inst.fov_arcmin or 15.0) / 60.0 * 0.5, 1e-3)
    clusters: list[list[int]] = []
    anchors: list[tuple[float, float]] = []
    unplaced: list[int] = []
    for i, meta in enumerate(lights):
        centre = _field_centre(meta)
        if centre is None:
            unplaced.append(i)
            continue
        ra, dec = centre
        for k, (ara, adec) in enumerate(anchors):
            d_ra = _delta_ra_deg(ra, ara) * np.cos(np.radians(adec))
            if float(np.hypot(d_ra, dec - adec)) <= link_deg:
                clusters[k].append(i)
                break
        else:
            clusters.append([i])
            anchors.append((ra, dec))
    if unplaced:
        if clusters:
            if len(clusters) > 1:
                log.warning("%d frame(s) have no usable pointing; assigning them "
                            "to the first field", len(unplaced))
            clusters[0] = sorted(clusters[0] + unplaced)
        else:
            clusters.append(unplaced)
    return clusters


def _acquire_reference(config: PipelineConfig, lights, inst) -> pd.DataFrame:
    """Reference catalog for one pointing group of science frames."""
    # 1) local file next to the raw frames
    local = Path(config.paths.raw) / "reference_catalog.csv"
    df = load_local_reference(local)
    if df is not None and len(df):
        return df
    # 2) online query centred on the field (first frame with a usable pointing)
    for meta0 in lights:
        centre = _field_centre(meta0)
        if centre is not None:
            ra, dec = centre
            radius_deg = (inst.fov_arcmin or 15.0) / 60.0 * 0.9
            return get_reference_catalog(config.photometry.ref_catalog, ra, dec, radius_deg)
    log.warning("No field centre available; skipping reference catalog")
    return pd.DataFrame({"ra": [], "dec": [], "mag": []})


def _merge_discoveries(parts: list[DiscoveryResult]) -> DiscoveryResult:
    """Combine per-field discovery results (unique obj_ids; ranked jointly)."""
    if not parts:
        return DiscoveryResult(pd.DataFrame(), [], 0, 0, 0, 0)
    if len(parts) == 1:
        return parts[0]
    objects = []
    cands: list[pd.DataFrame] = []
    offset = 0
    for part in parts:
        cand = part.candidates.copy()
        if len(cand):
            cand["obj_id"] = cand["obj_id"] + offset
            cands.append(cand)
        for o in part.objects:
            o.obj_id += offset
        objects.extend(part.objects)
        offset += len(part.objects)
    objects.sort(key=lambda o: o.rank_score, reverse=True)
    candidates = (pd.concat(cands, ignore_index=True)
                  .sort_values("rank_score", ascending=False, ignore_index=True)
                  if cands else pd.DataFrame())
    return DiscoveryResult(candidates=candidates, objects=objects,
                           n_objects=sum(p.n_objects for p in parts),
                           n_movers=sum(p.n_movers for p in parts),
                           n_transients=sum(p.n_transients for p in parts),
                           n_variables=sum(p.n_variables for p in parts))


def _run_summary_text(inst, qa_list, discovery, n_frames: int) -> str:
    """Human-readable narrative of the run (indexed by the Ask-ARISE assistant)."""
    def med(attr):
        vals = [getattr(q, attr) for q in qa_list if getattr(q, attr) == getattr(q, attr)]
        return float(np.median(vals)) if vals else float("nan")

    lines = [
        "ARISE run summary (latest run).",
        f"Instrument: {inst.name} ({inst.telescope}).",
        f"The pipeline reduced {n_frames} science frames.",
        f"Median seeing (FWHM) was {med('fwhm_arcsec'):.2f} arcsec.",
        f"Median photometric zero point was {med('zeropoint'):.3f} mag "
        f"with scatter {med('zp_scatter'):.3f} mag.",
        f"Median 5-sigma limiting magnitude was {med('limiting_mag'):.2f}.",
        f"Median astrometric residual was {med('astrom_rms_arcsec'):.3f} arcsec.",
    ]
    if discovery:
        lines.append(
            f"Discovery results: the run found and tracked {discovery.n_objects} objects, "
            f"of which {discovery.n_movers} moving object(s) (asteroid/NEO candidates), "
            f"{discovery.n_transients} transient(s) (new sources with no catalog match), "
            f"and {discovery.n_variables} variable star(s).")
        for _, r in discovery.candidates.head(8).iterrows():
            if r["kind"] in ("mover", "transient", "variable"):
                lines.append(
                    f"Candidate (discovered): {r['kind']} at RA {r['ra']:.5f} deg, "
                    f"Dec {r['dec']:.5f} deg, rank score {r['rank_score']:.3f}, "
                    f"seen in {r['n_det']} epochs. {r['notes']}")
    return "\n".join(lines)


def _sanitize_json(o):
    """Recursively replace NaN/inf with None (bare NaN is invalid strict JSON)."""
    if isinstance(o, dict):
        return {k: _sanitize_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize_json(v) for v in o]
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


def _log_summary(result: PipelineResult) -> None:
    d = result.discovery
    log.info("=" * 60)
    log.info("ARISE complete: %d science frames reduced", result.n_science)
    if result.frame_qa:
        fwhm = np.nanmedian([q.fwhm_arcsec for q in result.frame_qa])
        zp = np.nanmedian([q.zeropoint for q in result.frame_qa])
        lim = np.nanmedian([q.limiting_mag for q in result.frame_qa])
        log.info("Median seeing %.2f arcsec | median ZP %.3f | median limiting mag %.2f",
                 fwhm, zp, lim)
    if d:
        log.info("Discovery: %d objects | %d movers, %d transients, %d variables",
                 d.n_objects, d.n_movers, d.n_transients, d.n_variables)
        if len(d.candidates):
            top = d.candidates.iloc[0]
            log.info("Top candidate: %s (rank %.3f) - %s",
                     top["kind"], top["rank_score"], top["notes"])
    log.info("=" * 60)
