# ARISE

**Automated Reduction & Intelligent Source Extraction**

ARISE turns raw telescope frames into scientifically usable, calibrated
catalogs — and then hunts them for genuinely *new* sources: moving objects
(asteroids/NEOs), transients, and variable stars. It is a single-command,
offline-capable, config-profile-per-instrument pipeline built on the astropy
ecosystem, designed for the telescopes of **ARIES** (Aryabhatta Research
Institute of Observational Sciences, Nainital) and any other optical CCD imager.

It mirrors how modern production observatory pipelines (LCO *BANZAI*, Gemini
*DRAGONS*, ZTF, and Rubin/LSST alert production) actually work, scaled to a
single-institution facility.

---

## Why this exists

Every clear night, a telescope produces hundreds of raw frames that are useless
until an expert removes the instrument's fingerprints (bias, dark, flat), kills
cosmic rays, models the sky, extracts sources, and ties them to real sky
coordinates and magnitudes. Doing that by hand does not scale — and the
*interesting* objects (a new asteroid, a supernova, a flaring star) are easy to
miss in the flood. ARISE automates the entire path and puts the most promising
candidates at the top of a ranked list for a human to confirm.

---

## What it does (pipeline)

```
 ingest ─▶ master calibration ─▶ reduce ─▶ cosmic-ray rejection ─▶ sky background
        ─▶ detect + star mask ─▶ extract ─▶ astrometry ─▶ photometry
        ─▶ DISCOVERY (novelty + ranking) ─▶ QA metrics + HTML report
```

| Stage | What ARISE does | Engine |
|---|---|---|
| **Ingest** | Classify bias/dark/flat/science from headers via a per-instrument keyword map | `astropy.io.fits` |
| **Master calibration** | Sigma-clipped (MAD) **mean** master bias, dark-rate and per-filter flats; hot/bad-pixel map | `astropy.stats` |
| **Reduce** | `(raw − bias − dark·t) / flat` with a propagated read-noise + Poisson **variance** plane | ccdproc-style |
| **Cosmic rays** | L.A.Cosmic (van Dokkum) with the correct electron noise model; CR mask kept as a DQ plane | `astroscrappy` |
| **Sky background** | SExtractor-style 2-D mesh with **iterative source masking** | `photutils.Background2D` |
| **Detect + mask** | Matched-filter detection at *N*σ above the RMS map; watershed deblending; star mask | `photutils.segmentation` |
| **Extract** | Rich catalog: centroid, sky coords, segment/Kron/**fixed-aperture** flux + errors, SNR, FWHM, ellipticity, flags | `photutils.SourceCatalog` |
| **Astrometry** | Use + **refine** header WCS against a reference catalog; Gaia residual-RMS sanity gate (or plate-solve) | `astropy.wcs`, astrometry.net |
| **Photometry** | Zero point from **isolated** reference stars (sigma-clipped), airmass extinction, calibrated magnitudes, limiting mag | `photutils`, `astroquery` |
| **Discovery** | Cross-match novelty flag, moving-object linking, variability, real/bogus scoring, composite rank | ARISE |
| **QA + report** | Per-frame seeing/ZP/limiting-mag/astrometric-residual, ranked candidates, cutout stamps | self-contained HTML |

---

## The discovery engine

This is the point of ARISE. Detections are aggregated across the night into
per-sky-position objects (a *DIAObject* analog), and new sources are surfaced
through several converging signals, then combined into one composite rank:

- **Catalog novelty** — detections with no match in the reference catalog
  (Gaia DR3 / Pan-STARRS, or a local catalog offline).
- **Moving objects** — orphan detections linked into a constant-velocity
  tracklet across epochs (MOPS-style `findTracklets`/`linkTracklets`), reporting
  sky motion and position angle — asteroid / NEO candidates.
- **Variables** — catalogued stars whose calibrated light curve is a genuine
  outlier on the ensemble RMS-vs-magnitude relation across multiple epochs
  (robust to single-frame glitches).
- **Transients** — unmatched sources that are stationary and repeat.
- **Real vs bogus** — a quality score that demotes cosmic-ray residuals, edge
  junk, and one-frame artifacts.

The canonical "this is genuinely new" signal is *high SNR + repeated detection +
no catalog match + not a known mover* — which ARISE puts at the top of the queue.

### Discovery Dossiers — the week-of-vetting, automated

For each top candidate ARISE then does what a human vetting team spends
hours-to-days on, automatically:

- **Evidence pack** — epoch-by-epoch cutouts, light curve, fitted sky track.
- **Known-object vetting** — live cone-search of **SkyBoT** (IMCCE's
  known-solar-system-objects service) for movers and **SIMBAD** for stationary
  sources. *No match = potentially a genuine discovery.*
- **Follow-up prediction** — constant-velocity fit with predicted positions at
  +1 h and +24 h, so the object can be re-acquired tomorrow night.
- **MPC submission draft** — an 80-column astrometric report in the Minor
  Planet Center's optical format, ready to submit once an observatory code is
  filled in.
- **Verdict + recommended action** in plain language, plus a **night brief**
  (`night_brief.md`) summarising what a human should do in the morning.

Every check is best-effort with short timeouts — offline, the dossier still
builds and states which services could not be reached.

---

## Quick start

**Requirements:** Python 3.10+ (3.12 recommended). All dependencies install as
prebuilt wheels on Windows/Linux/macOS — no compilers, no external binaries.

### Windows (one-click)

1. Double-click **`setup.bat`** (installs dependencies — once).
2. Double-click **`run_demo.bat`**.

### Any platform (terminal)

```bash
pip install -r requirements.txt        # or: pip install -e .
python -m arise.cli demo --open        # generate a synthetic night + reduce it
```

### Web console (drag & drop)

Double-click **`run_app.bat`** (or `python -m arise.webapp`) and open
http://127.0.0.1:8770 :

- **Drop telescope frames** — FITS files, a ZIP of a night, or even a plain
  PNG/JPG (auto-converted; extraction-only mode) — the pipeline runs and the
  report appears inline with live progress.
- **Run demo night** — one click, no data needed.
- **Ask ARISE** — a built-in retrieval (RAG) assistant that answers questions
  from the project docs, the research brief, and *your latest run's own
  results* ("what did the last run discover?"). Fully offline; paste URLs
  (API docs, papers) to widen its knowledge base — they persist in `kb/urls.txt`.
  If `ANTHROPIC_API_KEY` is set it upgrades to fluent generated answers.
- **Personal knowledge** — point it at any folder of notes (an Obsidian vault
  is just a folder of markdown): list the path in `kb/sources.txt`, one per
  line, and Ask-ARISE answers from your own notes too.

### API keys (optional, all free)

Put keys in `config/keys.yaml` (gitignored — never commit it). Each unlocks a
feature; everything still runs without them:

| Key | Get it from | Unlocks |
|---|---|---|
| `astrometry_net` | nova.astrometry.net (profile page) | blind plate solving of frames with no WCS |
| `nasa` | api.nasa.gov (instant email signup) | NEO close-approach context in mover dossiers |
| `nvidia` (list) | build.nvidia.com | fluent generated Ask-ARISE answers (keys rotate on rate limits) |
| `anthropic` | console.anthropic.com | preferred generative backend when present |

The demo synthesises a realistic observing night — calibration frames plus a
science sequence containing a **hidden moving asteroid, a transient, and a
variable star** — runs the full pipeline, and opens an HTML report. Because the
planted objects have known truth values, the run is self-validating: ARISE
recovers all three and ranks them at the top.

### Run on your own data

```bash
# point ARISE at a folder of FITS frames (bias/dark/flat/science together)
arise run --raw /path/to/night --instrument dot_4kx4k --base out --open

# or use a config file
arise run --config config/dfot_demo.yaml
```

Other commands: `arise instruments` (list profiles), `arise synth` (only make
synthetic data), `arise init-config my.yaml` (write a documented config).

---

## Instrument profiles

`arise instruments` lists them. Built in: `dot_4kx4k` (3.6 m DOT 4Kx4K IMAGER),
`adfosc` (3.6 m DOT + ADFOSC), `dfot_2kx2k` (1.3 m DFOT), `st_1m` (1.04 m
Sampurnanand), and `generic`. Each profile carries gain, read noise, pixel
scale, saturation, filters and a FITS header-keyword map. **Header values always
override the profile**, so a slightly wrong default never corrupts a reduction,
and unlisted instruments work via `generic` + header keywords or a custom
profile in your config.

---

## Outputs

```
data/master/     master_bias.fits, master_dark_rate.fits, master_flat_<filter>.fits
data/reduced/    reduced_<frame>.fits    (multi-extension: SCI + ERR + DQ)
data/catalogs/   <frame>_catalog.csv, all_sources.csv, candidates.csv (ranked)
data/reports/    arise_report.html, qa_summary.json, arise.log
```

Every reduced frame is stamped with ARISE provenance keywords; the run log and
QA JSON make each reduction reproducible and auditable.

---

## Example demo result

A 1.3 m DFOT synthetic night (6 science frames), reduced end-to-end:

```
median seeing (FWHM) : 2.64 arcsec
median zero point    : 24.16 mag   (scatter 0.012 mag)
astrometric residual : 0.20 arcsec
5-sigma limiting mag  : 18.0
Discovery: moving asteroid (3.0 arcsec/min, 4 epochs linked)  ranked #1
           transient (no catalog match, 4 epochs)             ranked #2
           variable star (amp 0.51 mag, reduced chi2 1300)    ranked #3
```

All three planted objects recovered, correctly classified, with no false
positives above the reporting threshold.

---

## Design notes for reviewers

The design and its parameter choices are documented, with citations to the
primary literature and to the astropy/observatory-pipeline docs, in
[`docs/RESEARCH.md`](docs/RESEARCH.md). Highlights:

- **Correct noise handling end-to-end** — read-noise + Poisson variance is built
  and propagated; the CR mask is kept as a DQ plane (cleaned pixels are flagged,
  not trusted). ARISE avoids the classic *double-gain* bug in CR cleaning.
- **Photometry-safe by default** — denoising is off; the science array is never
  blurred (blurring biases centroids/photometry and correlates noise).
- **Per-instrument, header-driven** gain/read-noise/pixel-scale; all radii and
  thresholds derive from the real pixel scale.
- **Validated astrometry** — a Gaia/reference residual-RMS gate catches
  confident-but-wrong solutions instead of trusting them blindly.
- **Discovery that mirrors ZTF/Rubin** — novelty flag, tracklet linking, robust
  variability, real/bogus scoring, composite ranking.

## Roadmap

Difference imaging (ZOGY/HOTPANTS) against a template; a trained real/bogus CNN
on instrument-specific stamps; MPChecker/`digest2` vetting of movers;
NIR-array (up-the-ramp) and ILMT drift-scan reduction paths; parallel
multi-core frame processing.

## Testing

```bash
pip install pytest && pytest -q
```

The suite includes unit tests for the building blocks and an end-to-end
integration test that generates a synthetic night and asserts ARISE recovers
the planted asteroid, transient, and variable with the right parameters.

## License

MIT.
