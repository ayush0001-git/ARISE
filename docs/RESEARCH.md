# ARISE — Automated Reduction & Intelligent Source Extraction
## Authoritative Design Brief (ARIES Observatory Pipeline)

---

## 1. Executive Summary

**ARISE** is a pure-Python, offline-capable, config-profile-driven pipeline that takes raw FITS frames from ARIES telescopes, produces calibrated science images with propagated uncertainty and data-quality masks, extracts a rich source catalog with full astrometric and photometric calibration, and then surfaces **new** astronomical sources — transients, movers (asteroids/NEOs), and variables — by difference imaging, catalog cross-match, moving-object linking, and a real-bogus ranking layer. It is architected to mirror how modern production observatory pipelines (LCO BANZAI, Gemini DRAGONS, ZTF, Rubin/LSST alert production) actually work, scaled to a single-institution facility.

The design rests on four load-bearing decisions, each defensible from first principles:

1. **Pure-Python, external-binary-optional core.** The reduction/detection/photometry/calibration stack is built entirely on the astropy ecosystem — `astropy`, `ccdproc`, `photutils`, `astroscrappy`, `sep`, `astroquery`, `reproject` [1][2][3]. Heavy C binaries (astrometry.net `solve-field`, SExtractor, PSFEx, SWarp, SCAMP, HOTPANTS) are supported as **optional accelerators with pure-Python fallbacks**. This is a direct correction of RedPipe's biggest portability weakness — a hard IRAF/pyraf dependency — and is what makes ARISE installable on the user's Windows machine and runnable offline [1].

2. **Config-profile-per-instrument, header-driven per-frame.** ARIES instruments expose *user-selectable* gain and readout speed (DOT 4Kx4K: 5 gains × 3 speeds; DFOT: 2 speeds), so gain, read-noise, saturation, pixel scale, filter set, and the FITS header-keyword map are **read per-frame from the header, falling back to a per-instrument profile** — never hard-coded [1]. Hard-coding a single gain/read-noise/pixel-scale corrupts uncertainty maps, cosmic-ray rejection, saturation handling, and every aperture radius across instruments [1].

3. **Detector-type branching.** Optical CCDs (bias/dark/flat) and NIR arrays (up-the-ramp slope fit + reference-pixel + non-linearity) are genuinely different reduction physics. ARISE selects a `ccd` backend or an `ir_array` backend on `detector_type`, keeping photometry/astrometry/calibration shared downstream [1]. This is what lets one pipeline serve DOT 4Kx4K *and* TANSPEC/TIRCAM2 correctly instead of producing garbage on NIR data.

4. **Fixed, provenance-logged order of operations with data+uncertainty+mask carried together.** Every frame is a `CCDData` object with explicit unit, `StdDevUncertainty`, and boolean mask from ingest onward; every stage is non-mutating, propagates all three planes, and stamps a provenance keyword into the FITS header [2]. This is the BANZAI/DRAGONS reproducibility model and it is what makes ARISE credible as *production* software rather than a demo script.

The discovery layer is not bolted on — it is the point. ARISE emulates the ZTF/Rubin alert-production architecture (detrend → register/PSF/zero-point → subtract → detect → real-bogus → catalog cross-match → link/vary → composite rank), persisting a per-sky-position object record (a DIAObject analog) so variability, moving-object linking, and de-duplication all fall out of one aggregated history [6].

---

## 2. ARIES Instrument Context

ARIES operates telescopes at **Manora Peak, Nainital** (1.04 m Sampurnanand) and at **Devasthal** (~2450 m: 3.6 m DOT, 1.3 m DFOT, 4 m ILMT) [1]. Each detector below becomes a config profile carrying: gain table, read-noise table, pixel scale, saturation, default filters, default extinction coefficients, and the header-keyword map. Gain/read-noise are read *per frame* where the header exposes them.

| Instrument (profile) | Telescope | Detector | Format / pixel | Pixel scale | FOV | Gain (e⁻/ADU) | Read noise (e⁻) | Bands | detector_type |
|---|---|---|---|---|---|---|---|---|---|
| **DOT-4Kx4K** | 3.6 m DOT (axial f/9) | STA4150 CCD, LN₂-cooled | 4096² · 15 µm | ~0.095″/px | 6.5′×6.5′ | 1, 2, 3, 5, 10 (selectable) | per gain×speed (100k/500k/1M Hz) | UBVRI + ugriz + Clear | ccd |
| **ADFOSC-imaging** | 3.6 m DOT (axial) | E2V CCD231-84, −120 °C | 4096² (61.4 mm) | ~0.2″/px | 13.6′×13.6′ | ~1 | ~6 | broad/narrow-band | ccd |
| **DFOT-2Kx2K** | 1.3 m DFOT (f/4) | Andor E2V, −80 °C | 2048² · 13.5 µm | ~0.53″/px | 18′×18′ | 0.7 @31 kHz / 2 @1 MHz | ~2.5 @31 kHz / ~7 @1 MHz | UBVRI, ugriz, Hα/[OIII]/[SII] | ccd |
| **DFOT-EMCCD** | 1.3 m DFOT | Andor EMCCD E2V, −90 °C | 512² · 16 µm | — | high-cadence | (EM regime) | (EM regime) | as above | ccd |
| **Sampurnanand-4Kx4K / 2Kx2K** | 1.04 m ST (f/13) | 4Kx4K / 2Kx2K CCD | 24 µm | 0.37″/px (2K) | ~13′×13′ | (profile) | (profile) | UBVRI (+ pol) | ccd |
| **ILMT-TDI** | 4 m ILMT (zenith) | CCD, **drift-scan** | 4096² | ~0.33″/px | 22.3′ strip | (profile) | (profile) | SDSS g′r′i′ | ccd (TDI path) |
| **TIRCAM2** *(optional)* | 3.6 m DOT (side-port1) | Raytheon InSb Aladdin-III, 35 K | 512² | 0.169″/px | 86.5″×86.5″ | (profile) | (profile) | J,H,K,PAH,nbL | ir_array |
| **TANSPEC** *(optional)* | 3.6 m DOT (axial) | Teledyne H2RG HgCdTe | 2048² | 0.245″/px (imager) | 60″×60″ | ~1.07 (high-gain) | ~24.4 | 0.55–2.5 µm | ir_array |

**Two special code paths beyond stare-mode CCDs:**
- **ILMT-TDI** cannot use frame-based flats or standard stare astrometry. It needs column-wise response calibration, 102.35 s effective exposure per 22.3′ strip, and drift-scan astrometry/de-trailing [1].
- **NIR arrays (TIRCAM2, TANSPEC)** have *no simple bias frame*; they need sample-up-the-ramp (SUTR/NDR) slope fitting, Savitzky–Golay reference-pixel 1/f correction, and inverse B-spline non-linearity correction (~13% flux recovery), following the HxRGproc/pyTANSPEC pattern [1].

---

## 3. Pipeline Stages

The canonical CCD stage order is enforced (BANZAI/DRAGONS/ccdproc reference): **mask BPM → overscan → trim → (non-linearity) → bias → dark(scaled) → gain → flat → (fringe) → cosmic-ray → background → detect+mask → extract → astrometry → photometry → discovery → QA** [2]. Every frame is a `CCDData` (data + unit + `StdDevUncertainty` + boolean mask); every stage is non-mutating and writes a provenance header keyword.

### 3.1 Ingest & frame classification
- **Algorithm:** header-driven auto-classification (science / bias / dark / flat / arc) followed by human-in-the-loop QA accept/reject, the pyTANSPEC pattern [1].
- **Library/function:** `ccdproc.ImageFileCollection(directory).summary` + `.files_filtered(imagetyp=..., filter=...)`; `astropy.io.fits` for headers [2].
- **Key params/defaults:** per-instrument **header-keyword map** (`OBJECT`, `IMAGETYP`/`IMAGETYPE`, `FILTER`/`FILTER1`/`FILTER2`, `EXPTIME`, `GAIN`, `RDNOISE`, `DATE-OBS`, `RA`/`DEC`, `AIRMASS`, + TANSPEC `GRATING`/`SLIT`). Read `GAIN`/`RDNOISE`/binning/readout-speed from the header per frame; fall back to profile [1].
- **Order:** first stage; builds the `CCDData` with `.unit`, then `ccdproc.create_deviation(ccd, gain=<Q e-/adu>, readnoise=<Q e->)` on the **raw ADU** frame so the Poisson term is correct before any gain scaling [2].
- **Pitfalls:** a rigid header parser fails across DOT/DFOT/ST/ILMT/TANSPEC — keyword mapping **must** be per-instrument configurable [1]. Wrong `image_width/height` later breaks plate solving. SEP requires native-byte-order C-contiguous float arrays: apply `data.byteswap().newbyteorder()` (or `.astype(np.float64)`) on big-endian FITS data read on a little-endian host [4].

### 3.2 Master-calibration construction
- **Algorithm:** sigma-clipped **average** combine with robust center/scale (averaging N frames beats median: read noise ↓ √N) [2].
- **Library/function:** `ccdproc.combine(files, method='average', sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5, sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, mem_limit=350e6)`; flats add `scale=inv_median` where `inv_median=lambda a: 1/np.median(a)` [2].
- **Key params/defaults:** bias = combine zero-second frames; dark = **bias-subtracted** darks combined; flat = bias+dark-removed, **each scaled by 1/median**, combined **per FILTER**, then normalized. `mem_limit` low on modest hardware [2].
- **Order:** masters built before any science calibration; associate by full metadata (filter, exposure, binning, detector/ROI, date proximity) — never by filename [2].
- **Pitfalls:** scaling a dark that still contains bias corrupts the pedestal (bias is not exposure-dependent) — always bias-subtract darks before exposure scaling [2]. Never combine flats of different filters. Plain mean lets cosmic rays/hot outliers survive; plain median of a small stack throws away ~20% of the √N read-noise gain [2]. Add a **master-frame QA gate**: MAD-based pixel-wise comparison of a new master to the previous good master before promotion, catching warm-camera/bad-night calibrations (BANZAI approach) [2].

### 3.3 Calibrate (instrumental-signature removal)
- **Algorithm:** the canonical CCD chain in one deterministic order [2].
- **Library/function:** `ccdproc.ccd_process(ccd, oscan='[201:232,1:100]', trim='[1:200,1:100]', error=True, gain=<Q>, readnoise=<Q>, master_bias=..., dark_frame=..., master_flat=..., exposure_key='EXPTIME', exposure_unit=u.second, dark_scale=True, bad_pixel_mask=bpm)` runs overscan→trim→gain→BPM→uncertainty→bias→(scaled)dark→flat in the correct order [2]. Overscan modeling: `ccdproc.subtract_overscan(..., median=True)` or a low-order `astropy.modeling.models.Polynomial1D`/`Chebyshev1D`; trim is a **separate** `ccdproc.trim_image` [2].
- **NIR branch (`ir_array`):** SUTR slope fit + reference-pixel (Savitzky–Golay) 1/f correction + inverse-B-spline non-linearity, per the HxRGproc/pyTANSPEC pattern — **not** CCD bias/flat logic [1].
- **Non-linearity (optional, CCD):** custom per-pixel polynomial/lookup on `CCDData.data`, applied **after bias, before flat** (ccdproc has no primitive) [2].
- **Fringe (optional, red bands):** gated on filter I/i/z — median-combine object-masked flat-fielded frames into a master fringe, measure per-frame amplitude, **subtract** the scaled fringe **after** flat-fielding (fringe is additive, not multiplicative) [1][2].
- **Key params/defaults:** be consistent — if you overscan/trim science frames, do the identical operation on darks and flats [2]. Use a constant/low-order (1–3) overscan model; median across columns is CR-robust [2]. `flat_correct(min_value=...)` floors low flat pixels against divide-by-zero blow-up [2].
- **Pitfalls:** inconsistent overscan/trim across frame types → shape mismatch / residual gradient [2]. Applying non-linearity after flat-fielding is wrong [2]. Folding fringe into the flat is wrong [2]. Mismatched calibration association (wrong filter/binning, stale warm-camera dark) silently biases photometry — enforce freshness limits (BANZAI flags flats older than ~1–2 weeks) [2]. Skipping defringing on Devasthal thinned-CCD i/z frames leaves large red-band systematics [1].

### 3.4 Cosmic-ray rejection
- **Algorithm:** **frame-count aware.** If ≥2 registered exposures of the field exist → multi-frame sigma-clipped/median combine (removes CRs with *zero* interpolation, no PSF damage — the gold standard). Only for N==1 → single-frame L.A.Cosmic (van Dokkum 2001 Laplacian edge detection) [3][2].
- **Library/function:** multi-frame: `ccdproc.Combiner(...).sigma_clipping(...).median_combine()`. Single-frame: `ccdproc.cosmicray_lacosmic(ccd, sigclip=..., objlim=5.0, gain=1.0, readnoise=..., satlevel=..., niter=4, sepmed=True, cleantype='meanmask', fsmode='median')` wrapping `astroscrappy.detect_cosmics` → `(crmask, cleanarr)` [3].
- **Key params/defaults:** run **in electrons** (apply gain, pass true readnoise & satlevel). Ground-based default `sigclip≈6–7` (van Dokkum's 5/4.5 is too aggressive; the astropy guide used 7 to cut thousands of false detections to ~70 real CRs) [3]. **Raise `objlim` above 5** if bright/undersampled star cores get flagged; use `fsmode='convolve'` with a matched PSF for well-sampled/spectroscopic data [3].
- **Order:** AFTER bias/dark/flat and gain, **BEFORE** sky/background subtraction (L.A.Cosmic estimates its own background). Mask known bad pixels/columns and saturated stars first [3].
- **Pitfalls:** **double gain application** — `cosmicray_lacosmic` has `gain_apply=True` by default and multiplies output by gain; feed electron data with `gain=1.0` (align units) [3][2]. Wrong noise model (data not in electrons / wrong readnoise) is the #1 cause of both false positives and misses [3]. Sky-subtracting before CR breaks detection [3]. Treat the CR mask as a first-class **DQ plane** so downstream photometry *excludes* rather than trusts cleaned pixels [3]. *(Optional upgrade: `deepCR` U-Net — but its pretrained weights are HST ACS-WFC/WFC3-UVIS only and detector-specific; do not apply to ARIES data without retraining [3].)*

### 3.5 Denoise (optional, separate artifact only)
- **Algorithm:** **never blur the photometric science array.** If a denoised product is wanted (display/detection only), produce a *separate labeled* artifact via a flux-preserving method: starlet / à-trous isotropic-undecimated wavelet (astronomy standard, matches near-Gaussian isotropic stellar profiles), or Anscombe VST + BM3D with the Makitalo–Foi **exact** unbiased inverse [3].
- **Library/function:** starlet (à-trous) implementation; `bm3d` + `astropy`/`scikit-image` for the VST path [3].
- **Key params/defaults:** Anscombe = `2*sqrt(x+3/8)`; biased below ~20 counts/pixel — only use at high counts and only for the display copy [3].
- **Order:** off the main science path entirely; the calibrated frame for photometry is never overwritten.
- **Pitfalls:** plain Gaussian/median blur widens the PSF (FWHM), lowers peak counts, biases centroids for asymmetric PSFs, and correlates neighboring-pixel noise — which breaks the independent-Gaussian assumption behind aperture-photometry error bars and detection significances [3]. For pixel replacement prefer `astropy.convolution.interpolate_replace_nans(image, Gaussian2DKernel(...))` (NaN-flag only CR/bad pixels, interpolate only those, good pixels bit-for-bit unchanged) and **flag every replaced pixel** in the DQ plane rather than trusting it as a measurement [3].

### 3.6 Background modeling
- **Algorithm:** SExtractor 2D background **mesh** + mode estimator, run in **two passes with source masking** (documented to reach within 0.2% of true background, 1.5% of true RMS) [4].
- **Library/function:** `photutils.background.Background2D(data, box_size, filter_size=(3,3), sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=SExtractorBackground(), mask=source_mask, coverage_mask=...)` for the rich path; `sep.Background(data, bw=64, bh=64, fw=3, fh=3)` → `bkg.back()`, `bkg.rms()` for the fast SExtractor-identical path [4].
- **Key params/defaults:** `SExtractorBackground = 2.5*median − 1.5*mean`, falling back to median when `|mean−median|/std ≥ 0.3` [4]. `box_size` a few × the largest source (start 64 px; tune per instrument), `filter_size=(3,3)`. High-altitude Devasthal → very low sky; use robust sigma-clipped estimation [1][4]. Persist **background map and RMS map as FITS extensions** [4].
- **Order:** first pass → detect → dilate a source mask (`SegmentationImage.make_source_mask`) → second pass with `mask=` → use the RMS map as the detection-threshold basis and in aperture errors [4].
- **Pitfalls:** under-masking sources biases sky *high* and under-measures faint flux; too-large `box_size` erases real gradients; too-small absorbs source flux into the sky [4]. Threshold on the **2D RMS map**, not a scalar, for frames with varying noise [4].

### 3.7 Detect + mask
- **Algorithm:** convolve with a PSF-matched Gaussian to maximize faint-source S/N, then connected-component segmentation with N-sigma-above-RMS thresholding [4].
- **Library/function:** `kernel = make_2dgaussian_kernel(fwhm, size=...)`; `threshold = bkg.background + n_sigma*bkg.background_rms` (`detect_threshold`); `segment_map = detect_sources(convolved_data, threshold, npixels=..., connectivity=8)` [4].
- **Key params/defaults:** `n_sigma = 5` (matches LSST/Rubin) [4][6]; `npixels`/`DETECT_MINAREA` ~5–10 to suppress noise spikes [4]. Star-finder pass for the stellar sample: `DAOStarFinder(threshold=5*std, fwhm=3.0, sharpness_range=(0.2,1.0), roundness_range=(-1.0,1.0))` after `sigma_clipped_stats` [4].
- **Order:** on the *masked-background-subtracted* frame; pass the **same convolved image** to `SourceCatalog` for centroids/shapes, but measure **fluxes on the unconvolved** background-subtracted image [4].
- **Pitfalls:** **photutils v3.0 renamed args** (`npixels→n_pixels`, `nlevels→n_levels`, `sharplo/sharphi→sharpness_range`, `roundlo/roundhi→roundness_range`, `brightest→n_brightest`, `peakmax→peak_max`) — pin the version and code to the new names [4]. Centroid/shape/flux measured on inconsistent (convolved vs unconvolved) images shifts astrometry/photometry [4]. Mask saturated/bad pixels (`coverage_mask`) before science use [4].

### 3.8 Extract (deblend, measure, catalog)
- **Algorithm:** multi-threshold watershed deblending + Kron/FLUX_AUTO elliptical adaptive aperture, into a rich per-source catalog [4].
- **Library/function:** deblend `deblend_sources(convolved_data, segment_map, n_pixels, n_levels=32, contrast=0.001)`; measure `SourceCatalog(data, segment_map, convolved_data=..., error=..., wcs=...)`; fast path `sep.extract(data_sub, 1.5, err=bkg.globalrms, minarea=5, deblend_nthresh=32, deblend_cont=0.005, filter_type='matched')` [4]. Kron via SEP: `kron_radius(...,6.0)` → `sum_ellipse(...,2.5*kronrad)` with r_min=1.75 circular fallback (`PHOT_AUTOPARAMS 2.5,3.5`) [4].
- **Key params/defaults:** deblend `n_levels=32`, `contrast/deblend_cont≈0.001–0.005` (raise/disable in sparse fields; lower in crowded, watch over-splitting) [4]. **Always OR the Kron radius flag into the photometry flag** [4].
- **Order:** detect → deblend → measure → star-finder pass → write catalog + provenance [4].
- **Rich catalog schema:** id, xcentroid/ycentroid + `sky_centroid` (WCS), shape (semimajor/semiminor_sigma, orientation, ellipticity, eccentricity, elongation, fwhm), **multiple fluxes with errors** (segment/isophotal, Kron/AUTO, fixed-aperture via `sum_circle`, PSF), quality (area, npix, peak, local_background, sharpness/roundness), and a **unified FLAGS bitmask** merging SEP/SExtractor semantics (OBJ_MERGED, OBJ_TRUNC, APER_TRUNC, APER_HASMASKED, APER_NONPOSITIVE, kron-fallback-to-circle) [4].
- **Pitfalls:** `sep.extract` `thresh` is in **sigma only when `err=` is supplied** — forgetting `err` makes it an absolute pixel value [4]. Kron FLUX_AUTO misses low-surface-brightness wings (2.5·Kron ≈ 90–96%) — apply an aperture correction for absolute magnitudes; `kron_flux` returns NaN for non-finite shape / fully-masked sources [4]. Propagate a full error map (read noise + Poisson via gain + background RMS) into every flux [4].

### 3.9 Astrometry (WCS solve)
- **Algorithm:** bounded plate solving (astrometry.net quad-hashing) from the extracted, flux-sorted source list, then SIP-aware pixel→world, validated against Gaia [5].
- **Library/function:** local `solve-field` preferred (offline); `astroquery.astrometry_net.AstrometryNet().solve_from_source_list(x, y, image_width, image_height, solve_timeout=120, scale_units='arcsecperpix', scale_lower=..., scale_upper=..., center_ra=..., center_dec=..., radius=..., crpix_center=True, tweak_order=2)` as network fallback [5]. Then `wcs = WCS(header)`; `sky = wcs.pixel_to_world(x, y)` or `wcs.all_pix2world(x, y, 0)` (SIP-aware) [5].
- **Key params/defaults:** source list **must be flux-descending sorted**; `image_width/height` must match the actual (trimmed) frame [5]. **Always constrain the solver** with scale bounds + `center_ra/dec + radius` from the header pointing — turns multi-minute blind solves into seconds and prevents false solves [5].
- **Order:** after extraction; branch on the three outcomes — `isinstance(result, fits.Header)` (success) vs empty dict `{}` (failure) vs `TimeoutError` (capture `submission_id` and re-poll `monitor_submission`) [5].
- **Pitfalls:** doing `WCS({})` on a failed solve raises confusing errors — branch on return type first [5]. Using `wcs.wcs_pix2world` (core WCS only) drops SIP distortion; use `all_pix2world`/`pixel_to_world`. Applying SIP twice double-corrects [5]. Astrometry.net can return a confident **wrong** field when unconstrained — **always run a Gaia residual-RMS sanity gate** (require sub-arcsec, ideally ≪ pixel scale) and store residual RMS + N_matched as per-frame QA [5].

### 3.10 Photometry (zero-point / extinction / color calibration)
- **Algorithm:** instrumental mags → per-star zero point → sigma-clipped ZP; optional airmass/color least-squares [5].
- **Library/function:** flux via `photutils` `aperture_photometry`/`ApertureStats` on **background-subtracted** data; ZP via `astropy.stats.sigma_clipped_stats`/`SigmaClip`. Reference: `astroquery.gaia` (Gaia DR3, `ROW_LIMIT=-1`, `launch_job_async` ADQL), `gaiaxpy`/GSPC synthetic UBVRI/ugriz, or ATLAS Refcat2 (native griz) / PS1 DR2 / SDSS / APASS / 2MASS via `astroquery.mast`/`sdss`/`vizier` [5].
- **Key params/defaults:** `m_inst = −2.5·log10(counts/t_exp)`; per-star `ZP_i = m_ref_i − m_inst_i`; frame ZP = sigma-clipped median, scatter = calibration uncertainty [5]. Full form: `m_std = m_inst + ZP − k·X + c·(color)`; single frame at ~constant airmass folds `k·X` into ZP [5]. Default extinction priors (good high-altitude site, mag/airmass): U~0.55, B~0.25, V~0.15, R~0.09, I~0.06; ship Devasthal/Manora defaults from Kumar et al. 2022 [1][5].
- **Calibration-star cuts:** unsaturated, high-SNR (not faintest), isolated, `ruwe<1.4`, valid BP/RP, moderate color; **propagate Gaia proper motion from epoch 2016.0** with `SkyCoord(...).apply_space_motion(new_obstime=obs_time)` before matching [5]. Cross-match: `idx, d2d, d3d = src.match_to_catalog_sky(cat)`, keep `d2d < tol` where `tol ≈ WCS_RMS + fraction of seeing FWHM` (~1–2″) [5].
- **Order:** after WCS solve + Gaia validation; query reference for the solved footprint → PM-propagate → match → ZP → `m_cal = m_inst + ZP` for every source [5].
- **Pitfalls:** treating Gaia G as standard V/g carries large color-dependent error — use a color term or GSPC/GaiaXPy synthetic bands / Refcat2 native bands [5]. Contaminating ZP with saturated/blended/variable/CR-hit stars skews even a median [5]. Not background-subtracting before `aperture_photometry`/`ApertureStats` inflates fluxes [5]. Catalog coverage gaps: PS1 DR2 dec > −30°, APASS shallow (~10–17), 2MASS NIR-only — pick by hemisphere/band/depth [5].

### 3.11 Candidate / novelty ranking (discovery)
*(Full strategy in §5; stage summary here.)*
- **Algorithm:** difference imaging (ZOGY default, HOTPANTS fallback) → threshold `|S_corr|≥5` → real-bogus CNN/RF filter → catalog cross-match novelty flag → moving-object linking + variability across epochs → composite rank [6].
- **Library/function:** `ZOGY` (pmvreeswijk) / `HOTPANTS`; `photutils` detection on the difference image; `braai`-style CNN or `sklearn.ensemble.RandomForestClassifier`; `astroquery` (gaia/mast/sdss/vizier/mpc); `THOR` or MOPS-style linking; `astropy.timeseries.LombScargle` + variability indices [6].
- **Order:** after photometric calibration; writes candidate rows into the per-position DIAObject store [6].
- **Pitfalls:** detailed in §5.

### 3.12 QA + report
- **Algorithm:** provenance-logged, metrics-gated reproducible reporting (ILMT/pyTANSPEC/BANZAI approach) [1][2].
- **Library/function:** FITS `HISTORY` + custom step keywords (bias done, flat done, CR-cleaned, WCS solved) auto-written by ccdproc (do **not** pass `add_keyword=None`) [2]; multi-extension FITS output (SCI, ERR/VAR, BPM/DQ, CAT) like BANZAI/DRAGONS [2].
- **Key metrics/defaults:** per-frame — WCS residual RMS, N_matched, ZP, ZP scatter, N_calib_stars, extinction k, airmass, catalog + version, match tolerance, FWHM, ellipticity, sky brightness [5]. Master-frame QA gate (MAD comparison to previous good master) before promotion [2].
- **Order:** final; also the automatic verification gate — run `photutils` detection + aperture photometry before/after the CR stage and assert real-source flux/FWHM/centroid stability within tolerance; alert if star cores were flagged or flux changed beyond noise [3].
- **Pitfalls:** losing the `CCDData.mask` via raw-numpy ops leaks bad pixels into stacks/catalogs — keep everything in `CCDData`/NDData so data+uncertainty+mask stay coupled [2]. Not stamping every parameter into headers breaks reproducibility given cross-library default differences [4].

---

## 4. Dependency List (pip) with Windows Notes

**Core (all pure-Python wheels, install cleanly on Windows):**
```
astropy            # FITS I/O, WCS+SIP, units, coordinates, time, stats
ccdproc            # CCD detrending, combine, ccd_process, cosmicray_lacosmic, ccdmask
photutils          # Background2D, segmentation, SourceCatalog, DAO/IRAFStarFinder, aperture/PSF phot  (PIN — v3.0 renamed args)
astroscrappy       # L.A.Cosmic CR engine (Cython wheel on PyPI)
sep                # fast SExtractor-identical background/extraction  (use sep-pypi wheel; native byte-order arrays)
astroquery         # Gaia DR3, PS1(MAST), SDSS, VizieR(2MASS/APASS/USNO/Refcat2), astrometry_net, MPC
reproject          # pure-Python WCS resampling/alignment (SWarp fallback)
numpy scipy pandas matplotlib
```
**Optional discovery / advanced (pure-Python, fine on Windows):**
```
gaiaxpy            # Gaia BP/RP -> standardised UBVRI/ugriz  (patch >=1.2.4 for PS1 y-band fix)
scikit-learn       # Random-Forest real-bogus, classical ML
scikit-image       # Hough/Radon streak detection, wavelets
bm3d               # optional VST+BM3D display denoise (never on science array)
```

**Windows-hard dependencies — flag and provide fallback:**

| Dependency | Windows status | Fallback in ARISE |
|---|---|---|
| **astrometry.net `solve-field`** | No native Windows build; needs WSL2 or Docker | `astroquery.astrometry_net` **network** solve (nova) as fallback; bundle index files for local WSL solve for offline runs [5] |
| **SExtractor / PSFEx / SWarp / SCAMP (Astromatic C)** | No official Windows binaries; WSL/conda-forge | Pure-Python: `sep` (extraction), `photutils` (PSF/catalog), `reproject` (resampling). ARISE never *requires* these [1][4] |
| **HOTPANTS (C)** | Compile under WSL or use `conda`/prebuilt | `ZOGY` (Python) is the default subtraction backend; HOTPANTS is the optional robustness fallback [6] |
| **ZOGY (pmvreeswijk)** | Runs on Windows but its deps (solve-field, SExtractor, PSFEx, SWarp, pyfftw) are the hard part | Provide a reduced ZOGY path using `sep`+`photutils`+`reproject`; `pyfftw` has Windows wheels [6] |
| **deepCR / braai (TF/Keras/PyTorch)** | GPU stack heavy on Windows | Optional; default CR is astroscrappy, default real-bogus is `sklearn` RF; deep models are pluggable upgrades [3][6] |
| **THOR** | Python but heavy scientific deps | MOPS-style kd-tree linking (`scipy.spatial`) as the built-in linker; THOR optional for sparse cadence [6] |

**Recommendation:** ship a `conda`/`mamba` environment file (conda-forge carries `astropy`, `ccdproc`, `photutils`, `astroscrappy`, `sep`, `astrometry`, `sextractor`, `swarp`, `scamp`, `hotpants` built for Windows/WSL) as the *first-class* install path, with a `pip` requirements file for the pure-Python core, and treat all C binaries as **auto-detected optional accelerators** [1].

---

## 5. Discovery Strategy — How ARISE Surfaces NEW Sources

ARISE surfaces novelty through **five converging signals**, aggregated per sky position (a DIAObject analog) and combined into one composite rank. This mirrors ZTF alert packets (which ship `rb`/`drb`/`sgscore`/`distpsnr`/`isdiffpos`/`ndethist`/`ssdistnr`) and Rubin alert production (5σ DIASource → DIAObject → broker) [6].

**(A) Difference imaging → transient detection.**
Register images to sub-pixel accuracy (SCAMP/astrometry.net against Gaia), match PSFs (PSFEx), and match photometric zero-points and background **before** subtraction [6]. Default backend **ZOGY** (Zackay, Ofek & Gal-Yam 2016): closed-form Fourier subtraction producing a white-noise difference image `D`, matched-filter score `S`, and per-pixel significance `S_corr` (in σ units) that is the provably optimal point-source detection statistic, plus optimal PSF flux `F_psf ± F_psferr` [6]. **Detect at `|S_corr| ≥ 5`.** The `S_corr` denominator **must include the astrometric-noise term `V_ast`** or bright stars produce dipole false positives that flood the candidate list [6]. **HOTPANTS** (Alard–Lupton kernel matching) is the pluggable fallback for undersampled / poor-PSF data where ZOGY's white-Gaussian/accurate-PSF assumptions break [6].

**(B) Real-bogus ML ranking.**
Feed the classifier the standard **science + reference + difference cutout triplet** (~63×63 px) and keep `isdiffpos` (positive/negative subtraction) [6]. Default engine: `sklearn` **Random Forest** on ~20–38 hand-engineered features (autoScan/PS1 lineage — trains well on few labels, interpretable) [6]. Optional upgrade: a `braai`-style **CNN** (0.7% FN at threshold 0.5 vs 10.7% for RF on ZTF) [6]. **Build and version-control a labeled training set of ARISE's own instrument stamps** — ZTF/PS1-trained weights do not transfer without fine-tuning because artifacts (ghosts, bad columns, crosstalk) are telescope-specific [6]. The real-bogus threshold is a **tunable config**, not hard-coded; store the score *and* raw features [6].

**(C) Catalog cross-match novelty flag (all sources, not just diffs).**
Cone-search each detection against **Gaia DR3, PS1 DR2, SDSS, and MPC ephemerides** via `astroquery` at **1–2″** (tuned to seeing/pixel scale), using **proper-motion-propagated** Gaia positions [5][6]. Classify as a known stellar source if the Gaia match has parallax significance > ~8, or |PM| > 3×error, or is flagged variable; store PS1 `sgscore` (star/galaxy, →1 = star) and `distpsnr` per candidate [6]. A detection with **no catalog match in any survey AND not a known moving object (MPChecker)** is the **highest-value novelty candidate** — it goes to the top of the ranked queue [5][6]. Tolerance discipline: too small → PM/parallax-shifted stars flagged as false "novel"; too large → real transients near host galaxies over-rejected [6].

**(D) Moving-object linking (asteroids / NEOs).**
Fast movers **trail within one exposure** and are systematically missed by point-source detection — run a dedicated **streak/Radon path** (`skimage.transform.probabilistic_hough_line` / Radon; optional U-Net+Hough à la ASTA) with **multi-exposure confirmation** to suppress the high single-exposure false-positive rate [6]. For point-like movers, link across epochs: built-in **MOPS-style** `findTracklets`→`linkTracklets` on kd-trees (`scipy.spatial`) for regular cadence, optional **THOR** (test-orbit + generalized-Hough, tracklet-less) for sparse/irregular cadence [6]. Require **≥3 detections forming a self-consistent tracklet/track validated by orbit determination**, then run **MPChecker/NEOChecker + `digest2`** (D2 score 0–100) to separate known objects from genuine new NEO candidates (MPC posts to NEOCP at **D2 ≥ 65**) [6].

**(E) Variability across epochs.**
Over each DIAObject's multi-epoch light curve compute **robust** indices — **IQR** of magnitudes (timescale-independent, outlier-resistant) + **1/η** (von Neumann, variance/mean-square-successive-difference), plus Stetson J/K and χ²-vs-constant — and a **Lomb–Scargle** periodogram (`astropy.timeseries.LombScargle`) for periodicity [6]. Use robust indices (Sokolovsky+2017 recommends IQR + 1/η) because underestimated error bars break naive χ² and flag false variables [6].

**Composite candidate rank** (surfaced top-first): a multi-factor score, **not** a single cut — combining `S_corr` significance, real-bogus probability, **catalog-novelty (no Gaia/PS1/SDSS/MPC match)**, temporal behavior (new vs known, moving vs stationary, variable vs constant), and `ndethist`/light-curve consistency [6]. The canonical "this is genuinely new" signal is **high `S_corr` + high real-bogus + repeated detection + no catalog match + not a known mover** [6].

---

## 6. What Makes a Scientist Say "This Is Real Work"

1. **Correct, propagated uncertainty and DQ masks end-to-end.** Every output is multi-extension FITS (SCI + ERR/VAR + BPM/DQ + CAT), uncertainty built via `create_deviation` on raw ADU (right Poisson term) and propagated through every stage; CR/interpolated pixels are *flagged, not trusted* [2][3]. This alone separates ARISE from a demo script.

2. **Per-instrument, per-frame gain/read-noise/pixel-scale handling.** Reading selectable gain/speed from the header (DOT 4Kx4K 5 gains × 3 speeds; DFOT 2 speeds) and driving all radii/thresholds from the actual pixel scale — not hard-coded constants — is exactly what a DOT/DFOT observer will check first [1].

3. **Reproducible provenance + automated master-frame QA.** Full FITS `HISTORY`/step keywords, every parameter stamped, and a BANZAI-style MAD gate that rejects a warm-camera master before it corrupts a night's data [2].

4. **Astrometric validation gate, not blind trust.** A Gaia residual-RMS sanity check (sub-arcsec, ≪ pixel scale) that catches astrometry.net's confident-but-wrong solutions, with stored residual RMS + N_matched per frame [5].

5. **Photometric calibration tied to a real system with color terms.** ZP + extinction + color-term calibration against Gaia DR3 **synthetic photometry (GSPC/GaiaXPy)** or ATLAS Refcat2 native bands — not raw Gaia G — reproducing the DOT 4Kx4K Kumar et al. 2022 workflow with Devasthal/Manora default extinction priors [1][5].

6. **Detector-physics-correct branching.** Genuinely different CCD vs NIR-array (SUTR/reference-pixel/non-linearity) and stare vs ILMT-TDI drift-scan paths — proving the author understands that one linear pipeline cannot serve DOT 4Kx4K, TANSPEC, and ILMT [1].

7. **A discovery layer that mirrors ZTF/Rubin alert production.** ZOGY `S_corr≥5` with the astrometric-noise term, real-bogus scoring on science/ref/diff triplets, DIAObject aggregation, MPChecker/digest2 vetting for movers, and robust variability indices — the actual state-of-the-art architecture, at facility scale [6].

8. **Science-safe denoising discipline.** Refusing to blur the photometric array, keeping any denoised product a separate labeled artifact, and using flux-preserving starlet/à-trous or exact-inverse Anscombe only where valid — a subtlety that signals the author knows how correlated noise corrupts error bars [3].

---

## 7. References

1. ARIES instruments & pipelines — Pandey et al. 2018 (DOT 4Kx4K first-light imager), arXiv:1711.05422 · Kumar et al. 2022 (4Kx4K photometric calibration), arXiv:2111.13018 · Baug et al. 2018 (TIRCAM2), arXiv:1802.05008 · Sharma et al. 2022 (TANSPEC), arXiv:2207.07878 · Ghosh et al. 2023 (pyTANSPEC), arXiv:2212.04815 · RedPipe (Singh 2021, ASCL 2106.024), https://github.com/sPaMFouR/RedPipe · ILMT automated pipeline, arXiv:2311.04713 · ILMT overview, arXiv:2502.00564 · DOT capabilities, arXiv:1908.02531.
2. CCD calibration — ccdproc reduction toolbox, https://ccdproc.readthedocs.io/en/latest/reduction_toolbox.html · ccdproc `combine` API, https://ccdproc.readthedocs.io/en/latest/api/ccdproc.combine.html · Astropy CCD Reduction & Photometry Guide, https://www.astropy.org/ccd-reduction-and-photometry-guide/ · McCully et al. 2018 (BANZAI), arXiv:1811.04163 · Labrie et al. 2023 (DRAGONS), arXiv:2310.03048 · Schirmer 2013 (THELI), ApJS 209, 21 · Fringe defringing, arXiv:1201.2336.
3. Cosmic rays & denoising — van Dokkum 2001 (L.A.Cosmic), arXiv:astro-ph/0108003 · astroscrappy `detect_cosmics`, https://astroscrappy.readthedocs.io/ · ccdproc `cosmicray_lacosmic`, https://ccdproc.readthedocs.io/en/latest/api/ccdproc.cosmicray_lacosmic.html · Zhang & Bloom 2020 (deepCR), arXiv:1907.09500 · Chen et al. 2024 (label-free deepCR), doi:10.3847/1538-4357/ad1602 · Popowicz et al. 2016, arXiv:1608.02452 · `astropy.convolution.interpolate_replace_nans`, https://docs.astropy.org/en/stable/api/astropy.convolution.interpolate_replace_nans.html.
4. Background & source extraction — Bertin & Arnouts 1996 (SExtractor), A&AS 117, 393 · SExtractor 2.28 docs, https://astromatic.github.io/sextractor/ · photutils Background, https://photutils.readthedocs.io/en/stable/user_guide/background.html · photutils Segmentation, https://photutils.readthedocs.io/en/stable/user_guide/segmentation.html · SEP docs, https://sep.readthedocs.io/ · SEP source (sep.pyx), https://github.com/sep-developers/sep · LSST/Rubin Scarlet Lite deblending, https://pipelines.lsst.io/modules/lsst.meas.extensions.scarlet/overview.html.
5. Astrometric & photometric calibration — astroquery astrometry_net, https://astroquery.readthedocs.io/en/latest/astrometry_net/astrometry_net.html · astroquery Gaia, https://astroquery.readthedocs.io/en/latest/gaia/gaia.html · Astropy matching, https://docs.astropy.org/en/stable/coordinates/matchsep.html · astropy.wcs, https://docs.astropy.org/en/latest/wcs/index.html · photutils aperture, https://photutils.readthedocs.io/en/stable/user_guide/aperture.html · Tonry et al. 2018 (ATLAS Refcat2), arXiv:1809.09157 · Gaia GSPC, https://gaia.aip.de/metadata/gaiadr3/synthetic_photometry_gspc/ · Dhillon PHY217 photometric calibration, https://vikdhillon.staff.shef.ac.uk/teaching/phy217/instruments/phy217_inst_photcal.html.
6. Discovery / alerts — Zackay, Ofek & Gal-Yam 2016 (ZOGY), arXiv:1601.02655 · pmvreeswijk/ZOGY, https://github.com/pmvreeswijk/ZOGY · HOTPANTS (Becker 2015), ASCL 1504.004 · Duev et al. 2019 (braai), https://github.com/dmitryduev/braai · Goldstein et al. 2015 (autoScan), https://portal.nersc.gov/project/dessn/autoscan/ · Denneau et al. 2013 (MOPS) · Moeyens et al. 2021 (THOR), arXiv:2105.01056 · digest2, https://github.com/soniakeys/digest2 · MPC MPChecker/NEOChecker, https://www.minorplanetcenter.org/iau/MPCStatus.html · Sokolovsky et al. 2017 (variability indices), MNRAS 464, 274 · Patterson et al. 2019 (ZTF alerts), arXiv:1902.02227 · Rubin alerts & brokers, https://rubinobservatory.org/for-scientists/data-products/alerts-and-brokers.
