"""Configuration model for ARISE.

Two layers:

* :class:`Instrument` -- fixed physical properties of a telescope+detector
  (gain, read noise, pixel scale, saturation ...). A small registry of ARIES
  instruments ships built in; users can add their own in YAML.
* :class:`PipelineConfig` -- per-run choices: which stages to run and their
  parameters, plus paths. Loads from / dumps to YAML so every run is
  reproducible from a single file.

Numeric instrument values marked ``# APPROX`` are sensible published-order
defaults and are overridden automatically from FITS headers when the relevant
keyword is present, so a slightly wrong default never corrupts a reduction.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .logs import get_logger

log = get_logger("config")


# --------------------------------------------------------------------------- #
# Instruments
# --------------------------------------------------------------------------- #
@dataclass
class HeaderMap:
    """Maps ARISE's logical header fields to this instrument's FITS keywords.

    Multiple candidate keywords may be given per field; the first present in a
    header wins. This is what lets one pipeline read frames from many
    instruments without hand-editing headers.
    """

    imagetyp: tuple[str, ...] = ("IMAGETYP", "IMGTYPE", "OBSTYPE", "IMTYPE")
    exptime: tuple[str, ...] = ("EXPTIME", "EXPOSURE", "ITIME", "TELAPSE")
    filt: tuple[str, ...] = ("FILTER", "FILTER1", "FILTNAM", "INSFILTE")
    obj: tuple[str, ...] = ("OBJECT", "TARGET", "OBJNAME")
    dateobs: tuple[str, ...] = ("DATE-OBS", "DATE_OBS", "MJD-OBS", "JD")
    ra: tuple[str, ...] = ("RA", "OBJCTRA", "CRVAL1", "TELRA")
    dec: tuple[str, ...] = ("DEC", "OBJCTDEC", "CRVAL2", "TELDEC")
    gain: tuple[str, ...] = ("GAIN", "EGAIN", "CCDGAIN")
    rdnoise: tuple[str, ...] = ("RDNOISE", "READNOIS", "RON", "ENOISE")
    airmass: tuple[str, ...] = ("AIRMASS", "SECZ")

    # IMAGETYP string values that identify each frame class (matched case- and
    # whitespace-insensitively, as a substring).
    bias_values: tuple[str, ...] = ("bias", "zero")
    dark_values: tuple[str, ...] = ("dark",)
    flat_values: tuple[str, ...] = ("flat", "dome flat", "sky flat", "twilight")
    light_values: tuple[str, ...] = ("light", "object", "science", "target", "sci")


@dataclass
class Instrument:
    """Physical description of a telescope + detector."""

    name: str
    telescope: str = ""
    gain: float = 1.0            # e-/ADU
    read_noise: float = 5.0      # e- RMS
    pixel_scale: float = 1.0     # arcsec / pixel
    saturation: float = 60000.0  # ADU at which pixels saturate / go non-linear
    dark_current: float = 0.0    # e- / pixel / second (informational)
    filters: tuple[str, ...] = ("U", "B", "V", "R", "I")
    fov_arcmin: float | None = None    # field of view (square side), arcmin
    detector: str = ""
    header: HeaderMap = field(default_factory=HeaderMap)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Instrument":
        d = dict(d)
        hdr = d.pop("header", None)
        inst = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if isinstance(hdr, dict):
            # coerce lists (from YAML) to tuples for the HeaderMap
            hdr = {k: tuple(v) if isinstance(v, list) else v for k, v in hdr.items()}
            inst.header = HeaderMap(**{k: v for k, v in hdr.items() if k in HeaderMap.__dataclass_fields__})
        return inst


# --- ARIES instrument registry -------------------------------------------- #
# Values are published-order-of-magnitude defaults; confirm against your own
# frames' headers. ARISE always prefers header values when present.
INSTRUMENTS: dict[str, Instrument] = {
    "generic": Instrument(
        name="generic",
        telescope="unknown",
        gain=1.5, read_noise=6.0, pixel_scale=1.0, saturation=60000.0,
        detector="unspecified CCD",
        notes="Fallback profile; header keywords override every field.",
    ),
    # 3.6 m Devasthal Optical Telescope + 4Kx4K CCD IMAGER
    "dot_4kx4k": Instrument(
        name="dot_4kx4k",
        telescope="3.6m Devasthal Optical Telescope (DOT)",
        gain=2.0,            # selectable 1/2/3/5/10 e-/ADU; header GAIN overrides
        read_noise=4.0,      # varies with gain x readout speed (100k/500k/1M Hz)
        pixel_scale=0.095,   # arcsec/pix (unbinned), 15 um pixels
        saturation=65000.0,
        dark_current=0.0,
        filters=("U", "B", "V", "R", "I", "u", "g", "r", "i", "z"),
        fov_arcmin=6.5,      # 4096x4096 -> ~6.5' x 6.5'
        detector="STA4150 4096x4096 CCD (4Kx4K IMAGER)",
        notes="ARIES flagship 3.6m; deep imaging, transients, clusters, AGN. "
              "Gain/readnoise are header-driven (5 gains x 3 speeds).",
    ),
    # 3.6 m DOT + ADFOSC faint-object spectrograph & camera (imaging mode)
    "adfosc": Instrument(
        name="adfosc",
        telescope="3.6m Devasthal Optical Telescope (DOT) + ADFOSC",
        gain=1.0, read_noise=6.0, pixel_scale=0.20, saturation=65000.0,
        filters=("U", "B", "V", "R", "I", "u", "g", "r", "i", "z"),
        fov_arcmin=13.6,
        detector="E2V CCD231-84 4096x4096 CCD (-120 C)",
        notes="Wide-field imaging + low-res spectroscopy on the 3.6m.",
    ),
    # 1.3 m Devasthal Fast Optical Telescope + 2Kx2K CCD
    "dfot_2kx2k": Instrument(
        name="dfot_2kx2k",
        telescope="1.3m Devasthal Fast Optical Telescope (DFOT)",
        gain=2.0,            # APPROX
        read_noise=7.0,      # APPROX
        pixel_scale=0.535,   # APPROX arcsec/pix
        saturation=55000.0,
        filters=("U", "B", "V", "R", "I", "H-alpha"),
        fov_arcmin=18.0,
        detector="Andor E2V 2048x2048 CCD (-80 C)",
        notes="Fast time-domain work: variable stars, exoplanet transits, follow-up.",
    ),
    # 1.04 m Sampurnanand Telescope + 4Kx4K / 2Kx2K CCD
    "st_1m": Instrument(
        name="st_1m",
        telescope="1.04m Sampurnanand Telescope (ST), Manora Peak",
        gain=10.0,           # APPROX (older instrument)
        read_noise=5.0,      # APPROX
        pixel_scale=0.37,    # APPROX arcsec/pix (4Kx4K)
        saturation=55000.0,
        filters=("U", "B", "V", "R", "I"),
        fov_arcmin=13.0,     # APPROX
        detector="4096x4096 CCD",
        notes="Legacy ARIES workhorse for photometry of variables & clusters.",
    ),
}


def get_instrument(name: str) -> Instrument:
    key = (name or "generic").strip().lower()
    if key not in INSTRUMENTS:
        raise KeyError(
            f"Unknown instrument '{name}'. Known: {sorted(INSTRUMENTS)}. "
            f"Add a custom one under 'instrument:' in your config YAML."
        )
    return copy.deepcopy(INSTRUMENTS[key])


# --------------------------------------------------------------------------- #
# Pipeline configuration
# --------------------------------------------------------------------------- #
@dataclass
class CosmicRayConfig:
    enabled: bool = True
    method: str = "lacosmic"       # lacosmic | deepcr | none
    sigclip: float = 6.0           # ground-based default (4.5 clips star cores)
    sigfrac: float = 0.3
    objlim: float = 5.0
    niter: int = 4


@dataclass
class DenoiseConfig:
    enabled: bool = False          # off by default: aggressive denoise harms photometry
    method: str = "none"           # none | bilateral | tv | wavelet
    strength: float = 1.0


@dataclass
class BackgroundConfig:
    box_size: int = 64             # background mesh cell (pixels)
    filter_size: int = 3           # median filter over the mesh
    estimator: str = "sextractor"  # sextractor | median | mmm
    mask_sources_iters: int = 2    # iterations of detect->mask->re-estimate


@dataclass
class DetectConfig:
    nsigma: float = 3.0            # detection threshold in units of background RMS
    npixels: int = 5              # min connected pixels above threshold
    deblend: bool = True
    deblend_nlevels: int = 32
    deblend_contrast: float = 0.005
    kernel_fwhm: float = 3.0       # matched-filter smoothing FWHM (pixels)
    aperture_scale: float = 1.5    # fixed-aperture radius = scale * median FWHM
                                   # (~SNR-optimal for point sources; limits crowding)


@dataclass
class AstrometryConfig:
    enabled: bool = True
    use_header_wcs: bool = True    # trust an existing WCS if the header has one
    solver: str = "auto"           # auto | astrometry_net | none
    scale_low: float | None = None    # arcsec/pix bounds for the solver (else from instrument)
    scale_high: float | None = None


@dataclass
class PhotometryConfig:
    enabled: bool = True
    ref_catalog: str = "gaia"      # gaia | panstarrs | skymapper | none
    match_radius_arcsec: float = 2.0
    min_ref_stars: int = 8
    aperture_scale: float = 2.5    # aperture radius = scale * median FWHM


@dataclass
class DiscoveryConfig:
    enabled: bool = True
    catalog_match_radius_arcsec: float = 2.5
    flag_unmatched: bool = True            # sources absent from reference catalogs
    moving_object_link: bool = True        # link detections across a sequence
    max_motion_arcsec_per_min: float = 60.0
    min_tracklet_points: int = 3
    variability: bool = True               # flux changes across epochs
    difference_imaging: bool = False       # requires a reference/template frame
    min_snr: float = 5.0
    top_n: int = 50                        # how many ranked candidates to report
    dossiers: bool = True                  # per-candidate science dossiers
    dossiers_online: bool = True           # allow live SkyBoT/JPL/SIMBAD checks


@dataclass
class Paths:
    raw: str = "data/raw"
    master: str = "data/master"
    reduced: str = "data/reduced"
    catalogs: str = "data/catalogs"
    reports: str = "data/reports"


@dataclass
class PipelineConfig:
    instrument: str = "generic"
    # allow a fully inline custom instrument dict to override the named preset
    instrument_overrides: dict[str, Any] = field(default_factory=dict)

    paths: Paths = field(default_factory=Paths)
    log_level: str = "INFO"
    n_jobs: int = 1                # reserved: parallel frame workers not yet implemented (always serial)

    cosmic_ray: CosmicRayConfig = field(default_factory=CosmicRayConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    astrometry: AstrometryConfig = field(default_factory=AstrometryConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    # ---- convenience ------------------------------------------------------ #
    def resolve_instrument(self) -> Instrument:
        inst = get_instrument(self.instrument)
        if self.instrument_overrides:
            overrides = dict(self.instrument_overrides)  # copy: never mutate the config
            hdr_override = overrides.pop("header", None)
            base = inst.to_dict()
            base.update(overrides)
            if isinstance(hdr_override, dict):
                # per-field merge over the preset's header map (to_dict gives plain dicts;
                # Instrument.from_dict coerces YAML lists to tuples and drops unknown fields)
                base["header"].update(hdr_override)
            inst = Instrument.from_dict(base)
        return inst

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(_dataclass_to_plain(self), fh, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        data = dict(data or {})
        cfg = cls()
        _apply_nested(cfg, data)
        if isinstance(cfg.n_jobs, int) and cfg.n_jobs > 1:
            log.warning(
                "n_jobs=%d requested but parallel processing is not yet implemented; "
                "running serially.", cfg.n_jobs,
            )
        return cfg


# --------------------------------------------------------------------------- #
# (de)serialisation helpers
# --------------------------------------------------------------------------- #
_SUBCONFIGS = {
    "paths": Paths,
    "cosmic_ray": CosmicRayConfig,
    "denoise": DenoiseConfig,
    "background": BackgroundConfig,
    "detect": DetectConfig,
    "astrometry": AstrometryConfig,
    "photometry": PhotometryConfig,
    "discovery": DiscoveryConfig,
}


def _apply_nested(cfg: PipelineConfig, data: dict[str, Any]) -> None:
    unknown_top: list[str] = []
    for key, value in data.items():
        if key in _SUBCONFIGS:
            if not isinstance(value, dict):
                log.warning(
                    "Config section '%s' should be a mapping of options, got %s; "
                    "section ignored (defaults kept).", key, type(value).__name__,
                )
                continue
            sub = getattr(cfg, key)
            fields = type(sub).__dataclass_fields__
            bad = [k for k in value if k not in fields]
            if bad:
                log.warning(
                    "Unknown key(s) in config section '%s' ignored (check for typos): %s. "
                    "Valid options: %s.", key, ", ".join(bad), ", ".join(fields),
                )
            for k, v in value.items():
                if k in fields:
                    setattr(sub, k, v)
        elif key in PipelineConfig.__dataclass_fields__:
            setattr(cfg, key, value)
        else:
            unknown_top.append(key)
    if unknown_top:
        log.warning(
            "Unknown top-level config key(s) ignored (check for typos): %s. "
            "Valid keys: %s.", ", ".join(unknown_top),
            ", ".join(PipelineConfig.__dataclass_fields__),
        )


def _dataclass_to_plain(obj: Any) -> Any:
    """asdict, but tuples become lists so PyYAML emits clean scalars."""
    d = asdict(obj)

    def clean(x: Any) -> Any:
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    return clean(d)
