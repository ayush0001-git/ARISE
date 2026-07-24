"""FITS input/output and header interpretation for ARISE.

The rest of the pipeline never touches raw FITS keywords directly -- it goes
through :func:`read_frame` / :func:`resolve` / :func:`classify_frame`, which
use the instrument's :class:`~arise.config.HeaderMap` so one code path handles
frames from many instruments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits

from .config import HeaderMap, Instrument
from . import __version__


# --------------------------------------------------------------------------- #
# header value resolution
# --------------------------------------------------------------------------- #
def resolve(header: fits.Header, candidates: Iterable[str], default: Any = None) -> Any:
    """Return the value of the first present keyword in ``candidates``."""
    for key in candidates:
        if key in header:
            val = header[key]
            if val not in ("", None):
                return val
    return default


def classify_frame(header: fits.Header, hmap: HeaderMap) -> str:
    """Classify a frame as bias/dark/flat/light/unknown.

    Uses IMAGETYP-like keywords first; falls back to exposure-time heuristics
    (a zero-second exposure is a bias) so unlabelled frames still sort sensibly.
    """
    raw = resolve(header, hmap.imagetyp, "")
    tag = str(raw).strip().lower()

    def _match(values: tuple[str, ...]) -> bool:
        return any(v in tag for v in values)

    if tag:
        if _match(hmap.bias_values):
            return "bias"
        if _match(hmap.dark_values):
            return "dark"
        if _match(hmap.flat_values):
            return "flat"
        if _match(hmap.light_values):
            return "light"

    # heuristic fallback from exposure time
    exp = resolve(header, hmap.exptime, None)
    try:
        exp = float(exp)
        if exp == 0.0:
            return "bias"
    except (TypeError, ValueError):
        pass
    return "unknown"


@dataclass
class FrameMeta:
    """Instrument-independent view of a frame's metadata."""

    path: Path
    ftype: str            # bias | dark | flat | light | unknown
    exptime: float = 0.0
    filt: str = "NONE"
    obj: str = ""
    dateobs: str = ""
    ra: float | None = None
    dec: float | None = None
    gain: float = 1.0
    read_noise: float = 5.0
    airmass: float | None = None
    naxis1: int = 0
    naxis2: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        """Grouping key for master-frame construction (filter, rounded exptime)."""
        return (str(self.filt), f"{self.exptime:.1f}")


def _coerce_coord(value: Any, is_ra: bool = False) -> float | None:
    """Best-effort convert an RA/DEC header value to decimal degrees.

    Numeric values and plain decimal strings are already degrees (CRVAL-style).
    Sexagesimal strings follow FITS keyword convention: hours for RA
    ("HH:MM:SS", converted x15 to degrees when ``is_ra``), degrees for DEC.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    # sexagesimal "HH:MM:SS" / "DD:MM:SS" or space separated
    for sep in (":", " "):
        if sep in s:
            try:
                parts = [float(p) for p in s.replace("  ", " ").split(sep)]
                sign = -1.0 if parts[0] < 0 or s.strip().startswith("-") else 1.0
                deg = abs(parts[0]) + (parts[1] if len(parts) > 1 else 0) / 60.0 + (
                    parts[2] if len(parts) > 2 else 0
                ) / 3600.0
                # sexagesimal RA is conventionally hours -> degrees; a leading
                # field >= 24 can only be degrees already, so leave it alone
                if is_ra and abs(parts[0]) < 24.0:
                    deg *= 15.0
                return sign * deg
            except (ValueError, IndexError):
                return None
    try:
        return float(s)
    except ValueError:
        return None


def read_meta(path: str | Path, inst: Instrument) -> FrameMeta:
    """Read only the header and build a :class:`FrameMeta` (cheap; no pixels)."""
    path = Path(path)
    hmap = inst.header
    with fits.open(path, memmap=False) as hdul:
        hdr = _primary_header_with_data(hdul)
        ftype = classify_frame(hdr, hmap)
        exptime = _as_float(resolve(hdr, hmap.exptime, 0.0), 0.0)
        gain = _as_float(resolve(hdr, hmap.gain, inst.gain), inst.gain)
        rdn = _as_float(resolve(hdr, hmap.rdnoise, inst.read_noise), inst.read_noise)
        ra = _coerce_coord(resolve(hdr, hmap.ra, None), is_ra=True)
        dec = _coerce_coord(resolve(hdr, hmap.dec, None))
        airmass = resolve(hdr, hmap.airmass, None)
        return FrameMeta(
            path=path,
            ftype=ftype,
            exptime=exptime,
            filt=str(resolve(hdr, hmap.filt, "NONE")).strip() or "NONE",
            obj=str(resolve(hdr, hmap.obj, "")).strip(),
            dateobs=str(resolve(hdr, hmap.dateobs, "")).strip(),
            ra=ra,
            dec=dec,
            gain=gain,
            read_noise=rdn,
            airmass=_as_float(airmass, None) if airmass is not None else None,
            naxis1=int(hdr.get("NAXIS1", 0)),
            naxis2=int(hdr.get("NAXIS2", 0)),
        )


def read_frame(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Return (image as float32, primary header with data)."""
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        hdr = _primary_header_with_data(hdul)
        idx = _first_image_index(hdul)
        data = np.asarray(hdul[idx].data, dtype=np.float32)
    return data, hdr


def write_frame(
    path: str | Path,
    data: np.ndarray,
    header: fits.Header | None = None,
    history: Iterable[str] | None = None,
    extra_cards: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> Path:
    """Write ``data`` to a FITS file, stamping ARISE provenance into the header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = header.copy() if header is not None else fits.Header()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hdr["ARISEVER"] = (__version__, "ARISE pipeline version")
    hdr["ARISEUTC"] = (stamp, "UTC of this ARISE processing step")
    if extra_cards:
        for k, v in extra_cards.items():
            hdr[k] = v
    for line in history or []:
        hdr.add_history(f"ARISE: {line}")
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=hdr).writeto(
        path, overwrite=overwrite
    )
    return path


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _first_image_index(hdul: fits.HDUList) -> int:
    for i, hdu in enumerate(hdul):
        if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) >= 2:
            return i
    return 0


def _primary_header_with_data(hdul: fits.HDUList) -> fits.Header:
    """Return a merged header: primary + the image HDU that carries the data."""
    idx = _first_image_index(hdul)
    hdr = hdul[0].header.copy()
    if idx != 0:
        for card in hdul[idx].header.cards:
            if card.keyword not in ("XTENSION", "PCOUNT", "GCOUNT") and card.keyword:
                hdr[card.keyword] = (card.value, card.comment)
    return hdr


def _as_float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
