"""Stage 1 -- ingest.

Scan a directory of raw FITS frames, classify each (bias/dark/flat/light) using
the instrument header map, and group them for master-frame construction.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Instrument
from ..fitsio import read_meta, FrameMeta
from ..logs import get_logger

log = get_logger("ingest")

_FITS_EXT = (".fits", ".fit", ".fts", ".fits.gz", ".fits.fz", ".fz")


@dataclass
class FrameSet:
    """All frames from one raw directory, grouped by type."""

    bias: list[FrameMeta] = field(default_factory=list)
    dark: list[FrameMeta] = field(default_factory=list)
    flat: list[FrameMeta] = field(default_factory=list)
    light: list[FrameMeta] = field(default_factory=list)
    unknown: list[FrameMeta] = field(default_factory=list)

    def flats_by_filter(self) -> dict[str, list[FrameMeta]]:
        out: dict[str, list[FrameMeta]] = defaultdict(list)
        for m in self.flat:
            out[m.filt].append(m)
        return dict(out)

    def darks_by_exptime(self) -> dict[float, list[FrameMeta]]:
        out: dict[float, list[FrameMeta]] = defaultdict(list)
        for m in self.dark:
            out[round(m.exptime, 1)].append(m)
        return dict(out)

    def lights_sorted(self) -> list[FrameMeta]:
        """Science frames ordered in time (by DATE-OBS then filename)."""
        return sorted(self.light, key=lambda m: (m.dateobs, m.path.name))

    def summary(self) -> str:
        return (f"{len(self.bias)} bias, {len(self.dark)} dark, "
                f"{len(self.flat)} flat, {len(self.light)} light, "
                f"{len(self.unknown)} unknown")


def _iter_fits(directory: Path):
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.name.lower().endswith(_FITS_EXT):
            yield p


def ingest(raw_dir: str | Path, inst: Instrument) -> FrameSet:
    """Read headers for every FITS file in ``raw_dir`` and classify them."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw directory not found: {raw_dir}")

    fs = FrameSet()
    n = 0
    for path in _iter_fits(raw_dir):
        try:
            meta = read_meta(path, inst)
        except Exception as exc:  # a corrupt frame shouldn't kill the run
            log.warning("Skipping unreadable frame %s (%s)", path.name, exc)
            continue
        getattr(fs, meta.ftype).append(meta)
        n += 1

    if fs.unknown:
        log.warning("%d frame(s) could not be classified; set IMAGETYP or check the "
                    "instrument profile.", len(fs.unknown))
    log.info("Ingested %d frames from %s -> %s", n, raw_dir, fs.summary())
    return fs
