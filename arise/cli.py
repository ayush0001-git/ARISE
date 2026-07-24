"""ARISE command-line interface.

    arise demo                 generate a synthetic night and run the full pipeline
    arise run --raw DIR        run the pipeline on a directory of FITS frames
    arise synth --out DIR      only generate a synthetic night
    arise init-config FILE     write a documented default config YAML
    arise instruments          list built-in instrument profiles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import PipelineConfig, INSTRUMENTS
from .logs import setup_logging, get_logger

log = get_logger("cli")


def _set_paths(cfg: PipelineConfig, base: str | Path) -> PipelineConfig:
    base = Path(base)
    cfg.paths.raw = str(base / "raw")
    cfg.paths.master = str(base / "master")
    cfg.paths.reduced = str(base / "reduced")
    cfg.paths.catalogs = str(base / "catalogs")
    cfg.paths.reports = str(base / "reports")
    return cfg


def _print_summary(result) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        _print_summary_plain(result)
        return
    console = Console()
    disc = result.discovery
    if disc is not None and len(disc.candidates):
        t = Table(title="Top discovery candidates", header_style="bold cyan")
        for c in ["kind", "rank_score", "ra", "dec", "n_det", "mean_snr", "notes"]:
            t.add_column(c)
        for _, r in disc.candidates.head(10).iterrows():
            t.add_row(str(r["kind"]), f"{r['rank_score']:.3f}", f"{r['ra']:.5f}",
                      f"{r['dec']:.5f}", str(r["n_det"]), f"{r['mean_snr']:.0f}",
                      str(r["notes"])[:60])
        console.print(t)
    if result.outputs.get("report"):
        console.print(f"\n[bold green]Report:[/] {result.outputs['report']}")


def _print_summary_plain(result) -> None:
    disc = result.discovery
    if disc is not None and len(disc.candidates):
        print("\nTop candidates:")
        print(disc.candidates.head(10).to_string(index=False))
    if result.outputs.get("report"):
        print(f"\nReport: {result.outputs['report']}")


# --------------------------------------------------------------------------- #
def cmd_demo(args) -> int:
    from .synth import generate_night, SynthConfig
    from .pipeline import run_pipeline

    setup_logging(args.log_level)
    base = Path(args.outdir)
    raw = base / "raw"
    scfg = SynthConfig(nx=args.size, ny=args.size, n_science=args.frames)
    log.info("Generating synthetic night in %s ...", raw)
    generate_night(raw, args.instrument, scfg)

    cfg = _set_paths(PipelineConfig(instrument=args.instrument), base)
    cfg.log_level = args.log_level
    result = run_pipeline(cfg)
    _print_summary(result)
    _maybe_open(result, args.open)
    return 0


def cmd_run(args) -> int:
    from .pipeline import run_pipeline

    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
    else:
        cfg = PipelineConfig(instrument=args.instrument or "generic")
        if args.base:
            _set_paths(cfg, args.base)
        if args.raw:
            cfg.paths.raw = args.raw
    overridden = None
    if args.instrument:
        # an explicit --instrument wins over the config file's instrument
        if args.config and args.instrument != cfg.instrument:
            overridden = cfg.instrument
        cfg.instrument = args.instrument
    if args.log_level:
        # an explicit --log-level wins; otherwise keep the config file's log_level
        cfg.log_level = args.log_level
    setup_logging(cfg.log_level)
    if overridden:
        log.info("Overriding config instrument '%s' with --instrument '%s'",
                 overridden, cfg.instrument)
    result = run_pipeline(cfg)
    _print_summary(result)
    _maybe_open(result, args.open)
    return 0


def cmd_synth(args) -> int:
    from .synth import generate_night, SynthConfig
    setup_logging(args.log_level)
    scfg = SynthConfig(nx=args.size, ny=args.size, n_science=args.frames)
    generate_night(args.outdir, args.instrument, scfg)
    print(f"Synthetic night written to {args.outdir}")
    return 0


def cmd_init_config(args) -> int:
    cfg = PipelineConfig(instrument=args.instrument)
    if args.base:
        _set_paths(cfg, args.base)
    cfg.to_yaml(args.path)
    print(f"Wrote default config to {args.path}")
    return 0


def cmd_instruments(args) -> int:
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        t = Table(title="ARISE instrument profiles", header_style="bold cyan")
        for c in ["profile", "telescope", "detector", "gain", "read noise", "pix scale", "FOV"]:
            t.add_column(c)
        for name, inst in INSTRUMENTS.items():
            t.add_row(name, inst.telescope, inst.detector, f"{inst.gain:g}",
                      f"{inst.read_noise:g}", f"{inst.pixel_scale:g}\"",
                      f"{inst.fov_arcmin or '-'}'")
        console.print(t)
    except Exception:
        for name, inst in INSTRUMENTS.items():
            print(f"{name:14s} {inst.telescope}  gain={inst.gain} rn={inst.read_noise} "
                  f"scale={inst.pixel_scale}\"/px")
    return 0


def _maybe_open(result, do_open: bool) -> None:
    if not do_open:
        return
    report = result.outputs.get("report")
    if report:
        import webbrowser
        webbrowser.open(Path(report).resolve().as_uri())


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arise", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"ARISE {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="generate a synthetic night and run the pipeline")
    d.add_argument("--instrument", default="dfot_2kx2k", choices=list(INSTRUMENTS))
    d.add_argument("--outdir", default="data", help="base output directory")
    d.add_argument("--size", type=int, default=1024, help="synthetic image size (pixels)")
    d.add_argument("--frames", type=int, default=6, help="number of science frames")
    d.add_argument("--open", action="store_true", help="open the HTML report when done")
    d.add_argument("--log-level", default="INFO")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="run the pipeline on a FITS directory or config")
    r.add_argument("--config", help="pipeline config YAML")
    r.add_argument("--raw", help="directory of raw FITS frames")
    r.add_argument("--base", help="base output directory (raw/master/reduced/...)")
    r.add_argument("--instrument", default=None, choices=list(INSTRUMENTS),
                   help="instrument profile; overrides the config file's instrument "
                        "when given with --config (default: generic)")
    r.add_argument("--open", action="store_true")
    r.add_argument("--log-level", default=None,
                   help="logging level; overrides the config file's log_level "
                        "when given with --config (default: INFO)")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("synth", help="only generate a synthetic night")
    s.add_argument("--outdir", default="data/raw")
    s.add_argument("--instrument", default="dfot_2kx2k", choices=list(INSTRUMENTS))
    s.add_argument("--size", type=int, default=1024)
    s.add_argument("--frames", type=int, default=6)
    s.add_argument("--log-level", default="INFO")
    s.set_defaults(func=cmd_synth)

    c = sub.add_parser("init-config", help="write a default config YAML")
    c.add_argument("path")
    c.add_argument("--instrument", default="generic", choices=list(INSTRUMENTS))
    c.add_argument("--base", help="base output directory to bake into the config")
    c.set_defaults(func=cmd_init_config)

    i = sub.add_parser("instruments", help="list built-in instrument profiles")
    i.set_defaults(func=cmd_instruments)
    return p


def main(argv=None) -> int:
    from .keys import load_keys
    load_keys()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        log.error("ARISE failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
