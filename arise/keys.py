"""Local API-key management.

Keys live in ``config/keys.yaml`` (gitignored) and are exported to environment
variables at startup so every stage can read them the same way. Every key is
optional -- a missing key just degrades the related feature gracefully.

Env vars set:
    ASTROMETRY_NET_API_KEY   blind plate solving (nova.astrometry.net)
    NASA_API_KEY             NeoWs context checks (api.nasa.gov)
    NVIDIA_API_KEYS          comma-separated, generative Ask-ARISE answers
    ANTHROPIC_API_KEY        preferred generative backend when present
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from .logs import get_logger

log = get_logger("keys")

_ENV_MAP = {
    "astrometry_net": "ASTROMETRY_NET_API_KEY",
    "nasa": "NASA_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def load_keys(project_root: str | Path | None = None) -> dict[str, str]:
    """Read config/keys.yaml (if present) into os.environ. Idempotent."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    path = root / "config" / "keys.yaml"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("Could not parse %s: %s", path, exc)
        return loaded
    if not isinstance(data, dict):
        log.warning("Ignoring %s: expected a mapping of key names to values, got %s",
                    path, type(data).__name__)
        return loaded

    for field, env in _ENV_MAP.items():
        val = data.get(field)
        if val and not os.environ.get(env):
            os.environ[env] = str(val).strip()
            loaded[env] = "set"
    nvidia = data.get("nvidia") or []
    if isinstance(nvidia, str):
        nvidia = [nvidia]
    nvidia = [str(k).strip() for k in nvidia if str(k).strip()]
    if nvidia and not os.environ.get("NVIDIA_API_KEYS"):
        os.environ["NVIDIA_API_KEYS"] = ",".join(nvidia)
        loaded["NVIDIA_API_KEYS"] = f"{len(nvidia)} key(s)"
    if loaded:
        log.info("Loaded local API keys: %s", ", ".join(sorted(loaded)))
    return loaded
