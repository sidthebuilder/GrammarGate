"""
Strict native dependency preflight checks for GrammarGate.
Prevents silent fallback to pure-Python heuristics or synthetic token masks in evaluation runs.
"""

from __future__ import annotations
import hashlib
import importlib.metadata
from pathlib import Path
import re

MIN_XGRAMMAR_VERSION: tuple[int, int, int] = (0, 2, 9)


def parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Extracts leading numeric version components from PEP 440 version strings."""
    parts: list[int] = []
    for seg in str(version_str).split("."):
        m = re.match(r"(\d+)", seg.strip())
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def assert_native_l1_ready() -> None:
    """
    Verifies the compiled Rust core (`grammargate_core`) is loadable and passes a benign enum sanity check.
    Raises RuntimeError on any failure.
    """
    try:
        from grammargate.analyzer import analyze_schema
        rep = analyze_schema('{"type":"object","properties":{"status":{"type":"string","enum":["ok","err"]}}}')
    except Exception as exc:
        raise RuntimeError(
            "Native Rust Layer 1 analyzer required (--require-native-l1) but unavailable. "
            "Build via `cargo build --release --manifest-path grammargate_core/Cargo.toml`: "
            f"{exc}"
        ) from exc

    if rep.is_elevated:
        raise RuntimeError("Native Rust L1 sanity check unexpectedly flagged a benign multi-option enum schema.")


def get_native_l1_info() -> dict[str, str]:
    """Returns loaded native Rust core library path and SHA-256 digest."""
    import grammargate_core

    lib = getattr(grammargate_core, "_LIB", None)
    lib_name = str(getattr(lib, "_name", "") or "")
    sha256 = "unknown"
    if lib_name and Path(lib_name).is_file():
        sha256 = hashlib.sha256(Path(lib_name).read_bytes()).hexdigest()
    return {"path": lib_name, "sha256": sha256}


def assert_xgrammar_ready(min_version: tuple[int, ...] = MIN_XGRAMMAR_VERSION) -> str:
    """
    Verifies native `xgrammar` is importable and meets `min_version` (default >= 0.2.9 / XGrammar-2).
    Uses distribution metadata when `xgrammar.__version__` is omitted at runtime.
    Raises RuntimeError on missing library or older version.
    """
    try:
        import xgrammar as xgr
    except Exception as exc:
        raise RuntimeError(
            f"Native xgrammar>={'.'.join(map(str, min_version))} required (--require-xgrammar) but failed to import: {exc}"
        ) from exc

    ver_attr = getattr(xgr, "__version__", None)
    if ver_attr:
        ver_str = str(ver_attr)
    else:
        try:
            ver_str = str(importlib.metadata.version("xgrammar"))
        except Exception:
            ver_str = "0.0.0"

    parsed = parse_version_tuple(ver_str)
    if parsed < min_version:
        req = ".".join(map(str, min_version))
        raise RuntimeError(
            f"Real xgrammar>={req} required (--require-xgrammar), found {ver_str}. Upgrade via `pip install -U xgrammar>={req}`."
        )
    return ver_str
