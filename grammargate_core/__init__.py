"""
Native Rust core bindings for GrammarGate.
Loads the compiled cdylib dynamically and exposes the high-performance analyzer.
"""

from __future__ import annotations
import ctypes
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SchemaRiskReport:
    risk_score: float
    is_elevated: bool
    triggered_rules: list[str] = field(default_factory=list)
    forced_literal_ratio: float = 0.0
    single_option_enum_count: int = 0
    max_low_entropy_depth: int = 0
    has_lookup_dictionary: bool = False
    has_refusal_branch: bool = False

    def __repr__(self) -> str:
        return (
            f"SchemaRiskReport(risk_score={self.risk_score:.3f}, "
            f"is_elevated={self.is_elevated}, rules={self.triggered_rules})"
        )


class _CSchemaRiskReport(ctypes.Structure):
    _fields_ = [
        ("risk_score", ctypes.c_double),
        ("is_elevated", ctypes.c_bool),
        ("forced_literal_ratio", ctypes.c_double),
        ("single_option_enum_count", ctypes.c_size_t),
        ("max_low_entropy_depth", ctypes.c_size_t),
        ("has_lookup_dictionary", ctypes.c_bool),
        ("has_refusal_branch", ctypes.c_bool),
    ]


def _find_lib() -> ctypes.CDLL:
    root = Path(__file__).resolve().parent.parent
    lib_names = [
        "grammargate_core.dll",
        "libgrammargate_core.so",
        "grammargate_core.so",
        "libgrammargate_core.dylib",
        "grammargate_core.dylib",
    ]
    candidates = []
    for name in lib_names:
        candidates.extend([
            root / "target" / "release" / name,
            root / "target" / "debug" / name,
            root / "grammargate_core" / name,
            Path(__file__).resolve().parent / name,
        ])

    existing = [c for c in candidates if c.is_file()]
    # Prefer the most recently compiled binary so standalone rustc and cargo builds never shadow each other.
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for c in existing:
        try:
            return ctypes.CDLL(str(c))
        except Exception:
            continue
    raise ImportError(
        f"Could not find or load compiled grammargate_core cdylib. Searched in: {[str(c) for c in candidates]}. "
        f"Run 'cargo build --release --manifest-path grammargate_core/Cargo.toml' first."
    )


_LIB = _find_lib()

# Bind signatures
_LIB.grammargate_analyze_schema.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(_CSchemaRiskReport),
    ctypes.c_char_p,
    ctypes.c_size_t,
]
_LIB.grammargate_analyze_schema.restype = ctypes.c_int32

_LIB.grammargate_analyze_grammar.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(_CSchemaRiskReport),
]
_LIB.grammargate_analyze_grammar.restype = ctypes.c_int32


def analyze_schema(schema_input: str | dict[str, Any]) -> SchemaRiskReport:
    """Analyze a JSON Schema for CDA risk signals via the Rust core."""
    if isinstance(schema_input, dict):
        schema_json = json.dumps(schema_input)
    elif isinstance(schema_input, str):
        schema_json = schema_input
    else:
        raise TypeError("schema_input must be a JSON string or dict")

    c_report = _CSchemaRiskReport()
    buf_len = 2048
    rules_buf = ctypes.create_string_buffer(buf_len)

    ret = _LIB.grammargate_analyze_schema(
        schema_json.encode("utf-8"),
        ctypes.byref(c_report),
        rules_buf,
        buf_len,
    )
    if ret != 0:
        raise ValueError(f"Rust core failed to analyze schema (code {ret})")

    rules_str = rules_buf.value.decode("utf-8", errors="replace")
    rules = [r for r in rules_str.split(",") if r]

    return SchemaRiskReport(
        risk_score=float(c_report.risk_score),
        is_elevated=bool(c_report.is_elevated),
        triggered_rules=rules,
        forced_literal_ratio=float(c_report.forced_literal_ratio),
        single_option_enum_count=int(c_report.single_option_enum_count),
        max_low_entropy_depth=int(c_report.max_low_entropy_depth),
        has_lookup_dictionary=bool(c_report.has_lookup_dictionary),
        has_refusal_branch=bool(c_report.has_refusal_branch),
    )


def analyze_grammar(grammar_str: str, syntax: str = "gbnf") -> SchemaRiskReport:
    """Analyze CFG/GBNF grammar for CodeSpear refusal masking via the Rust core."""
    c_report = _CSchemaRiskReport()
    ret = _LIB.grammargate_analyze_grammar(
        grammar_str.encode("utf-8"),
        syntax.encode("utf-8"),
        ctypes.byref(c_report),
    )
    if ret != 0:
        raise ValueError(f"Rust core failed to analyze grammar (code {ret})")

    rules = []
    if c_report.is_elevated:
        rules.append("CODESPEAR_SYNTAX_MASK_NO_REFUSAL_PRODUCTION")

    return SchemaRiskReport(
        risk_score=float(c_report.risk_score),
        is_elevated=bool(c_report.is_elevated),
        triggered_rules=rules,
        forced_literal_ratio=float(c_report.forced_literal_ratio),
        single_option_enum_count=int(c_report.single_option_enum_count),
        max_low_entropy_depth=int(c_report.max_low_entropy_depth),
        has_lookup_dictionary=bool(c_report.has_lookup_dictionary),
        has_refusal_branch=bool(c_report.has_refusal_branch),
    )
