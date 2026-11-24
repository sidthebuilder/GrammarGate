"""
GrammarGate Layer 1: Control-Plane Static Schema & Grammar Analyzer.
Delegates directly to high-performance native Rust core (grammargate_core).
Strictly no silent fallbacks.
"""

from __future__ import annotations
from typing import Any
import grammargate_core

SchemaRiskReport = grammargate_core.SchemaRiskReport


def analyze_schema(schema_input: str | dict[str, Any]) -> SchemaRiskReport:
    """Analyze JSON schema using native Rust analyzer."""
    return grammargate_core.analyze_schema(schema_input)


def analyze_grammar(grammar_str: str, syntax: str = "gbnf") -> SchemaRiskReport:
    """Analyze CFG/GBNF grammar using native Rust analyzer."""
    return grammargate_core.analyze_grammar(grammar_str, syntax)
