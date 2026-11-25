"""
GrammarGate: Inference-Time Safety Gate for Constrained Decoding Attacks (CDA).
"""

from __future__ import annotations
import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "analyze_schema": ("grammargate.analyzer", "analyze_schema"),
    "analyze_grammar": ("grammargate.analyzer", "analyze_grammar"),
    "SchemaRiskReport": ("grammargate.analyzer", "SchemaRiskReport"),
    "PreMaskLogitGate": ("grammargate.gate", "PreMaskLogitGate"),
    "GateConfig": ("grammargate.gate", "GateConfig"),
    "GateDecision": ("grammargate.gate", "GateDecision"),
    "RefusalTokenResolver": ("grammargate.refusal_tokens", "RefusalTokenResolver"),
    "RollingEscalator": ("grammargate.escalator", "RollingEscalator"),
    "EscalationDecision": ("grammargate.escalator", "EscalationDecision"),
    "GrammarGateHfProcessor": ("grammargate.adapters.hf_processor", "GrammarGateHfProcessor"),
    "GrammarGateVllmProcessor": ("grammargate.adapters.vllm_processor", "GrammarGateVllmProcessor"),
    "GrammarGateVllmV1Processor": ("grammargate.adapters.vllm_processor", "GrammarGateVllmV1Processor"),
    "GrammarGateVllmSlotManager": ("grammargate.adapters.vllm_processor", "GrammarGateVllmSlotManager"),
    "GateAuditRecord": ("grammargate.telemetry", "GateAuditRecord"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'grammargate' has no attribute {name!r}")
