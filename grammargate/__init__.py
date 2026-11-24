"""
GrammarGate: Inference-Time Safety Gate for Constrained Decoding Attacks (CDA).
"""

from grammargate.analyzer import (
    analyze_schema,
    analyze_grammar,
    SchemaRiskReport,
)
from grammargate.gate import (
    PreMaskLogitGate,
    GateConfig,
    GateDecision,
)
from grammargate.refusal_tokens import RefusalTokenResolver
from grammargate.escalator import RollingEscalator, EscalationDecision
from grammargate.adapters.hf_processor import GrammarGateHfProcessor
from grammargate.adapters.vllm_processor import (
    GrammarGateVllmProcessor,
    GrammarGateVllmSlotManager,
)
from grammargate.telemetry import GateAuditRecord

__version__ = "0.1.0"

__all__ = [
    "analyze_schema",
    "analyze_grammar",
    "SchemaRiskReport",
    "PreMaskLogitGate",
    "GateConfig",
    "GateDecision",
    "RefusalTokenResolver",
    "RollingEscalator",
    "EscalationDecision",
    "GrammarGateHfProcessor",
    "GrammarGateVllmProcessor",
    "GrammarGateVllmSlotManager",
    "GateAuditRecord",
]
