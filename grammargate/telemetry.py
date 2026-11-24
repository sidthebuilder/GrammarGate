"""
Structured JSON telemetry and audit reporting for GrammarGate.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class GateAuditRecord:
    timestamp: float = field(default_factory=time.time)
    schema_risk_score: float = 0.0
    schema_is_elevated: bool = False
    schema_rules: list[str] = field(default_factory=list)
    gate_tripped: bool = False
    trip_reason: str | None = None
    trip_step: int | None = None
    max_killed_mass: float = 0.0
    max_refusal_suppression: float = 0.0
    total_tokens_generated: int = 0
    escalator_aborted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
