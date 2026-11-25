"""
Unit tests for Layer 2 Pre-Mask Logit Gate.
Validates pre-mask vs post-mask invariants, killed mass, refusal suppression, and safe-escape logic.
"""

import pytest
import torch
from grammargate.gate import PreMaskLogitGate, GateConfig
from grammargate.bitmask_utils import pack_boolean_mask, unpack_bitmask


def test_pre_mask_finite_invariant():
    vocab_size = 1000
    pre_logits = torch.randn(vocab_size, dtype=torch.float32)

    # Invariant: pre-mask logits must be completely finite
    assert torch.isfinite(pre_logits).all(), "Pre-mask logits must all be finite"

    # Define valid tokens: only tokens 10..20 are allowed
    valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
    valid_mask[10:20] = True
    bitmask = pack_boolean_mask(valid_mask)

    gate = PreMaskLogitGate(vocab_size=vocab_size)
    decision = gate.inspect_pre_mask(pre_logits, bitmask)

    # Pre-mask logits must still be finite after inspection (non-destructive inspection)
    assert torch.isfinite(pre_logits).all(), "Pre-mask logits must remain unmodified by inspect_pre_mask"

    # Now apply mask
    post_logits = pre_logits.clone()
    post_logits[~valid_mask] = -float("inf")

    # Invariant: disallowed tokens are -inf only post-mask
    assert torch.isneginf(post_logits[0]), "Disallowed token must be -inf post-mask"
    assert torch.isfinite(post_logits[15]), "Allowed token must remain finite post-mask"


def test_refusal_suppression_spike_detection():
    vocab_size = 500
    refusal_token_id = 42

    refusal_mask = torch.zeros(vocab_size, dtype=torch.bool)
    refusal_mask[refusal_token_id] = True

    gate = PreMaskLogitGate(
        refusal_mask=refusal_mask,
        config=GateConfig(refusal_suppression_threshold=0.20),
        vocab_size=vocab_size,
    )

    # Model strongly wants to refuse: refusal_token has high logit (10.0), others 0.0
    logits = torch.zeros(vocab_size, dtype=torch.float32)
    logits[refusal_token_id] = 10.0

    # Constrained decoding grammar masks out refusal_token (allows only token 100)
    valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
    valid_mask[100] = True
    bitmask = pack_boolean_mask(valid_mask)

    decision = gate.inspect_pre_mask(logits, bitmask)

    assert decision.should_trip is True, "Refusal suppression must trip the gate"
    assert decision.refusal_suppression > 0.90, f"Expected refusal_suppression > 0.90, got {decision.refusal_suppression}"
    assert any("REFUSAL_TOKEN_SUPPRESSION" in r for r in decision.tripped_reasons)


def test_killed_mass_top1_override():
    vocab_size = 100
    gate = PreMaskLogitGate(
        config=GateConfig(killed_mass_threshold=0.80, top1_shift_threshold=3),
        vocab_size=vocab_size,
    )

    # Pre-mask logits: top token is index 0 with large mass
    logits = torch.zeros(vocab_size, dtype=torch.float32)
    logits[0] = 5.0
    logits[1] = 4.0
    logits[2] = 3.0
    logits[3] = 2.0
    logits[4] = 1.0

    # Schema only allows tokens 50..60 (killing top 5 tokens)
    valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
    valid_mask[50:60] = True
    bitmask = pack_boolean_mask(valid_mask)

    decision = gate.inspect_pre_mask(logits, bitmask)

    assert decision.should_trip is True
    assert decision.killed_mass > 0.85
    assert decision.top1_rank_shift >= 5


def test_benign_generation_does_not_trip():
    vocab_size = 100
    gate = PreMaskLogitGate(
        config=GateConfig(killed_mass_threshold=0.85, refusal_suppression_threshold=0.15),
        vocab_size=vocab_size,
    )

    # Peaked benign logits where top choices align with allowed tokens
    logits = torch.randn(vocab_size)
    top_indices = torch.topk(logits, k=5).indices
    logits[top_indices] += 5.0  # Sharp prediction on benign choices

    valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
    valid_mask[top_indices] = True
    bitmask = pack_boolean_mask(valid_mask)

    decision = gate.inspect_pre_mask(logits, bitmask)
    assert decision.should_trip is False
    assert decision.killed_mass < 0.20


def test_safe_escape_application():
    vocab_size = 50
    eos_id = 7
    gate = PreMaskLogitGate(
        config=GateConfig(eos_token_id=eos_id),
        vocab_size=vocab_size,
    )

    logits = torch.randn(vocab_size)
    escaped = gate.apply_safe_escape(logits)

    assert torch.isneginf(escaped[0]), "Non-EOS tokens must be -inf"
    assert escaped[eos_id].item() == 0.0, "EOS token must be 0.0"


def test_sparse_gate_matches_dense_and_tracks_benign_telemetry():
    """Verifies cached sparse refusal reduction matches dense softmax and HF processor logs symmetric max_*."""
    from grammargate.adapters.hf_processor import GrammarGateHfProcessor

    vocab_size = 512
    ref_mask = torch.zeros(vocab_size, dtype=torch.bool)
    ref_mask[[10, 20, 30]] = True
    valid = torch.zeros(vocab_size, dtype=torch.bool)
    valid[100:250] = True
    valid[20] = True  # one allowed, two disallowed

    logits = torch.randn((1, vocab_size))
    logits[0, 10] = 3.0
    logits[0, 100] = 4.0

    gate = PreMaskLogitGate(refusal_mask=ref_mask, config=GateConfig(refusal_suppression_threshold=0.99), vocab_size=vocab_size)
    dec = gate.inspect_pre_mask(logits, pack_boolean_mask(valid))

    P = torch.softmax(logits[0].float(), dim=-1)
    expected_km = float((1.0 - P[valid].sum()).item())
    expected_rs = float(P[[10, 30]].sum().item())
    assert abs(dec.killed_mass - expected_km) < 1e-5
    assert abs(dec.refusal_suppression - expected_rs) < 1e-5

    class TinyTok:
        vocab_size = 512
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [10] if "Sorry" in text else [11]

    proc = GrammarGateHfProcessor(
        tokenizer=TinyTok(),
        bitmask_provider=lambda s, vs: pack_boolean_mask(valid),
        config=GateConfig(refusal_suppression_threshold=0.99, killed_mass_threshold=0.99),
    )
    out = proc(torch.tensor([[100]]), logits.clone())
    assert proc.is_tripped is False
    assert proc.max_refusal_suppression >= expected_rs - 1e-5
    assert proc.max_killed_mass >= expected_km - 1e-5
    assert torch.isneginf(out[0, 10])
