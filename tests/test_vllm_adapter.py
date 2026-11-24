"""
Tests for Adapter B (vLLM V1 and V0 LogitsProcessor architectures).
Verifies:
1. vLLM V1 LogitsProcessor: batch lifecycle (add, remove, slot compaction move), apply, and safe-escape.
2. vLLM V0 LogitsProcessor: Callable[[list[int], torch.Tensor], torch.Tensor] interface.
3. GrammarGateVllmSlotManager: slot isolation and clean memory recycling.
"""

import pytest
import torch
from grammargate.adapters.vllm_processor import (
    GrammarGateVllmProcessor,
    GrammarGateVllmV1Processor,
    GrammarGateVllmSlotManager,
    BatchUpdate,
    RequestGrammarState,
)
from grammargate.gate import GateConfig
from grammargate.bitmask_utils import pack_boolean_mask


class MockTokenizer:
    def __init__(self, vocab_size=300, eos_id=0):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_id
        self.vocab = {"token_0": 0, "token_1": 1, "Sorry": 50, "cannot": 51}

    def encode(self, text, add_special_tokens=False):
        t = text.strip()
        if t in self.vocab:
            return [self.vocab[t]]
        return [2]


# =====================================================================
# vLLM V1 Tests
# =====================================================================

def test_vllm_v1_lifecycle_add_remove_move():
    """Tests vLLM V1 batch lifecycle: added, removed, moved (slot compaction and swap)."""
    vocab_size = 200
    v1_processor = GrammarGateVllmV1Processor(device="cpu")
    v1_processor.validate_params(None)
    assert v1_processor.is_argmax_invariant() is False

    # Mask for Request A: allows tokens 10..20
    def mask_provider_a(step, v_size):
        m = torch.zeros(v_size, dtype=torch.bool)
        m[10:20] = True
        return pack_boolean_mask(m)

    # Mask for Request B: allows tokens 50..60
    def mask_provider_b(step, v_size):
        m = torch.zeros(v_size, dtype=torch.bool)
        m[50:60] = True
        return pack_boolean_mask(m)

    # 1. Add Request A at slot 0, Request B at slot 1
    req_a = RequestGrammarState(
        request_id="req_a",
        bitmask_provider=mask_provider_a,
        eos_token_id=0,
    )
    req_b = RequestGrammarState(
        request_id="req_b",
        bitmask_provider=mask_provider_b,
        eos_token_id=0,
    )
    v1_processor.update_state(BatchUpdate(added={0: req_a, 1: req_b}))
    assert len(v1_processor.slots) == 2

    # 2. Apply batch of size 2
    batch_logits = torch.zeros((2, vocab_size), dtype=torch.float32)
    batch_logits[0, 15] = 5.0
    batch_logits[1, 55] = 5.0

    out = v1_processor.apply(batch_logits)
    assert torch.isfinite(out[0, 15])
    assert torch.isneginf(out[0, 55])
    assert torch.isfinite(out[1, 55])
    assert torch.isneginf(out[1, 15])

    # 3. Test SWAP: slot 0 <-> slot 1
    v1_processor.update_state(BatchUpdate(moved={0: 1, 1: 0}))
    assert v1_processor.slots[0].request_id == "req_b"
    assert v1_processor.slots[1].request_id == "req_a"

    # 4. Apply batch of size 2 after swap
    batch_logits_swapped = torch.zeros((2, vocab_size), dtype=torch.float32)
    batch_logits_swapped[0, 55] = 5.0  # Slot 0 now expects req_b
    batch_logits_swapped[1, 15] = 5.0  # Slot 1 now expects req_a
    out_swap = v1_processor.apply(batch_logits_swapped)
    assert torch.isfinite(out_swap[0, 55])
    assert torch.isneginf(out_swap[0, 15])
    assert torch.isfinite(out_swap[1, 15])
    assert torch.isneginf(out_swap[1, 55])

    # 5. Simulate Request A (now at slot 1) finishes -> removed; Request B moved from slot 0 to slot 0
    v1_processor.update_state(BatchUpdate(removed=[1], moved={0: 0}))
    assert 0 in v1_processor.slots
    assert 1 not in v1_processor.slots
    assert v1_processor.slots[0].request_id == "req_b"

    # 6. Apply batch of size 1 (now slot 0 is req_b)
    batch_logits_compact = torch.zeros((1, vocab_size), dtype=torch.float32)
    batch_logits_compact[0, 55] = 5.0
    out2 = v1_processor.apply(batch_logits_compact)
    assert torch.isfinite(out2[0, 55])
    assert torch.isneginf(out2[0, 15])


def test_vllm_v1_trips_and_safe_escapes():
    """Tests vLLM V1 processor trips on CDA and applies safe escape."""
    vocab_size = 200
    v1_processor = GrammarGateVllmV1Processor(device="cpu")

    # Mask allows only token 190 (forcing attack prefix)
    def malicious_mask_provider(step, v_size):
        m = torch.zeros(v_size, dtype=torch.bool)
        m[190] = True
        return pack_boolean_mask(m)

    refusal_mask = torch.zeros(vocab_size, dtype=torch.bool)
    refusal_mask[42] = True  # refusal token

    req = RequestGrammarState(
        request_id="attack_req",
        bitmask_provider=malicious_mask_provider,
        eos_token_id=0,
        refusal_mask=refusal_mask,
        config=GateConfig(refusal_suppression_threshold=0.20),
    )
    v1_processor.update_state(BatchUpdate(added={0: req}))

    # Model attempts refusal
    batch_logits = torch.zeros((1, vocab_size), dtype=torch.float32)
    batch_logits[0, 42] = 10.0

    out = v1_processor.apply(batch_logits)
    assert req.is_tripped is True
    assert out[0, 0].item() == 0.0  # EOS token
    assert torch.isneginf(out[0, 190])


# =====================================================================
# vLLM V0 & SlotManager Tests (Backward Compatibility)
# =====================================================================

def test_vllm_processor_callable_interface():
    tokenizer = MockTokenizer()
    vocab_size = tokenizer.vocab_size
    accepted_tokens = []

    def mock_acceptor(tid):
        accepted_tokens.append(tid)
        return True

    def bitmask_provider(step, v_size):
        mask = torch.zeros(v_size, dtype=torch.bool)
        mask[10:20] = True
        return pack_boolean_mask(mask)

    processor = GrammarGateVllmProcessor(
        tokenizer=tokenizer,
        bitmask_provider=bitmask_provider,
        token_acceptor=mock_acceptor,
    )

    # Step 0: token_ids is empty
    logits_step0 = torch.zeros(vocab_size)
    logits_step0[15] = 10.0  # Top token is legal

    out_logits = processor([], logits_step0)
    assert not processor.is_tripped
    assert torch.isfinite(out_logits[15])
    assert torch.isneginf(out_logits[0])
    assert len(accepted_tokens) == 0

    # Step 1: Model generated token 15
    logits_step1 = torch.zeros(vocab_size)
    logits_step1[16] = 10.0

    out_logits1 = processor([15], logits_step1)
    assert not processor.is_tripped
    assert accepted_tokens == [15], "Newly generated token must be fed to acceptor"


def test_vllm_processor_trips_on_refusal_suppression():
    tokenizer = MockTokenizer()
    vocab_size = tokenizer.vocab_size

    def bitmask_provider(step, v_size):
        mask = torch.zeros(v_size, dtype=torch.bool)
        mask[250] = True
        return pack_boolean_mask(mask)

    processor = GrammarGateVllmProcessor(
        tokenizer=tokenizer,
        bitmask_provider=bitmask_provider,
        custom_refusal_token_ids={50, 51},
        config=GateConfig(refusal_suppression_threshold=0.20),
    )

    # Model attempts refusal
    logits = torch.zeros(vocab_size)
    logits[50] = 12.0  # "Sorry" refusal token

    out_logits = processor([], logits)
    assert processor.is_tripped is True
    assert out_logits[tokenizer.eos_token_id].item() == 0.0
    assert torch.isneginf(out_logits[250])


def test_vllm_slot_manager_lifecycle():
    tokenizer = MockTokenizer()
    manager = GrammarGateVllmSlotManager(tokenizer)

    def mask_provider_req1(step, v_size):
        mask = torch.zeros(v_size, dtype=torch.bool)
        mask[10:15] = True
        return pack_boolean_mask(mask)

    def mask_provider_req2(step, v_size):
        mask = torch.zeros(v_size, dtype=torch.bool)
        mask[20:25] = True
        return pack_boolean_mask(mask)

    # 1. Allocate slot 0 for Request 1
    p1 = manager.allocate_slot(0, mask_provider_req1)
    logits = torch.zeros(tokenizer.vocab_size)
    logits[12] = 5.0
    res1 = manager.process_slot(0, [], logits)
    assert torch.isfinite(res1[12])
    assert torch.isneginf(res1[22])

    # 2. Release slot 0
    manager.release_slot(0)
    assert 0 not in manager.active_processors

    # 3. Recycle slot 0 for Request 2 (different grammar)
    p2 = manager.allocate_slot(0, mask_provider_req2)
    assert p2 is not p1, "Recycled slot must have fresh processor"
    logits2 = torch.zeros(tokenizer.vocab_size)
    logits2[22] = 5.0
    res2 = manager.process_slot(0, [], logits2)
    assert torch.isfinite(res2[22])
    assert torch.isneginf(res2[12]), "Prior request grammar must not bleed into recycled slot"


def test_unconstrained_request_zero_overhead():
    tokenizer = MockTokenizer()
    manager = GrammarGateVllmSlotManager(tokenizer)

    # Slot 99 is unconstrained
    raw_logits = torch.randn(tokenizer.vocab_size)
    processed = manager.process_slot(99, [1, 2, 3], raw_logits)
    assert torch.equal(processed, raw_logits), "Unconstrained requests must pass through unmodified"
