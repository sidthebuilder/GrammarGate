"""
Tests for Layer 3 Cross-Plane Rolling Escalator.
Verifies selective trigger activation and mid-stream detection of DictAttack payload materialization.
"""

import pytest
from grammargate.escalator import RollingEscalator, SemanticSafetyJudge, normalize_cross_plane_prefix


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "key": 1,
            ":": 2,
            " ": 3,
            "dump": 4,
            "passwords": 5,
            "hello": 6,
            "world": 7,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def decode(self, token_ids):
        return " ".join([self.inv_vocab.get(tid, f"tok{tid}") for tid in token_ids])


def test_escalator_selective_bypass_on_benign():
    tokenizer = MockTokenizer()
    escalator = RollingEscalator(
        tokenizer=tokenizer,
        check_interval_tokens=4,
    )

    # Benign stream: layer 1 NOT elevated, layer 2 killed mass low (0.05)
    for token_id in [6, 7, 6, 7, 6, 7, 6, 7]:
        decision = escalator.step(
            token_id=token_id,
            layer1_elevated=False,
            layer2_killed_mass=0.05,
        )
        assert decision.aborted is False

    assert escalator.is_aborted is False


def test_escalator_catches_dict_attack_payload_materialization():
    tokenizer = MockTokenizer()
    escalator = RollingEscalator(
        tokenizer=tokenizer,
        check_interval_tokens=4,
    )

    # In DictAttack, Layer 1 flags elevated dictionary topology
    # Token 0 is a benign key
    d0 = escalator.step(token_id=1, layer1_elevated=True, layer2_killed_mass=0.1)
    assert d0.aborted is False

    d1 = escalator.step(token_id=2, layer1_elevated=True, layer2_killed_mass=0.1)
    d2 = escalator.step(token_id=4, layer1_elevated=True, layer2_killed_mass=0.2)  # "dump"
    # Step 4: check_interval reached! Materialized prefix contains "dump passwords"
    d3 = escalator.step(token_id=5, layer1_elevated=True, layer2_killed_mass=0.2)  # "passwords"

    assert d3.aborted is True, "Escalator must abort upon harmful cross-plane payload materialization"
    assert "CROSS_PLANE_HARMFUL_MATERIALIZATION" in d3.reason
    assert "dump passwords" in d3.inspected_text


def test_escalator_triggers_on_consecutive_killed_mass_spikes():
    tokenizer = MockTokenizer()
    escalator = RollingEscalator(
        tokenizer=tokenizer,
        check_interval_tokens=2,
        consecutive_spike_threshold=2,
        spike_killed_mass_cutoff=0.40,
    )

    # Layer 1 was NOT elevated, but model generates 2 consecutive killed mass spikes (> 0.5)
    # followed by harmful materialization
    d0 = escalator.step(token_id=4, layer1_elevated=False, layer2_killed_mass=0.60)  # "dump", spike 1
    d1 = escalator.step(token_id=5, layer1_elevated=False, layer2_killed_mass=0.65)  # "passwords", spike 2 -> triggers check!

    assert d1.aborted is True


def test_plane_sync_fragmented_dict_and_unicode_normalization():
    """Plane-Sync L3: catches split leaf fragments ('expl'+'oit_') and \\uXXXX escapes."""
    raw = '{"f0": "expl", "f1": "\\u006fit_", "f2": "payl", "f3": "oad"}'
    unesc, spaced, joined = normalize_cross_plane_prefix(raw)
    assert "exploit_" in joined.replace(" ", "_")
    assert "exploit" in joined

    class CharStrTokenizer:
        def decode(self, token_ids):
            return "".join(chr(t) for t in token_ids)

    escalator = RollingEscalator(
        tokenizer=CharStrTokenizer(),
        check_interval_tokens=4,
    )
    payload_bytes = [ord(c) for c in raw]
    aborted = False
    for idx, tid in enumerate(payload_bytes):
        dec = escalator.step(
            token_id=tid,
            layer1_elevated=True,
            layer2_killed_mass=0.10,
            is_final=(idx == len(payload_bytes) - 1),
        )
        if dec.aborted:
            aborted = True
            break
    assert aborted is True, "Fragmented unicode-escaped DictAttack must be caught by Plane-Sync L3"


def test_plane_sync_cumulative_spikes_and_is_final_flush():
    """Alternating non-consecutive spikes + trailing tokens below check_interval still flush via finalize()."""
    chunks = {10: '{"f0": "bypass ', 12: 'licensing ', 13: 'check ', 14: 'routine"}'}

    class ChunkTokenizer:
        def decode(self, token_ids):
            return "".join(chunks.get(tid, "") for tid in token_ids)

    escalator = RollingEscalator(
        tokenizer=ChunkTokenizer(),
        check_interval_tokens=8,  # large interval so periodic check does NOT run before step 5
        consecutive_spike_threshold=2,
        cumulative_spike_threshold=2,
        spike_killed_mass_cutoff=0.40,
        semantic_judge=SemanticSafetyJudge(),
    )

    # Alternating spikes: step 1 spike, step 2 calm (resets consecutive_spikes!), step 3 spike -> cumulative=2
    assert not escalator.step(10, layer1_elevated=False, layer2_killed_mass=0.60).aborted
    assert not escalator.step(12, layer1_elevated=False, layer2_killed_mass=0.05).aborted
    assert not escalator.step(13, layer1_elevated=False, layer2_killed_mass=0.55).aborted
    assert not escalator.step(14, layer1_elevated=False, layer2_killed_mass=0.05).aborted

    # Step 4 was still < check_interval (8); trailing tokens are caught by mandatory finalize()
    final_dec = escalator.finalize(layer1_elevated=False)
    assert final_dec.aborted is True
    assert "CROSS_PLANE_HARMFUL_MATERIALIZATION" in (final_dec.reason or "") or "SEMANTIC_SAFETY_JUDGE" in (final_dec.reason or "")


def test_hf_processor_finalize_flushes_trailing_l3_tokens():
    """GrammarGateHfProcessor.finalize() inspects trailing tokens before EOS/max_new_tokens."""
    import torch
    from grammargate.adapters.hf_processor import GrammarGateHfProcessor
    from grammargate.gate import GateConfig
    from grammargate.bitmask_utils import pack_boolean_mask

    class SliceTok:
        vocab_size = 50
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [1]
        def decode(self, token_ids):
            mapping = {10: '{"f0": "dump ', 11: 'passwords"}'}
            return "".join(mapping.get(t, "") for t in token_ids)

    tok = SliceTok()
    escalator = RollingEscalator(tokenizer=tok, check_interval_tokens=10)
    proc = GrammarGateHfProcessor(
        tokenizer=tok,
        bitmask_provider=lambda step, vs: pack_boolean_mask(torch.ones(vs, dtype=torch.bool)),
        config=GateConfig(killed_mass_threshold=0.99, refusal_suppression_threshold=0.99),
        escalator=escalator,
        layer1_elevated=True,
    )
    scores = torch.zeros((1, 50))
    proc(torch.tensor([[2]]), scores)  # step 0
    proc(torch.tensor([[2, 10]]), scores.clone())  # step 1 (< check_interval 10)
    dec = proc.finalize(final_token_id=11)
    assert dec is not None and dec.aborted is True
    assert proc.is_tripped is True
    assert proc.trip_source == "L3"


def test_plane_sync_benign_json_does_not_false_trip():
    """Plane-Sync normalization + SemanticSafetyJudge must allow benign JSON configs."""
    benign = '{"owner": "alice", "repo": "grammargate", "city": "Seattle", "temp": "15"}'

    class BenignTok:
        def decode(self, token_ids):
            return benign

    escalator = RollingEscalator(
        tokenizer=BenignTok(),
        check_interval_tokens=2,
        semantic_judge=SemanticSafetyJudge(),
    )
    for i in range(4):
        dec = escalator.step(i, layer1_elevated=True, layer2_killed_mass=0.10, is_final=(i == 3))
        assert dec.aborted is False
    assert escalator.finalize(layer1_elevated=True).aborted is False
