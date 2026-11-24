"""
Tests for Layer 3 Cross-Plane Rolling Escalator.
Verifies selective trigger activation and mid-stream detection of DictAttack payload materialization.
"""

import pytest
from grammargate.escalator import RollingEscalator


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
