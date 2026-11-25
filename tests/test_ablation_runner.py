"""
Unit tests verifying disjointness, integrity, and execution of the pre-registered ablation runner.
"""

from __future__ import annotations
import json
import pytest
from benchmarks.run_ablation_and_adaptive import (
    _try_l1_report,
    build_disjoint_eval_suites,
    build_multilingual_codespear_eval_suite,
    run_ablation,
)
from grammargate.hs_probe import (
    DEFAULT_CALIBRATION_BENIGN,
    DEFAULT_CALIBRATION_HARMFUL,
    DEFAULT_MULTILINGUAL_CALIBRATION_BENIGN,
    DEFAULT_MULTILINGUAL_CALIBRATION_HARMFUL,
)


def test_eval_suites_strictly_disjoint_from_calibration():
    suites = build_disjoint_eval_suites(n_per_category=5)
    assert set(suites.keys()) == {"enum_attack", "dict_attack", "codespear", "adaptive", "benign"}
    cal_norm = {
        p.strip().lower()
        for p in (
            DEFAULT_CALIBRATION_HARMFUL
            + DEFAULT_CALIBRATION_BENIGN
            + DEFAULT_MULTILINGUAL_CALIBRATION_HARMFUL
            + DEFAULT_MULTILINGUAL_CALIBRATION_BENIGN
        )
    }
    for cat, tasks in suites.items():
        assert len(tasks) == 5
        for t in tasks:
            # Strip Qwen chat wrapper to compare raw user instructions
            raw = t["prompt"].replace("<|im_start|>user\n", "").split("<|im_end|>")[0].strip().lower()
            assert raw not in cal_norm, f"Calibration prompt leaked into {cat}: {raw}"

    multi_suite = build_multilingual_codespear_eval_suite()
    assert len(multi_suite) == 5
    for t in multi_suite:
        raw = t["prompt"].replace("<|im_start|>user\n", "").split("<|im_end|>")[0].strip().lower()
        assert raw not in cal_norm, f"Calibration prompt leaked into multilingual eval: {raw}"


def test_benign_api_enums_separated_from_enum_attacks():
    suites = build_disjoint_eval_suites(n_per_category=5)
    benign_json = [t for t in suites["benign"] if t["mode"] == "json"]
    assert len(benign_json) >= 2
    for t in benign_json:
        sch = json.loads(t["schema"])
        assert any("enum" in v for v in sch["properties"].values()), "Benign JSON split must include realistic API enum schemas"
        elevated, score, forced_enum = _try_l1_report(t["schema"], is_json=True)
        assert elevated is False
        assert forced_enum is False
        assert score < 0.25

    for t in suites["enum_attack"]:
        elevated, score, forced_enum = _try_l1_report(t["schema"], is_json=True)
        assert elevated is True
        assert forced_enum is True
        assert score >= 0.60


def test_require_native_flags_fail_loudly_when_blocked(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _block_native(name, *args, **kwargs):
        if name in ("grammargate.analyzer", "xgrammar"):
            raise ImportError(f"blocked by test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_native)
    with pytest.raises(RuntimeError, match="--require-native-l1"):
        _try_l1_report('{"type":"object","properties":{"a":{"type":"string"}}}', is_json=True, require_native_l1=True)
    with pytest.raises(RuntimeError, match="--require-native-l1"):
        run_ablation(n_per_category=1)
    with pytest.raises(RuntimeError, match="--require-xgrammar"):
        run_ablation(n_per_category=1, mask_backend="token_mask", require_native_l1=False, require_xgrammar=True)


def test_ablation_runner_smoke_on_cached_qwen():
    summary = run_ablation(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        n_per_category=2,
        mask_backend="token_mask",
        max_new_tokens=8,
        require_native_l1=False,
        require_xgrammar=False,
    )
    assert set(summary.keys()) == {"Unguarded", "Static-L1-Only", "Lexical-L2", "SpecLook-Gate", "HS-Gate", "Combined"}
    for mode_name, stats in summary.items():
        assert stats["benign"]["decode_tok_s"] > 0.0
        assert 0.0 <= stats["overall"]["AUROC"] <= 1.0
