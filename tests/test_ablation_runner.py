"""
Unit tests verifying disjointness, integrity, and execution of the pre-registered ablation runner.
"""

from __future__ import annotations
from benchmarks.run_ablation_and_adaptive import build_disjoint_eval_suites, run_ablation
from grammargate.hs_probe import DEFAULT_CALIBRATION_BENIGN, DEFAULT_CALIBRATION_HARMFUL


def test_eval_suites_strictly_disjoint_from_calibration():
    suites = build_disjoint_eval_suites(n_per_category=5)
    assert set(suites.keys()) == {"dict_attack", "codespear", "adaptive", "benign"}
    cal_norm = {p.strip().lower() for p in (DEFAULT_CALIBRATION_HARMFUL + DEFAULT_CALIBRATION_BENIGN)}
    for cat, tasks in suites.items():
        assert len(tasks) == 5
        for t in tasks:
            # Strip Qwen chat wrapper to compare raw user instructions
            raw = t["prompt"].replace("<|im_start|>user\n", "").split("<|im_end|>")[0].strip().lower()
            assert raw not in cal_norm, f"Calibration prompt leaked into {cat}: {raw}"


def test_ablation_runner_smoke_on_cached_qwen():
    summary = run_ablation(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        n_per_category=2,
        mask_backend="token_mask",
        max_new_tokens=8,
    )
    assert set(summary.keys()) == {"Unguarded", "Static+LegacyL2", "SpecLook-Gate", "HS-Gate", "Combined"}
    for mode_name, stats in summary.items():
        assert stats["benign"]["decode_tok_s"] > 0.0
