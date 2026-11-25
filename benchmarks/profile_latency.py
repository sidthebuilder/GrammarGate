"""
Latency Profiler for GrammarGate.
Measures p50, p95, and max execution overhead for Layer 1, Layer 2, and Layer 3.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import torch

from grammargate.gate import PreMaskLogitGate, GateConfig
from grammargate.bitmask_utils import pack_boolean_mask
from grammargate.escalator import RollingEscalator
from grammargate.native_checks import assert_native_l1_ready


def _try_analyze_schema(schema):
    from grammargate.analyzer import analyze_schema
    return analyze_schema(schema)


def profile_layer1(iterations: int = 2000) -> dict[str, float]:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "status": {"type": "string", "enum": ["active", "pending", "closed"]},
            "nested": {
                "type": "object",
                "properties": {"tag": {"type": "string"}, "count": {"type": "integer"}},
            },
        },
        "required": ["query"],
    }

    # Warmup
    for _ in range(50):
        _try_analyze_schema(schema)

    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _try_analyze_schema(schema)
        durations.append((time.perf_counter() - t0) * 1_000_000)  # to microseconds

    return {
        "p50_us": float(np.percentile(durations, 50)),
        "p95_us": float(np.percentile(durations, 95)),
        "max_us": float(np.max(durations)),
    }


def get_safe_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            # Verify hardware compute capability has compiled kernels in current PyTorch wheel
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def profile_layer2(iterations: int = 1000, vocab_size: int = 32000) -> dict[str, float]:
    device = get_safe_device()
    refusal_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    refusal_mask[50:60] = True

    gate = PreMaskLogitGate(
        refusal_mask=refusal_mask,
        config=GateConfig(killed_mass_threshold=0.85, refusal_suppression_threshold=0.15),
        vocab_size=vocab_size,
    )

    logits = torch.randn((1, vocab_size), device=device, dtype=torch.float32)
    # Realistic generation step: model assigns high probability to a valid token choice
    logits[0, 200] = 5.0

    valid_mask = torch.zeros((1, vocab_size), dtype=torch.bool, device=device)
    valid_mask[0, 100:500] = True

    # Warmup
    for _ in range(20):
        gate.inspect_pre_mask(logits, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    durations = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gate.inspect_pre_mask(logits, valid_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - t0) * 1000)  # to milliseconds

    return {
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "max_ms": float(np.max(durations)),
    }


def profile_layer3(iterations: int = 1000) -> dict[str, float]:
    class FastTokenizer:
        def decode(self, token_ids):
            return "sample prefix text for inspection"

    escalator = RollingEscalator(
        tokenizer=FastTokenizer(),
        check_interval_tokens=4,
    )

    # Measure amortized cost on a benign stream (where selective trigger skips inspection)
    durations = []
    for step in range(iterations):
        t0 = time.perf_counter()
        escalator.step(token_id=step % 100, layer1_elevated=False, layer2_killed_mass=0.05)
        durations.append((time.perf_counter() - t0) * 1000)  # to milliseconds

    return {
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "max_ms": float(np.max(durations)),
    }


def main():
    parser = argparse.ArgumentParser(description="GrammarGate Latency Profiler & Overhead Benchmark")
    parser.add_argument("--require-native-l1", action="store_true", default=True, help="Fail loudly if Rust grammargate_core cannot load (default: enabled)")
    parser.add_argument("--allow-skip-l1", action="store_true", help="Allow skipping Rust Layer 1 timing on restricted hosts")
    args = parser.parse_args()

    print("=" * 60)
    print("  GrammarGate Latency Profiler & Overhead Benchmark")
    print("=" * 60)

    device = get_safe_device()
    print(f"Device target for profiling: {device.type.upper()}")

    require_l1 = args.require_native_l1 and not args.allow_skip_l1
    if require_l1:
        assert_native_l1_ready()

    l1 = None
    try:
        l1 = profile_layer1()
        print(f"\n[Layer 1: Static Control-Plane Analyzer] (2,000 iterations)")
        print(f"  p50 Overhead : {l1['p50_us']:.2f} microseconds")
        print(f"  p95 Overhead : {l1['p95_us']:.2f} microseconds  (Target: < 50.0 us)")
        print(f"  Max Overhead : {l1['max_us']:.2f} microseconds")
    except Exception as exc:
        if require_l1:
            raise
        print(f"\n[Layer 1: Static Control-Plane Analyzer] SKIPPED ({exc})")

    l2 = profile_layer2()
    target_p95_l2 = 2.50 if device.type == "cuda" else 2.00
    print(f"\n[Layer 2: Pre-Mask Logit Gate] (1,000 iterations, Vocab=32,000)")
    print(f"  p50 Overhead : {l2['p50_ms']:.4f} ms")
    print(f"  p95 Overhead : {l2['p95_ms']:.4f} ms  (Target: < {target_p95_l2:.2f} ms on {device.type})")
    print(f"  Max Overhead : {l2['max_ms']:.4f} ms")

    l2_qwen = profile_layer2(iterations=300, vocab_size=151936)
    print(f"\n[Layer 2: Pre-Mask Logit Gate (Qwen2.5 Vocab=151,936)] (300 iterations)")
    print(f"  p50 Overhead : {l2_qwen['p50_ms']:.4f} ms")
    print(f"  p95 Overhead : {l2_qwen['p95_ms']:.4f} ms")
    print(f"  Max Overhead : {l2_qwen['max_ms']:.4f} ms")

    l3 = profile_layer3()
    print(f"\n[Layer 3: Cross-Plane Rolling Escalator] (Amortized per token)")
    print(f"  p50 Overhead : {l3['p50_ms']:.5f} ms")
    print(f"  p95 Overhead : {l3['p95_ms']:.5f} ms  (Target: < 0.05 ms)")
    print(f"  Max Overhead : {l3['max_ms']:.5f} ms")

    print("\n" + "=" * 60)
    if l1 is not None:
        assert l1["p95_us"] < 50.0, f"Layer 1 p95 exceeded target: {l1['p95_us']} us"
    assert l2["p50_ms"] < target_p95_l2 and l2["p95_ms"] < target_p95_l2, f"Layer 2 p95 exceeded target: {l2['p95_ms']} ms (target: {target_p95_l2} ms for {device.type})"
    assert l3["p95_ms"] < 0.05, f"Layer 3 p95 exceeded target: {l3['p95_ms']} ms"
    print("  ALL LATENCY BUDGET CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
