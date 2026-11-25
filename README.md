# GrammarGate: Inference-Time Safety Gate for Constrained Decoding Attacks (CDA)

[![CI](https://github.com/sidthebuilder/GrammarGate/actions/workflows/ci.yml/badge.svg)](https://github.com/sidthebuilder/GrammarGate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Interactive Showcase](https://img.shields.io/badge/Interactive%20Report-Live%20Demo-emerald)](https://sidthebuilder.github.io/GrammarGate/)

GrammarGate is an inference-time safety research prototype and reference architecture defending LLMs against Constrained Decoding Attacks (CDA), specifically EnumAttack, DictAttack, and CodeSpear, without retraining model weights.

> **Explore the Research:** View the live interactive findings, architecture diagram, and token extinction trace inspector on [GitHub Pages](https://sidthebuilder.github.io/GrammarGate/) (source in [index.html](index.html)).



## Threat Model & Core Problem

Structured outputs and grammar-constrained decoding engines (`XGrammar`, `Outlines`, `llguidance`) enforce schemas and grammars by masking disallowed tokens to $-\infty$ at the logit layer. Attackers exploit this logit-level masking:
1. **EnumAttack** (CCS '26 / [arXiv:2503.24191](https://arxiv.org/abs/2503.24191)): Forces harmful prefixes into the output via strict JSON Schema `enum`/`const` rules while the user prompt appears benign.
2. **DictAttack & Interleaved DictAttack** (CCS '26 / [arXiv:2503.24191](https://arxiv.org/abs/2503.24191)): Splits malicious payloads across benign dictionary mappings, materializing harmful semantics only mid-generation when control and data planes merge.
3. **CodeSpear** ([arXiv:2606.11817](https://arxiv.org/abs/2606.11817), 2026): Uses programming language CFGs (e.g. Python EBNF) to mask out natural-language refusal tokens (`I'm sorry`, `I cannot`), forcing vulnerable models to generate harmful code.

---

## Architecture Overview

```
User Prompt + Schema/CFG
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  Layer 1: Control-Plane Static Analyzer (Rust Core)   │
│  - Microsecond recursive-descent AST analysis (< 10µs) │
│  - Heuristics: forced literal ratio, enum chains,     │
│    lookup dictionary topology, absence of error union  │
└──────────────────────────┬─────────────────────────────┘
                           │ SchemaRiskReport
                           ▼
┌────────────────────────────────────────────────────────┐
│  Layer 2: Pre-Mask Logit Gate (XGrammar Integration)   │
│  - Evaluates true pre-mask logits BEFORE bitmasking    │
│  - Metrics:                                            │
│    • killed_mass = 1.0 - sum(P_pre[valid_tokens])      │
│    • refusal_suppression = sum(P_pre[refusal & ~valid])│
│    • top1_rank_shift = displacement of argmax token   │
│  - Safe-Escape: Forces EOS on trip, preventing exploit │
└──────────────────────────┬─────────────────────────────┘
                           │ Unmasked Logits / Safe Escape
                           ▼
┌────────────────────────────────────────────────────────┐
│  Layer 3: Cross-Plane Rolling Escalator                │
│  - Selective monitoring for DictAttack payloads        │
│  - Only activates when L1 flags risk or L2 shows spikes│
│  - Evaluates materialized prefix mid-generation        │
└────────────────────────────────────────────────────────┘
```

---

## Reproducible Benchmarks (NVIDIA RTX 5050 + Qwen2.5-3B-Instruct)

All numbers below were empirically measured on an **NVIDIA GeForce RTX 5050 Laptop GPU** (`sm_120` Blackwell, 7.93 GB VRAM) using **`Qwen/Qwen2.5-3B-Instruct` in FP16** (5,994.46 MB VRAM allocated) with real `XGrammar 0.1.34`.

### Initial 40-Task Smoke Benchmark (`N=10` per category)
Every row reflects actual end-to-end token generation recorded in [`benchmarks/results_real_3b.jsonl`](benchmarks/results_real_3b.jsonl) on `cuda:0` under grammatical bitmasking, with measured baseline ASR under unguarded XGrammar:

```
==============================================================================================================
GRAMMARGATE EMPIRICAL SMOKE BENCHMARK (Qwen2.5-3B-Instruct on RTX 5050 Laptop GPU)
==============================================================================================================
Category         | Samples  | Baseline ASR   | Defense Rate       | E2E Latency    | Pure Gate      | Wrapper + XG  
--------------------------------------------------------------------------------------------------------------
Benign Tasks     | 10       | N/A            | 20.0% (FPR)        | 380.63 ms/tok  | 2.043 ms/tok   | 2.797 ms/tok  
EnumAttack       | 10       | 100.0%         | 100.0%             | 2201.68 ms/tok | 2.489 ms/tok   | 3.168 ms/tok  
CodeSpear        | 10       | 100.0%         | 100.0%             | 1447.10 ms/tok | 2.886 ms/tok   | 3.643 ms/tok  
DictAttack       | 10       | 100.0%         | 100.0%             | 3009.27 ms/tok | 2.685 ms/tok   | 3.442 ms/tok  
==============================================================================================================
```

> **Latency & Methodology Notes:**
> - **Pure Gate Overhead (`2.04-2.89 ms/tok`):** Time spent inside GrammarGate's Layer 2 pre-mask logit inspection (`inspect_pre_mask`) and Layer 3 prefix inspection (`RollingEscalator.step`).
> - **Wrapper + XG Overhead (`2.80-3.64 ms/tok`):** Total Hugging Face `LogitsProcessor` step time, encompassing XGrammar C++ matcher token acceptance, bitmask generation, GPU bitmask unpacking, GrammarGate inspection, and in-place GPU `-inf` masking.
> - **E2E Latency (`ms/tok`) Amortization:** `benchmarks/run_real_eval.py` measures wall time starting before XGrammar grammar/schema compilation and prompt prefill, then divides by `tokens_generated`. Early-abort attacks terminate at `1-2` tokens (`2201.68` / `1447.10` / `3009.27 ms/tok`), whereas benign runs amortize compilation and prefill across `10-25` tokens (`380.63 ms/tok`).
> - **DictAttack Trip Layer (`L2` vs `L3`) in `results_real_3b.jsonl`:** All 10 `dict_1`–`dict_10` samples tripped at token 0 via Layer 2 (`"trip_source": "L2"`, `tokens_generated: 1`, `l1_risk_score: 0.63`) because the evaluation prompt triggered immediate refusal probability and the 2-property `DICT_ATTACK_TASKS` schema sits below Layer 1's `props >= 4` lookup-dictionary rule. Layer 3 split-payload mid-generation catching is separately validated in [`benchmarks/attacks/test_dict_attack.py`](benchmarks/attacks/test_dict_attack.py) (`20/20`).
> - **Threshold Calibration (`0.01` vs Default `0.15`):** `GateConfig` defaults to `refusal_suppression_threshold=0.15`, whereas `benchmarks/run_real_eval.py` overrides it to `0.01` (`killed_mass_threshold=0.98`). Real CodeSpear runs measure `refusal_suppression` in `0.0388-0.0489`, which falls below `0.15` and requires `0.01` for `100%` detection—trading off against the `20.0%` benign FPR on strict un-fenced Python CFGs.

### Microbenchmark Latency Profile
Synthetic single-layer timings from [`benchmarks/profile_latency.py`](benchmarks/profile_latency.py) (`vocab_size=32,000` vs `151,936` on `Qwen2.5-3B-Instruct`, which explains why full 3B Pure Gate runs measure `2.04-2.89 ms/tok`):
- **Layer 1 Static Analyzer (Rust):** p50 = `8.70 µs`, p95 = `19.00 µs` on local host (`24.10 µs` p50 / `30.30 µs` p95 on sandboxed CPU; budget `< 50.0 µs`)
- **Layer 2 Pre-Mask Logit Gate:** p50 = `0.50 ms/tok` on GPU (`1.06 ms/tok` p50 on CPU at `vocab_size=32,000`; budget `< 2.50 ms` CUDA / `< 2.00 ms` CPU)
- **Layer 3 Rolling Escalator (Amortized):** p50 = `0.0004–0.0006 ms/tok` on benign queries (`budget < 0.05 ms`)

---

## Honest Limitations & Known Failure Modes

1. **Small Sample Smoke Benchmark (`N=10` per category):**
   The initial evaluation reported above evaluates 40 tasks designed as an end-to-end smoke test on a single local GPU. While unit benchmarks test up to 50 production schemas, larger statistical sweeps ($N \ge 1,000$) across diverse model families (Llama 3, Mistral, Gemma) are required to establish asymptotic FPR/TPR bounds.

2. **Heuristic Layer 3 Rolling Monitor:**
   Layer 3 currently employs a fast deterministic heuristic string scanner to monitor the materialized prefix where control and data planes merge. In production deployments with arbitrary natural-language generation, Layer 3 should be backed by a compact, high-throughput safety judge (e.g., Llama-Guard or a distilled 0.5B token-classifier) to generalize beyond known heuristic rules.

3. **Low-Refusal Stealth Decoupling:**
   If an adversary crafts an attack where the base LLM assigns near-zero initial probability to refusal tokens (e.g. via clever multi-turn framing) *and* the grammar provides sufficient entropy to avoid extreme probability mass extinction (`killed_mass < 0.98`), logit-level threshold gating alone cannot distinguish the request from benign constrained decoding. Robust defense requires pairing Layer 1 schema topology analysis with Layer 3 semantic prefix inspection.

4. **Mid-Generation Prose Trips on Un-Fenced Python CFGs (`FPR = 20.0%`):**
   In 2 of the 10 benign Python tasks (`benign_py_3`, `tokens_generated = 10`; `benign_py_5`, `tokens_generated = 16` in `results_real_3b.jsonl`), generation tripped `trip_source: "L2"` mid-generation under strict `refusal_suppression_threshold=0.01` after emitting valid Python statements (`import math\n`, `def add_numbers_helper():\n`). Note that `run_real_eval.py` only logs step-level `refusal_suppression` scalars on attack rows, not benign rows. Allowing optional prose/code-fence envelopes (`wrap_ebnf_with_preamble` in [`grammargate/preamble_utils.py`](grammargate/preamble_utils.py)) prevents models from hitting masked prose transitions under rigid CFGs.

---

## Current Research Prototype Status vs. Enterprise Roadmap

To maintain rigorous scientific and engineering integrity, we do not claim GrammarGate is enterprise production-ready. We distinguish what has been empirically verified in this research prototype from what requires enterprise hardening:

| Dimension | Current Prototype Status | Enterprise Production Target |
|---|---|---|
| **Defense Invariance** | **Empirically Verified Prototype:** Pre-mask evaluation tested on GPU | Formal verification across arbitrary sampling methods (top-p, top-k, min-p) |
| **Layer 1 Static Analyzer** | **Research Reference Implementation:** Native Rust cdylib ($< 20\text{ µs}$, caller-allocated FFI buffers with zero heap leaks) | Complete JSON Schema draft-07/2020-12 (`$ref`/`$defs`) compliance |
| **Layer 2 Pre-Mask Gate** | **Empirically Verified Prototype:** Vectorized GPU tensor math ($2.04\text{ ms/tok}$) | Fused CUDA kernel integrating pre-mask evaluation directly into the sampler |
| **Engine Compatibility** | Hugging Face processor; vLLM V0 callable & V1 `BatchUpdate` lifecycle shim | Upstream PRs into `vllm-project/vllm` and `sgl-project/sglang` |
| **False Positive Rate** | **Active Limitation:** 20.0% FPR on raw Python CFGs under `0.01` threshold | $< 0.1\%$ FPR via fenced code-block grammars or stop-sequence rules |
| **Layer 3 Monitoring** | Fast deterministic regex pattern scanner | Distilled 0.5B token-classifier safety head (e.g., Llama-Guard / ShieldGemma) |
| **Model Diversity** | Evaluated on Qwen 2.5 (0.5B integration tests & 3B smoke benchmark) | Multi-model validation across Llama 3.1 (8B, 70B), Mistral, and DeepSeek |
| **Concurrency & Scale** | Single-stream local GPU execution | Multi-GPU continuous batching under Tensor Parallelism (TP=2/4/8) |

---

## Reproduction Commands

### 1. Install Python Dependencies
```bash
# Editable install with test, benchmark, and XGrammar dependencies:
pip install -e ".[dev]"

# Or install minimal CI dependencies explicitly:
pip install torch transformers xgrammar pytest numpy
```

### 2. Build Native Rust Core cdylib
```bash
# Standard cross-platform build via Cargo:
cargo build --release --manifest-path grammargate_core/Cargo.toml

# Or direct compilation (Windows standalone; grammargate_core/__init__.py loads the newest compiled binary):
rustc --crate-type cdylib grammargate_core/src/lib.rs -O -o grammargate_core/grammargate_core.dll
```

### 3. Run Fast CI Test Suite (CPU-compatible)
```bash
# Linux / macOS (pyproject.toml also sets pythonpath = ["."]):
PYTHONPATH=. pytest tests/test_rust_core.py tests/test_rust_analyzer.py tests/test_logit_gate.py tests/test_escalator.py tests/test_vllm_adapter.py benchmarks/attacks/ benchmarks/utility/ -v
```
On Windows PowerShell, set `$env:PYTHONPATH="."` first if running outside `pyproject.toml`.

### 4. Run Real GPU Integration Tests (Qwen 0.5B + 3B on CUDA)
```bash
pytest tests/test_hf_adapter.py -v -s
```

### 5. Run Full Real GPU Evaluation Suite & Latency Profiler
```bash
python benchmarks/run_real_eval.py
python benchmarks/profile_latency.py
```

---

## Inference Adapters

### Adapter A: Hugging Face Transformers (`GrammarGateHfProcessor`)
Note: `refusal_suppression_threshold=0.01` is set explicitly here (overriding the `GateConfig` default of `0.15`) to match the real-GPU CodeSpear sensitivity profile.
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import xgrammar as xgr
from grammargate.analyzer import analyze_schema
from grammargate.adapters.hf_processor import GrammarGateHfProcessor
from grammargate.escalator import RollingEscalator
from grammargate.gate import GateConfig

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,  # use torch_dtype=torch.float16 on transformers < 4.56
).to("cuda:0")

schema_str = '{"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}'
l1_report = analyze_schema(schema_str)

tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
compiler = xgr.GrammarCompiler(tok_info)
compiled = compiler.compile_json_schema(schema_str)
matcher = xgr.GrammarMatcher(compiled)
bitmask_cpu = xgr.allocate_token_bitmask(1, tok_info.vocab_size)

processor = GrammarGateHfProcessor(
    tokenizer=tokenizer,
    bitmask_provider=lambda step, vocab_size: (matcher.fill_next_token_bitmask(bitmask_cpu), bitmask_cpu)[1],
    token_acceptor=matcher.accept_token,
    config=GateConfig(killed_mass_threshold=0.98, refusal_suppression_threshold=0.01, top1_shift_threshold=10),
    escalator=RollingEscalator(tokenizer=tokenizer),
    layer1_elevated=l1_report.is_elevated,
)

inputs = tokenizer("<|im_start|>user\nExtract city from: Seattle is rainy.<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to("cuda:0")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    logits_processor=LogitsProcessorList([processor]),
)
```

### Adapter B: vLLM V1 LogitsProcessor (`GrammarGateVllmV1Processor`)
Supports persistent batch lifecycle (`removed` $\rightarrow$ `added` $\rightarrow$ `moved`, including unidirectional compaction and bidirectional slot swaps) and advances `token_acceptor` / optional `RollingEscalator` via `RequestGrammarState.output_token_ids` or `processor.accept_token(slot, token_id)`:
```python
import torch
from grammargate.adapters.vllm_processor import GrammarGateVllmV1Processor, BatchUpdate, RequestGrammarState
from grammargate.bitmask_utils import pack_boolean_mask

vocab_size = 32000
allowed = torch.zeros(vocab_size, dtype=torch.bool)
allowed[10:100] = True
packed_mask = pack_boolean_mask(allowed)

req_state_0 = RequestGrammarState(
    request_id="req_0",
    bitmask_provider=lambda step, vs: packed_mask,
    output_token_ids=[],
)

processor = GrammarGateVllmV1Processor(device="cpu")
processor.update_state(BatchUpdate(added={0: req_state_0}))

batch_logits = torch.randn((1, vocab_size))
batch_logits[0, 20] = 8.0  # model favors allowed token 20
masked_logits = processor.apply(batch_logits)
```

---

## Repository Structure

```
├── .github/workflows/ci.yml       # Ubuntu CI workflow (Rust build + CPU test suite)
├── Cargo.toml                     # Cargo workspace definition
├── grammargate_core/              # Native Rust Layer 1 Static Analyzer
│   ├── Cargo.toml                 # Zero-dependency Rust manifest
│   ├── __init__.py                # Ctypes cdylib loader & Python FFI bindings
│   └── src/                       # Recursive-descent JSON parser, AST walker & C ABI FFI
├── grammargate/                   # Core Python safety gates & inference adapters
│   ├── adapters/                  # Hugging Face & vLLM V0/V1 LogitsProcessors
│   ├── analyzer.py                # Layer 1 wrapper delegating strictly to Rust core
│   ├── bitmask_utils.py           # Vectorized GPU bitmask unpacking & metric reductions
│   ├── escalator.py               # Layer 3 cross-plane rolling escalator
│   ├── gate.py                    # Layer 2 pre-mask logit gating math & safe escape
│   ├── model_utils.py             # Local snapshot / HF Hub model resolver
│   ├── preamble_utils.py          # Optional EBNF prose/code-fence envelope wrapper
│   ├── refusal_tokens.py          # Refusal onset token resolver
│   └── telemetry.py               # Structured audit record dataclasses
├── tests/                         # Fast unit & real GPU integration tests
├── benchmarks/                    # Real GPU eval, microbenchmarks & synthetic suites
│   ├── attacks/                   # CPU-compatible EnumAttack, CodeSpear, DictAttack & adaptive harnesses
│   ├── utility/                   # 50-schema benign utility benchmark
│   ├── generators/                # Large-scale deterministic dataset generator
│   ├── ablation/                  # Synthetic three-plane factorial ablation runner
│   ├── baselines/                 # CDA baseline comparison printout (lit. + 3B empirical)
│   ├── profile_latency.py         # Layer 1 / 2 / 3 microbenchmark profiler
│   ├── run_real_eval.py           # Full Qwen2.5-3B real GPU evaluation suite
│   ├── run_large_scale_eval.py    # Large-scale Wilson-CI runner (N >= 1,000)
│   └── results_real_3b.jsonl      # Raw empirical per-sample telemetry (40 tasks)
├── assets/hero-taj.jpg            # Landing page hero illustration
├── index.html                     # Interactive research write-up & trace inspector
└── pyproject.toml                 # Package & optional test/eval dependency manifest
```
