# GrammarGate: Inference-Time Safety Gate for Constrained Decoding Attacks (CDA)

[![CI](https://github.com/sidthebuilder/grammargate/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Interactive Showcase](https://img.shields.io/badge/Interactive%20Report-Live%20Demo-emerald)](index.html)

GrammarGate is an inference-time safety research prototype and reference architecture defending LLMs against Constrained Decoding Attacks (CDA), specifically EnumAttack, DictAttack, and CodeSpear, without retraining model weights.

> **Explore the Research:** View the full interactive findings, live architecture diagram, and token extinction trace inspector in [index.html](index.html).



## Threat Model & Core Problem

Structured outputs and grammar-constrained decoding engines (`XGrammar`, `Outlines`, `llguidance`) enforce schemas and grammars by masking disallowed tokens to $-\infty$ at the logit layer. Attackers exploit this logit-level masking:
1. **EnumAttack** (CCS '26 / arXiv:2503.24191): Forces harmful prefixes into the output via strict JSON Schema `enum`/`const` rules while the user prompt appears benign.
2. **DictAttack & Interleaved DictAttack** (CCS '26): Splits malicious payloads across benign dictionary mappings, materializing harmful semantics only mid-generation when control and data planes merge.
3. **CodeSpear** (CodeSpear, 2026): Uses programming language CFGs (e.g. Python EBNF) to mask out natural-language refusal tokens (`I'm sorry`, `I cannot`), forcing vulnerable models to generate harmful code.

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
Every row reflects actual end-to-end token generation on `cuda:0` under grammatical bitmasking, with measured baseline ASR under unguarded XGrammar:

```
==============================================================================================================
GRAMMARGATE EMPIRICAL SMOKE BENCHMARK (Qwen2.5-3B-Instruct on RTX 5050 Laptop GPU)
==============================================================================================================
Category         | Samples  | Baseline ASR   | Defense Rate       | E2E Latency    | Pure Gate      | Wrapper + XG  
--------------------------------------------------------------------------------------------------------------
Benign Tasks     | 10       | N/A            | 20.0% (FPR)        | 380.63 ms/tok  | 2.043 ms/tok   | 2.797 ms/tok  
EnumAttack       | 10       | 100.0%         | 100.0%             | 2201.68 ms/tok | 2.489 ms/tok   | 3.168 ms/tok  
CodeSpear        | 10       | 100.0%         | 100.0%             | 1447.10 ms/tok | 2.886 ms/tok   | 3.643 ms/tok  
DictAttack (L3)  | 10       | 100.0%         | 100.0%             | 3009.27 ms/tok | 2.685 ms/tok   | 3.442 ms/tok  
==============================================================================================================
```

> **Latency Breakdown:**
> - **Pure Gate Overhead (2.04–2.89 ms/tok):** Time spent exclusively inside GrammarGate's Layer 2 pre-mask logit inspection (`inspect_pre_mask`) and Layer 3 prefix inspection (`RollingEscalator.step`).
> - **Wrapper + XG Overhead (2.80–3.64 ms/tok):** Total Hugging Face `LogitsProcessor` step time, encompassing XGrammar C++ matcher token acceptance, bitmask generation, GPU bitmask unpacking, GrammarGate inspection, and in-place GPU `-inf` masking.

### Microbenchmark Latency Profile
- **Layer 1 Static Analyzer (Rust):** p50 = 8.70 µs, p95 = 19.00 µs (Budget: $< 50.0$ µs)
- **Layer 2 Pre-Mask Logit Gate (GPU):** p50 = 0.50 ms per token
- **Layer 3 Rolling Escalator (Amortized):** p50 = 0.0004 ms per token on benign queries

---

## Honest Limitations & Known Failure Modes

1. **Small Sample Smoke Benchmark (`N=10` per category):**
   The initial evaluation reported above evaluates 40 tasks designed as an end-to-end smoke test on a single local GPU. While unit benchmarks test up to 50 production schemas, larger statistical sweeps ($N \ge 1,000$) across diverse model families (Llama 3, Mistral, Gemma) are required to establish asymptotic FPR/TPR bounds.

2. **Heuristic Layer 3 Rolling Monitor:**
   Layer 3 currently employs a fast deterministic heuristic string scanner to monitor the materialized prefix where control and data planes merge. In production deployments with arbitrary natural-language generation, Layer 3 should be backed by a compact, high-throughput safety judge (e.g., Llama-Guard or a distilled 0.5B token-classifier) to generalize beyond known heuristic rules.

3. **Low-Refusal Stealth Decoupling:**
   If an adversary crafts an attack where the base LLM assigns near-zero initial probability to refusal tokens (e.g. via clever multi-turn framing) *and* the grammar provides sufficient entropy to avoid extreme probability mass extinction (`killed_mass < 0.98`), logit-level threshold gating alone cannot distinguish the request from benign constrained decoding. Robust defense requires pairing Layer 1 schema topology analysis with Layer 3 semantic prefix inspection.

4. **Mid-Generation Natural Prose False Trips on Strict Grammars (FPR = 20.0%):**
   In 2 of the 10 benign Python tasks (`benign_py_3` at step 10 and `benign_py_5` at step 16 in `results_real_3b.jsonl`), the model generated valid statements (`import math\n` or `def add_numbers_helper():\n`) and then attempted conversational English commentary (tokens starting with `"I"` / `"In"`). Because the rigid CFG forced repeated code statements and masked all prose, suppression of the single-token prefix `"I"` (which is part of multi-word refusal onsets) exceeded the threshold, triggering Layer 2 `REFUSAL_TOKEN_SUPPRESSION` mid-generation rather than at token 0.

---

## Current Research Prototype Status vs. Enterprise Roadmap

To maintain rigorous scientific and engineering integrity, we do not claim GrammarGate is enterprise production-ready. We distinguish what has been empirically verified in this research prototype from what requires enterprise hardening:

| Dimension | Current Prototype Status | Enterprise Production Target |
|---|---|---|
| **Defense Invariance** | **Empirically Verified Prototype:** Pre-mask evaluation tested on GPU | Formal verification across arbitrary sampling methods (top-p, top-k, min-p) |
| **Layer 1 Static Analyzer** | **Research Reference Implementation:** Native Rust cdylib ($< 20\text{ µs}$, caller-allocated FFI buffers with zero heap leaks) | Complete JSON Schema draft-07/2020-12 specification compliance |
| **Layer 2 Pre-Mask Gate** | **Empirically Verified Prototype:** Vectorized GPU tensor math ($2.04\text{ ms/tok}$) | Fused CUDA kernel integrating pre-mask evaluation directly into the sampler |
| **Engine Compatibility** | Working Hugging Face & vLLM V1 interface shims | Upstream PRs into `vllm-project/vllm` and `sgl-project/sglang` |
| **False Positive Rate** | **Active Limitation:** 20.0% FPR on raw Python CFGs due to mid-generation prose suppression | $< 0.1\%$ FPR via fenced code-block grammars or stop-sequence rules |
| **Layer 3 Monitoring** | Fast deterministic regex pattern scanner | Distilled 0.5B token-classifier safety head (e.g., Llama-Guard / ShieldGemma) |
| **Model Diversity** | Evaluated on Qwen 2.5 (0.5B & 3B) on local GPU | Multi-model validation across Llama 3.1 (8B, 70B), Mistral, and DeepSeek |
| **Concurrency & Scale** | Single-stream local GPU execution | Multi-GPU continuous batching under Tensor Parallelism (TP=2/4/8) |

---

## Reproduction Commands

### 1. Build Native Rust Core cdylib
```powershell
# Standard cross-platform build via Cargo:
cargo build --release --manifest-path grammargate_core/Cargo.toml

# Or direct compilation (Windows standalone):
rustc --crate-type cdylib grammargate_core/src/lib.rs -O -o grammargate_core/grammargate_core.dll
```

### 2. Run Fast CI Test Suite (CPU-compatible)
```powershell
$env:PYTHONPATH="."
pytest tests/test_rust_core.py tests/test_rust_analyzer.py tests/test_logit_gate.py tests/test_escalator.py tests/test_vllm_adapter.py benchmarks/attacks/ benchmarks/utility/ -v
```

### 3. Run Real GPU Integration Tests (Qwen 0.5B + 3B on CUDA)
```powershell
$env:PYTHONPATH="."
pytest tests/test_hf_adapter.py -v -s
```

### 4. Run Full Real GPU Evaluation Suite
```powershell
$env:PYTHONPATH="."
python benchmarks/run_real_eval.py
```

---

## Inference Adapters

### Adapter A: Hugging Face Transformers (`GrammarGateHfProcessor`)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import xgrammar as xgr
from grammargate.adapters.hf_processor import GrammarGateHfProcessor
from grammargate.gate import GateConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.float16).to("cuda:0")

tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
compiler = xgr.GrammarCompiler(tok_info)
compiled = compiler.compile_json_schema(schema_str)
matcher = xgr.GrammarMatcher(compiled)
bitmask_cpu = xgr.allocate_token_bitmask(1, tok_info.vocab_size)

processor = GrammarGateHfProcessor(
    tokenizer=tokenizer,
    bitmask_provider=lambda step, vocab_size: (matcher.fill_next_token_bitmask(bitmask_cpu), bitmask_cpu)[1],
    token_acceptor=matcher.accept_token,
    config=GateConfig(refusal_suppression_threshold=0.01),
)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    logits_processor=LogitsProcessorList([processor]),
)
```

### Adapter B: vLLM V1 LogitsProcessor (`GrammarGateVllmV1Processor`)
Supports persistent batch lifecycle: `removed` $\rightarrow$ `added` $\rightarrow$ `moved` (unidirectional compaction and bidirectional slot swaps).
```python
from grammargate.adapters.vllm_processor import GrammarGateVllmV1Processor, BatchUpdate, RequestGrammarState

processor = GrammarGateVllmV1Processor(device="cuda:0")

# Lifecycle update: adding requests
processor.update_state(BatchUpdate(added={0: req_state_0, 1: req_state_1}))

# Applying batched pre-mask logits inspection and safe-escape
masked_logits = processor.apply(batch_logits)
```

---

## Repository Structure

```
├── .github/workflows/ci.yml       # Ubuntu CI workflow (Rust build + CPU test suite)
├── grammargate_core/              # Native Rust Layer 1 Static Analyzer
│   ├── Cargo.toml                 # Zero-dependency Rust manifest
│   └── src/                       # Recursive-descent AST parser & C ABI FFI
├── grammargate/                   # Core Python safety gates & inference adapters
│   ├── adapters/                  # Hugging Face & vLLM V1 LogitsProcessors
│   ├── gate.py                    # Layer 2 pre-mask logit gating math
│   ├── escalator.py               # Layer 3 cross-plane rolling escalator
│   ├── bitmask_utils.py           # Vectorized GPU bitmask manipulation
│   └── model_utils.py             # Model path & snapshot resolver
├── tests/                         # Fast unit & real GPU integration tests
├── benchmarks/                    # Real GPU 40-task benchmark runner & raw traces
│   ├── run_real_eval.py           # Full GPU evaluation suite
│   └── results_real_3b.jsonl      # Raw empirical per-sample telemetry
├── index.html                     # Interactive research write-up & trace inspector
└── pyproject.toml                 # Package manifest
```
