"""
Modal GPU deployment wrapper for GrammarGate v1.0 Multi-Model Evaluation on A10G / H100.

Enforces zero fallbacks (native Rust Layer 1 + XGrammar 0.2.9), builds and tests the native
Rust `cdylib` inside the image, caches HF models in a persistent Modal Volume, supports a
fast `--smoke` preflight run, and collects raw JSONL traces + statistical summaries under
`benchmarks/results_v1/`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None


DEFAULT_PRINCIPAL_MODELS = (
    "Qwen/Qwen3-4B-Instruct-2507,"
    "microsoft/Phi-4-mini-instruct,"
    "allenai/OLMo-2-1124-7B-Instruct"
)
A10G_HOURLY_USD = 1.10

if modal is not None:
    app = modal.App("grammargate-v1-principal-eval")
    hf_cache_vol = modal.Volume.from_name("grammargate-hf-cache", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("curl", "build-essential")
        .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
        .env({
            "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
            "HF_HOME": "/cache/hf",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        })
        .pip_install(
            "torch>=2.5.0",
            "transformers>=4.51.0",
            "accelerate>=0.34.0",
            "xgrammar==0.2.9",
            "datasets>=2.20.0",
            "huggingface_hub>=0.25.0",
            "pyarrow>=15.0.0",
            "sentence-transformers>=2.6.0",
            "scipy>=1.11.0",
            "pytest",
            "numpy",
        )
        .add_local_dir(
            ".",
            remote_path="/workspace",
            ignore=[
                "hf_cache",
                "hf_cache/**",
                ".git",
                ".git/**",
                ".pytest_cache",
                ".pytest_cache/**",
                "**/__pycache__",
                "target",
                "target/**",
                "grammargate_core/target",
                "grammargate_core/target/**",
                "benchmarks/results_*.jsonl",
            ],
        )
        .run_commands(
            "cd /workspace && cargo test --manifest-path grammargate_core/Cargo.toml --verbose",
            "cd /workspace && cargo build --release --manifest-path grammargate_core/Cargo.toml",
            "cd /workspace && cp grammargate_core/target/release/libgrammargate_core.so grammargate/grammargate_core.so",
            "cd /workspace && pytest tests/test_rust_core.py tests/test_rust_analyzer.py tests/test_native_xgrammar_ci.py -v",
        )
    )

    def _run_principal_worker(
        model_id: str,
        n_jsb: int,
        n_humaneval: int,
        n_jbb: int,
        jbb_split: str,
        max_new_tokens: int,
        decode_regimes: str,
        stochastic_n: int,
        seed: int,
    ) -> dict:
        import sys
        sys.path.insert(0, "/workspace")

        from benchmarks.run_principal_eval import evaluate_model, get_runtime_provenance

        prov = get_runtime_provenance()
        print(
            "[Modal Startup] versions:",
            json.dumps({k: prov[k] for k in ("torch", "transformers", "xgrammar", "device_name", "grammargate_core_sha256")}),
        )

        res_dir = Path("/workspace/benchmarks/results_v1")
        summary = evaluate_model(
            model_id=model_id,
            n_jsb=n_jsb,
            n_humaneval=n_humaneval,
            n_jbb=n_jbb,
            jbb_split=jbb_split,
            max_new_tokens=max_new_tokens,
            decode_regimes=decode_regimes,
            stochastic_n=stochastic_n,
            seed=seed,
            results_dir=res_dir,
        )
        hf_cache_vol.commit()

        raw_files = {}
        for p in res_dir.glob("raw_*.jsonl"):
            raw_files[p.name] = p.read_text(encoding="utf-8")
        manifest_file = res_dir / "dataset_manifest.json"
        manifest_text = manifest_file.read_text(encoding="utf-8") if manifest_file.exists() else ""

        return {
            "model_id": model_id,
            "summary": summary,
            "raw_files": raw_files,
            "manifest_json": manifest_text,
        }

    @app.function(
        image=image,
        gpu="A10G",
        timeout=14400,
        volumes={"/cache/hf": hf_cache_vol},
    )
    def run_modal_principal_a10g(
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        n_jsb: int = 100,
        n_humaneval: int = 100,
        n_jbb: int = 100,
        jbb_split: str = "all",
        max_new_tokens: int = 48,
        decode_regimes: str = "all",
        stochastic_n: int = 25,
        seed: int = 2026,
    ) -> dict:
        return _run_principal_worker(
            model_id, n_jsb, n_humaneval, n_jbb, jbb_split, max_new_tokens, decode_regimes, stochastic_n, seed
        )

    @app.function(
        image=image,
        gpu="H100",
        timeout=10800,
        volumes={"/cache/hf": hf_cache_vol},
    )
    def run_modal_principal_h100(
        model_id: str = "allenai/OLMo-2-1124-7B-Instruct",
        n_jsb: int = 100,
        n_humaneval: int = 100,
        n_jbb: int = 100,
        jbb_split: str = "all",
        max_new_tokens: int = 48,
        decode_regimes: str = "all",
        stochastic_n: int = 25,
        seed: int = 2026,
    ) -> dict:
        return _run_principal_worker(
            model_id, n_jsb, n_humaneval, n_jbb, jbb_split, max_new_tokens, decode_regimes, stochastic_n, seed
        )

    @app.local_entrypoint()
    def main(
        models: str = DEFAULT_PRINCIPAL_MODELS,
        gpu: str = "A10G",
        smoke: bool = False,
        n_jsb: int = 100,
        n_humaneval: int = 100,
        n_jbb: int = 100,
        jbb_split: str = "all",
        max_new_tokens: int = 48,
        decode_regimes: str = "all",
        stochastic_n: int = 25,
        seed: int = 2026,
    ):
        worker = run_modal_principal_h100 if gpu.upper() == "H100" else run_modal_principal_a10g

        if smoke:
            smoke_model = "Qwen/Qwen3-4B-Instruct-2507"
            out = worker.remote(
                model_id=smoke_model,
                n_jsb=5,
                n_humaneval=5,
                n_jbb=5,
                jbb_split=jbb_split,
                max_new_tokens=32,
                decode_regimes="greedy",
                stochastic_n=0,
                seed=seed,
            )
            summary = out["summary"]
            reg_key = next(iter(summary["regimes"].keys()))
            modes = summary["regimes"][reg_key]["modes"]
            comb = modes.get("Combined", {})
            hs_m = modes.get("HS-Gate-Routed", {})
            cstats = summary["compute_stats"]
            est_hours_seq = cstats["projected_full_sweep_hours_sequential"]
            est_cost_usd = round(est_hours_seq * A10G_HOURLY_USD, 2)
            print("=== MODAL SMOKE REPORT ===")
            print("Provenance:", json.dumps(summary["provenance"], indent=2))
            print("Native Rust L1 (µs):", summary["rust_l1_microbenchmark"])
            print("Combined latency (ms/tok):", comb.get("latency"))
            print("Combined pure_gate_latency (ms/tok):", comb.get("pure_gate_latency"))
            print("HS-Gate-Routed pure_gate_latency (ms/tok):", hs_m.get("pure_gate_latency"))
            print("Combined component means (ms/tok):", comb.get("component_means_ms_tok"))
            print("Compute projection:", cstats, f"est_A10G_cost_usd~=${est_cost_usd}")
            local_results_dir = Path("benchmarks") / "results_v1"
            local_results_dir.mkdir(parents=True, exist_ok=True)
            with open(local_results_dir / "smoke_qwen3_4b.json", "w", encoding="utf-8") as sf:
                json.dump(summary, sf, indent=2)
            return

        targets = [m.strip() for m in models.split(",") if m.strip()]
        args_list = [
            (m, n_jsb, n_humaneval, n_jbb, jbb_split, max_new_tokens, decode_regimes, stochastic_n, seed)
            for m in targets
        ]
        local_results_dir = Path("benchmarks") / "results_v1"
        local_results_dir.mkdir(parents=True, exist_ok=True)

        consolidated = {"version": "1.0.0", "gpu": gpu.upper(), "models": {}}
        for out in worker.starmap(args_list):
            model_id = out["model_id"]
            slug = model_id.replace("/", "__").replace(" ", "_")
            consolidated["models"][model_id] = out["summary"]
            with open(local_results_dir / f"summary_{slug}.json", "w", encoding="utf-8") as sf:
                json.dump(out["summary"], sf, indent=2)
            for fname, content in out["raw_files"].items():
                (local_results_dir / fname).write_text(content, encoding="utf-8")
            if out.get("manifest_json"):
                (local_results_dir / "dataset_manifest.json").write_text(out["manifest_json"], encoding="utf-8")
            print(f"Saved Modal v1.0 results for {model_id}")

        with open(local_results_dir / "principal_eval_report.json", "w", encoding="utf-8") as rf:
            json.dump(consolidated, rf, indent=2)
        print(f"Consolidated report saved to {local_results_dir / 'principal_eval_report.json'}")
