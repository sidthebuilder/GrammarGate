"""
Modal GPU deployment wrapper for GrammarGate v1.0 Multi-Model Evaluation on A10G / H100.

Enforces zero fallbacks (`--require-native-l1` and `--require-xgrammar`), builds the native
Rust `cdylib`, verifies `cargo test`, downloads public HF corpora & models into a persistent
Modal Volume, runs `benchmarks/run_principal_eval.py` across ungated model families, and
saves local JSON/JSONL artifacts into `benchmarks/results_v1/`.
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
    "mistralai/Mistral-7B-Instruct-v0.3"
)

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
            "torch>=2.4.0",
            "transformers>=4.48.0",
            "accelerate>=0.34.0",
            "xgrammar>=0.2.9",
            "vllm>=0.7.0",
            "datasets>=2.20.0",
            "huggingface_hub>=0.25.0",
            "pyarrow>=15.0.0",
            "sentence-transformers>=2.6.0",
            "scipy>=1.11.0",
            "pytest",
            "numpy",
        )
        .add_local_dir(".", remote_path="/workspace")
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
        seed: int,
    ) -> dict:
        import sys
        sys.path.insert(0, "/workspace")

        from grammargate.native_checks import assert_native_l1_ready, assert_xgrammar_ready
        assert_native_l1_ready()
        assert_xgrammar_ready()

        from benchmarks.run_principal_eval import evaluate_model

        res_dir = Path("/workspace/benchmarks/results_v1")
        summary = evaluate_model(
            model_id=model_id,
            n_jsb=n_jsb,
            n_humaneval=n_humaneval,
            n_jbb=n_jbb,
            jbb_split=jbb_split,
            max_new_tokens=max_new_tokens,
            decode_regimes=decode_regimes,
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
        timeout=10800,
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
        seed: int = 2026,
    ) -> dict:
        return _run_principal_worker(
            model_id, n_jsb, n_humaneval, n_jbb, jbb_split, max_new_tokens, decode_regimes, seed
        )

    @app.function(
        image=image,
        gpu="H100",
        timeout=7200,
        volumes={"/cache/hf": hf_cache_vol},
    )
    def run_modal_principal_h100(
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        n_jsb: int = 100,
        n_humaneval: int = 100,
        n_jbb: int = 100,
        jbb_split: str = "all",
        max_new_tokens: int = 48,
        decode_regimes: str = "all",
        seed: int = 2026,
    ) -> dict:
        return _run_principal_worker(
            model_id, n_jsb, n_humaneval, n_jbb, jbb_split, max_new_tokens, decode_regimes, seed
        )

    @app.function(
        image=image,
        gpu="A10G",
        timeout=3600,
        volumes={"/cache/hf": hf_cache_vol},
    )
    def run_modal_ablation(
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        n_per_category: int = 50,
        multilingual_calibration: bool = False,
    ):
        import sys
        sys.path.insert(0, "/workspace")

        from grammargate.native_checks import assert_native_l1_ready, assert_xgrammar_ready
        assert_native_l1_ready()
        assert_xgrammar_ready()

        from benchmarks.run_ablation_and_adaptive import run_ablation
        safe_slug = model_id.replace("/", "_")
        out_path = f"/workspace/benchmarks/results_modal_{safe_slug}.jsonl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        res = run_ablation(
            model_id=model_id,
            n_per_category=max(50, int(n_per_category)),
            mask_backend="xgrammar",
            max_new_tokens=25,
            output_jsonl=out_path,
            require_native_l1=True,
            require_xgrammar=True,
            multilingual_calibration=multilingual_calibration,
        )
        hf_cache_vol.commit()
        return res

    @app.local_entrypoint()
    def main(
        models: str = DEFAULT_PRINCIPAL_MODELS,
        gpu: str = "A10G",
        n_jsb: int = 100,
        n_humaneval: int = 100,
        n_jbb: int = 100,
        jbb_split: str = "all",
        max_new_tokens: int = 48,
        decode_regimes: str = "all",
        seed: int = 2026,
        legacy_ablation: bool = False,
    ):
        targets = [m.strip() for m in models.split(",") if m.strip()]
        if legacy_ablation:
            for target in targets:
                res = run_modal_ablation.remote(model_id=target)
                print(f"Legacy modal ablation ({target}):", res)
            return

        worker = run_modal_principal_h100 if gpu.upper() == "H100" else run_modal_principal_a10g
        args_list = [
            (m, n_jsb, n_humaneval, n_jbb, jbb_split, max_new_tokens, decode_regimes, seed)
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
