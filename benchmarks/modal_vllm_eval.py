"""
Cloud GPU Runner for GrammarGate Ablation & Adaptive Benchmarks (Modal + XGrammar).

Usage (from Linux/macOS/WSL with `modal` CLI configured):
    modal run benchmarks/modal_vllm_eval.py --model-id Qwen/Qwen2.5-3B-Instruct --n-per-category 50
    modal run benchmarks/modal_vllm_eval.py --models Qwen/Qwen2.5-3B-Instruct,google/gemma-2-2b-it --n-per-category 50
"""

from __future__ import annotations

try:
    import modal
except ImportError:
    modal = None


DEFAULT_MODEL_FAMILIES = "Qwen/Qwen2.5-3B-Instruct,meta-llama/Llama-3.1-8B-Instruct"

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("curl", "build-essential")
        .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
        .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
        .pip_install("torch>=2.4.0", "transformers>=4.48.0", "xgrammar>=0.1.34", "vllm>=0.7.0", "pytest", "numpy")
        .add_local_dir(".", remote_path="/workspace")
        .run_commands(
            "cd /workspace && cargo test --manifest-path grammargate_core/Cargo.toml --verbose",
            "cd /workspace && cargo build --release --manifest-path grammargate_core/Cargo.toml",
            "cd /workspace && mkdir -p grammargate_core && find target -name 'libgrammargate_core.so' -exec cp {} grammargate_core/ \\;",
        )
    )

    app = modal.App("grammargate-eval")

    @app.function(image=image, gpu="A10G", timeout=3600)
    def run_modal_ablation(
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        n_per_category: int = 50,
        multilingual_calibration: bool = False,
    ):
        import os
        import sys
        sys.path.insert(0, "/workspace")

        # Fail loudly if native Rust L1 analyzer or real XGrammar cannot load
        import xgrammar as xgr  # noqa: F401
        from grammargate.analyzer import analyze_schema
        test_rep = analyze_schema('{"type":"object","properties":{"status":{"type":"string","enum":["ok","err"]}}}')
        if test_rep.is_elevated:
            raise RuntimeError("Benign multi-option enum schema unexpectedly elevated in Rust L1 sanity check")

        from benchmarks.run_ablation_and_adaptive import run_ablation
        safe_slug = model_id.replace("/", "_")
        out_path = f"/workspace/benchmarks/results_modal_{safe_slug}.jsonl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return run_ablation(
            model_id=model_id,
            n_per_category=max(50, int(n_per_category)),
            mask_backend="xgrammar",
            max_new_tokens=25,
            output_jsonl=out_path,
            require_native_l1=True,
            require_xgrammar=True,
            multilingual_calibration=multilingual_calibration,
        )

    @app.local_entrypoint()
    def main(
        model_id: str | None = None,
        models: str = "Qwen/Qwen2.5-3B-Instruct",
        n_per_category: int = 50,
        multilingual_calibration: bool = False,
    ):
        targets = [model_id] if model_id else [m.strip() for m in models.split(",") if m.strip()]
        results = {}
        for target in targets:
            summary = run_modal_ablation.remote(
                model_id=target,
                n_per_category=n_per_category,
                multilingual_calibration=multilingual_calibration,
            )
            results[target] = summary
            print(f"Modal ablation summary ({target}):", summary)
