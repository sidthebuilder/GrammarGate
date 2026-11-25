"""
Cloud GPU Runner for GrammarGate Ablation & Adaptive Benchmarks (Modal + vLLM / XGrammar).

Usage (from Linux/macOS/WSL with `modal` CLI configured):
    modal run benchmarks/modal_vllm_eval.py --model-id Qwen/Qwen2.5-3B-Instruct --n-per-category 50
"""

from __future__ import annotations

try:
    import modal
except ImportError:
    modal = None


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("curl", "build-essential")
        .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
        .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
        .pip_install("torch>=2.4.0", "transformers>=4.48.0", "xgrammar>=0.1.34", "vllm>=0.7.0", "pytest", "numpy")
        .add_local_dir(".", remote_path="/workspace")
        .run_commands(
            "cd /workspace && cargo build --release --manifest-path grammargate_core/Cargo.toml",
            "cd /workspace && mkdir -p grammargate_core && find target -name 'libgrammargate_core.so' -exec cp {} grammargate_core/ \\;",
        )
    )

    app = modal.App("grammargate-eval")

    @app.function(image=image, gpu="A10G", timeout=1800)
    def run_modal_ablation(model_id: str = "Qwen/Qwen2.5-3B-Instruct", n_per_category: int = 50):
        import sys
        sys.path.insert(0, "/workspace")
        from benchmarks.run_ablation_and_adaptive import run_ablation
        return run_ablation(
            model_id=model_id,
            n_per_category=n_per_category,
            mask_backend="xgrammar",
            max_new_tokens=25,
            output_jsonl="/workspace/benchmarks/results_modal_ablation.jsonl",
        )

    @app.local_entrypoint()
    def main(model_id: str = "Qwen/Qwen2.5-3B-Instruct", n_per_category: int = 50):
        summary = run_modal_ablation.remote(model_id=model_id, n_per_category=n_per_category)
        print("Modal ablation summary:", summary)
