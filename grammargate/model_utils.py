"""
Model resolution utilities for finding local cached weights or falling back to HF Hub.
"""

from __future__ import annotations
import os
from pathlib import Path


def resolve_model_path(repo_id: str, env_var: str | None = None) -> str:
    """
    Resolves the model path:
    1. Checks the specified environment variable override if provided.
    2. Searches for downloaded snapshots in local hf_cache/hub if present.
    3. Falls back cleanly to the Hugging Face repo ID.
    """
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val and os.path.exists(env_val):
            return env_val

    # Check local workspace hf_cache
    root = Path(__file__).resolve().parent.parent
    local_hub = root / "hf_cache" / "hub"
    if local_hub.is_dir():
        model_dir = local_hub / f"models--{repo_id.replace('/', '--')}" / "snapshots"
        if model_dir.is_dir():
            snaps = [p for p in model_dir.iterdir() if p.is_dir()]
            if snaps:
                return str(snaps[0])

    return repo_id
