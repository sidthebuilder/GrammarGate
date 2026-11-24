"""
Vectorized bitmask manipulation and logit reduction utilities.
"""

from __future__ import annotations
import math
import torch


def unpack_bitmask(bitmask: torch.Tensor, vocab_size: int, target_device: torch.device | str | None = None) -> torch.Tensor:
    """
    Unpacks an XGrammar uint32/int32 packed bitmask into a boolean tensor of shape (batch_size, vocab_size).
    Bit k is 1 if token k is allowed, 0 if disallowed.
    Optionally unpacks directly on target_device (e.g. CUDA) to eliminate CPU unpack and transfer latency.
    """
    if bitmask.dtype == torch.bool:
        res = bitmask[..., :vocab_size] if bitmask.shape[-1] != vocab_size else bitmask
        if target_device is not None and res.device != torch.device(target_device):
            res = res.to(target_device)
        return res

    # If bitmask is packed 32-bit integers
    batch_size = bitmask.shape[0] if bitmask.ndim > 1 else 1
    flat_bitmask = bitmask.view(batch_size, -1)

    # Number of 32-bit words needed
    words_needed = math.ceil(vocab_size / 32)
    flat_bitmask = flat_bitmask[:, :words_needed]

    target = torch.device(target_device) if target_device is not None else flat_bitmask.device
    if target != flat_bitmask.device:
        flat_bitmask = flat_bitmask.to(target, non_blocking=True)

    # Unpack bits: each int32/uint32 expands to 32 booleans
    # Using bitwise shifts on target device
    shifts = torch.arange(32, device=target, dtype=torch.int32)
    unpacked = (flat_bitmask.unsqueeze(-1) >> shifts) & 1
    unpacked_bool = (unpacked != 0).reshape(batch_size, -1)

    if unpacked_bool.shape[-1] < vocab_size:
        pad = torch.zeros(
            (batch_size, vocab_size - unpacked_bool.shape[-1]),
            dtype=torch.bool,
            device=target,
        )
        unpacked_bool = torch.cat([unpacked_bool, pad], dim=-1)
    else:
        unpacked_bool = unpacked_bool[:, :vocab_size]

    return unpacked_bool


def pack_boolean_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Packs a boolean tensor of shape (batch_size, vocab_size) into an XGrammar-compatible int32 tensor.
    """
    if valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)
    batch_size, vocab_size = valid_mask.shape
    words_needed = math.ceil(vocab_size / 32)
    padded_len = words_needed * 32

    if vocab_size < padded_len:
        pad = torch.zeros(
            (batch_size, padded_len - vocab_size),
            dtype=torch.bool,
            device=valid_mask.device,
        )
        padded_mask = torch.cat([valid_mask, pad], dim=-1)
    else:
        padded_mask = valid_mask

    chunks = padded_mask.view(batch_size, words_needed, 32).to(torch.int32)
    shifts = torch.arange(32, device=valid_mask.device, dtype=torch.int32)
    packed = (chunks << shifts).sum(dim=-1).to(torch.int32)
    return packed


def compute_metrics_vectorized(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    refusal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes pre-mask logit metrics in a single vectorized pass:
    - killed_mass: 1.0 - sum(P_pre[valid])
    - refusal_suppression: sum(P_pre[refusal & ~valid])
    - top1_rank_shift: rank of top valid token in pre-mask distribution (0 if top1 is valid)

    Returns:
        (killed_mass, refusal_suppression, top1_rank_shift) as 1D tensors of length batch_size.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)

    batch_size, vocab_size = logits.shape

    if valid_mask.shape[-1] < vocab_size:
        pad = torch.zeros((valid_mask.shape[0], vocab_size - valid_mask.shape[-1]), dtype=torch.bool, device=valid_mask.device)
        valid_mask = torch.cat([valid_mask, pad], dim=-1)
    elif valid_mask.shape[-1] > vocab_size:
        valid_mask = valid_mask[:, :vocab_size]

    # 1. Softmax over pre-mask logits
    P_pre = torch.softmax(logits.float(), dim=-1)

    # 2. Killed mass: total probability suppressed by the grammar
    # Zero out disallowed tokens to sum valid mass
    valid_mass = (P_pre * valid_mask.float()).sum(dim=-1)
    killed_mass = (1.0 - valid_mass).clamp(min=0.0, max=1.0)

    # 3. Refusal suppression: mass allocated to refusal onset tokens that are disallowed
    if refusal_mask.ndim == 1:
        refusal_mask = refusal_mask.unsqueeze(0)

    if refusal_mask.shape[-1] < vocab_size:
        pad = torch.zeros((refusal_mask.shape[0], vocab_size - refusal_mask.shape[-1]), dtype=torch.bool, device=refusal_mask.device)
        refusal_mask = torch.cat([refusal_mask, pad], dim=-1)
    elif refusal_mask.shape[-1] > vocab_size:
        refusal_mask = refusal_mask[:, :vocab_size]

    refusal_suppressed_mask = refusal_mask & (~valid_mask)
    if refusal_suppressed_mask.any():
        refusal_suppression = (P_pre * refusal_suppressed_mask.float()).sum(dim=-1)
    else:
        refusal_suppression = torch.zeros(batch_size, device=logits.device, dtype=torch.float32)

    # 4. Top-1 rank shift
    top1_pre = torch.argmax(logits, dim=-1)  # shape: (batch_size,)
    top1_is_valid = valid_mask.gather(1, top1_pre.unsqueeze(1)).squeeze(1)  # bool (batch_size,)

    if top1_is_valid.all():
        top1_rank_shift = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    else:
        top_valid_logits = torch.where(valid_mask, logits, -float("inf")).max(dim=-1, keepdim=True).values
        rank_shifts = (logits > top_valid_logits).sum(dim=-1)
        top1_rank_shift = torch.where(top1_is_valid, torch.zeros_like(rank_shifts), rank_shifts)

    return killed_mass, refusal_suppression, top1_rank_shift
