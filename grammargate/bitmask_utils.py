"""
Vectorized bitmask manipulation and logit reduction utilities.
"""

from __future__ import annotations
import math
import torch


_SHIFTS_CACHE: dict[torch.device, torch.Tensor] = {}
_ZERO_FLOAT_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_shifts(device: torch.device) -> torch.Tensor:
    cached = _SHIFTS_CACHE.get(device)
    if cached is None:
        cached = torch.arange(32, device=device, dtype=torch.int32)
        _SHIFTS_CACHE[device] = cached
    return cached


def _get_zero(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _ZERO_FLOAT_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(0.0, device=device, dtype=dtype)
        _ZERO_FLOAT_CACHE[key] = cached
    return cached


def unpack_bitmask(bitmask: torch.Tensor, vocab_size: int, target_device: torch.device | str | None = None) -> torch.Tensor:
    """
    Unpacks an XGrammar uint32/int32 packed bitmask into a boolean tensor of shape (batch_size, vocab_size).
    Bit k is 1 if token k is allowed, 0 if disallowed.
    Uses cached bit-shift vectors per device to avoid per-step allocations.
    """
    if bitmask.dtype == torch.bool:
        res = bitmask[..., :vocab_size] if bitmask.shape[-1] != vocab_size else bitmask
        if res.ndim == 1:
            res = res.unsqueeze(0)
        if target_device is not None and res.device != torch.device(target_device):
            res = res.to(target_device)
        return res

    batch_size = bitmask.shape[0] if bitmask.ndim > 1 else 1
    flat_bitmask = bitmask.view(batch_size, -1)

    words_needed = math.ceil(vocab_size / 32)
    flat_bitmask = flat_bitmask[:, :words_needed]

    target = torch.device(target_device) if target_device is not None else flat_bitmask.device
    if target != flat_bitmask.device:
        flat_bitmask = flat_bitmask.to(target, non_blocking=True)

    shifts = _get_shifts(target)
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
    shifts = _get_shifts(valid_mask.device)
    packed = (chunks << shifts).sum(dim=-1).to(torch.int32)
    return packed


def compute_metrics_vectorized(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    refusal_mask: torch.Tensor,
    refusal_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes pre-mask logit metrics in a single zero-sync vectorized pass:
    - killed_mass: 1.0 - sum(P_pre[valid])
    - refusal_suppression: sum(P_pre[refusal & ~valid])
    - top1_rank_shift: rank of top valid token in pre-mask distribution (0 if top1 is valid)
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

    logits_f = logits if logits.dtype == torch.float32 else logits.float()
    P_pre = torch.softmax(logits_f, dim=-1)
    zero_f = _get_zero(logits_f.device, torch.float32)

    valid_mass = torch.where(valid_mask, P_pre, zero_f).sum(dim=-1)
    killed_mass = (1.0 - valid_mass).clamp(min=0.0, max=1.0)

    if refusal_indices is not None:
        if refusal_indices.numel() > 0:
            disallowed_ref = ~valid_mask[:, refusal_indices]
            refusal_suppression = (P_pre[:, refusal_indices] * disallowed_ref).sum(dim=-1)
        else:
            refusal_suppression = torch.zeros(batch_size, device=logits.device, dtype=torch.float32)
    else:
        if refusal_mask.ndim == 1:
            refusal_mask = refusal_mask.unsqueeze(0)
        if refusal_mask.shape[-1] < vocab_size:
            pad = torch.zeros((refusal_mask.shape[0], vocab_size - refusal_mask.shape[-1]), dtype=torch.bool, device=refusal_mask.device)
            refusal_mask = torch.cat([refusal_mask, pad], dim=-1)
        elif refusal_mask.shape[-1] > vocab_size:
            refusal_mask = refusal_mask[:, :vocab_size]
        refusal_suppressed_mask = refusal_mask & (~valid_mask)
        refusal_suppression = torch.where(refusal_suppressed_mask, P_pre, zero_f).sum(dim=-1)

    top1_pre = torch.argmax(logits_f, dim=-1)
    top1_is_valid = valid_mask.gather(1, top1_pre.unsqueeze(1)).squeeze(1)

    top_valid_logits = logits_f.masked_fill(~valid_mask, -float("inf")).max(dim=-1, keepdim=True).values
    rank_shifts = (logits_f > top_valid_logits).sum(dim=-1)
    zero_i = torch.zeros_like(rank_shifts)
    top1_rank_shift = torch.where(top1_is_valid, zero_i, rank_shifts)

    return killed_mass, refusal_suppression, top1_rank_shift
