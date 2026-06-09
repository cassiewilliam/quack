# Copyright (c) 2025. SonicMoE fused per-expert offset metadata.
"""Fused per-expert offset metadata for grouped MoE, in a single Triton kernel.

Given per-expert token counts ``split_sizes`` [G], the grouped-GEMM path needs the prefix-sum
offsets ``[0, s0, s0+s1, ..., M]`` [G+1] in two forms:
  * ``cu_seqlens`` (int32) -- for the QuACK grouped-GEMM ``cu_seqlens_m`` / ``cu_seqlens_k``.
  * ``base_offsets`` (int64) -- for GroupedTensor ``tensor_offsets`` (= base_offsets * stride).

Mirrors the metadata fusion in Dao-AILab/sonic-moe (``sonicmoe/functional/triton_kernels``): one
Triton kernel does the prefix-sum, replacing the scattered torch ``splits_to_offsets`` +
``pad(cumsum())`` (a CUB DeviceScan + pad-fill + dtype cast = several tiny launch-bound kernels).
Compute once in the forward and reuse ``base_offsets`` in the backward.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_offsets_kernel(split_ptr, cu_ptr, base_ptr, G, BLOCK: tl.constexpr):
    # Single program (G small, BLOCK = next_pow2(G)). Exclusive prefix sum + total at index G.
    offs = tl.arange(0, BLOCK)
    mask = offs < G
    s = tl.load(split_ptr + offs, mask=mask, other=0).to(tl.int64)
    cum = tl.cumsum(s, axis=0)  # inclusive prefix sum
    excl = cum - s              # exclusive prefix sum == [0, s0, s0+s1, ...] == cu_seqlens[0:G]
    tl.store(cu_ptr + offs, excl.to(tl.int32), mask=mask)
    tl.store(base_ptr + offs, excl, mask=mask)
    total = tl.sum(s)           # == M, the last offset
    tl.store(cu_ptr + G, total.to(tl.int32))
    tl.store(base_ptr + G, total)


def compute_moe_offsets(split_sizes: torch.Tensor):
    """Per-expert prefix-sum offsets ``[0, s0, ..., M]`` [G+1] in one fused Triton kernel.

    Args:
        split_sizes: int tensor [G] of per-expert token counts.
    Returns:
        (cu_seqlens int32 [G+1], base_offsets int64 [G+1]) -- same values, dtypes for the two
        consumers (grouped-GEMM cu_seqlens vs GroupedTensor offsets).
    """
    assert split_sizes.ndim == 1, "split_sizes must be 1D [G]"
    G = split_sizes.numel()
    device = split_sizes.device
    cu = torch.empty(G + 1, dtype=torch.int32, device=device)
    base = torch.empty(G + 1, dtype=torch.int64, device=device)
    BLOCK = triton.next_power_of_2(max(G, 1))
    _moe_offsets_kernel[(1,)](split_sizes, cu, base, G, BLOCK=BLOCK)
    return cu, base
