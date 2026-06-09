# Copyright (c) 2025, quack contributors.
"""Python interface to the BF16 MegaMoE persistent megakernel (forward / inference).

Loads the TVM-FFI shared lib built from ``csrc/mega_moe`` (which wraps Transformer
Engine's ``Instance<...>::launch_ep`` BF16 megakernel) and exposes a torch-friendly
``mega_moe(...)`` entry. Mirrors quack's TVM-FFI usage and the loader convention of
``/Users/min.yang/github/mega_moe/python/mega_moe/__init__.py``.

Build the lib on an SM100 (Blackwell) box first (see ``csrc/mega_moe/CMakeLists.txt``)::

    export QUACK_MEGA_MOE_LIB=/path/to/build/libquack_mega_moe.so

Usage (single-GPU stub, num_ranks=1)::

    import torch
    from quack.mega_moe import MoEConfig, mega_moe
    cfg = MoEConfig(num_max_tokens_per_rank=256, hidden=2048, intermediate_hidden=512,
                    num_experts=256, num_topk=12, num_ranks=4, num_sms=148)
    y = mega_moe(x_bf16, w1_bf16, w2_bf16, topk_idx_i64, topk_w_f32, cfg, rank=0)

CAVEATS:
  * BF16, **forward-only** (no fused backward — training backward stays on QuACK GEMMs).
  * The (shape) must be REGISTERED in csrc/mega_moe/mega_moe_ffi.cu's MEGA_MOE_INSTANCES,
    else launch raises "unregistered shape".
  * Multi-rank (num_ranks > 1) needs NVLink **symmetric memory**: ``peer_ptrs`` must list
    every rank's symmetric-buffer base device address (IPC-exchanged). This module only
    auto-handles the single-rank stub; pass ``peer_ptrs`` yourself for true EP.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import tvm_ffi  # apache-tvm-ffi
except Exception as e:  # pragma: no cover
    tvm_ffi = None
    _IMPORT_ERR = e

_LIB_ENV = "QUACK_MEGA_MOE_LIB"
_MODULE = None


def load(lib_path: Optional[str] = None):
    """Load (and cache) the compiled TVM-FFI module. Reads ``QUACK_MEGA_MOE_LIB`` by default."""
    global _MODULE
    if _MODULE is not None and lib_path is None:
        return _MODULE
    if tvm_ffi is None:
        raise ImportError(f"tvm_ffi not available: {_IMPORT_ERR}")
    path = lib_path or os.environ.get(_LIB_ENV)
    if not path:
        raise ValueError(
            f"set {_LIB_ENV}=/path/to/libquack_mega_moe.so (build via csrc/mega_moe/CMakeLists.txt)"
        )
    mod = tvm_ffi.load_module(path)
    if lib_path is None:
        _MODULE = mod
    return mod


@dataclass
class MoEConfig:
    """Mirror of transformer_engine::mega_moe::MoEConfig (the runtime shape/topology)."""

    num_max_tokens_per_rank: int
    hidden: int
    intermediate_hidden: int
    num_experts: int
    num_topk: int
    num_ranks: int = 1
    num_sms: int = 148

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_experts // self.num_ranks

    def meta(self, num_tokens: int, rank: int = 0) -> torch.Tensor:
        """Pack into the CPU int64 meta tensor the binding expects (order: mega_moe_ffi.cu)."""
        return torch.tensor(
            [num_tokens, self.num_max_tokens_per_rank, self.hidden, self.intermediate_hidden,
             self.num_experts, self.num_topk, self.num_ranks, rank, self.num_sms],
            dtype=torch.int64, device="cpu",
        )


def _layout(mod, meta: torch.Tensor) -> dict:
    """Query the symmetric-buffer byte layout for this shape (raises if unregistered)."""
    out = torch.empty(7, dtype=torch.int64, device="cpu")
    mod.mega_moe_bf16_layout(meta, out)
    o = out.tolist()
    if o[0] < 0:
        raise RuntimeError(
            "MegaMoE: shape not registered in csrc/mega_moe/mega_moe_ffi.cu (MEGA_MOE_INSTANCES). "
            "Add an Instance<...> line for this (nmaxtok,H,I,E,topk,nranks)."
        )
    keys = ["total_bytes", "x_off", "topk_idx_off", "topk_w_off",
            "l1_acts_off", "l2_acts_off", "num_max_pool_tokens"]
    return dict(zip(keys, o))


def _region(buf: torch.Tensor, off: int, shape, dtype: torch.dtype) -> torch.Tensor:
    """Carve a typed [shape] view at byte-offset `off` of a contiguous uint8 buffer.

    The layout offsets from compute_layout() are >=1024-aligned, so the reinterpret is safe.
    """
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    flat = buf[off:off + nbytes].view(dtype)
    return flat.view(*shape)


def mega_moe(
    x: torch.Tensor,          # [num_tokens, H]  bf16, CUDA (this rank's input tokens)
    w1: torch.Tensor,         # [Le*2I, H]       bf16, CUDA (this rank's LOCAL L1 weights, contiguous)
    w2: torch.Tensor,         # [Le*H,  I]       bf16, CUDA (this rank's LOCAL L2 weights, contiguous)
    topk_idx: torch.Tensor,   # [num_tokens, topk] int64, CUDA
    topk_w: torch.Tensor,     # [num_tokens, topk] float32, CUDA
    cfg: MoEConfig,
    rank: int = 0,
    peer_ptrs: Optional[torch.Tensor] = None,  # CPU int64 [num_ranks] symmetric-buffer bases; None -> single-rank stub
    out: Optional[torch.Tensor] = None,
    sym_buffer: Optional[torch.Tensor] = None,  # reuse a preallocated uint8 CUDA buffer (>= total_bytes)
    lib_path: Optional[str] = None,
) -> torch.Tensor:
    """Run the BF16 MegaMoE megakernel and return ``y`` [num_tokens, H] bf16.

    Single-rank (cfg.num_ranks == 1): allocates the symmetric buffer, fills x/topk into it,
    launches, returns y. Multi-rank: you must pass ``sym_buffer`` (this rank's symmetric buffer,
    already IPC-registered at the same address on all ranks) and ``peer_ptrs`` (all ranks' bases),
    with x/topk pre-filled into this rank's buffer at the layout offsets.
    """
    mod = load(lib_path)
    assert x.is_cuda and x.dtype == torch.bfloat16, "x must be CUDA bf16"
    num_tokens, H = x.shape
    assert H == cfg.hidden, f"x hidden {H} != cfg.hidden {cfg.hidden}"
    meta = cfg.meta(num_tokens, rank)
    lay = _layout(mod, meta)

    if cfg.num_ranks == 1:
        if sym_buffer is None:
            sym_buffer = torch.empty(lay["total_bytes"], dtype=torch.uint8, device=x.device)
        # Fill this rank's symmetric buffer: x [nt,H] bf16, topk_idx [nt,topk] i64, topk_w [nt,topk] f32.
        _region(sym_buffer, lay["x_off"], (num_tokens, H), torch.bfloat16).copy_(x)
        _region(sym_buffer, lay["topk_idx_off"], (num_tokens, cfg.num_topk), torch.int64).copy_(topk_idx)
        _region(sym_buffer, lay["topk_w_off"], (num_tokens, cfg.num_topk), torch.float32).copy_(topk_w)
        if peer_ptrs is None:
            peer_ptrs = torch.tensor([sym_buffer.data_ptr()], dtype=torch.int64, device="cpu")
    else:
        if sym_buffer is None or peer_ptrs is None:
            raise ValueError(
                "multi-rank MegaMoE needs sym_buffer (this rank's symmetric buffer with x/topk "
                "pre-filled at layout offsets) and peer_ptrs (all ranks' symmetric-buffer bases). "
                "Symmetric-memory rendezvous (IPC) is the caller's responsibility."
            )

    if out is None:
        out = torch.empty(num_tokens, cfg.hidden, dtype=torch.bfloat16, device=x.device)
    mod.mega_moe_bf16(out, w1.contiguous(), w2.contiguous(), peer_ptrs, meta)
    return out


def smem_bytes(cfg: MoEConfig, num_tokens: int = 1, lib_path: Optional[str] = None) -> int:
    """Dynamic shared-memory bytes the kernel uses for this shape (-1 if unregistered)."""
    return int(load(lib_path).mega_moe_bf16_smem(cfg.meta(num_tokens)))
