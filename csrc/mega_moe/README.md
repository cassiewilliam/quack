# quack ⟵ TE BF16 MegaMoE (forward megakernel)

Exposes Transformer-Engine's **BF16 MegaMoE** persistent megakernel (Dispatch → L1 →
SwiGLU → L2 → Combine fused in one warp-specialized kernel, output-stationary, NVLink
symmetric-memory comm) to quack's Python layer via **TVM-FFI** — matching quack's existing
TVM-FFI export style (no pybind / torch C++ extension).

```
csrc/mega_moe/          (single flat include root)
  mega_moe_ffi.cu       quack TVM-FFI binding: Instance<...> shapes + matches()-dispatch,
                        exports mega_moe_bf16 / _layout / _smem      (the only authored file)
  CMakeLists.txt        builds libquack_mega_moe.so (sm_100a, links tvm_ffi)
  mega_moe_launch.cuh   VENDORED TE host layer (Instance / launch_ep / compute_layout)
  mega_moe_config.h     VENDORED MoEConfig / TileConfig / get_tile_config
  impls/                VENDORED DeepGEMM kernel: sm100_bf16_mega_moe.cuh
  comm/ common/ layout/ mma/ ptx/ scheduler/   VENDORED DeepGEMM deps (common/profiler.cuh = per-SM profiler)
  LICENSE               DeepGEMM license (vendored kernel)
quack/mega_moe.py       loader (tvm_ffi.load_module) + MoEConfig + mega_moe(...) wrapper
```

The kernel is NOT re-implemented here — everything except `mega_moe_ffi.cu` is a verbatim copy of
TE's `common/gemm/{mega_moe,megamoe_vendor}` (which itself vendors DeepGEMM), flattened into this
one directory so all `#include`s (`<mega_moe_launch.cuh>`, `<impls/...>`, `<common/...>`, …)
resolve under the single include root `csrc/mega_moe`. quack only adds the shape
registration + TVM-FFI wrapper. Only CUTLASS + tvm-ffi are external.

## Build (SM100 / Blackwell box only — cannot build on non-CUDA hosts)

```bash
cmake -B build csrc/mega_moe \
  -DQUACK_CUTLASS_DIR=/path/to/cutlass \
  -DQUACK_TVM_FFI_DIR=$(python -c "import tvm_ffi,os;print(os.path.dirname(tvm_ffi.__file__))")
cmake --build build -j
export QUACK_MEGA_MOE_LIB=$PWD/build/libquack_mega_moe.so
```

## Use

```python
import torch
from quack.mega_moe import MoEConfig, mega_moe
cfg = MoEConfig(num_max_tokens_per_rank=256, hidden=2048, intermediate_hidden=512,
                num_experts=256, num_topk=12, num_ranks=4, num_sms=148)   # = the REAL/REAL8 reg shape
y = mega_moe(x, w1, w2, topk_idx, topk_w, cfg, rank=0)   # single-rank stub auto-allocs sym buffer
```

## Status / caveats (read before relying on it)

- **BF16, forward-only.** No fused backward megakernel — training backward stays on the
  decomposed QuACK path (`forward_fused_moe`/`backward_fused_moe`). This op is for
  forward/inference.
- **Shape must be registered.** `mega_moe_ffi.cu`'s `MEGA_MOE_INSTANCES` lists the validated
  shapes (EP2/EP4/REAL/REALB/REAL8/RBAL4/RBAL8, vendored from `qa/megamoe_bf16_launch.cu`).
  Add a `using <Name> = mm::Instance<...>` + `F(<Name>)` for any new shape, with tile params ==
  `get_tile_config(cfg)` + `get_num_max_pool_tokens(cfg)` (`mega_moe/mega_moe_config.h`).
- **Real-shape numerics unvalidated.** Only the EP=2/EP=4 toy config is validated upstream
  (rel=0.0056, cos=1.0). Larger shapes are best-effort and MUST be re-validated.
- **gate/up interleave mismatch (open blocker for real-weight drop-in):** this kernel pairs
  gate/up at granularity **8**, but mcore's fused-MoE convention is **32** — reconcile before
  feeding real model weights (see `mega_moe_config.h::kGluInterleaveGranularity`).
- **Multi-rank EP needs NVLink symmetric memory + IPC rendezvous** (peer_ptrs of all ranks'
  symmetric-buffer bases). `quack/mega_moe.py` auto-handles only the single-rank stub; for true
  EP you allocate/IPC-register the symmetric buffer and pass `sym_buffer` + `peer_ptrs` yourself.
- **NVLink-domain only** (in-kernel pull/push) — no inter-node RDMA (unlike DeepEP).
- **NOT built/tested on this dev host** (no CUDA / SM100). Compile + numerically verify on B200.
