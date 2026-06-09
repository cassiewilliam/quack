// Copyright (c) 2025, quack contributors.
// TVM-FFI binding for the vendored BF16 MegaMoE forward megakernel.
// Registers the Instance<...> shape specializations and wraps the matches()-dispatch
// in typed functions; the kernel headers are vendored flat alongside this file. See README
// for caveats (bf16, forward-only, EP/symmetric-memory, shape registration).

#include <cstdint>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <mega_moe_launch.cuh>

namespace ffi = tvm::ffi;
using ffi::TensorView;
namespace mm = transformer_engine::mega_moe;

namespace {

#define MM_CHECK(cond, msg) TVM_FFI_CHECK((cond), ValueError) << msg

// Registered shapes (from qa/megamoe_bf16_launch.cu). Tile params must match
// get_tile_config(cfg) + get_num_max_pool_tokens(cfg). Add a line + F(...) for a new shape.
//                          NMaxTok   H    I    E   TK  EPW  BM  BN  BK  SBM  NPool   STG  DTH  NETH  ETH  NSMS  NR
using EP2   = mm::Instance<  128,    256, 256,  16,  4,   8,  64, 128, 64,  32,   2688,  4,  128, 128, 128,  148,  2>;
using EP4   = mm::Instance<  128,    256, 256,  16,  4,   4,  64, 128, 64,  32,   3072,  4,  128, 128, 128,  148,  4>;
using REAL  = mm::Instance<  256,   2048, 512, 256, 12,  64,  64, 128, 64,  32,  24576,  4,  128, 128, 128,  148,  4>;
using REALB = mm::Instance< 2048,   2048, 512, 256, 12,  64,  64, 128, 64,  32, 110592,  4,  128, 128, 128,  148,  4>;
using REAL8 = mm::Instance<  256,   2048, 512, 256, 12,  32,  64, 128, 64,  32,  30720,  4,  128, 128, 128,  148,  8>;
using RBAL4 = mm::Instance< 4096,   2048, 512, 256, 12,  64,  64, 128, 64,  32, 110592,  4,  128, 128, 128,  148,  4>;
using RBAL8 = mm::Instance< 16384,  2048, 512, 256, 12,  32,  64, 128, 64,  32, 399360,  4,  128, 128, 128,  148,  8>;
#define MEGA_MOE_INSTANCES(F) F(EP2) F(EP4) F(REAL) F(REALB) F(REAL8) F(RBAL4) F(RBAL8)

// meta int64: [num_tokens, nmaxtok, H, I, E, topk, nranks, rank, nsms]
mm::MoEConfig cfg_from_meta(const int64_t* m) {
  mm::MoEConfig c;
  c.num_max_tokens_per_rank = (uint32_t)m[1];
  c.hidden                  = (uint32_t)m[2];
  c.intermediate            = (uint32_t)m[3];
  c.num_experts             = (uint32_t)m[4];
  c.num_topk                = (uint32_t)m[5];
  c.num_ranks               = (uint32_t)m[6];
  c.num_sms                 = (uint32_t)m[8];
  return c;
}

// meta / peer_ptrs are small CPU int64 tensors (read on host).
const int64_t* host_i64(TensorView t, const char* name) {
  MM_CHECK(t.dtype() == ffi::DataType::Int(64), std::string(name) + " must be int64");
  MM_CHECK(t.device().device_type == kDLCPU, std::string(name) + " must be on CPU");
  return static_cast<const int64_t*>(t.data_ptr());
}

cudaStream_t current_stream(DLDevice dev) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
}

}  // namespace

// Dynamic shared-memory bytes for this shape (-1 if unregistered).
int64_t MegaMoEBf16Smem(TensorView meta) {
  const mm::MoEConfig cfg = cfg_from_meta(host_i64(meta, "meta"));
#define F(I) if (I::matches(cfg)) return (int64_t)I::smem_bytes();
  MEGA_MOE_INSTANCES(F)
#undef F
  return -1;
}

// Symmetric-buffer byte layout into CPU int64 out[>=7]:
//   [total_bytes, x_off, topk_idx_off, topk_w_off, l1_acts_off, l2_acts_off, num_max_pool_tokens]
// out[0] = -1 if the shape is unregistered.
void MegaMoEBf16Layout(TensorView meta, TensorView out) {
  const mm::MoEConfig cfg = cfg_from_meta(host_i64(meta, "meta"));
  MM_CHECK(out.dtype() == ffi::DataType::Int(64) && out.device().device_type == kDLCPU,
           "out must be a CPU int64 tensor");
  MM_CHECK(out.numel() >= 7, "out must have >= 7 int64 elements");
  int64_t* o = static_cast<int64_t*>(out.data_ptr());
#define F(I) if (I::matches(cfg)) { auto L = I::compute_layout();                       \
    o[0]=L.total_bytes; o[1]=L.x_off; o[2]=L.topk_idx_off; o[3]=L.topk_w_off;           \
    o[4]=L.l1_acts_off; o[5]=L.l2_acts_off; o[6]=L.num_max_pool_tokens; return; }
  MEGA_MOE_INSTANCES(F)
#undef F
  o[0] = -1;
}

// Launch the megakernel in-place into y [num_tokens, H] bf16. w1/w2 are this rank's local L1/L2
// weights (contiguous bf16); x/topk_idx/topk_w must already sit in this rank's symmetric buffer
// (peer_ptrs[rank]) at the offsets from MegaMoEBf16Layout.
void MegaMoEBf16(TensorView y, TensorView w1, TensorView w2,
                 TensorView peer_ptrs, TensorView meta) {
  const int64_t* m = host_i64(meta, "meta");
  const mm::MoEConfig cfg = cfg_from_meta(m);
  const int num_tokens = (int)m[0];
  const int rank_idx   = (int)m[7];

  MM_CHECK(y.device().device_type == kDLCUDA, "y must be a CUDA tensor");
  MM_CHECK(y.IsContiguous() && w1.IsContiguous() && w2.IsContiguous(), "y/w1/w2 must be contiguous");
  const int64_t* sym_ptrs = host_i64(peer_ptrs, "peer_ptrs");
  MM_CHECK((uint32_t)peer_ptrs.numel() == cfg.num_ranks, "peer_ptrs length must equal num_ranks");

  int64_t l1_off = -1, l2_off = -1;
#define F(I) if (I::matches(cfg)) { auto L = I::compute_layout(); l1_off=L.l1_acts_off; l2_off=L.l2_acts_off; }
  MEGA_MOE_INSTANCES(F)
#undef F
  MM_CHECK(l1_off >= 0, "MegaMoE: unregistered shape -- add an Instance<...> to MEGA_MOE_INSTANCES");

  cudaStream_t stream = current_stream(y.device());
  int rc = -1;
#define F(I) if (I::matches(cfg)) rc = (int)I::launch_ep(                                \
    y.data_ptr(), sym_ptrs, rank_idx, l1_off, l2_off,                                    \
    w1.data_ptr(), w2.data_ptr(), (uint32_t)num_tokens, stream, /*dbg=*/nullptr);
  MEGA_MOE_INSTANCES(F)
#undef F
  MM_CHECK(rc == (int)cudaSuccess, "MegaMoE launch_ep failed, cudaError=" + std::to_string(rc));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_bf16, MegaMoEBf16);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_bf16_layout, MegaMoEBf16Layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_bf16_smem, MegaMoEBf16Smem);
