/*************************************************************************
 * Mega-MoE (Phase 4) BF16 host launcher — config-driven.
 *
 * Wraps the fully-templated fused kernel (megamoe_vendor sm100_bf16_mega_moe)
 * in an Instance<...> whose static members (smem_bytes / compute_layout /
 * launch_ep) are the EP=2/EP=4-VALIDATED launch mechanics, now parameterized
 * by the template params instead of file-scope T_* constants. A runtime
 * MoEConfig is dispatched to the matching pre-instantiated Instance.
 *
 * This replaces the hardcoded qa/megamoe_bf16_launch.cu T_* block: register
 * the (shape -> instance) specializations once, then launch by runtime cfg.
 *************************************************************************/
#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_MEGA_MOE_LAUNCH_CUH_
#define TRANSFORMER_ENGINE_COMMON_GEMM_MEGA_MOE_LAUNCH_CUH_

#include <cstdint>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <impls/sm100_bf16_mega_moe.cuh>

#include "mega_moe_config.h"

namespace transformer_engine {
namespace mega_moe {

// Symmetric-buffer segment offsets (BF16, no SF). base=nullptr -> byte offsets.
struct BufferOffsets {
  int64_t total_bytes = 0;
  int64_t x_off = 0, topk_idx_off = 0, topk_w_off = 0, l1_acts_off = 0, l2_acts_off = 0;
  int64_t num_max_pool_tokens = 0;
  // L2 backward: per-pool-slot src metadata (TokenSrcMetadata{rank,token,topk}, 3xu32) and per-local-
  // expert recv-count-sum (u64). With these + BLOCK_M, the host reconstructs the per-expert pool ranges
  // and the pool-slot -> source-token map needed for the dispatch/combine transpose (scatter/gather).
  int64_t metadata_off = 0;        // offset of token_src_metadata[num_max_pool_tokens]
  int64_t recv_count_sum_off = 0;  // offset of expert_recv_count_sum[experts_per_rank] (u64)
};

// 2D TMA descriptor (bf16). With a non-zero swizzle the smem inner box must be
// swizzle_bytes/elem (one swizzle atom); this override matches the kernel's
// make_umma_desc expectation. Runtime args, so not templated.
inline CUtensorMap make_tma_2d(void* ptr, int gin, int gout, int sin, int sout,
                               int gstride, int sw) {
  if (sw != 0) sin = sw / static_cast<int>(sizeof(nv_bfloat16));
  CUtensorMap tm;
  const cuuint64_t gd[2] = {(cuuint64_t)gin, (cuuint64_t)gout};
  const cuuint32_t sd[2] = {(cuuint32_t)sin, (cuuint32_t)sout};
  const cuuint64_t gs[1] = {(cuuint64_t)(gstride * (int)sizeof(nv_bfloat16))};
  const cuuint32_t es[2] = {1, 1};
  CUtensorMapSwizzle s = sw == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
                       : sw == 64  ? CU_TENSOR_MAP_SWIZZLE_64B
                       : sw == 32  ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE;
  cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, gd, gs, sd, es,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, s, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tm;
}

// 5D TMA descriptor for the L1 weights: delivers the kernel's gran-8 gate/up interleave straight
// from CONTIGUOUS HBM weights [LE, 2I, H] (rows are [gate(I); up(I)] per expert). The row axis is
// decomposed (group=I/8, gu=2, w8=8); dims inner->outer are [H, w8, gu, group, expert], with the
// gu dim strided by I*H (jump gate->up) and expert by 2I*H. Box loads [BK, 8, 2, n_groups_box, 1].
// Bit-identical to host to_kernel_gran8 + 2D TMA (verified qa/test_tma_interleave.cu, sw=128) so the
// kernel epilogue is unchanged while HBM weights stay contiguous (Muon-safe, no host permute).
inline CUtensorMap make_tma_5d_glu(void* ptr, int H, int I, int LE, int BK, int n_groups_box, int sw) {
  int sin = (sw != 0) ? sw / static_cast<int>(sizeof(nv_bfloat16)) : BK;
  CUtensorMap tm{};
  const cuuint64_t gd[5] = {(cuuint64_t)H, 8, 2, (cuuint64_t)(I / 8), (cuuint64_t)LE};
  const cuuint64_t gs[4] = {
      (cuuint64_t)((size_t)H * sizeof(nv_bfloat16)),           // w8: +1 row
      (cuuint64_t)((size_t)I * H * sizeof(nv_bfloat16)),       // gu: +I rows (gate->up)
      (cuuint64_t)((size_t)8 * H * sizeof(nv_bfloat16)),       // group: +8 rows
      (cuuint64_t)((size_t)2 * I * H * sizeof(nv_bfloat16))};  // expert: +2I rows
  const cuuint32_t sd[5] = {(cuuint32_t)sin, 8, 2, (cuuint32_t)n_groups_box, 1};
  const cuuint32_t es[5] = {1, 1, 1, 1, 1};
  CUtensorMapSwizzle s = sw == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
                       : sw == 64  ? CU_TENSOR_MAP_SWIZZLE_64B
                       : sw == 32  ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE;
  cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, ptr, gd, gs, sd, es,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, s, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tm;
}

// One compile-time specialization of the fused kernel + its launch mechanics.
// Template params 1:1 with sm100_bf16_mega_moe_impl (CLAMP finite sentinel: an
// infinity float-NTTP breaks the __global__ registration stub).
template <uint32_t NMaxTok, uint32_t H, uint32_t I, uint32_t E, uint32_t TOPK, uint32_t EPW,
          uint32_t BM, uint32_t BN, uint32_t BK, uint32_t SBM, uint32_t NPool, uint32_t STAGES,
          uint32_t DTH, uint32_t NETH, uint32_t ETH, uint32_t NSMS, uint32_t NRANKS,
          bool FAST = false>
struct Instance {
  static constexpr float CLAMP = 1.0e30f;
  static constexpr uint32_t kBlockM = BM;   // pool block size (per-expert pool ranges are BLOCK_M-aligned)
  static constexpr uint32_t LE = E / NRANKS;
  static constexpr uint32_t LOAD_BM = BM / 2, LOAD_BN = BN;
  static constexpr uint32_t kSwizzle = BK * sizeof(nv_bfloat16);          // BF16: 64*2 = 128
  static constexpr uint32_t kNumThreads = DTH + NETH + ETH;
  static constexpr uint32_t kDispWarps = DTH / 32;
  static constexpr uint32_t kEpiWarps = ETH / 32;
  static constexpr uint32_t kEpiWG = kEpiWarps / 4;
  static constexpr uint32_t kEpiStages = 2, kTMAStoreStages = 2;
  static constexpr uint32_t L1_OUT_BN = BN / 2;

  static constexpr auto kernel_ptr = &deep_gemm::sm100_bf16_mega_moe_impl<
      NMaxTok, H, I, E, TOPK, EPW, BM, BN, BK, SBM, NPool, STAGES,
      DTH, NETH, ETH, NSMS, NRANKS, CLAMP, FAST>;

  static constexpr uint32_t alignup(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

  // Dynamic shared-memory bytes — mirrors the kernel's smem layout exactly.
  // (Validated: this returns 93440 for the EP=2/EP=4 toy config.)
  static constexpr uint32_t smem_bytes() {
    const uint32_t EXPERT_COUNT = alignup(E * 4u, 1024u);
    const uint32_t SEND_BUFFER = alignup(H * (uint32_t)sizeof(nv_bfloat16) * kDispWarps, 1024u);
    const uint32_t A = LOAD_BM * BK * (uint32_t)sizeof(nv_bfloat16);
    const uint32_t Bw = LOAD_BN * BK * (uint32_t)sizeof(nv_bfloat16);
    const uint32_t CD_L1 = kEpiWG * SBM * L1_OUT_BN * (uint32_t)sizeof(nv_bfloat16) * kTMAStoreStages;
    const uint32_t CD_L2 = kEpiWG * SBM * BN * (uint32_t)sizeof(nv_bfloat16);
    const uint32_t CD = CD_L1 > CD_L2 ? CD_L1 : CD_L2;
    const uint32_t before_barriers = EXPERT_COUNT + SEND_BUFFER + CD + STAGES * (A + Bw);
    const uint32_t n_barriers = kDispWarps + STAGES * 2 + kEpiStages * 2 + kEpiWarps * 2;
    return before_barriers + n_barriers * 8u + 64u;
  }

  // Symmetric-buffer layout via the vendored Workspace + Buffer helpers (BF16, no SF).
  static BufferOffsets compute_layout() {
    using namespace deep_gemm::layout;
    Workspace ws(nullptr, (int)NRANKS, (int)E, (int)NMaxTok, (int)TOPK);
    const auto bf16_token = Data(H * sizeof(nv_bfloat16));
    const auto bf16_inter = Data(I * sizeof(nv_bfloat16));
    const auto topk_idx_l = Data(TOPK * sizeof(int64_t), false);
    const auto topk_w_l   = Data(TOPK * sizeof(float), false);
    const auto l1tw_l     = Data(sizeof(float), false);
    const auto np = ws.num_max_pool_tokens;
    const auto input_token    = Buffer(bf16_token, 1, NMaxTok, ws.get_end_ptr());
    const auto input_topk_idx = Buffer(topk_idx_l, 1, NMaxTok, input_token.get_end_ptr());
    const auto input_topk_w   = Buffer(topk_w_l,   1, NMaxTok, input_topk_idx.get_end_ptr());
    const auto l1_token       = Buffer(bf16_token, 1, np,      input_topk_w.get_end_ptr());
    const auto l1_topk_w      = Buffer(l1tw_l,     1, np,      l1_token.get_end_ptr());
    const auto l2_token       = Buffer(bf16_inter, 1, np,      l1_topk_w.get_end_ptr());
    const auto combine        = Buffer(bf16_token, TOPK, NMaxTok, l2_token.get_end_ptr());
    BufferOffsets o;
    o.total_bytes = reinterpret_cast<int64_t>(combine.get_end_ptr());
    o.x_off = reinterpret_cast<int64_t>(input_token.base);
    o.topk_idx_off = reinterpret_cast<int64_t>(input_topk_idx.base);
    o.topk_w_off = reinterpret_cast<int64_t>(input_topk_w.base);
    o.l1_acts_off = reinterpret_cast<int64_t>(l1_token.base);
    o.l2_acts_off = reinterpret_cast<int64_t>(l2_token.base);
    o.num_max_pool_tokens = (int64_t)np;
    // L2 backward offsets (host-callable Workspace accessors; base=nullptr -> raw byte offsets).
    o.metadata_off = reinterpret_cast<int64_t>(ws.get_token_src_metadata_ptr(0));
    o.recv_count_sum_off = reinterpret_cast<int64_t>(ws.get_expert_recv_count_sum_ptr(0));
    return o;
  }

  // Launch on `stream`. sym_buffer_ptrs[NRANKS] = all ranks' symmetric-buffer base
  // device addresses; x/topk pre-filled into this rank's buffer at compute_layout offsets;
  // w1/w2 this rank's LOCAL experts (LE=E/NRANKS): w1[LE*2I,H], w2[LE*H,I].
  static cudaError_t launch_ep(void* y, const int64_t* sym_buffer_ptrs, int rank_idx,
                               int64_t l1_acts_off, int64_t l2_acts_off,
                               void* w1, void* w2, uint32_t num_tokens, cudaStream_t stream,
                               void* dbg = nullptr) {
    auto* base = reinterpret_cast<uint8_t*>(sym_buffer_ptrs[rank_idx]);
    void* l1_acts = base + l1_acts_off;
    void* l2_acts = base + l2_acts_off;

    auto tm_l1_acts    = make_tma_2d(l1_acts, H, NPool, BK, LOAD_BM, H, kSwizzle);
    auto tm_l1_weights = make_tma_5d_glu(w1, H, I, LE, BK, BN / 16, kSwizzle);  // contiguous w1, gran-8 via TMA
    auto tm_l1_output  = make_tma_2d(l2_acts, I, NPool, BN / 2, SBM, I, kSwizzle);
    auto tm_l2_acts    = make_tma_2d(l2_acts, I, NPool, BK, LOAD_BM, I, kSwizzle);
    auto tm_l2_weights = make_tma_2d(w2, I, LE * H, BK, LOAD_BN, I, kSwizzle);

    std::vector<int64_t> ptrs(sym_buffer_ptrs, sym_buffer_ptrs + NRANKS);
    deep_gemm::layout::SymBuffer<NRANKS> sym(ptrs, rank_idx);

    const uint32_t smem = smem_bytes();
    cudaError_t e = cudaFuncSetAttribute((const void*)kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NSMS, 1, 1);
    cfg.blockDim = dim3(kNumThreads, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {2, 1, 1};
    cfg.attrs = attr;
    cfg.numAttrs = 1;

    int* recv_stats = nullptr;
    return cudaLaunchKernelEx(&cfg, kernel_ptr, y, recv_stats, num_tokens, dbg, sym,
                              tm_l1_acts, tm_l1_weights, tm_l1_output, tm_l2_acts, tm_l2_weights);
  }

  // Does this specialization serve the given runtime shape? (tile params NPool/EPW/...
  // are assumed consistent with get_tile_config(cfg) at registration time.)
  static bool matches(const MoEConfig& c) {
    return c.num_max_tokens_per_rank == NMaxTok && c.hidden == H && c.intermediate == I &&
           c.num_experts == E && c.num_topk == TOPK && c.num_ranks == NRANKS;
  }
};

}  // namespace mega_moe
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_MEGA_MOE_LAUNCH_CUH_
