/*************************************************************************
 * Mega-MoE (Phase 4) host-side config — runtime mirror of the kernel's
 * compile-time template parameters, so the launcher can derive a self-
 * consistent tile from a runtime MoE shape instead of hardcoding T_*.
 *
 * BF16 adaptation of the upstream mega_moe host API (github mega_moe:
 * include/mega_moe/shapes.h + workspace.h), with all FP8/FP4 MX scale-
 * factor state removed (no Recipe / *_sf buffer segments).
 *
 * The fused kernel (megamoe_vendor/.../sm100_bf16_mega_moe.cuh) is fully
 * templated, so a given (MoEConfig, TileConfig) selects one pre-instantiated
 * specialization; get_tile_config() picks a known-good tile for the shape.
 *************************************************************************/
#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_MEGA_MOE_CONFIG_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_MEGA_MOE_CONFIG_H_

#include <cstdint>

namespace transformer_engine {
namespace mega_moe {

// ---------------------------------------------------------------------------
// MoE shape / topology — the part of the kernel's template params that comes
// from the model config (not tunable). Mirrors upstream MoEConfig (no Recipe).
// ---------------------------------------------------------------------------
struct MoEConfig {
  uint32_t num_max_tokens_per_rank = 0;  // per-rank token capacity (aligned up to block_m)
  uint32_t hidden = 0;                    // H: in/out hidden dim
  uint32_t intermediate = 0;             // I: per-expert FFN intermediate (gate/up each = I)
  uint32_t num_experts = 0;              // global expert count
  uint32_t num_topk = 0;                 // experts routed per token

  uint32_t num_ranks = 1;                // EP world size (NVLink domain)
  uint32_t num_sms = 148;                // persistent grid size (B200 = 148; must be even for 2-CTA cluster)

  float activation_clamp = 0.0f;         // SwiGLU pre-clamp (0 -> use a finite sentinel, no clamp)
  bool fast_math = false;                // fast silu/exp approximation

  // derived
  constexpr uint32_t num_experts_per_rank() const { return num_experts / num_ranks; }
  constexpr uint32_t l1_shape_n() const { return intermediate * 2; }  // gate||up
  constexpr uint32_t l1_shape_k() const { return hidden; }
  constexpr uint32_t l2_shape_n() const { return hidden; }
  constexpr uint32_t l2_shape_k() const { return intermediate; }
};

// ---------------------------------------------------------------------------
// Tile / pipeline / thread config — decoupled from MoEConfig for tuning.
// Defaults are the EP=2/EP=4-VALIDATED BF16 tile (rel=0.0056, cos=1.0).
//
// BF16 INVARIANTS (do not change without re-validating the kernel):
//   * block_k == 64  — BF16(2B): block_k*sizeof = 128B = exactly one TMA swizzle
//     atom. (FP8(1B) used 128; 128*2=256B is an invalid swizzle for BF16.)
//   * block_n == 128 — L1 out N is halved by SwiGLU into L1_OUT_BLOCK_N = 64.
//   * num_non_epilogue_threads == 128 (strict kernel static_assert).
// ---------------------------------------------------------------------------
struct TileConfig {
  uint32_t block_m = 64;                  // token / GEMM M block
  uint32_t block_n = 128;                 // weight-out N block (SwiGLU halves to 64)
  uint32_t block_k = 64;                  // K block — BF16 fixed at 64 (one 128B swizzle atom)
  uint32_t store_block_m = 32;            // epilogue store M block
  uint32_t num_stages = 4;                // K-dim software pipeline depth

  uint32_t num_experts_per_wave = 1;      // experts processed per scheduler wave (must divide experts_per_rank)

  uint32_t num_dispatch_threads = 128;    // dispatch warps  (% 128 == 0)
  uint32_t num_non_epilogue_threads = 128;  // GEMM TMA + MMA (strict == 128)
  uint32_t num_epilogue_threads = 128;    // epilogue + combine (% 128 == 0)
};

// ---------------------------------------------------------------------------
// Weight gate/up interleave granularity the SwiGLU epilogue expects.
// The epilogue pairs gate/up WITHIN one TMEM load, so the 2I weight columns
// must be interleaved as [g0..g7,u0..u7,g8..g15,...]. Empirically the BF16
// kernel (BLOCK_K=64 re-tile) pairs at granularity 8 (swept 8/16/32 -> only 8
// gives cos=1.0). NOTE: the mcore fused-MoE convention is interleave_size=32;
// reconciling the two (likely a TMEM-layout consequence of the BLOCK_K change)
// is the open blocker before drop-in integration with the real model weights.
constexpr uint32_t kGluInterleaveGranularity = 8;

// ---------------------------------------------------------------------------
// Pool capacity — worst-case tokens a rank's shared expert pool can hold,
// plus per-expert block_m alignment padding. Mirrors the kernel's
// get_num_max_pool_tokens (layout/mega_moe.cuh): the kLCMCandidateBlockM(384)
// / kMaxCandidateBlockM(192) candidate-block-M alignment is baked in here so
// the host doesn't depend on the vendored header for the bound.
constexpr uint32_t kLCMCandidateBlockM = 384;
constexpr uint32_t kMaxCandidateBlockM = 192;

constexpr uint32_t constexpr_align(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }
constexpr uint32_t constexpr_min(uint32_t a, uint32_t b) { return a < b ? a : b; }

constexpr uint32_t get_num_max_pool_tokens(const MoEConfig& cfg) {
  const uint32_t num_max_recv = cfg.num_ranks * cfg.num_max_tokens_per_rank;
  const uint32_t epr = cfg.num_experts_per_rank();
  const uint32_t experts_per_token = constexpr_min(cfg.num_topk, epr);
  return constexpr_align(num_max_recv * experts_per_token + epr * (kMaxCandidateBlockM - 1u),
                         kLCMCandidateBlockM);
}

// ---------------------------------------------------------------------------
// Heuristic: derive a self-consistent BF16 TileConfig from the MoE shape.
//
// HONEST STATUS: the EP=2/EP=4 toy config (~32-64 tokens/expert, E<=16) is
// VALIDATED with the defaults below. The branches for larger shapes (real
// model: E=256, top-12, EP=4) are best-effort and MUST be re-validated when
// the real shape is first run (the "先攻克" step) — in particular
// num_experts_per_wave for large experts_per_rank, and block_m for higher
// tokens-per-expert. The authoritative source is the deep_gemm JIT
// get_best_config; this encodes only what the BF16 port has validated so far.
inline TileConfig get_tile_config(const MoEConfig& cfg) {
  TileConfig t;  // BF16-fixed: block_n=128, block_k=64; threads=128/128/128
  // All local experts in one wave (mirrors the validated EP=2/EP=4 choice);
  // a too-small wave deadlocks the L1->L2 arrival logic. For very large
  // experts_per_rank this likely needs a divisor < epr — TBD at real shape.
  const uint32_t epr = cfg.num_experts_per_rank();
  t.num_experts_per_wave = epr == 0 ? 1 : epr;
  t.block_m = 64;
  t.store_block_m = 32;
  t.num_stages = 4;
  return t;
}

}  // namespace mega_moe
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_MEGA_MOE_CONFIG_H_
