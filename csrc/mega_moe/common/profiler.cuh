// =============================================================================
// mega/profiler.cuh —— per-SM Perfetto timeline probe (device side, shared across kernels)
// -----------------------------------------------------------------------------
// A **generic** probe in the blackwell_mega_kernel repo, shared by all mega kernels
// (mega_moe / the future mega_ffn / mega_attention ...). Modeled on FlashInfer's
// profiler.cuh (commit c802a05): emit begin/end events for each warp-role region,
// with each event carrying (timestamp, block_id, sm_id, event_id, event_type), written
// into a global buffer; after the run, use common/tools/export_perfetto.py to convert
// it into a Perfetto trace (one track per SM).
//
// Design points:
//   * Entirely toggled by the MEGA_ENABLE_PROFILER macro. **When undefined, all macros
//     expand to nothing**, zero overhead.
//   * Each event record is 8 bytes: union{ {num_blocks,num_groups} header; {tag,delta_time} event }
//     tag bit fields: [1:0]=event_type  [11:2]=event_idx  [23:12]=block_id  [31:24]=sm_id
//   * sm_id  ← `mov.u32 %0, %smid;`; timestamp ← `mov.u32 %0, %globaltimer_lo;`
//   * buffer write offset: each (block, group) has a fixed slot, no atomic:
//       slot   = 1 + block_id * num_groups + group_id   (+1 to skip the header)
//       stride = num_blocks * num_groups
//
// Each kernel defines its own event id enum (an int is enough), and agrees to match the
// event name table in export_perfetto.py one-to-one. This header does not bind the
// event semantics of any specific kernel.
// =============================================================================
#pragma once

#include <cstdint>

namespace mega::prof {

struct Entry {
    union {
        struct { uint32_t num_blocks; uint32_t num_groups; };
        struct { uint32_t tag; uint32_t delta_time; };
        uint64_t raw;
    };
};

}  // namespace mega::prof

// -----------------------------------------------------------------------------
// Toggle: off by default, zero overhead. To enable: compile with -DMEGA_ENABLE_PROFILER
// -----------------------------------------------------------------------------
#ifdef MEGA_ENABLE_PROFILER

namespace mega::prof {

enum EventType : uint32_t { kBegin = 0, kEnd = 1, kInstant = 2 };

static constexpr uint32_t kTypeShift  = 0;
static constexpr uint32_t kIdxShift   = 2;
static constexpr uint32_t kBlockShift = 12;
static constexpr uint32_t kSmShift    = 24;

__device__ __forceinline__ uint32_t read_smid() {
    uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s;
}
__device__ __forceinline__ uint32_t read_globaltimer_lo() {
    uint32_t t; asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(t)); return t;
}
// Low-overhead per-SM clock (a few cycles, doesn't serialize like %globaltimer).
// Per-SM (not cross-SM synchronized), but fine for per-SM-track timelines.
// Default timestamp source unless MEGA_PROFILER_USE_GLOBALTIMER is set.
__device__ __forceinline__ uint32_t read_clock_lo() {
    return static_cast<uint32_t>(clock64());
}
#ifdef MEGA_PROFILER_USE_GLOBALTIMER
#define MEGA_PROF_NOW() ::mega::prof::read_globaltimer_lo()
#else
#define MEGA_PROF_NOW() ::mega::prof::read_clock_lo()
#endif
__device__ __forceinline__ uint32_t encode_tag(uint32_t sm, uint32_t block,
                                                uint32_t event_idx, uint32_t type) {
    return (sm << kSmShift) | (block << kBlockShift) |
           (event_idx << kIdxShift) | (type << kTypeShift);
}

// Stateless, group-aware write: re-derives blockIdx/gridDim/smid and takes the
// buffer pointer at EACH call. Holds NO state across calls — critical because
// warp-specialized kernels run `reg_reconfig`/setmaxnreg, which would corrupt any
// cached register state carried across the kernel body.
//   `group` = warp-role track [0, num_groups). Each (block, group) gets its own
//   begin/end slots. `active` selects the single writer thread (a role's leader).
//   Layout: slot = 1 + (block*num_groups + group) + cursor * (num_blocks*num_groups).
//   event_idx is stored in the tag (we set event_idx = group = role).
__device__ __forceinline__ void write_event(void* buf, bool active, uint32_t group,
                                             uint32_t num_groups, uint32_t cursor,
                                             uint32_t event_idx, uint32_t type) {
    if (buf == nullptr || !active) return;
    const uint32_t blk = blockIdx.x, nbk = gridDim.x;
    Entry e;
    e.tag = encode_tag(read_smid(), blk, event_idx, type);
    e.delta_time = MEGA_PROF_NOW();
    const uint32_t slot = 1 + (blk * num_groups + group);
    reinterpret_cast<Entry*>(buf)[slot + (size_t)cursor * (nbk * num_groups)] = e;
}

__device__ __forceinline__ void write_header(void* buf, uint32_t num_groups) {
    if (buf == nullptr || blockIdx.x != 0 || threadIdx.x != 0) return;
    Entry h; h.num_blocks = gridDim.x; h.num_groups = num_groups;
    reinterpret_cast<Entry*>(buf)[0] = h;
}

}  // namespace mega::prof

// Group-aware macros. `act` = is-this-thread-the-role-leader; `g` = group/role;
// `ng` = num_groups; `ev` = event id (we pass ev == g so the role is in the tag).
#define MEGA_PROFILER_INIT(p_buf, ng)             ::mega::prof::write_header((p_buf), (ng))
#define MEGA_PROFILE_BEGIN(p_buf, act, g, ng, ev) ::mega::prof::write_event((p_buf), (act), (g), (ng), 0u, (uint32_t)(ev), ::mega::prof::kBegin)
#define MEGA_PROFILE_END(p_buf, act, g, ng, ev)   ::mega::prof::write_event((p_buf), (act), (g), (ng), 1u, (uint32_t)(ev), ::mega::prof::kEnd)
// explicit cursor (for per-iteration probes inside loops; begin=2*iter, end=2*iter+1)
#define MEGA_PROFILE_BEGIN_AT(p_buf, act, g, ng, cur, ev) ::mega::prof::write_event((p_buf), (act), (g), (ng), (cur), (uint32_t)(ev), ::mega::prof::kBegin)
#define MEGA_PROFILE_END_AT(p_buf, act, g, ng, cur, ev)   ::mega::prof::write_event((p_buf), (act), (g), (ng), (cur), (uint32_t)(ev), ::mega::prof::kEnd)

#else  // ---------------- off: expand to nothing, zero cost ----------------

#define MEGA_PROFILER_INIT(p_buf, ng)             ((void)0)
#define MEGA_PROFILE_BEGIN(p_buf, act, g, ng, ev) ((void)0)
#define MEGA_PROFILE_END(p_buf, act, g, ng, ev)   ((void)0)
#define MEGA_PROFILE_BEGIN_AT(p_buf, act, g, ng, cur, ev) ((void)0)
#define MEGA_PROFILE_END_AT(p_buf, act, g, ng, cur, ev)   ((void)0)

#endif  // MEGA_ENABLE_PROFILER
