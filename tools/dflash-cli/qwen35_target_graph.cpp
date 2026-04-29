// Forward pass of Qwen3.5-27B (qwen35 hybrid) in pure ggml.
//
// Translates llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp` into
// our standalone library, hardcoded for Qwen3.5-27B dimensions. No
// llama.cpp runtime is linked — only ggml ops.
//
// Architecture highlights:
//   - 64 layers; every 4th (il % 4 == 3) is full attention, rest are Gated DeltaNet
//   - Full-attention Q projection is PACKED with a gate (attn_q has width 2*q_dim)
//   - Full attention uses M-RoPE with sections [11,11,10,0]
//   - Flash attention is GQA 24/4, causal
//   - Delta-net uses ggml_ssm_conv for the 1D conv + ggml_gated_delta_net for the recurrence
//   - FFN is SwiGLU (w_gate * silu, element-wise multiply with w_up, then w_down)
//
// State (persisted in TargetCache across calls):
//   - attn_k[16], attn_v[16]     : KV cache for full-attn layers, f16
//   - conv_state[48]             : 1D conv recurrence state, f32
//   - ssm_state[48]              : delta-net recurrent state (head_v^2 × H_v), f32
//
// Key dimensions (all hardcoded via DFLASH27B_* macros):
//   n_embd           = 5120
//   n_head           = 24    head_dim = 256   q_dim = n_head * head_dim = 6144
//   n_head_kv        = 4     kv_dim = 4 * 256 = 1024
//   n_ff             = 17408
//   d_inner (ssm)    = 6144
//   d_state (ssm)    = 128
//   dt_rank (ssm)    = 48    (num_v_heads)
//   n_group (ssm)    = 16    (num_k_heads)
//   head_v_dim       = d_inner / dt_rank = 128
//   head_k_dim       = d_state           = 128
//   conv_kernel      = 4

#include "internal.h"
#include "delta_net_chunked.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash27b {

// ─── Local qwen35 constants (from the GGUF, hardcoded for this model) ─
// These complement the DFLASH27B_* macros in dflash27b.h with qwen35-specific
// hparams that differ from the draft (which uses plain Qwen3 dims).
namespace q35 {
constexpr int N_HEAD        = 24;
constexpr int N_HEAD_KV     = 4;
constexpr int HEAD_DIM      = 256;   // key_length == value_length
constexpr int Q_DIM         = N_HEAD * HEAD_DIM;    // 6144
constexpr int KV_DIM        = N_HEAD_KV * HEAD_DIM; // 1024
constexpr int FFN_DIM       = 17408;

constexpr int SSM_D_INNER   = 6144;
constexpr int SSM_D_STATE   = 128;
constexpr int SSM_DT_RANK   = 48;
constexpr int SSM_N_GROUP   = 16;
constexpr int SSM_CONV_KERN = 4;

// Derived
constexpr int HEAD_V_DIM    = SSM_D_INNER / SSM_DT_RANK;  // 128
constexpr int HEAD_K_DIM    = SSM_D_STATE;                // 128
constexpr int CONV_CHANNELS = SSM_D_INNER + 2 * SSM_N_GROUP * SSM_D_STATE; // 6144 + 2*16*128 = 10240

constexpr float EPS         = 1e-6f;
constexpr float ROPE_THETA  = 10000000.0f;
}  // namespace q35

// ─── TargetCache allocation ─────────────────────────────────────────

bool create_target_cache(const TargetWeights & w,
                         int max_ctx,
                         int max_verify_tokens,
                         ggml_backend_t backend,
                         TargetCache & out,
                         int fa_sink_padded,
                         ggml_tensor * const * external_K,
                         ggml_tensor * const * external_V,
                         int n_external_layers,
                         int slot_index) {
    out.backend = backend;
    out.max_ctx = max_ctx;
    out.cur_pos = 0;
    out.sink_padded_alloc = fa_sink_padded;
    if (max_verify_tokens <= 0) {
        max_verify_tokens = DFLASH27B_DRAFT_BLOCK_SIZE;
    }

    const int n_full_attn = w.n_layer / w.full_attention_interval; // 16
    const int n_delta     = w.n_layer - n_full_attn;               // 48

    // Validate external K/V borrow: both non-null, both same length, length matches.
    const bool borrow_kv = (external_K != nullptr && external_V != nullptr);
    if (borrow_kv) {
        if (n_external_layers != n_full_attn) {
            set_last_error("create_target_cache: borrow K/V layer count mismatch");
            return false;
        }
        if (slot_index < 0) {
            set_last_error("create_target_cache: slot_index must be >= 0");
            return false;
        }
    }
    out.kv_borrowed = borrow_kv;

    out.attn_k.assign(n_full_attn, nullptr);
    out.attn_v.assign(n_full_attn, nullptr);
    out.attn_k_combined.assign(n_full_attn, nullptr);
    out.attn_v_combined.assign(n_full_attn, nullptr);
    out.sink_built_for_lcp.assign(n_full_attn, -1);
    out.ssm_state.assign(n_delta, nullptr);
    out.conv_state.assign(n_delta, nullptr);
    out.ssm_state_snap.assign(n_delta, nullptr);
    out.conv_state_snap.assign(n_delta, nullptr);
    out.ssm_state_anchors.assign(TargetCache::DFLASH_ANCHOR_SLOTS, std::vector<ggml_tensor *>(n_delta, nullptr));
    out.conv_state_anchors.assign(TargetCache::DFLASH_ANCHOR_SLOTS, std::vector<ggml_tensor *>(n_delta, nullptr));
    out.anchor_positions.fill(0);
    out.ssm_intermediate.assign(n_delta, nullptr);
    out.conv_input_cache.assign(n_delta, nullptr);

    // Size the cache ggml context to hold all state tensors.
    //   per full-attn layer  : 2 (K, V) + 2 if fa_sink_padded > 0 (K_sink, V_sink)
    //   per delta-net layer  : 6 + 2*K (ssm, conv, ssm_snap, conv_snap,
    //                             K × ssm_anchor, K × conv_anchor,
    //                             ssm_intermediate, conv_input_cache)
    //   top-level            : 1 (target_feat)
    const int sink_tensors_per_layer = (fa_sink_padded > 0) ? 2 : 0;
    const int n_tensors = (2 + sink_tensors_per_layer) * n_full_attn + (6 + 2 * TargetCache::DFLASH_ANCHOR_SLOTS) * n_delta + 1;
    ggml_init_params ip{};
    ip.mem_size   = (size_t)(n_tensors + 32) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("cache ggml_init failed"); return false; }

    // Create the KV cache tensors (one set per full-attn layer).
    //
    // Env overrides (checked in order; last wins):
    //   DFLASH27B_KV_F16=1  → f16 (regression baseline)
    //   DFLASH27B_KV_Q4=1   → Q4_0 (8× vs f16, required for 128K on 24 GB, ~3% AL hit)
    //   DFLASH27B_KV_TQ3=1  → TQ3_0 (TurboQuant, 2.7× smaller than Q8_0)
    //
    // Default: Q8_0 — best quality/memory tradeoff at short context.
    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;
    if (const char * s = std::getenv("DFLASH27B_KV_F16")) {
        if (std::atoi(s) != 0) { kv_k_type = GGML_TYPE_F16; kv_v_type = GGML_TYPE_F16; }
    }
    if (const char * s = std::getenv("DFLASH27B_KV_Q4")) {
        if (std::atoi(s) != 0) { kv_k_type = GGML_TYPE_Q4_0; kv_v_type = GGML_TYPE_Q4_0; }
    }
    // TQ3_0 requires the F32→TQ3_0 CPY kernel (wired in ggml-cuda/cpy.cu)
    // and fattn support (already there, see fattn-common.cuh).
    if (const char * s = std::getenv("DFLASH27B_KV_TQ3")) {
        if (std::atoi(s) != 0) { kv_k_type = GGML_TYPE_TQ3_0; kv_v_type = GGML_TYPE_TQ3_0; }
    }
    // Independent K / V override so callers can mix, e.g. K=Q8_0 V=TQ3_0.
    auto parse_type = [](const char * s) -> ggml_type {
        if (!s) return GGML_TYPE_COUNT;
        if (std::strcmp(s, "f16")   == 0) return GGML_TYPE_F16;
        if (std::strcmp(s, "q8_0")  == 0) return GGML_TYPE_Q8_0;
        if (std::strcmp(s, "q4_0")  == 0) return GGML_TYPE_Q4_0;
        if (std::strcmp(s, "tq3_0") == 0) return GGML_TYPE_TQ3_0;
        return GGML_TYPE_COUNT;
    };
    if (ggml_type t = parse_type(std::getenv("DFLASH27B_KV_K")); t != GGML_TYPE_COUNT) kv_k_type = t;
    if (ggml_type t = parse_type(std::getenv("DFLASH27B_KV_V")); t != GGML_TYPE_COUNT) kv_v_type = t;
    int fa_idx = 0, dn_idx = 0;
    for (int il = 0; il < w.n_layer; il++) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        if (is_attn) {
            ggml_tensor * K = nullptr;
            ggml_tensor * V = nullptr;
            char name[64];
            if (borrow_kv) {
                // Borrow path: build non-owning views into the external
                // llama_kv_cache layer K/V tensors. Source layout is
                // `[head_dim*n_kv_heads, max_ctx, n_stream]`. We view it as
                // dflash's `[head_dim, max_ctx, n_kv_heads]` — strides become
                // non-standard (`nb2 < nb1`) but every kernel we touch
                // handles them via byte-offset math. See RFC Phase 1.
                ggml_tensor * src_K = external_K[fa_idx];
                ggml_tensor * src_V = external_V[fa_idx];
                if (!src_K || !src_V) {
                    set_last_error("create_target_cache: borrow K/V is null at layer index");
                    return false;
                }
                if ((int) src_K->ne[1] < max_ctx) {
                    set_last_error("create_target_cache: borrow K source has ne[1] < max_ctx");
                    return false;
                }
                // Type override: when borrowing, the external tensor's type
                // wins — env vars DFLASH27B_KV_K/V are ignored on this path
                // because the caller already owns the type via -ctk/-ctv.
                kv_k_type = src_K->type;
                kv_v_type = src_V->type;
                const size_t qk_K = (size_t) ggml_blck_size(kv_k_type);
                const size_t ts_K = (size_t) ggml_type_size(kv_k_type);
                const size_t qk_V = (size_t) ggml_blck_size(kv_v_type);
                const size_t ts_V = (size_t) ggml_type_size(kv_v_type);
                if (q35::HEAD_DIM % qk_K || q35::HEAD_DIM % qk_V) {
                    set_last_error("create_target_cache: HEAD_DIM not divisible by quant block size");
                    return false;
                }
                const size_t off_K = (size_t) slot_index * src_K->nb[2];
                const size_t off_V = (size_t) slot_index * src_V->nb[2];
                K = ggml_view_3d(out.ctx, src_K,
                                 q35::HEAD_DIM, max_ctx, q35::N_HEAD_KV,
                                 src_K->nb[1],                      // c-axis stride (full row)
                                 (q35::HEAD_DIM / qk_K) * ts_K,     // h-axis stride (head_dim worth)
                                 off_K);
                V = ggml_view_3d(out.ctx, src_V,
                                 q35::HEAD_DIM, max_ctx, q35::N_HEAD_KV,
                                 src_V->nb[1],
                                 (q35::HEAD_DIM / qk_V) * ts_V,
                                 off_V);
                std::snprintf(name, sizeof(name), "borrow_cache_k_%d", il);
                ggml_set_name(K, name);
                std::snprintf(name, sizeof(name), "borrow_cache_v_%d", il);
                ggml_set_name(V, name);
            } else {
                // Legacy path: allocate fresh K/V tensors in the cache's
                // own ggml context (will be backed by `out.buf` after the
                // ggml_backend_alloc_ctx_tensors call below).
                // [head_dim, max_ctx, n_head_kv]
                K = ggml_new_tensor_3d(out.ctx, kv_k_type,
                                       q35::HEAD_DIM, max_ctx, q35::N_HEAD_KV);
                V = ggml_new_tensor_3d(out.ctx, kv_v_type,
                                       q35::HEAD_DIM, max_ctx, q35::N_HEAD_KV);
                std::snprintf(name, sizeof(name), "cache_k_%d", il);
                ggml_set_name(K, name);
                std::snprintf(name, sizeof(name), "cache_v_%d", il);
                ggml_set_name(V, name);
            }
            out.attn_k[fa_idx] = K;
            out.attn_v[fa_idx] = V;

            // Persistent K_combined / V_combined for the attention-sink path.
            // Sized to hold sink + max window + spec-decode tokens. Sink portion is
            // reused across decode steps; window portion is rebuilt every call.
            if (fa_sink_padded > 0) {
                const int combined_len = fa_sink_padded + TargetCache::MAX_FA_WINDOW_PLUS_TOKENS;
                ggml_tensor * Kc = ggml_new_tensor_3d(out.ctx, kv_k_type,
                                                      q35::HEAD_DIM, combined_len, q35::N_HEAD_KV);
                ggml_tensor * Vc = ggml_new_tensor_3d(out.ctx, kv_v_type,
                                                      q35::HEAD_DIM, combined_len, q35::N_HEAD_KV);
                std::snprintf(name, sizeof(name), "cache_k_combined_%d", il); ggml_set_name(Kc, name);
                std::snprintf(name, sizeof(name), "cache_v_combined_%d", il); ggml_set_name(Vc, name);
                out.attn_k_combined[fa_idx] = Kc;
                out.attn_v_combined[fa_idx] = Vc;
            }
            fa_idx++;
        } else {
            // ssm_state: [head_v_dim, head_v_dim, num_v_heads]
            ggml_tensor * S  = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32,
                                                  q35::HEAD_V_DIM, q35::HEAD_V_DIM, q35::SSM_DT_RANK);
            ggml_tensor * Sn = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32,
                                                  q35::HEAD_V_DIM, q35::HEAD_V_DIM, q35::SSM_DT_RANK);
            ggml_tensor * Sa[TargetCache::DFLASH_ANCHOR_SLOTS];
            for (int k = 0; k < TargetCache::DFLASH_ANCHOR_SLOTS; k++) {
                Sa[k] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32,
                                           q35::HEAD_V_DIM, q35::HEAD_V_DIM, q35::SSM_DT_RANK);
            }
            // conv_state: [kernel-1, conv_channels]
            ggml_tensor * C  = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                                                  q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS);
            ggml_tensor * Cn = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                                                  q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS);
            ggml_tensor * Ca[TargetCache::DFLASH_ANCHOR_SLOTS];
            for (int k = 0; k < TargetCache::DFLASH_ANCHOR_SLOTS; k++) {
                Ca[k] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                                           q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS);
            }
            // ssm_intermediate: [S_v, S_v, H_v, max_verify_tokens] — one SSM
            // state per verify-block token. Sized to cover the largest verify
            // n_tokens we'll use (chain q_len=16 or DDTree 1+budget).
            // Stored in f16 to halve memory (~3 MB → 1.5 MB per layer per slot),
            // letting us fit budgets up to ~50 on 24 GB. The gated_delta_net
            // kernel converts f32 ↔ f16 on write/read via store/load_inter_state.
            ggml_tensor * Si = ggml_new_tensor_4d(out.ctx, GGML_TYPE_F16,
                                                  q35::HEAD_V_DIM, q35::HEAD_V_DIM,
                                                  q35::SSM_DT_RANK, max_verify_tokens);
            // conv_input_cache: [(K-1) + max_verify_tokens, conv_channels, 1]
            // — the full conv_input tensor captured during verify.
            ggml_tensor * Ci = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32,
                                                  (q35::SSM_CONV_KERN - 1) + max_verify_tokens,
                                                  q35::CONV_CHANNELS, 1);
            char name[64];
            std::snprintf(name, sizeof(name), "ssm_state_%d", il);         ggml_set_name(S,  name);
            std::snprintf(name, sizeof(name), "conv_state_%d", il);        ggml_set_name(C,  name);
            std::snprintf(name, sizeof(name), "ssm_state_snap_%d", il);    ggml_set_name(Sn, name);
            std::snprintf(name, sizeof(name), "conv_state_snap_%d", il);   ggml_set_name(Cn, name);
            for (int k = 0; k < TargetCache::DFLASH_ANCHOR_SLOTS; k++) {
                std::snprintf(name, sizeof(name), "ssm_state_anchor%d_%d", k, il);
                ggml_set_name(Sa[k], name);
                std::snprintf(name, sizeof(name), "conv_state_anchor%d_%d", k, il);
                ggml_set_name(Ca[k], name);
            }
            std::snprintf(name, sizeof(name), "ssm_intermediate_%d", il);  ggml_set_name(Si, name);
            std::snprintf(name, sizeof(name), "conv_input_cache_%d", il);  ggml_set_name(Ci, name);
            out.ssm_state[dn_idx]       = S;
            out.conv_state[dn_idx]      = C;
            out.ssm_state_snap[dn_idx]  = Sn;
            out.conv_state_snap[dn_idx] = Cn;
            for (int k = 0; k < TargetCache::DFLASH_ANCHOR_SLOTS; k++) {
                out.ssm_state_anchors[k][dn_idx]  = Sa[k];
                out.conv_state_anchors[k][dn_idx] = Ca[k];
            }
            out.ssm_intermediate[dn_idx] = Si;
            out.conv_input_cache[dn_idx] = Ci;
            dn_idx++;
        }
    }

    // Rolling target_feat buffer: [5*hidden, target_feat_len] bf16.
    //
    // target_feat_len is capped (default 4096) instead of growing to max_ctx,
    // because the draft only ever reads the last DRAFT_CTX_MAX=2048 positions
    // (see test_dflash.cpp). Cap = 2 * DRAFT_CTX_MAX to leave margin for
    // prefill batching and replay. Writes use `slot = kv_start % cap`; reads
    // produce a contiguous view of the last `draft_ctx` entries by handling
    // the wrap-around on the host side.
    //
    // At max_ctx=131072 this shrinks target_feat from 6.6 GB to 0.2 GB —
    // the difference that makes long context fit.
    constexpr int TARGET_FEAT_CAP_DEFAULT = 4096;
    out.target_feat_cap = std::min(max_ctx, TARGET_FEAT_CAP_DEFAULT);
    {
        const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * w.n_embd;  // 25600
        out.target_feat = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, fc_in, out.target_feat_cap);
        ggml_set_name(out.target_feat, "target_feat");
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed for target cache");
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // Zero-initialize all state tensors. We'll need a scratch zero buffer
    // since ggml_backend_tensor_memset isn't always available.
    // Use a big-enough zero buffer and iterate. Skip non-owning views —
    // those reference external buffers (e.g. the borrowed llama_kv_cache
    // K/V) where the host is responsible for initialisation. Zeroing
    // them here would clobber data the caller expects to keep.
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t != nullptr;
         t = ggml_get_next_tensor(out.ctx, t)) {
        if (t->view_src != nullptr) continue;
        size_t nb = ggml_nbytes(t);
        size_t off = 0;
        while (off < nb) {
            size_t chunk = std::min(nb - off, zeros.size());
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
            off += chunk;
        }
    }

    return true;
}

void free_target_cache(TargetCache & c) {
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
    c.ssm_state.clear();
    c.conv_state.clear();
    c.ssm_state_snap.clear();
    c.conv_state_snap.clear();
    c.ssm_state_anchors.clear();
    c.conv_state_anchors.clear();
    c.anchor_positions.fill(0);
    c.ssm_intermediate.clear();
    c.conv_input_cache.clear();
    c.target_feat = nullptr;
    c.cur_pos = 0;
    c.attn_k_combined.clear();
    c.attn_v_combined.clear();
    c.sink_built_for_lcp.clear();
    c.sink_padded_alloc = 0;
}

// Snapshot/restore SSM+conv state for speculative rollback. Uses device-side
// tensor copy (ggml_backend_tensor_copy). Called outside of any compute graph.
void snapshot_ssm_state(TargetCache & c) {
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        ggml_backend_tensor_copy(c.ssm_state[i], c.ssm_state_snap[i]);
        ggml_backend_tensor_copy(c.conv_state[i], c.conv_state_snap[i]);
    }
}

void restore_ssm_state(TargetCache & c) {
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        ggml_backend_tensor_copy(c.ssm_state_snap[i], c.ssm_state[i]);
        ggml_backend_tensor_copy(c.conv_state_snap[i], c.conv_state[i]);
    }
}

// Longer-timescale snapshot into a specific slot of the multi-anchor
// ring. The session writes to each of DFLASH_ANCHOR_SLOTS at
// geometrically-spaced positions during the last full prefill so a
// follow-up call can pick the best anchor <= lcp to rewind to.
void snapshot_anchor_state(TargetCache & c, int slot) {
    GGML_ASSERT(slot >= 0 && slot < TargetCache::DFLASH_ANCHOR_SLOTS);
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        ggml_backend_tensor_copy(c.ssm_state[i],  c.ssm_state_anchors[slot][i]);
        ggml_backend_tensor_copy(c.conv_state[i], c.conv_state_anchors[slot][i]);
    }
}

void restore_anchor_state(TargetCache & c, int slot) {
    GGML_ASSERT(slot >= 0 && slot < TargetCache::DFLASH_ANCHOR_SLOTS);
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        ggml_backend_tensor_copy(c.ssm_state_anchors[slot][i],  c.ssm_state[i]);
        ggml_backend_tensor_copy(c.conv_state_anchors[slot][i], c.conv_state[i]);
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

static ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * weight, float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

static ggml_tensor * build_swiglu_ffn(ggml_context * ctx, ggml_tensor * cur,
                                      const TargetLayer & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.w_gate, cur);   // [inter, n_tokens]
    if (L.w_gate_s) gate = ggml_mul(ctx, gate, L.w_gate_s);  // NVFP4 scale2
    gate = ggml_silu(ctx, gate);
    ggml_tensor * up = ggml_mul_mat(ctx, L.w_up, cur);
    if (L.w_up_s) up = ggml_mul(ctx, up, L.w_up_s);          // NVFP4 scale2
    ggml_tensor * gu = ggml_mul(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat(ctx, L.w_down, gu);    // [hidden, n_tokens]
    if (L.w_down_s) down = ggml_mul(ctx, down, L.w_down_s);  // NVFP4 scale2
    return down;
}

// Full-attention block (matches llama.cpp's build_layer_attn for qwen35)
//
// `cache_k` / `cache_v` are the persistent KV buffers for this layer
// (shape [head_dim, max_ctx, n_head_kv] f16). We write the new K/V for
// `n_tokens` new positions starting at `kv_start`, then run causal attention
// over [0..kv_start + n_tokens).
static ggml_tensor * build_full_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetLayer & L,
    ggml_tensor * cur,              // [hidden, n_tokens]
    ggml_tensor * positions,        // [n_tokens] i32
    const int * rope_sections,
    ggml_tensor * cache_k,          // [head_dim, max_ctx, n_head_kv]
    ggml_tensor * cache_v,          // [head_dim, max_ctx, n_head_kv]
    ggml_tensor * attn_mask,        // [kv_len, n_tokens] f32 or nullptr
    ggml_tensor * kv_idxs,          // [n_tokens] i32, set_rows index for TQ3_0 K
    int kv_start,
    int n_tokens,
    int fa_window,                  // sliding window: 0 = full attention
    int fa_sink,                    // attention sinks: keep first K KV positions visible (Xiao 2023). 0 = no sink.
    TargetCache * cache_ref,        // for K_combined persistent + sink_built_for_lcp tracking
    int fa_idx                      // index into cache_ref->attn_k_combined / sink_built_for_lcp
) {
    // ── Q projection (packed Q || gate), shape [2*q_dim, n_tokens]
    ggml_tensor * QG = ggml_mul_mat(ctx, L.wq, cur);
    if (L.wq_s) QG = ggml_mul(ctx, QG, L.wq_s);
    // Reshape to [head_dim*2, n_head, n_tokens] so we can view the Q and gate halves
    QG = ggml_reshape_3d(ctx, QG, q35::HEAD_DIM * 2, q35::N_HEAD, n_tokens);

    // Q half: view at offset 0, stride head_dim*2
    // Layout: [head_dim, n_head, n_tokens]
    ggml_tensor * Q = ggml_view_3d(ctx, QG,
        q35::HEAD_DIM, q35::N_HEAD, n_tokens,
        ggml_element_size(QG) * q35::HEAD_DIM * 2,                 // nb1: stride over n_head
        ggml_element_size(QG) * q35::HEAD_DIM * 2 * q35::N_HEAD,   // nb2: stride over n_tokens
        /*offset*/ 0);
    Q = rms_norm_mul(ctx, Q, L.q_norm, q35::EPS);

    // Gate half: view at offset head_dim
    ggml_tensor * gate = ggml_view_3d(ctx, QG,
        q35::HEAD_DIM, q35::N_HEAD, n_tokens,
        ggml_element_size(QG) * q35::HEAD_DIM * 2,
        ggml_element_size(QG) * q35::HEAD_DIM * 2 * q35::N_HEAD,
        ggml_element_size(QG) * q35::HEAD_DIM);
    gate = ggml_cont_2d(ctx, gate, q35::HEAD_DIM * q35::N_HEAD, n_tokens);  // [q_dim, n_tokens]

    // ── K and V projections
    ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);   // [kv_dim, n_tokens]
    if (L.wk_s) Kcur = ggml_mul(ctx, Kcur, L.wk_s);
    ggml_tensor * Vcur = ggml_mul_mat(ctx, L.wv, cur);   // [kv_dim, n_tokens]
    if (L.wv_s) Vcur = ggml_mul(ctx, Vcur, L.wv_s);

    Kcur = ggml_reshape_3d(ctx, Kcur, q35::HEAD_DIM, q35::N_HEAD_KV, n_tokens);
    Kcur = rms_norm_mul(ctx, Kcur, L.k_norm, q35::EPS);
    Vcur = ggml_reshape_3d(ctx, Vcur, q35::HEAD_DIM, q35::N_HEAD_KV, n_tokens);

    // ── M-RoPE (multi-axis rotary). n_rot = HEAD_DIM/4 * 4 ? Actually
    //    ggml_rope_multi takes n_dims = the number of dims to rotate; for
    //    qwen35 that's rope.dimension_count=64 (out of head_dim=256).
    int n_rot = 64;  // qwen35.rope.dimension_count
    int sections[4];
    for (int i = 0; i < 4; i++) sections[i] = rope_sections[i];

    Q = ggml_rope_multi(ctx, Q, positions, /*freq_factors=*/nullptr,
                        n_rot, sections, GGML_ROPE_TYPE_MROPE,
                        /*n_ctx_orig=*/0, q35::ROPE_THETA, 1.0f,
                        0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_multi(ctx, Kcur, positions, nullptr,
                           n_rot, sections, GGML_ROPE_TYPE_MROPE,
                           0, q35::ROPE_THETA, 1.0f,
                           0.0f, 1.0f, 0.0f, 0.0f);

    // ── Write K/V into the persistent cache at slot [kv_start..kv_start+n_tokens)
    //
    // cache_k is [head_dim, max_ctx, n_head_kv]. We want to copy Kcur
    // [head_dim, n_head_kv, n_tokens] into cache_k[:, kv_start:kv_start+n_tokens, :].
    //
    // Easiest: transpose Kcur to [head_dim, n_tokens, n_head_kv] so its axes
    // line up with cache_k's [head_dim, max_ctx, n_head_kv], then view a slice
    // of cache_k and copy.
    ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head_kv]
    ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head_kv]

    // K-write: cpy onto a strided 3D view of the TQ3_0 cache silently
    // corrupts blocks (cpy_blck reads 32 contiguous floats but the view
    // makes the 32 elements span across rows). Use ggml_set_rows with the
    // caller-supplied [n_tokens] i32 indices when the cache is quantized
    // (also when kv_idxs is provided); fall back to the original cpy on
    // f16/Q8_0 where strided cpy is well-tested.
    const bool use_set_rows = (kv_idxs != nullptr) &&
                              (cache_k->type == GGML_TYPE_TQ3_0 ||
                               cache_v->type == GGML_TYPE_TQ3_0);
    if (use_set_rows) {
        Kcur_T = ggml_cont(ctx, Kcur_T);
        Vcur_T = ggml_cont(ctx, Vcur_T);
        // ggml_set_rows constraint: c->ne[0] == b->ne[1], c->ne[1]==1.
        ggml_tensor * idx2d = ggml_reshape_2d(ctx, kv_idxs, n_tokens, 1);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_k, Kcur_T, idx2d));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_v, Vcur_T, idx2d));
    } else {
        ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
            q35::HEAD_DIM, n_tokens, q35::N_HEAD_KV,
            cache_k->nb[1], cache_k->nb[2],
            cache_k->nb[1] * kv_start);
        ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
            q35::HEAD_DIM, n_tokens, q35::N_HEAD_KV,
            cache_v->nb[1], cache_v->nb[2],
            cache_v->nb[1] * kv_start);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));
    }

    // Invalidate sink cache if THIS write touches positions < fa_sink (e.g. cold
    // prefill or LCP-divergent re-prefill from server-side cache reuse). Done in
    // CPU code at graph build time — the actual cache_k/cache_v writes scheduled
    // above will land before any subsequent sink-portion read.
    if (cache_ref && fa_sink > 0 && kv_start < fa_sink && cache_ref->sink_built_for_lcp.size() > (size_t) fa_idx) {
        cache_ref->sink_built_for_lcp[fa_idx] = -1;
    }

    // ── Flash attention slicing
    //
    // 1) fa_window > 0, fa_sink == 0  → standard SWA (last fa_window positions). Ours commit a19e198c6.
    //    Drops everything before kv_start - fa_window from attention.
    // 2) fa_sink > 0, fa_window > 0    → attention sinks (Xiao 2023 + Luce PR #26 follow-up):
    //    K/V exposed to FA = concat(sink_view[0..fa_sink], window_view[kv_start-fa_window..kv_start+n_tokens]).
    //    Keeps the system prompt + tool definitions (first ~fa_sink positions) attendable while
    //    still bounding FA cost to O(fa_sink + fa_window) at long contexts. Required for agent
    //    workloads (Dark Jarvis SOUL.md) where dropping the system prompt = identity loss.
    // 3) fa_window == 0                → full attention (default, pre-PR-26 behaviour).
    const int kv_len = kv_start + n_tokens;
    const bool use_sink =
        (fa_sink > 0 && fa_window > 0 && kv_start > fa_sink + fa_window);
    // When fa_sink is configured but threshold not met, force FULL attention
    // (win_start=0). Plain SWA would drop the system prompt — fa_sink>0 is the
    // user opt-in to "keep system prompt visible until fa_sink+fa_window threshold,
    // then switch to sink+window".
    const int win_start = use_sink
                              ? (kv_start - fa_window)
                              : ((fa_sink == 0 && fa_window > 0 && kv_start > fa_window) ? (kv_start - fa_window) : 0);
    const int win_len = kv_len - win_start;

    // FA kernel alignment requirements for the kv view length:
    //   - f16 / Q4_0 / Q8_0 paths: stride 1 (FA accepts any length)
    //   - TQ3_0 / TURBO* paths: need K->ne[1] divisible by FATTN_KQ_STRIDE=256
    //     (see fattn.cu:1348 can_use_vector_kernel and similar checks).
    // The K/V cache types are picked from env in this file, so we read them
    // back from the existing tensors and bump the pad to 256 when either
    // side is a 3-bit format. The caller's attn_mask is built using
    // g_kq_stride_pad (set to 256 by session.cpp at init when kv_tbq is on,
    // or here we mirror it locally) so positions beyond real kv_len get -inf.
    auto needs_fa_stride_256 = [](ggml_type t) {
        return t == GGML_TYPE_TQ3_0 || t == GGML_TYPE_TURBO2_0 ||
               t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 ||
               t == GGML_TYPE_TURBO3_TCQ || t == GGML_TYPE_TURBO2_TCQ;
    };
    const int fattn_stride  = (needs_fa_stride_256(cache_k->type) || needs_fa_stride_256(cache_v->type)) ? 256 : 1;
    const int win_len_padded = ((win_len + fattn_stride - 1) / fattn_stride) * fattn_stride;

    // Q needs to be [head_dim, n_tokens, n_head] for flash_attn_ext
    ggml_tensor * Qfa = ggml_permute(ctx, Q, 0, 2, 1, 3);   // [head_dim, n_tokens, n_head]
    Qfa = ggml_cont(ctx, Qfa);

    ggml_tensor * Kfa;
    ggml_tensor * Vfa;
    if (use_sink) {
        // Persistent K_combined/V_combined in cache_buf (allocated in create_target_cache).
        // Sink portion (positions 0..sink_padded) reused across decode steps when
        // sink_built_for_lcp[fa_idx] >= sink_padded. Window portion rebuilt every call.
        const int win_seg_len = win_len;  // window segment length INCLUDING new tokens
        const int sink_padded = ((fa_sink + fattn_stride - 1) / fattn_stride) * fattn_stride;
        const int total_target = ((sink_padded + win_seg_len + fattn_stride - 1) / fattn_stride) * fattn_stride;
        const int win_seg_padded = total_target - sink_padded;

        GGML_ASSERT(cache_ref && cache_ref->attn_k_combined.size() > (size_t) fa_idx
                    && cache_ref->attn_k_combined[fa_idx]
                    && "sink path requires cache_ref with attn_k_combined allocated (fa_sink_padded > 0 in create_target_cache)");
        ggml_tensor * K_combined = cache_ref->attn_k_combined[fa_idx];
        ggml_tensor * V_combined = cache_ref->attn_v_combined[fa_idx];

        // Bounds: ensure win_seg_padded fits the persistent buffer slack.
        GGML_ASSERT(sink_padded + win_seg_padded <= K_combined->ne[1]
                    && "K_combined too small for sink_padded + win_seg_padded; bump TargetCache::MAX_FA_WINDOW_PLUS_TOKENS");

        // Sink portion: rebuild if invalidated, otherwise reuse persistent data.
        const bool sink_stale = cache_ref->sink_built_for_lcp[fa_idx] < sink_padded;
        if (sink_stale) {
            ggml_tensor * K_sink_view = ggml_view_3d(ctx, cache_k,
                q35::HEAD_DIM, sink_padded, q35::N_HEAD_KV,
                cache_k->nb[1], cache_k->nb[2], 0);
            ggml_tensor * K_sink_slot = ggml_view_3d(ctx, K_combined,
                q35::HEAD_DIM, sink_padded, q35::N_HEAD_KV,
                K_combined->nb[1], K_combined->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, K_sink_view, K_sink_slot));

            ggml_tensor * V_sink_view = ggml_view_3d(ctx, cache_v,
                q35::HEAD_DIM, sink_padded, q35::N_HEAD_KV,
                cache_v->nb[1], cache_v->nb[2], 0);
            ggml_tensor * V_sink_slot = ggml_view_3d(ctx, V_combined,
                q35::HEAD_DIM, sink_padded, q35::N_HEAD_KV,
                V_combined->nb[1], V_combined->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, V_sink_view, V_sink_slot));
            cache_ref->sink_built_for_lcp[fa_idx] = sink_padded;
        }

        // Window portion: always rebuild (sliding window).
        ggml_tensor * K_win_view = ggml_view_3d(ctx, cache_k,
            q35::HEAD_DIM, win_seg_padded, q35::N_HEAD_KV,
            cache_k->nb[1], cache_k->nb[2], cache_k->nb[1] * win_start);
        ggml_tensor * K_win_slot = ggml_view_3d(ctx, K_combined,
            q35::HEAD_DIM, win_seg_padded, q35::N_HEAD_KV,
            K_combined->nb[1], K_combined->nb[2], K_combined->nb[1] * sink_padded);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, K_win_view, K_win_slot));

        ggml_tensor * V_win_view = ggml_view_3d(ctx, cache_v,
            q35::HEAD_DIM, win_seg_padded, q35::N_HEAD_KV,
            cache_v->nb[1], cache_v->nb[2], cache_v->nb[1] * win_start);
        ggml_tensor * V_win_slot = ggml_view_3d(ctx, V_combined,
            q35::HEAD_DIM, win_seg_padded, q35::N_HEAD_KV,
            V_combined->nb[1], V_combined->nb[2], V_combined->nb[1] * sink_padded);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, V_win_view, V_win_slot));

        // FA reads from K_combined / V_combined directly (sized to total_target rows).
        Kfa = ggml_view_3d(ctx, K_combined,
            q35::HEAD_DIM, sink_padded + win_seg_padded, q35::N_HEAD_KV,
            K_combined->nb[1], K_combined->nb[2], 0);
        Vfa = ggml_view_3d(ctx, V_combined,
            q35::HEAD_DIM, sink_padded + win_seg_padded, q35::N_HEAD_KV,
            V_combined->nb[1], V_combined->nb[2], 0);
    } else {
        // Single windowed view (or full attention when fa_window=0 / kv_start <= fa_window).
        Kfa = ggml_view_3d(ctx, cache_k,
            q35::HEAD_DIM, win_len_padded, q35::N_HEAD_KV,
            cache_k->nb[1], cache_k->nb[2], cache_k->nb[1] * win_start);
        Vfa = ggml_view_3d(ctx, cache_v,
            q35::HEAD_DIM, win_len_padded, q35::N_HEAD_KV,
            cache_v->nb[1], cache_v->nb[2], cache_v->nb[1] * win_start);
    }

    // Causal mask: for n_tokens==1 we don't need one (a single query attending
    // to all keys is trivially causal). For n_tokens>1 the caller must provide
    // a mask shaped [kv_len, n_tokens] with 0 for attendable positions and
    // -inf for positions beyond the causal boundary.
    const float kq_scale = 1.0f / std::sqrt((float)q35::HEAD_DIM);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                             kq_scale, 0.0f, 0.0f);
    // attn: [head_dim, n_head, n_tokens] (permuted)
    attn = ggml_reshape_2d(ctx, attn, q35::Q_DIM, n_tokens);

    // ── Apply the sigmoid gate from the packed Q
    ggml_tensor * gate_sig = ggml_sigmoid(ctx, gate);
    attn = ggml_mul(ctx, attn, gate_sig);

    // ── Output projection
    attn = ggml_mul_mat(ctx, L.wo, attn);  // [hidden, n_tokens]
    if (L.wo_s) attn = ggml_mul(ctx, attn, L.wo_s);
    return attn;
}

// Gated DeltaNet block using the fused ggml_gated_delta_net primitive.
//
// Matches the semantics of llama.cpp's build_layer_attn_linear + build_delta_net_fused.
// Updates cache->conv_state and cache->ssm_state in place.
//
// When `cap` is non-null, the function populates `cap->ssm_intermediate_states`
// with a view into the gated_delta_net result's per-step recurrent states and
// `cap->conv_input` with the concatenated conv input (old state + new tokens),
// both of which are marked as graph outputs so the caller can rollback SSM and
// conv state to any intermediate step commit_n-1 without a replay forward pass.
static ggml_tensor * build_delta_net_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetLayer & L,
    ggml_tensor * cur,            // [hidden, n_tokens]
    ggml_tensor * conv_state,     // [kernel-1, conv_channels] persistent
    ggml_tensor * ssm_state,      // [head_v_dim, head_v_dim, num_v_heads] persistent
    int n_tokens,
    DeltaNetCapture * cap,        // optional: populated on capture_delta_intermediate
    ggml_tensor * parent_ids      // optional [n_tokens] i32; tree mode when non-null
) {
    const int d_inner      = q35::SSM_D_INNER;
    const int head_k_dim   = q35::HEAD_K_DIM;   // 128
    const int num_k_heads  = q35::SSM_N_GROUP;  // 16
    const int num_v_heads  = q35::SSM_DT_RANK;  // 48
    const int head_v_dim   = q35::HEAD_V_DIM;   // 128
    const int n_seqs       = 1;
    const int n_seq_tokens = n_tokens;

    // ── qkv_mixed = wqkv @ cur         [10240, n_tokens]
    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, L.wqkv, cur);
    if (L.wqkv_s) qkv_mixed = ggml_mul(ctx, qkv_mixed, L.wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, q35::CONV_CHANNELS, n_seq_tokens, n_seqs);

    // ── z = wqkv_gate @ cur            [inner, n_tokens]
    ggml_tensor * z = ggml_mul_mat(ctx, L.wqkv_gate, cur);
    if (L.wqkv_gate_s) z = ggml_mul(ctx, z, L.wqkv_gate_s);

    // ── beta = ssm_beta @ cur          [dt_rank, n_tokens]
    ggml_tensor * beta = ggml_mul_mat(ctx, L.ssm_beta, cur);
    if (L.ssm_beta_s) beta = ggml_mul(ctx, beta, L.ssm_beta_s);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);

    // ── alpha = ssm_alpha @ cur        [dt_rank, n_tokens]
    //    alpha = alpha + ssm_dt_bias          (per-head bias)
    //    alpha = softplus(alpha)
    //    g     = alpha * ssm_a                (-A_log.exp() * softplus)
    ggml_tensor * alpha = ggml_mul_mat(ctx, L.ssm_alpha, cur);
    if (L.ssm_alpha_s) alpha = ggml_mul(ctx, alpha, L.ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    alpha = ggml_add(ctx, alpha, L.ssm_dt_bias);
    alpha = ggml_softplus(ctx, alpha);
    ggml_tensor * g_tensor = ggml_mul(ctx, alpha, L.ssm_a);
    g_tensor = ggml_reshape_4d(ctx, g_tensor, 1, num_v_heads, n_seq_tokens, n_seqs);

    // ── Fetch conv state [kernel-1, conv_channels] and prepend to qkv_mixed
    //    along the token axis to form the convolution input.
    ggml_tensor * conv_states_r = ggml_reshape_3d(ctx, conv_state,
        q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS, n_seqs);

    // qkv_mixed currently is [conv_channels, n_tokens, n_seqs]; we need
    // [n_tokens, conv_channels, n_seqs] to concat on dim 0.
    ggml_tensor * qkv_T = ggml_transpose(ctx, qkv_mixed);

    ggml_tensor * conv_input = ggml_concat(ctx, conv_states_r, qkv_T, 0);
    // conv_input: [kernel-1 + n_tokens, conv_channels, n_seqs]

    // For spec-decode rollback: copy the full conv_input into the persistent
    // cache buffer via an in-graph ggml_cpy. This avoids marking conv_input as
    // a graph output (which would force the gallocr to preserve its memory
    // past graph_compute). After graph_compute, the cache buffer's data is
    // always valid; the rollback code slices it at commit_n.
    if (cap && cap->conv_input) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, conv_input, cap->conv_input));
    }

    // ── Save the last (kernel-1) steps back to conv_state
    ggml_tensor * last_conv = ggml_view_3d(ctx, conv_input,
        q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        (conv_input->ne[0] - (q35::SSM_CONV_KERN - 1)) * ggml_element_size(conv_input));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last_conv, conv_state));

    // ── 1D conv + silu
    //    Tree mode: use the parent-chain-aware variant so sibling nodes gather
    //    their conv window from their actual tree parent instead of the DFS
    //    predecessor. Without this, siblings get garbage logits (the conv
    //    output would mix unrelated branches).
    ggml_tensor * conv_out = parent_ids
        ? ggml_ssm_conv_tree(ctx, conv_input, L.ssm_conv1d, parent_ids)
        : ggml_ssm_conv     (ctx, conv_input, L.ssm_conv1d);
    conv_out = ggml_silu(ctx, conv_out);

    // conv_out: [conv_channels, n_tokens, n_seqs]
    const int64_t q_offset = 0;
    const int64_t k_offset = num_k_heads * head_k_dim;
    const int64_t v_offset = 2 * num_k_heads * head_k_dim;

    const size_t elt = ggml_element_size(conv_out);
    const size_t row_size = q35::CONV_CHANNELS * elt;

    ggml_tensor * q_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        q_offset * elt);
    ggml_tensor * k_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        k_offset * elt);
    ggml_tensor * v_c = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        head_v_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        v_offset * elt);

    // L2 norm on Q and K
    q_c = ggml_l2_norm(ctx, q_c, q35::EPS);
    k_c = ggml_l2_norm(ctx, k_c, q35::EPS);

    // Repeat Q and K from num_k_heads to num_v_heads so they match V's layout
    // (only needed if not using the fused op's broadcast support).
    if (num_k_heads != num_v_heads) {
        q_c = ggml_repeat_4d(ctx, q_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_c = ggml_repeat_4d(ctx, k_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // ── SSM state (recurrent): reshape to [S_v, S_v, H_v, n_seqs]
    ggml_tensor * s = ggml_reshape_4d(ctx, ssm_state,
        head_v_dim, head_v_dim, num_v_heads, n_seqs);

    // ── Fused Gated DeltaNet op — returns packed (output | new_state [| intermediates]).
    //    In tree mode, the kernel uses parent_ids to reload state at DFS
    //    branch transitions (ported from sglang's retrieve_parent_token path).
    //    When `cap->ssm_intermediate_states` is present AND we are in tree
    //    mode, use the _tree_persist variant: the kernel writes per-token
    //    intermediate states DIRECTLY into the persistent cache buffer,
    //    eliminating the downstream ggml_cpy that would otherwise copy them.
    //    Saves ~5-10 ms per verify step (memory-bandwidth bound) on 27B.
    ggml_tensor * persist_inter = (parent_ids && cap && cap->ssm_intermediate_states)
        ? cap->ssm_intermediate_states
        : nullptr;

    // Chunked delta-net path: chain-only (no parent_ids), no per-token
    // capture (no cap). Ported from llama.cpp
    // src/models/delta-net-base.cpp::build_delta_net_chunking. At n_tokens=16
    // and 48 delta-net layers it eliminates the serial per-token loop that
    // dominates target-verify compute at long ctx. Currently OFF by
    // default — port produces correct shape but slightly wrong final state,
    // causing AL degradation and loopy output. Set DFLASH27B_CHUNKED=1 to
    // opt in for A/B testing while debugging.
    bool use_chunked = false;
    if (!parent_ids && !cap && n_seq_tokens > 1) {
        if (const char * s_env = std::getenv("DFLASH27B_CHUNKED")) {
            use_chunked = (std::atoi(s_env) != 0);
        }
    }

    ggml_tensor * output = nullptr;
    ggml_tensor * new_state = nullptr;

    if (use_chunked) {
        auto r = build_delta_net_chunked(ctx, q_c, k_c, v_c, g_tensor, beta, s);
        output    = r.output;
        new_state = r.new_state;
        goto after_delta_net;
    }

    ggml_tensor * result;
    result =
        persist_inter
            ? ggml_gated_delta_net_tree_persist(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids, persist_inter)
            : (parent_ids
                ? ggml_gated_delta_net_tree(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids)
                : ggml_gated_delta_net     (ctx, q_c, k_c, v_c, g_tensor, beta, s));

    // Slice output and new_state out of the packed result
    {
    const int64_t S_v = head_v_dim;
    const int64_t H_v = num_v_heads;
    const size_t r_elt = ggml_element_size(result);
    output = ggml_view_4d(ctx, result,
        S_v, H_v, n_seq_tokens, n_seqs,
        S_v * r_elt,
        S_v * H_v * r_elt,
        S_v * H_v * n_seq_tokens * r_elt,
        0);
    new_state = ggml_view_4d(ctx, result,
        S_v, S_v, H_v, n_seqs,
        S_v * r_elt,
        S_v * S_v * r_elt,
        S_v * S_v * H_v * r_elt,
        S_v * H_v * n_seq_tokens * n_seqs * r_elt);

    // Persist new_state back to cache
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ssm_state));

    // Expose per-step intermediate states for spec-decode rollback. The patched
    // ggml_gated_delta_net kernel appends an intermediate-states region to the
    // result tensor after the final-state slot. Layout in result->data:
    //   [ attn_out: S_v*H_v*n_seq_tokens*n_seqs floats
    //   | final_state: S_v*S_v*H_v*n_seqs floats
    //   | intermediate_states: S_v*S_v*H_v*n_seq_tokens*n_seqs floats ]
    //
    // Instead of marking the whole `result` tensor as a graph output (which
    // forces gallocr to preserve ~50 MB per layer × 48 layers of otherwise
    // transient memory and inflates graph_build by ~35 ms), we create a VIEW
    // into the intermediate region and ggml_cpy it into the persistent cache
    // buffer cap->ssm_intermediate_states. The gallocr is unaware of the
    // persistent cache, so verify_build stays cheap. Matches SGLang's
    // mamba_caches.intermediate_ssm pattern.
    if (cap && cap->ssm_intermediate_states && !persist_inter) {
        // Legacy cpy path: only used when the kernel wrote intermediates into
        // its own result region (i.e. when we did NOT use _tree_persist).
        // The _tree_persist variant writes directly to the cache buffer and
        // this cpy becomes redundant, saving ~5-10 ms per verify step.
        const size_t inter_offset =
            S_v * H_v * n_seq_tokens * n_seqs * r_elt        // attn output region
          + S_v * S_v * H_v * n_seqs * r_elt;                // final-state region
        ggml_tensor * inter_view = ggml_view_4d(ctx, result,
            S_v, S_v, H_v, n_seq_tokens,
            S_v * r_elt,
            S_v * S_v * r_elt,
            S_v * S_v * H_v * r_elt,
            inter_offset);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx, inter_view, cap->ssm_intermediate_states));
    }
    } // end of block started at `{` before `const int64_t S_v = head_v_dim;`

after_delta_net:
    // Chunked path writes directly into the same ssm_state slot via its 4D
    // view `s` (which is a live view over ssm_state), using the same cpy
    // pattern the sequential path uses for `new_state`. Sequential path's
    // cpy was already emitted above; guard this second cpy on use_chunked
    // so we don't double-write.
    if (use_chunked) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, s));
    }

    // ── Gated output norm: rms_norm(output) * silu(z_4d)
    ggml_tensor * z_4d = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * output_n = ggml_rms_norm(ctx, output, q35::EPS);
    output_n = ggml_mul(ctx, output_n, L.ssm_norm);
    ggml_tensor * z_silu  = ggml_silu(ctx, z_4d);
    output_n = ggml_mul(ctx, output_n, z_silu);

    // Reshape to [d_inner, n_tokens]
    ggml_tensor * flat = ggml_reshape_3d(ctx, output_n,
        head_v_dim * num_v_heads, n_seq_tokens, n_seqs);

    // Output projection
    ggml_tensor * out = ggml_mul_mat(ctx, L.ssm_out, flat);
    if (L.ssm_out_s) out = ggml_mul(ctx, out, L.ssm_out_s);
    out = ggml_reshape_2d(ctx, out, q35::N_HEAD * 0 + DFLASH27B_TARGET_HIDDEN, n_seq_tokens * n_seqs);
    return out;
}

// ─── Main graph builder ─────────────────────────────────────────────

QwenGraphOutputs build_qwen35_graph(
    ggml_context *         ctx,
    ggml_cgraph *          gf,
    const TargetWeights &  w,
    TargetCache &          cache,
    const QwenGraphInputs & in) {

    const int n_tokens = in.n_tokens;

    // 1. Caller supplies pre-embedded inputs via in.inp_embed (CPU lookup done
    //    ahead of time, zero GPU cost for the embedding table).
    ggml_tensor * inpL = in.inp_embed;

    int fa_idx = 0, dn_idx = 0;

    // If the caller requested capture, size the output list to the total delta-
    // net layer count so we can index by dn_idx as we iterate the layers.
    QwenGraphOutputs og_early{};
    if (in.capture_delta_intermediate) {
        const int n_full_attn = w.n_layer / w.full_attention_interval;
        const int n_delta     = w.n_layer - n_full_attn;
        og_early.delta_captures.resize(n_delta);
    }

    // DFlash target layer IDs for feature capture: {1, 16, 31, 46, 61}
    // HF hidden_states[lid+1] convention — capture AFTER layer 'lid' runs.
    static const int CAPTURE_LAYERS[DFLASH27B_DRAFT_N_TARGET_LAYERS] =
        { 1, 16, 31, 46, 61 };

    const int hidden = w.n_embd;
    const float eps  = q35::EPS;

    for (int il = 0; il < w.n_layer; il++) {
        const TargetLayer & L = w.layers[il];
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);

        ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        ggml_tensor * cur = rms_norm_mul(ctx, inpL, L.attn_norm, eps);

        if (is_attn) {
            cur = build_full_attn_block(ctx, gf, L, cur, in.positions, w.rope_sections,
                                        cache.attn_k[fa_idx], cache.attn_v[fa_idx],
                                        in.attn_mask, in.kv_idxs,
                                        in.kv_start, n_tokens, in.fa_window, in.fa_sink,
                                        &cache, fa_idx);
            fa_idx++;
        } else {
            DeltaNetCapture * cap_ptr = nullptr;
            if (in.capture_delta_intermediate) {
                cap_ptr = &og_early.delta_captures[dn_idx];
                // Point at the persistent per-layer cache buffers so
                // build_delta_net_block can ggml_cpy into them during graph
                // execution. The caller (test_dflash.cpp spec loop) reads from
                // these tensors post-compute; their ->data pointers are always
                // valid because they're cache-resident, not gallocr-managed.
                cap_ptr->ssm_intermediate_states = cache.ssm_intermediate[dn_idx];
                cap_ptr->conv_input              = cache.conv_input_cache[dn_idx];
            }
            cur = build_delta_net_block(ctx, gf, L, cur,
                                        cache.conv_state[dn_idx], cache.ssm_state[dn_idx],
                                        n_tokens, cap_ptr, in.parent_ids);
            dn_idx++;
        }

        // Residual
        cur = ggml_add(ctx, cur, inpSA);

        // Post-attention norm (before FFN)
        ggml_tensor * ffn_residual = cur;
        ggml_tensor * post = rms_norm_mul(ctx, cur, L.attn_post_norm, eps);

        // SwiGLU FFN
        ggml_tensor * ffn = build_swiglu_ffn(ctx, post, L);
        cur = ggml_add(ctx, ffn, ffn_residual);

        // ── DFlash layer feature capture ──
        // Write `cur` into the rolling target_feat buffer. The buffer is a
        // ring of `target_feat_cap` slots; position P maps to slot P%cap.
        // Within a single build call we may straddle the wrap boundary, so
        // we split the copy into up to two contiguous ggml_cpy ops.
        if (in.capture_layers && cache.target_feat) {
            int capture_idx = -1;
            for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
                if (CAPTURE_LAYERS[k] == il) { capture_idx = k; break; }
            }
            if (capture_idx >= 0) {
                const size_t elt        = ggml_element_size(cache.target_feat);
                const size_t col_stride = cache.target_feat->nb[1];
                const int    cap        = cache.target_feat_cap;
                const int    slot_start = in.kv_start % cap;
                const int    pre_n      = std::min(n_tokens, cap - slot_start);
                const int    post_n    = n_tokens - pre_n;

                ggml_tensor * cur_2d = ggml_reshape_2d(ctx, cur, hidden, n_tokens);

                // First slice: [slot_start..slot_start+pre_n) in the ring.
                {
                    const size_t offset =
                        (size_t)slot_start * col_stride +
                        (size_t)capture_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, pre_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        hidden, pre_n, cur_2d->nb[1], 0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }

                // Second slice: wrap-around at [0..post_n) if needed.
                if (post_n > 0) {
                    const size_t offset =
                        (size_t)capture_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, post_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        hidden, post_n, cur_2d->nb[1],
                        (size_t)pre_n * cur_2d->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }
            }
        }

        inpL = cur;
    }

    // 2. Final norm
    ggml_tensor * out = rms_norm_mul(ctx, inpL, w.out_norm, q35::EPS);

    // 3. LM head
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, out);
    if (w.output_s) logits = ggml_mul(ctx, logits, w.output_s);
    ggml_set_name(logits, "logits");

    ggml_build_forward_expand(gf, logits);

    QwenGraphOutputs og = std::move(og_early);
    og.logits = logits;
    return og;
}

} // namespace dflash27b
