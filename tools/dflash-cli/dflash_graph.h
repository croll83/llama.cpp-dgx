// Shared inputs/outputs for the DFlash draft graph builder.
#pragma once

#include "ggml.h"
#include <vector>

namespace dflash27b {

struct DraftWeights; // fwd

struct DraftGraphInputs {
    int           ctx_len;          // length of target_hidden_cat along ne[1]
    ggml_tensor * noise_embed;      // [hidden, q_len=16, 1] f32
    ggml_tensor * target_hidden_cat;// [5*hidden, ctx_len, 1] f32
    ggml_tensor * positions_q;      // [q_len] i32   values [ctx_len..ctx_len+q_len-1]
    ggml_tensor * positions_k;      // [ctx_len+q_len] i32   values [0..ctx_len+q_len-1]
    // Optional: if non-null, the graph projects final hidden states through
    // this LM head (shape [hidden, vocab]) and returns logits instead of
    // hidden states. Used for DFlash integration where the draft shares the
    // target's lm_head.
    ggml_tensor * lm_head;

    // Causal SWA + full-causal masks for the draft attention. When the draft
    // model is non-causal (Qwen3.5 generation, swa_window == 0) both stay null
    // and flash_attn gets nullptr. For Qwen3.6-27B-DFlash (swa_window > 0)
    // SWA layers use attn_mask_swa while the full-attention layer(s) use
    // attn_mask_full (causal without the window constraint). Shape on both:
    // [kv_pad, q_pad] F16, 0 for kept and -INF for masked positions.
    ggml_tensor * attn_mask_swa  = nullptr;
    ggml_tensor * attn_mask_full = nullptr;

    // Per-layer flag: true => use attn_mask_swa, false => use attn_mask_full.
    // Null when the draft is non-causal (both masks null anyway).
    const std::vector<bool> * layer_is_swa = nullptr;
};

struct DraftGraphOutputs {
    ggml_tensor * hidden_states;    // [hidden, q_len, 1]  (always set)
    ggml_tensor * logits;           // [vocab, q_len, 1]   (non-null iff lm_head was provided)
};

DraftGraphOutputs build_draft_graph(
    ggml_context *            ctx,
    const DraftWeights &      w,
    const DraftGraphInputs &  in);

} // namespace dflash27b
