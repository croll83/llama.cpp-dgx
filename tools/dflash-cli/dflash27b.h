// dflash27b — CUDA library for DFlash speculative decoding of Qwen3.5/3.6-27B
// with the z-lab DFlash draft model.
//
// Public API: a `dflash_session_t` handle owns the target cache / draft
// weights / step graph and exposes the three verbs needed by a host server:
//
//   create → prefill → decode_step ×N → prefill (append) → decode_step ...
//
// `prefill(..., kv_start)` allows APPEND-mode continuation when the caller
// has already committed tokens 0..kv_start-1 in a previous call, enabling
// prompt caching at the server layer.

#ifndef DFLASH27B_H
#define DFLASH27B_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Model config (compile-time constants for this target/draft pair) ─

#define DFLASH27B_TARGET_HIDDEN        5120
#define DFLASH27B_TARGET_LAYERS        64
// NOTE: the `DFLASH27B_TARGET_N_*` / `_HEAD_DIM` macros below are DRAFT
// dimensions (z-lab draft: 32 Q heads, 8 KV heads, 128 head_dim). The TARGET
// Qwen3.5-27B qwen35 hybrid uses 24 Q heads, 4 KV heads, 256 head_dim, which
// live in `internal.h` (n_embd_head_k/v, N_HEAD, N_HEAD_KV). Naming is
// historical — do not change without updating safetensors_draft.cpp +
// qwen3_dflash_graph.cpp which consume these as draft-side constants.
#define DFLASH27B_TARGET_N_HEADS       32
#define DFLASH27B_TARGET_N_KV_HEADS    8
#define DFLASH27B_TARGET_HEAD_DIM      128
#define DFLASH27B_TARGET_INTERMEDIATE  17408
#define DFLASH27B_TARGET_VOCAB         248320
#define DFLASH27B_ROPE_THETA           10000000.0f
#define DFLASH27B_RMS_EPS              1e-6f

#define DFLASH27B_DRAFT_LAYERS         5
#define DFLASH27B_DRAFT_BLOCK_SIZE     16
#define DFLASH27B_DRAFT_N_TARGET_LAYERS 5  // fc projects 5*hidden -> hidden
#define DFLASH27B_DRAFT_MASK_TOKEN_ID  248070

// ─── Diagnostics ──────────────────────────────────────────────────

// Most recent error from any API call. Thread-safe.
const char * dflash27b_last_error(void);

// ─── Session API (public) ─────────────────────────────────────────

// Opaque session handle. Owns the target cache, step graph and all
// per-request scratch. Weights are either owned (legacy single-session
// create) or borrowed from a shared dflash_weights_t. Not thread-safe:
// callers must serialize access around a single session instance.
typedef struct dflash_session_s dflash_session_t;

// Opaque shared-weights handle. Holds target GGUF + draft safetensors on
// the GPU backend and can be shared across multiple sessions (e.g. one
// per llama-server slot). The caller owns the lifetime: it must outlive
// every session created from it and be released with dflash_weights_free.
typedef struct dflash_weights_s dflash_weights_t;

typedef struct {
    int   max_ctx;              // max tokens the target KV cache will hold
    int   ddtree_budget;        // DDtree node budget (22 matches lucebox default)
    int   ddtree;               // 1 = DDtree verify (tree), 0 = linear chain verify
    int   fast_rollback;        // 1 = kernel-level SSM rollback for tree verify
    int   ddtree_chain_seed;    // 1 = pre-seed full chain before best-first
    float ddtree_temp;          // softmax temperature for top-K draft extraction
    int   kv_tbq;               // 1 = align mask stride to 256 (TBQ FA kernels)
    int   prefill_ubatch;       // 0 = auto (16 for <=2048 prompts, 192 otherwise)
    int   fa_window;            // sliding window for FA layers (0 = full attn). PR #26 default 2048.
    int   fa_sink;              // attention sinks (Xiao 2023): always keep first K KV positions visible. 0 = no sink.
} dflash_session_params_t;

// Sensible defaults matching lucebox test_dflash --fast-rollback --ddtree --ddtree-budget=22.
dflash_session_params_t dflash_session_default_params(void);

// Forward decl — the session creation takes a ggml CUDA backend handle that
// lives in ggml-base.h. We forward-decl to keep this header minimal.
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

// Create a session bound to the given target GGUF and draft safetensors
// file. The session internally owns the weights and frees them on destroy.
// Convenience wrapper over dflash_weights_load + dflash_session_create_shared;
// prefer the shared form when building multi-slot servers to avoid loading
// ~20 GiB of weights per slot.
// Returns NULL on error; use dflash27b_last_error() for details.
dflash_session_t * dflash_session_create(const char * target_gguf,
                                          const char * draft_safetensors,
                                          const dflash_session_params_t * params,
                                          ggml_backend_t backend);

// Load target GGUF + draft safetensors once into backend-resident buffers
// usable by many sessions. Returns NULL on error.
dflash_weights_t * dflash_weights_load(const char * target_gguf,
                                        const char * draft_safetensors,
                                        ggml_backend_t backend);

// Borrow target weights from an already-loaded llama_model — saves ~15 GB
// on GPU vs dflash_weights_load when llama-server keeps a resident model
// for mmproj support. The draft is still loaded normally from
// . The caller must keep the host llama_model alive for
// as long as the returned dflash_weights_t is in use; freeing the weights
// does NOT free the host model.
//
//  may be empty if the host llama_model already keeps
// tok_embd on a host (CPU-mapped) buffer; otherwise it is used to mmap the
// embedding rows for the CpuEmbedder.
dflash_weights_t * dflash_weights_load_borrow(const struct llama_model * lm,
                                               const char * target_gguf_path,
                                               const char * draft_safetensors,
                                               ggml_backend_t backend);

// Free shared weights. Sessions created from these weights must all be
// destroyed first — it is a use-after-free otherwise.
void dflash_weights_free(dflash_weights_t * w);

// Create a session that borrows the given shared weights. The caller
// retains ownership of `weights`; the session holds a non-owning pointer.
// Returns NULL on error; the weights handle is unchanged on failure.
dflash_session_t * dflash_session_create_shared(dflash_weights_t * weights,
                                                 const dflash_session_params_t * params,
                                                 ggml_backend_t backend);

void dflash_session_destroy(dflash_session_t * s);

// Reset the target cache + SSM / conv state + step graph. The next prefill
// must use kv_start=0. Weights and draft stay resident.
int dflash_session_reset(dflash_session_t * s);

// Prefill `n_tokens` tokens starting at absolute KV position `kv_start`.
//
// kv_start must equal dflash_session_kv_end(s) — callers that want to
// rewind must call dflash_session_reset() first (the SSM/conv state cannot
// be rolled back to an arbitrary position without a snapshot).
//
// Returns 0 on success, -1 on error.
int dflash_session_prefill(dflash_session_t * s,
                            const int32_t * tokens, int n_tokens,
                            int kv_start);

// Emit the next committed token and optionally advance decode. This is a
// token-granular step — the session internally runs one DDtree verify round
// (or chain verify, per params) and commits up to `max_out` tokens. Returns
// the number of tokens written to `out` (0..max_out). 0 means no progress
// (hit internal limit).
int dflash_session_decode_step(dflash_session_t * s,
                                int32_t * out, int max_out);

// Absolute KV position at end of the prefilled + decoded region.
int dflash_session_kv_end(const dflash_session_t * s);

// Last logits seed token (useful for picking up bonus after prefill).
int32_t dflash_session_last_tok(const dflash_session_t * s);

// Per-request stats captured during the most recent dflash_session_run.
// Populated on success (return >= 0); contents are undefined after error.
typedef struct {
    int    n_generated;       // committed tokens emitted through the callback
    int    n_draft_steps;     // number of draft-forward / verify rounds
    int    n_accept_sum;      // total draft tokens accepted across all rounds
    int    prefill_tokens;    // tokens processed in the prefill phase
    double prefill_seconds;   // wall clock of prefill
    double decode_seconds;    // wall clock of the draft+verify loop
} dflash_session_stats_t;

// Copy the last-run stats into `out`. Zero-fills `out` if the session has
// never been run. Safe to call between runs; overwritten on the next run.
void dflash_session_get_last_stats(const dflash_session_t * s,
                                    dflash_session_stats_t * out);

// Absolute KV position of the session's most recent end-of-prefill
// snapshot. Returns 0 if no prefill has run yet. For multi-anchor
// sessions this returns the MAXIMUM stored anchor position; callers
// that want to pick an anchor by an upper bound (e.g. lcp of a new
// prompt) should prefer dflash_session_best_anchor_pos below.
int dflash_session_anchor_pos(const dflash_session_t * s);

// Returns the largest stored anchor position <= target_pos, or 0 if no
// anchor satisfies that bound. Useful when the caller knows it can
// only trust a rewind that sits within its lcp against the cached
// token stream: passing target_pos = lcp yields the best rewind candidate.
int dflash_session_best_anchor_pos(const dflash_session_t * s, int target_pos);

// Restore SSM + conv state from the MAX-positioned anchor snapshot and
// rewind kv_end back to that position. Equivalent to
// dflash_session_rewind_to_pos(s, dflash_session_anchor_pos(s)).
// Returns 0 on success, -1 on error.
int dflash_session_rewind_to_anchor(dflash_session_t * s);

// Rewind to the best anchor whose position is <= target_pos (typically
// the caller's lcp). Returns the actual rewind position on success
// (0 < ret <= target_pos), or -1 if no suitable anchor exists.
int dflash_session_rewind_to_pos(dflash_session_t * s, int target_pos);

// Callback invoked for each committed token. Return non-zero to keep going,
// zero to abort (server-side generation stop).
typedef int (*dflash_token_cb)(int32_t token, void * user_data);

// Runs one request end-to-end: optional reset + prefill + decode.
//   prompt_ids / n_prompt  tokens to prefill (full prompt for RESET mode,
//                          delta tokens for APPEND mode)
//   n_gen                  max tokens to generate
//   append_mode            0 = reset cache before prefill, 1 = continue
//                          from dflash_session_kv_end() (prompt caching)
//   cb / user_data         optional per-token callback
// Returns number of committed tokens on success, -1 on error.
int dflash_session_run(dflash_session_t * s,
                       const int32_t * prompt_ids, int n_prompt,
                       int n_gen,
                       int append_mode,
                       dflash_token_cb cb,
                       void * user_data);

#ifdef __cplusplus
}
#endif

#endif // DFLASH27B_H
