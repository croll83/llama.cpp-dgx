# DFlash MTP integration (Phase C)

This fork of llama.cpp v5 integrates the [DFlash](https://github.com/z-lab/DFlash)
Multi-Token-Prediction speculative decoding (draft by [z-lab](https://huggingface.co/z-lab))
directly into `llama-server`. When `--dflash` is set on the CLI, the per-slot
decode path in `update_slots()` is replaced with a native call into the
`dflash27b` library; prompts are tokenized / chat-templated by `llama-server`
as usual, DDtree verify runs on-device via the library's own target cache,
SSM state and draft graph.

## Build

```bash
cd llama-cpp-v5
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120;121"
cmake --build . -j$(nproc) --target llama-server
```

Outputs:
- `build/bin/llama-server`       — main binary (works with or without `--dflash`)
- `build/bin/llama-dflash`       — standalone CLI for direct dflash experiments
- `build/bin/llama-dflash-server`— Phase 2 subprocess-wrapper server (kept for
  diagnostic comparison; `llama-server --dflash` is the preferred path)

## Run (Phase C)

```bash
llama-server \
    -m /path/to/target.gguf \
    --dflash \
    --dflash-draft /path/to/draft_model.safetensors \
    --dflash-budget 22 \
    --dflash-max-ctx 16384 \
    --host 0.0.0.0 --port 30000 \
    -c 16384 -np 2 -ngl 99 \
    --alias my-model
```

`-np N` spins up `N` slots, each with its own DFlash session but sharing
the target GGUF + draft safetensors on VRAM (loaded once via
`dflash_weights_load`). GPU compute still serializes on a single device;
the split buys per-slot state isolation (KV + SSM) so concurrent chat
turns don't corrupt each other.

Then the standard OpenAI API is served:
- `GET  /v1/models`              — lists `my-model`
- `POST /v1/chat/completions`    — non-stream + SSE stream supported

## Flags

| Flag                  | Default         | Notes                                             |
|-----------------------|-----------------|---------------------------------------------------|
| `--dflash`            | off             | Enable DFlash slot mode                           |
| `--dflash-draft PATH` | (required)      | DFlash draft `.safetensors` file                  |
| `--dflash-budget N`   | 22              | DDtree node budget per verify                     |
| `--dflash-max-ctx N`  | = `--ctx-size`  | Max context for the DFlash target cache           |

## MVP caveats

- ~~`-np 1` only (single slot).~~ *(Fixed — weights load once via
  `dflash_weights_load`, per-slot sessions allocated with
  `dflash_session_create_shared`. Concurrent slots produce distinct per-slot
  output; GPU compute still serializes on the single CUDA device, but
  KV + SSM state is fully isolated per slot so slot 0 and slot 1 can hold
  independent chat histories.)*
- Greedy decoding only — the llama-server sampler is bypassed for tokens
  produced by DFlash. `temperature`, `top_p`, `grammar`, `logit_bias` are
  ignored.
- ~~The main model weights are loaded twice~~ *(Fixed in commit 0efab257a — main model now loads vocab_only when --dflash is set, saving ~14 GB VRAM. process_slot_dflash restores slot.n_ctx to the dflash target cache size so the normal in-context-size check keeps working.)*
- ~~Prompt caching only hits when the new prompt is a strict extension of
  the cached token stream.~~ *(Partially fixed — the session now drops
  an SSM + conv anchor ~32 tokens before the end of each prefill, and
  process_slot_dflash rewinds to it when the follow-up LCP clears the
  anchor but doesn't match the full cached stream. This recovers the
  common chat-template case where only the last ~20 tokens re-tokenize
  differently across turns. Works both ways: strict-extension still
  uses the classic append path; mid-prompt divergence still has to
  fall back to reset. APIs: `dflash_session_anchor_pos` +
  `dflash_session_rewind_to_anchor`.)*
- `cpy.cu` in ggml-cuda has no `F32 → TURBO*/TQ3_0` kernels, so dflash's
  explicit Q/K/V → KV copy cannot use TurboQuant-KV types yet. Follow-up.

## Library API (`tools/dflash-cli/dflash27b.h`)

Intended for direct use in other integrations. Use the shared-weights form
when building multi-slot hosts (loads weights once for the process):

```c
// Shared weights (multi-slot):
dflash_weights_t * dflash_weights_load(target_gguf, draft_safetensors, backend);
dflash_session_t * dflash_session_create_shared(weights, params, backend);
void               dflash_weights_free(weights);

// Self-contained (one-shot CLIs, single slot):
dflash_session_t * dflash_session_create(target_gguf, draft_safetensors,
                                          params, backend);

// Common:
int  dflash_session_run(s, prompt_ids, n_prompt, n_gen, append_mode,
                         token_cb, user_data);
int  dflash_session_kv_end(s);
int  dflash_session_reset(s);
void dflash_session_destroy(s);

// Per-run stats + cross-turn rewind:
void dflash_session_get_last_stats(s, /*out*/ &stats);
int  dflash_session_anchor_pos(s);          // 0 if no anchor yet
int  dflash_session_rewind_to_anchor(s);    // 0 on success, -1 on error
```

Lifetime rule: every session created from shared weights must be destroyed
before `dflash_weights_free` is called. Set `append_mode=1` to reuse cached
KV + SSM state from a previous run on the same session (callers must ensure
`prompt_ids` is the delta past `dflash_session_kv_end`).

## Measured results (GB10, 27B TQ3_4S target + Qwen3.6 draft, ddtree-budget=22)

Single-request baseline:

| Metric                         | Value          |
|--------------------------------|----------------|
| llama-server VRAM (vocab_only) | **19.7 GiB**   |
| VRAM without vocab_only        | 35.3 GiB       |
| Sequential requests (50 × max_tokens=20) | 100% success |
| Accept rate (per verify)       | ~40%           |
| tok/s wall-clock (predictable prompt) | 30-40   |
| tok/s wall-clock (reasoning prompt)   | 15-20   |
| tok/s dflash-internal (predicted_per_second) | 100-210 |
| Streaming SSE (delta per token)| works          |

### Multi-turn chat with anchor rewind (16K ctx, 641-token system prompt)

Ran 4 consecutive `/v1/chat/completions` turns on the same slot, each
appending user+assistant to the history (see `tools/dflash-cli/bench_chat.py`
in the tree for the exact prompt). The anchor rewind kicks in starting at
turn 2 and drops prefill by ~6× because the ~600 token system prompt
prefix doesn't have to be re-ingested on every turn.

| Turn | prompt_n | prefill_ms | decode_ms | tok/s | cached |
|------|---------:|-----------:|----------:|------:|-------:|
| 1 (fresh)  | 641 | 5316 | 2264 | 26.5 | 0   |
| 2 (anchor) | 106 |  895 | 2600 | 23.1 | 624 |
| 3 (anchor) | 114 | 1001 | 3130 | 19.2 | 704 |
| 4 (anchor) | 108 |  898 | 2974 | 20.2 | 800 |

`cached` is the number of tokens that came from the anchor snapshot
(surfaced to the client via `timings.cache_n` / `usage.prompt_tokens_details.cached_tokens`).
Wall-clock drops from 7.6s on the cold turn to ~3.5-4.2s on subsequent
turns even though the history keeps growing.

## Known limitations / follow-ups

1. Greedy only — llama-server's sampler is bypassed. Temperature /
   top_p / grammar / logit_bias are ignored.
2. Prompt caching: strict extension works; cross-turn with chat-template
   re-tokenization now works when divergence happens in the last ~32
   tokens via anchor rewind (see MVP caveats). Earlier divergence still
   triggers full reset. Multi-slot routing needs `--slot-prompt-similarity`
   to pin follow-ups to the slot that has the warm anchor — default LRU
   can land on the wrong slot.
3. TurboQuant V-cache not wired in — ggml-cuda/cpy.cu has no F32 → TQ3/
   TURBO* kernels so dflash's Q/K/V→KV copy can't use them.
4. GPU compute still serializes on one CUDA device — two concurrent slots
   time-share the same device. True parallel compute would require
   multi-GPU shards or CUDA stream pipelining inside session_run.
