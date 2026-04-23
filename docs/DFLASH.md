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
    -c 16384 -np 1 -ngl 99 \
    --alias my-model
```

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

- `-np 1` only (single slot). Multi-slot would need one `dflash_session_t`
  per slot.
- Greedy decoding only — the llama-server sampler is bypassed for tokens
  produced by DFlash. `temperature`, `top_p`, `grammar`, `logit_bias` are
  ignored.
- The main model weights are loaded twice (once by `llama_model` for
  tokenizer / chat-template / sampler; once by `dflash_session` for
  verify). The second copy can be elided later by switching the main load
  to `vocab_only` when `--dflash` is set.
- Prompt caching only hits when the new prompt is a strict extension of
  the cached token stream. Chat-template re-tokenization across turns can
  break strict extension; a follow-up will either align
  `slot.prompt.tokens` with the template view or teach the dflash session
  to rewind KV via an SSM snapshot.
- `cpy.cu` in ggml-cuda has no `F32 → TURBO*/TQ3_0` kernels, so dflash's
  explicit Q/K/V → KV copy cannot use TurboQuant-KV types yet. Follow-up.

## Library API (`tools/dflash-cli/dflash27b.h`)

Intended for direct use in other integrations:

```c
dflash_session_t * dflash_session_create(target_gguf, draft_safetensors,
                                          params, backend);
int  dflash_session_run(s, prompt_ids, n_prompt, n_gen, append_mode,
                         token_cb, user_data);
int  dflash_session_kv_end(s);
int  dflash_session_reset(s);
void dflash_session_destroy(s);
```

Set `append_mode=1` to reuse cached KV + SSM state from a previous run
(callers must ensure `prompt_ids` is the delta past `dflash_session_kv_end`).
