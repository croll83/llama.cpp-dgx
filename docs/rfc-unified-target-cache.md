# RFC: Unify target K/V cache between standard llama_kv_cache and dflash session

**Status**: draft, author Claude with Marco, 2026-04-29  
**Scope**: tools/dflash-cli + src/llama-kv-cache + tools/server  
**Estimate**: 2-4 weeks (high uncertainty on FA-kernel stride compatibility)  
**Savings target**: -7 GiB resident on np=2 + remove redundant prefill on text↔vision context switches

## Problem

When `--dflash --mmproj` are both set, llama-server allocates **two
independent K/V caches** for the same target model:

- **`llama_kv_cache`** (allocated by `llama-context` init): used by the
  standard `llama_decode` path, which serves multimodal/vision requests.
  At `-c 262144 -np 2` with `-ctk q8_0 -ctv q8_0` this is **8.7 GiB**.
- **`dflash session cache`** (allocated by `qwen35_target_graph::create_target_cache`):
  used by the dflash custom graph for text-only requests. At
  `--dflash-max-ctx 131072 -np 2` with `DFLASH27B_KV_K/V=q8_0` this is
  another **9.1 GiB** (main K + V; sink K_combined/V_combined and SSM
  state add ~3 GiB more but those have no equivalent on the standard side).

Both caches store K/V for the **same 16 full-attention layers of the same
target model** with the **same projection weights**. The K/V values
written are bit-identical (same model, same input embeddings → same
projections). The duplication exists purely because the two paths were
historically wired through different ggml_context allocations.

**Cost in production**:

1. **Memory**: ~8.7 GiB redundant (the standard side's K+V duplicate of
   the dflash side's K+V for the same model).
2. **Latency on context switches**: when hermes alternates between
   text-only (dflash path) and image-bearing (standard path) requests on
   the same slot, the second path has to re-prefill the full prompt
   because the previous path's KV is in a separate buffer.
3. **Cache reuse mismatch**: llama-server's prompt-cache (the cross-request
   prefix-match index) stores hashes per slot. With two caches, a hit on
   one path doesn't help the other path.

## Layout analysis

### llama_kv_cache (standard side)

Construction at `src/llama-kv-cache.cpp:373`:

```cpp
const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il)
                                       : hparams.n_embd_v_gqa_max();
ggml_tensor * k = ggml_new_tensor_3d(ctx, layer_type_k,
                                     n_embd_k_gqa, kv_size, n_stream);
ggml_tensor * v = ggml_new_tensor_3d(ctx, layer_type_v,
                                     n_embd_v_gqa, kv_size, n_stream);
```

For Qwen3.6 with `flash_attn = on` (so `v_trans = false`):
- `n_embd_k_gqa = head_dim × n_kv_heads = 256 × 4 = 1024`
- `kv_size = n_ctx_per_seq × n_seq_max effective = 131072`
- `n_stream = 1` (or `n_seq_max` if multistream enabled)

So K shape: `[1024, 131072, 1]` Q8_0, V same shape.

### dflash TargetCache (custom side)

Construction at `tools/dflash-cli/qwen35_target_graph.cpp:158-163`:

```cpp
ggml_tensor * K = ggml_new_tensor_3d(out.ctx, kv_k_type,
                                     q35::HEAD_DIM, max_ctx, q35::N_HEAD_KV);
ggml_tensor * V = ggml_new_tensor_3d(out.ctx, kv_v_type,
                                     q35::HEAD_DIM, max_ctx, q35::N_HEAD_KV);
```

So K shape: `[256, 131072, 4]` Q8_0, V same shape.

### Equivalence

- Total elements: `1024 × 131072 × 1 = 256 × 131072 × 4 = 134,217,728`
  ✓
- Total bytes Q8_0: 134M × 34/32 = ~143 MB per layer × 16 layers × 2
  (K+V) × 2 slots ≈ **9 GiB per side**, matching observed.
- Logical dimension axis assignment: differs.
  - llama: `[d × h_kv, c, s]` packs the heads onto axis 0
  - dflash: `[d, c, h_kv]` separates heads onto axis 2

The two are **isomorphic via permute** when `n_stream = 1`. The dflash
view of llama's K layer is:
```cpp
ggml_view_3d(ctx, llama_K_il,
             /* ne0 */ HEAD_DIM,        // 256
             /* ne1 */ MAX_CTX,         // 131072
             /* ne2 */ N_HEAD_KV,       // 4
             /* nb1 */ n_embd_k_gqa * type_size,  // 1024 elems = 1088 B (Q8_0)
             /* nb2 */ HEAD_DIM * type_size / qk, // 256/32 = 8 blocks = 272 B
             /* off */ 0);
```

The strides are **non-standard**: `nb2 < nb1`. Natural-order ggml tensors
have `nb0 ≤ nb1 ≤ nb2`. Some kernels assume natural strides. Need to
verify each kernel that touches dflash K/V tensors accepts custom strides.

## Risk: kernel stride compatibility

Kernels that read dflash K/V:

1. **FA kernel** (`ggml-cuda/fattn.cu`): reads K and V via head/position
   indexing. Has strides_K[0..2], strides_V[0..2]. Need to verify it
   handles `nb_K[2] < nb_K[1]`.
2. **`cpy_q_q`** (our recent template, `ggml-cuda/cpy.cu:178`): reads
   `(i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03*nb03` — explicit byte-offset
   math, accepts any strides. ✓
3. **`set_rows_cuda<i32, q8_0>`** (sink path): reads dst tensor strides
   via nb. Need to verify.
4. **GDN attention block** (qwen35_target_graph.cpp): builds K_combined
   sink + window. Uses `ggml_cpy` and `ggml_view_3d`. Needs strides
   propagated correctly.

**Mitigation if FA kernel fails**:
- Allocate target K/V in dflash format on the standard side too — flip
  the standard `llama_kv_cache` to use dflash layout via a new
  `--kv-layout-dflash` flag. Backward-incompatible for non-dflash
  builds; only enable when `--dflash` is set.

## Implementation phases

### Phase 1: layout-compatibility validation (1-2 days)

- Write a unit-test harness that:
  1. Allocates a Q8_0 tensor in llama_kv_cache layout `[1024, 131072, 1]`
  2. Creates a non-standard view with dflash strides `[256, 131072, 4]`
  3. Runs FA kernel + cpy + set_rows on the view
  4. Compares results to a reference run on a natively-shaped dflash
     tensor.
- Check fattn.cu handles `nb_K[2] < nb_K[1]`. Likely needs a force-VEC
  predicate update if MMA path doesn't support it.
- Fallback: if FA breaks, abandon phase 2 and pursue alternate plan
  (allocate llama_kv_cache in dflash layout — same memory savings, less
  flexibility on the standard path).

### Phase 2: thread llama_kv_cache pointers into dflash (3-5 days)

- New API: `dflash_session_borrow_llama_kv(s, llama_context *)` — takes
  the slot's llama_context, extracts the per-layer K/V tensors via
  llama_kv_cache::layers[il].k/v, stores pointers in TargetCache.
- Refactor `create_target_cache` to skip K/V allocation when borrow
  pointers are provided. Allocate only the dflash-specific structures
  (SSM, K_combined, anchor snapshots) which have no llama_kv_cache
  equivalent.
- Update qwen35_target_graph build path to view through borrowed K/V
  tensors instead of native dflash tensors.

### Phase 3: wire server-context (1-2 days)

- `tools/server/server-context.cpp::process_slot_dflash`: pass the slot's
  `llama_context *` into `dflash_session_borrow_llama_kv` at slot
  init.
- Make the borrow path opt-in via a flag `--dflash-share-kv` initially,
  default off. Enable by default once stable.

### Phase 4: cross-path cache hit (2-3 days)

- When the standard llama_decode path writes K/V to a position, the
  dflash session sees that data on a subsequent text-only call to the
  same slot.
- Update dflash's `kv_end` and `prefill_kv_offset` to honor the standard
  path's writes (currently those are dflash-internal counters).
- Test: send image+text request → reply → text-only follow-up → verify
  no re-prefill of the image-context text.

### Phase 5: cleanup (1 day)

- Remove `DFLASH27B_KV_K` / `DFLASH27B_KV_V` env overrides (the K/V type
  comes from `-ctk`/`-ctv` flags now).
- Document the share-kv flag in README.
- Delete dead code in TargetCache for the standalone allocation path.

## Memory savings detail (post phase 5)

| voce | before | after | Δ |
|---|---:|---:|---:|
| llama_kv_cache K (standard side) | 4.4 GiB | 4.4 GiB | 0 |
| llama_kv_cache V (standard side) | 4.4 GiB | 4.4 GiB | 0 |
| dflash K main (per slot × 2) | 4.55 GiB | 0 (borrowed) | −4.55 |
| dflash V main (per slot × 2) | 4.55 GiB | 0 (borrowed) | −4.55 |
| dflash V_combined sink | 1.4 GiB | 1.4 GiB | 0 |
| dflash SSM state (1 anchor × 2 slot) | 1.2 GiB | 1.2 GiB | 0 |
| compute scratch | ~2 GiB | ~2 GiB | 0 |
| **total target-side** | **~22 GiB** | **~13 GiB** | **−9.1 GiB** |

Also removed: redundant prefill on text↔vision context switches, prompt
cache cross-path hits.

## Rollout

- Phase 1-2 land behind `--dflash-share-kv=experimental`.
- Production keeps the standalone path until phase 3 stabilises.
- Once phase 4 completes, flip default to share-kv on text+vision
  configs.

## Status today (2026-04-29)

Investigation done. Layout shown to be view-equivalent. Strides
non-standard. Phase 1 (kernel stride validation) is the next concrete
step. Has to wait for a non-production window since it requires server
restart per iteration.
