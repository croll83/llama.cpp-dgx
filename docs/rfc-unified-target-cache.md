# RFC: Unify target K/V cache between standard llama_kv_cache and dflash session

**Status**: Phase 1 + 2 + 4-defensive + 5 LANDED 2026-04-29.
Commits: `cab95737a` (anchor 4→2 + RFC), `67b9e0c54` (Phase 2.1 dflash
borrow API), `03ea32aef` (Phase 2.2 wiring + public llama API). Followup
commit adds Phase 4 defensive desync guard. Full Phase 4 (SSM state
sharing) deferred indefinitely — out of scope for the savings we wanted.

**Scope**: tools/dflash-cli + src/llama-kv-cache + src/llama-context + tools/server  
**Estimate (revised)**: 1 day for Phase 1+2+3+4-defensive+5 (vs original 2-4 weeks). Mostly because Phase 1 kernel-stride validation by code inspection found NO patches needed.  
**Savings achieved**: **−9 GiB resident** on np=2 (predicted ~9.1 GiB, measured 63 GiB → 54 GiB after enabling `DFLASH27B_SHARE_KV=1`).

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

### Phase 4: cross-path cache hit (deferred — see Status)

The naive plan was: when the standard llama_decode path writes K/V to a
position, the dflash session sees that data on a subsequent text-only
call to the same slot. Update dflash's `kv_end` and `prefill_kv_offset`
to honor the standard path's writes.

**Reality after implementation**: this isn't enough. K/V sharing is half
the picture — the dflash custom graph also maintains its own SSM state
for the 48 GDN layers (in `cache.ssm_state`), K_combined sink buffers,
and anchor snapshots. None of those have an equivalent on the standard
llama_decode path. So even if dflash detects the K/V advance from a
vision request, it cannot continue from that position without rebuilding
SSM state — which means re-running the full forward through all the
prefix anyway.

**Phase 4 defensive (delivered)**: detect kv_end desync between dflash's
internal counter and the server's prefill_off. When mismatch (e.g. after
a vision request used the slot via the standard path), force
`dflash_session_reset` and full re-prefill. Costs one redundant prefill
round but keeps the cache consistent. The standard path's K/V writes are
preserved bit-for-bit since both paths produce identical K/V projections
from the same target weights — dflash's prefill just overwrites with the
same values, then advances SSM state correctly.

**Phase 4 full** (true cross-path cache hit on text↔vision interleave)
would require sharing SSM state too — multi-week refactor of
`llama_memory_recurrent` to expose F32 SSM tensors per layer to dflash.
Deferred indefinitely; the use-case (frequent text↔vision interleave on
same slot) isn't load-bearing for hermes-dark.

### Phase 5: cleanup (delivered)

Done in this RFC update + commit `03ea32aef` and the followup that
landed Phase 4-defensive:

- Documented the `DFLASH27B_SHARE_KV` flag in README, including the
  layout-stride caveat and the kv_end-desync defensive reset.
- Updated the memory savings stack table to include the share-kv row.
- The `DFLASH27B_KV_K` / `_V` env overrides are KEPT (they still apply
  on the legacy alloc path; share-kv ignores them since the type comes
  from `-ctk`/`-ctv`).
- Dead code removal in TargetCache for the standalone allocation path
  is NOT done — the legacy path is still default until production
  validation flips share-kv on by default.

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

**Not delivered** (Phase 4 full): cross-path cache hit on text↔vision
context switches still costs one redundant prefill round (now via the
defensive `dflash_session_reset` rather than a buffer separation, but
same latency). Quantitatively: ~30-60s for a 22K-token context on GB10.
Negligible cost on a text-only agent workload (the defensive check is a
single int compare, never triggers).

## Rollout

- Phases 1+2+3+4-defensive+5 landed 2026-04-29.
- Default for `DFLASH27B_SHARE_KV` is **0 (off)** until production
  validation completes a few days of hermes-dark traffic.
- Flip to default-on after that, and remove the env (always-on) in a
  follow-up commit.

## Status today (2026-04-29 final)

DELIVERED: Phase 1 (validation by kernel inspection — no patches needed),
Phase 2 (borrow API + server-context wiring), Phase 4-defensive (kv_end
desync guard), Phase 5 (README + RFC docs).

DEFERRED: Phase 4 full (SSM state sharing across paths). Out of scope
given hermes-dark's text-only-dominant workload.

PRODUCTION STATE: server up with `DFLASH27B_SHARE_KV=1` env, AEON-7 XS
body, V=Q8 KV, np=2, -c 262144. Stable RAM **~54 GiB total** (38 GiB net
of OS overhead), full 131K context per slot on both dflash agent and
mmproj vision paths. Validated text-only at 24, 5K, 22K, 45K context
points. Vision path validated by code inspection (Phase 1 strides
confirmed compatible) but not by live request — todo on the validation
sweep.
