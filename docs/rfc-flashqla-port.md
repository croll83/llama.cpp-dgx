# RFC: Port FlashQLA chunked GDN forward to llama.cpp-dgx

**Status**: Phase 0 reconnaissance done 2026-04-30. Implementation pending — multi-day work.  
**Scope**: ggml/src/ggml-cuda/gated_delta_net.cu (new chunk-fused path) + dispatcher logic in tools/dflash-cli/qwen35_target_graph.cpp  
**Estimate**: 3-5 days for a correctness-validated port; another 1-2 days for performance tuning on GB10.  
**Savings target**: 2-3× forward speedup on GDN layers (= 48 of 64 layers in Qwen3.6) → estimated +30-50 % on prefill speed for long-context agent workloads. The GDN forward is the dominant cost on dflash prefill at ≥20K context (currently ~50 s for 22K tokens; should drop to ~25-30 s with FlashQLA).

## Context

The Qwen team released [FlashQLA](https://github.com/QwenLM/FlashQLA) on
2026-04-29: TileLang-built linear-attention kernels purpose-built for
agentic AI on consumer/Blackwell hardware. Headline numbers: **2-3×
forward speedup, 2× backward** vs the FLA Triton baseline.

For our llama.cpp-dgx fork, the relevant kernel is the chunked
`fused_chunk_gdr_fwd` — it processes a sequence in chunks (default 64
tokens) with parallelism-friendly math and warp-specialized pipelining,
in contrast to our current `gated_delta_net.cu` which is a per-token
recurrent loop. The chunking is what unlocks the 2-3× — for a 22K-token
prefill ubatch we go from ~22000 sequential SSM updates to ~344 chunks
of parallel work.

## Phase 0: feasibility on GB10 (done 2026-04-30)

Cleanly imports and runs TileLang JIT on sm_121a after patching two
SM gates that only allow sm_90:

- `flash_qla/ops/gated_delta_rule/chunk/__init__.py`
- `flash_qla/ops/gated_delta_rule/chunk/cp_context.py`

Both gates are pure import-time checks (`get_target_compute_version() == "9.0"`).
Patch to `in ("9.0", "10.0", "12.0", "12.1")` is sufficient. TileLang
itself reports compute version "12.1" on GB10 and generates code
without errors.

Hardware constraints discovered:

- **`kkt_solve_kernel` hardcodes DK=128** (line 310, `assert K == 128`).
  Qwen3.6 GDN layers use DK=128 for the linear-attn projections (the
  full-attn layers use DK=256 but those don't go through the GDN
  kernel). Our current target (AEON-7 XS) is fine here.
- **TileLang JIT compile time on first call is long** — `kkt_solve` has
  not finished compiling within 10 minutes in our smoke test. May need
  ahead-of-time compilation via `tilelang.compile` to a serialized form,
  or accept the cold-start cost on server boot (model load already
  takes ~12 min so an extra 5-10 min is tolerable if cached).
- The `ChunkGatedDeltaRuleFunction.apply()` invocation in upstream
  `__init__.py` passes 10 args but the `forward` signature expects 9
  — bug in upstream FlashQLA. Patch: drop the trailing
  `use_qk_l2norm_in_kernel` from the apply call (the kwarg is already
  applied via `q = l2norm(q); k = l2norm(k)` earlier in the wrapper).

## Phase 1: end-to-end correctness benchmark on FlashQLA (1 day)

Goal: confirm FlashQLA produces numerically-equivalent output to our
`gated_delta_net.cu` on a single-chunk and multi-chunk input.

- Wait out the compile, time the cold + warm calls.
- Cross-check with `tests/ref_gdr.py` from the FlashQLA repo.
- Measure speedup on a representative workload (T=4096, H=4, Hg=4,
  DK=128, DV=128 — matches Qwen3.6 GDN).

Decision point: if speedup is real (≥1.5×), proceed to phase 2. If
not, abandon — the GDN compute time isn't actually the bottleneck.

## Phase 2: extract CUDA from TileLang JIT (2 days)

TileLang is a TVM-derived DSL that lowers high-level tile descriptions
to CUDA source. The actual `.cu` it generates is the artifact we need
for llama.cpp.

- Find the JIT cache directory (typically `~/.cache/tilelang/` or
  `/tmp/tilelang/`).
- Extract the generated `.cu` for `tilelang_fused_chunk_gdr_fwd_kernel`.
- Inspect the WGMMA / WMMA instruction selection — should be sm_120a/sm_121a
  variants on Blackwell (different from Hopper's WGMMA).
- Identify dependencies on TileLang runtime (cp.async helpers, etc) —
  these need lightweight equivalents in our CUDA codebase or
  inlining.

## Phase 3: integrate into ggml-cuda (2-3 days)

Add a new entry point in `gated_delta_net.cu`:

```cpp
template <int CHUNK_SIZE = 64>
__global__ void gated_delta_net_chunk_fused_cuda(...);
```

Dispatcher in `ggml_cuda_op_gated_delta_net`: when `n_tokens >=
CHUNK_THRESHOLD` (e.g. 64), use the chunked kernel; otherwise fall
back to the existing per-token kernel (still useful for decode where
we process 1-token-at-a-time during DDtree verify).

The chunked kernel won't support TREE_MODE — that's only meaningful in
single-token decode. Prefill is monotonic, no rollback.

## Phase 4: validation + perf (1-2 days)

- Numerical: compare chunked output to per-token output on a Qwen3.6
  prefill ubatch (192 tokens). Should match within fp16 noise.
- Perf: benchmark prefill on 22K context. Target: 50 s → 25-30 s.
- Long-context: 45K (sink path), 100K (full).
- Production stress: full hermes-dark agent run.

## Phase 5: cleanup

- Document the chunk-threshold tuning in README.
- If the chunk kernel proves robust, consider replacing the per-token
  kernel entirely (decode = chunk_size=1 single-token call).

## Open risks

1. **WGMMA on Blackwell**: TileLang might emit Hopper-specific WGMMA
   instructions that don't map cleanly to GB10's tcgen05 MMA. If so,
   the generated .cu won't compile or will produce wrong results.
   Phase 1 numerical check will catch this.
2. **TileLang compile time as a regression**: if we keep TileLang as a
   runtime dep, server cold start grows. Mitigation: AOT compile into
   a static `.cu` checked into the repo.
3. **Cross-architecture portability**: the kernel as ported will only
   work on sm_120/sm_121. We'd need separate paths for sm_80 (older
   GPUs) and sm_90 (Hopper). Not a concern for our specific GB10 box,
   but worth noting before upstreaming.

## Status today (2026-04-30)

Phase 0 done. TileLang installed at `/tmp/flashqla-venv`, FlashQLA
patched at `/home/jarvis/flashqla-recon/FlashQLA/`. Phase 1 (e2e
correctness benchmark) is the next concrete step. No code merged into
the fork yet.
