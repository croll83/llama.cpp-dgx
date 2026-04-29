# RFC: Port FlashQLA chunked GDN forward to llama.cpp-dgx

**Status**: Phase 0 + Phase 1 done 2026-04-30. FlashQLA does NOT fit on GB10
hardware as-shipped — kernel tiles assume Hopper sm_90's 228 KB shared
memory per block; GB10 sm_121a tops out at 99 KB. As-is port impossible.

**Scope (revised)**: from-scratch chunk-fused kernel in ggml-cuda following
FlashQLA's algorithmic insights but with GB10-appropriate tiles.
**Estimate (revised)**: ~5-10 days of CUDA kernel work, not a port.
**Savings (revised)**: TBD. Will only know after a GB10-targeted chunk kernel exists. The original 2-3× claim is Hopper-specific, not transferable.

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

## Phase 1 RESULT (done 2026-04-30): SHOWSTOPPER

FlashQLA's `prepare_h` kernel allocates ~192 KB of dynamic shared memory
per block. GB10 sm_121a hardware reports:

- `cudaDevAttrMaxSharedMemoryPerBlockOptin` = **101376 bytes (99 KB)**
- `cudaDevAttrMaxSharedMemoryPerMultiprocessor` = 102400 bytes (100 KB)

Runtime fails immediately with:
```
tvm.error.InternalError: Failed to set the allowed dynamic shared memory size to 196608
```

Compute breakdown (`prepare_h.py` line 103+, with default
num_stages=2, block_S=64, DK=128, DV=128, bf16):

| buffer | size |
|---|---|
| k_shared (2-stage) | 32 KB |
| v_shared (2-stage) | 32 KB |
| a_shared (2-stage) | 16 KB |
| g/b_shared (2-stage, fp32) | 1 KB |
| h_shared | 32 KB |
| x_shared | 16 KB |
| y_shared | 16 KB |
| m_shared_L/R | 32 KB |
| **subtotal** | **~177 KB** |
| + alignment padding etc | → **~192 KB observed** |

The kernel was sized for Hopper sm_90 which has 228 KB per-block opt-in.
Same kernel won't fit on:
- sm_120 (RTX 5090 / B200 consumer): 99 KB
- sm_121a (GB10 / Spark): 99 KB
- Pre-Hopper (sm_80, sm_86, sm_89): 100-163 KB

Only sm_90 (Hopper H100 / H200) and sm_100 (B100/B200 datacenter) have
the headroom. AEON-7's RTX5090 / RTX-PRO-6000 numbers in their README
are with the MTP head, NOT FlashQLA (which they only validate on H100/B100).

**Workarounds and their cost:**

1. **`num_stages = 1`**: halves k/v/a/g/b shared. Saves ~80 KB → fits
   to ~112 KB → still over budget. Also kills the double-buffered
   pipeline that gives the 2-3× speedup.
2. **`block_DV = 64`**: halves V tile. Saves ~32 KB → ~160 KB still
   over. And halves throughput on DV axis.
3. **Both 1 + 2**: ~80 KB → fits! But probably ~1× vs our current
   per-token kernel — speedup vanishes.
4. **`chunk_size = 32`**: halves block_S. Saves ~24 KB → ~168 KB still
   over.

No combination retains the speedup AND fits in 99 KB. The kernel
algorithm fundamentally requires Hopper's 228 KB headroom.

## Phase 1.5 (cancelled)

Cross-check with `tests/ref_gdr.py` skipped — kernel can't run.

- Cross-check with `tests/ref_gdr.py` from the FlashQLA repo.
- Measure speedup on a representative workload (T=4096, H=4, Hg=4,
  DK=128, DV=128 — matches Qwen3.6 GDN).

Decision point: if speedup is real (≥1.5×), proceed to phase 2. If
not, abandon — the GDN compute time isn't actually the bottleneck.

## Phase 2 (DEFERRED): extract CUDA from TileLang JIT

Skipped — Phase 1 showed the kernel can't run on GB10 anyway. Even if we
extracted the CUDA, it would be sized for 192 KB shared and would fail
to launch.

If we ever want to revisit:
- Find the JIT cache directory (typically `~/.cache/tilelang/` or `/tmp/tilelang/`).
- Extract the generated `.cu` for `tilelang_fused_chunk_gdr_fwd_kernel`.
- Inspect the WGMMA / WMMA instruction selection — should be sm_120a/sm_121a
  variants on Blackwell (different from Hopper's WGMMA).

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

## Status today (2026-04-30 final)

Phase 0 + Phase 1 done. **As-is port to GB10 impossible** —
192 KB shared > 99 KB GB10 limit. Original 2-3× speedup claim from the
FlashQLA blog is Hopper-specific.

Phases 2-5 of this RFC are abandoned for the as-is port. New
alternative:

## Alternative path: GB10-targeted from-scratch chunk kernel

Take the algorithmic insights from FlashQLA — chunked GDN forward
with parallelism-friendly math, gate-driven CP, warp-specialised
pipeline — and write a NEW CUDA kernel sized for 99 KB shared budget.

Tile shape budget for 99 KB:
- num_stages = 1
- block_S (chunk) = 64
- DK = 128, DV = 64 (split DV across blocks, output combined)
- → ~80 KB shared per block, fits with ~20 KB scratch headroom
- Estimated speedup: 1.3-1.7× over our current per-token kernel
  (still meaningful for prefill-dominant agent workloads)

Estimate: 5-10 days of CUDA kernel work. Numerical validation against
existing per-token kernel as ground truth.

This is fundamentally a kernel engineering project, not a port.

## Reconnaissance artifacts kept

- `/home/jarvis/flashqla-recon/FlashQLA/` — patched FlashQLA repo
  (SM gates relaxed, apply()/forward() arg-count fix)
- `/tmp/flashqla-venv/` — virtualenv with tilelang 0.1.8 + torch 2.11
- `/tmp/flashqla_phase1.py` — correctness test script (cancelled at
  shared-mem failure)
- `/tmp/flashqla_phase1.log` — full run log incl. shared-mem error
