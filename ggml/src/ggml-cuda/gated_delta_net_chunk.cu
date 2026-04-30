// gated_delta_net_chunk.cu — chunked GDN forward for prefill paths.
//
// Companion to gated_delta_net.cu (per-token decode kernel). Active when
// n_tokens >= GDN_CHUNK_THRESHOLD (= 64) on prefill ubatches; the per-token
// kernel still handles decode (n_tokens=1) and DDtree verify (small chunks).
//
// Designed for GB10 sm_121a with the 99 KB shared memory per-block budget.
// FlashQLA's Hopper-targeted kernels need 192 KB so we cannot use them as-is;
// this is a from-scratch GB10-tiled re-implementation following FlashQLA's
// algorithmic split (chunk_local_cumsum -> kkt_solve -> prepare_h ->
// fused_fwd) but with smaller tiles (chunk_size=64, DV split into 2 sub-blocks
// of DV_BLK=64).
//
// Reference RFC: docs/rfc-gdn-chunk-kernel.md.
//
// Status: Phases 1-4 implemented. Phase 5 (E2E validation) and Phase 6
// (perf tuning) follow. Dispatcher gating in ggml-cuda.cu.

#include "gated_delta_net.cuh"
#include "common.cuh"
#include "convert.cuh"
#include "gated_delta_net_chunk_kernels.cuh"

#include <cstdlib>
#include <cuda_bf16.h>

// ─── Helper kernels for fp32 ↔ bf16 + state transpose ──────────────────────
//
// The per-token GDN op stores h-state as fp32 in [seq, head, dv, dk] layout
// (transposed: state[col*S_v + i] = h[i, col]). The chunked kernels expect
// h_initial / h_final as bf16 in [b, h, dk, dv] layout (no transpose). These
// helpers do the format conversion in a single fused kernel each.

// Layout converter: state_f32 [B, H, DV_outer, DK_inner] (row-major, transposed
// w.r.t. h) → h_bf16 [B, H, DK, DV] (row-major, untransposed).
__global__ void gdn_chunk_state_f32_to_bf16_transpose(
    const float * __restrict__ src,         // [B, H, DV, DK]
    __nv_bfloat16 * __restrict__ dst,       // [B, H, DK, DV]
    int B, int H, int DK, int DV) {

    const int dv = blockIdx.x * blockDim.x + threadIdx.x;
    const int dk = blockIdx.y * blockDim.y + threadIdx.y;
    if (dv >= DV || dk >= DK) return;

    const int bh = blockIdx.z;
    const int b  = bh / H;
    const int h  = bh % H;

    const int src_off = ((b * H + h) * DV + dv) * DK + dk;     // [B, H, DV, DK]
    const int dst_off = ((b * H + h) * DK + dk) * DV + dv;     // [B, H, DK, DV]
    dst[dst_off] = __float2bfloat16(src[src_off]);
}

// Reverse: h_bf16 [B, H, DK, DV] → state_f32 [B, H, DV, DK].
__global__ void gdn_chunk_state_bf16_to_f32_transpose(
    const __nv_bfloat16 * __restrict__ src, // [B, H, DK, DV]
    float * __restrict__ dst,               // [B, H, DV, DK]
    int B, int H, int DK, int DV) {

    const int dk = blockIdx.x * blockDim.x + threadIdx.x;
    const int dv = blockIdx.y * blockDim.y + threadIdx.y;
    if (dk >= DK || dv >= DV) return;

    const int bh = blockIdx.z;
    const int b  = bh / H;
    const int h  = bh % H;

    const int src_off = ((b * H + h) * DK + dk) * DV + dv;     // [B, H, DK, DV]
    const int dst_off = ((b * H + h) * DV + dv) * DK + dk;     // [B, H, DV, DK]
    dst[dst_off] = __bfloat162float(src[src_off]);
}

// Trivial fp32 → bf16 contiguous converter for Q/K/V.
__global__ void gdn_chunk_f32_to_bf16(
    const float * __restrict__ src,
    __nv_bfloat16 * __restrict__ dst,
    int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(src[i]);
}

// bf16 → fp32 contiguous (for O → attn_data) plus optional scale.
__global__ void gdn_chunk_bf16_to_f32_scaled(
    const __nv_bfloat16 * __restrict__ src,
    float * __restrict__ dst,
    int64_t n,
    float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]) * scale;
}

// ─── Public entry point ─────────────────────────────────────────────────────
//
// Chunked counterpart of ggml_cuda_op_gated_delta_net. Caller must ensure:
//   - parent_ids == nullptr (no tree mode)
//   - kda == false           (per-head scalar gate, not per-element)
//   - S_v == 128             (only DK=DV=128 instantiated)
//   - n_tokens >= GDN_CHUNK_THRESHOLD (= 64)
//
// Tensor layout (matching ggml_cuda_op_gated_delta_net):
//   src[0] q, src[1] k, src[2] v: fp32 [S_v, H, n_tokens, n_seqs]
//   src[3] g                    : fp32 [1, H, n_tokens, n_seqs]
//   src[4] beta                 : fp32 [1, H, n_tokens, n_seqs]
//   src[5] state                : fp32 [S_v, S_v, H, n_seqs] (transposed)
//   dst                         : fp32, three concat'd regions:
//                                   - attn = S_v * H * n_tokens * n_seqs
//                                   - state_out = S_v * S_v * H * n_seqs
//                                   - inter = n_tokens * S_v * S_v * H * n_seqs
void ggml_cuda_op_gated_delta_net_chunk(ggml_backend_cuda_context & ctx,
                                         ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(S_v == 128 && "chunk path only handles S_v=128");
    GGML_ASSERT(n_tokens >= 64 && "chunk path requires n_tokens >= 64");
    GGML_ASSERT(src_g->ne[0] == 1 && "chunk path does not handle KDA (per-element gate)");

    // Contiguity guard: the chunk kernels assume Q/K/V/g/beta are fully
    // contiguous in the [B, T, H, *] / [B, T, H] layouts. Fall back to
    // per-token if any input is strided.
    auto fully_contig_qkv = [](ggml_tensor * t, int64_t S_v, int64_t H,
                               int64_t n_tokens, int64_t n_seqs) -> bool {
        if (!ggml_is_contiguous(t)) return false;
        if (t->ne[0] != S_v || t->ne[1] != H ||
            t->ne[2] != n_tokens || t->ne[3] != n_seqs) return false;
        return true;
    };
    if (!fully_contig_qkv(src_q, S_v, H, n_tokens, n_seqs) ||
        !fully_contig_qkv(src_k, S_v, H, n_tokens, n_seqs) ||
        !fully_contig_qkv(src_v, S_v, H, n_tokens, n_seqs)) {
        ggml_cuda_op_gated_delta_net(ctx, dst);
        return;
    }
    if (!ggml_is_contiguous(src_g)    ||
        !ggml_is_contiguous(src_beta) ||
        !ggml_is_contiguous(src_state)) {
        ggml_cuda_op_gated_delta_net(ctx, dst);
        return;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    constexpr int CHUNK_SIZE = 64;
    constexpr int DK         = 128;
    constexpr int DV_BLK     = 64;
    const int     DV         = (int) S_v;          // = 128
    const int     B          = (int) n_seqs;
    const int     T          = (int) n_tokens;
    const int     Hh         = (int) H;
    const int     num_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;

    const int64_t n_BTH    = (int64_t) B * T * Hh;
    const int64_t n_BTH_DK = n_BTH * DK;
    const int64_t n_BTH_DV = n_BTH * DV;
    const int64_t n_BTH_S  = n_BTH * CHUNK_SIZE;
    const int64_t n_BHDD   = (int64_t) B * Hh * DK * DV;
    const int64_t n_HPC    = (int64_t) B * num_chunks * Hh * DK * DV;

    // ── Allocate scratch buffers in the cuda pool.
    ggml_cuda_pool_alloc<__nv_bfloat16> Q_bf(pool, n_BTH_DK);
    ggml_cuda_pool_alloc<__nv_bfloat16> K_bf(pool, n_BTH_DK);
    ggml_cuda_pool_alloc<__nv_bfloat16> V_bf(pool, n_BTH_DV);
    ggml_cuda_pool_alloc<float>         g_cum(pool, n_BTH);
    ggml_cuda_pool_alloc<__nv_bfloat16> A_sol(pool, n_BTH_S);
    ggml_cuda_pool_alloc<__nv_bfloat16> h_init(pool, n_BHDD);
    ggml_cuda_pool_alloc<__nv_bfloat16> h_per_chunk(pool, n_HPC);
    ggml_cuda_pool_alloc<__nv_bfloat16> h_final(pool, n_BHDD);
    ggml_cuda_pool_alloc<__nv_bfloat16> O_bf(pool, n_BTH_DV);

    // ── 1. Convert Q, K, V from fp32 → bf16 (contiguous; layout already matches).
    {
        const int  block = 256;
        const auto run   = [&](const float * src, __nv_bfloat16 * dst, int64_t n) {
            const int grid = (int) ((n + block - 1) / block);
            gdn_chunk_f32_to_bf16<<<grid, block, 0, stream>>>(src, dst, n);
        };
        run((const float *) src_q->data, Q_bf.get(), n_BTH_DK);
        run((const float *) src_k->data, K_bf.get(), n_BTH_DK);
        run((const float *) src_v->data, V_bf.get(), n_BTH_DV);
    }

    // ── 2. Transpose-and-cast h_initial: [B, H, DV, DK] fp32 → [B, H, DK, DV] bf16.
    {
        dim3 block(16, 16);
        dim3 grid((DV + 15) / 16, (DK + 15) / 16, B * Hh);
        gdn_chunk_state_f32_to_bf16_transpose<<<grid, block, 0, stream>>>(
            (const float *) src_state->data, h_init.get(), B, Hh, DK, DV);
    }

    // ── 3. chunk_local_cumsum: g_fp32 → g_cum fp32.
    {
        dim3 grid(num_chunks, Hh, B);
        dim3 block(CHUNK_SIZE);
        gdn_chunk_local_cumsum_kernel<<<grid, block, 0, stream>>>(
            (const float *) src_g->data, g_cum.get(), B, T, Hh, CHUNK_SIZE);
    }

    // ── 4. KKT solve: K_bf16 + beta + g_cum → A_sol bf16.
    launch_gdn_chunk_kkt_solve<CHUNK_SIZE, DK>(
        K_bf.get(), (const float *) src_beta->data, g_cum.get(),
        A_sol.get(), B, T, Hh, stream);

    // ── 5. prepare_h: K, V, g_cum, A_sol, h_initial → h_per_chunk, h_final.
    launch_gdn_chunk_prepare_h<CHUNK_SIZE, DK, DV_BLK>(
        K_bf.get(), V_bf.get(), g_cum.get(), A_sol.get(),
        h_init.get(), h_per_chunk.get(), h_final.get(),
        B, T, Hh, DV, stream);

    // ── 6. fused_fwd: Q, K, V, A_sol, g_cum, h_per_chunk → O bf16.
    launch_gdn_chunk_fused_fwd<CHUNK_SIZE, DK, DV_BLK>(
        Q_bf.get(), K_bf.get(), V_bf.get(), A_sol.get(), g_cum.get(),
        h_per_chunk.get(), O_bf.get(), B, T, Hh, DV, stream);

    // ── 7. Write outputs back into dst:
    //   dst region 0 = attn (B*T*H*DV fp32, layout [seq, token, head, dv])
    //   dst region 1 = state_out (B*H*S_v*S_v fp32, layout [seq, head, dv, dk])
    //   dst region 2 = intermediates (skip; not used in non-tree mode)
    float *       dst_d        = (float *) dst->data;
    const int64_t attn_elems   = n_BTH_DV;
    const int64_t state_elems  = (int64_t) B * Hh * S_v * S_v;
    float *       attn_dst     = dst_d;
    float *       state_dst    = dst_d + attn_elems;

    const float scale = 1.0f / sqrtf((float) S_v);
    {
        const int  block = 256;
        const int  grid  = (int) ((attn_elems + block - 1) / block);
        gdn_chunk_bf16_to_f32_scaled<<<grid, block, 0, stream>>>(
            O_bf.get(), attn_dst, attn_elems, scale);
    }

    {
        dim3 block(16, 16);
        dim3 grid((DK + 15) / 16, (DV + 15) / 16, B * Hh);
        gdn_chunk_state_bf16_to_f32_transpose<<<grid, block, 0, stream>>>(
            h_final.get(), state_dst, B, Hh, DK, DV);
    }

    (void) state_elems;  // intermediates region untouched (chunk path is not tree mode)
}

// ─── Dispatch wrapper ─────────────────────────────────────────────────────
// Decides whether to route to the chunked path or fall back to the per-token
// kernel. Called from ggml-cuda.cu's GGML_OP_GATED_DELTA_NET case.
//
// Env-var override: GGML_GDN_CHUNK_DISABLE=1 forces the per-token path (for
// A/B perf comparison and as an escape-hatch if a regression is discovered
// in production).
void ggml_cuda_op_gated_delta_net_dispatch(ggml_backend_cuda_context & ctx,
                                            ggml_tensor * dst) {
    static const bool chunk_disabled = []{
        const char * env = std::getenv("GGML_GDN_CHUNK_DISABLE");
        return env && env[0] == '1';
    }();

    ggml_tensor * src_v      = dst->src[2];
    ggml_tensor * src_g      = dst->src[3];
    ggml_tensor * src_parent = (dst->src[6]);  // optional tree-mode parent ids

    const int64_t S_v      = src_v->ne[0];
    const int64_t n_tokens = src_v->ne[2];
    const bool    kda      = (src_g->ne[0] == S_v);
    const bool    tree     = (src_parent != nullptr);

    // Gate: prefill batch + non-tree + non-KDA + S_v=128.
    if (!chunk_disabled && !tree && !kda &&
        S_v == 128 && n_tokens >= GDN_CHUNK_THRESHOLD) {
        ggml_cuda_op_gated_delta_net_chunk(ctx, dst);
    } else {
        ggml_cuda_op_gated_delta_net(ctx, dst);
    }
}
