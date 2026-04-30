// gated_delta_net_chunk_kernels.cuh — header-only kernel definitions for
// the chunked GDN forward path. Designed for GB10 sm_121a (99 KB shared
// per-block budget). See docs/rfc-gdn-chunk-kernel.md for design rationale.
//
// This header has NO ggml dependency, so it can be included from both
// the production .cu file (gated_delta_net_chunk.cu, plus dispatcher) and
// from standalone test executables.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

// Default tile parameters (overridable from command line).
#ifndef GDN_CHUNK_SIZE
#define GDN_CHUNK_SIZE 64
#endif
#ifndef GDN_DV_BLK
#define GDN_DV_BLK     64
#endif

// Threshold below which the chunked path falls back to the per-token kernel.
// Equal to one CHUNK_SIZE — there's no benefit to chunking a single chunk.
#ifndef GDN_CHUNK_THRESHOLD
#define GDN_CHUNK_THRESHOLD 64
#endif

// ─── Phase 1.2: chunk-local cumsum of g ────────────────────────────────────
//
// Computes G_cumsum[c, i] = sum_{j<=i within chunk c} g[c*S + j] per
// (B, H) work unit. The chunk-local boundary means each chunk's cumsum
// resets to 0 (no inter-chunk carry — that lives in the SSM state instead).
//
// Layout: g and g_cumsum are [B, T, H] with row-major stride
// offset(b, t, h) = (b*T + t)*H + h.
//
// Grid: (num_chunks, H, B). Block: 64 threads (one per chunk position).
// chunk_size is currently fixed at 64 (= 2 warps).
__global__ inline void gdn_chunk_local_cumsum_kernel(
    const float * __restrict__ g,
    float       * __restrict__ g_cumsum,
    int B, int T, int H, int chunk_size) {

    const int chunk_idx = blockIdx.x;
    const int h         = blockIdx.y;
    const int b         = blockIdx.z;
    const int t_local   = threadIdx.x;
    const int t_global  = chunk_idx * chunk_size + t_local;

    const bool in_range = (t_local < chunk_size) && (t_global < T);

    const int offset = (b * T + t_global) * H + h;
    float val = in_range ? g[offset] : 0.0f;

    // Warp-inclusive prefix sum.
    const int lane    = t_local & 31;
    const int warp_id = t_local >> 5;
    #pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        const float v = __shfl_up_sync(0xFFFFFFFFu, val, delta);
        if (lane >= delta) val += v;
    }

    // Inter-warp combine (chunk_size=64 → 2 warps).
    __shared__ float warp_total[2];
    if (lane == 31) warp_total[warp_id] = val;
    __syncthreads();
    if (warp_id == 1) val += warp_total[0];

    if (in_range) {
        g_cumsum[offset] = val;
    }
}

// ─── Phase 1.3: KKT solve for A matrix ─────────────────────────────────────
//
// Solves the per-chunk lower-triangular system
//
//     (I - L) · A_sol = diag(beta)
//
// where L is the strict-lower-triangular matrix
//
//     L[i, j] = beta[i] · <K[i], K[j]> · exp(G_cum[j-1] - G_cum[i])  for j < i
//             = 0                                                    for j >= i
//
// (with G_cum[-1] := 0 by convention for j=0). The solution is the
// chunk-local "alpha" matrix used by prepare_h to weight V before applying
// to the SSM state. By induction the result is also strict-lower-triangular
// with diagonal equal to beta.
template <int CHUNK_SIZE, int DK>
__global__ void gdn_chunk_kkt_solve_kernel(
    const __nv_bfloat16 * __restrict__ K,
    const float         * __restrict__ beta,
    const float         * __restrict__ g_cumsum,
    __nv_bfloat16       * __restrict__ A_sol,
    int B, int T, int H) {

    constexpr int S = CHUNK_SIZE;

    const int chunk_idx = blockIdx.x;
    const int h         = blockIdx.y;
    const int b         = blockIdx.z;
    const int j         = threadIdx.x;        // column index (one per thread)

    const int t_for_thread = chunk_idx * S + j;
    const bool in_range    = (j < S) && (t_for_thread < T);

    extern __shared__ __align__(16) char gdn_smem_raw[];
    __nv_bfloat16 * K_tile  = reinterpret_cast<__nv_bfloat16 *>(gdn_smem_raw);          // [S, DK]
    float         * KK_dot  = reinterpret_cast<float *>(K_tile + S * DK);                 // [S, S]
    float         * X       = KK_dot + S * S;                                             // [S, S]
    float         * beta_t  = X + S * S;                                                  // [S]
    float         * g_t     = beta_t + S;                                                 // [S]

    // Cooperative load: each thread loads its row of K, plus its scalars.
    const int K_row_off = ((b * T + t_for_thread) * H + h) * DK;
    if (in_range) {
        #pragma unroll 8
        for (int d = 0; d < DK; ++d) {
            K_tile[j * DK + d] = K[K_row_off + d];
        }
        const int scalar_off = (b * T + t_for_thread) * H + h;
        beta_t[j] = beta[scalar_off];
        g_t[j]    = g_cumsum[scalar_off];
    } else {
        #pragma unroll 8
        for (int d = 0; d < DK; ++d) {
            K_tile[j * DK + d] = __float2bfloat16(0.0f);
        }
        beta_t[j] = 0.0f;
        g_t[j]    = 0.0f;
    }
    __syncthreads();

    // KK_dot[j, k] = <K[j], K[k]> for all k ∈ [0, S).
    #pragma unroll 4
    for (int k = 0; k < S; ++k) {
        float dot = 0.0f;
        #pragma unroll 16
        for (int d = 0; d < DK; ++d) {
            dot += __bfloat162float(K_tile[j * DK + d]) *
                   __bfloat162float(K_tile[k * DK + d]);
        }
        KK_dot[j * S + k] = dot;
    }
    __syncthreads();

    // Initialise X = diag(beta).
    #pragma unroll 4
    for (int row = 0; row < S; ++row) {
        X[row * S + j] = (row == j) ? beta_t[row] : 0.0f;
    }
    __syncthreads();

    // Forward substitution: row by row, each thread is one column j.
    for (int i = 1; i < S; ++i) {
        if (j < i) {
            const float beta_i = beta_t[i];
            const float g_i    = g_t[i];
            float sum = 0.0f;
            for (int k = j; k < i; ++k) {
                const float g_prev = (k > 0) ? g_t[k - 1] : 0.0f;
                const float L_ik   = beta_i * KK_dot[i * S + k] * __expf(g_prev - g_i);
                sum += L_ik * X[k * S + j];
            }
            X[i * S + j] = sum;
        }
        __syncthreads();
    }

    // Write output.
    if (in_range) {
        const int A_row_off = ((b * T + t_for_thread) * H + h) * S;
        #pragma unroll 8
        for (int col = 0; col < S; ++col) {
            A_sol[A_row_off + col] = __float2bfloat16(X[j * S + col]);
        }
    }
}

// Launcher for kkt_solve. Sets the dynamic-shared-memory attribute before
// launch (the kernel exceeds the default 48 KB shared at DK > 64).
template <int CHUNK_SIZE, int DK>
inline void launch_gdn_chunk_kkt_solve(
    const __nv_bfloat16 * K,
    const float         * beta,
    const float         * g_cumsum,
    __nv_bfloat16       * A_sol,
    int B, int T, int H, cudaStream_t stream) {

    constexpr size_t S       = CHUNK_SIZE;
    constexpr size_t SMEM_K  = S * DK * sizeof(__nv_bfloat16);
    constexpr size_t SMEM_KK = S * S  * sizeof(float);
    constexpr size_t SMEM_X  = S * S  * sizeof(float);
    constexpr size_t SMEM_SC = S      * sizeof(float) * 2;
    constexpr size_t SMEM_TOTAL = SMEM_K + SMEM_KK + SMEM_X + SMEM_SC;
    static_assert(SMEM_TOTAL <= 64 * 1024,
                  "kkt_solve shared-mem budget exceeded — increase opt-in cap or reduce DK/S");

    auto kernel_ptr = gdn_chunk_kkt_solve_kernel<CHUNK_SIZE, DK>;
    cudaFuncSetAttribute(kernel_ptr,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int) SMEM_TOTAL);

    const int num_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;
    dim3 grid(num_chunks, H, B);
    dim3 block(CHUNK_SIZE);
    kernel_ptr<<<grid, block, SMEM_TOTAL, stream>>>(K, beta, g_cumsum, A_sol, B, T, H);
}

// ─── Phase 2: prepare_h kernel ─────────────────────────────────────────────
//
// Advances the SSM state h across the chunks of one (B, H, DV-block) work
// unit. h is a [DK, DV_BLK] matrix kept in registers (one strip per warp,
// one column-set per thread). For each chunk we:
//
//   1. Save h to h_per_chunk[c] (the state at the START of this chunk).
//   2. For each token t in the chunk (sequential):
//        u_t = β_t · (v_t - K_t · h_register)             [length DV_BLK]
//        h   *= α_t   (= exp(g_cum[t] - g_cum[t-1]))
//        h[k, :] += K_t[k] · u_t[:]    (rank-1 update)
//   3. Continue to the next chunk (h carries over).
//
// At the end, write h to h_final.
//
// This is the "per-token within chunk" formulation — sequential in t, but
// every token reuses the same shared K/V tile so we save 64x the global
// memory bandwidth vs the legacy per-token kernel that reloads per token.
// Phase 6 will explore an A_sol / tensor-core variant if speedup is below
// target.
//
// Block layout: 128 threads = 4 warps. h has DK=128 rows × DV_BLK=64 cols.
//   - Each thread holds (DK / num_threads_per_col) rows × DV_BLK cols of h
//     in registers. With 128 threads in a 1D block and 64 columns, a clean
//     mapping is: thread t handles dv = t & 63, dk = (t >> 6) gives 0/1
//     telling which half of DK rows. So each thread holds 64 dk-rows × 1
//     dv-col → 64 fp32 values → 256 bytes per thread → 32 KB total.
//   That is effectively a 64-rows × 64-cols sub-tile per (warp pair).
//   The full h_register is split: thread (dk_half, dv_col) owns h[dk_half*64 + 0..63, dv_col].
//
// Shared layout per CTA (DK=128, DV_BLK=64, S=64):
//   K_tile  : [S, DK]      bf16 = 16 KB
//   V_tile  : [S, DV_BLK]  bf16 =  8 KB
//   gc_tile : [S]          f32  = 0.25 KB
//   b_tile  : [S]          f32  = 0.25 KB
//   total ~25 KB
template <int CHUNK_SIZE, int DK, int DV_BLK>
__global__ void gdn_chunk_prepare_h_kernel(
    const __nv_bfloat16 * __restrict__ K,         // [B, T, H, DK]
    const __nv_bfloat16 * __restrict__ V,         // [B, T, H, DV]
    const float         * __restrict__ g_cumsum,  // [B, T, H]
    const float         * __restrict__ beta,      // [B, T, H]
    const __nv_bfloat16 * __restrict__ h_initial, // [B, H, DK, DV] or null = zeros
    __nv_bfloat16       * __restrict__ h_per_chunk, // [B, num_chunks, H, DK, DV]
    __nv_bfloat16       * __restrict__ h_final,     // [B, H, DK, DV]
    int B, int T, int H, int DV, int dv_blk_off) {

    constexpr int S = CHUNK_SIZE;
    static_assert(DK == 128, "Phase 2 prepare_h is hardcoded for DK=128 (will generalise later)");
    static_assert(DV_BLK == 64, "Phase 2 prepare_h is hardcoded for DV_BLK=64");

    const int dv_blk_idx = blockIdx.x;       // [0, num_dv_blocks)
    const int h_idx      = blockIdx.y;       // [0, H)
    const int b_idx      = blockIdx.z;       // [0, B)

    const int dv_off    = dv_blk_off + dv_blk_idx * DV_BLK;
    const int num_chunks = (T + S - 1) / S;

    // Thread layout: tid in [0, 128). Map to (dk_half, dv_col):
    //   dv_col  = tid & 63   (which DV column this thread owns)
    //   dk_half = tid >> 6   (0 → DK rows [0,64), 1 → [64,128))
    const int tid     = threadIdx.x;
    const int dv_col  = tid & 63;
    const int dk_half = tid >> 6;

    // h_register holds 64 fp32 values: h[dk_half*64 + 0..63, dv_col].
    float h_reg[64];

    // ── Initialise h: from h_initial or zeros.
    if (h_initial != nullptr) {
        const int h0_base = ((b_idx * H + h_idx) * DK) * DV;
        #pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            const int dk = dk_half * 64 + k;
            h_reg[k] = __bfloat162float(h_initial[h0_base + dk * DV + (dv_off + dv_col)]);
        }
    } else {
        #pragma unroll 8
        for (int k = 0; k < 64; ++k) h_reg[k] = 0.0f;
    }

    // ── Shared tiles for the current chunk.
    __shared__ __nv_bfloat16 K_tile[S * DK];
    __shared__ __nv_bfloat16 V_tile[S * DV_BLK];
    __shared__ float          gc_tile[S];
    __shared__ float          b_tile[S];
    __shared__ float          dot_scratch[2 * 64];   // [dk_half, dv_col] partial-dot exchange

    // ── Outer chunk loop (sequential).
    for (int c = 0; c < num_chunks; ++c) {
        // -- Save h to h_per_chunk[c] (state at the START of chunk c).
        const int hpc_off = (((b_idx * num_chunks + c) * H + h_idx) * DK) * DV;
        #pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            const int dk = dk_half * 64 + k;
            h_per_chunk[hpc_off + dk * DV + (dv_off + dv_col)] = __float2bfloat16(h_reg[k]);
        }

        // -- Cooperative load of K, V, g_cum, beta tiles for this chunk.
        const int t_chunk_start = c * S;
        // Each thread loads: K[t_local, :] for some t_local, etc. Threads
        // share work via t_local = tid & (S-1); for the per-row 128-element
        // K we have 128 threads loading 1 K element each per "t_local" slot
        // → S * DK / 128 = 64 elements per thread total over S iterations.
        // Simpler: t_local = tid % S, each thread handles its row across
        // S/(num_threads/S) = 0.5 rows... messy. Use a flat loop instead.
        for (int idx = tid; idx < S * DK; idx += 128) {
            const int t_local = idx / DK;
            const int dk      = idx % DK;
            const int t_g     = t_chunk_start + t_local;
            __nv_bfloat16 v = __float2bfloat16(0.0f);
            if (t_g < T) {
                v = K[((b_idx * T + t_g) * H + h_idx) * DK + dk];
            }
            K_tile[t_local * DK + dk] = v;
        }
        for (int idx = tid; idx < S * DV_BLK; idx += 128) {
            const int t_local = idx / DV_BLK;
            const int dv_lc   = idx % DV_BLK;
            const int t_g     = t_chunk_start + t_local;
            __nv_bfloat16 v = __float2bfloat16(0.0f);
            if (t_g < T) {
                v = V[((b_idx * T + t_g) * H + h_idx) * DV + (dv_off + dv_lc)];
            }
            V_tile[t_local * DV_BLK + dv_lc] = v;
        }
        for (int t_local = tid; t_local < S; t_local += 128) {
            const int t_g = t_chunk_start + t_local;
            const bool ok = t_g < T;
            const int sc  = (b_idx * T + t_g) * H + h_idx;
            gc_tile[t_local] = ok ? g_cumsum[sc] : 0.0f;
            b_tile [t_local] = ok ? beta[sc]     : 0.0f;
        }
        __syncthreads();

        // -- Inner per-token loop (sequential).
        const int t_in_chunk_max = min(S, T - t_chunk_start);
        for (int t_local = 0; t_local < t_in_chunk_max; ++t_local) {
            const float beta_t = b_tile[t_local];
            const float g_t    = gc_tile[t_local];
            const float g_prev = (t_local > 0) ? gc_tile[t_local - 1]
                                                : 0.0f;  // chunk-local cumsum starts at 0
            const float alpha_t = __expf(g_t - g_prev);

            // u_t[dv_col] = beta_t * (V_tile[t_local, dv_col] - sum_k K_tile[t_local, k] * h_reg[..., dv_col])
            //
            // Each thread computes u_t for ITS OWN dv_col but the K_tile dot
            // product spans ALL DK=128 rows. With our register layout, two
            // threads (dk_half=0 and dk_half=1) own complementary halves of
            // h_reg. Use warp shuffle to combine their partial dot products.
            float dot = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < 64; ++k) {
                const int dk = dk_half * 64 + k;
                dot += __bfloat162float(K_tile[t_local * DK + dk]) * h_reg[k];
            }
            // Reduce across the dk_half partition: thread (0, dv_col) and
            // thread (1, dv_col) own the same dv_col but different dk halves.
            // They are in DIFFERENT warps (since tid differs by 64). Combine
            // their partial dots via the dot_scratch shared buffer.
            dot_scratch[dk_half * 64 + dv_col] = dot;
            __syncthreads();
            const float dot_full = dot_scratch[0 * 64 + dv_col] +
                                   dot_scratch[1 * 64 + dv_col];

            const float v_t   = __bfloat162float(V_tile[t_local * DV_BLK + dv_col]);
            const float u_t   = beta_t * (v_t - dot_full);

            // h *= alpha_t (every register element)
            #pragma unroll 8
            for (int k = 0; k < 64; ++k) {
                h_reg[k] *= alpha_t;
            }
            // h[dk, dv_col] += K_tile[t_local, dk] * u_t  (for our dk_half rows)
            #pragma unroll 8
            for (int k = 0; k < 64; ++k) {
                const int dk = dk_half * 64 + k;
                const float k_val = __bfloat162float(K_tile[t_local * DK + dk]);
                h_reg[k] += k_val * u_t;
            }
            __syncthreads();  // ensure dot_scratch is reusable next iteration
        }
        __syncthreads();
    }

    // ── Write h_final (state after the last chunk).
    if (h_final != nullptr) {
        const int hf_off = ((b_idx * H + h_idx) * DK) * DV;
        #pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            const int dk = dk_half * 64 + k;
            h_final[hf_off + dk * DV + (dv_off + dv_col)] = __float2bfloat16(h_reg[k]);
        }
    }
}

// Launcher for prepare_h.
template <int CHUNK_SIZE, int DK, int DV_BLK>
inline void launch_gdn_chunk_prepare_h(
    const __nv_bfloat16 * K,
    const __nv_bfloat16 * V,
    const float         * g_cumsum,
    const float         * beta,
    const __nv_bfloat16 * h_initial,
    __nv_bfloat16       * h_per_chunk,
    __nv_bfloat16       * h_final,
    int B, int T, int H, int DV, cudaStream_t stream) {

    const int num_dv_blocks = DV / DV_BLK;
    dim3 grid(num_dv_blocks, H, B);
    dim3 block(128);

    for (int dv_blk_off_idx = 0; dv_blk_off_idx < num_dv_blocks; ++dv_blk_off_idx) {
        // We loop here only conceptually — the kernel uses blockIdx.x to pick
        // the dv block, so we launch a single grid spanning all of them.
        (void) dv_blk_off_idx;
    }
    gdn_chunk_prepare_h_kernel<CHUNK_SIZE, DK, DV_BLK>
        <<<grid, block, 0, stream>>>(K, V, g_cumsum, beta, h_initial,
                                      h_per_chunk, h_final,
                                      B, T, H, DV, /*dv_blk_off=*/0);
}

// ─── Phase 3: fused_fwd kernel ─────────────────────────────────────────────
//
// Computes the GDN output for one (B, H, chunk, DV-block) tile:
//
//   y_t[d] = exp(g_cum[t]) · (q_t · h_{c-1})[d]                    ← inter-chunk
//          + Σ_{k≤t in chunk} exp(g_cum[t] - g_cum[k]) ·
//                              (q_t · k_k) · U_k[d]                ← intra-chunk
//
// where  U = A_sol_chunk @ V_eff,
//        V_eff[t, d] = V[t, d] - (K[t] · h_{c-1})[d].
//
// Intuition: A_sol substitutes the recursive (v_j - k_j h) terms within the
// chunk; V_eff folds the inter-chunk h dependency into a single modified V.
//
// Block layout: 4 warps = 128 threads. Each warp owns rows [w*16, (w+1)*16)
// of the 64×64 output. wmma 16×16×16 bf16 with fp32 accumulators throughout.
//
// Grid:
//   x = chunk_idx * num_dv_blocks + dv_blk_idx
//   y = h
//   z = b
// (We keep dv_blk fast-varying so consecutive blocks share the same chunk
// — improves L2 reuse on Q/K/V for the chunk.)
//
// Shared budget for S=64, DK=128, DV_BLK=64 (= 80.25 KB, under 99 KB):
//   Q_tile      [S, DK]       bf16  = 16 KB
//   K_tile      [S, DK]       bf16  = 16 KB
//   A_tile      [S, S]        bf16  =  8 KB   (later reused as attn_bf16 then O_intra_bf16 storage)
//   V_tile      [S, DV_BLK]   bf16  =  8 KB   (V → V_eff_bf16 → U_bf16)
//   h_tile      [DK, DV_BLK]  bf16  = 16 KB
//   gc_tile     [S]           fp32  = 0.25 KB
//   scratch_fp32 [S, max(S, DV_BLK)] fp32 = 16 KB (= [64,64], time-mux'd)
//
// Pipeline (each step gated by __syncthreads):
//   load Q, K, V→bf16, A_sol, gc, h
//   STEP-V: scratch_fp32 ← K @ h  (wmma 16x16x16 bf16 → fp32 acc)
//           V_tile ← bf16(V - scratch_fp32)
//   STEP-U: scratch_fp32 ← A_sol @ V_eff  (wmma)
//           V_tile ← bf16(scratch_fp32)            // V_tile now holds U_bf16
//   STEP-QK: scratch_fp32 ← Q @ K^T  (wmma, fragment_b col_major)
//           A_tile ← bf16(scratch_fp32 · decay · tri_lower_with_diag)   // attn matrix
//   STEP-O_intra: scratch_fp32 ← A_tile @ V_tile  (wmma)         // O_intra fp32
//                 V_tile ← bf16(scratch_fp32)                    // O_intra_bf16 stash
//   STEP-Qh: scratch_fp32 ← Q @ h_tile  (wmma)
//   FINAL: O[t,dv] = bf2f(V_tile[t,dv]) + exp(gc[t]) · scratch_fp32[t,dv]
template <int CHUNK_SIZE, int DK, int DV_BLK>
__global__ void gdn_chunk_fused_fwd_kernel(
    const __nv_bfloat16 * __restrict__ Q,           // [B, T, H, DK]
    const __nv_bfloat16 * __restrict__ K,           // [B, T, H, DK]
    const __nv_bfloat16 * __restrict__ V,           // [B, T, H, DV]
    const __nv_bfloat16 * __restrict__ A_sol,       // [B, T, H, S]
    const float         * __restrict__ g_cumsum,    // [B, T, H]
    const __nv_bfloat16 * __restrict__ h_per_chunk, // [B, num_chunks, H, DK, DV]
    __nv_bfloat16       * __restrict__ O,           // [B, T, H, DV]
    int B, int T, int H, int DV, int num_dv_blocks) {

    constexpr int S = CHUNK_SIZE;
    static_assert(DK == 128 && DV_BLK == 64 && CHUNK_SIZE == 64,
                  "fused_fwd_wmma is hardcoded for S=64, DK=128, DV_BLK=64");

    namespace wmma = nvcuda::wmma;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

    const int chunk_idx  = blockIdx.x / num_dv_blocks;
    const int dv_blk_idx = blockIdx.x % num_dv_blocks;
    const int h_idx      = blockIdx.y;
    const int b_idx      = blockIdx.z;
    const int dv_off     = dv_blk_idx * DV_BLK;
    const int num_chunks = (T + S - 1) / S;
    const int t_chunk_start = chunk_idx * S;

    const int tid     = threadIdx.x;     // 0..127
    const int warp_id = tid >> 5;        // 0..3
    const int lane    = tid & 31;
    (void) lane;

    extern __shared__ __align__(16) char gdn_fwd_smem[];
    __nv_bfloat16 * Q_tile     = reinterpret_cast<__nv_bfloat16 *>(gdn_fwd_smem);                                       // 16 KB
    __nv_bfloat16 * K_tile     = Q_tile + S * DK;                                                                       // 16 KB
    __nv_bfloat16 * A_tile     = K_tile + S * DK;                                                                       //  8 KB (later attn_bf16)
    __nv_bfloat16 * V_tile     = A_tile + S * S;                                                                        //  8 KB (V → V_eff → U → O_intra)
    __nv_bfloat16 * h_tile     = V_tile + S * DV_BLK;                                                                   // 16 KB
    float         * gc_tile    = reinterpret_cast<float *>(h_tile + DK * DV_BLK);                                       // 0.25 KB
    float         * scratch    = gc_tile + S;                                                                           // 16 KB ([S, max(S, DV_BLK)] fp32 = 64*64*4 = 16 KB)

    // ── Cooperative load of all tiles for this chunk (128 threads).
    // Q, K [S, DK]: 8192 bf16 each → 64 per thread.
    #pragma unroll 4
    for (int idx = tid; idx < S * DK; idx += 128) {
        const int t_local = idx / DK;
        const int dk      = idx % DK;
        const int t_g     = t_chunk_start + t_local;
        const bool ok     = (t_g < T);
        const int off     = ((b_idx * T + t_g) * H + h_idx) * DK + dk;
        Q_tile[t_local * DK + dk] = ok ? Q[off] : __float2bfloat16(0.0f);
        K_tile[t_local * DK + dk] = ok ? K[off] : __float2bfloat16(0.0f);
    }
    // V [S, DV_BLK]: 4096 bf16 → 32 per thread.
    for (int idx = tid; idx < S * DV_BLK; idx += 128) {
        const int t_local = idx / DV_BLK;
        const int dv_lc   = idx % DV_BLK;
        const int t_g     = t_chunk_start + t_local;
        __nv_bfloat16 v   = (t_g < T)
                            ? V[((b_idx * T + t_g) * H + h_idx) * DV + (dv_off + dv_lc)]
                            : __float2bfloat16(0.0f);
        V_tile[t_local * DV_BLK + dv_lc] = v;
    }
    // A_sol [S, S].
    for (int idx = tid; idx < S * S; idx += 128) {
        const int i = idx / S;
        const int j = idx % S;
        const int t_g = t_chunk_start + i;
        A_tile[i * S + j] = (t_g < T)
                            ? A_sol[((b_idx * T + t_g) * H + h_idx) * S + j]
                            : __float2bfloat16(0.0f);
    }
    for (int t_local = tid; t_local < S; t_local += 128) {
        const int t_g = t_chunk_start + t_local;
        gc_tile[t_local] = (t_g < T) ? g_cumsum[(b_idx * T + t_g) * H + h_idx] : 0.0f;
    }
    // h_tile [DK, DV_BLK]: 8192 bf16 → 64 per thread.
    #pragma unroll 4
    for (int idx = tid; idx < DK * DV_BLK; idx += 128) {
        const int dk    = idx / DV_BLK;
        const int dv_lc = idx % DV_BLK;
        const int hpc_off = (((b_idx * num_chunks + chunk_idx) * H + h_idx) * DK + dk) * DV
                            + (dv_off + dv_lc);
        h_tile[dk * DV_BLK + dv_lc] = h_per_chunk[hpc_off];
    }
    __syncthreads();

    // ───────────────────────────────────────────────────────────────────────
    // STEP-V: V_eff = V - K @ h    [S, DV_BLK] = [S, DK] @ [DK, DV_BLK]
    // Each warp does row-tile [warp_id*16, +15]. 4 col tiles × 8 inner steps each.
    // ───────────────────────────────────────────────────────────────────────
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) wmma::fill_fragment(acc[c], 0.0f);

        const int row_off = warp_id * WMMA_M;
        #pragma unroll
        for (int dk_block = 0; dk_block < DK; dk_block += WMMA_K) {
            wmma::load_matrix_sync(a_frag, K_tile + row_off * DK + dk_block, DK);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                wmma::load_matrix_sync(b_frag, h_tile + dk_block * DV_BLK + c * WMMA_N, DV_BLK);
                wmma::mma_sync(acc[c], a_frag, b_frag, acc[c]);
            }
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            wmma::store_matrix_sync(scratch + row_off * DV_BLK + c * WMMA_N, acc[c],
                                     DV_BLK, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // V_tile bf16 ← V - scratch. (Both [S, DV_BLK].)
    for (int idx = tid; idx < S * DV_BLK; idx += 128) {
        float v_eff = __bfloat162float(V_tile[idx]) - scratch[idx];
        V_tile[idx] = __float2bfloat16(v_eff);
    }
    __syncthreads();

    // ───────────────────────────────────────────────────────────────────────
    // STEP-U: U = A_sol @ V_eff    [S, DV_BLK] = [S, S] @ [S, DV_BLK]
    // ───────────────────────────────────────────────────────────────────────
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) wmma::fill_fragment(acc[c], 0.0f);

        const int row_off = warp_id * WMMA_M;
        #pragma unroll
        for (int kb = 0; kb < S; kb += WMMA_K) {
            wmma::load_matrix_sync(a_frag, A_tile + row_off * S + kb, S);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                wmma::load_matrix_sync(b_frag, V_tile + kb * DV_BLK + c * WMMA_N, DV_BLK);
                wmma::mma_sync(acc[c], a_frag, b_frag, acc[c]);
            }
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            wmma::store_matrix_sync(scratch + row_off * DV_BLK + c * WMMA_N, acc[c],
                                     DV_BLK, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // V_tile bf16 ← U_fp32 cast. (V_tile now holds U_bf16, ready for the
    // attn @ U matmul below.)
    for (int idx = tid; idx < S * DV_BLK; idx += 128) {
        V_tile[idx] = __float2bfloat16(scratch[idx]);
    }
    __syncthreads();

    // ───────────────────────────────────────────────────────────────────────
    // STEP-QK: QK = Q @ K^T    [S, S] = [S, DK] @ [DK, S]
    // K^T is achieved via fragment_b col_major (logical view of K_tile).
    // ───────────────────────────────────────────────────────────────────────
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) wmma::fill_fragment(acc[c], 0.0f);

        const int row_off = warp_id * WMMA_M;
        #pragma unroll
        for (int dk_block = 0; dk_block < DK; dk_block += WMMA_K) {
            wmma::load_matrix_sync(a_frag, Q_tile + row_off * DK + dk_block, DK);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                // Load K viewed as [DK, S] col_major. Tile starting at logical
                // (dk_block, c*16): base = K_tile + (c*16)*DK + dk_block, ldm=DK.
                wmma::load_matrix_sync(b_frag,
                    K_tile + (c * WMMA_N) * DK + dk_block, DK);
                wmma::mma_sync(acc[c], a_frag, b_frag, acc[c]);
            }
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            wmma::store_matrix_sync(scratch + row_off * S + c * WMMA_N, acc[c],
                                     S, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // attn[t, k] = scratch[t, k] · exp(gc[t] - gc[k])  for k ≤ t  else 0.
    // Cast to bf16 in A_tile (was the A_sol storage; A_sol is no longer needed).
    for (int idx = tid; idx < S * S; idx += 128) {
        const int t = idx / S;
        const int k = idx % S;
        float v = 0.0f;
        if (k <= t) {
            v = scratch[idx] * __expf(gc_tile[t] - gc_tile[k]);
        }
        A_tile[idx] = __float2bfloat16(v);
    }
    __syncthreads();

    // ───────────────────────────────────────────────────────────────────────
    // STEP-O_intra: O_intra = attn @ U   [S, DV_BLK] = [S, S] @ [S, DV_BLK]
    // ───────────────────────────────────────────────────────────────────────
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) wmma::fill_fragment(acc[c], 0.0f);

        const int row_off = warp_id * WMMA_M;
        #pragma unroll
        for (int kb = 0; kb < S; kb += WMMA_K) {
            wmma::load_matrix_sync(a_frag, A_tile + row_off * S + kb, S);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                wmma::load_matrix_sync(b_frag, V_tile + kb * DV_BLK + c * WMMA_N, DV_BLK);
                wmma::mma_sync(acc[c], a_frag, b_frag, acc[c]);
            }
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            wmma::store_matrix_sync(scratch + row_off * DV_BLK + c * WMMA_N, acc[c],
                                     DV_BLK, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // V_tile bf16 ← O_intra_fp32 cast. (V_tile becomes O_intra_bf16 stash.)
    for (int idx = tid; idx < S * DV_BLK; idx += 128) {
        V_tile[idx] = __float2bfloat16(scratch[idx]);
    }
    __syncthreads();

    // ───────────────────────────────────────────────────────────────────────
    // STEP-Qh: Qh = Q @ h_tile    [S, DV_BLK] = [S, DK] @ [DK, DV_BLK]
    // ───────────────────────────────────────────────────────────────────────
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) wmma::fill_fragment(acc[c], 0.0f);

        const int row_off = warp_id * WMMA_M;
        #pragma unroll
        for (int dk_block = 0; dk_block < DK; dk_block += WMMA_K) {
            wmma::load_matrix_sync(a_frag, Q_tile + row_off * DK + dk_block, DK);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                wmma::load_matrix_sync(b_frag, h_tile + dk_block * DV_BLK + c * WMMA_N, DV_BLK);
                wmma::mma_sync(acc[c], a_frag, b_frag, acc[c]);
            }
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            wmma::store_matrix_sync(scratch + row_off * DV_BLK + c * WMMA_N, acc[c],
                                     DV_BLK, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // ── FINAL: O[t, dv] = bf2f(V_tile[t,dv]) + exp(gc[t]) · scratch[t,dv]
    for (int idx = tid; idx < S * DV_BLK; idx += 128) {
        const int t     = idx / DV_BLK;
        const int dv_lc = idx % DV_BLK;
        const int t_g   = t_chunk_start + t;
        if (t_g >= T) continue;
        const float o_intra = __bfloat162float(V_tile[idx]);
        const float o_inter = __expf(gc_tile[t]) * scratch[idx];
        const float y       = o_intra + o_inter;
        const int   o_off   = ((b_idx * T + t_g) * H + h_idx) * DV + (dv_off + dv_lc);
        O[o_off] = __float2bfloat16(y);
    }
}

// Launcher for fused_fwd. Sets the dynamic-shared-memory attribute (≈80 KB
// exceeds the default 48 KB cap).
template <int CHUNK_SIZE, int DK, int DV_BLK>
inline void launch_gdn_chunk_fused_fwd(
    const __nv_bfloat16 * Q,
    const __nv_bfloat16 * K,
    const __nv_bfloat16 * V,
    const __nv_bfloat16 * A_sol,
    const float         * g_cumsum,
    const __nv_bfloat16 * h_per_chunk,
    __nv_bfloat16       * O,
    int B, int T, int H, int DV, cudaStream_t stream) {

    constexpr int S = CHUNK_SIZE;
    // wmma layout: Q+K bf16, A bf16, V_tile bf16 (V→V_eff→U→O_intra), h bf16,
    // gc fp32, scratch fp32 [S, max(S, DV_BLK)] = 16 KB.
    constexpr size_t SMEM_Q       = S * DK     * sizeof(__nv_bfloat16);   // 16 KB
    constexpr size_t SMEM_K       = S * DK     * sizeof(__nv_bfloat16);   // 16 KB
    constexpr size_t SMEM_A       = S * S      * sizeof(__nv_bfloat16);   //  8 KB
    constexpr size_t SMEM_V       = S * DV_BLK * sizeof(__nv_bfloat16);   //  8 KB
    constexpr size_t SMEM_H       = DK * DV_BLK* sizeof(__nv_bfloat16);   // 16 KB
    constexpr size_t SMEM_GC      = S          * sizeof(float);           //  0.25 KB
    constexpr size_t SMEM_SCRATCH = (size_t)(S * (S > DV_BLK ? S : DV_BLK)) * sizeof(float);  // 16 KB
    constexpr size_t SMEM_TOTAL   = SMEM_Q + SMEM_K + SMEM_A + SMEM_V + SMEM_H + SMEM_GC + SMEM_SCRATCH;
    static_assert(SMEM_TOTAL <= 99 * 1024,
                  "fused_fwd shared budget exceeded sm_121a 99 KB opt-in cap");

    auto kernel_ptr = gdn_chunk_fused_fwd_kernel<CHUNK_SIZE, DK, DV_BLK>;
    cudaFuncSetAttribute(kernel_ptr,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int) SMEM_TOTAL);

    const int num_chunks    = (T + S - 1) / S;
    const int num_dv_blocks = DV / DV_BLK;
    dim3 grid(num_chunks * num_dv_blocks, H, B);
    dim3 block(128);    // 4 warps for wmma
    kernel_ptr<<<grid, block, SMEM_TOTAL, stream>>>(
        Q, K, V, A_sol, g_cumsum, h_per_chunk, O, B, T, H, DV, num_dv_blocks);
}
