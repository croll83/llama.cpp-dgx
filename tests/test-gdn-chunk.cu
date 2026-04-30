// test_gdn_chunk.cu — standalone CUDA test for the chunk-fused GDN kernels.
//
// Loads a fixture (`/tmp/gdn_chunk_fixture.bin`) produced by
// tools/test/gdn_chunk_ref.py, runs the in-tree CUDA kernels, and compares
// against the PyTorch ground-truth. Prints PASS/FAIL with element-wise
// max-abs-diff stats for each kernel.
//
// Build standalone:
//   nvcc -O2 -std=c++17 -arch=sm_121a \
//       /tmp/test_gdn_chunk.cu \
//       -I /home/jarvis/llama-cpp-v5/ggml/src/ggml-cuda \
//       -I /home/jarvis/llama-cpp-v5/ggml/src \
//       -I /home/jarvis/llama-cpp-v5/ggml/include \
//       -o /tmp/test_gdn_chunk

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(2); \
    } \
} while (0)

#include "gated_delta_net_chunk_kernels.cuh"

// ─── Fixture loader ─────────────────────────────────────────────────────────

struct Fixture {
    int B, T, H, DK, DV, S;
    std::vector<float>    g;            // [B*T*H]
    std::vector<uint16_t> Q_bits;       // [B*T*H*DK]
    std::vector<uint16_t> K_bits;       // [B*T*H*DK]
    std::vector<uint16_t> V_bits;       // [B*T*H*DV]
    std::vector<float>    beta;         // [B*T*H]
    std::vector<float>    g_cum_exp;    // [B*T*H]
    std::vector<uint16_t> A_sol_exp;    // [B*T*H*S]
    std::vector<uint16_t> h_initial;    // [B*H*DK*DV]
    std::vector<uint16_t> hpc_exp;      // [B*nc*H*DK*DV]
    std::vector<uint16_t> hfinal_exp;   // [B*H*DK*DV]
    std::vector<uint16_t> O_exp;        // [B*T*H*DV]
};

static Fixture load_fixture(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }

    Fixture f;
    int hdr[6];
    in.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    f.B  = hdr[0]; f.T = hdr[1]; f.H = hdr[2];
    f.DK = hdr[3]; f.DV = hdr[4]; f.S  = hdr[5];

    const int  num_chunks = (f.T + f.S - 1) / f.S;
    const size_t n_BTH    = (size_t) f.B * f.T * f.H;
    const size_t n_BTH_DK = n_BTH * f.DK;
    const size_t n_BTH_DV = n_BTH * f.DV;
    const size_t n_BTH_S  = n_BTH * f.S;
    const size_t n_BHDD   = (size_t) f.B * f.H * f.DK * f.DV;
    const size_t n_HPC    = (size_t) f.B * num_chunks * f.H * f.DK * f.DV;

    f.g.resize(n_BTH);
    f.Q_bits.resize(n_BTH_DK);
    f.K_bits.resize(n_BTH_DK);
    f.V_bits.resize(n_BTH_DV);
    f.beta.resize(n_BTH);
    f.g_cum_exp.resize(n_BTH);
    f.A_sol_exp.resize(n_BTH_S);
    f.h_initial.resize(n_BHDD);
    f.hpc_exp.resize(n_HPC);
    f.hfinal_exp.resize(n_BHDD);
    f.O_exp.resize(n_BTH_DV);

    in.read(reinterpret_cast<char *>(f.g.data()),         n_BTH * sizeof(float));
    in.read(reinterpret_cast<char *>(f.Q_bits.data()),    n_BTH_DK * sizeof(uint16_t));
    in.read(reinterpret_cast<char *>(f.K_bits.data()),    n_BTH_DK * sizeof(uint16_t));
    in.read(reinterpret_cast<char *>(f.V_bits.data()),    n_BTH_DV * sizeof(uint16_t));
    in.read(reinterpret_cast<char *>(f.beta.data()),      n_BTH * sizeof(float));
    in.read(reinterpret_cast<char *>(f.g_cum_exp.data()), n_BTH * sizeof(float));
    in.read(reinterpret_cast<char *>(f.A_sol_exp.data()), n_BTH_S * sizeof(uint16_t));
    in.read(reinterpret_cast<char *>(f.h_initial.data()), n_BHDD * sizeof(uint16_t));
    in.read(reinterpret_cast<char *>(f.hpc_exp.data()),   n_HPC * sizeof(uint16_t));
    in.read(reinterpret_cast<char *>(f.hfinal_exp.data()),n_BHDD * sizeof(uint16_t));
    in.read(reinterpret_cast<char *>(f.O_exp.data()),     n_BTH_DV * sizeof(uint16_t));
    if (!in.good()) { std::fprintf(stderr, "fixture read short at %s\n", path); std::exit(2); }
    return f;
}

// ─── Compare helpers ────────────────────────────────────────────────────────

struct DiffStats {
    double max_abs;
    double mean_abs;
    double max_rel;
    size_t argmax;
};

static DiffStats compare_f32(const std::vector<float> & got,
                             const std::vector<float> & exp) {
    DiffStats s{0, 0, 0, 0};
    double sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double)got[i] - (double)exp[i]);
        if (d > s.max_abs) { s.max_abs = d; s.argmax = i; }
        sum += d;
        double denom = std::max(std::fabs((double)exp[i]), 1e-6);
        double r = d / denom;
        if (r > s.max_rel) s.max_rel = r;
    }
    s.mean_abs = sum / (double) got.size();
    return s;
}

static DiffStats compare_bf16(const std::vector<uint16_t> & got_bits,
                              const std::vector<uint16_t> & exp_bits) {
    DiffStats s{0, 0, 0, 0};
    double sum = 0.0;
    for (size_t i = 0; i < got_bits.size(); ++i) {
        uint32_t got_u = ((uint32_t) got_bits[i]) << 16;
        uint32_t exp_u = ((uint32_t) exp_bits[i]) << 16;
        float got_f, exp_f;
        std::memcpy(&got_f, &got_u, 4);
        std::memcpy(&exp_f, &exp_u, 4);
        double d = std::fabs((double)got_f - (double)exp_f);
        if (d > s.max_abs) { s.max_abs = d; s.argmax = i; }
        sum += d;
        double denom = std::max(std::fabs((double)exp_f), 1e-6);
        double r = d / denom;
        if (r > s.max_rel) s.max_rel = r;
    }
    s.mean_abs = sum / (double) got_bits.size();
    return s;
}

// ─── Tests ─────────────────────────────────────────────────────────────────

static int test_cumsum(const Fixture & f) {
    std::printf("\n[test cumsum] B=%d T=%d H=%d (chunk_size=%d)\n", f.B, f.T, f.H, f.S);

    const size_t n_BTH = (size_t) f.B * f.T * f.H;

    float *d_g, *d_g_cum;
    CUDA_CHECK(cudaMalloc(&d_g,     n_BTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_cum, n_BTH * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_g, f.g.data(), n_BTH * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_g_cum, 0,      n_BTH * sizeof(float)));

    const int num_chunks = (f.T + f.S - 1) / f.S;
    dim3 grid(num_chunks, f.H, f.B);
    dim3 block(f.S);
    gdn_chunk_local_cumsum_kernel<<<grid, block>>>(d_g, d_g_cum, f.B, f.T, f.H, f.S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(n_BTH);
    CUDA_CHECK(cudaMemcpy(got.data(), d_g_cum, n_BTH * sizeof(float), cudaMemcpyDeviceToHost));

    DiffStats s = compare_f32(got, f.g_cum_exp);
    std::printf("  max_abs_diff = %.6e   mean_abs_diff = %.6e   max_rel = %.6e\n",
                s.max_abs, s.mean_abs, s.max_rel);
    const bool pass = s.max_abs < 1e-4;
    std::printf("  %s\n", pass ? "PASS ✓" : "FAIL ✗");

    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_g_cum));
    return pass ? 0 : 1;
}

static int test_kkt_solve(const Fixture & f) {
    std::printf("\n[test kkt_solve] B=%d T=%d H=%d DK=%d S=%d\n", f.B, f.T, f.H, f.DK, f.S);
    if (f.DK != 128 || f.S != 64) {
        std::printf("  SKIP — kernel currently instantiated only for DK=128, S=64\n");
        return 0;
    }

    const size_t n_BTH    = (size_t) f.B * f.T * f.H;
    const size_t n_BTH_DK = n_BTH * f.DK;
    const size_t n_BTH_S  = n_BTH * f.S;

    __nv_bfloat16 *d_K, *d_A_sol;
    float *d_beta, *d_g_cum;
    CUDA_CHECK(cudaMalloc(&d_K,     n_BTH_DK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_beta,  n_BTH    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_cum, n_BTH    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_sol, n_BTH_S  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_K,     f.K_bits.data(),    n_BTH_DK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  f.beta.data(),      n_BTH    * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_cum, f.g_cum_exp.data(), n_BTH    * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_A_sol, 0, n_BTH_S * sizeof(__nv_bfloat16)));

    launch_gdn_chunk_kkt_solve<64, 128>(d_K, d_beta, d_g_cum, d_A_sol, f.B, f.T, f.H, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got(n_BTH_S);
    CUDA_CHECK(cudaMemcpy(got.data(), d_A_sol, n_BTH_S * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    DiffStats s = compare_bf16(got, f.A_sol_exp);
    std::printf("  max_abs_diff = %.6e   mean_abs_diff = %.6e   max_rel = %.6e\n",
                s.max_abs, s.mean_abs, s.max_rel);
    const bool pass = s.max_abs < 5e-3 && s.max_rel < 5e-2;
    std::printf("  %s\n", pass ? "PASS ✓" : "FAIL ✗");

    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_g_cum));
    CUDA_CHECK(cudaFree(d_A_sol));
    return pass ? 0 : 1;
}

static int test_prepare_h(const Fixture & f) {
    std::printf("\n[test prepare_h] B=%d T=%d H=%d DK=%d DV=%d S=%d\n",
                f.B, f.T, f.H, f.DK, f.DV, f.S);
    if (f.DK != 128 || f.DV != 128 || f.S != 64) {
        std::printf("  SKIP — kernel currently instantiated only for DK=128, DV=128, S=64\n");
        return 0;
    }

    const int num_chunks = (f.T + f.S - 1) / f.S;
    const size_t n_BTH    = (size_t) f.B * f.T * f.H;
    const size_t n_BTH_DK = n_BTH * f.DK;
    const size_t n_BTH_DV = n_BTH * f.DV;
    const size_t n_BHDD   = (size_t) f.B * f.H * f.DK * f.DV;
    const size_t n_HPC    = (size_t) f.B * num_chunks * f.H * f.DK * f.DV;

    __nv_bfloat16 *d_K, *d_V, *d_h0, *d_hpc, *d_hfinal;
    float *d_beta, *d_g_cum;
    CUDA_CHECK(cudaMalloc(&d_K,      n_BTH_DK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V,      n_BTH_DV * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_beta,   n_BTH    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_cum,  n_BTH    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h0,     n_BHDD   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_hpc,    n_HPC    * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_hfinal, n_BHDD   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_K,     f.K_bits.data(),    n_BTH_DK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,     f.V_bits.data(),    n_BTH_DV * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  f.beta.data(),      n_BTH    * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_cum, f.g_cum_exp.data(), n_BTH    * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0,    f.h_initial.data(), n_BHDD   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hpc,    0, n_HPC  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_hfinal, 0, n_BHDD * sizeof(__nv_bfloat16)));

    // Pass A_sol_exp (precomputed reference) into prepare_h.
    __nv_bfloat16 * d_A_sol_in;
    CUDA_CHECK(cudaMalloc(&d_A_sol_in, (size_t) f.B * f.T * f.H * f.S * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_A_sol_in, f.A_sol_exp.data(),
                          (size_t) f.B * f.T * f.H * f.S * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    launch_gdn_chunk_prepare_h<64, 128, 64>(d_K, d_V, d_g_cum, d_A_sol_in,
                                              d_h0, d_hpc, d_hfinal,
                                              f.B, f.T, f.H, f.DV, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got_hpc(n_HPC);
    std::vector<uint16_t> got_hf(n_BHDD);
    CUDA_CHECK(cudaMemcpy(got_hpc.data(), d_hpc,    n_HPC  * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_hf.data(),  d_hfinal, n_BHDD * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    DiffStats s_hpc = compare_bf16(got_hpc, f.hpc_exp);
    DiffStats s_hf  = compare_bf16(got_hf,  f.hfinal_exp);
    std::printf("  h_per_chunk: max_abs=%.6e mean_abs=%.6e max_rel=%.6e\n",
                s_hpc.max_abs, s_hpc.mean_abs, s_hpc.max_rel);
    std::printf("  h_final:     max_abs=%.6e mean_abs=%.6e max_rel=%.6e\n",
                s_hf.max_abs, s_hf.mean_abs, s_hf.max_rel);

    const bool pass = (s_hpc.max_abs < 5e-2) && (s_hf.max_abs < 5e-2);
    std::printf("  %s\n", pass ? "PASS ✓" : "FAIL ✗");

    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_g_cum));
    CUDA_CHECK(cudaFree(d_h0));
    CUDA_CHECK(cudaFree(d_hpc));
    CUDA_CHECK(cudaFree(d_hfinal));
    CUDA_CHECK(cudaFree(d_A_sol_in));
    return pass ? 0 : 1;
}

static int test_fused_fwd(const Fixture & f) {
    std::printf("\n[test fused_fwd] B=%d T=%d H=%d DK=%d DV=%d S=%d\n",
                f.B, f.T, f.H, f.DK, f.DV, f.S);
    if (f.DK != 128 || f.DV != 128 || f.S != 64) {
        std::printf("  SKIP — kernel currently instantiated only for DK=128, DV=128, S=64\n");
        return 0;
    }

    const int num_chunks = (f.T + f.S - 1) / f.S;
    const size_t n_BTH    = (size_t) f.B * f.T * f.H;
    const size_t n_BTH_DK = n_BTH * f.DK;
    const size_t n_BTH_DV = n_BTH * f.DV;
    const size_t n_BTH_S  = n_BTH * f.S;
    const size_t n_HPC    = (size_t) f.B * num_chunks * f.H * f.DK * f.DV;

    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_A, *d_hpc, *d_O;
    float *d_g_cum;
    CUDA_CHECK(cudaMalloc(&d_Q,     n_BTH_DK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K,     n_BTH_DK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V,     n_BTH_DV * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_A,     n_BTH_S  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_g_cum, n_BTH    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hpc,   n_HPC    * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_O,     n_BTH_DV * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_Q,     f.Q_bits.data(),    n_BTH_DK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,     f.K_bits.data(),    n_BTH_DK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,     f.V_bits.data(),    n_BTH_DV * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A,     f.A_sol_exp.data(), n_BTH_S  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_cum, f.g_cum_exp.data(), n_BTH    * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hpc,   f.hpc_exp.data(),   n_HPC    * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, n_BTH_DV * sizeof(__nv_bfloat16)));

    launch_gdn_chunk_fused_fwd<64, 128, 64>(d_Q, d_K, d_V, d_A, d_g_cum,
                                              d_hpc, d_O, f.B, f.T, f.H, f.DV, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got(n_BTH_DV);
    CUDA_CHECK(cudaMemcpy(got.data(), d_O, n_BTH_DV * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    DiffStats s = compare_bf16(got, f.O_exp);

    // Compute the max_abs of the *expected* values to scale the tolerance
    // (bf16 quantization noise is ~|x|·2⁻⁸ at full mantissa).
    double abs_max_exp = 0.0;
    for (size_t i = 0; i < f.O_exp.size(); ++i) {
        uint32_t u = ((uint32_t) f.O_exp[i]) << 16;
        float v;  std::memcpy(&v, &u, 4);
        if (std::fabs((double)v) > abs_max_exp) abs_max_exp = std::fabs((double)v);
    }
    std::printf("  O: max_abs=%.6e mean_abs=%.6e max_rel=%.6e (argmax=%zu, |exp|max=%.4f)\n",
                s.max_abs, s.mean_abs, s.max_rel, s.argmax, abs_max_exp);

    // bf16 mantissa is 7 bits → relative precision ≈ 2⁻⁸ ≈ 4e-3. After O(S)
    // multiplied accumulations the noise floor is a few × 2⁻⁸ × |x|max.
    const double abs_tol = 1.0e-2 * abs_max_exp + 5e-3;       // ~1% of max + epsilon
    const double mean_tol = 5.0e-3;                            // 0.5% of mean magnitude
    const bool pass = (s.max_abs < abs_tol) && (s.mean_abs < mean_tol);
    std::printf("  abs_tol=%.4e  mean_tol=%.4e  → %s\n",
                abs_tol, mean_tol, pass ? "PASS ✓" : "FAIL ✗");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_g_cum));
    CUDA_CHECK(cudaFree(d_hpc));
    CUDA_CHECK(cudaFree(d_O));
    return pass ? 0 : 1;
}

// ─── Micro-benchmark mode ──────────────────────────────────────────────────
// Times each kernel individually using CUDA events, plus the full pipeline.
// Reports min/median/mean over N iterations after a warmup.

struct BenchStats {
    double min_us, median_us, mean_us, max_us;
};

static BenchStats compute_stats(std::vector<double> & ms) {
    std::sort(ms.begin(), ms.end());
    BenchStats s;
    s.min_us    = ms.front() * 1000.0;
    s.max_us    = ms.back()  * 1000.0;
    s.median_us = ms[ms.size() / 2] * 1000.0;
    double sum = 0.0;
    for (double v : ms) sum += v;
    s.mean_us = (sum / ms.size()) * 1000.0;
    return s;
}

static int bench(const Fixture & f, int n_iters) {
    if (f.DK != 128 || f.DV != 128 || f.S != 64) {
        std::printf("BENCH: SKIP (kernels only instantiated for DK=128 DV=128 S=64)\n");
        return 0;
    }
    std::printf("\n=== microbench: B=%d T=%d H=%d DK=%d DV=%d S=%d  iters=%d ===\n",
                f.B, f.T, f.H, f.DK, f.DV, f.S, n_iters);

    const int num_chunks = (f.T + f.S - 1) / f.S;
    const size_t n_BTH    = (size_t) f.B * f.T * f.H;
    const size_t n_BTH_DK = n_BTH * f.DK;
    const size_t n_BTH_DV = n_BTH * f.DV;
    const size_t n_BTH_S  = n_BTH * f.S;
    const size_t n_BHDD   = (size_t) f.B * f.H * f.DK * f.DV;
    const size_t n_HPC    = (size_t) f.B * num_chunks * f.H * f.DK * f.DV;

    float *d_g, *d_g_cum, *d_beta;
    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_A, *d_h0, *d_hpc, *d_hf, *d_O;
    CUDA_CHECK(cudaMalloc(&d_g,     n_BTH    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_cum, n_BTH    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta,  n_BTH    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q,     n_BTH_DK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K,     n_BTH_DK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V,     n_BTH_DV * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_A,     n_BTH_S  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_h0,    n_BHDD   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_hpc,   n_HPC    * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_hf,    n_BHDD   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_O,     n_BTH_DV * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_g,     f.g.data(),         n_BTH    * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_cum, f.g_cum_exp.data(), n_BTH    * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  f.beta.data(),      n_BTH    * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q,     f.Q_bits.data(),    n_BTH_DK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,     f.K_bits.data(),    n_BTH_DK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,     f.V_bits.data(),    n_BTH_DV * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A,     f.A_sol_exp.data(), n_BTH_S  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0,    f.h_initial.data(), n_BHDD   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hpc,   f.hpc_exp.data(),   n_HPC    * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto bench_kernel = [&](const char * name, auto && launcher) {
        // warmup
        for (int i = 0; i < 3; ++i) launcher();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times(n_iters);
        for (int i = 0; i < n_iters; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            launcher();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            times[i] = ms;
        }
        BenchStats s = compute_stats(times);
        std::printf("  %-14s: min=%7.2f us  median=%7.2f us  mean=%7.2f us  max=%7.2f us\n",
                    name, s.min_us, s.median_us, s.mean_us, s.max_us);
        return s.median_us;
    };

    // 1. cumsum
    double t_cumsum = bench_kernel("cumsum", [&]{
        dim3 grid(num_chunks, f.H, f.B);
        dim3 block(f.S);
        gdn_chunk_local_cumsum_kernel<<<grid, block>>>(d_g, d_g_cum, f.B, f.T, f.H, f.S);
    });

    // 2. kkt_solve
    double t_kkt = bench_kernel("kkt_solve", [&]{
        launch_gdn_chunk_kkt_solve<64, 128>(d_K, d_beta, d_g_cum, d_A, f.B, f.T, f.H, 0);
    });

    // 3. prepare_h
    double t_ph = bench_kernel("prepare_h", [&]{
        launch_gdn_chunk_prepare_h<64, 128, 64>(
            d_K, d_V, d_g_cum, d_A, d_h0, d_hpc, d_hf,
            f.B, f.T, f.H, f.DV, 0);
    });

    // 4. fused_fwd
    double t_ff = bench_kernel("fused_fwd", [&]{
        launch_gdn_chunk_fused_fwd<64, 128, 64>(
            d_Q, d_K, d_V, d_A, d_g_cum, d_hpc, d_O,
            f.B, f.T, f.H, f.DV, 0);
    });

    // 5. full pipeline (chained, no upload/download)
    double t_pipe = bench_kernel("FULL pipeline", [&]{
        dim3 grid(num_chunks, f.H, f.B);
        dim3 block(f.S);
        gdn_chunk_local_cumsum_kernel<<<grid, block>>>(d_g, d_g_cum, f.B, f.T, f.H, f.S);
        launch_gdn_chunk_kkt_solve<64, 128>(d_K, d_beta, d_g_cum, d_A, f.B, f.T, f.H, 0);
        launch_gdn_chunk_prepare_h<64, 128, 64>(
            d_K, d_V, d_g_cum, d_A, d_h0, d_hpc, d_hf,
            f.B, f.T, f.H, f.DV, 0);
        launch_gdn_chunk_fused_fwd<64, 128, 64>(
            d_Q, d_K, d_V, d_A, d_g_cum, d_hpc, d_O,
            f.B, f.T, f.H, f.DV, 0);
    });

    const double sum_indiv = t_cumsum + t_kkt + t_ph + t_ff;
    std::printf("  ─────────────────────────────────────────────────────\n");
    std::printf("  Σ individual = %.2f us   pipeline = %.2f us   overlap savings = %.2f us\n",
                sum_indiv, t_pipe, sum_indiv - t_pipe);
    std::printf("  bottleneck breakdown (median):\n");
    std::printf("    cumsum:    %5.1f%%  (%.2f us)\n", 100.0 * t_cumsum / sum_indiv, t_cumsum);
    std::printf("    kkt_solve: %5.1f%%  (%.2f us)\n", 100.0 * t_kkt    / sum_indiv, t_kkt);
    std::printf("    prepare_h: %5.1f%%  (%.2f us)\n", 100.0 * t_ph     / sum_indiv, t_ph);
    std::printf("    fused_fwd: %5.1f%%  (%.2f us)\n", 100.0 * t_ff     / sum_indiv, t_ff);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_g); cudaFree(d_g_cum); cudaFree(d_beta);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_A);
    cudaFree(d_h0); cudaFree(d_hpc); cudaFree(d_hf); cudaFree(d_O);
    return 0;
}

int main(int argc, char ** argv) {
    const char * fixture = (argc > 1) ? argv[1] : "/tmp/gdn_chunk_fixture.bin";
    int  n_bench_iters   = 0;
    bool do_tests        = true;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) {
            n_bench_iters = (i + 1 < argc) ? std::atoi(argv[i + 1]) : 100;
            i++;
        } else if (std::strcmp(argv[i], "--no-tests") == 0) {
            do_tests = false;
        }
    }

    Fixture f = load_fixture(fixture);
    std::printf("loaded fixture %s: B=%d T=%d H=%d DK=%d DV=%d S=%d\n",
                fixture, f.B, f.T, f.H, f.DK, f.DV, f.S);

    int fails = 0;
    if (do_tests) {
        fails += test_cumsum(f);
        fails += test_kkt_solve(f);
        fails += test_prepare_h(f);
        fails += test_fused_fwd(f);
        std::printf("\n=== %d test(s) failed ===\n", fails);
    }
    if (n_bench_iters > 0) {
        bench(f, n_bench_iters);
    }
    return fails;
}
