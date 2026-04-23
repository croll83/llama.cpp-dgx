#include "common.cuh"

// Dynamic Walsh-Hadamard Transform — supports any power-of-2 group size
// Group size = head_dim, passed via op_params[1]
// Uses shared memory for groups > 32 (warp size)

// Sign pattern from golden ratio hash
static __device__ float turbo_sign(int i) {
    return ((((unsigned)i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

static __global__ void turbo_wht_kernel(const float * __restrict__ src, float * __restrict__ dst,
                                         const int64_t n_total, const int group_size, const int direction) {
    extern __shared__ float smem[];

    const int64_t group_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t base = group_id * group_size;

    if (base + tid >= n_total) return;

    // Load + apply first signs (forward) or just load (inverse)
    float val;
    if (direction == 0) {
        val = src[base + tid] * turbo_sign(tid);
    } else {
        val = src[base + tid];
    }
    smem[tid] = val;
    __syncthreads();

    // WHT butterfly in shared memory
    for (int step = 1; step < group_size; step <<= 1) {
        int partner = tid ^ step;
        float other = smem[partner];
        __syncthreads();
        if (tid & step) {
            smem[tid] = other - val;
        } else {
            smem[tid] = other + val;
        }
        val = smem[tid];
        __syncthreads();
    }

    // Normalize + apply second signs (inverse) or just normalize (forward)
    float norm = 1.0f / sqrtf((float)group_size);
    if (direction == 0) {
        dst[base + tid] = val * norm;
    } else {
        dst[base + tid] = val * norm * turbo_sign(tid);
#include "turbo-wht.cuh"

// Sign arrays for FWHT rotation (from turbo-wht.h, seed=42)
static __constant__ float d_turbo_wht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1};
static __constant__ float d_turbo_wht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1};

// One block per 128-element group. 128 threads per block.
static __global__ void k_turbo_wht(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements, const int direction) {

    const int64_t group = blockIdx.x;
    const int64_t offset = group * 128;
    if (offset >= n_elements) return;

    const float * s_first  = (direction == 0) ? d_turbo_wht_s1 : d_turbo_wht_s2;
    const float * s_second = (direction == 0) ? d_turbo_wht_s2 : d_turbo_wht_s1;

    __shared__ float buf[128];

    // Load and apply first signs
    if (threadIdx.x < 128) {
        buf[threadIdx.x] = src[offset + threadIdx.x] * s_first[threadIdx.x];
    }
    __syncthreads();

    // Parallel FWHT butterfly: 64 threads, 7 passes
    for (int h = 1; h < 128; h *= 2) {
        if (threadIdx.x < 64) {
            int j = (threadIdx.x / h) * (2 * h) + (threadIdx.x % h);
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b; buf[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize and apply second signs, write output
    constexpr float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    if (threadIdx.x < 128) {
        dst[offset + threadIdx.x] = buf[threadIdx.x] * inv_sqrt_128 * s_second[threadIdx.x];
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;

    int32_t params[2];
    memcpy(params, dst->op_params, sizeof(params));
    const int direction = params[0];
    // Group size from ne[0] (head_dim) — must be power of 2
    const int group_size = (int)src->ne[0];
    GGML_ASSERT((group_size & (group_size - 1)) == 0); // power of 2

    const int64_t n_total = ggml_nelements(src);
    const int64_t n_groups = n_total / group_size;

    cudaStream_t stream = ctx.stream();
    turbo_wht_kernel<<<n_groups, group_size, group_size * sizeof(float), stream>>>(
        src_d, dst_d, n_total, group_size, direction);
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const float * src_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_elements = ggml_nelements(src0);
    const int64_t n_groups = n_elements / 128;

    k_turbo_wht<<<(int)n_groups, 128, 0, stream>>>(src_d, dst_d, n_elements, direction);
}
