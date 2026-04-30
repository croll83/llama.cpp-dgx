#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Chunked prefill path. Caller must gate on n_tokens >= 64 and tree_mode == false.
// See gated_delta_net_chunk.cu and docs/rfc-gdn-chunk-kernel.md.
void ggml_cuda_op_gated_delta_net_chunk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Routes between the per-token (decode / tree) and chunked (prefill) kernels.
// The chunked path is taken iff:
//   - parent_ids (src[6]) is null (no tree mode)
//   - g is per-head scalar (not KDA per-element)
//   - S_v == 128
//   - n_tokens >= 64
// Falls back to ggml_cuda_op_gated_delta_net otherwise.
void ggml_cuda_op_gated_delta_net_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
