#include <cuda_runtime.h>

#include <cstdint>

// Optimized IVF scoring kernel.
// Instead of evaluating every vector in the union of all clusters against every
// query in the batch, the host flattens only valid (query, candidate) pairs.
// This removes invalid work at the cost of an extra q_ids indirection.
__global__ void ivf_compact_score_kernel(const float* __restrict__ base,
                                         const float* __restrict__ queries,
                                         const int* __restrict__ candidate_ids,
                                         const int* __restrict__ query_ids,
                                         float* __restrict__ scores,
                                         int pair_count,
                                         int d) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pair_count) return;

    const int id = candidate_ids[tid];
    const int q = query_ids[tid];
    float dot = 0.0f;
    for (int j = 0; j < d; ++j) {
        dot += base[id * d + j] * queries[q * d + j];
    }
    scores[tid] = 1.0f - dot;
}

float launch_ivf_compact_score_kernel(const float* d_base,
                                      const float* d_queries,
                                      const int* d_candidate_ids,
                                      const int* d_query_ids,
                                      float* d_scores,
                                      int pair_count,
                                      int d,
                                      int block_size) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int grid = (pair_count + block_size - 1) / block_size;
    cudaEventRecord(start);
    ivf_compact_score_kernel<<<grid, block_size>>>(
        d_base, d_queries, d_candidate_ids, d_query_ids, d_scores, pair_count, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms;
}
