#include <cuda_runtime.h>

#include <cstdint>

// IVF matrix baseline from the lab handout.
// For one selected cluster, evaluate all vectors in that cluster against all
// queries in the current batch. The host repeats this for the union of clusters
// selected by the batch, then keeps scores only for queries that really probed
// the current cluster. The wasted work is measured as computed_pairs/useful_pairs.
__global__ void ivf_cluster_score_kernel(const float* __restrict__ base,
                                         const float* __restrict__ queries,
                                         const int* __restrict__ candidate_ids,
                                         float* __restrict__ scores,
                                         int candidate_count,
                                         int d,
                                         int batch) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = candidate_count * batch;
    if (tid >= total) return;

    const int q = tid / candidate_count;
    const int local = tid - q * candidate_count;
    const int id = candidate_ids[local];
    float dot = 0.0f;
    for (int j = 0; j < d; ++j) {
        dot += base[id * d + j] * queries[q * d + j];
    }
    scores[tid] = 1.0f - dot;
}

float launch_ivf_cluster_score_kernel(const float* d_base,
                                      const float* d_queries,
                                      const int* d_candidate_ids,
                                      float* d_scores,
                                      int candidate_count,
                                      int d,
                                      int batch,
                                      int block_size) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int total = candidate_count * batch;
    const int grid = (total + block_size - 1) / block_size;
    cudaEventRecord(start);
    ivf_cluster_score_kernel<<<grid, block_size>>>(
        d_base, d_queries, d_candidate_ids, d_scores, candidate_count, d, batch);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms;
}
