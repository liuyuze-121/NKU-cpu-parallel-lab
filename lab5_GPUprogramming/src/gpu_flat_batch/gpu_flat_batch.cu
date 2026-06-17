#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

struct GpuHit {
    float distance;
    uint32_t id;
};

// Matrix baseline required by the lab handout:
// base is N x d, queries is batch x d, scores is batch x N.
// Each CUDA thread computes one inner product. This is intentionally simple
// and is used as the baseline for batch-size tuning.
__global__ void flat_score_kernel(const float* __restrict__ base,
                                  const float* __restrict__ queries,
                                  float* __restrict__ scores,
                                  int n,
                                  int d,
                                  int batch) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * batch;
    if (tid >= total) return;

    const int q = tid / n;
    const int i = tid - q * n;
    float dot = 0.0f;
    for (int j = 0; j < d; ++j) {
        dot += base[i * d + j] * queries[q * d + j];
    }
    scores[tid] = 1.0f - dot;
}

std::vector<std::vector<GpuHit>> select_topk_on_host(const float* scores,
                                                     int n,
                                                     int batch,
                                                     int k) {
    std::vector<std::vector<GpuHit>> output(batch);
    for (int q = 0; q < batch; ++q) {
        std::priority_queue<std::pair<float, uint32_t>> heap;
        const float* row = scores + static_cast<size_t>(q) * n;
        for (int i = 0; i < n; ++i) {
            const float dist = row[i];
            if (static_cast<int>(heap.size()) < k) {
                heap.push({dist, static_cast<uint32_t>(i)});
            } else if (dist < heap.top().first) {
                heap.push({dist, static_cast<uint32_t>(i)});
                heap.pop();
            }
        }
        auto& hits = output[q];
        hits.reserve(k);
        while (!heap.empty()) {
            hits.push_back({heap.top().first, heap.top().second});
            heap.pop();
        }
        std::sort(hits.begin(), hits.end(), [](const GpuHit& a, const GpuHit& b) {
            if (a.distance != b.distance) return a.distance < b.distance;
            return a.id < b.id;
        });
    }
    return output;
}

float launch_flat_score_kernel(const float* d_base,
                               const float* d_queries,
                               float* d_scores,
                               int n,
                               int d,
                               int batch,
                               int block_size) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int total = n * batch;
    const int grid = (total + block_size - 1) / block_size;
    cudaEventRecord(start);
    flat_score_kernel<<<grid, block_size>>>(d_base, d_queries, d_scores, n, d, batch);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms;
}
