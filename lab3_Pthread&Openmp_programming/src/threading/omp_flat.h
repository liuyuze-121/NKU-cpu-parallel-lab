#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <arm_neon.h>
#include <omp.h>

static float inner_product_neon_omp(const float* a, const float* b, size_t dim) {
    float32x4_t sum0 = vmovq_n_f32(0.0f);
    float32x4_t sum1 = vmovq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        sum0 = vfmaq_f32(sum0, a0, b0);
        sum1 = vfmaq_f32(sum1, a1, b1);
    }
    float extra = 0.0f;
    for (; i < dim; i++) extra += a[i] * b[i];
    sum0 = vaddq_f32(sum0, sum1);
    return vaddvq_f32(sum0) + extra;
}

std::priority_queue<std::pair<float, uint32_t>> omp_flat_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    int num_threads = omp_get_max_threads();
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> locals(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < base_number; ++i) {
            float ip = inner_product_neon_omp(base + i * vecdim, query, vecdim);
            float dis = 1.0f - ip;

            auto& local = locals[tid];
            if (local.size() < k) {
                local.push({dis, (uint32_t)i});
            } else if (dis < local.top().first) {
                local.push({dis, (uint32_t)i});
                local.pop();
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> global;
    for (auto& local : locals) {
        while (!local.empty()) {
            auto top = local.top();
            local.pop();
            if (global.size() < k) {
                global.push(top);
            } else if (top.first < global.top().first) {
                global.push(top);
                global.pop();
            }
        }
    }
    return global;
}