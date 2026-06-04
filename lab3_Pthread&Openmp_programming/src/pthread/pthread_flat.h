#pragma once
#include <pthread.h>
#include <queue>
#include <vector>
#include <cstdint>
#include "../simd/simd_search.h"

struct FlatThreadData {
    float* base;
    float* query;
    size_t start_idx;
    size_t end_idx;
    size_t vecdim;
    size_t k;
    std::priority_queue<std::pair<float, uint32_t>>* result;
};

void* pthread_flat_worker(void* arg) {
    FlatThreadData* data = (FlatThreadData*)arg;
    auto& res = *(data->result);

    for (size_t i = data->start_idx; i < data->end_idx; ++i) {
        float ip = inner_product_neon(data->base + i * data->vecdim, data->query, data->vecdim);
        float dis = 1.0f - ip;

        if (res.size() < data->k) {
            res.push({dis, (uint32_t)i});
        } else if (dis < res.top().first) {
            res.push({dis, (uint32_t)i});
            res.pop();
        }
    }
    return nullptr;
}

std::priority_queue<std::pair<float, uint32_t>> pthread_flat_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, int num_threads)
{
    std::vector<pthread_t> threads(num_threads);
    std::vector<FlatThreadData> thread_data(num_threads);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(num_threads);

    size_t chunk_size = base_number / num_threads;
    size_t remainder = base_number % num_threads;

    for (int t = 0; t < num_threads; ++t) {
        thread_data[t].base = base;
        thread_data[t].query = query;
        thread_data[t].start_idx = t * chunk_size + std::min((size_t)t, remainder);
        thread_data[t].end_idx = thread_data[t].start_idx + chunk_size + (t < (int)remainder ? 1 : 0);
        thread_data[t].vecdim = vecdim;
        thread_data[t].k = k;
        thread_data[t].result = &local_results[t];

        pthread_create(&threads[t], nullptr, pthread_flat_worker, &thread_data[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }

    std::priority_queue<std::pair<float, uint32_t>> global;
    for (auto& local : local_results) {
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