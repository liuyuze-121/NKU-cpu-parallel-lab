#pragma once
#include <pthread.h>
#include <queue>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <algorithm>
#include <arm_neon.h>

class IVFFlatSearcherPthread {
public:
    int nlist_;
    size_t n_, dim_;
    float* base_;
    std::vector<float> centroids_;
    std::vector<std::vector<uint32_t>> lists_;

    IVFFlatSearcherPthread(int nlist, float* base, size_t n, size_t dim)
        : nlist_(nlist), base_(base), n_(n), dim_(dim) {}

    static float inner_product_neon(const float* a, const float* b, size_t dim) {
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

    void train() {
        centroids_.resize((size_t)nlist_ * dim_);

        for (int c = 0; c < nlist_; ++c) {
            size_t idx = rand() % n_;
            memcpy(centroids_.data() + c * dim_, base_ + idx * dim_, dim_ * sizeof(float));
        }

        for (int iter = 0; iter < 20; ++iter) {
            std::vector<int> assignments(n_);
            for (size_t i = 0; i < n_; ++i) {
                float best_sim = -FLT_MAX;
                int best_c = 0;
                for (int c = 0; c < nlist_; ++c) {
                    float sim = inner_product_neon(base_ + i * dim_, centroids_.data() + c * dim_, dim_);
                    if (sim > best_sim) {
                        best_sim = sim;
                        best_c = c;
                    }
                }
                assignments[i] = best_c;
            }

            std::vector<int> counts(nlist_, 0);
            std::vector<float> new_centroids((size_t)nlist_ * dim_, 0.0f);

            for (size_t i = 0; i < n_; ++i) {
                int c = assignments[i];
                counts[c]++;
                for (size_t d = 0; d < dim_; ++d) {
                    new_centroids[c * dim_ + d] += base_[i * dim_ + d];
                }
            }

            for (int c = 0; c < nlist_; ++c) {
                if (counts[c] > 0) {
                    for (size_t d = 0; d < dim_; ++d) {
                        centroids_[c * dim_ + d] = new_centroids[c * dim_ + d] / counts[c];
                    }
                }
            }
        }
    }

    void assign_base() {
        lists_.resize(nlist_);
        for (size_t i = 0; i < n_; ++i) {
            float best_sim = -FLT_MAX;
            int best_c = 0;
            for (int c = 0; c < nlist_; ++c) {
                float sim = inner_product_neon(base_ + i * dim_, centroids_.data() + c * dim_, dim_);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_c = c;
                }
            }
            lists_[best_c].push_back((uint32_t)i);
        }
    }

    struct IVFThreadData {
        IVFFlatSearcherPthread* ivf;
        float* query;
        std::vector<uint32_t>* candidates;
        size_t k;
        std::priority_queue<std::pair<float, uint32_t>>* result;
    };

    static void* ivf_search_worker(void* arg) {
        IVFThreadData* data = (IVFThreadData*)arg;
        auto& res = *(data->result);

        for (uint32_t idx : *(data->candidates)) {
            float ip = inner_product_neon(data->ivf->base_ + idx * data->ivf->dim_, data->query, data->ivf->dim_);
            float dis = 1.0f - ip;

            if (res.size() < data->k) {
                res.push({dis, idx});
            } else if (dis < res.top().first) {
                res.push({dis, idx});
                res.pop();
            }
        }
        return nullptr;
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        float* query, size_t k, int nprobe, int num_threads)
    {
        std::vector<std::pair<float, int>> centroid_dists;
        for (int c = 0; c < nlist_; ++c) {
            float ip = inner_product_neon(query, centroids_.data() + c * dim_, dim_);
            centroid_dists.emplace_back(1.0f - ip, c);
        }

        std::sort(centroid_dists.begin(), centroid_dists.end());
        std::vector<int> selected_lists;
        for (int i = 0; i < nprobe && i < nlist_; ++i) {
            selected_lists.push_back(centroid_dists[i].second);
        }

        std::vector<std::vector<uint32_t>> list_chunks(num_threads);
        int chunk_idx = 0;
        for (int list_id : selected_lists) {
            for (uint32_t idx : lists_[list_id]) {
                list_chunks[chunk_idx % num_threads].push_back(idx);
                chunk_idx++;
            }
        }

        pthread_t threads[num_threads];
        IVFThreadData thread_data[num_threads];
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            thread_data[t].ivf = this;
            thread_data[t].query = query;
            thread_data[t].candidates = &list_chunks[t];
            thread_data[t].k = k;
            thread_data[t].result = &local_results[t];

            pthread_create(&threads[t], nullptr, ivf_search_worker, &thread_data[t]);
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
};