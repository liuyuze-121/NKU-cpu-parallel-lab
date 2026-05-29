#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <algorithm>
#include <arm_neon.h>
#include <omp.h>

class IVFFlatSearcherOpenMP {
public:
    int nlist_;
    size_t n_, dim_;
    float* base_;
    std::vector<float> centroids_;
    std::vector<std::vector<uint32_t>> lists_;

    IVFFlatSearcherOpenMP(int nlist, float* base, size_t n, size_t dim)
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

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < nlist_; ++c) {
            size_t idx = rand() % n_;
            memcpy(centroids_.data() + c * dim_, base_ + idx * dim_, dim_ * sizeof(float));
        }

        for (int iter = 0; iter < 20; ++iter) {
            std::vector<int> assignments(n_);

            #pragma omp parallel for schedule(static)
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

            #pragma omp parallel for schedule(static)
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
        #pragma omp parallel
        {
            std::vector<std::vector<uint32_t>> local_lists(nlist_);

            #pragma omp for schedule(static)
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
                local_lists[best_c].push_back((uint32_t)i);
            }

            #pragma omp critical
            {
                for (int c = 0; c < nlist_; ++c) {
                    lists_[c].insert(lists_[c].end(), local_lists[c].begin(), local_lists[c].end());
                }
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k, int nprobe) {
        std::vector<std::pair<float, int>> centroid_dists(nlist_);

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < nlist_; ++c) {
            float ip = inner_product_neon(query, centroids_.data() + c * dim_, dim_);
            centroid_dists[c] = {1.0f - ip, c};
        }

        std::sort(centroid_dists.begin(), centroid_dists.end());

        std::vector<uint32_t> candidates;
        for (int i = 0; i < nprobe && i < nlist_; ++i) {
            int list_id = centroid_dists[i].second;
            candidates.insert(candidates.end(), lists_[list_id].begin(), lists_[list_id].end());
        }

        int num_threads = omp_get_max_threads();
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> locals(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(guided, 256)
            for (size_t t = 0; t < candidates.size(); ++t) {
                uint32_t idx = candidates[t];
                float ip = inner_product_neon(base_ + idx * dim_, query, dim_);
                float dis = 1.0f - ip;

                auto& local = locals[tid];
                if (local.size() < k) {
                    local.push({dis, idx});
                } else if (dis < local.top().first) {
                    local.push({dis, idx});
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
};