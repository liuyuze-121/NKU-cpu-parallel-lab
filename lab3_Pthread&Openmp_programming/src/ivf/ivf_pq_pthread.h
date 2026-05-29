#pragma once
#include <pthread.h>
#include <queue>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <algorithm>
#include <cstring>
#include <arm_neon.h>

class IVFPQSearcherPthread {
public:
    int nlist_;
    int M_, ksub_, dsub_;
    size_t n_, dim_;
    float* base_;

    std::vector<float> coarse_centroids_;
    std::vector<std::vector<uint32_t>> lists_;

    std::vector<float> pq_centroids_;
    std::vector<uint8_t> codes_;
    std::vector<float> norms_;

    IVFPQSearcherPthread(int nlist, int M, int ksub, float* base, size_t n, size_t dim)
        : nlist_(nlist), M_(M), ksub_(ksub), base_(base), n_(n), dim_(dim) {
        dsub_ = (int)(dim / M);
    }

    static float l2_neon(const float* a, const float* b, int d) {
        float32x4_t s0 = vmovq_n_f32(0.0f);
        float32x4_t s1 = vmovq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= d; i += 8) {
            float32x4_t a0 = vld1q_f32(a + i), a1 = vld1q_f32(a + i + 4);
            float32x4_t b0 = vld1q_f32(b + i), b1 = vld1q_f32(b + i + 4);
            float32x4_t d0 = vsubq_f32(a0, b0), d1 = vsubq_f32(a1, b1);
            s0 = vfmaq_f32(s0, d0, d0);
            s1 = vfmaq_f32(s1, d1, d1);
        }
        s0 = vaddq_f32(s0, s1);
        float r = vaddvq_f32(s0);
        for (; i < d; i++) { float df = a[i] - b[i]; r += df * df; }
        return r;
    }

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

    void train_coarse() {
        coarse_centroids_.resize((size_t)nlist_ * dim_);
        for (int c = 0; c < nlist_; ++c) {
            size_t idx = rand() % n_;
            memcpy(coarse_centroids_.data() + c * dim_, base_ + idx * dim_, dim_ * sizeof(float));
        }

        for (int iter = 0; iter < 20; ++iter) {
            std::vector<int> assignments(n_);
            for (size_t i = 0; i < n_; ++i) {
                float best_sim = -FLT_MAX;
                int best_c = 0;
                for (int c = 0; c < nlist_; ++c) {
                    float sim = inner_product_neon(base_ + i * dim_, coarse_centroids_.data() + c * dim_, dim_);
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
                        coarse_centroids_[c * dim_ + d] = new_centroids[c * dim_ + d] / counts[c];
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
                float sim = inner_product_neon(base_ + i * dim_, coarse_centroids_.data() + c * dim_, dim_);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_c = c;
                }
            }
            lists_[best_c].push_back((uint32_t)i);
        }
    }

    void train_pq() {
        pq_centroids_.resize((size_t)M_ * ksub_ * dsub_);

        for (int m = 0; m < M_; ++m) {
            std::vector<float> sub_data(n_ * dsub_);
            for (size_t i = 0; i < n_; ++i) {
                memcpy(sub_data.data() + i * dsub_, base_ + i * dim_ + (size_t)m * dsub_, dsub_ * sizeof(float));
            }

            float* sc = pq_centroids_.data() + (size_t)m * ksub_ * dsub_;
            for (int k = 0; k < ksub_; ++k) {
                size_t idx = rand() % n_;
                memcpy(sc + (size_t)k * dsub_, sub_data.data() + idx * dsub_, dsub_ * sizeof(float));
            }

            std::vector<int> assignments(n_);
            for (int iter = 0; iter < 10; ++iter) {
                for (size_t i = 0; i < n_; ++i) {
                    float best_dist = FLT_MAX;
                    int best_k = 0;
                    for (int k = 0; k < ksub_; ++k) {
                        float dist = l2_neon(sub_data.data() + i * dsub_, sc + (size_t)k * dsub_, dsub_);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }
                    assignments[i] = best_k;
                }

                std::vector<int> counts(ksub_, 0);
                std::vector<float> new_centroids((size_t)ksub_ * dsub_, 0.0f);

                for (size_t i = 0; i < n_; ++i) {
                    int k = assignments[i];
                    counts[k]++;
                    for (int d = 0; d < dsub_; ++d) {
                        new_centroids[(size_t)k * dsub_ + d] += sub_data[i * dsub_ + d];
                    }
                }

                for (int k = 0; k < ksub_; ++k) {
                    if (counts[k] > 0) {
                        for (int d = 0; d < dsub_; ++d) {
                            sc[(size_t)k * dsub_ + d] = new_centroids[(size_t)k * dsub_ + d] / counts[k];
                        }
                    }
                }
            }
        }
    }

    void encode_all() {
        codes_.resize(n_ * M_);
        norms_.resize(n_);

        for (size_t i = 0; i < n_; ++i) {
            const float* v = base_ + i * dim_;
            uint8_t* code = codes_.data() + i * M_;

            for (int m = 0; m < M_; ++m) {
                const float* sv = v + (size_t)m * dsub_;
                const float* sc = pq_centroids_.data() + (size_t)m * ksub_ * dsub_;
                float best_dist = FLT_MAX;
                int best_k = 0;
                for (int k = 0; k < ksub_; ++k) {
                    float dist = l2_neon(sv, sc + (size_t)k * dsub_, dsub_);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }
                code[m] = (uint8_t)best_k;
            }

            float nrm = 0;
            for (size_t d = 0; d < dim_; ++d) {
                nrm += base_[i * dim_ + d] * base_[i * dim_ + d];
            }
            norms_[i] = nrm;
        }
    }

    struct IVFPQThreadData {
        IVFPQSearcherPthread* ivfpq;
        float* lut;
        float q_norm;
        std::vector<uint32_t>* candidates;
        size_t k;
        size_t p;
        std::vector<std::pair<float, uint32_t>>* local_cands;
    };

    static void* ivfpq_search_worker(void* arg) {
        IVFPQThreadData* data = (IVFPQThreadData*)arg;
        auto& cands = *(data->local_cands);

        for (uint32_t idx : *(data->candidates)) {
            const uint8_t* c = data->ivfpq->codes_.data() + idx * data->ivfpq->M_;
            float l2_dist = 0.0f;
            for (int m = 0; m < data->ivfpq->M_; m++) {
                l2_dist += data->lut[(size_t)m * data->ivfpq->ksub_ + c[m]];
            }
            float ip = (data->ivfpq->norms_[idx] + data->q_norm - l2_dist) / 2.0f;
            float dist = 1.0f - ip;

            if (cands.size() < data->p) {
                cands.push_back({dist, idx});
                if (cands.size() == data->p)
                    std::make_heap(cands.begin(), cands.end());
            } else if (dist < cands.front().first) {
                std::pop_heap(cands.begin(), cands.end());
                cands.back() = {dist, idx};
                std::push_heap(cands.begin(), cands.end());
            }
        }
        return nullptr;
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        float* query, size_t k, int nprobe, size_t p = 2000, int num_threads = 4)
    {
        std::vector<float> lut((size_t)M_ * ksub_);

        for (int m = 0; m < M_; m++) {
            const float* sq = query + (size_t)m * dsub_;
            const float* sc = pq_centroids_.data() + (size_t)m * ksub_ * dsub_;
            for (int ki = 0; ki < ksub_; ki++) {
                lut[(size_t)m * ksub_ + ki] = l2_neon(sq, sc + (size_t)ki * dsub_, dsub_);
            }
        }

        float q_norm = 0;
        for (size_t d = 0; d < dim_; d++) q_norm += query[d] * query[d];

        std::vector<std::pair<float, int>> centroid_dists;
        for (int c = 0; c < nlist_; ++c) {
            float ip = inner_product_neon(query, coarse_centroids_.data() + c * dim_, dim_);
            centroid_dists.emplace_back(1.0f - ip, c);
        }

        std::sort(centroid_dists.begin(), centroid_dists.end());

        std::vector<uint32_t> candidates;
        for (int i = 0; i < nprobe && i < nlist_; ++i) {
            int list_id = centroid_dists[i].second;
            candidates.insert(candidates.end(), lists_[list_id].begin(), lists_[list_id].end());
        }

        std::vector<std::vector<uint32_t>> chunks(num_threads);
        for (size_t i = 0; i < candidates.size(); ++i) {
            chunks[i % num_threads].push_back(candidates[i]);
        }

        pthread_t threads[num_threads];
        IVFPQThreadData thread_data[num_threads];
        std::vector<std::vector<std::pair<float, uint32_t>>> local_cands(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            thread_data[t].ivfpq = this;
            thread_data[t].lut = lut.data();
            thread_data[t].q_norm = q_norm;
            thread_data[t].candidates = &chunks[t];
            thread_data[t].k = k;
            thread_data[t].p = p;
            thread_data[t].local_cands = &local_cands[t];

            pthread_create(&threads[t], nullptr, ivfpq_search_worker, &thread_data[t]);
        }

        for (int t = 0; t < num_threads; ++t) {
            pthread_join(threads[t], nullptr);
        }

        std::vector<std::pair<float, uint32_t>> global_cands;
        for (auto& lc : local_cands) {
            global_cands.insert(global_cands.end(), lc.begin(), lc.end());
        }

        std::sort(global_cands.begin(), global_cands.end());
        if (global_cands.size() > p) {
            global_cands.resize(p);
        }

        std::priority_queue<std::pair<float, uint32_t>> res;
        for (auto& cand : global_cands) {
            float ip = 0.0f;
            const float* bv = base_ + cand.second * dim_;
            for (size_t d = 0; d < dim_; d++) ip += bv[d] * query[d];
            float dis = 1.0f - ip;
            if (res.size() < k) {
                res.push({dis, cand.second});
            } else if (dis < res.top().first) {
                res.push({dis, cand.second});
                res.pop();
            }
        }
        return res;
    }
};