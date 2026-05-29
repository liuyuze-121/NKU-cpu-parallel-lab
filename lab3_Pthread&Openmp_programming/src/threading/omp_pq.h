#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <cstring>
#include <algorithm>
#include <arm_neon.h>
#include <omp.h>

class PQSearcherOpenMP {
public:
    int M_, ksub_, dsub_;
    size_t n_, dim_;
    std::vector<float> centroids_;
    std::vector<uint8_t> codes_;
    std::vector<float> norms_;
    float* base_;

    PQSearcherOpenMP(int M, int ksub, float* base, size_t n, size_t dim)
        : M_(M), ksub_(ksub), base_(base), n_(n), dim_(dim) {
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

    void train(size_t train_n = 0) {
        if (train_n == 0) train_n = n_;
        centroids_.resize((size_t)M_ * ksub_ * dsub_);

        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M_; ++m) {
            std::vector<float> sub_data(train_n * dsub_);
            for (size_t i = 0; i < train_n; ++i) {
                memcpy(sub_data.data() + i * dsub_, base_ + i * dim_ + (size_t)m * dsub_, dsub_ * sizeof(float));
            }

            float* sc = centroids_.data() + (size_t)m * ksub_ * dsub_;
            for (int k = 0; k < ksub_; ++k) {
                size_t idx = rand() % train_n;
                memcpy(sc + (size_t)k * dsub_, sub_data.data() + idx * dsub_, dsub_ * sizeof(float));
            }

            std::vector<int> assignments(train_n);
            for (int iter = 0; iter < 10; ++iter) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < train_n; ++i) {
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

                for (size_t i = 0; i < train_n; ++i) {
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

        #pragma omp parallel for schedule(dynamic, 256)
        for (size_t i = 0; i < n_; ++i) {
            const float* v = base_ + i * dim_;
            uint8_t* code = codes_.data() + i * M_;

            for (int m = 0; m < M_; ++m) {
                const float* sv = v + (size_t)m * dsub_;
                const float* sc = centroids_.data() + (size_t)m * ksub_ * dsub_;
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

    void build_lut(const float* query, float* lut) {
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M_; m++) {
            const float* sq = query + (size_t)m * dsub_;
            const float* sc = centroids_.data() + (size_t)m * ksub_ * dsub_;
            for (int k = 0; k < ksub_; k++)
                lut[(size_t)m * ksub_ + k] = l2_neon(sq, sc + (size_t)k * dsub_, dsub_);
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k, size_t p = 2000) {
        std::vector<float> lut((size_t)M_ * ksub_);
        build_lut(query, lut.data());

        float q_norm = 0;
        for (size_t d = 0; d < dim_; d++) q_norm += query[d] * query[d];

        int num_threads = omp_get_max_threads();
        std::vector<std::vector<std::pair<float, uint32_t>>> local_cands(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static, 512)
            for (size_t i = 0; i < n_; ++i) {
                const uint8_t* c = codes_.data() + i * M_;
                float l2_dist = 0.0f;
                for (int m = 0; m < M_; m++)
                    l2_dist += lut[(size_t)m * ksub_ + c[m]];
                float ip = (norms_[i] + q_norm - l2_dist) / 2.0f;
                float dist = 1.0f - ip;

                auto& cands = local_cands[tid];
                if (cands.size() < p) {
                    cands.push_back({dist, (uint32_t)i});
                    if (cands.size() == p)
                        std::make_heap(cands.begin(), cands.end());
                } else if (dist < cands.front().first) {
                    std::pop_heap(cands.begin(), cands.end());
                    cands.back() = {dist, (uint32_t)i};
                    std::push_heap(cands.begin(), cands.end());
                }
            }
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