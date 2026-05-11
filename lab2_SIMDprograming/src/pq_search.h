#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <cstring>
#include <arm_neon.h>
#include <omp.h>

class PQSearcher {
public:
    int M_, ksub_, dsub_;
    size_t n_, dim_;
    std::vector<float> centroids_;
    std::vector<uint8_t> codes_;
    std::vector<float> norms_;
    float* base_;

    PQSearcher(int M, int ksub, float* base, size_t n, size_t dim)
        : M_(M), ksub_(ksub), base_(base), n_(n), dim_(dim)
    {
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
        #pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < M_; m++)
            train_sub(m, train_n);
    }

    void encode_all() {
        codes_.resize(n_ * M_);
        norms_.resize(n_);
        #pragma omp parallel for schedule(dynamic, 256)
        for (size_t i = 0; i < n_; i++) {
            encode_one(base_ + i * dim_, codes_.data() + i * M_);
            float nrm = 0;
            for (size_t d = 0; d < dim_; d++) nrm += base_[i * dim_ + d] * base_[i * dim_ + d];
            norms_[i] = nrm;
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k, size_t p = 2000) {
        std::vector<float> lut((size_t)M_ * ksub_);
        build_lut(query, lut.data());

        float q_norm = 0;
        for (size_t d = 0; d < dim_; d++) q_norm += query[d] * query[d];

        std::vector<std::pair<float, uint32_t>> cands;
        cands.reserve(p);
        for (size_t i = 0; i < n_; i++) {
            const uint8_t* c = codes_.data() + i * M_;
            float l2_dist = 0.0f;
            for (int m = 0; m < M_; m++)
                l2_dist += lut[(size_t)m * ksub_ + c[m]];
            float ip = (norms_[i] + q_norm - l2_dist) / 2.0f;
            float dist = 1.0f - ip;
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

        std::priority_queue<std::pair<float, uint32_t>> res;
        for (auto& cand : cands) {
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

private:
    void init_kmeanspp(const float* data, float* cen, size_t n) {
        memcpy(cen, data, dsub_ * sizeof(float));
        std::vector<float> md(n, FLT_MAX);
        for (int k = 1; k < ksub_; k++) {
            float total = 0.0f;
            for (size_t i = 0; i < n; i++) {
                float d = l2_neon(data + i * dsub_, cen + (size_t)(k-1) * dsub_, dsub_);
                if (d < md[i]) md[i] = d;
                total += md[i];
            }
            float r = ((float)rand() / RAND_MAX) * total, cs = 0.0f;
            size_t ch = 0;
            for (size_t i = 0; i < n; i++) { cs += md[i]; if (cs >= r) { ch = i; break; } }
            memcpy(cen + (size_t)k * dsub_, data + ch * dsub_, dsub_ * sizeof(float));
        }
    }

    void train_sub(int m, size_t train_n) {
        float* sc = centroids_.data() + (size_t)m * ksub_ * dsub_;
        std::vector<float> sd(train_n * dsub_);
        for (size_t i = 0; i < train_n; i++)
            memcpy(sd.data() + i * dsub_, base_ + i * dim_ + (size_t)m * dsub_, dsub_ * sizeof(float));

        std::vector<float> ct((size_t)ksub_ * dsub_);
        init_kmeanspp(sd.data(), ct.data(), train_n);

        std::vector<int> asgn(train_n);
        for (int it = 0; it < 20; it++) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < train_n; i++) {
                float best = FLT_MAX; int bk = 0;
                for (int k = 0; k < ksub_; k++) {
                    float d = l2_neon(sd.data() + i * dsub_, ct.data() + (size_t)k * dsub_, dsub_);
                    if (d < best) { best = d; bk = k; }
                }
                asgn[i] = bk;
            }
            std::vector<int> cnt(ksub_, 0);
            memset(ct.data(), 0, (size_t)ksub_ * dsub_ * sizeof(float));
            for (size_t i = 0; i < train_n; i++) {
                int c = asgn[i]; cnt[c]++;
                for (int d = 0; d < dsub_; d++)
                    ct[(size_t)c * dsub_ + d] += sd[i * dsub_ + d];
            }
            for (int k = 0; k < ksub_; k++)
                if (cnt[k] > 0)
                    for (int d = 0; d < dsub_; d++)
                        ct[(size_t)k * dsub_ + d] /= cnt[k];
        }
        memcpy(sc, ct.data(), (size_t)ksub_ * dsub_ * sizeof(float));
    }

    void encode_one(const float* v, uint8_t* code) {
        for (int m = 0; m < M_; m++) {
            const float* sv = v + (size_t)m * dsub_;
            const float* sc = centroids_.data() + (size_t)m * ksub_ * dsub_;
            float best = FLT_MAX; int bk = 0;
            for (int k = 0; k < ksub_; k++) {
                float d = l2_neon(sv, sc + (size_t)k * dsub_, dsub_);
                if (d < best) { best = d; bk = k; }
            }
            code[m] = (uint8_t)bk;
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
};