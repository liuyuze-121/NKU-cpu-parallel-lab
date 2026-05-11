#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <algorithm>

class SQSearcher {
public:
    float* base_;
    size_t n_, dim_;
    std::vector<float> means_, scales_;
    std::vector<uint8_t> codes_;

    SQSearcher(float* base, size_t n, size_t dim)
        : base_(base), n_(n), dim_(dim) {}

    void train() {
        means_.resize(dim_);
        scales_.resize(dim_);
        
        for (size_t d = 0; d < dim_; d++) {
            float sum = 0.0f;
            for (size_t i = 0; i < n_; i++) {
                sum += base_[i * dim_ + d];
            }
            means_[d] = sum / n_;
        }

        for (size_t d = 0; d < dim_; d++) {
            float max_abs = 0.0f;
            for (size_t i = 0; i < n_; i++) {
                float val = std::abs(base_[i * dim_ + d] - means_[d]);
                if (val > max_abs) max_abs = val;
            }
            scales_[d] = max_abs / 127.0f;
            if (scales_[d] < 1e-10f) scales_[d] = 1.0f;
        }
    }

    void encode_all() {
        codes_.resize(n_ * dim_);
        for (size_t i = 0; i < n_; i++)
            encode_one(base_ + i * dim_, codes_.data() + i * dim_);
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        float* query, size_t k, size_t p)
    {
        std::vector<uint8_t> q_code(dim_);
        encode_one(query, q_code.data());

        std::vector<std::pair<float, uint32_t>> cands;
        cands.reserve(p);
        for (size_t i = 0; i < n_; i++) {
            float dist = sq_dist(q_code.data(), codes_.data() + i * dim_);
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
        for (auto& c : cands) {
            float ip = 0.0f;
            const float* bv = base_ + c.second * dim_;
            for (size_t d = 0; d < dim_; d++) ip += bv[d] * query[d];
            float dis = 1.0f - ip;
            if (res.size() < k) {
                res.push({dis, c.second});
            } else if (dis < res.top().first) {
                res.push({dis, c.second});
                res.pop();
            }
        }
        return res;
    }

private:
    void encode_one(const float* v, uint8_t* code) const {
        for (size_t d = 0; d < dim_; d++) {
            int val = (int)((v[d] - means_[d]) / scales_[d] + 128.0f);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            code[d] = (uint8_t)val;
        }
    }

    float sq_dist(const uint8_t* a, const uint8_t* b) const {
        float ip = 0.0f;
        for (size_t d = 0; d < dim_; d++) {
            ip += (float)(a[d] - 128) * (float)(b[d] - 128);
        }
        return -ip;
    }
};