#pragma once
#include <pthread.h>
#include <queue>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <arm_neon.h>
#include <iostream>

struct HNSWNode {
    std::vector<std::vector<int>> neighbors;
    float* data;

    HNSWNode(float* d) : data(d) {}
};

class HNSWSearcherPthread {
public:
    int M_;
    int efConstruction_;
    int ef_;
    int max_level_;
    size_t n_, dim_;
    float* base_;
    std::vector<HNSWNode*> nodes_;
    int enter_point_;

    HNSWSearcherPthread(int M, int efConstruction, int ef, float* base, size_t n, size_t dim)
        : M_(M), efConstruction_(efConstruction), ef_(ef), base_(base), n_(n), dim_(dim), max_level_(-1), enter_point_(-1) {}

    ~HNSWSearcherPthread() {
        for (auto node : nodes_) delete node;
    }

    static float l2_neon(const float* a, const float* b, size_t dim) {
        float32x4_t sum0 = vmovq_n_f32(0.0f);
        float32x4_t sum1 = vmovq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            float32x4_t a0 = vld1q_f32(a + i);
            float32x4_t a1 = vld1q_f32(a + i + 4);
            float32x4_t b0 = vld1q_f32(b + i);
            float32x4_t b1 = vld1q_f32(b + i + 4);
            float32x4_t d0 = vsubq_f32(a0, b0);
            float32x4_t d1 = vsubq_f32(a1, b1);
            sum0 = vfmaq_f32(sum0, d0, d0);
            sum1 = vfmaq_f32(sum1, d1, d1);
        }
        float extra = 0.0f;
        for (; i < dim; i++) {
            float diff = a[i] - b[i];
            extra += diff * diff;
        }
        return vaddvq_f32(vaddq_f32(sum0, sum1)) + extra;
    }

    void build() {
        nodes_.resize(n_);
        for (size_t i = 0; i < n_; ++i) {
            nodes_[i] = new HNSWNode(base_ + i * dim_);
        }

        int level0 = random_level();
        nodes_[0]->neighbors.resize(level0 + 1);
        max_level_ = level0;
        enter_point_ = 0;

        for (size_t i = 1; i < n_; ++i) {
            if (i % 20000 == 0) {
                std::cerr << "  Building progress: " << i << "/" << n_ << "\n";
            }

            int level = random_level();
            nodes_[i]->neighbors.resize(level + 1);

            int ep = enter_point_;
            float best_dist = l2_neon(base_ + i * dim_, nodes_[ep]->data, dim_);

            for (int l = max_level_; l > level; l--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (int neighbor : nodes_[ep]->neighbors[l]) {
                        float dist = l2_neon(base_ + i * dim_, nodes_[neighbor]->data, dim_);
                        if (dist < best_dist) {
                            best_dist = dist;
                            ep = neighbor;
                            changed = true;
                        }
                    }
                }
            }

            for (int l = std::min(level, max_level_); l >= 0; l--) {
                std::priority_queue<std::pair<float, int>> candidates;
                std::priority_queue<std::pair<float, int>> results;
                std::unordered_set<int> visited;

                float d_ep = l2_neon(base_ + i * dim_, nodes_[ep]->data, dim_);
                candidates.push({-d_ep, ep});
                results.push({-d_ep, ep});
                visited.insert(ep);

                while (!candidates.empty()) {
                    auto cur = candidates.top();
                    candidates.pop();
                    float cur_dist = -cur.first;
                    int cur_idx = cur.second;

                    if (results.size() >= (size_t)efConstruction_) {
                        float worst = -results.top().first;
                        if (cur_dist > worst) break;
                    }

                    for (int neighbor : nodes_[cur_idx]->neighbors[l]) {
                        if (!visited.count(neighbor)) {
                            visited.insert(neighbor);
                            float ndist = l2_neon(base_ + i * dim_, nodes_[neighbor]->data, dim_);

                            if (results.size() < (size_t)efConstruction_) {
                                candidates.push({-ndist, neighbor});
                                results.push({-ndist, neighbor});
                            } else {
                                float worst = -results.top().first;
                                if (ndist < worst) {
                                    candidates.push({-ndist, neighbor});
                                    results.pop();
                                    results.push({-ndist, neighbor});
                                }
                            }
                        }
                    }
                }

                int m_max = (l == 0) ? M_ * 2 : M_;

                while (!results.empty() && (int)nodes_[i]->neighbors[l].size() < m_max) {
                    int neighbor = results.top().second;
                    results.pop();

                    if (neighbor == (int)i) continue;

                    nodes_[i]->neighbors[l].push_back(neighbor);
                    if ((int)nodes_[neighbor]->neighbors[l].size() < m_max) {
                        nodes_[neighbor]->neighbors[l].push_back((int)i);
                    }
                }

                if (!nodes_[i]->neighbors[l].empty()) {
                    ep = nodes_[i]->neighbors[l][0];
                }
            }

            if (level > max_level_) {
                max_level_ = level;
                enter_point_ = (int)i;
            }
        }

        std::cerr << "  HNSW build completed, max_level=" << max_level_ << "\n";
    }

    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k) {
        int ep = enter_point_;
        float best_dist = l2_neon(query, nodes_[ep]->data, dim_);

        for (int l = max_level_; l > 0; l--) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int neighbor : nodes_[ep]->neighbors[l]) {
                    float dist = l2_neon(query, nodes_[neighbor]->data, dim_);
                    if (dist < best_dist) {
                        best_dist = dist;
                        ep = neighbor;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<float, int>> candidates;
        std::priority_queue<std::pair<float, int>> results;
        std::unordered_set<int> visited;

        float d_ep = l2_neon(query, nodes_[ep]->data, dim_);
        candidates.push({-d_ep, ep});
        results.push({-d_ep, ep});
        visited.insert(ep);

        int ef_search = std::max(ef_, (int)k);

        while (!candidates.empty()) {
            auto cur = candidates.top();
            candidates.pop();
            float cur_dist = -cur.first;
            int cur_idx = cur.second;

            if (results.size() >= (size_t)ef_search) {
                float worst = -results.top().first;
                if (cur_dist > worst) break;
            }

            for (int neighbor : nodes_[cur_idx]->neighbors[0]) {
                if (!visited.count(neighbor)) {
                    visited.insert(neighbor);
                    float ndist = l2_neon(query, nodes_[neighbor]->data, dim_);

                    if (results.size() < (size_t)ef_search) {
                        candidates.push({-ndist, neighbor});
                        results.push({-ndist, neighbor});
                    } else {
                        float worst = -results.top().first;
                        if (ndist < worst) {
                            candidates.push({-ndist, neighbor});
                            results.pop();
                            results.push({-ndist, neighbor});
                        }
                    }
                }
            }
        }

        std::priority_queue<std::pair<float, uint32_t>> res;
        while (!results.empty() && res.size() < k) {
            auto top = results.top();
            results.pop();
            res.push({-top.first, (uint32_t)top.second});
        }

        return res;
    }

private:
    int random_level() {
        float r = ((float)rand() / (float)RAND_MAX);
        if (r < 1e-6f) r = 1e-6f;
        float ml = 1.0f / std::log(M_ > 1 ? (float)M_ : 2.0f);
        int level = (int)(-std::log(r) * ml);
        if (level > 20) level = 20;
        return level;
    }
};