#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <algorithm>
#include <unordered_set>
#include <arm_neon.h>
#include <omp.h>

struct HNSWNodeOMP {
    std::vector<std::vector<int>> neighbors;
    float* data;

    HNSWNodeOMP(float* d) : data(d) {}
};

class HNSWSearcherOpenMP {
public:
    int M_;
    int efConstruction_;
    int ef_;
    int max_level_;
    size_t n_, dim_;
    float* base_;
    std::vector<HNSWNodeOMP*> nodes_;
    std::vector<int> enter_points_;

    HNSWSearcherOpenMP(int M, int efConstruction, int ef, float* base, size_t n, size_t dim)
        : M_(M), efConstruction_(efConstruction), ef_(ef), base_(base), n_(n), dim_(dim), max_level_(0) {}

    ~HNSWSearcherOpenMP() {
        for (auto node : nodes_) delete node;
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

    void build() {
        nodes_.resize(n_);
        for (size_t i = 0; i < n_; ++i) {
            nodes_[i] = new HNSWNodeOMP(base_ + i * dim_);
        }

        #pragma omp parallel for schedule(dynamic, 128)
        for (size_t i = 0; i < n_; ++i) {
            int level = random_level();

            #pragma omp critical
            {
                if (level > max_level_) {
                    max_level_ = level;
                    enter_points_.resize(max_level_ + 1, 0);
                }
            }

            nodes_[i]->neighbors.resize(level + 1);
            int ep = enter_points_[max_level_];

            for (int l = max_level_; l > level; l--) {
                std::priority_queue<std::pair<float, int>> candidates;
                std::unordered_set<int> visited;

                candidates.push({inner_product_neon(base_ + i * dim_, nodes_[ep]->data, dim_), ep});
                visited.insert(ep);

                while (!candidates.empty()) {
                    auto top = candidates.top();
                    candidates.pop();

                    for (int neighbor : nodes_[top.second]->neighbors[l]) {
                        if (!visited.count(neighbor)) {
                            visited.insert(neighbor);
                            float sim = inner_product_neon(base_ + i * dim_, nodes_[neighbor]->data, dim_);
                            candidates.push({sim, neighbor});
                            if (sim > inner_product_neon(base_ + i * dim_, nodes_[ep]->data, dim_)) {
                                ep = neighbor;
                            }
                        }
                    }
                }
            }

            for (int l = std::min(level, max_level_); l >= 0; l--) {
                std::priority_queue<std::pair<float, int>> candidates;
                std::unordered_set<int> visited;

                candidates.push({inner_product_neon(base_ + i * dim_, nodes_[ep]->data, dim_), ep});
                visited.insert(ep);

                std::vector<std::pair<float, int>> results;
                while (!candidates.empty()) {
                    auto top = candidates.top();
                    candidates.pop();

                    if (!results.empty() && top.first < results[0].first) break;

                    for (int neighbor : nodes_[top.second]->neighbors[l]) {
                        if (!visited.count(neighbor)) {
                            visited.insert(neighbor);
                            float sim = inner_product_neon(base_ + i * dim_, nodes_[neighbor]->data, dim_);
                            candidates.push({sim, neighbor});

                            auto it = results.begin();
                            while (it != results.end() && sim > it->first) ++it;
                            results.insert(it, {sim, neighbor});

                            if (results.size() > efConstruction_) {
                                results.pop_back();
                            }
                        }
                    }
                }

                #pragma omp critical
                {
                    for (auto& r : results) {
                        int other = r.second;
                        if (nodes_[other]->neighbors[l].size() < (size_t)M_) {
                            nodes_[other]->neighbors[l].push_back((int)i);
                        }

                        if (nodes_[i]->neighbors[l].size() < (size_t)M_) {
                            nodes_[i]->neighbors[l].push_back(other);
                        }
                    }
                }

                if (!results.empty()) {
                    ep = results[0].second;
                }
            }

            #pragma omp critical
            {
                enter_points_[level] = i;
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k) {
        int ep = enter_points_[max_level_];

        for (int l = max_level_; l > 0; l--) {
            std::priority_queue<std::pair<float, int>> candidates;
            std::unordered_set<int> visited;

            candidates.push({inner_product_neon(query, nodes_[ep]->data, dim_), ep});
            visited.insert(ep);

            while (!candidates.empty()) {
                auto top = candidates.top();
                candidates.pop();

                for (int neighbor : nodes_[top.second]->neighbors[l]) {
                    if (!visited.count(neighbor)) {
                        visited.insert(neighbor);
                        float sim = inner_product_neon(query, nodes_[neighbor]->data, dim_);
                        candidates.push({sim, neighbor});
                        if (sim > inner_product_neon(query, nodes_[ep]->data, dim_)) {
                            ep = neighbor;
                        }
                    }
                }
            }
        }

        std::priority_queue<std::pair<float, int>> candidates;
        std::unordered_set<int> visited;
        std::priority_queue<std::pair<float, int>> results;

        candidates.push({inner_product_neon(query, nodes_[ep]->data, dim_), ep});
        visited.insert(ep);
        results.push({inner_product_neon(query, nodes_[ep]->data, dim_), ep});

        while (!candidates.empty()) {
            auto top = candidates.top();
            candidates.pop();

            if (!results.empty() && top.first < results.top().first) break;

            for (int neighbor : nodes_[top.second]->neighbors[0]) {
                if (!visited.count(neighbor)) {
                    visited.insert(neighbor);
                    float sim = inner_product_neon(query, nodes_[neighbor]->data, dim_);

                    if (results.size() < k) {
                        results.push({sim, neighbor});
                        candidates.push({sim, neighbor});
                    } else if (sim > results.top().first) {
                        results.pop();
                        results.push({sim, neighbor});
                        candidates.push({sim, neighbor});
                    }
                }
            }
        }

        std::priority_queue<std::pair<float, uint32_t>> res;
        while (!results.empty()) {
            auto top = results.top();
            results.pop();
            res.push({1.0f - top.first, (uint32_t)top.second});
        }
        return res;
    }

private:
    int random_level() {
        float r = (float)rand() / RAND_MAX;
        int l = 0;
        while (r < 0.5 && l < 60) {
            l++;
            r = (float)rand() / RAND_MAX;
        }
        return l;
    }
};