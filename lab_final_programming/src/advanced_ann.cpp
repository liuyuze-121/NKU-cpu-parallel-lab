#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

using Clock = std::chrono::high_resolution_clock;

struct Matrix {
    std::size_t n = 0;
    std::size_t d = 0;
    std::vector<float> x;
};

struct GroundTruth {
    std::size_t n = 0;
    std::size_t k = 0;
    std::vector<std::uint32_t> id;
};

struct Timer {
    Clock::time_point t0;
    Timer() : t0(Clock::now()) {}
    double us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    }
    double s() const {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }
};

static std::string join_path(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    const char c = a.back();
    if (c == '/' || c == '\\') return a + b;
    return a + "/" + b;
}

static Matrix load_fbin(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::int32_t n = 0, d = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    in.read(reinterpret_cast<char *>(&d), sizeof(d));
    if (n <= 0 || d <= 0) throw std::runtime_error("bad fbin header " + path);
    Matrix m;
    m.n = static_cast<std::size_t>(n);
    m.d = static_cast<std::size_t>(d);
    m.x.resize(m.n * m.d);
    in.read(reinterpret_cast<char *>(m.x.data()), static_cast<std::streamsize>(m.x.size() * sizeof(float)));
    if (!in) throw std::runtime_error("short read " + path);
    return m;
}

static GroundTruth load_gt_first_u32_block(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::int32_t n = 0, k = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    in.read(reinterpret_cast<char *>(&k), sizeof(k));
    if (n <= 0 || k <= 0) throw std::runtime_error("bad gt header " + path);
    GroundTruth gt;
    gt.n = static_cast<std::size_t>(n);
    gt.k = static_cast<std::size_t>(k);
    gt.id.resize(gt.n * gt.k);
    in.read(reinterpret_cast<char *>(gt.id.data()), static_cast<std::streamsize>(gt.id.size() * sizeof(std::uint32_t)));
    if (!in) throw std::runtime_error("short read gt id block " + path);
    return gt;
}

static inline float dot_product(const float *a, const float *b, std::size_t d) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#ifdef __FMA__
        acc = _mm256_fmadd_ps(va, vb, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
#endif
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (std::size_t i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
#endif
}

static inline float l2_distance(const float *a, const float *b, std::size_t d) {
    float s = 0.0f;
    for (std::size_t i = 0; i < d; ++i) {
        const float v = a[i] - b[i];
        s += v * v;
    }
    return s;
}

static std::vector<std::uint32_t> make_sample_ids(std::size_t n, std::size_t sample_size, std::uint32_t seed) {
    sample_size = std::min(sample_size, n);
    std::vector<std::uint32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(sample_size);
    return ids;
}

static int nearest_l2(const float *x, const std::vector<float> &centers, int k, int d) {
    int best = 0;
    float best_dist = l2_distance(x, centers.data(), static_cast<std::size_t>(d));
    for (int c = 1; c < k; ++c) {
        const float dist = l2_distance(x, centers.data() + static_cast<std::size_t>(c) * d, static_cast<std::size_t>(d));
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }
    return best;
}

static std::vector<float> train_kmeans_dense(
    const std::vector<float> &data,
    int n,
    int d,
    int k,
    int iters,
    std::uint32_t seed,
    const std::string &tag
) {
    if (n < k) throw std::runtime_error("kmeans n < k for " + tag);
    std::vector<float> centers(static_cast<std::size_t>(k) * d);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    for (int c = 0; c < k; ++c) {
        std::memcpy(centers.data() + static_cast<std::size_t>(c) * d,
                    data.data() + static_cast<std::size_t>(order[c]) * d,
                    static_cast<std::size_t>(d) * sizeof(float));
    }

    std::vector<int> assign(n, 0);
    std::vector<float> sums(static_cast<std::size_t>(k) * d);
    std::vector<int> counts(k);

    for (int it = 0; it < iters; ++it) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            assign[i] = nearest_l2(data.data() + static_cast<std::size_t>(i) * d, centers, k, d);
        }
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (int i = 0; i < n; ++i) {
            const int c = assign[i];
            ++counts[c];
            const float *row = data.data() + static_cast<std::size_t>(i) * d;
            float *sum = sums.data() + static_cast<std::size_t>(c) * d;
            for (int j = 0; j < d; ++j) sum[j] += row[j];
        }
        for (int c = 0; c < k; ++c) {
            float *dst = centers.data() + static_cast<std::size_t>(c) * d;
            if (counts[c] == 0) {
                const int repl = order[(it * k + c) % n];
                std::memcpy(dst, data.data() + static_cast<std::size_t>(repl) * d, static_cast<std::size_t>(d) * sizeof(float));
            } else {
                const float inv = 1.0f / static_cast<float>(counts[c]);
                const float *sum = sums.data() + static_cast<std::size_t>(c) * d;
                for (int j = 0; j < d; ++j) dst[j] = sum[j] * inv;
            }
        }
    }
    return centers;
}

struct AdvancedIndex {
    int nlist = 64;
    int m = 16;
    int ksub = 64;
    int d = 0;
    int dsub = 0;
    bool opq_lite = false;

    std::vector<float> centroids;
    std::vector<int> assignment;
    std::vector<int> perm;
    std::vector<float> codebooks;      // [m][ksub][dsub]
    std::vector<std::uint8_t> codes;   // original id major [nb][m]

    std::vector<std::vector<std::uint32_t>> lists;
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> ids_reordered;
    std::vector<std::uint8_t> codes_reordered;
    std::vector<float> vecs_reordered;

    std::vector<std::vector<int>> centroid_graph;

    double coarse_train_s = 0.0;
    double assign_s = 0.0;
    double opq_s = 0.0;
    double pq_train_s = 0.0;
    double pq_encode_s = 0.0;
    double reorder_s = 0.0;
    double graph_s = 0.0;
    double total_build_s = 0.0;
};

static std::vector<float> gather_sample_rows(const Matrix &base, const std::vector<std::uint32_t> &sample_ids) {
    std::vector<float> sample(sample_ids.size() * base.d);
    for (std::size_t i = 0; i < sample_ids.size(); ++i) {
        std::memcpy(sample.data() + i * base.d, base.x.data() + static_cast<std::size_t>(sample_ids[i]) * base.d, base.d * sizeof(float));
    }
    return sample;
}

static std::vector<int> build_opq_lite_perm(
    const Matrix &base,
    const std::vector<float> &centroids,
    const std::vector<int> &assignment,
    const std::vector<std::uint32_t> &sample_ids,
    int nlist,
    int m
) {
    const int d = static_cast<int>(base.d);
    const int dsub = d / m;
    std::vector<double> mean(d, 0.0), var(d, 0.0);
    for (std::uint32_t id : sample_ids) {
        const int c = assignment[id];
        const float *x = base.x.data() + static_cast<std::size_t>(id) * d;
        const float *cent = centroids.data() + static_cast<std::size_t>(c) * d;
        for (int j = 0; j < d; ++j) mean[j] += static_cast<double>(x[j] - cent[j]);
    }
    for (double &v : mean) v /= static_cast<double>(sample_ids.size());
    for (std::uint32_t id : sample_ids) {
        const int c = assignment[id];
        const float *x = base.x.data() + static_cast<std::size_t>(id) * d;
        const float *cent = centroids.data() + static_cast<std::size_t>(c) * d;
        for (int j = 0; j < d; ++j) {
            const double r = static_cast<double>(x[j] - cent[j]) - mean[j];
            var[j] += r * r;
        }
    }
    std::vector<int> dims(d);
    std::iota(dims.begin(), dims.end(), 0);
    std::sort(dims.begin(), dims.end(), [&](int a, int b) { return var[a] > var[b]; });

    std::vector<int> perm(d, 0);
    std::vector<int> fill(m, 0);
    for (int rank = 0; rank < d; ++rank) {
        const int group = rank % m;
        const int slot = fill[group]++;
        perm[group * dsub + slot] = dims[rank];
    }
    (void)nlist;
    return perm;
}

static void build_centroid_graph(AdvancedIndex &idx, int degree) {
    Timer timer;
    idx.centroid_graph.assign(idx.nlist, {});
    for (int i = 0; i < idx.nlist; ++i) {
        std::vector<std::pair<float, int>> dist;
        dist.reserve(idx.nlist - 1);
        const float *ci = idx.centroids.data() + static_cast<std::size_t>(i) * idx.d;
        for (int j = 0; j < idx.nlist; ++j) {
            if (i == j) continue;
            const float *cj = idx.centroids.data() + static_cast<std::size_t>(j) * idx.d;
            dist.push_back({l2_distance(ci, cj, idx.d), j});
        }
        const int take = std::min<int>(degree, dist.size());
        std::nth_element(dist.begin(), dist.begin() + take, dist.end());
        std::sort(dist.begin(), dist.begin() + take);
        idx.centroid_graph[i].reserve(take);
        for (int t = 0; t < take; ++t) idx.centroid_graph[i].push_back(dist[t].second);
    }
    idx.graph_s = timer.s();
}

static AdvancedIndex build_index(
    const Matrix &base,
    int nlist,
    int m,
    int ksub,
    int kmeans_iters,
    int sample_size,
    bool opq_lite
) {
    Timer all;
    if (base.d % static_cast<std::size_t>(m) != 0) throw std::runtime_error("dimension must be divisible by M");
    AdvancedIndex idx;
    idx.nlist = nlist;
    idx.m = m;
    idx.ksub = ksub;
    idx.d = static_cast<int>(base.d);
    idx.dsub = idx.d / idx.m;
    idx.opq_lite = opq_lite;

    const auto sample_ids = make_sample_ids(base.n, static_cast<std::size_t>(sample_size), opq_lite ? 2027u : 2026u);
    {
        Timer timer;
        std::vector<float> coarse_sample = gather_sample_rows(base, sample_ids);
        idx.centroids = train_kmeans_dense(coarse_sample, static_cast<int>(sample_ids.size()), idx.d, idx.nlist, kmeans_iters, 13u, "coarse");
        idx.coarse_train_s = timer.s();
    }

    {
        Timer timer;
        idx.assignment.resize(base.n);
#pragma omp parallel for schedule(dynamic, 512)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(base.n); ++i) {
            idx.assignment[static_cast<std::size_t>(i)] = nearest_l2(base.x.data() + static_cast<std::size_t>(i) * idx.d, idx.centroids, idx.nlist, idx.d);
        }
        idx.lists.assign(idx.nlist, {});
        for (std::size_t i = 0; i < base.n; ++i) idx.lists[idx.assignment[i]].push_back(static_cast<std::uint32_t>(i));
        idx.assign_s = timer.s();
    }

    {
        Timer timer;
        if (opq_lite) {
            idx.perm = build_opq_lite_perm(base, idx.centroids, idx.assignment, sample_ids, idx.nlist, idx.m);
        } else {
            idx.perm.resize(idx.d);
            std::iota(idx.perm.begin(), idx.perm.end(), 0);
        }
        idx.opq_s = timer.s();
    }

    idx.codebooks.resize(static_cast<std::size_t>(idx.m) * idx.ksub * idx.dsub);
    {
        Timer timer;
        for (int sm = 0; sm < idx.m; ++sm) {
            std::vector<float> train(static_cast<std::size_t>(sample_ids.size()) * idx.dsub);
            for (std::size_t row = 0; row < sample_ids.size(); ++row) {
                const std::uint32_t id = sample_ids[row];
                const int c = idx.assignment[id];
                const float *x = base.x.data() + static_cast<std::size_t>(id) * idx.d;
                const float *cent = idx.centroids.data() + static_cast<std::size_t>(c) * idx.d;
                for (int j = 0; j < idx.dsub; ++j) {
                    const int dim = idx.perm[sm * idx.dsub + j];
                    train[row * idx.dsub + j] = x[dim] - cent[dim];
                }
            }
            std::vector<float> cb = train_kmeans_dense(train, static_cast<int>(sample_ids.size()), idx.dsub, idx.ksub, kmeans_iters, 101u + sm, "pq");
            std::memcpy(idx.codebooks.data() + static_cast<std::size_t>(sm) * idx.ksub * idx.dsub,
                        cb.data(),
                        cb.size() * sizeof(float));
        }
        idx.pq_train_s = timer.s();
    }

    {
        Timer timer;
        idx.codes.resize(base.n * idx.m);
#pragma omp parallel for schedule(dynamic, 256)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(base.n); ++ii) {
            const std::size_t id = static_cast<std::size_t>(ii);
            const int c = idx.assignment[id];
            const float *x = base.x.data() + id * idx.d;
            const float *cent = idx.centroids.data() + static_cast<std::size_t>(c) * idx.d;
            for (int sm = 0; sm < idx.m; ++sm) {
                int best = 0;
                float best_dist = std::numeric_limits<float>::max();
                const float *cb = idx.codebooks.data() + static_cast<std::size_t>(sm) * idx.ksub * idx.dsub;
                for (int code = 0; code < idx.ksub; ++code) {
                    const float *cw = cb + static_cast<std::size_t>(code) * idx.dsub;
                    float dist = 0.0f;
                    for (int j = 0; j < idx.dsub; ++j) {
                        const int dim = idx.perm[sm * idx.dsub + j];
                        const float r = (x[dim] - cent[dim]) - cw[j];
                        dist += r * r;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best = code;
                    }
                }
                idx.codes[id * idx.m + sm] = static_cast<std::uint8_t>(best);
            }
        }
        idx.pq_encode_s = timer.s();
    }

    {
        Timer timer;
        idx.offsets.assign(idx.nlist + 1, 0);
        for (int l = 0; l < idx.nlist; ++l) idx.offsets[l + 1] = idx.offsets[l] + static_cast<std::uint32_t>(idx.lists[l].size());
        idx.ids_reordered.resize(base.n);
        idx.codes_reordered.resize(base.n * idx.m);
        idx.vecs_reordered.resize(base.n * idx.d);
        for (int l = 0; l < idx.nlist; ++l) {
            std::uint32_t pos = idx.offsets[l];
            for (std::uint32_t id : idx.lists[l]) {
                idx.ids_reordered[pos] = id;
                std::memcpy(idx.codes_reordered.data() + static_cast<std::size_t>(pos) * idx.m,
                            idx.codes.data() + static_cast<std::size_t>(id) * idx.m,
                            static_cast<std::size_t>(idx.m));
                std::memcpy(idx.vecs_reordered.data() + static_cast<std::size_t>(pos) * idx.d,
                            base.x.data() + static_cast<std::size_t>(id) * idx.d,
                            static_cast<std::size_t>(idx.d) * sizeof(float));
                ++pos;
            }
        }
        idx.reorder_s = timer.s();
    }

    build_centroid_graph(idx, 12);
    idx.total_build_s = all.s();
    return idx;
}

struct Config {
    std::string name;
    bool opq_lite = false;
    bool hnsw_route = false;
    bool reordered = false;
    bool fused = false;
    bool parallel_queries = false;
    int nprobe = 16;
    int ef = 48;
    int rerank = 500;
};

struct QueryStat {
    double route_us = 0.0;
    double lut_us = 0.0;
    double scan_us = 0.0;
    double top_r_us = 0.0;
    double rerank_us = 0.0;
    std::size_t candidates = 0;
    std::size_t centroids_scored = 0;
    double recall = 0.0;
};

struct RunSummary {
    Config cfg;
    double recall = 0.0;
    double latency_us = 0.0;
    double qps = 0.0;
    double wall_s = 0.0;
    double route_us = 0.0;
    double lut_us = 0.0;
    double scan_us = 0.0;
    double top_r_us = 0.0;
    double rerank_us = 0.0;
    double avg_candidates = 0.0;
    double avg_centroids_scored = 0.0;
};

static std::vector<int> exact_centroid_route(const AdvancedIndex &idx, const float *q, int nprobe, std::size_t &scored) {
    scored = static_cast<std::size_t>(idx.nlist);
    std::vector<std::pair<float, int>> dist;
    dist.reserve(idx.nlist);
    for (int c = 0; c < idx.nlist; ++c) {
        const float *cent = idx.centroids.data() + static_cast<std::size_t>(c) * idx.d;
        dist.push_back({l2_distance(q, cent, idx.d), c});
    }
    const int take = std::min(nprobe, idx.nlist);
    std::nth_element(dist.begin(), dist.begin() + take, dist.end());
    std::sort(dist.begin(), dist.begin() + take);
    std::vector<int> out;
    out.reserve(take);
    for (int i = 0; i < take; ++i) out.push_back(dist[i].second);
    return out;
}

static std::vector<int> hnsw_centroid_route(const AdvancedIndex &idx, const float *q, int nprobe, int ef, std::size_t &scored) {
    struct Node {
        float dist;
        int id;
        bool operator<(const Node &o) const { return dist > o.dist; }
    };
    std::priority_queue<Node> cand;
    std::vector<char> seen(idx.nlist, 0);
    std::vector<Node> visited;
    const int entries[] = {0, idx.nlist / 4, idx.nlist / 2, (idx.nlist * 3) / 4};
    scored = 0;
    for (int e : entries) {
        if (e < 0 || e >= idx.nlist || seen[e]) continue;
        seen[e] = 1;
        const float *cent = idx.centroids.data() + static_cast<std::size_t>(e) * idx.d;
        cand.push({l2_distance(q, cent, idx.d), e});
        ++scored;
    }
    while (!cand.empty() && static_cast<int>(visited.size()) < ef) {
        Node cur = cand.top();
        cand.pop();
        visited.push_back(cur);
        for (int nb : idx.centroid_graph[cur.id]) {
            if (seen[nb]) continue;
            seen[nb] = 1;
            const float *cent = idx.centroids.data() + static_cast<std::size_t>(nb) * idx.d;
            cand.push({l2_distance(q, cent, idx.d), nb});
            ++scored;
        }
    }
    if (static_cast<int>(visited.size()) < nprobe) {
        for (int c = 0; c < idx.nlist; ++c) {
            if (seen[c]) continue;
            const float *cent = idx.centroids.data() + static_cast<std::size_t>(c) * idx.d;
            visited.push_back({l2_distance(q, cent, idx.d), c});
            ++scored;
            if (static_cast<int>(visited.size()) >= nprobe) break;
        }
    }
    std::sort(visited.begin(), visited.end(), [](const Node &a, const Node &b) { return a.dist < b.dist; });
    const int take = std::min<int>(nprobe, visited.size());
    std::vector<int> out;
    out.reserve(take);
    for (int i = 0; i < take; ++i) out.push_back(visited[i].id);
    return out;
}

static void build_lut_for_list(const AdvancedIndex &idx, const float *q, int list_id, std::vector<float> &lut) {
    lut.assign(static_cast<std::size_t>(idx.m) * idx.ksub, 0.0f);
    for (int sm = 0; sm < idx.m; ++sm) {
        const float *cb = idx.codebooks.data() + static_cast<std::size_t>(sm) * idx.ksub * idx.dsub;
        for (int code = 0; code < idx.ksub; ++code) {
            const float *cw = cb + static_cast<std::size_t>(code) * idx.dsub;
            float s = 0.0f;
            for (int j = 0; j < idx.dsub; ++j) {
                const int dim = idx.perm[sm * idx.dsub + j];
                s += q[dim] * cw[j];
            }
            lut[static_cast<std::size_t>(sm) * idx.ksub + code] = s;
        }
    }
    const float cent_score = dot_product(q, idx.centroids.data() + static_cast<std::size_t>(list_id) * idx.d, idx.d);
    for (float &v : lut) v += cent_score / static_cast<float>(idx.m);
}

struct ApproxCand {
    float score;
    std::uint32_t id;
    std::uint32_t pos;
};

struct MinScore {
    bool operator()(const ApproxCand &a, const ApproxCand &b) const {
        return a.score > b.score;
    }
};

static inline float pq_score_from_codes(const AdvancedIndex &idx, const std::vector<float> &lut, const std::uint8_t *code_ptr) {
    float s = 0.0f;
    for (int sm = 0; sm < idx.m; ++sm) {
        s += lut[static_cast<std::size_t>(sm) * idx.ksub + code_ptr[sm]];
    }
    return s;
}

static std::vector<std::uint32_t> search_one(
    const Matrix &base,
    const AdvancedIndex &idx,
    const Config &cfg,
    const float *q,
    int k,
    QueryStat &stat
) {
    Timer total_timer;

    Timer t_route;
    std::size_t scored = 0;
    std::vector<int> probes = cfg.hnsw_route
        ? hnsw_centroid_route(idx, q, cfg.nprobe, cfg.ef, scored)
        : exact_centroid_route(idx, q, cfg.nprobe, scored);
    stat.route_us = t_route.us();
    stat.centroids_scored = scored;

    std::vector<ApproxCand> top_r_vec;
    top_r_vec.reserve(static_cast<std::size_t>(cfg.rerank));
    std::vector<float> lut;

    if (cfg.fused) {
        Timer t_scan;
        std::priority_queue<ApproxCand, std::vector<ApproxCand>, MinScore> heap;
        for (int list_id : probes) {
            Timer t_lut;
            build_lut_for_list(idx, q, list_id, lut);
            stat.lut_us += t_lut.us();
            const std::uint32_t begin = idx.offsets[list_id];
            const std::uint32_t end = idx.offsets[list_id + 1];
            stat.candidates += static_cast<std::size_t>(end - begin);
            for (std::uint32_t pos = begin; pos < end; ++pos) {
                const std::uint8_t *code_ptr = idx.codes_reordered.data() + static_cast<std::size_t>(pos) * idx.m;
                const float score = pq_score_from_codes(idx, lut, code_ptr);
                ApproxCand cand{score, idx.ids_reordered[pos], pos};
                if (static_cast<int>(heap.size()) < cfg.rerank) {
                    heap.push(cand);
                } else if (score > heap.top().score) {
                    heap.pop();
                    heap.push(cand);
                }
            }
        }
        stat.scan_us = t_scan.us() - stat.lut_us;
        while (!heap.empty()) {
            top_r_vec.push_back(heap.top());
            heap.pop();
        }
    } else {
        std::vector<ApproxCand> cands;
        Timer t_scan;
        for (int list_id : probes) {
            Timer t_lut;
            build_lut_for_list(idx, q, list_id, lut);
            stat.lut_us += t_lut.us();
            if (cfg.reordered) {
                const std::uint32_t begin = idx.offsets[list_id];
                const std::uint32_t end = idx.offsets[list_id + 1];
                stat.candidates += static_cast<std::size_t>(end - begin);
                for (std::uint32_t pos = begin; pos < end; ++pos) {
                    const std::uint8_t *code_ptr = idx.codes_reordered.data() + static_cast<std::size_t>(pos) * idx.m;
                    cands.push_back({pq_score_from_codes(idx, lut, code_ptr), idx.ids_reordered[pos], pos});
                }
            } else {
                const auto &lst = idx.lists[list_id];
                stat.candidates += lst.size();
                for (std::uint32_t id : lst) {
                    const std::uint8_t *code_ptr = idx.codes.data() + static_cast<std::size_t>(id) * idx.m;
                    cands.push_back({pq_score_from_codes(idx, lut, code_ptr), id, id});
                }
            }
        }
        stat.scan_us = t_scan.us() - stat.lut_us;

        Timer t_topr;
        const int take = std::min<int>(cfg.rerank, cands.size());
        if (take > 0) {
            std::nth_element(cands.begin(), cands.begin() + take, cands.end(),
                             [](const ApproxCand &a, const ApproxCand &b) { return a.score > b.score; });
            cands.resize(take);
            top_r_vec.swap(cands);
        }
        stat.top_r_us = t_topr.us();
    }

    Timer t_rerank;
    struct ExactCand {
        float score;
        std::uint32_t id;
    };
    struct ExactMin {
        bool operator()(const ExactCand &a, const ExactCand &b) const { return a.score > b.score; }
    };
    std::priority_queue<ExactCand, std::vector<ExactCand>, ExactMin> exact_heap;
    for (const ApproxCand &cand : top_r_vec) {
        const float *vec = nullptr;
        if (cfg.reordered && cand.pos < idx.ids_reordered.size() && idx.ids_reordered[cand.pos] == cand.id) {
            vec = idx.vecs_reordered.data() + static_cast<std::size_t>(cand.pos) * idx.d;
        } else {
            vec = base.x.data() + static_cast<std::size_t>(cand.id) * idx.d;
        }
        const float score = dot_product(q, vec, idx.d);
        if (static_cast<int>(exact_heap.size()) < k) {
            exact_heap.push({score, cand.id});
        } else if (score > exact_heap.top().score) {
            exact_heap.pop();
            exact_heap.push({score, cand.id});
        }
    }
    std::vector<ExactCand> out;
    while (!exact_heap.empty()) {
        out.push_back(exact_heap.top());
        exact_heap.pop();
    }
    std::sort(out.begin(), out.end(), [](const ExactCand &a, const ExactCand &b) { return a.score > b.score; });
    std::vector<std::uint32_t> ids;
    ids.reserve(out.size());
    for (const auto &v : out) ids.push_back(v.id);
    stat.rerank_us = t_rerank.us();
    (void)total_timer;
    return ids;
}

static double recall_at_k(const std::vector<std::uint32_t> &pred, const GroundTruth &gt, std::size_t qi, int k) {
    std::unordered_set<std::uint32_t> truth;
    truth.reserve(static_cast<std::size_t>(k) * 2);
    for (int i = 0; i < k; ++i) truth.insert(gt.id[qi * gt.k + i]);
    int hit = 0;
    for (std::uint32_t id : pred) {
        if (truth.find(id) != truth.end()) ++hit;
    }
    return static_cast<double>(hit) / static_cast<double>(k);
}

static RunSummary run_config(
    const Matrix &base,
    const Matrix &queries,
    const GroundTruth &gt,
    const AdvancedIndex &idx,
    const Config &cfg,
    int query_count,
    int k
) {
    query_count = std::min<int>(query_count, static_cast<int>(queries.n));
    std::vector<QueryStat> stats(query_count);
    Timer wall;
    if (cfg.parallel_queries) {
#pragma omp parallel for schedule(dynamic, 8)
        for (int qi = 0; qi < query_count; ++qi) {
            QueryStat st;
            const float *q = queries.x.data() + static_cast<std::size_t>(qi) * queries.d;
            std::vector<std::uint32_t> pred = search_one(base, idx, cfg, q, k, st);
            st.recall = recall_at_k(pred, gt, static_cast<std::size_t>(qi), k);
            stats[qi] = st;
        }
    } else {
        for (int qi = 0; qi < query_count; ++qi) {
            QueryStat st;
            const float *q = queries.x.data() + static_cast<std::size_t>(qi) * queries.d;
            std::vector<std::uint32_t> pred = search_one(base, idx, cfg, q, k, st);
            st.recall = recall_at_k(pred, gt, static_cast<std::size_t>(qi), k);
            stats[qi] = st;
        }
    }
    const double wall_s = wall.s();
    RunSummary rs;
    rs.cfg = cfg;
    rs.wall_s = wall_s;
    for (const auto &st : stats) {
        rs.recall += st.recall;
        rs.route_us += st.route_us;
        rs.lut_us += st.lut_us;
        rs.scan_us += st.scan_us;
        rs.top_r_us += st.top_r_us;
        rs.rerank_us += st.rerank_us;
        rs.avg_candidates += static_cast<double>(st.candidates);
        rs.avg_centroids_scored += static_cast<double>(st.centroids_scored);
    }
    const double inv = 1.0 / static_cast<double>(query_count);
    rs.recall *= inv;
    rs.route_us *= inv;
    rs.lut_us *= inv;
    rs.scan_us *= inv;
    rs.top_r_us *= inv;
    rs.rerank_us *= inv;
    rs.avg_candidates *= inv;
    rs.avg_centroids_scored *= inv;
    rs.latency_us = (rs.route_us + rs.lut_us + rs.scan_us + rs.top_r_us + rs.rerank_us);
    rs.qps = static_cast<double>(query_count) / wall_s;
    if (cfg.parallel_queries) {
        rs.latency_us = wall_s * 1e6 / static_cast<double>(query_count);
    }
    return rs;
}

static std::string csv_bool(bool v) { return v ? "1" : "0"; }

static void write_build_csv(const std::string &path, const AdvancedIndex &plain, const AdvancedIndex &opq) {
    std::ofstream out(path);
    out << "index,opq_lite,nlist,M,ksub,coarse_train_s,assign_s,opq_s,pq_train_s,pq_encode_s,reorder_s,graph_s,total_build_s\n";
    auto row = [&](const std::string &name, const AdvancedIndex &idx) {
        out << name << "," << csv_bool(idx.opq_lite) << "," << idx.nlist << "," << idx.m << "," << idx.ksub << ","
            << idx.coarse_train_s << "," << idx.assign_s << "," << idx.opq_s << "," << idx.pq_train_s << ","
            << idx.pq_encode_s << "," << idx.reorder_s << "," << idx.graph_s << "," << idx.total_build_s << "\n";
    };
    row("Residual-IVF-PQ", plain);
    row("OPQ-lite-IVF-PQ", opq);
}

static void write_results_csv(const std::string &path, const std::vector<RunSummary> &rows, int query_count, int topk) {
    std::ofstream out(path);
    out << "method,query_count,topk,opq_lite,hnsw_route,reordered,fused,parallel_queries,nprobe,ef,rerank,recall_at_10,latency_us,qps,wall_s,avg_candidates,avg_centroids_scored\n";
    out << std::fixed << std::setprecision(6);
    for (const auto &r : rows) {
        const auto &c = r.cfg;
        out << c.name << "," << query_count << "," << topk << "," << csv_bool(c.opq_lite) << "," << csv_bool(c.hnsw_route) << ","
            << csv_bool(c.reordered) << "," << csv_bool(c.fused) << "," << csv_bool(c.parallel_queries) << ","
            << c.nprobe << "," << c.ef << "," << c.rerank << "," << r.recall << "," << r.latency_us << ","
            << r.qps << "," << r.wall_s << "," << r.avg_candidates << "," << r.avg_centroids_scored << "\n";
    }
}

static void write_profile_csv(const std::string &path, const std::vector<RunSummary> &rows) {
    std::ofstream out(path);
    out << "method,route_us,lut_us,pq_scan_or_fused_topr_us,unfused_topr_us,exact_rerank_topk_us\n";
    out << std::fixed << std::setprecision(6);
    for (const auto &r : rows) {
        out << r.cfg.name << "," << r.route_us << "," << r.lut_us << "," << r.scan_us << ","
            << r.top_r_us << "," << r.rerank_us << "\n";
    }
}

static std::string arg_value(int argc, char **argv, const std::string &key, const std::string &def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) return argv[i + 1];
    }
    return def;
}

static int arg_int(int argc, char **argv, const std::string &key, int def) {
    return std::stoi(arg_value(argc, argv, key, std::to_string(def)));
}

int main(int argc, char **argv) {
    try {
        const std::string data_dir = arg_value(argc, argv, "--data_dir", "C:/Users/Administrator/Desktop/ann_final_advanced/data");
        const std::string out_dir = arg_value(argc, argv, "--out_dir", "C:/Users/Administrator/Desktop/ann_final_advanced/results");
        const int query_count = arg_int(argc, argv, "--queries", 500);
        const int nlist = arg_int(argc, argv, "--nlist", 64);
        const int m = arg_int(argc, argv, "--M", 16);
        const int ksub = arg_int(argc, argv, "--ksub", 64);
        const int sample = arg_int(argc, argv, "--sample", 12000);
        const int iters = arg_int(argc, argv, "--iters", 5);
        const int topk = arg_int(argc, argv, "--topk", 10);

        std::cerr << "Loading DEEP100K from " << data_dir << "\n";
        Matrix base = load_fbin(join_path(data_dir, "DEEP100K.base.100k.fbin"));
        Matrix queries = load_fbin(join_path(data_dir, "DEEP100K.query.fbin"));
        GroundTruth gt = load_gt_first_u32_block(join_path(data_dir, "DEEP100K.gt.query.100k.top100.bin"));
        if (base.d != queries.d) throw std::runtime_error("base/query dimension mismatch");
        if (gt.n < queries.n || gt.k < static_cast<std::size_t>(topk)) throw std::runtime_error("ground truth shape mismatch");

#ifdef _OPENMP
        std::cerr << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#endif
#ifdef __AVX2__
        std::cerr << "AVX2 dot product enabled\n";
#else
        std::cerr << "AVX2 dot product not enabled\n";
#endif
        std::cerr << "base=" << base.n << "x" << base.d << ", query_count=" << query_count
                  << ", nlist=" << nlist << ", M=" << m << ", ksub=" << ksub << "\n";

        std::cerr << "Building residual IVF-PQ index...\n";
        AdvancedIndex plain = build_index(base, nlist, m, ksub, iters, sample, false);
        std::cerr << "Building OPQ-lite IVF-PQ index...\n";
        AdvancedIndex opq = build_index(base, nlist, m, ksub, iters, sample, true);

        std::vector<Config> configs = {
            {"A-Residual-IVF-PQ-Rerank-np8-R200", false, false, false, false, false, 8, 32, 200},
            {"A-Residual-IVF-PQ-Rerank-np16-R500", false, false, false, false, false, 16, 48, 500},
            {"A-Residual-IVF-PQ-Rerank-np24-R800", false, false, false, false, false, 24, 64, 800},
            {"A-Residual-IVF-PQ-Rerank-np32-R1000", false, false, false, false, false, 32, 64, 1000},

            {"B-OPQ-IVF-PQ-Rerank-np8-R200", true, false, false, false, false, 8, 32, 200},
            {"B-OPQ-IVF-PQ-Rerank-np16-R500", true, false, false, false, false, 16, 48, 500},
            {"B-OPQ-IVF-PQ-Rerank-np24-R800", true, false, false, false, false, 24, 64, 800},
            {"B-OPQ-IVF-PQ-Rerank-np32-R1000", true, false, false, false, false, 32, 64, 1000},

            {"C-IVF-HNSW-OPQ-PQ-Rerank-ef16", true, true, false, false, false, 16, 16, 500},
            {"C-IVF-HNSW-OPQ-PQ-Rerank-ef32", true, true, false, false, false, 16, 32, 500},
            {"C-IVF-HNSW-OPQ-PQ-Rerank-ef48", true, true, false, false, false, 16, 48, 500},
            {"C-IVF-HNSW-OPQ-PQ-Rerank-ef64", true, true, false, false, false, 16, 64, 500},

            {"D-Reordered-IVF-HNSW-OPQ-PQ-Rerank-np8-R200", true, true, true, false, false, 8, 32, 200},
            {"D-Reordered-IVF-HNSW-OPQ-PQ-Rerank-np16-R500", true, true, true, false, false, 16, 48, 500},
            {"D-Reordered-IVF-HNSW-OPQ-PQ-Rerank-np24-R800", true, true, true, false, false, 24, 64, 800},
            {"D-Fused-Reordered-IVF-HNSW-OPQ-PQ-Rerank-np8-R200", true, true, true, true, false, 8, 32, 200},
            {"D-Fused-Reordered-IVF-HNSW-OPQ-PQ-Rerank-np16-R500", true, true, true, true, false, 16, 48, 500},
            {"D-Fused-Reordered-IVF-HNSW-OPQ-PQ-Rerank-np24-R1000", true, true, true, true, false, 24, 64, 1000},

            {"E-Pipeline-OMP-SIMD-Fused-Reordered-np8-R200", true, true, true, true, true, 8, 32, 200},
            {"E-Pipeline-OMP-SIMD-Fused-Reordered-np16-R500", true, true, true, true, true, 16, 48, 500},
            {"E-Pipeline-OMP-SIMD-Fused-Reordered-np24-R1000", true, true, true, true, true, 24, 64, 1000}
        };

        std::vector<RunSummary> rows;
        for (const Config &cfg : configs) {
            const AdvancedIndex &idx = cfg.opq_lite ? opq : plain;
            std::cerr << "Running " << cfg.name << "...\n";
            rows.push_back(run_config(base, queries, gt, idx, cfg, query_count, topk));
            const RunSummary &r = rows.back();
            std::cerr << "  recall@10=" << r.recall << ", latency_us=" << r.latency_us
                      << ", qps=" << r.qps << ", candidates=" << r.avg_candidates << "\n";
        }

        write_build_csv(join_path(out_dir, "advanced_build_profile.csv"), plain, opq);
        write_results_csv(join_path(out_dir, "advanced_results.csv"), rows, query_count, topk);
        write_profile_csv(join_path(out_dir, "advanced_query_profile.csv"), rows);

        std::ofstream meta(join_path(out_dir, "run_config.txt"));
        meta << "data_dir=" << data_dir << "\n";
        meta << "query_count=" << query_count << "\n";
        meta << "topk=" << topk << "\n";
        meta << "nlist=" << nlist << "\n";
        meta << "M=" << m << "\n";
        meta << "ksub=" << ksub << "\n";
        meta << "sample=" << sample << "\n";
        meta << "kmeans_iters=" << iters << "\n";
#ifdef _OPENMP
        meta << "openmp_threads=" << omp_get_max_threads() << "\n";
#else
        meta << "openmp_threads=0\n";
#endif
#ifdef __AVX2__
        meta << "avx2=1\n";
#else
        meta << "avx2=0\n";
#endif
        std::cerr << "Done. Results written to " << out_dir << "\n";
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
