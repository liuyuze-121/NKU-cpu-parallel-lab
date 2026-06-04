#pragma once
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "../flat_scan.h"
#include "../omp/omp_flat.h"
#include "../pthread/pthread_flat.h"
#include "../omp/ivf_flat_omp.h"
#include "../pthread/ivf_flat_pthread.h"
#include "../omp/hnsw_omp.h"
#include "../pthread/hnsw_pthread.h"

struct MPIRunConfig {
    std::string algorithm = "ivf";
    std::string comm_mode = "blocking";
    std::string thread_mode = "none";
    std::string graph_mode = "shard_hnsw";
    int threads = 1;
    int mpi_queries = 64;
    int k = 10;
    int ivf_nlist = 128;
    int ivf_nprobe = 16;
    int hnsw_m = 16;
    int hnsw_ef_construction = 100;
    int hnsw_ef = 32;
    int vecdim = 96;
};

struct MPISearchResult {
    float avg_recall = 0.0f;
    double avg_latency_us = 0.0;
    double max_latency_us = 0.0;
    size_t query_count = 0;
};

struct MPIEnv {
    int world_size = 1;
    int world_rank = 0;
    bool initialized = false;
};

static inline std::priority_queue<std::pair<float, uint32_t>> mpi_merge_topk(
    const std::vector<std::pair<float, uint32_t>>& candidates,
    size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (q.size() < k) {
            q.push(candidates[i]);
        } else if (candidates[i].first < q.top().first) {
            q.push(candidates[i]);
            q.pop();
        }
    }
    return q;
}

static inline std::vector<std::pair<float, uint32_t>> mpi_priority_queue_to_sorted_vector(
    std::priority_queue<std::pair<float, uint32_t>> q)
{
    std::vector<std::pair<float, uint32_t>> out;
    while (!q.empty()) {
        out.push_back(q.top());
        q.pop();
    }
    std::sort(out.begin(), out.end(), [](const std::pair<float, uint32_t>& lhs, const std::pair<float, uint32_t>& rhs) {
        if (lhs.first != rhs.first) return lhs.first < rhs.first;
        return lhs.second < rhs.second;
    });
    return out;
}

static inline float mpi_compute_recall_from_sorted(
    const std::vector<std::pair<float, uint32_t>>& res,
    int* gt,
    size_t gt_dim,
    size_t query_idx,
    size_t k)
{
    std::set<uint32_t> gtset;
    for (size_t j = 0; j < k; ++j) {
        gtset.insert((uint32_t)gt[query_idx * gt_dim + j]);
    }
    size_t hit = 0;
    for (size_t i = 0; i < res.size(); ++i) {
        if (gtset.find(res[i].second) != gtset.end()) {
            ++hit;
        }
    }
    return k == 0 ? 0.0f : (float)hit / (float)k;
}

static inline MPIEnv mpi_setup(int* argc, char*** argv) {
    MPIEnv env;
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
        int required = MPI_THREAD_MULTIPLE;
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(argc, argv, required, &provided);
        env.initialized = true;
    }
    MPI_Comm_size(MPI_COMM_WORLD, &env.world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &env.world_rank);
    return env;
}

static inline void mpi_shutdown(const MPIEnv& env) {
    if (env.initialized) {
        MPI_Finalize();
    }
}

static inline void mpi_build_shard(
    const MPIEnv& env,
    float* base,
    size_t base_number,
    size_t vecdim,
    std::vector<float>& local_base,
    size_t& local_number,
    size_t& offset)
{
    size_t chunk = base_number / (size_t)env.world_size;
    size_t rem = base_number % (size_t)env.world_size;
    offset = chunk * (size_t)env.world_rank + std::min((size_t)env.world_rank, rem);
    local_number = chunk + ((size_t)env.world_rank < rem ? 1 : 0);
    local_base.resize(local_number * vecdim);
    if (local_number > 0) {
        memcpy(local_base.data(), base + offset * vecdim, local_number * vecdim * sizeof(float));
    }
}

static inline void mpi_shift_indices(std::vector<std::pair<float, uint32_t>>& res, size_t offset) {
    for (size_t i = 0; i < res.size(); ++i) {
        res[i].second = (uint32_t)(res[i].second + offset);
    }
}

static inline std::vector<std::pair<float, uint32_t>> mpi_local_search_flat(
    const MPIRunConfig& cfg,
    const std::vector<float>& local_base,
    size_t local_number,
    size_t vecdim,
    float* query)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    if (cfg.thread_mode == "openmp") {
        omp_set_num_threads(cfg.threads);
        q = omp_flat_search(const_cast<float*>(local_base.data()), query, local_number, vecdim, cfg.k);
    } else if (cfg.thread_mode == "pthread") {
        q = pthread_flat_search(const_cast<float*>(local_base.data()), query, local_number, vecdim, cfg.k, cfg.threads);
    } else {
        q = flat_search(const_cast<float*>(local_base.data()), query, local_number, vecdim, cfg.k);
    }
    return mpi_priority_queue_to_sorted_vector(q);
}

static inline std::vector<std::pair<float, uint32_t>> mpi_gather_blocking(
    const MPIEnv& env,
    const std::vector<std::pair<float, uint32_t>>& local_res,
    int k)
{
    std::vector<float> send_dis(k, FLT_MAX);
    std::vector<uint32_t> send_ids(k, UINT32_MAX);
    for (size_t i = 0; i < local_res.size() && i < (size_t)k; ++i) {
        send_dis[i] = local_res[i].first;
        send_ids[i] = local_res[i].second;
    }
    std::vector<float> recv_dis;
    std::vector<uint32_t> recv_ids;
    if (env.world_rank == 0) {
        recv_dis.resize((size_t)env.world_size * k);
        recv_ids.resize((size_t)env.world_size * k);
    }
    MPI_Gather(send_dis.data(), k, MPI_FLOAT, recv_dis.data(), k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(send_ids.data(), k, MPI_UNSIGNED, recv_ids.data(), k, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::vector<std::pair<float, uint32_t>> merged;
    if (env.world_rank == 0) {
        for (size_t i = 0; i < recv_dis.size(); ++i) {
            if (recv_ids[i] != UINT32_MAX) {
                merged.push_back(std::make_pair(recv_dis[i], recv_ids[i]));
            }
        }
    }
    return merged;
}

static inline std::vector<std::pair<float, uint32_t>> mpi_gather_nonblocking(
    const MPIEnv& env,
    const std::vector<std::pair<float, uint32_t>>& local_res,
    int k)
{
    std::vector<float> send_dis(k, FLT_MAX);
    std::vector<uint32_t> send_ids(k, UINT32_MAX);
    for (size_t i = 0; i < local_res.size() && i < (size_t)k; ++i) {
        send_dis[i] = local_res[i].first;
        send_ids[i] = local_res[i].second;
    }
    std::vector<float> recv_dis;
    std::vector<uint32_t> recv_ids;
    if (env.world_rank == 0) {
        recv_dis.resize((size_t)env.world_size * k);
        recv_ids.resize((size_t)env.world_size * k);
    }
    MPI_Request reqs[2];
    MPI_Igather(send_dis.data(), k, MPI_FLOAT, recv_dis.data(), k, MPI_FLOAT, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Igather(send_ids.data(), k, MPI_UNSIGNED, recv_ids.data(), k, MPI_UNSIGNED, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    std::vector<std::pair<float, uint32_t>> merged;
    if (env.world_rank == 0) {
        for (size_t i = 0; i < recv_dis.size(); ++i) {
            if (recv_ids[i] != UINT32_MAX) {
                merged.push_back(std::make_pair(recv_dis[i], recv_ids[i]));
            }
        }
    }
    return merged;
}

static inline std::vector<std::pair<float, uint32_t>> mpi_gather_rma(
    const MPIEnv& env,
    const std::vector<std::pair<float, uint32_t>>& local_res,
    int k)
{
    std::vector<float> send_dis(k, FLT_MAX);
    std::vector<uint32_t> send_ids(k, UINT32_MAX);
    for (size_t i = 0; i < local_res.size() && i < (size_t)k; ++i) {
        send_dis[i] = local_res[i].first;
        send_ids[i] = local_res[i].second;
    }
    float* root_dis = nullptr;
    uint32_t* root_ids = nullptr;
    MPI_Win win_dis, win_ids;
    if (env.world_rank == 0) {
        MPI_Win_allocate((MPI_Aint)((size_t)env.world_size * k * sizeof(float)), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &root_dis, &win_dis);
        MPI_Win_allocate((MPI_Aint)((size_t)env.world_size * k * sizeof(uint32_t)), sizeof(uint32_t), MPI_INFO_NULL, MPI_COMM_WORLD, &root_ids, &win_ids);
    } else {
        MPI_Win_allocate(0, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &root_dis, &win_dis);
        MPI_Win_allocate(0, sizeof(uint32_t), MPI_INFO_NULL, MPI_COMM_WORLD, &root_ids, &win_ids);
    }
    MPI_Win_fence(0, win_dis);
    MPI_Win_fence(0, win_ids);
    MPI_Put(send_dis.data(), k, MPI_FLOAT, 0, env.world_rank * k, k, MPI_FLOAT, win_dis);
    MPI_Put(send_ids.data(), k, MPI_UNSIGNED, 0, env.world_rank * k, k, MPI_UNSIGNED, win_ids);
    MPI_Win_fence(0, win_dis);
    MPI_Win_fence(0, win_ids);
    std::vector<std::pair<float, uint32_t>> merged;
    if (env.world_rank == 0) {
        for (int r = 0; r < env.world_size; ++r) {
            for (int i = 0; i < k; ++i) {
                uint32_t id = root_ids[r * k + i];
                if (id != UINT32_MAX) {
                    merged.push_back(std::make_pair(root_dis[r * k + i], id));
                }
            }
        }
    }
    MPI_Win_free(&win_dis);
    MPI_Win_free(&win_ids);
    return merged;
}

static inline std::vector<std::pair<float, uint32_t>> mpi_collect_results(
    const MPIEnv& env,
    const MPIRunConfig& cfg,
    const std::vector<std::pair<float, uint32_t>>& local_res)
{
    if (cfg.comm_mode == "nonblocking") {
        return mpi_gather_nonblocking(env, local_res, cfg.k);
    }
    if (cfg.comm_mode == "onesided") {
        return mpi_gather_rma(env, local_res, cfg.k);
    }
    return mpi_gather_blocking(env, local_res, cfg.k);
}
