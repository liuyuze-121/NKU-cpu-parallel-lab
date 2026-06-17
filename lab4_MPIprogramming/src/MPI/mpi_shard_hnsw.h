#pragma once
#include "common.h"

// 分片 HNSW：每个 MPI rank 独立构建一个 hnswlib::HierarchicalNSW 索引，
// 复用 omp/pthread 中已通过 hnswlib 基础图索引验证的实现。
// 与旧版相比：
//   1. 索引在 driver 入口处一次性构建，所有 query 共享；
//   2. 每条 query 只做 Bcast + 局部 HNSW search，不再有 per-query build 开销；
//   3. 通过 cfg.thread_mode 切换 rank 内使用 OpenMP 还是 pthread 的 query 级并行。
struct ShardHNSWContext {
    std::unique_ptr<HNSWSearcherOpenMP> omp_searcher;
    std::unique_ptr<HNSWSearcherPthread> pthread_searcher;
};

static inline ShardHNSWContext mpi_build_shard_hnsw(
    const MPIRunConfig& cfg,
    float* local_base,
    size_t local_number,
    size_t vecdim)
{
    ShardHNSWContext ctx;
    if (cfg.thread_mode == "openmp") {
        omp_set_num_threads(cfg.threads);
        ctx.omp_searcher.reset(new HNSWSearcherOpenMP(
            cfg.hnsw_m, cfg.hnsw_ef_construction, cfg.hnsw_ef,
            local_base, local_number, vecdim));
        ctx.omp_searcher->build();
    } else {
        ctx.pthread_searcher.reset(new HNSWSearcherPthread(
            cfg.hnsw_m, cfg.hnsw_ef_construction, cfg.hnsw_ef,
            local_base, local_number, vecdim));
        ctx.pthread_searcher->build();
    }
    return ctx;
}

static inline std::vector<std::pair<float, uint32_t>> mpi_local_search_shard_hnsw(
    ShardHNSWContext& ctx,
    const MPIRunConfig& cfg,
    float* query)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    if (ctx.omp_searcher) {
        q = ctx.omp_searcher->search(query, cfg.k);
    } else {
        q = ctx.pthread_searcher->search(query, cfg.k);
    }
    return mpi_priority_queue_to_sorted_vector(q);
}

static inline MPISearchResult mpi_run_shard_hnsw(
    const MPIEnv& env,
    const MPIRunConfig& cfg,
    float* base,
    size_t base_number,
    float* queries,
    int* gt,
    size_t query_number,
    size_t gt_dim,
    size_t vecdim)
{
    std::vector<float> local_base;
    size_t local_number = 0;
    size_t offset = 0;
    mpi_build_shard(env, base, base_number, vecdim, local_base, local_number, offset);

    // 索引在 driver 入口处只构建一次
    ShardHNSWContext shard_ctx = mpi_build_shard_hnsw(cfg, local_base.data(), local_number, vecdim);

    size_t query_count = std::min((size_t)cfg.mpi_queries, query_number);
    double recall_sum = 0.0;
    double latency_sum = 0.0;
    double max_latency = 0.0;

    std::vector<float> shared_query(vecdim);
    for (size_t qi = 0; qi < query_count; ++qi) {
        if (env.world_rank == 0) {
            memcpy(shared_query.data(), queries + qi * vecdim, vecdim * sizeof(float));
        }
        double start = 0.0;
        if (env.world_rank == 0) {
            start = MPI_Wtime();
        }
        if (cfg.comm_mode == "nonblocking") {
            MPI_Request req;
            MPI_Ibcast(shared_query.data(), (int)vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        } else {
            MPI_Bcast(shared_query.data(), (int)vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        std::vector<std::pair<float, uint32_t>> local_res =
            mpi_local_search_shard_hnsw(shard_ctx, cfg, shared_query.data());
        mpi_shift_indices(local_res, offset);
        std::vector<std::pair<float, uint32_t>> merged = mpi_collect_results(env, cfg, local_res);
        if (env.world_rank == 0) {
            std::priority_queue<std::pair<float, uint32_t>> final_q = mpi_merge_topk(merged, cfg.k);
            std::vector<std::pair<float, uint32_t>> final_sorted = mpi_priority_queue_to_sorted_vector(final_q);
            float recall = mpi_compute_recall_from_sorted(final_sorted, gt, gt_dim, qi, cfg.k);
            double elapsed = (MPI_Wtime() - start) * 1e6;
            recall_sum += recall;
            latency_sum += elapsed;
            if (elapsed > max_latency) {
                max_latency = elapsed;
            }
        }
    }

    MPISearchResult result;
    if (env.world_rank == 0 && query_count > 0) {
        result.avg_recall = (float)(recall_sum / (double)query_count);
        result.avg_latency_us = latency_sum / (double)query_count;
        result.max_latency_us = max_latency;
        result.query_count = query_count;
    }
    return result;
}
