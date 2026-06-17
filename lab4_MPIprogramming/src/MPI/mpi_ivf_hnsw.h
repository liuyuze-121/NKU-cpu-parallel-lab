#pragma once
#include "common.h"

// IVF+HNSW 组合图索引：
//   1. 每个 rank 的 IVF 在 driver 入口处只训练一次（centroids + 倒排链）；
//   2. 每条 query：coarse 选 top-nprobe 簇 → 抽取候选 → 在候选上临时构建
//      hnswlib::HierarchicalNSW → 局部 search → 回到主流程 merge。
// 与旧版相比，IVF 训练从 per-query 改为 once-for-all。
struct IVFHNSWContext {
    std::unique_ptr<IVFFlatSearcherPthread> coarse_pthread;
    std::unique_ptr<IVFFlatSearcherOpenMP> coarse_openmp;
    int nlist = 0;
};

static inline IVFHNSWContext mpi_build_ivf_hnsw(
    const MPIRunConfig& cfg,
    float* local_base,
    size_t local_number,
    size_t vecdim)
{
    IVFHNSWContext ctx;
    ctx.nlist = std::max(1, std::min(cfg.ivf_nlist, (int)local_number));
    if (cfg.thread_mode == "openmp") {
        omp_set_num_threads(cfg.threads);
        ctx.coarse_openmp.reset(new IVFFlatSearcherOpenMP(
            ctx.nlist, local_base, local_number, vecdim));
        ctx.coarse_openmp->train();
        ctx.coarse_openmp->assign_base();
    } else {
        ctx.coarse_pthread.reset(new IVFFlatSearcherPthread(
            ctx.nlist, local_base, local_number, vecdim));
        ctx.coarse_pthread->train();
        ctx.coarse_pthread->assign_base();
    }
    return ctx;
}

static inline std::vector<std::pair<float, uint32_t>> mpi_local_search_ivf_hnsw(
    IVFHNSWContext& ctx,
    const MPIRunConfig& cfg,
    float* query)
{
    using IPFn = float(*)(const float*, const float*, size_t);
    IPFn ip = (cfg.thread_mode == "openmp")
        ? IVFFlatSearcherOpenMP::inner_product_neon
        : IVFFlatSearcherPthread::inner_product_neon;

    int nlist = ctx.nlist;
    const float* centroids = (ctx.coarse_openmp)
        ? ctx.coarse_openmp->centroids_.data()
        : ctx.coarse_pthread->centroids_.data();
    const auto& lists = (ctx.coarse_openmp)
        ? ctx.coarse_openmp->lists_
        : ctx.coarse_pthread->lists_;

    std::vector<std::pair<float, int>> centroid_dists(nlist);
    for (int c = 0; c < nlist; ++c) {
        float ip_val = ip(query, centroids + c * cfg.vecdim, cfg.vecdim);
        centroid_dists[c] = std::make_pair(1.0f - ip_val, c);
    }
    std::sort(centroid_dists.begin(), centroid_dists.end());

    std::vector<uint32_t> candidates;
    int nprobe = std::min(cfg.ivf_nprobe, nlist);
    for (int i = 0; i < nprobe; ++i) {
        int list_id = centroid_dists[i].second;
        candidates.insert(candidates.end(), lists[list_id].begin(), lists[list_id].end());
    }
    if (candidates.empty()) {
        return {};
    }

    const float* local_base = (ctx.coarse_openmp)
        ? ctx.coarse_openmp->base_
        : ctx.coarse_pthread->base_;
    std::vector<float> candidate_base(candidates.size() * cfg.vecdim);
    for (size_t i = 0; i < candidates.size(); ++i) {
        memcpy(candidate_base.data() + i * cfg.vecdim,
               local_base + (size_t)candidates[i] * cfg.vecdim,
               cfg.vecdim * sizeof(float));
    }

    std::priority_queue<std::pair<float, uint32_t>> q;
    if (cfg.thread_mode == "openmp") {
        HNSWSearcherOpenMP searcher(cfg.hnsw_m, cfg.hnsw_ef_construction, cfg.hnsw_ef,
                                    candidate_base.data(), candidates.size(), cfg.vecdim);
        searcher.build();
        q = searcher.search(query, cfg.k);
    } else {
        HNSWSearcherPthread searcher(cfg.hnsw_m, cfg.hnsw_ef_construction, cfg.hnsw_ef,
                                     candidate_base.data(), candidates.size(), cfg.vecdim);
        searcher.build();
        q = searcher.search(query, cfg.k);
    }

    std::vector<std::pair<float, uint32_t>> res = mpi_priority_queue_to_sorted_vector(q);
    for (size_t i = 0; i < res.size(); ++i) {
        res[i].second = candidates[res[i].second];
    }
    return res;
}

static inline MPISearchResult mpi_run_ivf_hnsw(
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

    // IVF 在 driver 入口处只训练一次
    IVFHNSWContext ivf_ctx = mpi_build_ivf_hnsw(cfg, local_base.data(), local_number, vecdim);

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
            mpi_local_search_ivf_hnsw(ivf_ctx, cfg, shared_query.data());
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
