#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static const char* kPtx = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry flat_score_kernel(
    .param .u64 base,
    .param .u64 queries,
    .param .u64 scores,
    .param .u32 n,
    .param .u32 d,
    .param .u32 batch
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [base];
    ld.param.u64 %rd2, [queries];
    ld.param.u64 %rd3, [scores];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r2, [d];
    ld.param.u32 %r3, [batch];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r4, %r5, %r6;

    mul.lo.u32 %r8, %r1, %r3;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra FLAT_DONE;

    div.u32 %r9, %r7, %r1;
    mul.lo.u32 %r10, %r9, %r1;
    sub.u32 %r11, %r7, %r10;

    mov.u32 %r12, 0;
    mov.f32 %f1, 0f00000000;

FLAT_LOOP:
    setp.ge.u32 %p2, %r12, %r2;
    @%p2 bra FLAT_AFTER;

    mad.lo.u32 %r13, %r11, %r2, %r12;
    mul.wide.u32 %rd4, %r13, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    mad.lo.u32 %r14, %r9, %r2, %r12;
    mul.wide.u32 %rd6, %r14, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];

    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r12, %r12, 1;
    bra FLAT_LOOP;

FLAT_AFTER:
    mov.f32 %f4, 0f3f800000;
    sub.rn.f32 %f5, %f4, %f1;
    mul.wide.u32 %rd8, %r7, 4;
    add.u64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f5;

FLAT_DONE:
    ret;
}

.visible .entry indexed_score_kernel(
    .param .u64 base,
    .param .u64 queries,
    .param .u64 cand_ids,
    .param .u64 q_ids,
    .param .u64 scores,
    .param .u32 pairs,
    .param .u32 d
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [base];
    ld.param.u64 %rd2, [queries];
    ld.param.u64 %rd3, [cand_ids];
    ld.param.u64 %rd4, [q_ids];
    ld.param.u64 %rd5, [scores];
    ld.param.u32 %r1, [pairs];
    ld.param.u32 %r2, [d];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra INDEX_DONE;

    mul.wide.u32 %rd6, %r6, 4;
    add.u64 %rd7, %rd3, %rd6;
    ld.global.u32 %r7, [%rd7];
    add.u64 %rd8, %rd4, %rd6;
    ld.global.u32 %r8, [%rd8];

    mov.u32 %r9, 0;
    mov.f32 %f1, 0f00000000;

INDEX_LOOP:
    setp.ge.u32 %p2, %r9, %r2;
    @%p2 bra INDEX_AFTER;

    mad.lo.u32 %r10, %r7, %r2, %r9;
    mul.wide.u32 %rd9, %r10, 4;
    add.u64 %rd10, %rd1, %rd9;
    ld.global.f32 %f2, [%rd10];

    mad.lo.u32 %r11, %r8, %r2, %r9;
    mul.wide.u32 %rd11, %r11, 4;
    add.u64 %rd12, %rd2, %rd11;
    ld.global.f32 %f3, [%rd12];

    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r9, %r9, 1;
    bra INDEX_LOOP;

INDEX_AFTER:
    mov.f32 %f4, 0f3f800000;
    sub.rn.f32 %f5, %f4, %f1;
    mul.wide.u32 %rd13, %r6, 4;
    add.u64 %rd14, %rd5, %rd13;
    st.global.f32 [%rd14], %f5;

INDEX_DONE:
    ret;
}

.visible .entry cluster_score_kernel(
    .param .u64 base,
    .param .u64 queries,
    .param .u64 cand_ids,
    .param .u64 scores,
    .param .u32 cands,
    .param .u32 d,
    .param .u32 batch
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<22>;
    .reg .b64 %rd<22>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [base];
    ld.param.u64 %rd2, [queries];
    ld.param.u64 %rd3, [cand_ids];
    ld.param.u64 %rd4, [scores];
    ld.param.u32 %r1, [cands];
    ld.param.u32 %r2, [d];
    ld.param.u32 %r3, [batch];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r4, %r5, %r6;

    mul.lo.u32 %r8, %r1, %r3;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra CLUSTER_DONE;

    div.u32 %r9, %r7, %r1;
    mul.lo.u32 %r10, %r9, %r1;
    sub.u32 %r11, %r7, %r10;
    mul.wide.u32 %rd5, %r11, 4;
    add.u64 %rd6, %rd3, %rd5;
    ld.global.u32 %r12, [%rd6];

    mov.u32 %r13, 0;
    mov.f32 %f1, 0f00000000;

CLUSTER_LOOP:
    setp.ge.u32 %p2, %r13, %r2;
    @%p2 bra CLUSTER_AFTER;

    mad.lo.u32 %r14, %r12, %r2, %r13;
    mul.wide.u32 %rd7, %r14, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.f32 %f2, [%rd8];

    mad.lo.u32 %r15, %r9, %r2, %r13;
    mul.wide.u32 %rd9, %r15, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f3, [%rd10];

    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r13, %r13, 1;
    bra CLUSTER_LOOP;

CLUSTER_AFTER:
    mov.f32 %f4, 0f3f800000;
    sub.rn.f32 %f5, %f4, %f1;
    mul.wide.u32 %rd11, %r7, 4;
    add.u64 %rd12, %rd4, %rd11;
    st.global.f32 [%rd12], %f5;

CLUSTER_DONE:
    ret;
}
)ptx";

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::string fmt(double x, int digits = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(digits) << x;
    return oss.str();
}

static std::string csv_escape(const std::string& s) {
    bool quote = false;
    for (char ch : s) {
        if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') quote = true;
    }
    if (!quote) return s;
    std::string out = "\"";
    for (char ch : s) out += (ch == '"') ? "\"\"" : std::string(1, ch);
    out += "\"";
    return out;
}

static std::string sanitize_filename(const std::string& s) {
    std::string out;
    for (char ch : s) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '=' || ch == '.' || ch == '-';
        out.push_back(ok ? ch : '_');
    }
    while (out.find("__") != std::string::npos) out.replace(out.find("__"), 2, "_");
    if (!out.empty() && out.front() == '_') out.erase(out.begin());
    if (!out.empty() && out.back() == '_') out.pop_back();
    return out.empty() ? "default" : out;
}

template <typename T>
struct Matrix {
    size_t n = 0;
    size_t d = 0;
    std::vector<T> values;
    const T* row(size_t i) const { return values.data() + i * d; }
    T* row(size_t i) { return values.data() + i * d; }
};

static uint32_t read_u32(std::ifstream& in) {
    uint32_t x = 0;
    in.read(reinterpret_cast<char*>(&x), sizeof(x));
    if (!in) throw std::runtime_error("failed to read matrix header");
    return x;
}

template <typename T>
static Matrix<T> load_matrix(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    Matrix<T> mat;
    mat.n = read_u32(in);
    mat.d = read_u32(in);
    mat.values.resize(mat.n * mat.d);
    in.read(reinterpret_cast<char*>(mat.values.data()),
            static_cast<std::streamsize>(mat.values.size() * sizeof(T)));
    if (!in) throw std::runtime_error("failed to read body from " + path);
    return mat;
}

static float inner_product(const float* a, const float* b, size_t d) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < d; ++i) sum += a[i] * b[i];
    return sum;
}

static void push_topk(std::priority_queue<std::pair<float, int>>& heap,
                      float dist,
                      int id,
                      size_t k) {
    if (heap.size() < k) heap.push({dist, id});
    else if (dist < heap.top().first) {
        heap.push({dist, id});
        heap.pop();
    }
}

static float heap_recall(std::priority_queue<std::pair<float, int>> heap,
                         const Matrix<int>& gt,
                         size_t qi,
                         size_t k) {
    size_t ok = 0;
    while (!heap.empty()) {
        const int id = heap.top().second;
        heap.pop();
        for (size_t j = 0; j < k && j < gt.d; ++j) {
            if (gt.row(qi)[j] == id) {
                ++ok;
                break;
            }
        }
    }
    return k ? static_cast<float>(ok) / static_cast<float>(k) : 0.0f;
}

static double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    if (lo == hi) return values[lo];
    const double w = rank - static_cast<double>(lo);
    return values[lo] * (1.0 - w) + values[hi] * w;
}

class CudaDriver {
public:
    using CUresult = int;
    using CUdevice = int;
    using CUcontext = void*;
    using CUmodule = void*;
    using CUfunction = void*;
    using CUevent = void*;
    using CUdeviceptr = unsigned long long;

    CudaDriver() {
        lib_ = LoadLibraryA("nvcuda.dll");
        if (!lib_) throw std::runtime_error("cannot load nvcuda.dll");
        load_symbols();
        check(cuInit_(0), "cuInit");
        check(cuDeviceGet_(&device_, 0), "cuDeviceGet");
        check(cuCtxCreate_(&context_, 0, device_), "cuCtxCreate");
        check(cuModuleLoadData_(&module_, kPtx), "cuModuleLoadData");
        check(cuModuleGetFunction_(&flat_func_, module_, "flat_score_kernel"), "cuModuleGetFunction(flat)");
        check(cuModuleGetFunction_(&indexed_func_, module_, "indexed_score_kernel"), "cuModuleGetFunction(indexed)");
        check(cuModuleGetFunction_(&cluster_func_, module_, "cluster_score_kernel"), "cuModuleGetFunction(cluster)");
    }

    ~CudaDriver() {
        if (module_) cuModuleUnload_(module_);
        if (context_) cuCtxDestroy_(context_);
        if (lib_) FreeLibrary(lib_);
    }

    CUfunction flat_func() const { return flat_func_; }
    CUfunction indexed_func() const { return indexed_func_; }
    CUfunction cluster_func() const { return cluster_func_; }

    std::map<std::string, std::string> device_info() const {
        char name[256] = {};
        int major = 0, minor = 0, version = 0;
        check(cuDeviceGetName_(name, 256, device_), "cuDeviceGetName");
        check(cuDeviceComputeCapability_(&major, &minor, device_), "cuDeviceComputeCapability");
        check(cuDriverGetVersion_(&version), "cuDriverGetVersion");
        return {
            {"gpu", name},
            {"compute_capability", std::to_string(major) + "." + std::to_string(minor)},
            {"cuda_driver_version", std::to_string(version)}
        };
    }

    CUdeviceptr mem_alloc(size_t bytes) const {
        CUdeviceptr ptr = 0;
        check(cuMemAlloc_(&ptr, bytes), "cuMemAlloc");
        return ptr;
    }

    void mem_free(CUdeviceptr ptr) const {
        if (ptr) check(cuMemFree_(ptr), "cuMemFree");
    }

    double memcpy_h2d(CUdeviceptr dst, const void* src, size_t bytes) const {
        const auto begin = Clock::now();
        check(cuMemcpyHtoD_(dst, src, bytes), "cuMemcpyHtoD");
        return elapsed_ms(begin, Clock::now());
    }

    double memcpy_d2h(void* dst, CUdeviceptr src, size_t bytes) const {
        const auto begin = Clock::now();
        check(cuMemcpyDtoH_(dst, src, bytes), "cuMemcpyDtoH");
        return elapsed_ms(begin, Clock::now());
    }

    double launch(CUfunction func,
                  unsigned grid_x,
                  unsigned block_x,
                  std::vector<void*>& params) const {
        CUevent start = nullptr, stop = nullptr;
        check(cuEventCreate_(&start, 0), "cuEventCreate(start)");
        check(cuEventCreate_(&stop, 0), "cuEventCreate(stop)");
        check(cuEventRecord_(start, nullptr), "cuEventRecord(start)");
        check(cuLaunchKernel_(func, grid_x, 1, 1, block_x, 1, 1, 0, nullptr, params.data(), nullptr),
              "cuLaunchKernel");
        check(cuEventRecord_(stop, nullptr), "cuEventRecord(stop)");
        check(cuEventSynchronize_(stop), "cuEventSynchronize");
        float ms = 0.0f;
        check(cuEventElapsedTime_(&ms, start, stop), "cuEventElapsedTime");
        check(cuEventDestroy_(start), "cuEventDestroy(start)");
        check(cuEventDestroy_(stop), "cuEventDestroy(stop)");
        return ms;
    }

private:
    HMODULE lib_ = nullptr;
    CUdevice device_ = 0;
    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction flat_func_ = nullptr;
    CUfunction indexed_func_ = nullptr;
    CUfunction cluster_func_ = nullptr;

    using cuInit_t = CUresult (*)(unsigned);
    using cuDriverGetVersion_t = CUresult (*)(int*);
    using cuGetErrorString_t = CUresult (*)(CUresult, const char**);
    using cuDeviceGet_t = CUresult (*)(CUdevice*, int);
    using cuDeviceGetName_t = CUresult (*)(char*, int, CUdevice);
    using cuDeviceComputeCapability_t = CUresult (*)(int*, int*, CUdevice);
    using cuCtxCreate_t = CUresult (*)(CUcontext*, unsigned, CUdevice);
    using cuCtxDestroy_t = CUresult (*)(CUcontext);
    using cuModuleLoadData_t = CUresult (*)(CUmodule*, const void*);
    using cuModuleUnload_t = CUresult (*)(CUmodule);
    using cuModuleGetFunction_t = CUresult (*)(CUfunction*, CUmodule, const char*);
    using cuMemAlloc_t = CUresult (*)(CUdeviceptr*, size_t);
    using cuMemFree_t = CUresult (*)(CUdeviceptr);
    using cuMemcpyHtoD_t = CUresult (*)(CUdeviceptr, const void*, size_t);
    using cuMemcpyDtoH_t = CUresult (*)(void*, CUdeviceptr, size_t);
    using cuLaunchKernel_t = CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                          unsigned, unsigned, void*, void**, void**);
    using cuEventCreate_t = CUresult (*)(CUevent*, unsigned);
    using cuEventRecord_t = CUresult (*)(CUevent, void*);
    using cuEventSynchronize_t = CUresult (*)(CUevent);
    using cuEventElapsedTime_t = CUresult (*)(float*, CUevent, CUevent);
    using cuEventDestroy_t = CUresult (*)(CUevent);

    cuInit_t cuInit_ = nullptr;
    cuDriverGetVersion_t cuDriverGetVersion_ = nullptr;
    cuGetErrorString_t cuGetErrorString_ = nullptr;
    cuDeviceGet_t cuDeviceGet_ = nullptr;
    cuDeviceGetName_t cuDeviceGetName_ = nullptr;
    cuDeviceComputeCapability_t cuDeviceComputeCapability_ = nullptr;
    cuCtxCreate_t cuCtxCreate_ = nullptr;
    cuCtxDestroy_t cuCtxDestroy_ = nullptr;
    cuModuleLoadData_t cuModuleLoadData_ = nullptr;
    cuModuleUnload_t cuModuleUnload_ = nullptr;
    cuModuleGetFunction_t cuModuleGetFunction_ = nullptr;
    cuMemAlloc_t cuMemAlloc_ = nullptr;
    cuMemFree_t cuMemFree_ = nullptr;
    cuMemcpyHtoD_t cuMemcpyHtoD_ = nullptr;
    cuMemcpyDtoH_t cuMemcpyDtoH_ = nullptr;
    cuLaunchKernel_t cuLaunchKernel_ = nullptr;
    cuEventCreate_t cuEventCreate_ = nullptr;
    cuEventRecord_t cuEventRecord_ = nullptr;
    cuEventSynchronize_t cuEventSynchronize_ = nullptr;
    cuEventElapsedTime_t cuEventElapsedTime_ = nullptr;
    cuEventDestroy_t cuEventDestroy_ = nullptr;

    template <typename T>
    void load(T& fn, const char* name) {
        fn = reinterpret_cast<T>(GetProcAddress(lib_, name));
        if (!fn) throw std::runtime_error(std::string("missing CUDA symbol: ") + name);
    }

    void load_symbols() {
        load(cuInit_, "cuInit");
        load(cuDriverGetVersion_, "cuDriverGetVersion");
        load(cuGetErrorString_, "cuGetErrorString");
        load(cuDeviceGet_, "cuDeviceGet");
        load(cuDeviceGetName_, "cuDeviceGetName");
        load(cuDeviceComputeCapability_, "cuDeviceComputeCapability");
        load(cuCtxCreate_, "cuCtxCreate_v2");
        load(cuCtxDestroy_, "cuCtxDestroy_v2");
        load(cuModuleLoadData_, "cuModuleLoadData");
        load(cuModuleUnload_, "cuModuleUnload");
        load(cuModuleGetFunction_, "cuModuleGetFunction");
        load(cuMemAlloc_, "cuMemAlloc_v2");
        load(cuMemFree_, "cuMemFree_v2");
        load(cuMemcpyHtoD_, "cuMemcpyHtoD_v2");
        load(cuMemcpyDtoH_, "cuMemcpyDtoH_v2");
        load(cuLaunchKernel_, "cuLaunchKernel");
        load(cuEventCreate_, "cuEventCreate");
        load(cuEventRecord_, "cuEventRecord");
        load(cuEventSynchronize_, "cuEventSynchronize");
        load(cuEventElapsedTime_, "cuEventElapsedTime");
        load(cuEventDestroy_, "cuEventDestroy_v2");
    }

    void check(CUresult code, const char* where) const {
        if (code == 0) return;
        const char* msg = nullptr;
        if (cuGetErrorString_) cuGetErrorString_(code, &msg);
        std::ostringstream oss;
        oss << where << " failed: " << code;
        if (msg) oss << " " << msg;
        throw std::runtime_error(oss.str());
    }
};

struct ScopedDevicePtr {
    const CudaDriver* cuda = nullptr;
    CudaDriver::CUdeviceptr ptr = 0;
    ScopedDevicePtr() = default;
    ScopedDevicePtr(const CudaDriver& c, size_t bytes) : cuda(&c), ptr(c.mem_alloc(bytes)) {}
    ~ScopedDevicePtr() {
        if (cuda && ptr) cuda->mem_free(ptr);
    }
    ScopedDevicePtr(const ScopedDevicePtr&) = delete;
    ScopedDevicePtr& operator=(const ScopedDevicePtr&) = delete;
};

struct RunResult {
    std::string algorithm;
    std::string variant;
    std::string params;
    size_t queries = 0;
    size_t k = 0;
    size_t batch_size = 0;
    std::string nlist;
    std::string nprobe;
    double build_ms = 0.0;
    double avg_recall = 0.0;
    double avg_latency_us = 0.0;
    double p50_latency_us = 0.0;
    double p95_latency_us = 0.0;
    double max_latency_us = 0.0;
    double total_time_ms = 0.0;
    double kernel_ms = 0.0;
    double h2d_ms = 0.0;
    double d2h_ms = 0.0;
    double host_select_ms = 0.0;
    double avg_candidates = 0.0;
    double waste_ratio = 1.0;
    std::string measured_by = "cpp_nvidia_driver_api_ptx_jit";
};

struct PerQueryRow {
    size_t query = 0;
    double recall = 0.0;
    double latency_us = 0.0;
    size_t candidates = 0;
};

static void write_summary_csv(const fs::path& path, const std::vector<RunResult>& rows) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "algorithm,variant,params,queries,k,batch_size,nlist,nprobe,build_ms,avg_recall,"
           "avg_latency_us,p50_latency_us,p95_latency_us,max_latency_us,total_time_ms,kernel_ms,"
           "h2d_ms,d2h_ms,host_select_ms,avg_candidates,waste_ratio,measured_by\n";
    for (const auto& r : rows) {
        out << csv_escape(r.algorithm) << "," << csv_escape(r.variant) << "," << csv_escape(r.params)
            << "," << r.queries << "," << r.k << "," << r.batch_size << ","
            << csv_escape(r.nlist) << "," << csv_escape(r.nprobe) << ","
            << fmt(r.build_ms) << "," << fmt(r.avg_recall, 6) << ","
            << fmt(r.avg_latency_us) << "," << fmt(r.p50_latency_us) << ","
            << fmt(r.p95_latency_us) << "," << fmt(r.max_latency_us) << ","
            << fmt(r.total_time_ms) << "," << fmt(r.kernel_ms) << ","
            << fmt(r.h2d_ms) << "," << fmt(r.d2h_ms) << ","
            << fmt(r.host_select_ms) << "," << fmt(r.avg_candidates, 1) << ","
            << fmt(r.waste_ratio, 6) << "," << csv_escape(r.measured_by) << "\n";
    }
}

static void write_per_query_csv(const fs::path& path, const std::vector<PerQueryRow>& rows) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "query,recall,latency_us,candidates\n";
    for (const auto& r : rows) {
        out << r.query << "," << fmt(r.recall, 6) << "," << fmt(r.latency_us)
            << "," << r.candidates << "\n";
    }
}

static void write_run_files(const fs::path& out_dir,
                            const RunResult& result,
                            const std::vector<PerQueryRow>& per_query) {
    const std::string key = "q=" + std::to_string(result.queries) + ",k=" +
                            std::to_string(result.k) + "," + result.params;
    const fs::path run_dir = out_dir / result.algorithm / sanitize_filename(key);
    write_summary_csv(run_dir / "summary.csv", {result});
    write_per_query_csv(run_dir / "per_query.csv", per_query);
    std::ofstream md(run_dir / "result.md");
    md << "# " << result.algorithm << "\n\n";
    md << "- params: `" << result.params << "`\n";
    md << "- queries: `" << result.queries << "`, k: `" << result.k << "`\n";
    md << "- recall: `" << fmt(result.avg_recall, 6) << "`\n";
    md << "- avg latency us: `" << fmt(result.avg_latency_us) << "`\n";
    md << "- kernel ms: `" << fmt(result.kernel_ms) << "`\n";
}

static RunResult make_result(const std::string& algorithm,
                             const std::string& variant,
                             const std::string& params,
                             size_t queries,
                             size_t k,
                             size_t batch,
                             const std::string& nlist,
                             const std::string& nprobe,
                             double build_ms,
                             const std::vector<double>& recalls,
                             const std::vector<double>& latencies,
                             double total_ms,
                             double kernel_ms,
                             double h2d_ms,
                             double d2h_ms,
                             double select_ms,
                             double avg_candidates,
                             double waste_ratio) {
    RunResult r;
    r.algorithm = algorithm;
    r.variant = variant;
    r.params = params;
    r.queries = queries;
    r.k = k;
    r.batch_size = batch;
    r.nlist = nlist;
    r.nprobe = nprobe;
    r.build_ms = build_ms;
    r.avg_recall = recalls.empty() ? 0.0 : std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
    r.avg_latency_us = latencies.empty() ? 0.0 : std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    r.p50_latency_us = percentile(latencies, 50.0);
    r.p95_latency_us = percentile(latencies, 95.0);
    r.max_latency_us = latencies.empty() ? 0.0 : *std::max_element(latencies.begin(), latencies.end());
    r.total_time_ms = total_ms;
    r.kernel_ms = kernel_ms;
    r.h2d_ms = h2d_ms;
    r.d2h_ms = d2h_ms;
    r.host_select_ms = select_ms;
    r.avg_candidates = avg_candidates;
    r.waste_ratio = waste_ratio;
    return r;
}

class IVFIndex {
public:
    IVFIndex(const Matrix<float>& base, int nlist, int iters, unsigned seed)
        : base_(base), nlist_(nlist), iters_(iters), seed_(seed) {}

    void build() {
        const auto begin = Clock::now();
        centroids_.assign(static_cast<size_t>(nlist_) * base_.d, 0.0f);
        lists_.assign(static_cast<size_t>(nlist_), {});
        init_centroids();
        std::vector<int> assignment(base_.n, 0);
        for (int iter = 0; iter < iters_; ++iter) {
            for (size_t i = 0; i < base_.n; ++i) {
                assignment[i] = nearest_centroid(base_.row(i));
            }
            std::vector<float> next(centroids_.size(), 0.0f);
            std::vector<size_t> counts(static_cast<size_t>(nlist_), 0);
            for (size_t i = 0; i < base_.n; ++i) {
                const int c = assignment[i];
                ++counts[static_cast<size_t>(c)];
                float* dst = next.data() + static_cast<size_t>(c) * base_.d;
                const float* src = base_.row(i);
                for (size_t j = 0; j < base_.d; ++j) dst[j] += src[j];
            }
            for (int c = 0; c < nlist_; ++c) {
                if (counts[static_cast<size_t>(c)] == 0) continue;
                float* dst = centroids_.data() + static_cast<size_t>(c) * base_.d;
                const float inv = 1.0f / static_cast<float>(counts[static_cast<size_t>(c)]);
                for (size_t j = 0; j < base_.d; ++j) {
                    dst[j] = next[static_cast<size_t>(c) * base_.d + j] * inv;
                }
            }
        }
        for (size_t i = 0; i < base_.n; ++i) {
            const int c = nearest_centroid(base_.row(i));
            lists_[static_cast<size_t>(c)].push_back(static_cast<int>(i));
        }
        build_ms_ = elapsed_ms(begin, Clock::now());
    }

    std::vector<int> top_clusters(const float* query, int nprobe) const {
        std::vector<std::pair<float, int>> scores(static_cast<size_t>(nlist_));
        for (int c = 0; c < nlist_; ++c) {
            const float ip = inner_product(query, centroid(c), base_.d);
            scores[static_cast<size_t>(c)] = {1.0f - ip, c};
        }
        nprobe = std::min(nprobe, nlist_);
        std::partial_sort(scores.begin(), scores.begin() + nprobe, scores.end());
        std::vector<int> out(static_cast<size_t>(nprobe));
        for (int i = 0; i < nprobe; ++i) out[static_cast<size_t>(i)] = scores[static_cast<size_t>(i)].second;
        return out;
    }

    const std::vector<int>& list(int c) const { return lists_[static_cast<size_t>(c)]; }
    double build_ms() const { return build_ms_; }
    int nlist() const { return nlist_; }

private:
    const Matrix<float>& base_;
    int nlist_ = 0;
    int iters_ = 0;
    unsigned seed_ = 0;
    std::vector<float> centroids_;
    std::vector<std::vector<int>> lists_;
    double build_ms_ = 0.0;

    const float* centroid(int c) const {
        return centroids_.data() + static_cast<size_t>(c) * base_.d;
    }

    void init_centroids() {
        std::mt19937 rng(seed_);
        std::uniform_int_distribution<size_t> dist(0, base_.n - 1);
        std::unordered_set<size_t> used;
        for (int c = 0; c < nlist_; ++c) {
            size_t idx = dist(rng);
            while (used.find(idx) != used.end()) idx = (idx + 9973) % base_.n;
            used.insert(idx);
            std::memcpy(centroids_.data() + static_cast<size_t>(c) * base_.d,
                        base_.row(idx), base_.d * sizeof(float));
        }
    }

    int nearest_centroid(const float* v) const {
        int best = 0;
        float best_ip = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < nlist_; ++c) {
            const float ip = inner_product(v, centroid(c), base_.d);
            if (ip > best_ip) {
                best_ip = ip;
                best = c;
            }
        }
        return best;
    }
};

class GpuExperiment {
public:
    GpuExperiment(const Matrix<float>& base, const Matrix<float>& queries, const Matrix<int>& gt)
        : base_(base), queries_(queries), gt_(gt), cuda_(), d_base_(cuda_, base.values.size() * sizeof(float)) {
        base_upload_ms_ = cuda_.memcpy_h2d(d_base_.ptr, base.values.data(), base.values.size() * sizeof(float));
    }

    const CudaDriver& cuda() const { return cuda_; }
    double base_upload_ms() const { return base_upload_ms_; }

    RunResult run_flat(size_t query_limit, size_t k, size_t batch_size, const fs::path& out_dir) {
        query_limit = std::min(query_limit, queries_.n);
        const std::string params = "batch_size=" + std::to_string(batch_size) +
                                   ",selection=host_heap,metric=inner_product";
        std::vector<PerQueryRow> per_query;
        std::vector<double> latencies, recalls;
        per_query.reserve(query_limit);
        latencies.reserve(query_limit);
        recalls.reserve(query_limit);
        double total_ms = 0.0, h2d_ms = 0.0, d2h_ms = 0.0, kernel_ms = 0.0, select_ms = 0.0;
        const unsigned block = 256;
        ScopedDevicePtr d_query(cuda_, batch_size * queries_.d * sizeof(float));
        ScopedDevicePtr d_scores(cuda_, batch_size * base_.n * sizeof(float));
        std::vector<float> scores(batch_size * base_.n);
        for (size_t start = 0; start < query_limit; start += batch_size) {
            const size_t actual = std::min(batch_size, query_limit - start);
            const auto batch_begin = Clock::now();
            h2d_ms += cuda_.memcpy_h2d(d_query.ptr, queries_.row(start), actual * queries_.d * sizeof(float));
            const unsigned total = static_cast<unsigned>(actual * base_.n);
            const unsigned grid = (total + block - 1) / block;
            auto base_ptr = d_base_.ptr;
            auto query_ptr = d_query.ptr;
            auto score_ptr = d_scores.ptr;
            auto n = static_cast<unsigned>(base_.n);
            auto d = static_cast<unsigned>(base_.d);
            auto batch = static_cast<unsigned>(actual);
            std::vector<void*> args = {&base_ptr, &query_ptr, &score_ptr, &n, &d, &batch};
            kernel_ms += cuda_.launch(cuda_.flat_func(), grid, block, args);
            d2h_ms += cuda_.memcpy_d2h(scores.data(), d_scores.ptr, actual * base_.n * sizeof(float));
            const auto select_begin = Clock::now();
            std::vector<double> batch_recalls(actual, 0.0);
            for (size_t local = 0; local < actual; ++local) {
                std::priority_queue<std::pair<float, int>> heap;
                const float* row = scores.data() + local * base_.n;
                for (size_t i = 0; i < base_.n; ++i) push_topk(heap, row[i], static_cast<int>(i), k);
                const double recall = heap_recall(heap, gt_, start + local, k);
                recalls.push_back(recall);
                batch_recalls[local] = recall;
            }
            select_ms += elapsed_ms(select_begin, Clock::now());
            const double batch_ms = elapsed_ms(batch_begin, Clock::now());
            total_ms += batch_ms;
            const double latency_us = batch_ms * 1000.0 / static_cast<double>(actual);
            for (size_t local = 0; local < actual; ++local) {
                latencies.push_back(latency_us);
                per_query.push_back({start + local, batch_recalls[local], latency_us, base_.n});
            }
        }
        RunResult result = make_result("gpu_flat_batch", "matrix_baseline", params, query_limit, k,
                                       batch_size, "", "", 0.0, recalls, latencies, total_ms,
                                       kernel_ms, h2d_ms, d2h_ms, select_ms, static_cast<double>(base_.n), 1.0);
        write_run_files(out_dir, result, per_query);
        return result;
    }

    RunResult run_ivf_compact(size_t query_limit,
                              size_t k,
                              size_t batch_size,
                              int nlist,
                              int nprobe,
                              int iters,
                              unsigned seed,
                              const fs::path& out_dir) {
        query_limit = std::min(query_limit, queries_.n);
        IVFIndex index(base_, nlist, iters, seed);
        index.build();
        const std::string params = "nlist=" + std::to_string(nlist) + ",nprobe=" +
                                   std::to_string(nprobe) + ",batch_size=" +
                                   std::to_string(batch_size) + ",layout=compact_pairs,iters=" +
                                   std::to_string(iters);
        std::vector<PerQueryRow> per_query;
        std::vector<double> latencies, recalls;
        std::vector<double> cand_counts;
        per_query.reserve(query_limit);
        latencies.reserve(query_limit);
        recalls.reserve(query_limit);
        double total_ms = 0.0, h2d_ms = 0.0, d2h_ms = 0.0, kernel_ms = 0.0, select_ms = 0.0;
        const unsigned block = 256;
        ScopedDevicePtr d_query(cuda_, batch_size * queries_.d * sizeof(float));
        for (size_t start = 0; start < query_limit; start += batch_size) {
            const size_t actual = std::min(batch_size, query_limit - start);
            std::vector<int> cand_ids;
            std::vector<int> q_ids;
            std::vector<size_t> offsets(actual + 1, 0);
            for (size_t local = 0; local < actual; ++local) {
                const std::vector<int> clusters = index.top_clusters(queries_.row(start + local), nprobe);
                for (int c : clusters) {
                    const auto& list = index.list(c);
                    cand_ids.insert(cand_ids.end(), list.begin(), list.end());
                    q_ids.insert(q_ids.end(), list.size(), static_cast<int>(local));
                }
                offsets[local + 1] = cand_ids.size();
                cand_counts.push_back(static_cast<double>(offsets[local + 1] - offsets[local]));
            }
            std::vector<float> scores(cand_ids.size());
            ScopedDevicePtr d_cands(cuda_, cand_ids.size() * sizeof(int));
            ScopedDevicePtr d_qids(cuda_, q_ids.size() * sizeof(int));
            ScopedDevicePtr d_scores(cuda_, scores.size() * sizeof(float));
            const auto batch_begin = Clock::now();
            h2d_ms += cuda_.memcpy_h2d(d_query.ptr, queries_.row(start), actual * queries_.d * sizeof(float));
            h2d_ms += cuda_.memcpy_h2d(d_cands.ptr, cand_ids.data(), cand_ids.size() * sizeof(int));
            h2d_ms += cuda_.memcpy_h2d(d_qids.ptr, q_ids.data(), q_ids.size() * sizeof(int));
            const unsigned pairs = static_cast<unsigned>(cand_ids.size());
            const unsigned grid = (pairs + block - 1) / block;
            auto base_ptr = d_base_.ptr;
            auto query_ptr = d_query.ptr;
            auto cand_ptr = d_cands.ptr;
            auto qid_ptr = d_qids.ptr;
            auto score_ptr = d_scores.ptr;
            auto pair_count = pairs;
            auto d = static_cast<unsigned>(base_.d);
            std::vector<void*> args = {&base_ptr, &query_ptr, &cand_ptr, &qid_ptr, &score_ptr, &pair_count, &d};
            kernel_ms += cuda_.launch(cuda_.indexed_func(), grid, block, args);
            d2h_ms += cuda_.memcpy_d2h(scores.data(), d_scores.ptr, scores.size() * sizeof(float));
            const auto select_begin = Clock::now();
            std::vector<double> batch_recalls(actual, 0.0);
            for (size_t local = 0; local < actual; ++local) {
                std::priority_queue<std::pair<float, int>> heap;
                for (size_t p = offsets[local]; p < offsets[local + 1]; ++p) {
                    push_topk(heap, scores[p], cand_ids[p], k);
                }
                const double recall = heap_recall(heap, gt_, start + local, k);
                recalls.push_back(recall);
                batch_recalls[local] = recall;
            }
            select_ms += elapsed_ms(select_begin, Clock::now());
            const double batch_ms = elapsed_ms(batch_begin, Clock::now());
            total_ms += batch_ms;
            const double latency_us = batch_ms * 1000.0 / static_cast<double>(actual);
            for (size_t local = 0; local < actual; ++local) {
                latencies.push_back(latency_us);
                per_query.push_back({start + local, batch_recalls[local], latency_us, offsets[local + 1] - offsets[local]});
            }
        }
        const double avg_cands = cand_counts.empty() ? 0.0 : std::accumulate(cand_counts.begin(), cand_counts.end(), 0.0) / cand_counts.size();
        RunResult result = make_result("gpu_ivf_compact", "ivf_pair_compaction", params, query_limit, k,
                                       batch_size, std::to_string(nlist), std::to_string(nprobe),
                                       index.build_ms(), recalls, latencies, total_ms, kernel_ms,
                                       h2d_ms, d2h_ms, select_ms, avg_cands, 1.0);
        write_run_files(out_dir, result, per_query);
        return result;
    }

    RunResult run_ivf_union(size_t query_limit,
                            size_t k,
                            size_t batch_size,
                            int nlist,
                            int nprobe,
                            int iters,
                            unsigned seed,
                            const fs::path& out_dir) {
        query_limit = std::min(query_limit, queries_.n);
        IVFIndex index(base_, nlist, iters, seed);
        index.build();
        const std::string params = "nlist=" + std::to_string(nlist) + ",nprobe=" +
                                   std::to_string(nprobe) + ",batch_size=" +
                                   std::to_string(batch_size) + ",layout=batch_cluster_union,iters=" +
                                   std::to_string(iters);
        std::vector<PerQueryRow> per_query;
        std::vector<double> latencies, recalls;
        std::vector<double> cand_counts;
        per_query.reserve(query_limit);
        latencies.reserve(query_limit);
        recalls.reserve(query_limit);
        double total_ms = 0.0, h2d_ms = 0.0, d2h_ms = 0.0, kernel_ms = 0.0, select_ms = 0.0;
        double computed_pairs = 0.0, useful_pairs = 0.0;
        const unsigned block = 256;
        ScopedDevicePtr d_query(cuda_, batch_size * queries_.d * sizeof(float));
        for (size_t start = 0; start < query_limit; start += batch_size) {
            const size_t actual = std::min(batch_size, query_limit - start);
            std::vector<std::vector<int>> selected(actual);
            std::set<int> cluster_union;
            for (size_t local = 0; local < actual; ++local) {
                selected[local] = index.top_clusters(queries_.row(start + local), nprobe);
                cluster_union.insert(selected[local].begin(), selected[local].end());
            }
            std::vector<std::priority_queue<std::pair<float, int>>> heaps(actual);
            std::vector<size_t> per_q_cands(actual, 0);
            const auto batch_begin = Clock::now();
            h2d_ms += cuda_.memcpy_h2d(d_query.ptr, queries_.row(start), actual * queries_.d * sizeof(float));
            for (int c : cluster_union) {
                const auto& list = index.list(c);
                if (list.empty()) continue;
                std::vector<float> scores(actual * list.size());
                ScopedDevicePtr d_cands(cuda_, list.size() * sizeof(int));
                ScopedDevicePtr d_scores(cuda_, scores.size() * sizeof(float));
                h2d_ms += cuda_.memcpy_h2d(d_cands.ptr, list.data(), list.size() * sizeof(int));
                const unsigned total = static_cast<unsigned>(actual * list.size());
                computed_pairs += static_cast<double>(total);
                const unsigned grid = (total + block - 1) / block;
                auto base_ptr = d_base_.ptr;
                auto query_ptr = d_query.ptr;
                auto cand_ptr = d_cands.ptr;
                auto score_ptr = d_scores.ptr;
                auto cand_count = static_cast<unsigned>(list.size());
                auto d = static_cast<unsigned>(base_.d);
                auto batch = static_cast<unsigned>(actual);
                std::vector<void*> args = {&base_ptr, &query_ptr, &cand_ptr, &score_ptr, &cand_count, &d, &batch};
                kernel_ms += cuda_.launch(cuda_.cluster_func(), grid, block, args);
                d2h_ms += cuda_.memcpy_d2h(scores.data(), d_scores.ptr, scores.size() * sizeof(float));
                const auto select_begin = Clock::now();
                for (size_t local = 0; local < actual; ++local) {
                    if (std::find(selected[local].begin(), selected[local].end(), c) == selected[local].end()) continue;
                    useful_pairs += static_cast<double>(list.size());
                    per_q_cands[local] += list.size();
                    const float* row = scores.data() + local * list.size();
                    for (size_t j = 0; j < list.size(); ++j) push_topk(heaps[local], row[j], list[j], k);
                }
                select_ms += elapsed_ms(select_begin, Clock::now());
            }
            const double batch_ms = elapsed_ms(batch_begin, Clock::now());
            total_ms += batch_ms;
            const double latency_us = batch_ms * 1000.0 / static_cast<double>(actual);
            for (size_t local = 0; local < actual; ++local) {
                const double recall = heap_recall(heaps[local], gt_, start + local, k);
                recalls.push_back(recall);
                latencies.push_back(latency_us);
                cand_counts.push_back(static_cast<double>(per_q_cands[local]));
                per_query.push_back({start + local, recall, latency_us, per_q_cands[local]});
            }
        }
        const double avg_cands = cand_counts.empty() ? 0.0 : std::accumulate(cand_counts.begin(), cand_counts.end(), 0.0) / cand_counts.size();
        const double waste = useful_pairs > 0.0 ? computed_pairs / useful_pairs : 0.0;
        RunResult result = make_result("gpu_ivf_union", "ivf_cluster_matrix_baseline", params, query_limit, k,
                                       batch_size, std::to_string(nlist), std::to_string(nprobe),
                                       index.build_ms(), recalls, latencies, total_ms, kernel_ms,
                                       h2d_ms, d2h_ms, select_ms, avg_cands, waste);
        write_run_files(out_dir, result, per_query);
        return result;
    }

private:
    const Matrix<float>& base_;
    const Matrix<float>& queries_;
    const Matrix<int>& gt_;
    CudaDriver cuda_;
    ScopedDevicePtr d_base_;
    double base_upload_ms_ = 0.0;
};

struct Options {
    std::string base_path = "C:/Users/Administrator/Downloads/DEEP100K.base.100k.fbin";
    std::string query_path = "C:/Users/Administrator/Downloads/DEEP100K.query.fbin";
    std::string gt_path = "C:/Users/Administrator/Downloads/DEEP100K.gt.query.100k.top100.bin";
    fs::path out_dir = "data/gpu_experiments";
    unsigned seed = 20260616;
    bool clean = false;
    bool quick = false;
};

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const std::string& key) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            return argv[++i];
        };
        if (arg == "--base") opt.base_path = need(arg);
        else if (arg == "--query") opt.query_path = need(arg);
        else if (arg == "--gt") opt.gt_path = need(arg);
        else if (arg == "--out") opt.out_dir = need(arg);
        else if (arg == "--seed") opt.seed = static_cast<unsigned>(std::stoul(need(arg)));
        else if (arg == "--clean") opt.clean = true;
        else if (arg == "--quick") opt.quick = true;
        else if (arg == "--help") {
            std::cout << "Usage: gpu_ann_main [--base path] [--query path] [--gt path] [--out dir] [--clean] [--quick]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return opt;
}

static void write_dataset_manifest(const fs::path& out_dir,
                                   const Matrix<float>& base,
                                   const Matrix<float>& queries,
                                   const Matrix<int>& gt,
                                   const std::map<std::string, std::string>& device,
                                   double base_upload_ms,
                                   const Options& opt) {
    const fs::path dataset_dir = out_dir / "dataset";
    fs::create_directories(dataset_dir);
    for (const auto& p : {fs::path(opt.base_path), fs::path(opt.query_path), fs::path(opt.gt_path)}) {
        const fs::path dst = dataset_dir / p.filename();
        if (!fs::exists(dst) || fs::file_size(dst) != fs::file_size(p)) {
            fs::copy_file(p, dst, fs::copy_options::overwrite_existing);
        }
    }
    std::ofstream out(dataset_dir / "manifest.txt");
    out << "base_n=" << base.n << "\n";
    out << "base_dim=" << base.d << "\n";
    out << "query_n=" << queries.n << "\n";
    out << "query_dim=" << queries.d << "\n";
    out << "gt_n=" << gt.n << "\n";
    out << "gt_dim=" << gt.d << "\n";
    out << "metric=1-inner_product\n";
    out << "base_upload_ms=" << fmt(base_upload_ms) << "\n";
    out << "runner=gpu_ann_main.cpp\n";
    for (const auto& kv : device) out << kv.first << "=" << kv.second << "\n";
}

static void write_grouped_csvs(const fs::path& out_dir, const std::vector<RunResult>& results) {
    write_summary_csv(out_dir / "summary" / "gpu_results.csv", results);
    std::map<std::string, std::vector<RunResult>> grouped;
    for (const auto& r : results) grouped[r.algorithm].push_back(r);
    for (const auto& kv : grouped) {
        write_summary_csv(out_dir / "algorithm_csv" / (kv.first + ".csv"), kv.second);
    }
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        if (opt.clean && fs::exists(opt.out_dir)) {
            const std::string out = fs::absolute(opt.out_dir).string();
            if (out.find("gpu_experiments") == std::string::npos) {
                throw std::runtime_error("refuse to clean unexpected directory: " + out);
            }
            fs::remove_all(opt.out_dir);
        }
        fs::create_directories(opt.out_dir);

        std::cerr << "loading base: " << opt.base_path << "\n";
        Matrix<float> base = load_matrix<float>(opt.base_path);
        std::cerr << "loading query: " << opt.query_path << "\n";
        Matrix<float> queries = load_matrix<float>(opt.query_path);
        std::cerr << "loading gt: " << opt.gt_path << "\n";
        Matrix<int> gt = load_matrix<int>(opt.gt_path);
        if (base.d != queries.d) throw std::runtime_error("base/query dimension mismatch");
        if (queries.n != gt.n) throw std::runtime_error("query/gt count mismatch");

        GpuExperiment exp(base, queries, gt);
        const auto device = exp.cuda().device_info();
        std::cout << "GPU: " << device.at("gpu") << " cc=" << device.at("compute_capability")
                  << " driver=" << device.at("cuda_driver_version") << "\n";
        std::cout << "Base upload: " << fmt(exp.base_upload_ms()) << " ms\n";
        write_dataset_manifest(opt.out_dir, base, queries, gt, device, exp.base_upload_ms(), opt);

        std::vector<RunResult> results;
        auto save = [&]() { write_grouped_csvs(opt.out_dir, results); };

        if (opt.quick) {
            std::cout << "flat q=2000 k=10 batch=64\n";
            results.push_back(exp.run_flat(2000, 10, 64, opt.out_dir)); save();
            std::cout << "ivf_compact q=2000 k=10 nlist=64 nprobe=8 batch=32\n";
            results.push_back(exp.run_ivf_compact(2000, 10, 32, 64, 8, 4, opt.seed + 64, opt.out_dir)); save();
            std::cout << "ivf_union q=2000 k=10 nlist=64 nprobe=8 batch=16\n";
            results.push_back(exp.run_ivf_union(2000, 10, 16, 64, 8, 4, opt.seed + 64, opt.out_dir)); save();
        } else {
            const std::vector<std::tuple<size_t, size_t, size_t>> flat_configs = {
                {2000, 10, 32}, {2000, 10, 64}, {2000, 10, 128}, {4000, 10, 64},
                {2000, 1, 64}, {2000, 20, 64}, {2000, 50, 64}
            };
            for (const auto& [q, k, batch] : flat_configs) {
                std::cout << "flat q=" << q << " k=" << k << " batch=" << batch << "\n";
                results.push_back(exp.run_flat(q, k, batch, opt.out_dir)); save();
            }

            struct IVFConfig { size_t q; size_t k; int nlist; int nprobe; size_t batch; };
            const std::vector<IVFConfig> compact_configs = {
                {2000, 10, 32, 4, 32}, {2000, 10, 64, 8, 32}, {2000, 10, 64, 16, 32},
                {2000, 10, 128, 8, 32}, {4000, 10, 64, 8, 32}, {4000, 10, 128, 8, 32}
            };
            for (const auto& c : compact_configs) {
                std::cout << "ivf_compact q=" << c.q << " k=" << c.k << " nlist=" << c.nlist
                          << " nprobe=" << c.nprobe << " batch=" << c.batch << "\n";
                results.push_back(exp.run_ivf_compact(c.q, c.k, c.batch, c.nlist, c.nprobe, 4,
                                                      opt.seed + static_cast<unsigned>(c.nlist), opt.out_dir));
                save();
            }

            const std::vector<IVFConfig> union_configs = {
                {2000, 10, 64, 4, 16}, {2000, 10, 64, 8, 16}, {2000, 10, 128, 8, 16}
            };
            for (const auto& c : union_configs) {
                std::cout << "ivf_union q=" << c.q << " k=" << c.k << " nlist=" << c.nlist
                          << " nprobe=" << c.nprobe << " batch=" << c.batch << "\n";
                results.push_back(exp.run_ivf_union(c.q, c.k, c.batch, c.nlist, c.nprobe, 4,
                                                    opt.seed + static_cast<unsigned>(c.nlist), opt.out_dir));
                save();
            }
        }

        save();
        std::cout << "summary: " << fs::absolute(opt.out_dir / "summary" / "gpu_results.csv").string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
