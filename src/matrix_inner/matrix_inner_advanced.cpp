#include <iostream>
#include <windows.h>
#include <iomanip>
#include <fstream>
#include <vector>
#include <emmintrin.h>

using namespace std;

// 进阶实验结果结构体
struct Result {
    int N;
    double time_block_unroll; // 进阶算法1：循环分块+循环展开
    double time_simd;         // 进阶算法2：SSE SIMD向量化
};

// 全局数组定义
double** b;  
double* a;   
double* sum;

// 初始化数据
void init_data(int N) {
    for (int i = 0; i < N; i++) {
        a[i] = i;
        for (int j = 0; j < N; j++) {
            b[i][j] = i + j;
        }
    }
}

// ========== 进阶优化算法1：循环分块+4路循环展开 ==========
const int BLOCK_SIZE = 128;
void block_unroll(int N) {
    for (int i = 0; i < N; i++) {
        sum[i] = 0.0;
    }

    for (int j_block = 0; j_block < N; j_block += BLOCK_SIZE) {
        int j_end = (j_block + BLOCK_SIZE < N) ? (j_block + BLOCK_SIZE) : N;

        for (int j = j_block; j < j_end; j++) {
            double vec_val = a[j];
            int i = 0;
    
            for (; i <= N - 4; i += 4) {
                sum[i] += b[j][i] * vec_val;
                sum[i+1] += b[j][i+1] * vec_val;
                sum[i+2] += b[j][i+2] * vec_val;
                sum[i+3] += b[j][i+3] * vec_val;
            }

            for (; i < N; i++) {
                sum[i] += b[j][i] * vec_val;
            }
        }
    }
}

// ========== 进阶优化算法2：SSE SIMD向量化 ==========
void simd_vectorize(int N) {
    for (int i = 0; i < N; i++) {
        sum[i] = 0.0;
    }
    for (int j = 0; j < N; j++) {
        __m128d vec_val = _mm_set1_pd(a[j]);
        int i = 0;

        for (; i <= N - 2; i += 2) {
            __m128d mat_val = _mm_loadu_pd(&b[j][i]);
            __m128d prod = _mm_mul_pd(mat_val, vec_val);
            __m128d sum_val = _mm_loadu_pd(&sum[i]);
            sum_val = _mm_add_pd(sum_val, prod);
            _mm_storeu_pd(&sum[i], sum_val);
        }

        for (; i < N; i++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}

// Windows高精度计时
double measure_time(void (*func)(int), int N) {
    LARGE_INTEGER freq, begin, end;
    QueryPerformanceFrequency(&freq);
    int repeat = 100;
    QueryPerformanceCounter(&begin);
    for (int i = 0; i < repeat; i++) {
        func(N);
    }
    QueryPerformanceCounter(&end);
    return (end.QuadPart - begin.QuadPart) * 1000.0 / freq.QuadPart / repeat;
}

// 保存结果为CSV
void save_csv(const vector<Result>& results, const string& filename) {
    ofstream file(filename);
    file << "N,BlockUnroll_ms,SIMDVectorize_ms" << endl;
    for (const auto& r : results) {
        file << r.N << ","
             << fixed << setprecision(4) << r.time_block_unroll << ","
             << r.time_simd << endl;
    }
    file.close();
    cout << "进阶实验结果已保存到: " << filename << endl;
}

int main() {
    const int MAXN = 2048;


    b = new double*[MAXN];
    for (int i = 0; i < MAXN; i++) {
        b[i] = new double[MAXN];
    }
    a = new double[MAXN];
    sum = new double[MAXN];


    vector<int> N_values;
    for (int n = 128; n <= 2048; n += 128) {
        N_values.push_back(n);
    }
    vector<Result> results;



    for (int N : N_values) {
        init_data(N);


        double time_block = measure_time(block_unroll, N);
        double time_simd = measure_time(simd_vectorize, N);

        results.push_back({N, time_block, time_simd});


        cout << "N = " << setw(5) << N
             << " | 分块展开: " << fixed << setprecision(4) << time_block << " ms"
             << " | SIMD向量化: " << time_simd << " ms" << endl;
    }

    // 保存结果
    save_csv(results, "matrix_advanced_benchmark.csv");


    for (int i = 0; i < MAXN; i++) {
        delete[] b[i];
    }
    delete[] b;
    delete[] a;
    delete[] sum;

    return 0;
}