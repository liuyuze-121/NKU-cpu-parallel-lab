#include <iostream>
#include <windows.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace std;

struct Result {
    int N;
    double time_naive;
    double time_opt;
    double speedup;
};

double** b;  // N×N矩阵
double* a;   // 输入向量
double* sum; // 结果向量

void init_data(int N) {
    for (int i = 0; i < N; i++) {
        a[i] = i;
        for (int j = 0; j < N; j++) {
            b[i][j] = i + j;
        }
    }
}

// 平凡算法：按列访问
void col_major(int N) {
    for (int i = 0; i < N; i++) {
        sum[i] = 0.0;
        for (int j = 0; j < N; j++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}

// Cache优化算法：按行访问
void row_major(int N) {
    for (int i = 0; i < N; i++) {
        sum[i] = 0.0;
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}

// 高精度计时
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
    file << "N,Naive_ms,Optimized_ms,Speedup" << endl;
    for (const auto& r : results) {
        file << r.N << "," << fixed << setprecision(4) << r.time_naive << ","
             << r.time_opt << "," << fixed << setprecision(2) << r.speedup << endl;
    }
    file.close();
    cout << "结果已保存到: " << filename << endl;
}

int main() {
    const int MAXN = 2048;

    // 动态分配内存
    b = new double*[MAXN];
    for (int i = 0; i < MAXN; i++) {
        b[i] = new double[MAXN];
    }
    a = new double[MAXN];
    sum = new double[MAXN];

    // 测试的不同矩阵大小
    vector<int> N_values;
    for (int n = 128; n <= 2048; n += 128) {
        N_values.push_back(n);
    }
    vector<Result> results;

    // 遍历不同N值进行测试
    for (int N : N_values) {
        init_data(N);
        double time_naive = measure_time(col_major, N);
        double time_opt = measure_time(row_major, N);
        double speedup = time_naive / time_opt;

        results.push_back({N, time_naive, time_opt, speedup});

        cout << "N = " << setw(5) << N
             << " | 平凡算法: " << fixed << setprecision(4) << time_naive << " ms"
             << " | Cache优化: " << time_opt << " ms"
             << " | 加速比: " << fixed << setprecision(2) << speedup << "x" << endl;
    }

    save_csv(results, "matrix_inner_benchmark.csv");

    // 释放内存
    for (int i = 0; i < MAXN; i++) {
        delete[] b[i];
    }
    delete[] b;
    delete[] a;
    delete[] sum;

    return 0;
}