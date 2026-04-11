#include <iostream>
#include <windows.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace std;

struct SumResult {
    int N;
    double time_chain;    // 平凡链式算法耗时
    double time_two_way;  // 两路超标量优化耗时
    double speedup;
};

double* a; // 输入数组

// 初始化数据
void init_data(int N) {
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<double>(i); 
    }
}

// 算法1：平凡算法 - 逐个链式累加（串行，有数据依赖）
void sum_chain(int N) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }
    volatile double result = sum; 
}

// 算法2：超标量优化 - 两路链式累加（指令级并行）
void sum_two_way(int N) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    // 步长为2，两条链独立计算，无依赖
    for (int i = 0; i < N; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    volatile double result = sum1 + sum2;
}

// 高精度计时
double measure_time_sum(void (*func)(int), int N) {
    LARGE_INTEGER freq, begin, end;
    QueryPerformanceFrequency(&freq);
    

    int repeat = 10000; 
    
    QueryPerformanceCounter(&begin);
    for (int i = 0; i < repeat; i++) {
        func(N);
    }
    QueryPerformanceCounter(&end);
    
    return (end.QuadPart - begin.QuadPart) * 1000.0 / freq.QuadPart / repeat;
}

// 保存结果为CSV
void save_csv_sum(const vector<SumResult>& results, const string& filename) {
    ofstream file(filename);
    file << "N,Chain_ms,TwoWay_ms,Speedup" << endl;
    for (const auto& r : results) {
        file << r.N << "," << fixed << setprecision(6) << r.time_chain << ","
             << r.time_two_way << "," << fixed << setprecision(2) << r.speedup << endl;
    }
    file.close();
    cout << "结果已保存到: " << filename << endl;
}

int main() {
    const int MAXN = 2048;

    a = new double[MAXN];

    vector<int> N_values;
    for (int n = 128; n <= 2048; n += 128) {
        N_values.push_back(n);
    }
    vector<SumResult> results;


    for (int N : N_values) {
        init_data(N);
        double time_chain = measure_time_sum(sum_chain, N);
        double time_two_way = measure_time_sum(sum_two_way, N);
        double speedup = time_chain / time_two_way;

        results.push_back({N, time_chain, time_two_way, speedup});

        cout << "N = " << setw(8) << N
             << " | 链式: " << fixed << setprecision(6) << time_chain << " ms"
             << " | 两路: " << time_two_way << " ms"
             << " | 加速比: " << fixed << setprecision(2) << speedup << "x" << endl;
    }

    save_csv_sum(results, "sum_n_benchmark.csv");

    delete[] a;

    return 0;
}