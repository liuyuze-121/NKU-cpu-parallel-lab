#include <iostream>
#include <windows.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace std;


struct Result {
    int N;
    double time_adv_four_way;  // 进阶算法1：四路循环展开并行
    double time_adv_recursive;  // 进阶算法2：递归两两归并
};

double* a;   
double* sum; 

// 初始化数据
void init_data(int N) {
    for (int i = 0; i < N; i++) {
        a[i] = 1.0; // 固定值
    }
}

// ========== 进阶优化算法1：四路循环展开并行 ==========
void four_way_unroll(int N) {
    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    int i = 0;
 
    for (; i <= N - 4; i += 4) {
        sum1 += a[i];
        sum2 += a[i+1];
        sum3 += a[i+2];
        sum4 += a[i+3];
    }

    for (; i < N; i++) {
        sum1 += a[i];
    }

    sum[0] = sum1 + sum2 + sum3 + sum4;
}
// ========== 进阶优化算法2：递归两两归并 ==========
void recursive_pairwise(int N) {
    
    double* temp = new double[N];
    for (int i = 0; i < N; i++) {
        temp[i] = a[i];
    }

    // 原地归并（合并辅助函数）
    auto merge_helper = [&](auto&& self, double* arr, int n) {
        if (n <= 1) return;
        int half = n / 2;

        for (int i = 0; i < half; i++) {
            arr[i] += arr[n - i - 1];
        }

        self(self, arr, half);
    };

    merge_helper(merge_helper, temp, N);

    sum[0] = temp[0];
    delete[] temp;
}

// Windows高精度计时
double measure_time(void (*func)(int), int N) {
    LARGE_INTEGER freq, begin, end;
    QueryPerformanceFrequency(&freq);
    int repeat = 1000;
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
    file << "N,FourWayUnroll_ms,RecursivePairwise_ms" << endl;
    for (const auto& r : results) {
        file << r.N << ","
             << fixed << setprecision(4) << r.time_adv_four_way << ","
             << r.time_adv_recursive << endl;
    }
    file.close();
    
}

int main() {
    const int MAXN = 2048;

    a = new double[MAXN];
    sum = new double[1];

    vector<int> N_values;
    for (int n = 128; n <= 2048; n += 128) {
        N_values.push_back(n);
    }
    vector<Result> results;

    for (int N : N_values) {
        init_data(N);

        double time_four_way = measure_time(four_way_unroll, N);
        double time_recursive = measure_time(recursive_pairwise, N);

        results.push_back({N, time_four_way, time_recursive});

        cout << "N = " << setw(8) << N
             << " | 四路展开: " << fixed << setprecision(4) << time_four_way << " ms"
             << " | 递归归并: " << time_recursive << " ms" << endl;
    }

    save_csv(results, "sum_advanced_benchmark.csv");

    delete[] a;
    delete[] sum;

    return 0;
}