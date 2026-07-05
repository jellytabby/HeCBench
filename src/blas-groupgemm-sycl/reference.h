#ifndef REFERENCE_H
#define REFERENCE_H

#include <vector>
#include <cmath>
#include <cstdio>

// The host reference is always computed in float, independent of the device
// precision. Callers convert the device output back to float before verifying.

// Row-major reference for a grouped (variable-M) matmul, matching the oneDNN
// grouped-memory MoE layout:
//   src : [total_M, K]        (rows partitioned into groups by groupM)
//   W   : [num_groups, K, N]  (one K x N weight matrix per group)
//   dst : [total_M, N]
// For every group g and every one of its rows m:
//   dst[m, n] = sum_k src[m, k] * W[g, k, n]
inline void groupedGemm_ref(const std::vector<int> &groupM, int K, int N,
                            const std::vector<float> &src,
                            const std::vector<float> &W,
                            std::vector<float> &dst)
{
  size_t rowBase = 0;
  for (size_t g = 0; g < groupM.size(); g++) {
    const float *Wg = W.data() + g * (size_t)K * N;
    for (int m = 0; m < groupM[g]; m++) {
      const float *srow = src.data() + (rowBase + m) * (size_t)K;
      float *drow = dst.data() + (rowBase + m) * (size_t)N;
      for (int n = 0; n < N; n++) {
        float acc = 0;
        for (int kk = 0; kk < K; kk++)
          acc += srow[kk] * Wg[(size_t)kk * N + n];
        drow[n] = acc;
      }
    }
    rowBase += groupM[g];
  }
}

// Compare and report the maximum absolute error over all rows.
inline bool verify(const std::vector<float> &test,
                   const std::vector<float> &ref,
                   double relTol)
{
  double max_err = 0.0, max_ref = 0.0;
  for (size_t i = 0; i < ref.size(); i++) {
    max_err = std::fmax(max_err, std::fabs((double)test[i] - (double)ref[i]));
    max_ref = std::fmax(max_ref, std::fabs((double)ref[i]));
  }
  bool ok = max_err <= relTol * (max_ref + 1.0);
  printf("Maximum absolute error: %e (tolerance %e) -> %s\n",
         max_err, relTol * (max_ref + 1.0), ok ? "PASS" : "FAIL");
  return ok;
}

// Report aggregate throughput for a grouped matmul.
inline void performance(const std::vector<int> &groupM, int K, int N,
                        double avg_time_us)
{
  double total_ops = 0.0;
  for (size_t g = 0; g < groupM.size(); g++)
    total_ops += 2.0 * groupM[g] * K * N;
  double perf = total_ops / (avg_time_us * 1e3); // GFLOP/s
  const char *scale = "G";
  if (perf >= 1000) { perf /= 1000; scale = "T"; }
  printf("Average execution time: %.3f (us) | performance: %.2f %sFLOP/s\n",
         avg_time_us, perf, scale);
}

#endif
