#ifndef REFERENCE_H
#define REFERENCE_H

#include <vector>
#include <cmath>
#include <cstdio>

// The host reference is always computed in float, independent of the device
// precision. Callers convert the device output back to float before verifying.

// Column-major reference for a group of GEMMs (no transpose):
//   C[g] = alpha * A[g] * B[g] + beta * C[g]
// where A[g] is (m x k), B[g] is (k x n), C[g] is (m x n).
// Leading dimensions: lda = m, ldb = k, ldc = m.
inline void groupedGemm_ref(const std::vector<int> &m,
                            const std::vector<int> &n,
                            const std::vector<int> &k,
                            float alpha, float beta,
                            const std::vector<std::vector<float>> &A,
                            const std::vector<std::vector<float>> &B,
                            std::vector<std::vector<float>> &C)
{
  const int groups = (int)m.size();
  for (int g = 0; g < groups; g++) {
    const int M = m[g], N = n[g], K = k[g];
    const float *Ag = A[g].data();
    const float *Bg = B[g].data();
    float *Cg = C[g].data();
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
        float acc = 0;
        for (int p = 0; p < K; p++)
          acc += Ag[i + (size_t)p * M] * Bg[p + (size_t)j * K];
        Cg[i + (size_t)j * M] = alpha * acc + beta * Cg[i + (size_t)j * M];
      }
    }
  }
}

// Compare and report the maximum absolute error over all groups.
inline bool verify(const std::vector<std::vector<float>> &test,
                   const std::vector<std::vector<float>> &ref,
                   double tol)
{
  double max_err = 0.0;
  for (size_t g = 0; g < ref.size(); g++)
    for (size_t i = 0; i < ref[g].size(); i++)
      max_err = std::fmax(max_err, std::fabs((double)test[g][i] - (double)ref[g][i]));
  bool ok = max_err <= tol;
  printf("Maximum absolute error: %e (tolerance %e) -> %s\n",
         max_err, tol, ok ? "PASS" : "FAIL");
  return ok;
}

// Report aggregate throughput for a group of GEMMs.
inline void performance(const std::vector<int> &m,
                        const std::vector<int> &n,
                        const std::vector<int> &k,
                        double avg_time_us)
{
  double total_ops = 0.0;
  for (size_t g = 0; g < m.size(); g++)
    total_ops += 2.0 * m[g] * n[g] * k[g];
  double perf = total_ops / (avg_time_us * 1e3); // GFLOP/s
  const char *scale = "G";
  if (perf >= 1000) { perf /= 1000; scale = "T"; }
  printf("Average execution time: %.3f (us) | performance: %.2f %sFLOP/s\n",
         avg_time_us, perf, scale);
}

#endif
