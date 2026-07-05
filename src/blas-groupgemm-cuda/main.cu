// Grouped GEMM using the cuBLAS grouped-batched API (cublasGemmGroupedBatchedEx,
// available since CUDA 12.5).
//
// A grouped GEMM launches a group of independent GEMMs, each of which may have
// its own M, N, K sizes.  This mirrors the Triton "Group GEMM" tutorial
// (https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
// but relies on the vendor BLAS library to schedule the whole group in a single
// call instead of a hand-written persistent kernel.
//
// Real-world LLM use case (see main()): the Mixture-of-Experts (MoE) feed-forward
// layer used by Mixtral, DeepSeek-MoE, Qwen-MoE, GPT-OSS, etc.  A router assigns
// each token to one of E experts, so every expert processes a *different* number
// of tokens (varying M) while sharing the *same* weight shape (fixed N, K).  The
// fixed-size batched GEMM API cannot express this; grouped GEMM can, in a single
// launch.
//
// All matrices are stored column-major (cuBLAS default). For each group g:
//   C[g] = alpha * A[g] * B[g] + beta * C[g]
// with A[g] : m[g] x k[g], B[g] : k[g] x n[g], C[g] : m[g] x n[g].

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "reference.h"

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                            \
              cudaGetErrorString(err), __FILE__, __LINE__);                  \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t st = (call);                                             \
    if (st != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", st, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// Host-side conversions between float and the device element type.
template <typename T> static inline T   fromFloat(float x) { return (T)x; }
template <> inline __half                fromFloat<__half>(float x) { return __float2half(x); }
template <typename T> static inline float toFloat(T x) { return (float)x; }
template <> inline float                 toFloat<__half>(__half x) { return __half2float(x); }

template <typename T>
void groupedGemm(const char *name, cudaDataType_t dataType,
                 const std::vector<int> &m,
                 const std::vector<int> &n,
                 const std::vector<int> &k,
                 int repeat)
{
  printf(">>>>>>>>>>>>>>> %s grouped GEMM >>>>>>>>>>>>>>>\n", name);

  const int groups = (int)m.size();

  // Scalars use the compute-type precision: double for FP64, float otherwise.
  using Scalar = std::conditional_t<std::is_same_v<T, double>, double, float>;
  const cublasComputeType_t computeType =
      std::is_same_v<T, double> ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
  std::vector<Scalar> alpha_arr(groups, (Scalar)1);
  std::vector<Scalar> beta_arr(groups, (Scalar)0);

  // Reference inputs are generated in float; the device copies are converted.
  std::vector<std::vector<float>> hAf(groups), hBf(groups), hReff(groups);
  std::vector<std::vector<T>> hA(groups), hB(groups);
  std::vector<std::vector<float>> hCf(groups);
  srand48(123);
  for (int g = 0; g < groups; g++) {
    hAf[g].resize((size_t)m[g] * k[g]);
    hBf[g].resize((size_t)k[g] * n[g]);
    hReff[g].assign((size_t)m[g] * n[g], 0.f);
    hCf[g].resize((size_t)m[g] * n[g]);
    hA[g].resize(hAf[g].size());
    hB[g].resize(hBf[g].size());
    for (size_t i = 0; i < hAf[g].size(); i++) { hAf[g][i] = (float)drand48(); hA[g][i] = fromFloat<T>(hAf[g][i]); }
    for (size_t i = 0; i < hBf[g].size(); i++) { hBf[g][i] = (float)drand48(); hB[g][i] = fromFloat<T>(hBf[g][i]); }
  }

  // Device buffers, one allocation per matrix.
  std::vector<T*> dA(groups), dB(groups), dC(groups);
  for (int g = 0; g < groups; g++) {
    CHECK_CUDA(cudaMalloc(&dA[g], hA[g].size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&dB[g], hB[g].size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&dC[g], (size_t)m[g] * n[g] * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(dA[g], hA[g].data(), hA[g].size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB[g], hB[g].data(), hB[g].size() * sizeof(T), cudaMemcpyHostToDevice));
  }

  // The grouped-batched API takes the size/scale/transpose arrays on the host,
  // while the arrays of matrix pointers (one entry per matrix) must live in
  // device memory, just like the classic batched API.
  std::vector<cublasOperation_t> transa(groups, CUBLAS_OP_N);
  std::vector<cublasOperation_t> transb(groups, CUBLAS_OP_N);
  std::vector<int> lda(groups), ldb(groups), ldc(groups), group_size(groups, 1);
  std::vector<const void*> hAarray(groups), hBarray(groups);
  std::vector<void*> hCarray(groups);
  for (int g = 0; g < groups; g++) {
    lda[g] = m[g];
    ldb[g] = k[g];
    ldc[g] = m[g];
    hAarray[g] = dA[g];
    hBarray[g] = dB[g];
    hCarray[g] = dC[g];
  }

  const void **Aarray; const void **Barray; void **Carray;
  CHECK_CUDA(cudaMalloc(&Aarray, groups * sizeof(void*)));
  CHECK_CUDA(cudaMalloc(&Barray, groups * sizeof(void*)));
  CHECK_CUDA(cudaMalloc(&Carray, groups * sizeof(void*)));
  CHECK_CUDA(cudaMemcpy(Aarray, hAarray.data(), groups * sizeof(void*), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(Barray, hBarray.data(), groups * sizeof(void*), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(Carray, hCarray.data(), groups * sizeof(void*), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  auto launch = [&]() {
    return cublasGemmGroupedBatchedEx(handle, transa.data(), transb.data(),
                                      m.data(), n.data(), k.data(),
                                      alpha_arr.data(),
                                      Aarray, dataType, lda.data(),
                                      Barray, dataType, ldb.data(),
                                      beta_arr.data(),
                                      Carray, dataType, ldc.data(),
                                      groups, group_size.data(),
                                      computeType);
  };

  // Warmup and verify
  for (int r = 0; r < 30; r++)
    CHECK_CUBLAS(launch());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<std::vector<T>> hC(groups);
  for (int g = 0; g < groups; g++) {
    hC[g].resize((size_t)m[g] * n[g]);
    CHECK_CUDA(cudaMemcpy(hC[g].data(), dC[g], hC[g].size() * sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < hC[g].size(); i++) hCf[g][i] = toFloat<T>(hC[g][i]);
  }

  groupedGemm_ref(m, n, k, 1.f, 0.f, hAf, hBf, hReff);
  double tol = std::is_same_v<T, __half> ? 1.0 : (std::is_same_v<T, float> ? 1e-1 : 1e-3);
  verify(hCf, hReff, tol);

  // Benchmarking
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    CHECK_CUBLAS(launch());
  CHECK_CUDA(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  double avg_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                  * 1e-3 / repeat;
  performance(m, n, k, avg_us);

  for (int g = 0; g < groups; g++) {
    CHECK_CUDA(cudaFree(dA[g]));
    CHECK_CUDA(cudaFree(dB[g]));
    CHECK_CUDA(cudaFree(dC[g]));
  }
  CHECK_CUDA(cudaFree(Aarray));
  CHECK_CUDA(cudaFree(Barray));
  CHECK_CUDA(cudaFree(Carray));
  CHECK_CUBLAS(cublasDestroy(handle));
}

int main(int argc, char *argv[])
{
  // Usage: main [repeat] [num_experts] [hidden(K)] [inter(N)] [avg_tokens]
  const int repeat = (argc > 1) ? atoi(argv[1]) : 100;
  const int num_experts = (argc > 2) ? atoi(argv[2]) : 64;  // routed experts (8 Mixtral, 64 DeepSeek-V2)
  const int hidden      = (argc > 3) ? atoi(argv[3]) : 2048; // model hidden size      -> K
  const int inter       = (argc > 4) ? atoi(argv[4]) : 2048; // FFN intermediate size  -> N
  const int avg_tokens  = (argc > 5) ? atoi(argv[5]) : 16;  // avg tokens/expert (batch*seq*top_k / E)

  // Draw a skewed per-expert probability, then assign
  // num_tokens tokens accordingly so the load is imbalanced
  // Every expert is clamped to at least one token
  const long num_tokens = (long)num_experts * avg_tokens;
  srand48(123);
  std::vector<double> cdf(num_experts);
  double wsum = 0;
  for (int e = 0; e < num_experts; e++) { wsum += 0.2 + drand48(); cdf[e] = wsum; }

  std::vector<int> m(num_experts, 0), n(num_experts, inter), k(num_experts, hidden);
  for (long t = 0; t < num_tokens; t++) {
    double r = drand48() * wsum;
    int e = 0;
    while (e < num_experts - 1 && r > cdf[e]) e++;
    m[e]++;
  }
  for (int e = 0; e < num_experts; e++) if (m[e] == 0) m[e] = 1;  // no empty groups

  printf("MoE FFN grouped GEMM: %d experts, hidden(K)=%d, intermediate(N)=%d, repeat=%d\n",
         num_experts, hidden, inter, repeat);
#ifdef DEBUG
  int total_tokens = 0;
  for (int e = 0; e < num_experts; e++) {
    printf("  expert %d: tokens(M)=%d\n", e, m[e]);
    total_tokens += m[e];
  }
  printf("  total routed tokens = %d\n", total_tokens);
#endif

  groupedGemm<__half>("Half precision",   CUDA_R_16F, m, n, k, repeat);
  //groupedGemm<float> ("Single precision", CUDA_R_32F, m, n, k, repeat);
  //groupedGemm<double>("Double precision", CUDA_R_64F, m, n, k, repeat);

  return 0;
}
