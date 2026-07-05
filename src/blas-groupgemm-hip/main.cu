// Grouped GEMM using the hipBLASLt grouped-gemm extension
// (hipblaslt_ext::GroupedGemm).
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
// All matrices are stored column-major (hipBLASLt default). For each group g:
//   C[g] = alpha * A[g] * B[g] + beta * C[g]
// with A[g] : m[g] x k[g], B[g] : k[g] x n[g], C[g] : m[g] x n[g].
//
// Note: the set of element types supported by the grouped-gemm path depends on
// the GPU architecture and hipBLASLt version. Types with no available heuristic
// (e.g. FP32/FP64 on some architectures) are reported and skipped.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <type_traits>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include "../blas-groupgemm-cuda/reference.h"

#define CHECK_HIP(call)                                                      \
  do {                                                                       \
    hipError_t err = (call);                                                 \
    if (err != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s at %s:%d\n",                             \
              hipGetErrorString(err), __FILE__, __LINE__);                   \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CHECK_HIPBLAS(call)                                                  \
  do {                                                                       \
    hipblasStatus_t st = (call);                                             \
    if (st != HIPBLAS_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "hipBLAS error %d at %s:%d\n", st, __FILE__, __LINE__);\
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// Host-side conversions between float and the device element type.
template <typename T> static inline T   fromFloat(float x) { return (T)x; }
template <> inline __half                fromFloat<__half>(float x) { return __float2half(x); }
template <typename T> static inline float toFloat(T x) { return (float)x; }
template <> inline float                 toFloat<__half>(__half x) { return __half2float(x); }

template <typename T>
void groupedGemm(const char *name, hipDataType dataType,
                 const std::vector<int> &m,
                 const std::vector<int> &n,
                 const std::vector<int> &k,
                 int repeat)
{
  printf(">>>>>>>>>>>>>>> %s grouped GEMM >>>>>>>>>>>>>>>\n", name);

  const int groups = (int)m.size();

  // Scalars use the compute-type precision: double for FP64, float otherwise.
  using Scalar = std::conditional_t<std::is_same_v<T, double>, double, float>;
  const hipblasComputeType_t computeType =
      std::is_same_v<T, double> ? HIPBLAS_COMPUTE_64F : HIPBLAS_COMPUTE_32F;
  const Scalar alpha = (Scalar)1;
  const Scalar beta  = (Scalar)0;

  // Reference inputs are generated in float; the device copies are converted.
  std::vector<std::vector<float>> hAf(groups), hBf(groups), hReff(groups), hCf(groups);
  std::vector<std::vector<T>> hA(groups), hB(groups);
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
    CHECK_HIP(hipMalloc(&dA[g], hA[g].size() * sizeof(T)));
    CHECK_HIP(hipMalloc(&dB[g], hB[g].size() * sizeof(T)));
    CHECK_HIP(hipMalloc(&dC[g], (size_t)m[g] * n[g] * sizeof(T)));
    CHECK_HIP(hipMemcpy(dA[g], hA[g].data(), hA[g].size() * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB[g], hB[g].data(), hB[g].size() * sizeof(T), hipMemcpyHostToDevice));
  }

  hipblasLtHandle_t handle;
  CHECK_HIPBLAS(hipblasLtCreate(&handle));

  hipblaslt_ext::GroupedGemm gg(handle,
                                HIPBLAS_OP_N, HIPBLAS_OP_N,
                                dataType, dataType, dataType, dataType, computeType);

  // Describe every GEMM in the group. Default leading dimensions (column-major)
  // are inferred from the sizes and transpose modes set in the constructor.
  std::vector<int64_t> M(groups), N(groups), K(groups), batch(groups, 1);
  std::vector<hipblaslt_ext::GemmEpilogue> epilogue(groups);
  std::vector<hipblaslt_ext::GemmInputs> inputs(groups);
  for (int g = 0; g < groups; g++) {
    M[g] = m[g];
    N[g] = n[g];
    K[g] = k[g];
    inputs[g].setA(dA[g]);
    inputs[g].setB(dB[g]);
    inputs[g].setC(dC[g]);
    inputs[g].setD(dC[g]);
    inputs[g].setAlpha(&alpha);
    inputs[g].setBeta(&beta);
  }
  CHECK_HIPBLAS(gg.setProblem(M, N, K, batch, epilogue, inputs));

  // Query a heuristic algorithm for the whole group.
  const size_t workspaceSize = 32ULL * 1024 * 1024;
  hipblaslt_ext::GemmPreference pref;
  pref.setMaxWorkspaceBytes(workspaceSize);

  const int requestedAlgoCount = 32;
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
  hipblasStatus_t hst = gg.algoGetHeuristic(requestedAlgoCount, pref, heuristic);

  if (hst != HIPBLAS_STATUS_SUCCESS || heuristic.empty()) {
    printf("No heuristic algorithm available for this configuration (skipped)\n");
    for (int g = 0; g < groups; g++) {
      CHECK_HIP(hipFree(dA[g])); CHECK_HIP(hipFree(dB[g])); CHECK_HIP(hipFree(dC[g]));
    }
    CHECK_HIPBLAS(hipblasLtDestroy(handle));
    return;
  }

  // Pick the first heuristic algorithm that actually supports this problem and
  // fits within our workspace budget.
  int chosen = -1;
  size_t chosenWs = 0;
  for (size_t i = 0; i < heuristic.size(); i++) {
    size_t ws = 0;
    if (gg.isAlgoSupported(heuristic[i].algo, ws) == HIPBLAS_STATUS_SUCCESS &&
        ws <= workspaceSize) {
      chosen = (int)i;
      chosenWs = ws;
      break;
    }
  }
  if (chosen < 0) {
    printf("No supported algorithm within workspace budget (skipped)\n");
    for (int g = 0; g < groups; g++) {
      CHECK_HIP(hipFree(dA[g])); CHECK_HIP(hipFree(dB[g])); CHECK_HIP(hipFree(dC[g]));
    }
    CHECK_HIPBLAS(hipblasLtDestroy(handle));
    return;
  }
  (void)chosenWs;

  void *workspace = nullptr;
  CHECK_HIP(hipMalloc(&workspace, workspaceSize));
  // useUserArgs = false so we can launch with the simple run(stream) entry point
  // rather than supplying a device user-arguments buffer.
  CHECK_HIPBLAS(gg.initialize(heuristic[chosen].algo, workspace, false));

  // Warmup and verify
  for (int r = 0; r < 30; r++)
    CHECK_HIPBLAS(gg.run(0));
  CHECK_HIP(hipDeviceSynchronize());

  std::vector<std::vector<T>> hC(groups);
  for (int g = 0; g < groups; g++) {
    hC[g].resize((size_t)m[g] * n[g]);
    CHECK_HIP(hipMemcpy(hC[g].data(), dC[g], hC[g].size() * sizeof(T), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < hC[g].size(); i++) hCf[g][i] = toFloat<T>(hC[g][i]);
  }

  groupedGemm_ref(m, n, k, 1.f, 0.f, hAf, hBf, hReff);
  double tol = std::is_same_v<T, __half> ? 1.0 : (std::is_same_v<T, float> ? 1e-1 : 1e-3);
  verify(hCf, hReff, tol);

  // Benchmarking
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    CHECK_HIPBLAS(gg.run(0));
  CHECK_HIP(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  double avg_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                  * 1e-3 / repeat;
  performance(m, n, k, avg_us);

  CHECK_HIP(hipFree(workspace));
  for (int g = 0; g < groups; g++) {
    CHECK_HIP(hipFree(dA[g]));
    CHECK_HIP(hipFree(dB[g]));
    CHECK_HIP(hipFree(dC[g]));
  }
  CHECK_HIPBLAS(hipblasLtDestroy(handle));
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

  groupedGemm<__half>("Half precision",   HIP_R_16F, m, n, k, repeat);
  //groupedGemm<float> ("Single precision", HIP_R_32F, m, n, k, repeat);
  //groupedGemm<double>("Double precision", HIP_R_64F, m, n, k, repeat);

  return 0;
}
