#include <chrono>
#include <cstring>
#include "helper.h"

// MXFP6 block-scaled matmul in hipBLASLt requires ROCm >= 7.0 (the HIP_R_6F_E2M3
// data type and VEC32_UE8M0 scale mode) and a CDNA4 GPU (gfx950).
#if defined(HIP_VERSION_MAJOR) && (HIP_VERSION_MAJOR >= 7)
#define MXFP6_GEMM_SUPPORTED 1
#else
#define MXFP6_GEMM_SUPPORTED 0
#endif

#if MXFP6_GEMM_SUPPORTED

/// Block-scaled MXFP6 (FP6 E2M3) matmul with hipblasLtMatmul on CDNA4 (gfx950).
///
///   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
/// A and B are FP6 (HIP_R_6F_E2M3) packed four elements per three bytes along K.
/// The block scales are E8M0 (HIP_R_8F_UE8M0) with one scale per 32 elements
/// along the K dimension (scale mode HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0).
/// Output D is FP16.
///
/// A is stored (M,K) row-major and B is stored (N,K) row-major. To compute
/// A @ B^T on the column-major hipBLASLt we use transa = OP_T, transb = OP_N so
/// that A is interpreted as (K x M) and B as (K x N), producing D = (M x N).
void LtMxfp6Matmul(const int repeat,
                   hipblasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const float *alpha, /* host pointer */
                   const float *beta,  /* host pointer */
                   const void *a_scale, /* device pointer, UE8M0 block scales */
                   const void *A,       /* device pointer, packed FP6 (M,K) */
                   int lda,
                   const void *b_scale, /* device pointer, UE8M0 block scales */
                   const void *B,       /* device pointer, packed FP6 (N,K) */
                   int ldb,
                   __half *D,           /* device pointer, FP16 (M,N) */
                   int ldd,
                   void *workspace,
                   size_t workspaceSize) {
    hipblasLtMatmulDesc_t operationDesc = NULL;
    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    hipblasLtMatmulPreference_t preference = NULL;

    // A must be transposed and B non-transposed (TN format required by MX types).
    hipblasOperation_t transa = HIPBLAS_OP_T;
    hipblasOperation_t transb = HIPBLAS_OP_N;

    int returnedResults = 0;
    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    checkHipblasStatus(hipblasLtMatmulDescCreate(&operationDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Block-scaling: set the scale pointers and mark them as VEC32_UE8M0 tensors.
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    hipblasLtMatmulMatrixScale_t scaleMode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Create matrix descriptors. FP6 dimensions are in elements (packed 4/3 bytes).
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_6F_E2M3, k, m, lda));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_6F_E2M3, k, n, ldb));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_16F, m, n, ldd));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Ddesc, HIP_R_16F, m, n, ldd));

    checkHipblasStatus(hipblasLtMatmulPreferenceCreate(&preference));
    checkHipblasStatus(hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    checkHipblasStatus(hipblasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        printf("no heuristic function available for current configuration\n");
        if (preference) hipblasLtMatmulPreferenceDestroy(preference);
        if (Ddesc) hipblasLtMatrixLayoutDestroy(Ddesc);
        if (Cdesc) hipblasLtMatrixLayoutDestroy(Cdesc);
        if (Bdesc) hipblasLtMatrixLayoutDestroy(Bdesc);
        if (Adesc) hipblasLtMatrixLayoutDestroy(Adesc);
        if (operationDesc) hipblasLtMatmulDescDestroy(operationDesc);
        return;
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        checkHipblasStatus(hipblasLtMatmul(ltHandle,
                                           operationDesc,
                                           alpha, A, Adesc,
                                           B, Bdesc, beta,
                                           D, Cdesc,
                                           D, Ddesc,
                                           &heuristicResult.algo,
                                           workspace,
                                           workspaceSize,
                                           0));
    }

    checkHipStatus(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto ns = (time / repeat);
    printf("Average hipblasLtMatmul execution time %10.3f (us) | ", ns * 1e-3f);
    printf("Average hipblasLtMatmul performance %.1f (TFLOPS)\n", 2.f * m * k * n / ns * 1e-3f);

    if (preference) checkHipblasStatus(hipblasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkHipblasStatus(hipblasLtMatmulDescDestroy(operationDesc));
}

#endif // MXFP6_GEMM_SUPPORTED

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

#if !MXFP6_GEMM_SUPPORTED
    printf("Skipped: MXFP6 block-scaled matmul requires ROCm >= 7.0\n");
    return 0;
#else
    // Skip on GPUs other than CDNA4 (gfx950), which do not support MXFP6
    // block-scaled matmul.
    int device = 0;
    checkHipStatus(hipGetDevice(&device));
    hipDeviceProp_t prop;
    checkHipStatus(hipGetDeviceProperties(&prop, device));
    if (strncmp(prop.gcnArchName, "gfx950", 6) != 0) {
        printf("Skipped: MXFP6 block-scaled matmul requires a CDNA4 GPU (gfx950), "
               "found %s (%s)\n", prop.gcnArchName, prop.name);
        return 0;
    }

    // M = N = 8192, sweep K -- following the benchmark shapes in reference.py.
    const int M = 8192;
    const int N = 8192;
    const int Ks[5] = {512, 1024, 2048, 4096, 8192};

    for (int i = 0; i < 5; i++) {
        int m = M, n = N, k = Ks[i];
        printf("Matrix dimension (M, N, K) = (%d, %d, %d)\n", m, n, k);

        Mxfp6TestBench props(m, n, k, 1.0f, 0.0f, 32ULL * 1024 * 1024);

        LtMxfp6Matmul(repeat,
                      props.ltHandle,
                      props.m,
                      props.n,
                      props.k,
                      &props.alpha,
                      &props.beta,
                      props.AscaleDev, props.Adev, props.k, // A: (M,K), lda = K (elements)
                      props.BscaleDev, props.Bdev, props.k, // B: (N,K), ldb = K (elements)
                      props.Ddev, props.m,                  // D: (M,N), ldd = M
                      props.workspace,
                      props.workspaceSize);

        props.verify();
    }

    return 0;
#endif // MXFP6_GEMM_SUPPORTED
}
