# blas-mxfp8gemm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-mxfp8gemm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublasLt
)
