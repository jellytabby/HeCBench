# blas-mxfp6gemm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-mxfp6gemm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublasLt
)
