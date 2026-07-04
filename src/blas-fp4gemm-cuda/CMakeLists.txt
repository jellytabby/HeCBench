# blas-fp4gemm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-fp4gemm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublasLt
)
