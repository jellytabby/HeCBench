# blas-mxfp6gemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-mxfp6gemm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblaslt
)
