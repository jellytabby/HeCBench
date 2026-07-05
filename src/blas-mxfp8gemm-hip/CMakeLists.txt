# blas-mxfp8gemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-mxfp8gemm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblaslt
)
