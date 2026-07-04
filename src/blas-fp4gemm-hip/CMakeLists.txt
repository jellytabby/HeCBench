# blas-fp4gemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-fp4gemm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblaslt
)
