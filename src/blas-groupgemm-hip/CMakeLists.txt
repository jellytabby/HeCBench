# blas-groupgemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-groupgemm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblaslt
)
