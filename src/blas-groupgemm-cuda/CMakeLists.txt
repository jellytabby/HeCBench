# blas-groupgemm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-groupgemm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
)
