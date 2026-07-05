# blas-mxfp8gemm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-mxfp8gemm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -ldnnl
)
