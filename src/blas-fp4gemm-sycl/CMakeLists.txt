# blas-mxfp4gemm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-mxfp4gemm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -ldnnl
)
