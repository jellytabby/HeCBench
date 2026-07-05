# blas-groupgemm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-groupgemm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -ldnnl
)
