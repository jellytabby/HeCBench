# mlp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mlp
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES -ldnnl
)
