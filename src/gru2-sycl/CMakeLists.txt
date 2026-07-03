# gru2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gru2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
