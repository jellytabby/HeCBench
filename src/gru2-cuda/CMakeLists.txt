# gru2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gru2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
