# gru2-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gru2
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
