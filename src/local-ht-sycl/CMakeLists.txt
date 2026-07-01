# local-ht-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME local-ht
    MODEL sycl
    SOURCES
        src/main.cpp
        src/helper.cpp
    CATEGORIES bioinformatics
)
