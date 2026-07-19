#!/usr/bin/env bash

set -euo pipefail

BENCHMARKS=(rsbench
  xsbench
  blacksholes)

ml ninja

# cmake -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
#       -DCMAKE_EXE_LINKER_FLAGS=-no-pie \
#       -DCUDAToolkit_ROOT=/usr/tce/packages/cuda/cuda-13.1.1/ \
#       --preset cuda-sm90

for TARGET in $BENCHMARKS; do
  cmake --build build/cuda-sm90 --target="$TARGET"
done
      # -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      # -DMPI_C_COMPILER=/usr/tce/packages/mvapich2/mvapich2-2.3.7-intel-classic-2021.6.0-magic/bin/mpicc \
      # -DMPI_CXX_COMPILER=/usr/tce/packages/mvapich2/mvapich2-2.3.7-intel-classic-2021.6.0-magic/bin/mpic++ \

