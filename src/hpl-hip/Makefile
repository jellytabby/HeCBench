.PHONY: all clean run lapack

# Reference LAPACK is an external dependency of HPL. By default we build it
# in-tree under HeCBench/lapack/ so a fresh checkout is self-sufficient (no
# manual cmake step, no $(HOME)/lapack convention).
#
# To use an existing LAPACK install instead, point LAdir at its build dir:
#
#     make LAdir=/path/to/your/lapack/build_dir
#
# In that mode the `lapack` target is a no-op — the user is fully responsible
# for the install being complete. If any lib or header is missing, the inner
# Make.intel64 will fail at link time with a precise ld error naming the
# offending file.
LAPACK_DIR    ?= $(abspath $(CURDIR)/../../lapack)
LAPACK_BUILD  ?= $(LAPACK_DIR)/build_lapack_32
LAdir         ?= $(LAPACK_BUILD)
export LAdir

all: lapack
	$(MAKE) -C src/hpl-2.3

clean:
	$(MAKE) -C src/hpl-2.3 clean

ifeq ($(LAdir),$(LAPACK_BUILD))
# Default: build Reference LAPACK in-tree. libcblas.so is the sentinel — if
# it's present we assume the build is intact (we produced it).
lapack: $(LAPACK_BUILD)/lib/libcblas.so

$(LAPACK_BUILD)/lib/libcblas.so:
	@if [ ! -d "$(LAPACK_DIR)/.git" ] && [ ! -f "$(LAPACK_DIR)/CMakeLists.txt" ]; then \
	    echo "Cloning Reference LAPACK v3.12.1 into $(LAPACK_DIR)"; \
	    git clone --depth 1 --branch v3.12.1 \
	        https://github.com/Reference-LAPACK/lapack.git "$(LAPACK_DIR)"; \
	fi
	cmake -S "$(LAPACK_DIR)" -B "$(LAPACK_BUILD)" \
	    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
	    -DCBLAS=ON -DLAPACKE=ON \
	    -DBUILD_INDEX64=OFF -DBUILD_INDEX64_EXT_API=OFF \
	    -DBUILD_TESTING=OFF
	cmake --build "$(LAPACK_BUILD)" -j
else
# User supplied their own LAdir. Trust it; link-time errors (if any) will name
# exactly what's missing.
lapack:
	@echo "Using user-provided LAPACK at $(LAdir)"
endif

# Encodes the manual run procedure documented in README.md:
#   - copy the small-GPU input deck over HPL.dat
#   - extend LD_LIBRARY_PATH with the in-tree HIP shim and LAPACK
#   - launch xhpl under mpirun
run: all
	cd src/hpl-2.3/bin/intel64 && \
	cp ../../../../datafiles/HPL_small_gpu.dat HPL.dat && \
	LD_LIBRARY_PATH=../../src/cuda:$(LAdir)/lib:$(LD_LIBRARY_PATH) \
	mpirun -n 1 ./xhpl
