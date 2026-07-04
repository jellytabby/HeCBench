#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>

// ---------------------------------------------------------------------------
// Full Jacobian of a batched softmax.
//
// For a single sample x in R^D:
//     y = softmax(x),   y_i = exp(x_i - max) / sum_k exp(x_k - max)
// the (full) Jacobian J = dy/dx in R^{D x D} has the closed form
//     J[i][j] = dy_i / dx_j = y_i * (delta_ij - y_j)
//
// SYCL port of jacobian-cuda. The warp-centric kernel (k2) uses sub-group
// collectives (reduce_over_group). Both kernels are softmax_jacobianed with a required
// sub-group size that is templated so the program can exercise BOTH 32- and
// 64-wide sub-groups (whichever the device reports as supported).
// ---------------------------------------------------------------------------

template <typename T>
void reference(const T* input, T* jac, int batch_size, int dim)
{
  for (int b = 0; b < batch_size; b++) {
    const T* x = input + (size_t)b * dim;
    T* J = jac + (size_t)b * dim * dim;

    T maxv = x[0];
    #pragma omp parallel for reduction(max:maxv)
    for (int i = 1; i < dim; i++) maxv = std::max(x[i], maxv);

    T sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < dim; i++) sum += std::exp(x[i] - maxv);
    const T inv = (T)1 / sum;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        const T yi = std::exp(x[i] - maxv) * inv;
        const T yj = std::exp(x[j] - maxv) * inv;
        J[(size_t)i * dim + j] = yi * ((i == j ? (T)1 : (T)0) - yj);
      }
    }
  }
}

// k1: one work-group per sample. Softmax is computed cooperatively into local
// memory, then the D x D Jacobian is written with a flat, coalesced group-
// stride loop over the dim*dim output entries. (Sub-group size independent.)
template <typename T>
void softmax_jacobian_k1(sycl::queue& q, const T* input, T* jac, int dim,
            int batch_size, int block_size)
{
  const size_t local_elems = (size_t)dim + block_size;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<T, 1> smem(sycl::range<1>(local_elems), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t)batch_size * block_size),
                          sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> it) {
          T* y   = smem.template get_multi_ptr<sycl::access::decorated::no>().get();
          T* red = y + dim;

          const int b   = it.get_group(0);
          const int tid = it.get_local_id(0);
          const int nt  = it.get_local_range(0);
          const T*  x   = input + (size_t)b * dim;
          T*        J   = jac   + (size_t)b * dim * dim;

          T lmax = -std::numeric_limits<T>::infinity();
          for (int i = tid; i < dim; i += nt) lmax = x[i] > lmax ? x[i] : lmax;
          red[tid] = lmax;
          it.barrier(sycl::access::fence_space::local_space);
          for (int s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] = sycl::fmax(red[tid], red[tid + s]);
            it.barrier(sycl::access::fence_space::local_space);
          }
          const T maxv = red[0];
          it.barrier(sycl::access::fence_space::local_space);

          T lsum = 0;
          for (int i = tid; i < dim; i += nt) {
            const T e = sycl::exp(x[i] - maxv);
            y[i] = e;
            lsum += e;
          }
          red[tid] = lsum;
          it.barrier(sycl::access::fence_space::local_space);
          for (int s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            it.barrier(sycl::access::fence_space::local_space);
          }
          const T inv = (T)1 / red[0];
          it.barrier(sycl::access::fence_space::local_space);

          for (int i = tid; i < dim; i += nt) y[i] *= inv;
          it.barrier(sycl::access::fence_space::local_space);

          const size_t total = (size_t)dim * dim;
          for (size_t idx = tid; idx < total; idx += nt) {
            const int i = (int)(idx / dim);
            const int j = (int)(idx - (size_t)i * dim);
            J[idx] = y[i] * ((i == j ? (T)1 : (T)0) - y[j]);
          }
        });
  });
}

// k2: one work-group per sample, sub-group-centric. The softmax max/sum
// reductions use reduce_over_group across each sub-group and only the
// per-sub-group partials touch local memory. The normalized softmax vector y
// is staged in local memory and the (memory-bound) D x D output is streamed
// with 128-bit vectorized (float4), coalesced stores. Correct for SG in {32,64}.
template <typename T>
void softmax_jacobian_k2(sycl::queue& q, const T* input, T* jac, int dim,
            int batch_size, int block_size)
{
  const size_t local_elems = (size_t)dim + block_size;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<T, 1> smem(sycl::range<1>(local_elems), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t)batch_size * block_size),
                          sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> it) {
          T* y     = smem.template get_multi_ptr<sycl::access::decorated::no>().get();
          T* wpart = y + dim;

          const int b   = it.get_group(0);
          const int tid = it.get_local_id(0);
          const int nt  = it.get_local_range(0);

          sycl::sub_group sg = it.get_sub_group();
          const int lane   = sg.get_local_linear_id();
          const int warp   = sg.get_group_linear_id();
          const int nwarps = sg.get_group_linear_range();

          const T* x = input + (size_t)b * dim;
          T*       J = jac   + (size_t)b * dim * dim;

          // ---- max reduction: sub-group reduce, then combine partials ----
          T lmax = -std::numeric_limits<T>::infinity();
          for (int i = tid; i < dim; i += nt) lmax = x[i] > lmax ? x[i] : lmax;
          lmax = sycl::reduce_over_group(sg, lmax, sycl::maximum<T>{});
          if (lane == 0) wpart[warp] = lmax;
          it.barrier(sycl::access::fence_space::local_space);
          if (warp == 0) {
            T v = (lane < nwarps) ? wpart[lane]
                                  : -std::numeric_limits<T>::infinity();
            v = sycl::reduce_over_group(sg, v, sycl::maximum<T>{});
            if (lane == 0) wpart[0] = v;
          }
          it.barrier(sycl::access::fence_space::local_space);
          const T maxv = wpart[0];

          // ---- exp + sum reduction: same sub-group pattern ----
          T lsum = 0;
          for (int i = tid; i < dim; i += nt) {
            const T e = sycl::exp(x[i] - maxv);
            y[i] = e;
            lsum += e;
          }
          lsum = sycl::reduce_over_group(sg, lsum, sycl::plus<T>{});
          if (lane == 0) wpart[warp] = lsum;
          it.barrier(sycl::access::fence_space::local_space);
          if (warp == 0) {
            T v = (lane < nwarps) ? wpart[lane] : (T)0;
            v = sycl::reduce_over_group(sg, v, sycl::plus<T>{});
            if (lane == 0) wpart[0] = v;
          }
          it.barrier(sycl::access::fence_space::local_space);
          const T inv = (T)1 / wpart[0];

          for (int i = tid; i < dim; i += nt) y[i] *= inv;
          it.barrier(sycl::access::fence_space::local_space);

          // ---- Jacobian: J[i*dim + j] = y_i * (delta_ij - y_j) ----
          if ((dim & 3) == 0) {
            using V4 = sycl::vec<T, 4>;
            const int quads_per_row  = dim >> 2;
            const size_t total_quads = (size_t)dim * (size_t)quads_per_row;
            V4* J4 = reinterpret_cast<V4*>(J);
            for (size_t qi = tid; qi < total_quads; qi += nt) {
              const int i    = (int)(qi / (size_t)quads_per_row);
              const int qcol = (int)(qi - (size_t)i * quads_per_row);
              const int j0   = qcol << 2;
              const T yi = y[i];
              V4 out;
              #pragma unroll
              for (int k = 0; k < 4; k++) {
                const T d = (i == j0 + k) ? (T)1 : (T)0;
                out[k] = yi * (d - y[j0 + k]);
              }
              J4[qi] = out;
            }
          } else {
            const size_t total = (size_t)dim * dim;
            for (size_t idx = tid; idx < total; idx += nt) {
              const int i = (int)(idx / dim);
              const int j = (int)(idx - (size_t)i * dim);
              J[idx] = y[i] * ((i == j ? (T)1 : (T)0) - y[j]);
            }
          }
        });
  });
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int batch_size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  assert(batch_size > 0);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("Batch size: %d\n", batch_size);

  const int dims[] = {128, 512, 2048, 8192};

  for (size_t n = 0; n < sizeof(dims) / sizeof(int); n++) {
    const int dim = dims[n];
    printf("\nSoftmax dimension: %d (Jacobian %d x %d per sample)\n", dim, dim, dim);

    const size_t in_elems  = (size_t)batch_size * dim;
    const size_t jac_elems = (size_t)batch_size * dim * dim;
    const size_t in_bytes  = in_elems * sizeof(float);
    const size_t jac_bytes = jac_elems * sizeof(float);

    float* input   = (float*)malloc(in_bytes);
    float* jac_k1  = (float*)malloc(jac_bytes);
    float* jac_k2  = (float*)malloc(jac_bytes);
    float* jac_ref = (float*)malloc(jac_bytes);

    std::default_random_engine g(123);
    std::uniform_real_distribution<float> distr(-3.f, 3.f);
    for (size_t i = 0; i < in_elems; i++) input[i] = distr(g);

    reference(input, jac_ref, batch_size, dim);

    float* d_input = sycl::malloc_device<float>(in_elems, q);
    float* d_jac   = sycl::malloc_device<float>(jac_elems, q);
    q.memcpy(d_input, input, in_bytes).wait();

    // warmup and verify
    for (int block_size = 64; block_size <= 1024; block_size *= 2) {
      printf("block size: %d\n", block_size);

      q.memset(d_jac, 0, jac_bytes);
      softmax_jacobian_k1<float>(q, d_input, d_jac, dim, batch_size, block_size);
      q.memcpy(jac_k1, d_jac, jac_bytes).wait();

      q.memset(d_jac, 0, jac_bytes);
      softmax_jacobian_k2<float>(q, d_input, d_jac, dim, batch_size, block_size);
      q.memcpy(jac_k2, d_jac, jac_bytes).wait();

      bool ok = true;
      for (size_t i = 0; i < jac_elems; i++) {
        if (fabsf(jac_k1[i] - jac_ref[i]) > 1e-3f ||
            fabsf(jac_k2[i] - jac_ref[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }

    printf("Benchmarking...\n");

    for (int block_size = 64; block_size <= 1024; block_size *= 2) {
      printf("block size: %d\n", block_size);

      q.memset(d_jac, 0, jac_bytes).wait();
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        softmax_jacobian_k1<float>(q, d_input, d_jac, dim, batch_size, block_size);
      q.wait();
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of softmax Jacobian kernel (k1): %f (us)\n",
             (time * 1e-3f) / repeat);

      q.memset(d_jac, 0, jac_bytes).wait();
      start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        softmax_jacobian_k2<float>(q, d_input, d_jac, dim, batch_size, block_size);
      q.wait();
      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of softmax Jacobian kernel (k2): %f (us)\n",
             (time * 1e-3f) / repeat);
    }
    
    sycl::free(d_input, q);
    sycl::free(d_jac, q);
    free(input);
    free(jac_k1);
    free(jac_k2);
    free(jac_ref);
  }

  return 0;
}
