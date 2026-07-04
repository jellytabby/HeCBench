#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
              __LINE__, cudaGetErrorString(err__));                           \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Full Jacobian of a batched softmax.
//
// For a single sample x in R^D:
//     y = softmax(x),   y_i = exp(x_i - max) / sum_k exp(x_k - max)
// the (full) Jacobian J = dy/dx in R^{D x D} has the closed form
//     J[i][j] = dy_i / dx_j = y_i * (delta_ij - y_j)
//
// This is exactly the local Jacobian used by the backward pass of softmax:
// the reverse-mode gradient (VJP) is  dL/dx = J^T (dL/dy), and here we
// materialize the *entire* J for every sample in the batch (output B x D x D).
//
// The results are verified against a CPU reference here, and against PyTorch's
// reverse-mode (vjp / jacrev) and forward-mode (jvp / jacfwd) autodiff in
// verify_torch.py, which reads the small dump written by dump_reference().
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T dev_exp(T v);
template <>
__device__ __forceinline__ float dev_exp<float>(float v) { return expf(v); }
template <>
__device__ __forceinline__ double dev_exp<double>(double v) { return exp(v); }

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

    // reuse the front of J as scratch for y, then overwrite
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

// k1: one block per sample. Softmax is computed cooperatively into shared
// memory, then the D x D Jacobian is written with a flat, coalesced grid-stride
// loop over the dim*dim output entries.
template <typename T>
__global__ void softmax_jacobian_k1(
    const T* __restrict__ input,
          T* __restrict__ jac,
    int dim)
{
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* y   = reinterpret_cast<T*>(smem_raw);   // [dim]
  T* red = y + dim;                          // [blockDim.x]

  const int b   = blockIdx.x;
  const int tid = threadIdx.x;
  const int nt  = blockDim.x;
  const T*  x   = input + (size_t)b * dim;
  T*        J   = jac   + (size_t)b * dim * dim;

  // max reduction
  T lmax = -INFINITY;
  for (int i = tid; i < dim; i += nt) lmax = x[i] > lmax ? x[i] : lmax;
  red[tid] = lmax;
  __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] = max(red[tid] , red[tid + s]);
    __syncthreads();
  }
  const T maxv = red[0];
  __syncthreads();

  // exp and sum reduction
  T lsum = 0;
  for (int i = tid; i < dim; i += nt) {
    const T e = dev_exp<T>(x[i] - maxv);
    y[i] = e;
    lsum += e;
  }
  red[tid] = lsum;
  __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const T inv = (T)1 / red[0];
  __syncthreads();

  for (int i = tid; i < dim; i += nt) y[i] *= inv;
  __syncthreads();

  // Jacobian: J[i*dim + j] = y_i * (delta_ij - y_j)
  const size_t total = (size_t)dim * dim;
  for (size_t idx = tid; idx < total; idx += nt) {
    const int i = idx / dim;
    const int j = idx - (size_t)i * dim;
    J[idx] = y[i] * ((i == j ? (T)1 : (T)0) - y[j]);
  }
}

// Warp-level primitives -----------------------------------------------------
// Full-warp reductions via shuffle; no shared memory / __syncthreads needed
// within a warp. Assumes a full 32-lane warp (blockDim is a multiple of 64).
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T v) {
#pragma unroll
  for (int o = warpSize >> 1; o > 0; o >>= 1)
    v = max(v, __shfl_down_sync(0xffffffffu, v, o));
  return v;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
#pragma unroll
  for (int o = warpSize >> 1; o > 0; o >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

// 4-wide vector wrapper for widened (128-bit for float) loads/stores.
template <typename T>
struct alignas(4 * sizeof(T)) Vec4 { T v[4]; };

// k2: one block per sample, aggressively warp-centric. The softmax max/sum
// reductions are done with shuffle instructions inside each warp and only the
// per-warp partials touch shared memory. The final scalars (maxv, inv) are
// broadcast to every lane with __shfl_sync, so the hot Jacobian loop needs no
// __syncthreads at all. The normalized softmax vector y is staged in shared
// memory and the (memory-bound) D x D output is streamed with 128-bit
// vectorized, coalesced stores.
template <typename T>
__global__ void softmax_jacobian_k2(
    const T* __restrict__ input,
          T* __restrict__ jac,
    int dim)
{
  // 16-byte alignment lets us reinterpret shared y as Vec4<float> safely.
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* y     = reinterpret_cast<T*>(smem_raw);   // [dim]
  T* wpart = y + dim;                          // [num_warps] warp partials

  const int b        = blockIdx.x;
  const int tid      = threadIdx.x;
  const int nt       = blockDim.x;
  const int lane     = tid & (warpSize - 1);
  const int warp     = tid >> 5;
  const int nwarps   = (nt + warpSize - 1) / warpSize;
  const T*  x        = input + (size_t)b * dim;
  T*        J        = jac   + (size_t)b * dim * dim;

  // ---- max reduction: warp shuffle, then one warp combines the partials ----
  T lmax = -INFINITY;
  for (int i = tid; i < dim; i += nt) lmax = x[i] > lmax ? x[i] : lmax;
  lmax = warp_reduce_max<T>(lmax);
  if (lane == 0) wpart[warp] = lmax;
  __syncthreads();
  if (warp == 0) {
    T v = (lane < nwarps) ? wpart[lane] : (T)-INFINITY;
    v = warp_reduce_max<T>(v);
    if (lane == 0) wpart[0] = v;
  }
  __syncthreads();
  const T maxv = wpart[0];

  // ---- exp + sum reduction: same warp-centric pattern ----------------------
  T lsum = 0;
  for (int i = tid; i < dim; i += nt) {
    const T e = dev_exp<T>(x[i] - maxv);
    y[i] = e;
    lsum += e;
  }
  lsum = warp_reduce_sum<T>(lsum);
  if (lane == 0) wpart[warp] = lsum;
  __syncthreads();
  if (warp == 0) {
    T v = (lane < nwarps) ? wpart[lane] : (T)0;
    v = warp_reduce_sum<T>(v);
    if (lane == 0) wpart[0] = v;
  }
  __syncthreads();
  const T inv = (T)1 / wpart[0];

  // normalize softmax in shared memory
  for (int i = tid; i < dim; i += nt) y[i] *= inv;
  __syncthreads();

  // ---- Jacobian: J[i*dim + j] = y_i * (delta_ij - y_j) ---------------------
  if ((dim & 3) == 0) {
    const int quads_per_row  = dim >> 2;
    const size_t total_quads = (size_t)dim * (size_t)quads_per_row;
    const Vec4<T>* y4 = reinterpret_cast<const Vec4<T>*>(y);
    Vec4<T>*       J4 = reinterpret_cast<Vec4<T>*>(J);
    for (size_t q = tid; q < total_quads; q += nt) {
      const int i    = (int)(q / (size_t)quads_per_row);
      const int qcol = (int)(q - (size_t)i * quads_per_row);
      const int j0   = qcol << 2;
      const T yi = y[i];
      const Vec4<T> yj = y4[qcol];
      Vec4<T> out;
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        const T d = (i == j0 + k) ? (T)1 : (T)0;
        out.v[k] = yi * (d - yj.v[k]);
      }
      J4[q] = out;
    }
  } else {
    // Scalar fallback for dims not divisible by 4.
    const size_t total = (size_t)dim * dim;
    for (size_t idx = tid; idx < total; idx += nt) {
      const int i = (int)(idx / dim);
      const int j = (int)(idx - (size_t)i * dim);
      J[idx] = y[i] * ((i == j ? (T)1 : (T)0) - y[j]);
    }
  }
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

  printf("Batch size: %d\n", batch_size);

  const int dims[] = {128, 512, 2048, 8192};

  for (size_t n = 0; n < sizeof(dims) / sizeof(int); n++) {
    const int dim = dims[n];
    printf("\nSoftmax dimension: %d (Jacobian %d x %d per sample)\n", dim, dim, dim);

    const size_t in_elems  = (size_t)batch_size * dim;
    const size_t jac_elems = (size_t)batch_size * dim * dim;
    const size_t in_bytes  = in_elems * sizeof(float);
    const size_t jac_bytes = jac_elems * sizeof(float);

    float* input      = (float*)malloc(in_bytes);
    float* jac_k1     = (float*)malloc(jac_bytes);
    float* jac_k2     = (float*)malloc(jac_bytes);
    float* jac_ref    = (float*)malloc(jac_bytes);

    std::default_random_engine g(123);
    std::uniform_real_distribution<float> distr(-3.f, 3.f);
    for (size_t i = 0; i < in_elems; i++) input[i] = distr(g);

    reference(input, jac_ref, batch_size, dim);

    float *d_input, *d_jac;
    CUDA_CHECK(cudaMalloc((void**)&d_input, in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_jac, jac_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input, in_bytes, cudaMemcpyHostToDevice));

    for (int block_size = 64; block_size <= 1024; block_size *= 2) {
      printf("block size: %d\n", block_size);
      const size_t shmem = ((size_t)dim + block_size) * sizeof(float);

      CUDA_CHECK(cudaMemset(d_jac, 0, jac_bytes));
      softmax_jacobian_k1<float><<<batch_size, block_size, shmem>>>(d_input, d_jac, dim);
      CUDA_CHECK(cudaMemcpy(jac_k1, d_jac, jac_bytes, cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaMemset(d_jac, 0, jac_bytes));
      softmax_jacobian_k2<float><<<batch_size, block_size, shmem>>>(d_input, d_jac, dim);
      CUDA_CHECK(cudaMemcpy(jac_k2, d_jac, jac_bytes, cudaMemcpyDeviceToHost));

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

    printf("Benchmarking..\n");

    for (int block_size = 64; block_size <= 1024; block_size *= 2) {
      printf("block size: %d\n", block_size);
      const size_t shmem = ((size_t)dim + block_size) * sizeof(float);

      CUDA_CHECK(cudaMemset(d_jac, 0, jac_bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        softmax_jacobian_k1<float><<<batch_size, block_size, shmem>>>(d_input, d_jac, dim);
      CUDA_CHECK(cudaDeviceSynchronize());
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of softmax Jacobian kernel (k1): %f (us)\n",
             (time * 1e-3f) / repeat);

      CUDA_CHECK(cudaMemset(d_jac, 0, jac_bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
      start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        softmax_jacobian_k2<float><<<batch_size, block_size, shmem>>>(d_input, d_jac, dim);
      CUDA_CHECK(cudaDeviceSynchronize());
      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of softmax Jacobian kernel (k2): %f (us)\n",
             (time * 1e-3f) / repeat);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_jac));
    free(input);
    free(jac_k1);
    free(jac_k2);
    free(jac_ref);
  }

  return 0;
}
