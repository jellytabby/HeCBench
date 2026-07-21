#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

static void CheckError( hipError_t err, const char *file, int line ) {
  if (err != hipSuccess) {
    printf( "%s in %s at line %d\n", hipGetErrorString( err ), file, line );
  }
}
#define CHECK_ERROR( err ) (CheckError( err, __FILE__, __LINE__ ))

#define BLOCK_SIZE 256

#define ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define ONE_FP16  __ushort_as_half((unsigned short)0x3c00U)
#define ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x0000U)
#define ONE_BF16  __ushort_as_bfloat16((unsigned short)0x3f80U)

// fp16: 10 explicit + 1 implicit mantissa bits → max exact int = 2048
// bf16:  7 explicit + 1 implicit mantissa bits → max exact int = 256
template<typename T> constexpr int maxExactInt();
template<> constexpr int maxExactInt<__half>() { return 2048; }
template<> constexpr int maxExactInt<__hip_bfloat16>() { return 256; }

__global__
void f16AtomicOnGlobalMem(__half* result, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  __half2 *result_v = reinterpret_cast<__half2*>(result);
  __half2 val {ZERO_FP16, ONE_FP16};
  unsafeAtomicAdd(&result_v[tid % BLOCK_SIZE], val);
}

__global__
void f16AtomicOnGlobalMem(__hip_bfloat16* result, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  __hip_bfloat162 *result_v = reinterpret_cast<__hip_bfloat162*>(result);
  __hip_bfloat162 val {ZERO_BF16, ONE_BF16};
  unsafeAtomicAdd(&result_v[tid % BLOCK_SIZE], val);
}

template <typename T>
bool verify(const T* result, int hits_per_slot)
{
  float expected = (float)std::min(hits_per_slot, maxExactInt<T>());
  for (int i = 0; i < BLOCK_SIZE; i++) {
    float even = (float)result[2 * i];
    float odd  = (float)result[2 * i + 1];
    if (even != 0.0f || odd != expected) {
      printf("FAIL at slot %d: expected (0.0, %f), got (%f, %f)\n",
             i, expected, even, odd);
      return false;
    }
  }
  return true;
}

template <typename T>
bool atomicCost (int nelems, int repeat)
{
  size_t result_size = sizeof(T) * BLOCK_SIZE * 2;

  T* result = (T*) malloc (result_size);

  T *d_result;
  CHECK_ERROR( hipMalloc((void **)&d_result, result_size) );
  CHECK_ERROR( hipMemset(d_result, 0, result_size) );

  dim3 block (BLOCK_SIZE);
  dim3 grid ((nelems / 2  + BLOCK_SIZE - 1) / BLOCK_SIZE);

  f16AtomicOnGlobalMem<<<grid, block>>>(d_result, nelems/2);
  CHECK_ERROR( hipDeviceSynchronize() );
  CHECK_ERROR( hipMemcpy(result, d_result, result_size, hipMemcpyDeviceToHost) );

  int hits_per_slot = (nelems / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  bool pass = verify(result, hits_per_slot);
  printf("%s\n", pass ? "PASS" : "FAIL");

  free(result);
  if (!pass) {
    CHECK_ERROR(hipFree(d_result));
    return false;
  }

  CHECK_ERROR( hipMemset(d_result, 0, result_size) );
  CHECK_ERROR( hipDeviceSynchronize() );
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    f16AtomicOnGlobalMem<<<grid, block>>>(d_result, nelems/2);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of 16-bit floating-point atomic add on global memory: %f (us)\n",
          time * 1e-3f / repeat);
  CHECK_ERROR(hipFree(d_result));
  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: total number of elements (a multiple of 2)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  assert(nelems > 0 && (nelems % 2) == 0);

  printf("\nFP16 atomic add\n");
  bool fp16_pass = atomicCost<__half>(nelems, repeat);

  printf("\nBF16 atomic add\n");
  bool bf16_pass = atomicCost<__hip_bfloat16>(nelems, repeat);

  return (fp16_pass && bf16_pass) ? 0 : 1;
}
