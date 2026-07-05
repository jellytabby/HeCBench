// Grouped GEMM using the oneDNN matmul primitive with grouped-memory encoding.
//
// A grouped GEMM computes a group of matmuls where one dimension varies per
// group.  This mirrors the vendor grouped-batched GEMM used in
// blas-groupgemm-cuda, but uses the oneDNN grouped-memory format for
// variable-size batching (commonly used for Mixture-of-Experts):
//   https://uxlfoundation.github.io/oneDNN/dev_guide_grouped_mem.html
//   https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html
//
// Layout (row-major):
//   src : [total_M, K]        grouped encoding, M varies per group
//   W   : [num_groups, K, N]  regular dense 3D tensor (uniform per group)
//   dst : [total_M, N]        grouped encoding, matching src group boundaries
// For each group g and its rows: dst = src @ W[g].
//
// The grouped-memory format is an experimental oneDNN feature available since
// oneDNN v3.12 (the API and CMake flag first appeared in v3.12; v3.11 and
// earlier have neither).  It must be enabled at oneDNN build time, which
// defines DNNL_EXPERIMENTAL_GROUPED_MEMORY in dnnl_config.h; the code below is
// guarded on that macro and otherwise compiles to a graceful skip.
//
// Building oneDNN (>= 3.12) with grouped memory and the SYCL runtime, per
// https://uxlfoundation.github.io/oneDNN/dev_guide_build.html :
//
//   git clone --depth 1 --branch v3.12.2 \
//       https://github.com/uxlfoundation/oneDNN.git
//   cd oneDNN && mkdir build && cd build
//   CC=icx CXX=icpx cmake -G Ninja \
//       -DCMAKE_BUILD_TYPE=Release \
//       -DDNNL_CPU_RUNTIME=SYCL \
//       -DDNNL_GPU_RUNTIME=SYCL \
//       -DDNNL_GPU_VENDOR=INTEL \
//       -DDNNL_EXPERIMENTAL_GROUPED_MEMORY=ON \
//       -DCMAKE_INSTALL_PREFIX=<install_dir> ..
//   ninja install
//
// Note: on the current development branch the option is spelled
// -DONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON (it still defines the same macro).
// Then build and run this benchmark against that install:
//
//   make ONEDNN=yes ONEDNN_PATH=<install_dir>
//   LD_LIBRARY_PATH=<install_dir>/lib ./main 100

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <numeric>
#include <vector>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include "reference.h"

#if defined(DNNL_EXPERIMENTAL_GROUPED_MEMORY)

using namespace dnnl;
using sycl::half;
using bfloat16 = sycl::ext::oneapi::bfloat16;

// Host-side conversions between float and the device element type.
template <typename T> static inline T     fromFloat(float x) { return (T)x; }
template <typename T> static inline float toFloat(T x)        { return (float)x; }

// Copy a host vector into a oneDNN memory buffer (buffer `index`).
template <typename T>
static void writeBuffer(memory &mem, const std::vector<T> &data, int index = 0) {
  T *p = mem.map_data<T>(index);
  std::copy(data.begin(), data.end(), p);
  mem.unmap_data(p, index);
}

// Copy a oneDNN memory buffer (buffer `index`) into a host vector.
template <typename T>
static void readBuffer(memory &mem, std::vector<T> &data, int index = 0) {
  T *p = mem.map_data<T>(index);
  std::copy(p, p + data.size(), data.begin());
  mem.unmap_data(p, index);
}

template <typename T>
void groupedGemm(const char *name, memory::data_type dtype,
                 engine &eng, stream &strm,
                 const std::vector<int> &groupM, int K, int N,
                 int repeat)
{
  printf(">>>>>>>>>>>>>>> %s grouped GEMM >>>>>>>>>>>>>>>\n", name);

  const int num_groups = (int)groupM.size();
  const memory::dim total_M =
      std::accumulate(groupM.begin(), groupM.end(), memory::dim(0));

  // Cumulative (exclusive-end) group boundaries required by grouped memory.
  std::vector<int32_t> offsets(num_groups);
  offsets[0] = groupM[0];
  for (int g = 1; g < num_groups; g++) offsets[g] = offsets[g - 1] + groupM[g];

  // Reference inputs generated in float; device copies converted to T.
  std::vector<float> srcF((size_t)total_M * K);
  std::vector<float> wF((size_t)num_groups * K * N);
  std::vector<float> refF((size_t)total_M * N, 0.f);
  srand48(123);
  for (auto &v : srcF) v = (float)drand48();
  for (auto &v : wF)   v = (float)drand48();

  std::vector<T> srcT(srcF.size()), wT(wF.size());
  for (size_t i = 0; i < srcF.size(); i++) srcT[i] = fromFloat<T>(srcF[i]);
  for (size_t i = 0; i < wF.size(); i++)   wT[i]   = fromFloat<T>(wF[i]);

  // Grouped-memory descriptors (variable_dim_idx = 0 -> M varies per group).
  auto src_md = memory::desc::grouped({total_M, K}, dtype, 0, num_groups);
  auto dst_md = memory::desc::grouped({total_M, N}, dtype, 0, num_groups);
  auto w_md   = memory::desc({num_groups, K, N}, dtype, memory::format_tag::abc);

  matmul::primitive_desc pd;
  try {
    pd = matmul::primitive_desc(eng, src_md, w_md, dst_md);
  } catch (dnnl::error &e) {
    if (e.status == dnnl_unimplemented) {
      printf("no grouped matmul implementation available for current configuration\n");
      return;
    }
    throw;
  }
  auto prim = matmul(pd);

  memory src_mem(src_md, eng);
  memory dst_mem(dst_md, eng);
  memory w_mem(w_md, eng);

  writeBuffer(src_mem, srcT);            // buffer 0: concatenated values
  writeBuffer(w_mem, wT);
  writeBuffer(src_mem, offsets, 1);      // buffer 1: group offsets
  writeBuffer(dst_mem, offsets, 1);      // dst uses identical offsets

  std::unordered_map<int, memory> args = {
    {DNNL_ARG_SRC, src_mem},
    {DNNL_ARG_WEIGHTS, w_mem},
    {DNNL_ARG_DST, dst_mem},
  };

  // Warmup.
  prim.execute(strm, args);
  strm.wait();

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    prim.execute(strm, args);
  strm.wait();
  auto end = std::chrono::steady_clock::now();
  double avg_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                  * 1e-3 / repeat;
  performance(groupM, K, N, avg_us);

  std::vector<T> dstT((size_t)total_M * N);
  readBuffer(dst_mem, dstT);
  std::vector<float> dstF(dstT.size());
  for (size_t i = 0; i < dstT.size(); i++) dstF[i] = toFloat<T>(dstT[i]);

  groupedGemm_ref(groupM, K, N, srcF, wF, refF);
  double tol = std::is_same_v<T, float> ? 1e-2 : (std::is_same_v<T, half> ? 5e-2 : 1e-1);
  verify(dstF, refF, tol);
}
#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY

int main(int argc, char *argv[])
{
  // Usage: main [repeat] [num_experts] [hidden(K)] [inter(N)] [avg_tokens]
  const int repeat = (argc > 1) ? atoi(argv[1]) : 100;
  const int num_experts = (argc > 2) ? atoi(argv[2]) : 64;   // routed experts (8 Mixtral, 64 DeepSeek-V2)
  const int hidden      = (argc > 3) ? atoi(argv[3]) : 2048; // model hidden size      -> K
  const int inter       = (argc > 4) ? atoi(argv[4]) : 2048; // FFN intermediate size  -> N
  const int avg_tokens  = (argc > 5) ? atoi(argv[5]) : 16;   // avg tokens/expert (batch*seq*top_k / E)

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

#if !defined(DNNL_EXPERIMENTAL_GROUPED_MEMORY)
  (void)repeat; (void)num_experts; (void)hidden; (void)inter; (void)avg_tokens;
  printf("Skipped: grouped matmul requires oneDNN >= 3.12 built with "
         "DNNL_EXPERIMENTAL_GROUPED_MEMORY=ON\n");
  return 0;
#else
  auto eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
  auto strm = dnnl::sycl_interop::make_stream(eng, q);

  // Draw a skewed per-expert probability, then assign
  // num_tokens tokens accordingly so the load is imbalanced.
  // Every expert is clamped to at least one token.
  // Only the per-expert token count (M) varies; K and N are shared, matching
  // the grouped-memory layout (src [total_M, K], weights [num_experts, K, N]).
  const long num_tokens = (long)num_experts * avg_tokens;
  srand48(123);
  std::vector<double> cdf(num_experts);
  double wsum = 0;
  for (int e = 0; e < num_experts; e++) { wsum += 0.2 + drand48(); cdf[e] = wsum; }

  std::vector<int> groupM(num_experts, 0);
  for (long t = 0; t < num_tokens; t++) {
    double r = drand48() * wsum;
    int e = 0;
    while (e < num_experts - 1 && r > cdf[e]) e++;
    groupM[e]++;
  }
  for (int e = 0; e < num_experts; e++) if (groupM[e] == 0) groupM[e] = 1;  // no empty groups

  const int K = hidden;
  const int N = inter;

  printf("MoE FFN grouped GEMM: %d experts, hidden(K)=%d, intermediate(N)=%d, repeat=%d\n",
         num_experts, hidden, inter, repeat);
#ifdef DEBUG
  int total_tokens = 0;
  for (int e = 0; e < num_experts; e++) {
    printf("  expert %d: tokens(M)=%d\n", e, groupM[e]);
    total_tokens += groupM[e];
  }
  printf("  total routed tokens = %d\n", total_tokens);
#endif

  groupedGemm<half>    ("Half precision",   memory::data_type::f16,  eng, strm, groupM, K, N, repeat);
  //groupedGemm<float>   ("Single precision", memory::data_type::f32,  eng, strm, groupM, K, N, repeat);
  //groupedGemm<bfloat16>("BF16",             memory::data_type::bf16, eng, strm, groupM, K, N, repeat);

  return 0;
#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
}
