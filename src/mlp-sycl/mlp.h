#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

// Global SYCL queue and oneDNN engine/stream
#ifdef USE_GPU
sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

auto engine = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
auto stream = dnnl::sycl_interop::make_stream(engine, q);

// Run a single dense layer: out = W^T * input (optionally + bias, optionally ReLU).
//
// Matrix storage follows the cuBLASLt example (column-major):
//   weight: [ifeat x ofeat]
//   input:  [ifeat x batch]
//   output: [ofeat x batch]
//
template <typename scalar_t>
int mlp_gemm_dnnl(
    int batch,
    int ifeat,
    int ofeat,
    const scalar_t* weight,
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* bias,
    bool use_bias,
    bool use_relu) {

  using namespace dnnl;

  // Memory descriptors. The cuBLASLt example stores buffers in column-major
  // layout: input is [ifeat x batch], weight is [ifeat x ofeat], output is
  // [ofeat x batch]. Viewed row-major, input becomes [batch x ifeat] and
  // output becomes [batch x ofeat] (natural format_tag::ab). The weight
  // buffer's row-major view is [ofeat x ifeat]; we declare the oneDNN dims
  // as (ifeat, ofeat) and use format_tag::ba so the underlying strides
  // match the column-major memory.
  auto inp_md    = memory::desc({batch, ifeat}, memory::data_type::f32, memory::format_tag::ab);
  auto weight_md = memory::desc({ifeat, ofeat}, memory::data_type::f32, memory::format_tag::ba);
  auto out_md    = memory::desc({batch, ofeat}, memory::data_type::f32, memory::format_tag::ab);
  auto bias_md   = memory::desc({1, ofeat},     memory::data_type::f32, memory::format_tag::ab);

  // USM memory objects
  auto inp_mem    = sycl_interop::make_memory(inp_md,    engine, sycl_interop::memory_kind::usm,
                                              const_cast<scalar_t*>(input));
  auto weight_mem = sycl_interop::make_memory(weight_md, engine, sycl_interop::memory_kind::usm,
                                              const_cast<scalar_t*>(weight));
  auto out_mem    = sycl_interop::make_memory(out_md,    engine, sycl_interop::memory_kind::usm, output);

  // Post-op for ReLU epilogue
  primitive_attr matmul_attr;
  if (use_relu) {
    post_ops po;
    po.append_eltwise(algorithm::eltwise_relu, 0.0f, 0.0f);
    matmul_attr.set_post_ops(po);
  }

  // Primitive descriptor
  matmul::primitive_desc matmul_pd;
  if (use_bias) {
    matmul_pd = matmul::primitive_desc(engine, inp_md, weight_md, bias_md, out_md, matmul_attr);
  } else {
    matmul_pd = matmul::primitive_desc(engine, inp_md, weight_md, out_md, matmul_attr);
  }

  auto matmul_prim = matmul(matmul_pd);

  if (use_bias) {
    auto bias_mem = sycl_interop::make_memory(bias_md, engine, sycl_interop::memory_kind::usm,
                                              const_cast<scalar_t*>(bias));
    matmul_prim.execute(stream, {
        {DNNL_ARG_SRC,     inp_mem},
        {DNNL_ARG_WEIGHTS, weight_mem},
        {DNNL_ARG_BIAS,    bias_mem},
        {DNNL_ARG_DST,     out_mem}
    });
  } else {
    matmul_prim.execute(stream, {
        {DNNL_ARG_SRC,     inp_mem},
        {DNNL_ARG_WEIGHTS, weight_mem},
        {DNNL_ARG_DST,     out_mem}
    });
  }

  return 0;
}


// Does a simple MLP fprop (GEMM+bias).
// Can handle num_layers number of layers, each with its own shape. Output of layer i is assumed
// to be input of layer i+1. output_features, WPtr and BPtr are arrays of length num_layers, and
// must be in the same order i.e. WPtr[i] and BPtr[i] are respectively the weight and bias of layer
// 'i'.
template <typename T>
int mlp_fp(
    T* X,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T** BPtr,
    T* Y,
    T* reserved_space,
    int use_bias,
    int use_relu)
{
  T *weight, *input, *output, *bias = nullptr;
  T *reserved_space_x, *reserved_space_y;
  reserved_space_x = nullptr;
  reserved_space_y = reserved_space;

  for (int layer = 0; layer < num_layers; layer++) {
    weight = WPtr[layer];
    input  = (layer == 0) ? X : reserved_space_x;
    output = (layer == num_layers - 1) ? Y : reserved_space_y;
    if (use_bias) {
      bias = BPtr[layer];
    }
    int ifeat = (layer == 0) ? input_features : output_features[layer - 1];
    int ofeat = output_features[layer];

    int status = mlp_gemm_dnnl<T>(
        batch_size,
        ifeat,
        ofeat,
        weight,
        input,
        output,
        bias,
        use_bias == 1,
        use_relu == 1);
    if (status) return 1;

    reserved_space_x = reserved_space_y;
    reserved_space_y += ofeat * batch_size;
  }

  return 0;
}

template int mlp_fp<float>(
    float* X,
    int input_features,
    int batch_size,
    float** WPtr,
    int num_layers,
    int* output_features,
    float** BPtr,
    float* Y,
    float* reserved_space,
    int use_bias,
    int use_relu);
