#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "reference.h"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 128
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__device__ __forceinline__ float device_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// store GRU gates in reset, update, new order:
// weight_ih_l[k] = (W_ir | W_iz | W_in), weight_hh_l[k] = (W_hr | W_hz | W_hn).
// pack every layer into one flat array and pads W_ih rows to
// max(input_size, hidden_size) so layer 0 and deeper layers share one stride.
__device__ __forceinline__ size_t device_wih_offset(
    size_t layer,
    int gate,
    int hidden_idx,
    int input_idx,
    int hidden_size,
    int max_input_size) {
  return ((layer * 3 + gate) * hidden_size + hidden_idx) *
      max_input_size + input_idx;
}

__device__ __forceinline__ size_t device_whh_offset(
    size_t layer,
    int gate,
    int hidden_idx,
    int prev_hidden_idx,
    int hidden_size) {
  return ((layer * 3 + gate) * hidden_size + hidden_idx) *
      hidden_size + prev_hidden_idx;
}

__device__ __forceinline__ size_t device_bias_offset(
    size_t layer,
    int gate,
    int hidden_idx,
    int hidden_size) {
  return (layer * 3 + gate) * hidden_size + hidden_idx;
}

__global__ void fused_gru_layer_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h_prev,
    const float* __restrict__ w_ih,
    const float* __restrict__ w_hh,
    const float* __restrict__ b_ih,
    const float* __restrict__ b_hh,
    float* __restrict__ layer_out,
    float* __restrict__ h_next,
    float* __restrict__ output_t,
    int layer,
    int batch_size,
    int input_size,
    int hidden_size,
    int max_input_size) {
  const int hidden_tile = blockDim.x / warpSize;
  const int b = blockIdx.x;
  const int local_h = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int h = blockIdx.y * hidden_tile + local_h;
  if (b >= batch_size || h >= hidden_size) {
    return;
  }

  // four warps per block, one warp per hidden
  // element. Lanes in a warp tile the dot products and reduce them in-warp.
  const float* h_row = h_prev + (size_t)b * hidden_size;
  float input_gate[3];
  float hidden_gate[3];
  for (int gate = 0; gate < 3; ++gate) {
    float input_sum = 0.0f;
    for (int k = lane; k < input_size; k += warpSize) {
      input_sum += x[(size_t)b * input_size + k] *
          w_ih[device_wih_offset(layer, gate, h, k, hidden_size, max_input_size)];
    }

    float hidden_sum = 0.0f;
    for (int k = lane; k < hidden_size; k += warpSize) {
      hidden_sum += h_row[k] *
          w_hh[device_whh_offset(layer, gate, h, k, hidden_size)];
    }

    const unsigned mask = 0xffffffffu;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      input_sum += __shfl_down_sync(mask, input_sum, offset);
      hidden_sum += __shfl_down_sync(mask, hidden_sum, offset);
    }

    if (lane == 0) {
      input_gate[gate] =
          input_sum + b_ih[device_bias_offset(layer, gate, h, hidden_size)];
      hidden_gate[gate] =
          hidden_sum + b_hh[device_bias_offset(layer, gate, h, hidden_size)];
    }
  }

  if (lane == 0) {
    const size_t idx = (size_t)b * hidden_size + h;
    const float hx = h_prev[idx];
    const float r = device_sigmoid(input_gate[0] + hidden_gate[0]);
    const float z = device_sigmoid(input_gate[1] + hidden_gate[1]);
    const float n = tanhf(input_gate[2] + r * hidden_gate[2]);
    const float ht = n + z * (hx - n);
    h_next[idx] = ht;
    layer_out[idx] = ht;
    if (output_t != nullptr) {
      output_t[idx] = ht;
    }
  }
}

const float* gru(
    const float* d_input,
    const float* d_w_ih,
    const float* d_w_hh,
    const float* d_b_ih,
    const float* d_b_hh,
    const float* d_h0,
    float* d_output,
    float* d_hidden,
    float* d_hidden_a,
    float* d_layer_output_a,
    float* d_layer_output_b,
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers,
    int max_input_size) {
  constexpr int threads = THREADS_PER_BLOCK;
  static_assert(threads <= 1024,
                "THREADS_PER_BLOCK cannot exceed CUDA's block limit");
  static_assert(threads % 32 == 0,
                "THREADS_PER_BLOCK must be a multiple of warp size");
  constexpr int hidden_tile = threads / 32;
  const dim3 grid(batch_size, (hidden_size + hidden_tile - 1) / hidden_tile);

  const float* hidden_read = d_h0; // initial hidden state
  float* hidden_buffers[2] = {d_hidden, d_hidden_a};
  int hidden_write_idx = 0;

  for (int t = 0; t < seq_len; ++t) {

    float* hidden_write = hidden_buffers[hidden_write_idx];

    size_t input_offset = (size_t)t * batch_size * input_size;
    size_t output_offset = (size_t)t * batch_size * hidden_size;

    const float* current_input = d_input + input_offset;
    int current_input_size = input_size;

    for (int layer = 0; layer < num_layers; ++layer) {

      size_t hidden_offset = (size_t)layer * batch_size * hidden_size;

      float* current_output = (layer % 2 == 0) ? d_layer_output_a : d_layer_output_b;
      if (layer > 0) {
        current_input = (layer % 2 == 1) ? d_layer_output_a : d_layer_output_b;
        current_input_size = hidden_size;
      }

      const float* layer_hidden_read = hidden_read + hidden_offset;
      float* layer_hidden_write = hidden_write + hidden_offset;
      float* output_t = (layer == num_layers - 1) ? d_output + output_offset : nullptr;

      fused_gru_layer_kernel<<<grid, threads>>>(
          current_input, layer_hidden_read, d_w_ih, d_w_hh, d_b_ih, d_b_hh,
          current_output, layer_hidden_write, output_t, layer, batch_size,
          current_input_size, hidden_size, max_input_size);
    }

    hidden_read = hidden_write;
    hidden_write_idx = 1 - hidden_write_idx;
  }

  return hidden_read;
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::printf(
        "Usage: %s <seq_len> <batch_size> <input_size> <hidden_size> "
        "<num_layers> <repeat>\n",
        argv[0]);
    return 1;
  }

  const int seq_len = std::atoi(argv[1]);
  const int batch_size = std::atoi(argv[2]);
  const int input_size = std::atoi(argv[3]);
  const int hidden_size = std::atoi(argv[4]);
  const int num_layers = std::atoi(argv[5]);
  const int repeat = std::atoi(argv[6]);
  if (seq_len <= 0 || batch_size <= 0 || input_size <= 0 ||
      hidden_size <= 0 || num_layers <= 0 || repeat <= 0) {
    std::fprintf(stderr, "All arguments must be positive integers.\n");
    return 1;
  }

  const int max_input_size = std::max(input_size, hidden_size);
  const size_t input_elements = (size_t)seq_len * batch_size * input_size;
  const size_t output_elements = (size_t)seq_len * batch_size * hidden_size;
  const size_t hidden_state_elements = (size_t)num_layers * batch_size * hidden_size;
  const size_t w_ih_elements = (size_t)num_layers * 3 * hidden_size * max_input_size;
  const size_t w_hh_elements = (size_t)num_layers * 3 * hidden_size * hidden_size;
  const size_t bias_elements = (size_t)num_layers * 3 * hidden_size;
  const size_t layer_output_elements = (size_t)batch_size * hidden_size;

  // W_ih is padded to max_input_size because layer 0 has input_size columns,
  // while layers 1..N-1 have hidden_size columns.
  std::vector<float> h_input(input_elements);
  std::vector<float> h_w_ih(w_ih_elements, 0.0f);
  std::vector<float> h_w_hh(w_hh_elements);
  std::vector<float> h_b_ih(bias_elements);
  std::vector<float> h_b_hh(bias_elements);
  std::vector<float> h_h0(hidden_state_elements);
  std::vector<float> h_output(output_elements);
  std::vector<float> h_hidden(hidden_state_elements);
  std::vector<float> h_output_ref(output_elements);
  std::vector<float> h_hidden_ref(hidden_state_elements);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
  for (float& v : h_input) v = dist(gen);
  for (float& v : h_w_ih) v = dist(gen);
  for (float& v : h_w_hh) v = dist(gen);
  for (float& v : h_b_ih) v = dist(gen);
  for (float& v : h_b_hh) v = dist(gen);
  for (float& v : h_h0) v = dist(gen);

  float *d_input, *d_w_ih, *d_w_hh, *d_b_ih, *d_b_hh, *d_h0;
  float *d_output, *d_hidden, *d_hidden_a;
  float *d_layer_output_a, *d_layer_output_b;

  CUDA_CHECK(cudaMalloc((void**)&d_input, input_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_w_ih, w_ih_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_w_hh, w_hh_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_b_ih, bias_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_b_hh, bias_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_h0, hidden_state_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_output, output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_hidden, hidden_state_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_hidden_a, hidden_state_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_layer_output_a, layer_output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_layer_output_b, layer_output_elements * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w_ih, h_w_ih.data(), w_ih_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w_hh, h_w_hh.data(), w_hh_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_ih, h_b_ih.data(), bias_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_hh, h_b_hh.data(), bias_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_h0, h_h0.data(), hidden_state_elements * sizeof(float),
                        cudaMemcpyHostToDevice));

  const float* d_final_hidden = gru(
      d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0, d_output, d_hidden,
      d_hidden_a, d_layer_output_a, d_layer_output_b, seq_len, batch_size,
      input_size, hidden_size, num_layers, max_input_size);

  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                        output_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_hidden.data(), d_final_hidden,
                        hidden_state_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));

  gru_reference(h_input.data(), h_w_ih.data(), h_w_hh.data(), h_b_ih.data(),
                h_b_hh.data(), h_h0.data(), h_output_ref.data(),
                h_hidden_ref.data(), seq_len, batch_size, input_size,
                hidden_size, num_layers, max_input_size);

  float max_output_error = 0.0f;
  for (size_t i = 0; i < output_elements; ++i) {
    max_output_error = std::max(max_output_error,
                                std::fabs(h_output[i] - h_output_ref[i]));
  }
  float max_hidden_error = 0.0f;
  for (size_t i = 0; i < hidden_state_elements; ++i) {
    max_hidden_error = std::max(max_hidden_error,
                                std::fabs(h_hidden[i] - h_hidden_ref[i]));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    d_final_hidden = gru(
        d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0, d_output, d_hidden,
        d_hidden_a, d_layer_output_a, d_layer_output_b, seq_len, batch_size,
        input_size, hidden_size, num_layers, max_input_size);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::printf("Average execution time of multi_layer_gru: %f (us)\n",
              (elapsed_ns * 1e-3f) / repeat);
  std::printf("max_output_error: %.8e\n", max_output_error);
  std::printf("max_hidden_error: %.8e\n", max_hidden_error);
  bool ok = (max_output_error < 1e-4f && max_hidden_error < 1e-4f);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_w_ih));
  CUDA_CHECK(cudaFree(d_w_hh));
  CUDA_CHECK(cudaFree(d_b_ih));
  CUDA_CHECK(cudaFree(d_b_hh));
  CUDA_CHECK(cudaFree(d_h0));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_hidden));
  CUDA_CHECK(cudaFree(d_hidden_a));
  CUDA_CHECK(cudaFree(d_layer_output_a));
  CUDA_CHECK(cudaFree(d_layer_output_b));

  return 0;
}
