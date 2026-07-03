#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

#include "../gru2-cuda/reference.h"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

inline float device_sigmoid(float x) {
  return 1.0f / (1.0f + sycl::exp(-x));
}

inline size_t device_wih_offset(
    int layer,
    int gate,
    int hidden_idx,
    int input_idx,
    int hidden_size,
    int max_input_size) {
  return (((size_t)layer * 3 + gate) * hidden_size + hidden_idx) *
      max_input_size + input_idx;
}

inline size_t device_whh_offset(
    int layer,
    int gate,
    int hidden_idx,
    int prev_hidden_idx,
    int hidden_size) {
  return (((size_t)layer * 3 + gate) * hidden_size + hidden_idx) *
      hidden_size + prev_hidden_idx;
}

inline size_t device_bias_offset(
    int layer,
    int gate,
    int hidden_idx,
    int hidden_size) {
  return ((size_t)layer * 3 + gate) * hidden_size + hidden_idx;
}

const float* gru(
    sycl::queue& q,
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
    int max_input_size)
{
  constexpr int threads = THREADS_PER_BLOCK;
  static_assert(threads <= 1024,
                "THREADS_PER_BLOCK cannot exceed GPU block limit");
  static_assert(threads % 64 == 0,
                "THREADS_PER_BLOCK must support both 32- and 64-lane warps");

  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int warp_size = *r;
  const int hidden_tile = threads / warp_size;
  sycl::range<2> gws((hidden_size + hidden_tile - 1) / hidden_tile, batch_size * threads);
  sycl::range<2> lws(1, threads);

  const float* hidden_read = d_h0;
  float* hidden_buffers[2] = {d_hidden, d_hidden_a};
  int hidden_write_idx = 0;

  for (int t = 0; t < seq_len; ++t) {
    float* hidden_write = hidden_buffers[hidden_write_idx];

    const size_t input_offset = (size_t)t * batch_size * input_size;
    const size_t output_offset = (size_t)t * batch_size * hidden_size;

    const float* current_input = d_input + input_offset;
    int current_input_size = input_size;

    for (int layer = 0; layer < num_layers; ++layer) {
      const size_t hidden_offset = (size_t)layer * batch_size * hidden_size;

      float* current_output = (layer % 2 == 0) ? d_layer_output_a : d_layer_output_b;
      if (layer > 0) {
        current_input = (layer % 2 == 1) ? d_layer_output_a : d_layer_output_b;
        current_input_size = hidden_size;
      }

      const float* layer_hidden_read = hidden_read + hidden_offset;
      float* layer_hidden_write = hidden_write + hidden_offset;
      float* output_t = (layer == num_layers - 1) ? d_output + output_offset : nullptr;

      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class fused_gru_layer>(
            sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item) {
              const sycl::sub_group sg = item.get_sub_group();
              const int sg_size = sg.get_local_range()[0];
              const size_t hidden_tile = sg.get_group_range()[0]; // the number of subgroups
              const int b = item.get_group(1);
              const int local_h = sg.get_group_linear_id(); // index of the subgroup
              const int lane = sg.get_local_linear_id();
              const int h = item.get_group(0) * hidden_tile + local_h;

              if (b >= batch_size || h >= hidden_size) {
                return;
              }

              const float* h_row = layer_hidden_read + (size_t)b * hidden_size;

              float input_gate[3];
              float hidden_gate[3];
              for (int gate = 0; gate < 3; ++gate) {
                float input_sum = 0.0f;
                for (int k = lane; k < current_input_size; k += sg_size) {
                  input_sum +=
                      current_input[(size_t)b * current_input_size + k] *
                      d_w_ih[device_wih_offset(layer, gate, h, k, hidden_size, max_input_size)];
                }

                float hidden_sum = 0.0f;
                for (int k = lane; k < hidden_size; k += sg_size) {
                  hidden_sum += h_row[k] *
                      d_w_hh[device_whh_offset(layer, gate, h, k, hidden_size)];
                }

                for (int offset = sg_size / 2; offset > 0; offset /= 2) {
                  input_sum += sycl::shift_group_left(sg, input_sum, offset);
                  hidden_sum += sycl::shift_group_left(sg, hidden_sum, offset);
                }

                if (lane == 0) {
                  input_gate[gate] = input_sum +
                      d_b_ih[device_bias_offset(layer, gate, h, hidden_size)];
                  hidden_gate[gate] = hidden_sum +
                      d_b_hh[device_bias_offset(layer, gate, h, hidden_size)];
                }
              }

              if (lane == 0) {
                const size_t idx = (size_t)b * hidden_size + h;
                const float hx = layer_hidden_read[idx];
                const float r = device_sigmoid(input_gate[0] + hidden_gate[0]);
                const float z = device_sigmoid(input_gate[1] + hidden_gate[1]);
                const float n = sycl::tanh(input_gate[2] + r * hidden_gate[2]);
                const float ht = n + z * (hx - n);
                layer_hidden_write[idx] = ht;
                current_output[idx] = ht;
                if (output_t != nullptr) {
                  output_t[idx] = ht;
                }
              }
            });
      });
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float* d_input = sycl::malloc_device<float>(input_elements, q);
  float* d_w_ih = sycl::malloc_device<float>(w_ih_elements, q);
  float* d_w_hh = sycl::malloc_device<float>(w_hh_elements, q);
  float* d_b_ih = sycl::malloc_device<float>(bias_elements, q);
  float* d_b_hh = sycl::malloc_device<float>(bias_elements, q);
  float* d_h0 = sycl::malloc_device<float>(hidden_state_elements, q);
  float* d_output = sycl::malloc_device<float>(output_elements, q);
  float* d_hidden = sycl::malloc_device<float>(hidden_state_elements, q);
  float* d_hidden_a = sycl::malloc_device<float>(hidden_state_elements, q);
  float* d_layer_output_a = sycl::malloc_device<float>(layer_output_elements, q);
  float* d_layer_output_b = sycl::malloc_device<float>(layer_output_elements, q);

  q.memcpy(d_input, h_input.data(), input_elements * sizeof(float));
  q.memcpy(d_w_ih, h_w_ih.data(), w_ih_elements * sizeof(float));
  q.memcpy(d_w_hh, h_w_hh.data(), w_hh_elements * sizeof(float));
  q.memcpy(d_b_ih, h_b_ih.data(), bias_elements * sizeof(float));
  q.memcpy(d_b_hh, h_b_hh.data(), bias_elements * sizeof(float));
  q.memcpy(d_h0, h_h0.data(), hidden_state_elements * sizeof(float));

  const float* d_final_hidden = gru(
      q, d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0, d_output, d_hidden,
      d_hidden_a, d_layer_output_a, d_layer_output_b, seq_len, batch_size,
      input_size, hidden_size, num_layers, max_input_size);

  q.memcpy(h_output.data(), d_output, output_elements * sizeof(float));
  q.memcpy(h_hidden.data(), d_final_hidden, hidden_state_elements * sizeof(float));
  q.wait();

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

  q.wait();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    d_final_hidden = gru(q, d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0,
                         d_output, d_hidden, d_hidden_a, d_layer_output_a,
                         d_layer_output_b, seq_len, batch_size, input_size,
                         hidden_size, num_layers, max_input_size);
  }
  q.wait();
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::printf("Average execution time of multi_layer_gru: %f (us)\n",
              (elapsed_ns * 1e-3f) / repeat);
  std::printf("max_output_error: %.8e\n", max_output_error);
  std::printf("max_hidden_error: %.8e\n", max_hidden_error);
  bool ok = (max_output_error < 1e-4f && max_hidden_error < 1e-4f);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_input, q);
  sycl::free(d_w_ih, q);
  sycl::free(d_w_hh, q);
  sycl::free(d_b_ih, q);
  sycl::free(d_b_hh, q);
  sycl::free(d_h0, q);
  sycl::free(d_output, q);
  sycl::free(d_hidden, q);
  sycl::free(d_hidden_a, q);
  sycl::free(d_layer_output_a, q);
  sycl::free(d_layer_output_b, q);

  return 0;
}
