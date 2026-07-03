#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <omp.h>

#include "../gru2-cuda/reference.h"

#pragma omp declare target
inline float omp_target_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}
#pragma omp end declare target

// Multi-layer GRU. One work-item per (batch, hidden) output.
void gru(
    const float* input,
    const float* w_ih,
    const float* w_hh,
    const float* b_ih,
    const float* b_hh,
    const float* h0,
    float* output,
    float* hidden,
    float* hidden_next,
    float* layer_input,
    float* layer_output,
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers,
    int max_input_size)
{
  const size_t hidden_state_elements =
      (size_t)num_layers * batch_size * hidden_size;

#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < hidden_state_elements; ++i) {
    hidden[i] = h0[i];
  }

  bool read_from_next = false;
  bool write_to_next = true;
  for (int t = 0; t < seq_len; ++t) {
    for (int layer = 0; layer < num_layers; ++layer) {
#pragma omp target teams distribute parallel for collapse(2)
      for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < hidden_size; ++h) {
          const int current_input_size =
              (layer == 0) ? input_size : hidden_size;
          const size_t hidden_base =
              ((size_t)layer * batch_size + b) * hidden_size;
          float input_gate[3];
          float hidden_gate[3];

          for (int gate = 0; gate < 3; ++gate) {
            float input_sum =
                b_ih[((size_t)layer * 3 + gate) * hidden_size + h];
            for (int k = 0; k < current_input_size; ++k) {
              float x_value;
              if (layer == 0) {
                x_value = input[((size_t)t * batch_size + b) * input_size + k];
              } else {
                // Read the buffer written by the previous layer (ping-pong).
                const size_t in_idx = (size_t)b * hidden_size + k;
                x_value = ((layer & 1) == 0) ? layer_input[in_idx]
                                             : layer_output[in_idx];
              }
              const size_t wih_idx =
                  (((size_t)layer * 3 + gate) * hidden_size + h) *
                      max_input_size + k;
              input_sum += x_value * w_ih[wih_idx];
            }

            float hidden_sum =
                b_hh[((size_t)layer * 3 + gate) * hidden_size + h];
            for (int k = 0; k < hidden_size; ++k) {
              const size_t whh_idx =
                  (((size_t)layer * 3 + gate) * hidden_size + h) *
                      hidden_size + k;
              const float h_value = read_from_next
                  ? hidden_next[hidden_base + k]
                  : hidden[hidden_base + k];
              hidden_sum += h_value * w_hh[whh_idx];
            }

            input_gate[gate] = input_sum;
            hidden_gate[gate] = hidden_sum;
          }

          const float hx = read_from_next ? hidden_next[hidden_base + h]
                                          : hidden[hidden_base + h];
          const float r = omp_target_sigmoid(input_gate[0] + hidden_gate[0]);
          const float z = omp_target_sigmoid(input_gate[1] + hidden_gate[1]);
          const float n = tanhf(input_gate[2] + r * hidden_gate[2]);
          const float ht = n + z * (hx - n);
          const size_t idx = (size_t)b * hidden_size + h;
          if (write_to_next) {
            hidden_next[hidden_base + h] = ht;
          } else {
            hidden[hidden_base + h] = ht;
          }
          // Write this layer's output to the opposite ping-pong buffer.
          if ((layer & 1) == 0) {
            layer_output[idx] = ht;
          } else {
            layer_input[idx] = ht;
          }
          if (layer == num_layers - 1) {
            output[((size_t)t * batch_size + b) * hidden_size + h] = ht;
          }
        }
      }
    }
    // Advance the recurrent state once per timestep, after all layers.
    read_from_next = write_to_next;
    write_to_next = !write_to_next;
  }

  if (read_from_next) {
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < hidden_state_elements; ++i) {
      hidden[i] = hidden_next[i];
    }
  }
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
  const size_t hidden_state_elements =
      (size_t)num_layers * batch_size * hidden_size;
  const size_t w_ih_elements =
      (size_t)num_layers * 3 * hidden_size * max_input_size;
  const size_t w_hh_elements =
      (size_t)num_layers * 3 * hidden_size * hidden_size;
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
  std::vector<float> h_hidden_next(hidden_state_elements);
  std::vector<float> h_layer_input(layer_output_elements);
  std::vector<float> h_layer_output(layer_output_elements);
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

  gru_reference(h_input.data(), h_w_ih.data(), h_w_hh.data(), h_b_ih.data(),
                h_b_hh.data(), h_h0.data(), h_output_ref.data(),
                h_hidden_ref.data(), seq_len, batch_size, input_size,
                hidden_size, num_layers, max_input_size);

  float* input = h_input.data();
  float* w_ih = h_w_ih.data();
  float* w_hh = h_w_hh.data();
  float* b_ih = h_b_ih.data();
  float* b_hh = h_b_hh.data();
  float* h0 = h_h0.data();
  float* output = h_output.data();
  float* hidden = h_hidden.data();
  float* hidden_next = h_hidden_next.data();
  float* layer_input = h_layer_input.data();
  float* layer_output = h_layer_output.data();

#pragma omp target data map(to: input[0:input_elements],                     \
                                w_ih[0:w_ih_elements],                       \
                                w_hh[0:w_hh_elements],                       \
                                b_ih[0:bias_elements],                       \
                                b_hh[0:bias_elements],                       \
                                h0[0:hidden_state_elements])                 \
                        map(from: output[0:output_elements],                 \
                                  hidden[0:hidden_state_elements])           \
                        map(alloc: hidden_next[0:hidden_state_elements],      \
                                   layer_input[0:layer_output_elements],     \
                                   layer_output[0:layer_output_elements])
  {
    gru(input, w_ih, w_hh, b_ih, b_hh, h0, output, hidden, hidden_next,
        layer_input, layer_output, seq_len, batch_size, input_size,
        hidden_size, num_layers, max_input_size);
    

#pragma omp target update from(output[0:output_elements],                    \
                               hidden[0:hidden_state_elements])

    float max_output_error = 0.0f;
    float max_hidden_error = 0.0f;
    for (size_t i = 0; i < output_elements; ++i) {
      max_output_error = std::max(max_output_error,
                                  std::fabs(h_output[i] - h_output_ref[i]));
    }
    for (size_t i = 0; i < hidden_state_elements; ++i) {
      max_hidden_error = std::max(max_hidden_error,
                                  std::fabs(h_hidden[i] - h_hidden_ref[i]));
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      gru(input, w_ih, w_hh, b_ih, b_hh, h0, output, hidden, hidden_next,
          layer_input, layer_output, seq_len, batch_size, input_size,
          hidden_size, num_layers, max_input_size);
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    const bool ok = (max_output_error < 1e-4f && max_hidden_error < 1e-4f);
    printf("Average execution time of multi_layer_gru: %f (us)\n",
           (elapsed_ns * 1e-3f) / repeat);
    printf("max_output_error: %.8e\n", max_output_error);
    printf("max_hidden_error: %.8e\n", max_hidden_error);
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  return 0;
}
