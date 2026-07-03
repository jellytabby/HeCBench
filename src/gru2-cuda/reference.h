#ifndef GRU_CUDA_REFERENCE_H
#define GRU_CUDA_REFERENCE_H

#include <cmath>
#include <vector>

inline float gru_sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

// Gate order follows torch.nn.GRU: reset, update, new. PyTorch names the
// packed rows (W_ir | W_iz | W_in) and (W_hr | W_hz | W_hn).
inline size_t gru_wih_offset(
    int layer,
    int gate,
    int hidden_idx,
    int input_idx,
    int hidden_size,
    int max_input_size) {
  // W_ih is padded to max_input_size so all layers can share one flat layout.
  return (((size_t)layer * 3 + gate) * hidden_size + hidden_idx) *
      max_input_size + input_idx;
}

inline size_t gru_whh_offset(
    int layer,
    int gate,
    int hidden_idx,
    int prev_hidden_idx,
    int hidden_size) {
  return (((size_t)layer * 3 + gate) * hidden_size + hidden_idx) *
      hidden_size + prev_hidden_idx;
}

inline size_t gru_bias_offset(
    int layer,
    int gate,
    int hidden_idx,
    int hidden_size) {
  return ((size_t)layer * 3 + gate) * hidden_size + hidden_idx;
}

inline void gru_reference(
    const float* input,
    const float* w_ih,
    const float* w_hh,
    const float* b_ih,
    const float* b_hh,
    const float* h0,
    float* output,
    float* hn,
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers,
    int max_input_size) {
  const size_t hidden_state_size = (size_t)num_layers * batch_size * hidden_size;
  for (size_t i = 0; i < hidden_state_size; ++i) {
    hn[i] = h0[i];
  }

  std::vector<float> layer_in((size_t)batch_size * max_input_size);
  std::vector<float> layer_out((size_t)batch_size * hidden_size);

  // Advance through the sequence one time step at a time. At step t, each
  // layer reads h_(t-1) from hn and overwrites it with h_t.
  for (int t = 0; t < seq_len; ++t) {
    const float* current_input = input + (size_t)t * batch_size * input_size;
    int current_input_size = input_size;

    // Layer 0 consumes x_t. Deeper layers consume the h_t produced by the
    // previous layer at the same time step, matching PyTorch's stacked GRU.
    for (int layer = 0; layer < num_layers; ++layer) {
      if (layer > 0) {
        current_input = layer_in.data();
        current_input_size = hidden_size;
      }

      for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < hidden_size; ++h) {
          float input_gate[3];
          float hidden_gate[3];
          const float* h_prev = hn + ((size_t)layer * batch_size + b) * hidden_size;

          for (int gate = 0; gate < 3; ++gate) {
            // Compute x_t W_i*^T + b_i* for one gate and hidden element.
            float input_sum = b_ih[gru_bias_offset(layer, gate, h, hidden_size)];
            for (int k = 0; k < current_input_size; ++k) {
              input_sum += current_input[(size_t)b * current_input_size + k] *
                  w_ih[gru_wih_offset(
                      layer, gate, h, k, hidden_size, max_input_size)];
            }

            // Compute h_(t-1) W_h*^T + b_h* for the same gate.
            float hidden_sum = b_hh[gru_bias_offset(layer, gate, h, hidden_size)];
            for (int k = 0; k < hidden_size; ++k) {
              hidden_sum += h_prev[k] *
                  w_hh[gru_whh_offset(layer, gate, h, k, hidden_size)];
            }

            input_gate[gate] = input_sum;
            hidden_gate[gate] = hidden_sum;
          }

          // PyTorch GRU equations:
          // r = sigmoid(i_r + h_r), z = sigmoid(i_z + h_z),
          // n = tanh(i_n + r * h_n), h_t = (1 - z) * n + z * h_(t-1).
          const float r = gru_sigmoid(input_gate[0] + hidden_gate[0]);
          const float z = gru_sigmoid(input_gate[1] + hidden_gate[1]);
          const float n = std::tanh(input_gate[2] + r * hidden_gate[2]);
          layer_out[(size_t)b * hidden_size + h] = n + z * (h_prev[h] - n);
        }
      }

      for (int b = 0; b < batch_size; ++b) {
        float* h_dst = hn + ((size_t)layer * batch_size + b) * hidden_size;
        for (int h = 0; h < hidden_size; ++h) {
          // Save h_t both as this layer's recurrent state and as the input to
          // the next stacked layer.
          h_dst[h] = layer_out[(size_t)b * hidden_size + h];
          layer_in[(size_t)b * hidden_size + h] = layer_out[(size_t)b * hidden_size + h];
        }
      }
    }

    // output[t] contains only the final layer's h_t for this time step.
    for (int i = 0; i < batch_size * hidden_size; ++i) {
      output[(size_t)t * batch_size * hidden_size + i] = layer_out[i];
    }
  }
}

#endif
