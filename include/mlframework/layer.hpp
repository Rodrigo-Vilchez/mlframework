#pragma once

#include <vector>

#include "mlframework/tensor.hpp"

namespace mlf {

// Global training mode flag
inline bool& training_mode() {
    static bool flag = true;
    return flag;
}

struct TrainMode {
    TrainMode() { training_mode() = true; }
};
struct EvalMode {
    EvalMode() { training_mode() = false; }
};

class Linear {
   public:
    Linear(size_t in_features, size_t out_features);

    TensorPtr forward(TensorPtr x);
    std::vector<TensorPtr> parameters() const;

   private:
    TensorPtr weight_;
    TensorPtr bias_;
};

// Activations
TensorPtr relu(TensorPtr x);
TensorPtr sigmoid(TensorPtr x);
TensorPtr dropout(TensorPtr x, float p = 0.5F);

class BatchNorm1d {
   public:
    explicit BatchNorm1d(size_t num_features, float epsilon = 1e-5F, float momentum = 0.1F);

    TensorPtr forward(TensorPtr x);
    std::vector<TensorPtr> parameters() const;

   private:
    size_t num_features_;
    float epsilon_;
    float momentum_;

    TensorPtr gamma_;  // learnable scale
    TensorPtr beta_;   // learnable shift

    // running stats for inference (not learnable, not in parameters())
    std::vector<float> running_mean_;
    std::vector<float> running_var_;
};

}  // namespace mlf