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
    std::vector<TensorPtr> parameters();

   private:
    TensorPtr weight_;
    TensorPtr bias_;
};

// Activations
TensorPtr relu(TensorPtr x);
TensorPtr sigmoid(TensorPtr x);
TensorPtr dropout(TensorPtr x, float p = 0.5F);

}  // namespace mlf