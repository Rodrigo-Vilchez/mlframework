#pragma once

#include <vector>

#include "mlframework/tensor.hpp"

namespace mlf {

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

}  // namespace mlf