#pragma once

#include <vector>

#include "mlframework/layer.hpp"
#include "mlframework/tensor.hpp"

namespace mlf {

class MLP {
   public:
    MLP(size_t input_size, std::vector<size_t> hidden_sizes, size_t output_size,
        float dropout_ = 0.0F, bool use_batchnorm = false);

    TensorPtr forward(TensorPtr x);
    std::vector<TensorPtr> parameters();

   private:
    std::vector<Linear> layers_;
    std::vector<BatchNorm1d> bnorms_;
    float dropout_p_;
    bool use_batchnorm_;
};

}  // namespace mlf