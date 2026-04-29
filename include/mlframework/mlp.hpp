#pragma once

#include <vector>

#include "mlframework/layer.hpp"
#include "mlframework/tensor.hpp"

namespace mlf {

class MLP {
   public:
    MLP(size_t input_size, std::vector<size_t> hidden_sizes, size_t output_size);

    TensorPtr forward(TensorPtr x);
    std::vector<TensorPtr> parameters();

   private:
    std::vector<Linear> layers_;
};

}  // namespace mlf