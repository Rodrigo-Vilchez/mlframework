#pragma once

#include <vector>

#include "mlframework/tensor.hpp"

namespace mlf {

class SGD {
   public:
    SGD(std::vector<TensorPtr> params, float lr, float momentum = 0.0F);

    void step();
    void zero_grad();

   private:
    std::vector<TensorPtr> params_;
    float lr_;
    float momentum_;
    std::vector<std::vector<float>> velocity_;
};

}  // namespace mlf