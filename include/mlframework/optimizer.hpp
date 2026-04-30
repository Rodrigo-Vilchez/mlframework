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

class Adam {
   public:
    Adam(std::vector<TensorPtr> params, float lr = 0.001F, float beta1 = 0.9F, float beta2 = 0.999F,
         float epsilon = 1e-8F);

    void step();
    void zero_grad();

   private:
    std::vector<TensorPtr> params_;
    float lr_, beta1_, beta2_, epsilon_;
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
    size_t t_{0};
};

}  // namespace mlf