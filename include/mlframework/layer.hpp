#pragma once

#include <random>
#include <vector>

#include "mlframework/module.hpp"
#include "mlframework/ops.hpp"

namespace mlf {

class Linear : public Module {
   public:
    Linear(size_t in_features, size_t out_features);
    TensorPtr forward(TensorPtr x) override;

   private:
    TensorPtr weight_;
    TensorPtr bias_;
};

class BatchNorm1d : public Module {
   public:
    explicit BatchNorm1d(size_t num_features, float epsilon = 1e-5F, float momentum = 0.1F);
    TensorPtr forward(TensorPtr x) override;

   private:
    size_t num_features_;
    float epsilon_, momentum_;
    TensorPtr gamma_;
    TensorPtr beta_;
    std::vector<float> running_mean_;
    std::vector<float> running_var_;
};

class Conv2d : public Module {
   public:
    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1,
           size_t padding = 0);
    TensorPtr forward(TensorPtr x) override;

   private:
    size_t in_channels_, out_channels_, kernel_size_, stride_, padding_;
    TensorPtr weight_;
    TensorPtr bias_;
};

class MaxPool2d : public Module {
   public:
    explicit MaxPool2d(size_t kernel_size, size_t stride = 0);
    TensorPtr forward(TensorPtr x) override;

   private:
    size_t kernel_size_, stride_;
};

// activations - free functions, not modules
TensorPtr relu(TensorPtr x);
TensorPtr sigmoid(TensorPtr x);
TensorPtr dropout(TensorPtr x, float p = 0.5F);
TensorPtr flatten(TensorPtr x);

}  // namespace mlf