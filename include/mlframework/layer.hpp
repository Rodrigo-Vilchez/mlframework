#pragma once

#include <vector>

#include "mlframework/tensor.hpp"

namespace mlf {

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

class Conv2d {
   public:
    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1,
           size_t padding = 0);

    TensorPtr forward(TensorPtr x);  // x: {N, C, H, W}
    std::vector<TensorPtr> parameters() const;

   private:
    size_t in_channels_, out_channels_, kernel_size_, stride_, padding_;
    TensorPtr weight_;  // {out_channels, in_channels, K, K}
    TensorPtr bias_;    // {out_channels}
};

TensorPtr flatten(TensorPtr x);  // {N, ...} -> {N, numel/N}

class MaxPool2d {
   public:
    explicit MaxPool2d(size_t kernel_size, size_t stride = 0);
    // stride=0 means stride=kernel_size (non-overlapping windows)

    TensorPtr forward(TensorPtr x);  // x: {N, C, H, W}

   private:
    size_t kernel_size_, stride_;
};

}  // namespace mlf