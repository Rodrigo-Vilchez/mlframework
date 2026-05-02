#include "mlframework/mlp.hpp"

#include "mlframework/layer.hpp"

namespace mlf {

MLP::MLP(size_t input_size, std::vector<size_t> hidden_sizes, size_t output_size, float dropout_p,
         bool use_batchnorm)
    : dropout_p_(dropout_p), use_batchnorm_(use_batchnorm) {
    std::vector<size_t> sizes;
    sizes.push_back(input_size);
    for (size_t h : hidden_sizes) sizes.push_back(h);
    sizes.push_back(output_size);

    for (size_t i = 0; i + 1 < sizes.size(); i++) {
        layers_.push_back(std::make_shared<Linear>(sizes[i], sizes[i + 1]));
        register_module("linear" + std::to_string(i), *layers_.back());

        if (use_batchnorm_ && i + 2 < sizes.size()) {
            bnorms_.push_back(std::make_shared<BatchNorm1d>(sizes[i + 1]));
            register_module("bn" + std::to_string(i), *bnorms_.back());
        }
    }
}

TensorPtr MLP::forward(TensorPtr x) {
    TensorPtr out = x;
    size_t bn_idx = 0;
    for (size_t i = 0; i < layers_.size(); i++) {
        out = layers_[i]->forward(out);
        if (i + 1 < layers_.size()) {
            if (use_batchnorm_) out = bnorms_[bn_idx++]->forward(out);
            out = relu(out);
            out = dropout(out, dropout_p_);
        }
    }
    return out;
}

}  // namespace mlf