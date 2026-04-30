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
        layers_.emplace_back(sizes[i], sizes[i + 1]);
        if (use_batchnorm_ && i + 2 < sizes.size()) {
            bnorms_.emplace_back(sizes[i + 1]);
        }
    }
}

TensorPtr MLP::forward(TensorPtr x) {
    TensorPtr out = x;
    size_t bn_idx = 0;
    for (size_t i = 0; i < layers_.size(); i++) {
        out = layers_[i].forward(out);
        // ReLU on all layers except the last
        if (i + 1 < layers_.size()) {
            if (use_batchnorm_) {
                out = bnorms_[bn_idx++].forward(out);
            }
            out = relu(out);
            out = dropout(out, dropout_p_);
        }
    }
    return out;
}

std::vector<TensorPtr> MLP::parameters() {
    std::vector<TensorPtr> params;
    for (auto& layer : layers_) {
        for (auto& p : layer.parameters()) {
            params.push_back(p);
        }
    }
    for (auto& bn : bnorms_) {
        for (auto& p : bn.parameters()) {
            params.push_back(p);
        }
    }
    return params;
}

}  // namespace mlf