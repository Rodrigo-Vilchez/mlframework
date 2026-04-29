#include "mlframework/mlp.hpp"

namespace mlf {

MLP::MLP(size_t input_size, std::vector<size_t> hidden_sizes, size_t output_size) {
    std::vector<size_t> sizes;
    sizes.push_back(input_size);
    for (size_t h : hidden_sizes) sizes.push_back(h);
    sizes.push_back(output_size);

    for (size_t i = 0; i + 1 < sizes.size(); i++) {
        layers_.emplace_back(sizes[i], sizes[i + 1]);
    }
}

TensorPtr MLP::forward(TensorPtr x) {
    TensorPtr out = x;
    for (size_t i = 0; i < layers_.size(); i++) {
        out = layers_[i].forward(out);
        // ReLU on all layers except the last
        if (i + 1 < layers_.size()) {
            out = relu(out);
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
    return params;
}

}  // namespace mlf