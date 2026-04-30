#include "mlframework/layer.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

#include "mlframework/ops.hpp"

namespace mlf {

Linear::Linear(size_t in_features, size_t out_features) {
    // Xavier initialization: uniform(-1/sqrt(in), 1/sqrt(in))
    std::mt19937 rng(42);
    float bound = 1.0F / std::sqrt(static_cast<float>(in_features));
    std::uniform_real_distribution<float> dist(-bound, bound);

    std::vector<float> w_data(in_features * out_features);
    for (auto& v : w_data) v = dist(rng);

    weight_ = make_tensor({in_features, out_features}, w_data, true);
    bias_ = make_tensor({out_features}, std::vector<float>(out_features, 0.0F), true);
}

TensorPtr Linear::forward(TensorPtr x) {
    if (x->shape.size() != 2) {
        throw std::invalid_argument("Linear::forward expects a 2D input {batch, in_features}");
    }
    return ops::add_bias(ops::matmul(x, weight_), bias_);
}

std::vector<TensorPtr> Linear::parameters() { return {weight_, bias_}; }

// --- Activations ---

TensorPtr relu(TensorPtr x) {
    auto result = make_tensor(x->shape, x->requires_grad);
    for (size_t i = 0; i < x->numel(); i++) {
        result->data[i] = x->data[i] > 0.0F ? x->data[i] : 0.0F;
    }
    if (x->requires_grad) {
        result->inputs = {x};
        result->backward_fn = [x, r = result.get()]() {
            for (size_t i = 0; i < x->numel(); i++) {
                if (x->requires_grad) x->grad[i] += r->grad[i] * (x->data[i] > 0.0F ? 1.0F : 0.0F);
            }
        };
    }
    return result;
}

TensorPtr sigmoid(TensorPtr x) {
    auto result = make_tensor(x->shape, x->requires_grad);
    for (size_t i = 0; i < x->numel(); i++) {
        result->data[i] = 1.0F / (1.0F + std::exp(-x->data[i]));
    }
    if (x->requires_grad) {
        result->inputs = {x};
        result->backward_fn = [x, r = result.get()]() {
            for (size_t i = 0; i < x->numel(); i++) {
                if (x->requires_grad) {
                    float s = r->data[i];
                    x->grad[i] += r->grad[i] * s * (1.0F - s);
                }
            }
        };
    }
    return result;
}

}  // namespace mlf