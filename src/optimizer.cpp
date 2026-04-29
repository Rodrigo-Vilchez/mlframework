#include "mlframework/optimizer.hpp"

#include <stdexcept>

namespace mlf {

SGD::SGD(std::vector<TensorPtr> params, float lr, float momentum)
    : params_(std::move(params)), lr_(lr), momentum_(momentum) {
    if (lr_ <= 0.0F) {
        throw std::invalid_argument("learning rate must be positive");
    }
    if (momentum_ < 0.0F || momentum_ >= 1.0F) {
        throw std::invalid_argument("momentum must be in [0, 1)");
    }
    velocity_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); i++) {
        velocity_[i].assign(params_[i]->numel(), 0.0F);
    }
}

void SGD::step() {
    for (size_t i = 0; i < params_.size(); i++) {
        auto& p = params_[i];
        if (!p->requires_grad || p->grad.empty()) continue;
        for (size_t j = 0; j < p->numel(); j++) {
            velocity_[i][j] = momentum_ * velocity_[i][j] + p->grad[j];
            p->data[j] -= lr_ * velocity_[i][j];
        }
    }
}

void SGD::zero_grad() {
    for (auto& p : params_) {
        p->zero_grad();
    }
}

}  // namespace mlf