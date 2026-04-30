#include "mlframework/optimizer.hpp"

#include <cmath>
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

Adam::Adam(std::vector<TensorPtr> params, float lr, float beta1, float beta2, float epsilon)
    : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); i++) {
        m_[i].assign(params_[i]->numel(), 0.0F);
        v_[i].assign(params_[i]->numel(), 0.0F);
    }
}

void Adam::step() {
    t_++;
    float bc1 = 1.0F - std::pow(beta1_, static_cast<float>(t_));
    float bc2 = 1.0F - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params_.size(); i++) {
        auto& p = params_[i];
        if (!p->requires_grad || p->grad.empty()) continue;
        for (size_t j = 0; j < p->numel(); j++) {
            float g = p->grad[j];
            m_[i][j] = beta1_ * m_[i][j] + (1.0F - beta1_) * g;
            v_[i][j] = beta2_ * v_[i][j] + (1.0F - beta2_) * g * g;
            float m_hat = m_[i][j] / bc1;
            float v_hat = v_[i][j] / bc2;
            p->data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

void Adam::zero_grad() {
    for (auto& p : params_) {
        p->zero_grad();
    }
}

}  // namespace mlf