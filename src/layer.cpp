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

std::vector<TensorPtr> Linear::parameters() const { return {weight_, bias_}; }

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

TensorPtr dropout(TensorPtr x, float p) {
    if (!training_mode() || p == 0.0F) return x;

    static std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution dist(1.0F - p);

    auto result = make_tensor(x->shape, x->requires_grad);
    std::vector<float> mask(x->numel());

    float scale = 1.0F / (1.0F - p);  // inverted dropout scaling
    for (size_t i = 0; i < x->numel(); i++) {
        mask[i] = dist(rng) ? scale : 0.0F;
        result->data[i] = x->data[i] * mask[i];
    }

    if (x->requires_grad) {
        result->inputs = {x};
        result->backward_fn = [x, r = result.get(), mask]() {
            for (size_t i = 0; i < x->numel(); i++) {
                if (x->requires_grad) x->grad[i] += r->grad[i] * mask[i];
            }
        };
    }
    return result;
}

BatchNorm1d::BatchNorm1d(size_t num_features, float epsilon, float momentum)
    : num_features_(num_features), epsilon_(epsilon), momentum_(momentum) {
    gamma_ = make_tensor({num_features}, std::vector<float>(num_features, 1.0F), true);
    beta_ = make_tensor({num_features}, std::vector<float>(num_features, 0.0F), true);
    running_mean_.assign(num_features, 0.0F);
    running_var_.assign(num_features, 1.0F);
}

TensorPtr BatchNorm1d::forward(TensorPtr x) {
    if (x->shape.size() != 2 || x->shape[1] != num_features_) {
        throw std::invalid_argument("BatchNorm1d: input must be {batch, num_features}");
    }
    size_t B = x->shape[0];
    size_t F = num_features_;

    bool rg = x->requires_grad;
    auto result = make_tensor({B, F}, rg);

    if (training_mode()) {
        // compute batch mean and variance per feature
        std::vector<float> mean(F, 0.0F);
        std::vector<float> var(F, 0.0F);

        for (size_t j = 0; j < F; j++) {
            for (size_t i = 0; i < B; i++) {
                mean[j] += x->data[i * F + j];
            }
            mean[j] /= static_cast<float>(B);

            for (size_t i = 0; i < B; i++) {
                float diff = x->data[i * F + j] - mean[j];
                var[j] += diff * diff;
            }
            var[j] /= static_cast<float>(B);
        }

        // normalize and apply gamma/beta
        std::vector<float> x_norm(B * F);
        for (size_t i = 0; i < B; i++) {
            for (size_t j = 0; j < F; j++) {
                x_norm[i * F + j] = (x->data[i * F + j] - mean[j]) / std::sqrt(var[j] + epsilon_);
                result->data[i * F + j] = gamma_->data[j] * x_norm[i * F + j] + beta_->data[j];
            }
        }

        // update running stats
        for (size_t j = 0; j < F; j++) {
            running_mean_[j] = (1.0F - momentum_) * running_mean_[j] + momentum_ * mean[j];
            running_var_[j] = (1.0F - momentum_) * running_var_[j] + momentum_ * var[j];
        }

        if (rg) {
            result->inputs = {x, gamma_, beta_};
            result->backward_fn = [x, g = gamma_.get(), b = beta_.get(), r = result.get(), x_norm,
                                   mean, var, B, F, ep = epsilon_]() {
                std::vector<float> dgamma(F, 0.0F);
                std::vector<float> dbeta(F, 0.0F);
                std::vector<float> dx(B * F, 0.0F);

                for (size_t j = 0; j < F; j++) {
                    float inv_std = 1.0F / std::sqrt(var[j] + ep);

                    for (size_t i = 0; i < B; i++) {
                        float dout = r->grad[i * F + j];
                        dgamma[j] += dout * x_norm[i * F + j];
                        dbeta[j] += dout;
                    }

                    // gradient through normalization
                    float dxnorm_sum = 0.0F;
                    float dxnorm_xsum = 0.0F;
                    for (size_t i = 0; i < B; i++) {
                        float dxnorm = r->grad[i * F + j] * g->data[j];
                        dxnorm_sum += dxnorm;
                        dxnorm_xsum += dxnorm * x_norm[i * F + j];
                    }
                    for (size_t i = 0; i < B; i++) {
                        float dxnorm = r->grad[i * F + j] * g->data[j];
                        dx[i * F + j] = inv_std / static_cast<float>(B) *
                                        (static_cast<float>(B) * dxnorm - dxnorm_sum -
                                         x_norm[i * F + j] * dxnorm_xsum);
                    }
                }

                if (x->requires_grad) {
                    for (size_t i = 0; i < B * F; i++) x->grad[i] += dx[i];
                }
                for (size_t j = 0; j < F; j++) {
                    g->grad[j] += dgamma[j];
                    b->grad[j] += dbeta[j];
                }
            };
        }

    } else {
        // inference: use running stats
        for (size_t i = 0; i < B; i++) {
            for (size_t j = 0; j < F; j++) {
                float x_norm =
                    (x->data[i * F + j] - running_mean_[j]) / std::sqrt(running_var_[j] + epsilon_);
                result->data[i * F + j] = gamma_->data[j] * x_norm + beta_->data[j];
            }
        }
    }

    return result;
}

std::vector<TensorPtr> BatchNorm1d::parameters() const { return {gamma_, beta_}; }

}  // namespace mlf