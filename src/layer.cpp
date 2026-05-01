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

Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride,
               size_t padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {
    std::mt19937 rng(42);
    float bound = 1.0F / std::sqrt(static_cast<float>(in_channels * kernel_size * kernel_size));
    std::uniform_real_distribution<float> dist(-bound, bound);

    size_t w_size = out_channels * in_channels * kernel_size * kernel_size;
    std::vector<float> w_data(w_size);
    for (auto& v : w_data) v = dist(rng);

    weight_ = make_tensor({out_channels, in_channels, kernel_size, kernel_size}, w_data, true);
    bias_ = make_tensor({out_channels}, std::vector<float>(out_channels, 0.0F), true);
}

TensorPtr Conv2d::forward(TensorPtr x) {
    if (x->shape.size() != 4) {
        throw std::invalid_argument("Conv2d::forward expects {N, C, H, W}");
    }
    size_t N = x->shape[0];
    size_t C = x->shape[1];
    size_t H = x->shape[2];
    size_t W = x->shape[3];
    size_t K = kernel_size_;
    size_t F = out_channels_;
    size_t P = padding_;
    size_t S = stride_;
    size_t OH = (H + 2 * P - K) / S + 1;
    size_t OW = (W + 2 * P - K) / S + 1;

    bool rg = x->requires_grad || weight_->requires_grad;
    auto result = make_tensor({N, F, OH, OW}, rg);

    // forward pass — direct convolution
    for (size_t n = 0; n < N; n++) {
        for (size_t f = 0; f < F; f++) {
            for (size_t i = 0; i < OH; i++) {
                for (size_t j = 0; j < OW; j++) {
                    float sum = bias_->data[f];
                    for (size_t c = 0; c < C; c++) {
                        for (size_t ki = 0; ki < K; ki++) {
                            for (size_t kj = 0; kj < K; kj++) {
                                long ih = static_cast<long>(i * S + ki) - static_cast<long>(P);
                                long iw = static_cast<long>(j * S + kj) - static_cast<long>(P);
                                if (ih < 0 || iw < 0 || ih >= static_cast<long>(H) ||
                                    iw >= static_cast<long>(W))
                                    continue;
                                size_t x_idx = n * (C * H * W) + c * (H * W) +
                                               static_cast<size_t>(ih) * W +
                                               static_cast<size_t>(iw);
                                size_t w_idx = f * (C * K * K) + c * (K * K) + ki * K + kj;
                                sum += x->data[x_idx] * weight_->data[w_idx];
                            }
                        }
                    }
                    result->data[n * (F * OH * OW) + f * (OH * OW) + i * OW + j] = sum;
                }
            }
        }
    }

    if (rg) {
        result->inputs = {x, weight_, bias_};
        result->backward_fn = [x, w = weight_.get(), b = bias_.get(), r = result.get(), N, C, H, W,
                               F, K, OH, OW, P, S]() {
            for (size_t n = 0; n < N; n++) {
                for (size_t f = 0; f < F; f++) {
                    for (size_t i = 0; i < OH; i++) {
                        for (size_t j = 0; j < OW; j++) {
                            float g = r->grad[n * (F * OH * OW) + f * (OH * OW) + i * OW + j];

                            // grad_bias
                            if (b->requires_grad) b->grad[f] += g;

                            for (size_t c = 0; c < C; c++) {
                                for (size_t ki = 0; ki < K; ki++) {
                                    for (size_t kj = 0; kj < K; kj++) {
                                        long ih =
                                            static_cast<long>(i * S + ki) - static_cast<long>(P);
                                        long iw =
                                            static_cast<long>(j * S + kj) - static_cast<long>(P);
                                        if (ih < 0 || iw < 0 || ih >= static_cast<long>(H) ||
                                            iw >= static_cast<long>(W))
                                            continue;
                                        size_t x_idx = n * (C * H * W) + c * (H * W) +
                                                       static_cast<size_t>(ih) * W +
                                                       static_cast<size_t>(iw);
                                        size_t w_idx = f * (C * K * K) + c * (K * K) + ki * K + kj;
                                        // grad_weight
                                        if (w->requires_grad) w->grad[w_idx] += g * x->data[x_idx];
                                        // grad_input
                                        if (x->requires_grad) x->grad[x_idx] += g * w->data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
    }
    return result;
}

std::vector<TensorPtr> Conv2d::parameters() { return {weight_, bias_}; }

TensorPtr flatten(TensorPtr x) {
    size_t N = x->shape[0];
    size_t rest = x->numel() / N;
    auto result = make_tensor({N, rest}, x->requires_grad);
    result->data = x->data;

    if (x->requires_grad) {
        result->inputs = {x};
        result->backward_fn = [x, r = result.get()]() {
            for (size_t i = 0; i < x->numel(); i++) {
                x->grad[i] += r->grad[i];
            }
        };
    }
    return result;
}

}  // namespace mlf