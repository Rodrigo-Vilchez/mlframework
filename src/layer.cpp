#include "mlframework/layer.hpp"

#include <cblas.h>

#include <cmath>
#include <random>
#include <stdexcept>

#include "mlframework/ops.hpp"
#include "mlframework/ops_cuda.hpp"

namespace mlf {

Linear::Linear(size_t in_features, size_t out_features) {
    std::mt19937 rng(42);
    float bound = 1.0F / std::sqrt(static_cast<float>(in_features));
    std::uniform_real_distribution<float> dist(-bound, bound);

    std::vector<float> w_data(in_features * out_features);
    for (auto& v : w_data) v = dist(rng);

    weight_ = make_tensor({in_features, out_features}, w_data, true);
    bias_ = make_tensor({out_features}, std::vector<float>(out_features, 0.0F), true);

    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
}

TensorPtr Linear::forward(TensorPtr x) {
    if (x->shape.size() != 2) {
        throw std::invalid_argument("Linear::forward expects a 2D input {batch, in_features}");
    }
    if (x->device == Device::CUDA) {
        return cuda::add_bias_cuda(cuda::matmul_cuda(x, weight_), bias_);
    }
    return ops::add_bias(ops::matmul(x, weight_), bias_);
}

// --- Activations ---

TensorPtr relu(TensorPtr x) {
    if (x->device == Device::CUDA) return cuda::relu_cuda(x);
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

    bool on_cuda = (x->device == Device::CUDA);
    TensorPtr x_cpu = on_cuda ? to_device(x, Device::CPU) : x;

    static std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution dist(1.0F - p);

    auto result = make_tensor(x_cpu->shape, x_cpu->requires_grad);
    std::vector<float> mask(x->numel());

    float scale = 1.0F / (1.0F - p);  // inverted dropout scaling
    for (size_t i = 0; i < x_cpu->numel(); i++) {
        mask[i] = dist(rng) ? scale : 0.0F;
        result->data[i] = x_cpu->data[i] * mask[i];
    }

    if (x_cpu->requires_grad) {
        result->inputs = {x_cpu};
        result->backward_fn = [x_cpu, r = result.get(), mask]() {
            for (size_t i = 0; i < x_cpu->numel(); i++) {
                if (x_cpu->requires_grad) x_cpu->grad[i] += r->grad[i] * mask[i];
            }
        };
    }
    if (on_cuda) return to_device(result, Device::CUDA);
    return result;
}

BatchNorm1d::BatchNorm1d(size_t num_features, float epsilon, float momentum)
    : num_features_(num_features), epsilon_(epsilon), momentum_(momentum) {
    gamma_ = make_tensor({num_features}, std::vector<float>(num_features, 1.0F), true);
    beta_ = make_tensor({num_features}, std::vector<float>(num_features, 0.0F), true);
    running_mean_.assign(num_features, 0.0F);
    running_var_.assign(num_features, 1.0F);

    register_parameter("gamma", gamma_);
    register_parameter("beta", beta_);
}

TensorPtr BatchNorm1d::forward(TensorPtr x) {
    // CUDA: bring to CPU, compute, return to CUDA
    bool on_cuda = (x->device == Device::CUDA);
    TensorPtr x_cpu = on_cuda ? to_device(x, Device::CPU) : x;
    if (on_cuda) {
        x_cpu = mlf::to(x_cpu, Device::CPU);
        gamma_ = mlf::to(gamma_, Device::CPU);
        beta_ = mlf::to(beta_, Device::CPU);
    }

    if (x_cpu->shape.size() != 2 || x_cpu->shape[1] != num_features_) {
        throw std::invalid_argument("BatchNorm1d: input must be {batch, num_features}");
    }
    size_t B = x_cpu->shape[0];
    size_t F = num_features_;

    bool rg = x_cpu->requires_grad;
    auto result = make_tensor({B, F}, rg);

    if (training_mode()) {
        // compute batch mean and variance per feature
        std::vector<float> mean(F, 0.0F);
        std::vector<float> var(F, 0.0F);

        for (size_t j = 0; j < F; j++) {
            for (size_t i = 0; i < B; i++) {
                mean[j] += x_cpu->data[i * F + j];
            }
            mean[j] /= static_cast<float>(B);

            for (size_t i = 0; i < B; i++) {
                float diff = x_cpu->data[i * F + j] - mean[j];
                var[j] += diff * diff;
            }
            var[j] /= static_cast<float>(B);
        }

        // normalize and apply gamma/beta
        std::vector<float> x_norm(B * F);
        for (size_t i = 0; i < B; i++) {
            for (size_t j = 0; j < F; j++) {
                x_norm[i * F + j] =
                    (x_cpu->data[i * F + j] - mean[j]) / std::sqrt(var[j] + epsilon_);
                result->data[i * F + j] = gamma_->data[j] * x_norm[i * F + j] + beta_->data[j];
            }
        }

        // update running stats
        for (size_t j = 0; j < F; j++) {
            running_mean_[j] = (1.0F - momentum_) * running_mean_[j] + momentum_ * mean[j];
            running_var_[j] = (1.0F - momentum_) * running_var_[j] + momentum_ * var[j];
        }

        if (rg) {
            result->inputs = {x_cpu, gamma_, beta_};
            result->backward_fn = [x_cpu, g = gamma_.get(), b = beta_.get(), r = result.get(),
                                   x_norm, mean, var, B, F, ep = epsilon_]() {
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

                if (x_cpu->requires_grad) {
                    for (size_t i = 0; i < B * F; i++) x_cpu->grad[i] += dx[i];
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
                float x_norm = (x_cpu->data[i * F + j] - running_mean_[j]) /
                               std::sqrt(running_var_[j] + epsilon_);
                result->data[i * F + j] = gamma_->data[j] * x_norm + beta_->data[j];
            }
        }
    }

    if (on_cuda) return to_device(result, Device::CUDA);
    return result;
}

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

    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
}

TensorPtr Conv2d::forward(TensorPtr x) {
    if (x->shape.size() != 4) {
        throw std::invalid_argument("Conv2d::forward expects {N, C, H, W}");
    }

    bool on_cuda = (x->device == Device::CUDA);

    // local CPU views — to_device creates autograd nodes so grads
    // flow back to the originals (x, weight_, bias_) on CUDA
    TensorPtr x_cpu = on_cuda ? to_device(x, Device::CPU) : x;
    TensorPtr weight_cpu = on_cuda ? to_device(weight_, Device::CPU) : weight_;
    TensorPtr bias_cpu = on_cuda ? to_device(bias_, Device::CPU) : bias_;

    size_t N = x_cpu->shape[0];
    size_t C = x_cpu->shape[1];
    size_t H = x_cpu->shape[2];
    size_t W = x_cpu->shape[3];
    size_t K = kernel_size_;
    size_t F = out_channels_;
    size_t P = padding_;
    size_t S = stride_;
    size_t OH = (H + 2 * P - K) / S + 1;
    size_t OW = (W + 2 * P - K) / S + 1;

    size_t col_rows = C * K * K;
    size_t col_cols = OH * OW;
    std::vector<float> col(N * col_rows * col_cols, 0.0F);

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t ki = 0; ki < K; ki++) {
                for (size_t kj = 0; kj < K; kj++) {
                    size_t row = c * K * K + ki * K + kj;
                    for (size_t i = 0; i < OH; i++) {
                        for (size_t j = 0; j < OW; j++) {
                            long ih = static_cast<long>(i * S + ki) - static_cast<long>(P);
                            long iw = static_cast<long>(j * S + kj) - static_cast<long>(P);
                            size_t col_idx =
                                n * (col_rows * col_cols) + row * col_cols + i * OW + j;
                            if (ih >= 0 && iw >= 0 && ih < static_cast<long>(H) &&
                                iw < static_cast<long>(W)) {
                                col[col_idx] = x_cpu->data[n * (C * H * W) + c * (H * W) +
                                                           static_cast<size_t>(ih) * W +
                                                           static_cast<size_t>(iw)];
                            }
                        }
                    }
                }
            }
        }
    }

    bool rg = x_cpu->requires_grad || weight_cpu->requires_grad;
    auto result = make_tensor({N, F, OH, OW}, rg);

    for (size_t n = 0; n < N; n++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(F),
                    static_cast<int>(col_cols), static_cast<int>(col_rows), 1.0F,
                    weight_cpu->data.data(), static_cast<int>(col_rows),
                    col.data() + n * col_rows * col_cols, static_cast<int>(col_cols), 0.0F,
                    result->data.data() + n * F * col_cols, static_cast<int>(col_cols));
        for (size_t f = 0; f < F; f++)
            for (size_t k = 0; k < col_cols; k++)
                result->data[n * (F * col_cols) + f * col_cols + k] += bias_cpu->data[f];
    }

    if (rg) {
        result->inputs = {x_cpu, weight_cpu, bias_cpu};
        result->backward_fn = [x_cpu, w = weight_cpu.get(), b = bias_cpu.get(), r = result.get(), N,
                               C, H, W, F, K, OH, OW, P, S, col_rows, col_cols]() {
            // gradient checkpointing: recompute col
            std::vector<float> col(N * col_rows * col_cols, 0.0F);
            for (size_t n = 0; n < N; n++) {
                for (size_t c = 0; c < C; c++) {
                    for (size_t ki = 0; ki < K; ki++) {
                        for (size_t kj = 0; kj < K; kj++) {
                            size_t row = c * K * K + ki * K + kj;
                            for (size_t i = 0; i < OH; i++) {
                                for (size_t j = 0; j < OW; j++) {
                                    long ih = static_cast<long>(i * S + ki) - static_cast<long>(P);
                                    long iw = static_cast<long>(j * S + kj) - static_cast<long>(P);
                                    size_t col_idx =
                                        n * (col_rows * col_cols) + row * col_cols + i * OW + j;
                                    if (ih >= 0 && iw >= 0 && ih < static_cast<long>(H) &&
                                        iw < static_cast<long>(W)) {
                                        col[col_idx] = x_cpu->data[n * (C * H * W) + c * (H * W) +
                                                                   static_cast<size_t>(ih) * W +
                                                                   static_cast<size_t>(iw)];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (b->requires_grad) {
                for (size_t n = 0; n < N; n++)
                    for (size_t f = 0; f < F; f++)
                        for (size_t k = 0; k < col_cols; k++)
                            b->grad[f] += r->grad[n * (F * col_cols) + f * col_cols + k];
            }

            for (size_t n = 0; n < N; n++) {
                const float* grad_out = r->grad.data() + n * F * col_cols;
                const float* col_n = col.data() + n * col_rows * col_cols;

                if (w->requires_grad) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(F),
                                static_cast<int>(col_rows), static_cast<int>(col_cols), 1.0F,
                                grad_out, static_cast<int>(col_cols), col_n,
                                static_cast<int>(col_cols), 1.0F, w->grad.data(),
                                static_cast<int>(col_rows));
                }

                if (x_cpu->requires_grad) {
                    std::vector<float> grad_col(col_rows * col_cols, 0.0F);
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(col_rows),
                                static_cast<int>(col_cols), static_cast<int>(F), 1.0F,
                                w->data.data(), static_cast<int>(col_rows), grad_out,
                                static_cast<int>(col_cols), 0.0F, grad_col.data(),
                                static_cast<int>(col_cols));

                    for (size_t c = 0; c < C; c++) {
                        for (size_t ki = 0; ki < K; ki++) {
                            for (size_t kj = 0; kj < K; kj++) {
                                size_t row = c * K * K + ki * K + kj;
                                for (size_t i = 0; i < OH; i++) {
                                    for (size_t j = 0; j < OW; j++) {
                                        long ih =
                                            static_cast<long>(i * S + ki) - static_cast<long>(P);
                                        long iw =
                                            static_cast<long>(j * S + kj) - static_cast<long>(P);
                                        if (ih < 0 || iw < 0 || ih >= static_cast<long>(H) ||
                                            iw >= static_cast<long>(W))
                                            continue;
                                        x_cpu->grad[n * (C * H * W) + c * (H * W) +
                                                    static_cast<size_t>(ih) * W +
                                                    static_cast<size_t>(iw)] +=
                                            grad_col[row * col_cols + i * OW + j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    if (on_cuda) return to_device(result, Device::CUDA);
    return result;
}

TensorPtr flatten(TensorPtr x) {
    bool on_cuda = (x->device == Device::CUDA);
    TensorPtr x_cpu = on_cuda ? to_device(x, Device::CPU) : x;

    size_t N = x_cpu->shape[0];
    size_t rest = x_cpu->numel() / N;
    auto result = make_tensor({N, rest}, x_cpu->requires_grad);
    result->data = x_cpu->data;

    if (x_cpu->requires_grad) {
        result->inputs = {x_cpu};
        result->backward_fn = [x_cpu, r = result.get()]() {
            for (size_t i = 0; i < x_cpu->numel(); i++) {
                x_cpu->grad[i] += r->grad[i];
            }
        };
    }

    if (on_cuda) return to_device(result, Device::CUDA);
    return result;
}

MaxPool2d::MaxPool2d(size_t kernel_size, size_t stride)
    : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride) {}

TensorPtr MaxPool2d::forward(TensorPtr x) {
    if (x->shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d::forward expects {N, C, H, W}");
    }

    bool on_cuda = (x->device == Device::CUDA);
    TensorPtr x_cpu = on_cuda ? to_device(x, Device::CPU) : x;

    size_t N = x_cpu->shape[0];
    size_t C = x_cpu->shape[1];
    size_t H = x_cpu->shape[2];
    size_t W = x_cpu->shape[3];
    size_t K = kernel_size_;
    size_t S = stride_;
    size_t OH = (H - K) / S + 1;
    size_t OW = (W - K) / S + 1;

    auto result = make_tensor({N, C, OH, OW}, x_cpu->requires_grad);
    std::vector<size_t> argmax(N * C * OH * OW);

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t i = 0; i < OH; i++) {
                for (size_t j = 0; j < OW; j++) {
                    float max_val = -std::numeric_limits<float>::max();
                    size_t max_idx = 0;
                    for (size_t ki = 0; ki < K; ki++) {
                        for (size_t kj = 0; kj < K; kj++) {
                            size_t ih = i * S + ki;
                            size_t iw = j * S + kj;
                            size_t idx = n * (C * H * W) + c * (H * W) + ih * W + iw;
                            if (x_cpu->data[idx] > max_val) {
                                max_val = x_cpu->data[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    size_t out_idx = n * (C * OH * OW) + c * (OH * OW) + i * OW + j;
                    result->data[out_idx] = max_val;
                    argmax[out_idx] = max_idx;
                }
            }
        }
    }

    if (x_cpu->requires_grad) {
        result->inputs = {x_cpu};
        result->backward_fn = [x_cpu, r = result.get(), argmax]() {
            for (size_t i = 0; i < r->numel(); i++) {
                x_cpu->grad[argmax[i]] += r->grad[i];
            }
        };
    }

    if (on_cuda) return to_device(result, Device::CUDA);
    return result;
}

}  // namespace mlf