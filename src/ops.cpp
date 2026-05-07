#include "mlframework/ops.hpp"

#include <cblas.h>

#include <stdexcept>

#include "mlframework/ops_cuda.hpp"

namespace mlf::ops {

static void check_same_shape(const TensorPtr& a, const TensorPtr& b) {
    if (a->shape != b->shape) {
        throw std::invalid_argument("tensors must have the same shape");
    }
}

static bool any_requires_grad(const TensorPtr& a, const TensorPtr& b) {
    return !no_grad_mode() && (a->requires_grad || b->requires_grad);
}

TensorPtr add(TensorPtr a, TensorPtr b) {
    if (a->device == Device::CUDA) return cuda::add_cuda(a, b);
    check_same_shape(a, b);
    bool rg = any_requires_grad(a, b);
    auto result = make_tensor(a->shape, rg);
    for (size_t i = 0; i < a->numel(); i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get()]() {
            for (size_t i = 0; i < r->numel(); i++) {
                if (a->requires_grad) a->grad[i] += r->grad[i];
                if (b->requires_grad) b->grad[i] += r->grad[i];
            }
        };
    }
    return result;
}

TensorPtr sub(TensorPtr a, TensorPtr b) {
    if (a->device == Device::CUDA) return cuda::sub_cuda(a, b);
    check_same_shape(a, b);
    bool rg = any_requires_grad(a, b);
    auto result = make_tensor(a->shape, rg);
    for (size_t i = 0; i < a->numel(); i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get()]() {
            for (size_t i = 0; i < r->numel(); i++) {
                if (a->requires_grad) a->grad[i] += r->grad[i];
                if (b->requires_grad) b->grad[i] -= r->grad[i];
            }
        };
    }
    return result;
}

TensorPtr mul(TensorPtr a, TensorPtr b) {
    if (a->device == Device::CUDA) return cuda::mul_cuda(a, b);
    check_same_shape(a, b);
    bool rg = any_requires_grad(a, b);
    auto result = make_tensor(a->shape, rg);
    for (size_t i = 0; i < a->numel(); i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get()]() {
            for (size_t i = 0; i < r->numel(); i++) {
                if (a->requires_grad) a->grad[i] += r->grad[i] * b->data[i];
                if (b->requires_grad) b->grad[i] += r->grad[i] * a->data[i];
            }
        };
    }
    return result;
}

TensorPtr div(TensorPtr a, TensorPtr b) {
    if (a->device == Device::CUDA) return cuda::div_cuda(a, b);
    check_same_shape(a, b);
    bool rg = any_requires_grad(a, b);
    auto result = make_tensor(a->shape, rg);
    for (size_t i = 0; i < a->numel(); i++) {
        if (b->data[i] == 0.0F) {
            throw std::invalid_argument("division by zero");
        }
        result->data[i] = a->data[i] / b->data[i];
    }
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get()]() {
            for (size_t i = 0; i < r->numel(); i++) {
                if (a->requires_grad) a->grad[i] += r->grad[i] / b->data[i];
                if (b->requires_grad)
                    b->grad[i] -= r->grad[i] * a->data[i] / (b->data[i] * b->data[i]);
            }
        };
    }
    return result;
}

TensorPtr matmul(TensorPtr a, TensorPtr b) {
    if (a->device == Device::CUDA) return cuda::matmul_cuda(a, b);
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];
    if (K != b->shape[0]) {
        throw std::invalid_argument("incompatible shapes for matmul");
    }
    bool rg = any_requires_grad(a, b);
    auto result = make_tensor({M, N}, rg);

    /*
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < K; k++) {
                result->at({i, j}) += a->at({i, k}) * b->at({k, j});
            }
        }
    }
    */

    // Forward: result = A @ B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M), static_cast<int>(N),
                static_cast<int>(K), 1.0F, a->data.data(), static_cast<int>(K), b->data.data(),
                static_cast<int>(N), 0.0F, result->data.data(), static_cast<int>(N));

    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get(), M, K, N]() {
            /*
            for (size_t i = 0; i < M; i++) {
                for (size_t k = 0; k < K; k++) {
                    if (a->requires_grad) {
                        float g = 0.0F;
                        for (size_t j = 0; j < N; j++) {
                            g += r->grad[i * N + j] * b->data[k * N + j];
                        }
                        a->grad[i * K + k] += g;
                    }
                    if (b->requires_grad) {
                        for (size_t j = 0; j < N; j++) {
                            b->grad[k * N + j] += r->grad[i * N + j] * a->data[i * K + k];
                        }
                    }
                }
            }
            */
            if (a->requires_grad) {
                // grad_A = grad_C @ B^T
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(M),
                            static_cast<int>(K), static_cast<int>(N), 1.0F, r->grad.data(),
                            static_cast<int>(N), b->data.data(), static_cast<int>(N), 1.0F,
                            a->grad.data(), static_cast<int>(K));
            }
            if (b->requires_grad) {
                // grad_B = A^T @ grad_C
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(K),
                            static_cast<int>(N), static_cast<int>(M), 1.0F, a->data.data(),
                            static_cast<int>(K), r->grad.data(), static_cast<int>(N), 1.0F,
                            b->grad.data(), static_cast<int>(N));
            }
        };
    }
    return result;
}

TensorPtr add_bias(TensorPtr a, TensorPtr b) {
    if (a->device == Device::CUDA) return cuda::add_bias_cuda(a, b);
    if (a->shape.size() != 2 || b->shape.size() != 1) {
        throw std::invalid_argument("add_bias requires a 2D tensor and a 1D bias");
    }
    if (a->shape[1] != b->shape[0]) {
        throw std::invalid_argument("bias size must match a.shape[1]");
    }
    size_t M = a->shape[0];
    size_t N = a->shape[1];
    bool rg = a->requires_grad || b->requires_grad;
    auto result = make_tensor({M, N}, rg);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            result->data[i * N + j] = a->data[i * N + j] + b->data[j];
        }
    }
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get(), M, N]() {
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    if (a->requires_grad) a->grad[i * N + j] += r->grad[i * N + j];
                    if (b->requires_grad) b->grad[j] += r->grad[i * N + j];
                }
            }
        };
    }
    return result;
}

}  // namespace mlf::ops