#include "mlframework/ops.hpp"

#include <stdexcept>

namespace mlf::ops {

static void check_same_shape(const TensorPtr& a, const TensorPtr& b) {
    if (a->shape != b->shape) {
        throw std::invalid_argument("tensors must have the same shape");
    }
}

static bool any_requires_grad(const TensorPtr& a, const TensorPtr& b) {
    return a->requires_grad || b->requires_grad;
}

TensorPtr add(TensorPtr a, TensorPtr b) {
    check_same_shape(a, b);
    bool rg = any_requires_grad(a, b);
    auto result = make_tensor(a->shape, rg);
    for (size_t i = 0; i < a->numel(); i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, result]() {
            for (size_t i = 0; i < result->numel(); i++) {
                if (a->requires_grad) a->grad[i] += result->grad[i];
                if (b->requires_grad) b->grad[i] += result->grad[i];
            }
        };
    }
    return result;
}

TensorPtr sub(TensorPtr a, TensorPtr b) {
    check_same_shape(a, b);
    bool rg = any_requires_grad(a, b);
    auto result = make_tensor(a->shape, rg);
    for (size_t i = 0; i < a->numel(); i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, result]() {
            for (size_t i = 0; i < result->numel(); i++) {
                if (a->requires_grad) a->grad[i] += result->grad[i];
                if (b->requires_grad) b->grad[i] -= result->grad[i];
            }
        };
    }
    return result;
}

TensorPtr mul(TensorPtr a, TensorPtr b) {
    check_same_shape(a, b);
    bool rg = any_requires_grad(a, b);
    auto result = make_tensor(a->shape, rg);
    for (size_t i = 0; i < a->numel(); i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, result]() {
            for (size_t i = 0; i < result->numel(); i++) {
                if (a->requires_grad) a->grad[i] += result->grad[i] * b->data[i];
                if (b->requires_grad) b->grad[i] += result->grad[i] * a->data[i];
            }
        };
    }
    return result;
}

TensorPtr div(TensorPtr a, TensorPtr b) {
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
        result->backward_fn = [a, b, result]() {
            for (size_t i = 0; i < result->numel(); i++) {
                if (a->requires_grad) a->grad[i] += result->grad[i] / b->data[i];
                if (b->requires_grad)
                    b->grad[i] -= result->grad[i] * a->data[i] / (b->data[i] * b->data[i]);
            }
        };
    }
    return result;
}

TensorPtr matmul(TensorPtr a, TensorPtr b) {
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
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < K; k++) {
                result->at({i, j}) += a->at({i, k}) * b->at({k, j});
            }
        }
    }
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, result, M, K, N]() {
            for (size_t i = 0; i < M; i++) {
                for (size_t k = 0; k < K; k++) {
                    if (a->requires_grad) {
                        float g = 0.0F;
                        for (size_t j = 0; j < N; j++) {
                            g += result->grad[i * N + j] * b->data[k * N + j];
                        }
                        a->grad[i * K + k] += g;
                    }
                    if (b->requires_grad) {
                        for (size_t j = 0; j < N; j++) {
                            b->grad[k * N + j] += result->grad[i * N + j] * a->data[i * K + k];
                        }
                    }
                }
            }
        };
    }
    return result;
}

}  // namespace mlf::ops