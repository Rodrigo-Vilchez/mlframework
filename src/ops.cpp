#include "mlframework/ops.hpp"

#include <stdexcept>

namespace mlf::ops {

static void check_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("tensors must have the same shape");
    }
}

Tensor add(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b);
    Tensor result(a.shape);
    for (size_t i = 0; i < a.numel(); i++) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b);
    Tensor result(a.shape);
    for (size_t i = 0; i < a.numel(); i++) {
        result.data[i] = a.data[i] - b.data[i];
    }
    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b);
    Tensor result(a.shape);
    for (size_t i = 0; i < a.numel(); i++) {
        result.data[i] = a.data[i] * b.data[i];
    }
    return result;
}

Tensor div(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b);
    Tensor result(a.shape);
    for (size_t i = 0; i < a.numel(); i++) {
        if (b.data[i] == 0.0F) {
            throw std::invalid_argument("division by zero");
        }
        result.data[i] = a.data[i] / b.data[i];
    }
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];
    if (K != b.shape[0]) {
        throw std::invalid_argument("incompatible shapes for matmul");
    }
    Tensor result({M, N});
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < K; k++) {
                result.at({i, j}) += a.at({i, k}) * b.at({k, j});
            }
        }
    }
    return result;
}

}  // namespace mlf::ops