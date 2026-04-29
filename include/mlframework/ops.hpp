#pragma once

#include "mlframework/tensor.hpp"

namespace mlf::ops {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace mlf::ops