#pragma once

#include "mlframework/tensor.hpp"

namespace mlf::ops {

TensorPtr add(TensorPtr a, TensorPtr b);
TensorPtr sub(TensorPtr a, TensorPtr b);
TensorPtr mul(TensorPtr a, TensorPtr b);
TensorPtr div(TensorPtr a, TensorPtr b);
TensorPtr matmul(TensorPtr a, TensorPtr b);

}  // namespace mlf::ops