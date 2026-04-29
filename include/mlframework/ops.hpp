#pragma once

#include "mlframework/tensor.hpp"

namespace mlf::ops {

TensorPtr add(TensorPtr a, TensorPtr b);
TensorPtr sub(TensorPtr a, TensorPtr b);
TensorPtr mul(TensorPtr a, TensorPtr b);
TensorPtr div(TensorPtr a, TensorPtr b);
TensorPtr matmul(TensorPtr a, TensorPtr b);
// Adds bias 'b' of shape {N} to each row of 'a' of shape {M, N}
TensorPtr add_bias(TensorPtr a, TensorPtr b);

}  // namespace mlf::ops