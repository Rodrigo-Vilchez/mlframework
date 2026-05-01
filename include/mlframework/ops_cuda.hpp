#pragma once
#include "mlframework/tensor.hpp"

namespace mlf::cuda {
TensorPtr add_cuda(TensorPtr a, TensorPtr b);
TensorPtr sub_cuda(TensorPtr a, TensorPtr b);
TensorPtr mul_cuda(TensorPtr a, TensorPtr b);
TensorPtr matmul_cuda(TensorPtr a, TensorPtr b);
TensorPtr add_bias_cuda(TensorPtr a, TensorPtr b);
TensorPtr relu_cuda(TensorPtr x);
}  // namespace mlf::cuda