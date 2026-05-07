#pragma once
#include "mlframework/tensor.hpp"

namespace mlf::cuda {
TensorPtr add_cuda(TensorPtr a, TensorPtr b);
TensorPtr sub_cuda(TensorPtr a, TensorPtr b);
TensorPtr mul_cuda(TensorPtr a, TensorPtr b);
TensorPtr matmul_cuda(TensorPtr a, TensorPtr b);
TensorPtr div_cuda(TensorPtr a, TensorPtr b);
TensorPtr add_bias_cuda(TensorPtr a, TensorPtr b);
TensorPtr relu_cuda(TensorPtr x);
void adam_step_cuda(float* data, float* grad, float* m, float* v, float lr, float beta1,
                    float beta2, float epsilon, float bc1, float bc2, size_t n);
}  // namespace mlf::cuda