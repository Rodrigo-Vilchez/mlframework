#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "mlframework/tensor.hpp"

namespace mlf {

// custom deleter for cuda_data shared_ptr
static void cuda_deleter(float* ptr) {
    if (ptr) cudaFree(ptr);
}

static std::shared_ptr<float> alloc_cuda(size_t n) {
    float* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, n * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
    return std::shared_ptr<float>(ptr, cuda_deleter);
}

static void cuda_zero(float* ptr, size_t n) { cudaMemset(ptr, 0, n * sizeof(float)); }

TensorPtr to(TensorPtr x, Device device) {
    if (x->device == device) return x;

    auto result = make_tensor(x->shape, x->requires_grad, device);

    if (device == Device::CUDA) {
        // CPU → GPU
        result->cuda_data = alloc_cuda(x->numel());
        cudaMemcpy(result->cuda_data_ptr(), x->data.data(), x->numel() * sizeof(float),
                   cudaMemcpyHostToDevice);
        if (x->requires_grad) {
            result->cuda_grad = alloc_cuda(x->numel());
            cuda_zero(result->cuda_grad_ptr(), x->numel());
        }
        result->data.clear();  // free CPU buffer
        result->grad.clear();
    } else {
        // GPU → CPU
        result->data.resize(x->numel());
        cudaMemcpy(result->data.data(), x->cuda_data_ptr(), x->numel() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        if (x->requires_grad) {
            result->grad.resize(x->numel());
            cudaMemcpy(result->grad.data(), x->cuda_grad_ptr(), x->numel() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
    }
    return result;
}

// factory helpers used by make_tensor for CUDA device
void tensor_alloc_cuda(Tensor& t) {
    t.cuda_data = alloc_cuda(t.numel());
    cuda_zero(t.cuda_data.get(), t.numel());
    if (t.requires_grad) {
        t.cuda_grad = alloc_cuda(t.numel());
        cuda_zero(t.cuda_grad.get(), t.numel());
    }
}

}  // namespace mlf