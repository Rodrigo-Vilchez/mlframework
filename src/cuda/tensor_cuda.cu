#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "mlframework/tensor.hpp"

namespace mlf {

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

static void cuda_deleter(float* ptr) {
    if (ptr) {
        // context may already be destroyed at program exit - ignore error
        cudaError_t err = cudaFree(ptr);
        (void)err;
    }
}

static std::shared_ptr<float> alloc_cuda(size_t n) {
    float* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float)));
    return std::shared_ptr<float>(ptr, cuda_deleter);
}

static void cuda_zero(float* ptr, size_t n) { CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float))); }

TensorPtr to(TensorPtr x, Device device) {
    if (x->device == device) return x;
    auto result = make_tensor(x->shape, x->requires_grad, device);

    if (device == Device::CUDA) {
        result->cuda_data = alloc_cuda(x->numel());
        CUDA_CHECK(cudaMemcpy(result->cuda_data_ptr(), x->data.data(), x->numel() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (x->requires_grad) {
            result->cuda_grad = alloc_cuda(x->numel());
            cuda_zero(result->cuda_grad_ptr(), x->numel());
        }
        result->data.clear();
        result->grad.clear();
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
        result->data.resize(x->numel());
        CUDA_CHECK(cudaMemcpy(result->data.data(), x->cuda_data_ptr(), x->numel() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        if (x->requires_grad) {
            result->grad.resize(x->numel());
            CUDA_CHECK(cudaMemcpy(result->grad.data(), x->cuda_grad_ptr(),
                                  x->numel() * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    return result;
}

void tensor_alloc_cuda(Tensor& t) {
    t.cuda_data = alloc_cuda(t.numel());
    cuda_zero(t.cuda_data.get(), t.numel());
    if (t.requires_grad) {
        t.cuda_grad = alloc_cuda(t.numel());
        cuda_zero(t.cuda_grad.get(), t.numel());
    }
}

TensorPtr to_device(TensorPtr x, Device target) {
    if (x->device == target) return x;

    // physical transfer
    auto result = make_tensor(x->shape, x->requires_grad, target);

    if (target == Device::CUDA) {
        result->cuda_data = alloc_cuda(x->numel());
        CUDA_CHECK(cudaMemcpy(result->cuda_data_ptr(), x->data.data(), x->numel() * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (x->requires_grad) {
            result->cuda_grad = alloc_cuda(x->numel());
            cuda_zero(result->cuda_grad_ptr(), x->numel());
        }
        result->data.clear();
        result->grad.clear();
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
        result->data.resize(x->numel());
        CUDA_CHECK(cudaMemcpy(result->data.data(), x->cuda_data_ptr(), x->numel() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        if (x->requires_grad) {
            result->grad.assign(x->numel(), 0.0F);
        }
    }

    if (x->requires_grad) {
        result->inputs = {x};
        Device src = x->device;  // capture source device
        result->backward_fn = [x, r = result.get(), src]() {
            if (src == Device::CPU) {
                // x was CPU, result is CUDA
                // r->grad is on CUDA — transfer to CPU and accumulate into x->grad
                std::vector<float> tmp(x->numel());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(tmp.data(), r->cuda_grad_ptr(), x->numel() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < x->numel(); i++) {
                    x->grad[i] += tmp[i];
                }
            } else {
                // x was CUDA, result is CPU
                // r->grad is on CPU — transfer to CUDA and accumulate into x->cuda_grad
                std::vector<float> tmp(x->numel());
                CUDA_CHECK(cudaMemcpy(tmp.data(), x->cuda_grad_ptr(), x->numel() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < x->numel(); i++) {
                    tmp[i] += r->grad[i];
                }
                CUDA_CHECK(cudaMemcpy(x->cuda_grad_ptr(), tmp.data(), x->numel() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
        };
    }
    return result;
}

}  // namespace mlf