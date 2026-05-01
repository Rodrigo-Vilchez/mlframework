#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "mlframework/ops.hpp"

namespace mlf::cuda {

// cuBLAS handle — initialized once
static cublasHandle_t& cublas_handle() {
    static cublasHandle_t handle = []() {
        cublasHandle_t h;
        cublasCreate(&h);
        return h;
    }();
    return handle;
}

// element-wise kernels
__global__ void kernel_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void kernel_sub(const float* a, const float* b, float* c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

__global__ void kernel_mul(const float* a, const float* b, float* c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void kernel_relu(const float* x, float* y, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0F ? x[i] : 0.0F;
}

__global__ void kernel_relu_backward(const float* x, const float* grad_out, float* grad_in,
                                     size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] += grad_out[i] * (x[i] > 0.0F ? 1.0F : 0.0F);
}

__global__ void kernel_add_bias(const float* a, const float* b, float* c, size_t M, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) c[i] = a[i] + b[i % N];
}

__global__ void kernel_add_bias_backward_b(const float* grad, float* grad_b, size_t M, size_t N) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        float s = 0.0F;
        for (size_t i = 0; i < M; i++) s += grad[i * N + j];
        grad_b[j] += s;
    }
}

__global__ void kernel_mul_backward(const float* grad, const float* other, float* grad_in,
                                    size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] += grad[i] * other[i];
}

static dim3 blocks(size_t n, size_t threads = 256) {
    return dim3(static_cast<unsigned>((n + threads - 1) / threads));
}

// --- public CUDA ops ---

TensorPtr add_cuda(TensorPtr a, TensorPtr b) {
    bool rg = a->requires_grad || b->requires_grad;
    auto result = make_tensor(a->shape, rg, Device::CUDA);
    kernel_add<<<blocks(a->numel()), 256>>>(a->cuda_data_ptr(), b->cuda_data_ptr(),
                                            result->cuda_data_ptr(), a->numel());
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get()]() {
            size_t n = r->numel();
            if (a->requires_grad)
                kernel_add<<<blocks(n), 256>>>(a->cuda_grad_ptr(), r->cuda_grad_ptr(),
                                               a->cuda_grad_ptr(), n);
            if (b->requires_grad)
                kernel_add<<<blocks(n), 256>>>(b->cuda_grad_ptr(), r->cuda_grad_ptr(),
                                               b->cuda_grad_ptr(), n);
        };
    }
    return result;
}

TensorPtr sub_cuda(TensorPtr a, TensorPtr b) {
    bool rg = a->requires_grad || b->requires_grad;
    auto result = make_tensor(a->shape, rg, Device::CUDA);
    kernel_sub<<<blocks(a->numel()), 256>>>(a->cuda_data_ptr(), b->cuda_data_ptr(),
                                            result->cuda_data_ptr(), a->numel());
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get()]() {
            size_t n = r->numel();
            if (a->requires_grad)
                kernel_add<<<blocks(n), 256>>>(a->cuda_grad_ptr(), r->cuda_grad_ptr(),
                                               a->cuda_grad_ptr(), n);
            if (b->requires_grad)
                kernel_sub<<<blocks(n), 256>>>(b->cuda_grad_ptr(), r->cuda_grad_ptr(),
                                               b->cuda_grad_ptr(), n);
        };
    }
    return result;
}

TensorPtr mul_cuda(TensorPtr a, TensorPtr b) {
    bool rg = a->requires_grad || b->requires_grad;
    auto result = make_tensor(a->shape, rg, Device::CUDA);
    kernel_mul<<<blocks(a->numel()), 256>>>(a->cuda_data_ptr(), b->cuda_data_ptr(),
                                            result->cuda_data_ptr(), a->numel());
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get()]() {
            size_t n = r->numel();
            if (a->requires_grad)
                kernel_mul_backward<<<blocks(n), 256>>>(r->cuda_grad_ptr(), b->cuda_data_ptr(),
                                                        a->cuda_grad_ptr(), n);
            if (b->requires_grad)
                kernel_mul_backward<<<blocks(n), 256>>>(r->cuda_grad_ptr(), a->cuda_data_ptr(),
                                                        b->cuda_grad_ptr(), n);
        };
    }
    return result;
}

TensorPtr matmul_cuda(TensorPtr a, TensorPtr b) {
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];
    bool rg = a->requires_grad || b->requires_grad;
    auto result = make_tensor({M, N}, rg, Device::CUDA);

    float alpha = 1.0F, beta = 0.0F;
    // row-major A@B = col-major B@A (swap arguments)
    cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M),
                static_cast<int>(K), &alpha, b->cuda_data_ptr(), static_cast<int>(N),
                a->cuda_data_ptr(), static_cast<int>(K), &beta, result->cuda_data_ptr(),
                static_cast<int>(N));

    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get(), M, K, N]() {
            float alpha = 1.0F, beta = 1.0F;
            if (a->requires_grad) {
                // grad_A = grad_C @ B^T  →  col-major: B @ grad_C^T
                cublasSgemm(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(K),
                            static_cast<int>(M), static_cast<int>(N), &alpha, b->cuda_data_ptr(),
                            static_cast<int>(N), r->cuda_grad_ptr(), static_cast<int>(N), &beta,
                            a->cuda_grad_ptr(), static_cast<int>(K));
            }
            if (b->requires_grad) {
                // grad_B = A^T @ grad_C  →  col-major: grad_C @ A^T
                cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(N),
                            static_cast<int>(K), static_cast<int>(M), &alpha, r->cuda_grad_ptr(),
                            static_cast<int>(N), a->cuda_data_ptr(), static_cast<int>(K), &beta,
                            b->cuda_grad_ptr(), static_cast<int>(N));
            }
        };
    }
    return result;
}

TensorPtr add_bias_cuda(TensorPtr a, TensorPtr b) {
    size_t M = a->shape[0];
    size_t N = a->shape[1];
    bool rg = a->requires_grad || b->requires_grad;
    auto result = make_tensor({M, N}, rg, Device::CUDA);
    kernel_add_bias<<<blocks(M * N), 256>>>(a->cuda_data_ptr(), b->cuda_data_ptr(),
                                            result->cuda_data_ptr(), M, N);
    if (rg) {
        result->inputs = {a, b};
        result->backward_fn = [a, b, r = result.get(), M, N]() {
            if (a->requires_grad)
                kernel_add<<<blocks(M * N), 256>>>(a->cuda_grad_ptr(), r->cuda_grad_ptr(),
                                                   a->cuda_grad_ptr(), M * N);
            if (b->requires_grad)
                kernel_add_bias_backward_b<<<blocks(N), 256>>>(r->cuda_grad_ptr(),
                                                               b->cuda_grad_ptr(), M, N);
        };
    }
    return result;
}

TensorPtr relu_cuda(TensorPtr x) {
    auto result = make_tensor(x->shape, x->requires_grad, Device::CUDA);
    kernel_relu<<<blocks(x->numel()), 256>>>(x->cuda_data_ptr(), result->cuda_data_ptr(),
                                             x->numel());
    if (x->requires_grad) {
        result->inputs = {x};
        result->backward_fn = [x, r = result.get()]() {
            if (x->requires_grad)
                kernel_relu_backward<<<blocks(x->numel()), 256>>>(
                    x->cuda_data_ptr(), r->cuda_grad_ptr(), x->cuda_grad_ptr(), x->numel());
        };
    }
    return result;
}

}  // namespace mlf::cuda