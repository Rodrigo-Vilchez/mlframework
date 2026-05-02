#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace mlf {

enum class Device { CPU, CUDA };

struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

TensorPtr make_tensor(std::vector<size_t> shape, bool requires_grad = false,
                      Device device = Device::CPU);
TensorPtr make_tensor(std::vector<size_t> shape, std::vector<float> data,
                      bool requires_grad = false, Device device = Device::CPU);

TensorPtr to(TensorPtr x, Device device);
TensorPtr to_device(TensorPtr x, Device target);
TensorPtr reshape(TensorPtr x, std::vector<size_t> new_shape);

struct Tensor {
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<float> data;  // valid when device == CPU
    std::vector<float> grad;
    bool requires_grad{false};
    Device device{Device::CPU};

    // CUDA buffer with automatic cudaFree via custom deleter
    std::shared_ptr<float> cuda_data;
    std::shared_ptr<float> cuda_grad;

    std::function<void()> backward_fn;
    std::vector<TensorPtr> inputs;

    float& at(std::vector<size_t> indices);
    const float& at(std::vector<size_t> indices) const;

    size_t numel() const;
    void zero_grad();
    void backward();
    void print() const;

    // raw device pointer helpers
    float* cuda_data_ptr() { return cuda_data.get(); }
    const float* cuda_data_ptr() const { return cuda_data.get(); }
    float* cuda_grad_ptr() { return cuda_grad.get(); }

   private:
    friend TensorPtr make_tensor(std::vector<size_t>, bool, Device);
    friend TensorPtr make_tensor(std::vector<size_t>, std::vector<float>, bool, Device);

    explicit Tensor(std::vector<size_t> shape, bool requires_grad, Device device);
    Tensor(std::vector<size_t> shape, std::vector<float> data, bool requires_grad, Device device);

    void compute_strides();
    size_t flat_index(const std::vector<size_t>& indices) const;
};

// global flags
inline bool& no_grad_mode() {
    static bool f = false;
    return f;
}
struct NoGrad {
    NoGrad() { no_grad_mode() = true; }
    ~NoGrad() { no_grad_mode() = false; }
};
struct TrainMode {
    TrainMode() {}
};
struct EvalMode {
    EvalMode() {}
};
inline bool& training_mode() {
    static bool f = true;
    return f;
}

}  // namespace mlf