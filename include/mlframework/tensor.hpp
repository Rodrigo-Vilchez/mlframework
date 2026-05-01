#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace mlf {

struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

TensorPtr make_tensor(std::vector<size_t> shape, bool requires_grad = false);
TensorPtr make_tensor(std::vector<size_t> shape, std::vector<float> data,
                      bool requires_grad = false);

inline bool& no_grad_mode() {
    static bool flag = false;
    return flag;
}

struct NoGrad {
    NoGrad() { no_grad_mode() = true; }
    ~NoGrad() { no_grad_mode() = false; }
};

struct Tensor {
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<float> data;
    std::vector<float> grad;
    bool requires_grad{false};
    std::function<void()> backward_fn;
    std::vector<TensorPtr> inputs;

    float& at(std::vector<size_t> indices);
    const float& at(std::vector<size_t> indices) const;

    size_t numel() const;
    void zero_grad();
    void backward();
    void print() const;

   private:
    friend TensorPtr make_tensor(std::vector<size_t>, bool);
    friend TensorPtr make_tensor(std::vector<size_t>, std::vector<float>, bool);

    explicit Tensor(std::vector<size_t> shape, bool requires_grad = false);
    Tensor(std::vector<size_t> shape, std::vector<float> data, bool requires_grad = false);

    void compute_strides();
    size_t flat_index(const std::vector<size_t>& indices) const;
};

TensorPtr reshape(TensorPtr x, std::vector<size_t> new_shape);

}  // namespace mlf