#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mlf {

struct Tensor {
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<float> data;

    explicit Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, std::vector<float> data);

    // Access
    float& at(std::vector<size_t> indices);
    const float& at(std::vector<size_t> indices) const;

    // Methods
    size_t numel() const;
    void print() const;

   private:
    void compute_strides();
    size_t flat_index(const std::vector<size_t>& indices) const;
};

}  // namespace mlf