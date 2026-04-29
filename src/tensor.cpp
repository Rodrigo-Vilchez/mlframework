#include "mlframework/tensor.hpp"

#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace mlf {

Tensor::Tensor(std::vector<size_t> shape) : shape(std::move(shape)) {
    compute_strides();
    data.resize(numel(), 0.0F);
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data)
    : shape(std::move(shape)), data(std::move(data)) {
    if (this->data.size() != numel()) {
        throw std::invalid_argument("data size does not match shape");
    }
    compute_strides();
}

void Tensor::compute_strides() {
    strides.resize(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

size_t Tensor::flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("indices rank does not match tensor rank");
    }
    size_t pos = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("index out of bounds");
        }
        pos += indices[i] * strides[i];
    }
    return pos;
}

float& Tensor::at(std::vector<size_t> indices) { return data[flat_index(indices)]; }

const float& Tensor::at(std::vector<size_t> indices) const { return data[flat_index(indices)]; }

size_t Tensor::numel() const {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
}

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << data[i];
        if (i + 1 < data.size()) std::cout << ", ";
    }
    std::cout << "])\n";
}

}  // namespace mlf