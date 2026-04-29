#include "mlframework/tensor.hpp"

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace mlf {

TensorPtr make_tensor(std::vector<size_t> shape, bool requires_grad) {
    return std::shared_ptr<Tensor>(new Tensor(std::move(shape), requires_grad));
}

TensorPtr make_tensor(std::vector<size_t> shape, std::vector<float> data, bool requires_grad) {
    return std::shared_ptr<Tensor>(new Tensor(std::move(shape), std::move(data), requires_grad));
}

Tensor::Tensor(std::vector<size_t> shape, bool requires_grad)
    : shape(std::move(shape)), requires_grad(requires_grad) {
    compute_strides();
    data.resize(numel(), 0.0F);
    if (requires_grad) {
        grad.resize(numel(), 0.0F);
    }
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data, bool requires_grad)
    : shape(std::move(shape)), data(std::move(data)), requires_grad(requires_grad) {
    if (this->data.size() != numel()) {
        throw std::invalid_argument("data size does not match shape");
    }
    compute_strides();
    if (requires_grad) {
        grad.resize(numel(), 0.0F);
    }
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

void Tensor::zero_grad() { std::fill(grad.begin(), grad.end(), 0.0F); }

void Tensor::backward() {
    if (grad.empty()) {
        grad.assign(numel(), 0.0F);
    }
    std::fill(grad.begin(), grad.end(), 1.0F);

    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;

    std::function<void(Tensor*)> build_topo = [&](Tensor* node) {
        if (visited.count(node)) return;
        visited.insert(node);
        for (auto& inp : node->inputs) {
            build_topo(inp.get());
        }
        topo.push_back(node);
    };

    build_topo(this);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }
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