#include "mlframework/tensor.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "mlframework/cuda_internal.hpp"

namespace mlf {

TensorPtr make_tensor(std::vector<size_t> shape, bool requires_grad, Device device) {
    return std::shared_ptr<Tensor>(new Tensor(std::move(shape), requires_grad, device));
}

TensorPtr make_tensor(std::vector<size_t> shape, std::vector<float> data, bool requires_grad,
                      Device device) {
    return std::shared_ptr<Tensor>(
        new Tensor(std::move(shape), std::move(data), requires_grad, device));
}

Tensor::Tensor(std::vector<size_t> shape, bool requires_grad, Device dev)
    : shape(std::move(shape)), requires_grad(requires_grad), device(dev) {
    compute_strides();
    if (device == Device::CPU) {
        data.resize(numel(), 0.0F);
        if (requires_grad) grad.resize(numel(), 0.0F);
    } else {
        tensor_alloc_cuda(*this);
    }
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data, bool requires_grad, Device dev)
    : shape(std::move(shape)), data(std::move(data)), requires_grad(requires_grad), device(dev) {
    if (device == Device::CPU && this->data.size() != numel()) {
        throw std::invalid_argument("data size does not match shape");
    }
    compute_strides();
    if (device == Device::CUDA) {
        // upload data to GPU then clear CPU buffer
        tensor_alloc_cuda(*this);
        cudaMemcpy(cuda_data_ptr(), this->data.data(), numel() * sizeof(float),
                   cudaMemcpyHostToDevice);
        this->data.clear();
        this->grad.clear();
    } else {
        if (requires_grad) grad.resize(numel(), 0.0F);
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

TensorPtr reshape(TensorPtr x, std::vector<size_t> new_shape) {
    size_t new_numel = 1;
    for (size_t d : new_shape) new_numel *= d;
    if (new_numel != x->numel()) {
        throw std::invalid_argument("reshape: element count mismatch");
    }
    auto result = make_tensor(new_shape, x->requires_grad);
    result->data = x->data;

    if (x->requires_grad) {
        result->inputs = {x};
        result->backward_fn = [x, r = result.get()]() {
            for (size_t i = 0; i < x->numel(); i++) {
                x->grad[i] += r->grad[i];
            }
        };
    }
    return result;
}

}  // namespace mlf