#include <cassert>
#include <cmath>
#include <iostream>

#include "mlframework/cnn.hpp"
#include "mlframework/dataloader.hpp"
#include "mlframework/layer.hpp"
#include "mlframework/loss.hpp"
#include "mlframework/mlp.hpp"
#include "mlframework/ops.hpp"
#include "mlframework/ops_cuda.hpp"
#include "mlframework/optimizer.hpp"
#include "mlframework/tensor.hpp"

using namespace mlf;

void test_constructor_zeros() {
    auto t = make_tensor({2, 3});
    assert(t->numel() == 6);
    assert(t->at({0, 0}) == 0.0F);
    std::cout << "[OK] constructor zeros\n";
}

void test_constructor_data() {
    auto t = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    assert(t->at({0, 0}) == 1.0F);
    assert(t->at({1, 2}) == 6.0F);
    std::cout << "[OK] constructor with data\n";
}

void test_strides() {
    auto t = make_tensor({2, 3, 4});
    assert(t->strides[0] == 12);
    assert(t->strides[1] == 4);
    assert(t->strides[2] == 1);
    std::cout << "[OK] strides\n";
}

void test_write() {
    auto t = make_tensor({3, 3});
    t->at({1, 1}) = 42.0F;
    assert(t->at({1, 1}) == 42.0F);
    std::cout << "[OK] write via at()\n";
}

void test_add() {
    auto a = make_tensor({2, 2}, {1, 2, 3, 4});
    auto b = make_tensor({2, 2}, {5, 6, 7, 8});
    auto c = ops::add(a, b);
    assert(c->at({0, 0}) == 6.0F);
    assert(c->at({1, 1}) == 12.0F);
    std::cout << "[OK] add\n";
}

void test_sub() {
    auto a = make_tensor({2, 2}, {5, 6, 7, 8});
    auto b = make_tensor({2, 2}, {1, 2, 3, 4});
    auto c = ops::sub(a, b);
    assert(c->at({0, 0}) == 4.0F);
    assert(c->at({1, 1}) == 4.0F);
    std::cout << "[OK] sub\n";
}

void test_mul() {
    auto a = make_tensor({2, 2}, {1, 2, 3, 4});
    auto b = make_tensor({2, 2}, {2, 2, 2, 2});
    auto c = ops::mul(a, b);
    assert(c->at({0, 0}) == 2.0F);
    assert(c->at({1, 1}) == 8.0F);
    std::cout << "[OK] mul\n";
}

void test_matmul() {
    auto a = make_tensor({2, 2}, {1, 2, 3, 4});
    auto b = make_tensor({2, 2}, {5, 6, 7, 8});
    auto c = ops::matmul(a, b);
    assert(c->at({0, 0}) == 19.0F);
    assert(c->at({0, 1}) == 22.0F);
    assert(c->at({1, 0}) == 43.0F);
    assert(c->at({1, 1}) == 50.0F);
    std::cout << "[OK] matmul\n";
}

void test_grad_mul() {
    auto a = make_tensor({1}, {3.0F}, true);
    auto b = make_tensor({1}, {4.0F}, true);
    auto c = ops::mul(a, b);
    c->backward();
    assert(a->grad[0] == 4.0F);
    assert(b->grad[0] == 3.0F);
    std::cout << "[OK] grad mul\n";
}

void test_grad_add() {
    auto a = make_tensor({1}, {2.0F}, true);
    auto b = make_tensor({1}, {5.0F}, true);
    auto c = ops::add(a, b);
    c->backward();
    assert(a->grad[0] == 1.0F);
    assert(b->grad[0] == 1.0F);
    std::cout << "[OK] grad add\n";
}

void test_grad_chain() {
    // d = (a * b) + b  →  grad_a = b = 3, grad_b = a + 1 = 3
    auto a = make_tensor({1}, {2.0F}, true);
    auto b = make_tensor({1}, {3.0F}, true);
    auto c = ops::mul(a, b);
    auto d = ops::add(c, b);
    d->backward();
    assert(a->grad[0] == 3.0F);
    assert(b->grad[0] == 3.0F);
    std::cout << "[OK] grad chain rule\n";
}

void test_sgd_converges() {
    // f(x) = x^2, grad = 2x, minimum at x = 0
    auto x = make_tensor({1}, {5.0F}, true);
    SGD opt({x}, 0.1F);

    for (int i = 0; i < 50; i++) {
        opt.zero_grad();
        // manually set grad = 2x (forward not needed here)
        x->grad[0] = 2.0F * x->data[0];
        opt.step();
    }

    assert(x->data[0] < 0.01F);
    std::cout << "[OK] SGD converges\n";
}

void test_linear_forward_shape() {
    Linear fc(4, 3);
    auto x = make_tensor({2, 4}, std::vector<float>(8, 1.0F));
    auto y = fc.forward(x);
    assert(y->shape[0] == 2);
    assert(y->shape[1] == 3);
    std::cout << "[OK] linear forward shape\n";
}

void test_relu() {
    auto x = make_tensor({4}, {-2.0F, -1.0F, 0.0F, 3.0F}, true);
    auto y = relu(x);
    assert(y->data[0] == 0.0F);
    assert(y->data[1] == 0.0F);
    assert(y->data[2] == 0.0F);
    assert(y->data[3] == 3.0F);
    y->backward();
    assert(x->grad[0] == 0.0F);
    assert(x->grad[3] == 1.0F);
    std::cout << "[OK] relu + grad\n";
}

void test_sigmoid_range() {
    auto x = make_tensor({3}, {-10.0F, 0.0F, 10.0F});
    auto y = sigmoid(x);
    assert(y->data[0] < 0.01F);
    assert(std::abs(y->data[1] - 0.5F) < 1e-6F);
    assert(y->data[2] > 0.99F);
    std::cout << "[OK] sigmoid range\n";
}

void test_cross_entropy_perfect_prediction() {
    // high logits in the correct class implies loss near 0
    auto logits = make_tensor({2, 3}, {10.0F, 0.0F, 0.0F, 0.0F, 10.0F, 0.0F}, true);
    auto labels = make_tensor({2}, std::vector<float>{0.0F, 1.0F});
    auto loss = cross_entropy(logits, labels);
    assert(loss->data[0] < 0.001F);
    std::cout << "[OK] cross_entropy perfect prediction\n";
}

void test_cross_entropy_uniform_prediction() {
    // same logits implies loss = log(num_classes) = log(4)
    auto logits = make_tensor({1, 4}, {1.0F, 1.0F, 1.0F, 1.0F}, true);
    auto labels = make_tensor({1}, std::vector<float>{2.0F});
    auto loss = cross_entropy(logits, labels);
    float expected = std::log(4.0F);
    assert(std::abs(loss->data[0] - expected) < 1e-5F);
    std::cout << "[OK] cross_entropy uniform prediction\n";
}

void test_cross_entropy_backward() {
    // gradient check: grad of logits should be (p - y) / B
    auto logits = make_tensor({1, 3}, {1.0F, 2.0F, 3.0F}, true);
    auto labels = make_tensor({1}, std::vector<float>{2.0F});
    auto loss = cross_entropy(logits, labels);
    loss->backward();
    // correct class grad should be negative (p - 1), others positive (p - 0)
    assert(logits->grad[2] < 0.0F);
    assert(logits->grad[0] > 0.0F);
    assert(logits->grad[1] > 0.0F);
    std::cout << "[OK] cross_entropy backward\n";
}

void test_mnist_loader() {
    IDXLoader loader("data/mnist", 32, true);
    assert(loader.num_samples() == 60000);
    assert(loader.feature_size() == 784);
    assert(loader.has_next());

    Batch b = loader.next();
    assert(b.images->shape[0] == 32);
    assert(b.images->shape[1] == 784);
    assert(b.labels->shape[0] == 32);

    for (size_t i = 0; i < b.images->numel(); i++) {
        assert(b.images->data[i] >= 0.0F && b.images->data[i] <= 1.0F);
    }
    std::cout << "[OK] IDX loader (MNIST)\n";
}

void test_mlp_forward_shape() {
    MLP model(784, {128, 64}, 10);
    auto x = make_tensor({32, 784}, std::vector<float>(32 * 784, 0.5F));
    auto out = model.forward(x);
    assert(out->shape[0] == 32);
    assert(out->shape[1] == 10);
    std::cout << "[OK] MLP forward shape\n";
}

void test_mlp_backward_runs() {
    MLP model(784, {128, 64}, 10);
    auto x = make_tensor({4, 784}, std::vector<float>(4 * 784, 0.5F));
    auto labels = make_tensor({4}, std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F});
    auto logits = model.forward(x);
    auto loss = cross_entropy(logits, labels);
    loss->backward();
    // all parameter grads should be non-zero
    for (auto& p : model.parameters()) {
        bool any_nonzero = false;
        for (float g : p->grad) {
            if (g != 0.0F) {
                any_nonzero = true;
                break;
            }
        }
        assert(any_nonzero);
    }
    std::cout << "[OK] MLP backward runs\n";
}

void test_conv2d_forward_shape() {
    Conv2d conv(1, 8, 3, 1, 1);  // in=1, out=8, K=3, S=1, P=1
    auto x = make_tensor({2, 1, 8, 8}, std::vector<float>(2 * 64, 0.5F));
    auto y = conv.forward(x);
    assert(y->shape[0] == 2);
    assert(y->shape[1] == 8);
    assert(y->shape[2] == 8);  // padding=1 preserves spatial size
    assert(y->shape[3] == 8);
    std::cout << "[OK] conv2d forward shape with padding\n";
}

void test_conv2d_backward_runs() {
    Conv2d conv(1, 4, 3, 1, 1);
    auto x = make_tensor({1, 1, 4, 4}, std::vector<float>(16, 1.0F), true);
    auto labels = make_tensor({1}, std::vector<float>{0.0F});
    auto y = conv.forward(x);
    auto flat = flatten(y);
    // reduce to scalar via sum for simple backward test
    auto w = make_tensor({flat->shape[1], 1}, std::vector<float>(flat->shape[1], 1.0F), false);
    auto out = ops::matmul(flat, w);
    out->backward();
    bool any_nonzero = false;
    for (float g : x->grad)
        if (g != 0.0F) {
            any_nonzero = true;
            break;
        }
    assert(any_nonzero);
    std::cout << "[OK] conv2d backward runs\n";
}

void test_maxpool_forward_shape() {
    MaxPool2d pool(2);
    auto x = make_tensor({1, 4, 8, 8}, std::vector<float>(1 * 4 * 64, 1.0F));
    auto y = pool.forward(x);
    assert(y->shape[0] == 1);
    assert(y->shape[1] == 4);
    assert(y->shape[2] == 4);
    assert(y->shape[3] == 4);
    std::cout << "[OK] maxpool forward shape\n";
}

void test_maxpool_backward_routes_to_max() {
    MaxPool2d pool(2);
    // 1x1x2x2 input, max is at position [0,0,1,1] = 4.0
    auto x = make_tensor({1, 1, 2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}, true);
    auto y = pool.forward(x);
    y->backward();
    assert(x->grad[0] == 0.0F);
    assert(x->grad[3] == 1.0F);  // only the max gets gradient
    std::cout << "[OK] maxpool backward routes to max\n";
}

void test_cnn_forward_shape() {
    CNN model;
    // MNIST-like input: {2, 1, 28, 28}
    auto x = make_tensor({2, 1, 28, 28}, std::vector<float>(2 * 784, 0.5F));
    auto y = model.forward(x);
    assert(y->shape[0] == 2);
    assert(y->shape[1] == 10);
    std::cout << "[OK] CNN forward shape\n";
}

void test_cnn_backward_runs() {
    CNN model;
    auto x = make_tensor({2, 1, 28, 28}, std::vector<float>(2 * 784, 0.5F), true);
    auto labels = make_tensor({2}, std::vector<float>{0.0F, 1.0F});
    auto logits = model.forward(x);
    auto loss = cross_entropy(logits, labels);
    loss->backward();
    bool any_nonzero = false;
    for (auto& p : model.parameters()) {
        for (float g : p->grad) {
            if (g != 0.0F) {
                any_nonzero = true;
                break;
            }
        }
    }
    assert(any_nonzero);
    std::cout << "[OK] CNN backward runs\n";
}

void test_cuda_roundtrip() {
    auto a = make_tensor({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    auto a_gpu = to(a, Device::CUDA);
    auto a_cpu = to(a_gpu, Device::CPU);
    assert(a_cpu->data[0] == 1.0F);
    assert(a_cpu->data[3] == 4.0F);
    std::cout << "[OK] CUDA CPU->GPU->CPU roundtrip\n";
}

void test_cuda_matmul() {
    auto x = to(make_tensor({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}), Device::CUDA);
    auto y = to(make_tensor({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F}), Device::CUDA);
    auto z = ops::matmul(x, y);
    auto z_cpu = to(z, Device::CPU);
    assert(z_cpu->data[0] == 19.0F);
    assert(z_cpu->data[3] == 50.0F);
    std::cout << "[OK] CUDA matmul\n";
}

void test_cuda_relu() {
    auto x = to(make_tensor({4}, {-2.0F, -1.0F, 0.0F, 3.0F}, true), Device::CUDA);
    auto y = cuda::relu_cuda(x);
    auto y_cpu = to(y, Device::CPU);
    assert(y_cpu->data[0] == 0.0F);
    assert(y_cpu->data[3] == 3.0F);
    std::cout << "[OK] CUDA relu\n";
}

void test_to_device_backward() {
    // x on CPU, move to CUDA, compute, move back, backward
    auto x = make_tensor({4}, {1.0F, 2.0F, 3.0F, 4.0F}, true);
    auto x_gpu = to_device(x, Device::CUDA);     // CPU→CUDA node
    auto y_gpu = cuda::relu_cuda(x_gpu);         // relu on CUDA
    auto y_cpu = to_device(y_gpu, Device::CPU);  // CUDA→CPU node
    y_cpu->backward();                           // should flow back to x

    // relu grad: all inputs > 0, so grad = 1 for all
    assert(x->grad[0] == 1.0F);
    assert(x->grad[3] == 1.0F);
    std::cout << "[OK] to_device backward across CPU/CUDA boundary\n";
}

int main() {
    test_constructor_zeros();
    test_constructor_data();
    test_strides();
    test_write();
    test_add();
    test_sub();
    test_mul();
    test_matmul();
    test_grad_mul();
    test_grad_add();
    test_grad_chain();
    test_sgd_converges();
    test_linear_forward_shape();
    test_relu();
    test_sigmoid_range();
    test_cross_entropy_perfect_prediction();
    test_cross_entropy_uniform_prediction();
    test_cross_entropy_backward();
    test_mnist_loader();
    test_mlp_forward_shape();
    test_mlp_backward_runs();
    test_conv2d_forward_shape();
    test_conv2d_backward_runs();
    test_maxpool_forward_shape();
    test_maxpool_backward_routes_to_max();
    test_cnn_forward_shape();
    test_cnn_backward_runs();
    test_cuda_roundtrip();
    test_cuda_matmul();
    test_cuda_relu();
    test_to_device_backward();
    std::cout << "\nAll tests passed.\n";
    return 0;
}