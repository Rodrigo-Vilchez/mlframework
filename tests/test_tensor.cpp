#include <cassert>
#include <iostream>

#include "mlframework/ops.hpp"
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
    std::cout << "\nAll tests passed.\n";
    return 0;
}