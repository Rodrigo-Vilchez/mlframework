#include <cassert>
#include <iostream>

#include "mlframework/ops.hpp"
#include "mlframework/tensor.hpp"

using namespace mlf;

void test_constructor_zeros() {
    Tensor t({2, 3});
    assert(t.numel() == 6);
    assert(t.data.size() == 6);
    assert(t.at({0, 0}) == 0.0F);
    std::cout << "[OK] constructor zeros\n";
}

void test_constructor_data() {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    assert(t.at({0, 0}) == 1.0F);
    assert(t.at({0, 2}) == 3.0F);
    assert(t.at({1, 0}) == 4.0F);
    assert(t.at({1, 2}) == 6.0F);
    std::cout << "[OK] constructor with data\n";
}

void test_strides() {
    Tensor t({2, 3, 4});
    assert(t.strides[0] == 12);
    assert(t.strides[1] == 4);
    assert(t.strides[2] == 1);
    std::cout << "[OK] strides\n";
}

void test_write() {
    Tensor t({3, 3});
    t.at({1, 1}) = 42.0F;
    assert(t.at({1, 1}) == 42.0F);
    std::cout << "[OK] write via at()\n";
}

void test_add() {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});
    Tensor c = ops::add(a, b);
    assert(c.at({0, 0}) == 6.0F);
    assert(c.at({1, 1}) == 12.0F);
    std::cout << "[OK] add\n";
}

void test_sub() {
    Tensor a({2, 2}, {5, 6, 7, 8});
    Tensor b({2, 2}, {1, 2, 3, 4});
    Tensor c = ops::sub(a, b);
    assert(c.at({0, 0}) == 4.0F);
    assert(c.at({1, 1}) == 4.0F);
    std::cout << "[OK] sub\n";
}

void test_mul() {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {2, 2, 2, 2});
    Tensor c = ops::mul(a, b);
    assert(c.at({0, 0}) == 2.0F);
    assert(c.at({1, 1}) == 8.0F);
    std::cout << "[OK] mul\n";
}

void test_matmul() {
    // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
    // [3 4] x [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});
    Tensor c = ops::matmul(a, b);
    assert(c.at({0, 0}) == 19.0F);
    assert(c.at({0, 1}) == 22.0F);
    assert(c.at({1, 0}) == 43.0F);
    assert(c.at({1, 1}) == 50.0F);
    std::cout << "[OK] matmul\n";
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
    std::cout << "\nAll tests passed.\n";
    return 0;
}