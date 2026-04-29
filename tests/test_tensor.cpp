#include <cassert>
#include <iostream>

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

int main() {
    test_constructor_zeros();
    test_constructor_data();
    test_strides();
    test_write();
    std::cout << "\nAll tests passed.\n";
    return 0;
}