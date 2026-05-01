#pragma once

#include <vector>

#include "mlframework/layer.hpp"
#include "mlframework/tensor.hpp"

namespace mlf {

// Conv(1→32,K=3,P=1) → ReLU → MaxPool(2)   28x28 → 14x14
// Conv(32→64,K=3,P=1) → ReLU → MaxPool(2)  14x14 → 7x7
// Flatten                                    64*7*7 = 3136
// Linear(3136→128) → BatchNorm → ReLU → Dropout
// Linear(128→10)
class CNN {
   public:
    CNN(float dropout_p = 0.3F);

    TensorPtr forward(TensorPtr x);  // x: {N, 1, 28, 28}
    std::vector<TensorPtr> parameters() const;

   private:
    Conv2d conv1_{1, 32, 3, 1, 1};
    Conv2d conv2_{32, 64, 3, 1, 1};
    MaxPool2d pool_{2};
    Linear fc1_{3136, 128};
    BatchNorm1d bn1_{128};
    Linear fc2_{128, 10};
    float dropout_p_;
};

}  // namespace mlf