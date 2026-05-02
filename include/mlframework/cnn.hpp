#pragma once

#include <memory>

#include "mlframework/layer.hpp"
#include "mlframework/module.hpp"

namespace mlf {

class CNN : public Module {
   public:
    explicit CNN(float dropout_p = 0.3F);
    TensorPtr forward(TensorPtr x) override;

   private:
    std::shared_ptr<Conv2d> conv1_;
    std::shared_ptr<Conv2d> conv2_;
    std::shared_ptr<MaxPool2d> pool_;
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<BatchNorm1d> bn1_;
    std::shared_ptr<Linear> fc2_;
    float dropout_p_;
};

}  // namespace mlf