#include "mlframework/cnn.hpp"

namespace mlf {

CNN::CNN(float dropout_p) : dropout_p_(dropout_p) {
    conv1_ = std::make_shared<Conv2d>(1, 32, 3, 1, 1);
    conv2_ = std::make_shared<Conv2d>(32, 64, 3, 1, 1);
    pool_ = std::make_shared<MaxPool2d>(2);
    fc1_ = std::make_shared<Linear>(3136, 128);
    bn1_ = std::make_shared<BatchNorm1d>(128);
    fc2_ = std::make_shared<Linear>(128, 10);

    register_module("conv1", *conv1_);
    register_module("conv2", *conv2_);
    register_module("pool", *pool_);
    register_module("fc1", *fc1_);
    register_module("bn1", *bn1_);
    register_module("fc2", *fc2_);
}

TensorPtr CNN::forward(TensorPtr x) {
    auto out = conv1_->forward(x);
    out = relu(out);
    out = pool_->forward(out);
    out = conv2_->forward(out);
    out = relu(out);
    out = pool_->forward(out);
    out = flatten(out);
    out = fc1_->forward(out);
    out = bn1_->forward(out);
    out = relu(out);
    out = dropout(out, dropout_p_);
    out = fc2_->forward(out);
    return out;
}

}  // namespace mlf