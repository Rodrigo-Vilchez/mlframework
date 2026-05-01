#include "mlframework/cnn.hpp"

namespace mlf {

CNN::CNN(float dropout_p) : dropout_p_(dropout_p) {}

TensorPtr CNN::forward(TensorPtr x) {
    // block 1
    auto out = conv1_.forward(x);
    out = relu(out);
    out = pool_.forward(out);

    // block 2
    out = conv2_.forward(out);
    out = relu(out);
    out = pool_.forward(out);

    // classifier
    out = flatten(out);
    out = fc1_.forward(out);
    out = bn1_.forward(out);
    out = relu(out);
    out = dropout(out, dropout_p_);
    out = fc2_.forward(out);

    return out;
}

std::vector<TensorPtr> CNN::parameters() const {
    std::vector<TensorPtr> params;
    for (auto& p : conv1_.parameters()) params.push_back(p);
    for (auto& p : conv2_.parameters()) params.push_back(p);
    for (auto& p : fc1_.parameters()) params.push_back(p);
    for (auto& p : bn1_.parameters()) params.push_back(p);
    for (auto& p : fc2_.parameters()) params.push_back(p);
    return params;
}

}  // namespace mlf