#include "mlframework/cnn.hpp"

#include "mlframework/layer.hpp"

namespace mlf {

size_t CNN::compute_flat_size(const CNNConfig& cfg) {
    size_t H = cfg.input_size;
    size_t W = cfg.input_size;
    size_t C = cfg.input_channels;
    for (const auto& b1k : cfg.conv_layers) {
        H = (H + 2 * b1k.padding - b1k.kernel_size) / b1k.stride + 1;
        W = (W + 2 * b1k.padding - b1k.kernel_size) / b1k.stride + 1;
        C = b1k.out_channels;
        if (b1k.pool) {
            size_t ps = (b1k.pool_s == 0) ? b1k.pool_k : b1k.pool_s;
            H = (H - b1k.pool_k) / ps + 1;
            W = (W - b1k.pool_k) / ps + 1;
        }
    }
    return C * H * W;
}

CNN::CNN(const CNNConfig& cfg) : cfg_(cfg), dropout_p_(cfg.dropout_p) {
    const size_t flat = compute_flat_size(cfg);

    for (size_t i = 0; i < cfg.conv_layers.size(); i++) {
        const auto& b1k = cfg.conv_layers[i];
        size_t in_ch = (i == 0) ? cfg.input_channels : cfg.conv_layers[i - 1].out_channels;

        auto conv = std::make_shared<Conv2d>(in_ch, b1k.out_channels, b1k.kernel_size, b1k.stride,
                                             b1k.padding);
        conv_layers_.push_back(conv);
        register_module("conv" + std::to_string(i), *conv);

        if (b1k.pool) {
            size_t ps = (b1k.pool_s == 0) ? b1k.pool_k : b1k.pool_s;
            auto pool = std::make_shared<MaxPool2d>(b1k.pool_k, ps);
            pool_layers_.push_back(pool);
            register_module("pool" + std::to_string(i), *pool);
        } else {
            pool_layers_.push_back(nullptr);
        }
    }

    fc1_ = std::make_shared<Linear>(flat, cfg.hidden_dim);
    bn1_ = std::make_shared<BatchNorm1d>(cfg.hidden_dim);
    fc2_ = std::make_shared<Linear>(cfg.hidden_dim, cfg.output_size);

    register_module("fc1", *fc1_);
    register_module("bn1", *bn1_);
    register_module("fc2", *fc2_);
}

TensorPtr CNN::forward(TensorPtr x) {
    TensorPtr out = x;
    for (size_t i = 0; i < conv_layers_.size(); i++) {
        out = conv_layers_[i]->forward(out);
        out = relu(out);
        if (pool_layers_[i]) {
            out = pool_layers_[i]->forward(out);
        }
    }
    out = flatten(out);
    out = fc1_->forward(out);
    out = bn1_->forward(out);
    out = relu(out);
    out = dropout(out, dropout_p_);
    out = fc2_->forward(out);
    return out;
}

}  // namespace mlf