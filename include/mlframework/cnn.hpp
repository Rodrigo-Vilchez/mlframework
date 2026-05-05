#pragma once

#include <memory>
#include <vector>

#include "mlframework/layer.hpp"
#include "mlframework/module.hpp"

namespace mlf {

struct ConvBlockConfig {
    size_t out_channels;
    size_t kernel_size;
    size_t stride = 1;
    size_t padding = 0;
    bool pool = true;
    size_t pool_k = 2;
    size_t pool_s = 2;
};

struct CNNConfig {
    size_t input_channels = 1;
    size_t input_size = 28;
    std::vector<ConvBlockConfig> conv_layers;
    size_t hidden_dim = 128;
    size_t output_size = 10;
    float dropout_p = 0.3F;
};

inline CNNConfig default_cnn_config() {
    return CNNConfig{
        .input_channels = 1,
        .input_size = 28,
        .conv_layers =
            {
                {.out_channels = 32, .kernel_size = 3, .padding = 1, .pool = true},
                {.out_channels = 64, .kernel_size = 3, .padding = 1, .pool = true},
            },
        .hidden_dim = 128,
        .output_size = 10,
        .dropout_p = 0.3F,
    };
}

class CNN : public Module {
   public:
    explicit CNN(const CNNConfig& cfg = default_cnn_config());
    TensorPtr forward(TensorPtr x) override;
    const CNNConfig& config() const { return cfg_; }

   private:
    CNNConfig cfg_;
    std::vector<std::shared_ptr<Conv2d>> conv_layers_;
    std::vector<std::shared_ptr<MaxPool2d>> pool_layers_;  // nullptr if pool=false
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<BatchNorm1d> bn1_;
    std::shared_ptr<Linear> fc2_;
    float dropout_p_;

    static size_t compute_flat_size(const CNNConfig&);
};

}  // namespace mlf