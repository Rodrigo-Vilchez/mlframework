#pragma once

#include <string>

#include "mlframework/mlp.hpp"
#include "mlframework/optimizer.hpp"

namespace mlf {

struct ModelConfig {
    size_t input_size;
    std::vector<size_t> hidden_sizes;
    size_t output_size;
    float dropout_p;
    bool use_batchnorm;
};

void save_model(const MLP& model, const ModelConfig& config, const std::string& path);
ModelConfig load_model(MLP& model, const std::string& path);

void save_optimizer(const MLP& model, const Adam& opt, const CosineAnnealingWR& scheduler,
                    const std::string& path);

void load_optimizer(Adam& opt, CosineAnnealingWR& scheduler, const std::string& path);

}  // namespace mlf