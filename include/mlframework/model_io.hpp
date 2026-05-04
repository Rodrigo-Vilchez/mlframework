#pragma once

#include <string>
#include <variant>

#include "mlframework/cnn.hpp"
#include "mlframework/mlp.hpp"
#include "mlframework/module.hpp"
#include "mlframework/optimizer.hpp"

namespace mlf {

enum class ModelType : uint32_t { MLP = 1, CNN = 2 };

struct ModelConfig {
    size_t input_size;
    std::vector<size_t> hidden_sizes;
    size_t output_size;
    float dropout_p;
    bool use_batchnorm;
};

struct CNNConfig {
    float dropout_p;
};

using ConfigVariant = std::variant<ModelConfig, CNNConfig>;

// peek at file header without loading weights
ModelType peek_model_type(const std::string& path);

void save_model(const Module& model, ModelType type, const ConfigVariant& config,
                const std::string& path);

ConfigVariant load_model(Module& model, ModelType type, const std::string& path);

void save_optimizer(const Module& model, const Adam& opt, const CosineAnnealingWR& scheduler,
                    const std::string& path);

void load_optimizer(Adam& opt, CosineAnnealingWR& scheduler, const std::string& path);

}  // namespace mlf