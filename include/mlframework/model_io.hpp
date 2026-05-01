#pragma once

#include <string>

#include "mlframework/cnn.hpp"
#include "mlframework/mlp.hpp"

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

// Variant that wraps either MLP or CNN uniformly
struct Model {
    enum class Type { MLP, CNN };
    Type type;

    std::unique_ptr<MLP> mlp;
    std::unique_ptr<CNN> cnn;

    TensorPtr forward(TensorPtr x) { return type == Type::MLP ? mlp->forward(x) : cnn->forward(x); }

    std::vector<TensorPtr> parameters() const {
        return type == Type::MLP ? mlp->parameters() : cnn->parameters();
    }
};

}  // namespace mlf