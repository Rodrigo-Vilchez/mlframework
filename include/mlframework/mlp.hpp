#pragma once

#include <memory>
#include <vector>

#include "mlframework/layer.hpp"
#include "mlframework/module.hpp"

namespace mlf {

class MLP : public Module {
   public:
    MLP(size_t input_size, std::vector<size_t> hidden_sizes, size_t output_size,
        float dropout_p = 0.0F, bool use_batchnorm = false);

    TensorPtr forward(TensorPtr x) override;

   private:
    std::vector<std::shared_ptr<Linear>> layers_;
    std::vector<std::shared_ptr<BatchNorm1d>> bnorms_;
    float dropout_p_;
    bool use_batchnorm_;
};

}  // namespace mlf