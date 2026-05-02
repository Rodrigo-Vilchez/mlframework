#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlframework/tensor.hpp"

namespace mlf {

class Module {
   public:
    virtual ~Module() = default;

    virtual TensorPtr forward(TensorPtr x) = 0;

    // move all parameters and submodules to device
    void to(Device device);

    // collect all parameters recursively
    std::vector<TensorPtr> parameters() const;

    // zero all gradients recursively
    void zero_grad();

    void train() {
        training_ = true;
        propagate_mode(true);
    }
    void eval() {
        training_ = false;
        propagate_mode(false);
    }
    bool is_training() const { return training_; }

   protected:
    void register_parameter(const std::string& name, TensorPtr& param);
    void register_module(const std::string& name, Module& submodule);

   private:
    void propagate_mode(bool training);

    struct ParamEntry {
        std::string name;
        TensorPtr* param;
    };
    struct ModuleEntry {
        std::string name;
        Module* mod;
    };

    std::vector<ParamEntry> params_;
    std::vector<ModuleEntry> submodules_;
    bool training_{true};
};

}  // namespace mlf