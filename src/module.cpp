#include "mlframework/module.hpp"

namespace mlf {

void Module::register_parameter(const std::string& name, TensorPtr& param) {
    params_.push_back({name, &param});
}

void Module::register_module(const std::string& name, Module& submodule) {
    submodules_.push_back({name, &submodule});
}

void Module::to(Device device) {
    for (auto& e : params_) {
        *e.param = mlf::to(*e.param, device);
    }
    for (auto& e : submodules_) {
        e.mod->to(device);
    }
}

std::vector<TensorPtr> Module::parameters() const {
    std::vector<TensorPtr> result;
    for (auto& e : params_) {
        result.push_back(*e.param);
    }
    for (auto& e : submodules_) {
        for (auto& p : e.mod->parameters()) {
            result.push_back(p);
        }
    }
    return result;
}

void Module::zero_grad() {
    for (auto& e : params_) {
        (*e.param)->zero_grad();
    }
    for (auto& e : submodules_) {
        e.mod->zero_grad();
    }
}

void Module::propagate_mode(bool training) {
    training_ = training;
    for (auto& e : submodules_) {
        e.mod->propagate_mode(training);
    }
}

}  // namespace mlf