#pragma once

#include <vector>

#include "mlframework/tensor.hpp"

namespace mlf {

class SGD {
   public:
    SGD(std::vector<TensorPtr> params, float lr, float momentum = 0.0F);

    void set_lr(float lr) { lr_ = lr; }
    void step();
    void zero_grad();

   private:
    std::vector<TensorPtr> params_;
    float lr_;
    float momentum_;
    std::vector<std::vector<float>> velocity_;
};

class Adam {
   public:
    Adam(std::vector<TensorPtr> params, float lr = 0.001F, float beta1 = 0.9F, float beta2 = 0.999F,
         float epsilon = 1e-8F);

    void set_lr(float lr) { lr_ = lr; }
    void step();
    void zero_grad();

    // serialization interface
    size_t step_count() const { return t_; }
    const std::vector<std::vector<float>>& cpu_m() const { return m_; }
    const std::vector<std::vector<float>>& cpu_v() const { return v_; }
    const std::vector<std::shared_ptr<float>>& cuda_m() const { return cuda_m_; }
    const std::vector<std::shared_ptr<float>>& cuda_v() const { return cuda_v_; }
    size_t param_numel(size_t i) const { return params_[i]->numel(); }
    size_t num_params() const { return params_.size(); }
    bool on_cuda() const { return !params_.empty() && params_[0]->device == Device::CUDA; }

    void restore_state(size_t t, std::vector<std::vector<float>> m,
                       std::vector<std::vector<float>> v);

   private:
    std::vector<TensorPtr> params_;
    float lr_, beta1_, beta2_, epsilon_;
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
    size_t t_{0};
    std::vector<std::shared_ptr<float>> cuda_m_;
    std::vector<std::shared_ptr<float>> cuda_v_;
};

class CosineAnnealingWR {
   public:
    CosineAnnealingWR(float lr_max, float lr_min, size_t T0, float T_mult = 1.0F);

    float get_lr() const;
    void step();  // call once per epoch

    // serialization
    size_t t_cur() const { return t_cur_; }
    size_t T_cur() const { return T_cur_; }
    float lr_max() const { return lr_max_; }
    float lr_min() const { return lr_min_; }
    size_t T0() const { return T0_; }
    float T_mult() const { return T_mult_; }

    void restore_state(size_t t_cur, size_t T_cur) {
        t_cur_ = t_cur;
        T_cur_ = T_cur;
    }

   private:
    float lr_max_, lr_min_;
    size_t T0_;        // initial cycle length in epochs
    float T_mult_;     // cycle length multiplier on each restart
    size_t t_cur_{0};  // current position within cycle
    size_t T_cur_{0};  // current cycle length
};

}  // namespace mlf