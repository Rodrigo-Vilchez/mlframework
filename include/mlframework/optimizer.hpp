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

   private:
    float lr_max_, lr_min_;
    size_t T0_;        // initial cycle length in epochs
    float T_mult_;     // cycle length multiplier on each restart
    size_t t_cur_{0};  // current position within cycle
    size_t T_cur_{0};  // current cycle length
};

}  // namespace mlf