#include "mlframework/loss.hpp"

#include <cmath>
#include <stdexcept>

namespace mlf {

TensorPtr cross_entropy(TensorPtr logits, TensorPtr labels) {
    if (logits->shape.size() != 2) {
        throw std::invalid_argument("cross_entropy: logits must be 2D {batch, classes}");
    }

    // use to_device (autograd node) so gradients flow back to original device
    bool on_cuda = (logits->device == Device::CUDA);
    TensorPtr logits_cpu = on_cuda ? to_device(logits, Device::CPU) : logits;
    TensorPtr labels_cpu = on_cuda ? mlf::to(labels, Device::CPU) : labels;

    if (labels_cpu->shape.size() != 1 || labels_cpu->shape[0] != logits_cpu->shape[0]) {
        throw std::invalid_argument("cross_entropy: labels must be 1D {batch}");
    }

    size_t B = logits_cpu->shape[0];
    size_t C = logits_cpu->shape[1];

    std::vector<float> probs(B * C);
    for (size_t i = 0; i < B; i++) {
        float max_val = logits_cpu->data[i * C];
        for (size_t j = 1; j < C; j++) max_val = std::max(max_val, logits_cpu->data[i * C + j]);

        float sum_exp = 0.0F;
        for (size_t j = 0; j < C; j++) {
            probs[i * C + j] = std::exp(logits_cpu->data[i * C + j] - max_val);
            sum_exp += probs[i * C + j];
        }
        for (size_t j = 0; j < C; j++) probs[i * C + j] /= sum_exp;
    }

    float loss_val = 0.0F;
    for (size_t i = 0; i < B; i++) {
        size_t cls = static_cast<size_t>(labels_cpu->data[i]);
        loss_val -= std::log(probs[i * C + cls] + 1e-9F);
    }
    loss_val /= static_cast<float>(B);

    // result is always CPU scalar — backward starts here
    bool rg = logits_cpu->requires_grad;
    auto result = make_tensor({1}, {loss_val}, rg);

    if (rg) {
        // logits_cpu is in result->inputs — to_device's backward_fn
        // will automatically transfer logits_cpu->grad back to CUDA
        result->inputs = {logits_cpu, labels_cpu};
        result->backward_fn = [logits_cpu, labels_cpu, r = result.get(), probs, B, C]() {
            float grad_out = r->grad[0] / static_cast<float>(B);
            for (size_t i = 0; i < B; i++) {
                size_t cls = static_cast<size_t>(labels_cpu->data[i]);
                for (size_t j = 0; j < C; j++) {
                    float y = (j == cls) ? 1.0F : 0.0F;
                    logits_cpu->grad[i * C + j] += grad_out * (probs[i * C + j] - y);
                }
            }
        };
    }

    return result;
}

}  // namespace mlf