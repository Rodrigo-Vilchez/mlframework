#include "mlframework/loss.hpp"

#include <cmath>
#include <stdexcept>

namespace mlf {

TensorPtr cross_entropy(TensorPtr logits, TensorPtr labels) {
    if (logits->shape.size() != 2) {
        throw std::invalid_argument("cross_entropy: logits must be 2D {batch, classes}");
    }
    if (labels->shape.size() != 1 || labels->shape[0] != logits->shape[0]) {
        throw std::invalid_argument("cross_entropy: labels must be 1D {batch}");
    }

    size_t B = logits->shape[0];
    size_t C = logits->shape[1];

    // Softmax + log + NLL — fused for numerical stability
    std::vector<float> probs(B * C);

    for (size_t i = 0; i < B; i++) {
        // 1. max for stability
        float max_val = logits->data[i * C];
        for (size_t j = 1; j < C; j++) {
            max_val = std::max(max_val, logits->data[i * C + j]);
        }
        // 2. exp(x - max)
        float sum_exp = 0.0F;
        for (size_t j = 0; j < C; j++) {
            probs[i * C + j] = std::exp(logits->data[i * C + j] - max_val);
            sum_exp += probs[i * C + j];
        }
        // 3. normalize
        for (size_t j = 0; j < C; j++) {
            probs[i * C + j] /= sum_exp;
        }
    }

    // NLL loss: -log(p[correct_class]), averaged over batch
    float loss_val = 0.0F;
    for (size_t i = 0; i < B; i++) {
        size_t cls = static_cast<size_t>(labels->data[i]);
        loss_val -= std::log(probs[i * C + cls] + 1e-9F);
    }
    loss_val /= static_cast<float>(B);

    bool rg = logits->requires_grad;
    auto result = make_tensor({1}, {loss_val}, rg);

    if (rg) {
        result->inputs = {logits, labels};
        result->backward_fn = [logits, labels, result, probs, B, C]() {
            float grad_out = result->grad[0] / static_cast<float>(B);
            for (size_t i = 0; i < B; i++) {
                size_t cls = static_cast<size_t>(labels->data[i]);
                for (size_t j = 0; j < C; j++) {
                    float y = (j == cls) ? 1.0F : 0.0F;
                    logits->grad[i * C + j] += grad_out * (probs[i * C + j] - y);
                }
            }
        };
    }

    return result;
}

}  // namespace mlf