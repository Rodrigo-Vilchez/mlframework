#pragma once

#include "mlframework/tensor.hpp"

namespace mlf {

// logits: {batch, num_classes}, labels: {batch} (class indices as floats)
TensorPtr cross_entropy(TensorPtr logits, TensorPtr labels);

}  // namespace mlf