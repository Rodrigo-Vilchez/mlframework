#pragma once

#include <string>
#include <vector>

#include "mlframework/tensor.hpp"

namespace mlf {

struct Batch {
    TensorPtr images;  // {batch_size, 784}
    TensorPtr labels;  // {batch_size}
};

class MNISTLoader {
   public:
    MNISTLoader(const std::string& images_path, const std::string& labels_path, size_t batch_size,
                bool shuffle = true);

    void reset();  // shuffle + reset index for new epoch
    bool has_next() const;
    Batch next();

    size_t num_samples() const { return num_samples_; }
    size_t num_batches() const { return num_samples_ / batch_size_; }

   private:
    std::vector<float> images_;    // all images flat, normalized [0,1]
    std::vector<float> labels_;    // all labels
    std::vector<size_t> indices_;  // shuffled index

    size_t num_samples_;
    size_t batch_size_;
    size_t cursor_;
    bool shuffle_;

    static std::vector<float> read_images(const std::string& path);
    static std::vector<float> read_labels(const std::string& path);
};

}  // namespace mlf