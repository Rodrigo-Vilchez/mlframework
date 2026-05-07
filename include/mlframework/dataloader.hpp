#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "mlframework/tensor.hpp"

namespace mlf {

struct Batch {
    TensorPtr images;  // {batch_size, feature_size}
    TensorPtr labels;  // {batch_size}
};

class IDXLoader {
   public:
    IDXLoader(const std::string& base_path, size_t batch_size, bool train, bool shuffle = true);

    void reset();
    bool has_next() const;
    Batch next();

    size_t num_samples() const { return num_samples_; }
    size_t feature_size() const { return feature_size_; }
    size_t num_batches() const { return num_samples_ / batch_size_; }

   private:
    std::vector<float> images_;
    std::vector<float> labels_;
    std::vector<size_t> indices_;

    size_t num_samples_;
    size_t feature_size_;
    size_t batch_size_;
    size_t cursor_;
    bool shuffle_;

    static std::pair<std::vector<float>, size_t> read_images(const std::filesystem::path& path);
    static std::vector<float> read_labels(const std::filesystem::path& path);
};

}  // namespace mlf