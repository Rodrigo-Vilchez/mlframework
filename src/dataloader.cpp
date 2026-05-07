#include "mlframework/dataloader.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace mlf {

static uint32_t read_uint32(std::ifstream& f) {
    uint32_t val = 0;
    f.read(reinterpret_cast<char*>(&val), 4);
    return __builtin_bswap32(val);
}

std::pair<std::vector<float>, size_t> IDXLoader::read_images(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path.string());

    uint32_t magic = read_uint32(f);
    if ((magic >> 8) != 0x000008)
        throw std::runtime_error("Invalid IDX image file: " + path.string());

    uint8_t num_dims = magic & 0xFF;
    if (num_dims < 1) throw std::runtime_error("IDX image file has no dimensions");

    uint32_t n = read_uint32(f);

    // read remaining dimensions and compute feature_size
    size_t feature_size = 1;
    for (uint8_t d = 1; d < num_dims; d++) {
        uint32_t dim = read_uint32(f);
        feature_size *= dim;
    }

    std::vector<float> data(static_cast<size_t>(n) * feature_size);
    for (size_t i = 0; i < static_cast<size_t>(n) * feature_size; i++) {
        uint8_t byte = 0;
        f.read(reinterpret_cast<char*>(&byte), 1);
        data[i] = static_cast<float>(byte) / 255.0F;
    }
    return {data, feature_size};
}

std::vector<float> IDXLoader::read_labels(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path.string());

    uint32_t magic = read_uint32(f);
    if (magic != 0x00000801) throw std::runtime_error("Invalid IDX label file: " + path.string());

    uint32_t n = read_uint32(f);
    std::vector<float> labels(n);
    for (size_t i = 0; i < n; i++) {
        uint8_t byte = 0;
        f.read(reinterpret_cast<char*>(&byte), 1);
        labels[i] = static_cast<float>(byte);
    }
    return labels;
}

IDXLoader::IDXLoader(const std::string& base_path, size_t batch_size, bool train, bool shuffle)
    : batch_size_(batch_size), cursor_(0), shuffle_(shuffle) {
    namespace fs = std::filesystem;
    fs::path base(base_path);

    fs::path images_path =
        train ? base / "train-images-idx3-ubyte" : base / "t10k-images-idx3-ubyte";
    fs::path labels_path =
        train ? base / "train-labels-idx1-ubyte" : base / "t10k-labels-idx1-ubyte";

    auto [imgs, feat] = read_images(images_path);
    images_ = std::move(imgs);
    feature_size_ = feat;
    labels_ = read_labels(labels_path);
    num_samples_ = labels_.size();

    indices_.resize(num_samples_);
    std::iota(indices_.begin(), indices_.end(), 0);

    if (shuffle_) {
        std::mt19937 rng(42);
        std::shuffle(indices_.begin(), indices_.end(), rng);
    }
}

void IDXLoader::reset() {
    cursor_ = 0;
    if (shuffle_) {
        std::mt19937 rng(std::random_device{}());
        std::shuffle(indices_.begin(), indices_.end(), rng);
    }
}

bool IDXLoader::has_next() const { return cursor_ + batch_size_ <= num_samples_; }

Batch IDXLoader::next() {
    if (!has_next()) throw std::runtime_error("No more batches");

    std::vector<float> img_data(batch_size_ * feature_size_);
    std::vector<float> lbl_data(batch_size_);

    for (size_t i = 0; i < batch_size_; i++) {
        size_t idx = indices_[cursor_ + i];
        std::copy(images_.begin() + idx * feature_size_,
                  images_.begin() + idx * feature_size_ + feature_size_,
                  img_data.begin() + i * feature_size_);
        lbl_data[i] = labels_[idx];
    }

    cursor_ += batch_size_;
    return {make_tensor({batch_size_, feature_size_}, img_data),
            make_tensor({batch_size_}, lbl_data)};
}

}  // namespace mlf