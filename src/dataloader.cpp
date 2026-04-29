#include "mlframework/dataloader.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace mlf {

// IDX format stores integers in big-endian - swap bytes on x86
static uint32_t read_uint32(std::ifstream& f) {
    uint32_t val = 0;
    f.read(reinterpret_cast<char*>(&val), 4);
    return __builtin_bswap32(val);
}

std::vector<float> MNISTLoader::read_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic = read_uint32(f);
    if (magic != 2051) throw std::runtime_error("Invalid MNIST image file");

    uint32_t n = read_uint32(f);
    uint32_t rows = read_uint32(f);
    uint32_t cols = read_uint32(f);
    size_t pixels = rows * cols;

    std::vector<float> data(n * pixels);
    for (size_t i = 0; i < n * pixels; i++) {
        uint8_t byte = 0;
        f.read(reinterpret_cast<char*>(&byte), 1);
        data[i] = static_cast<float>(byte) / 255.0F;
    }
    return data;
}

std::vector<float> MNISTLoader::read_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic = read_uint32(f);
    if (magic != 2049) throw std::runtime_error("Invalid MNIST label file");

    uint32_t n = read_uint32(f);
    std::vector<float> labels(n);
    for (size_t i = 0; i < n; i++) {
        uint8_t byte = 0;
        f.read(reinterpret_cast<char*>(&byte), 1);
        labels[i] = static_cast<float>(byte);
    }
    return labels;
}

MNISTLoader::MNISTLoader(const std::string& images_path, const std::string& labels_path,
                         size_t batch_size, bool shuffle)
    : batch_size_(batch_size), cursor_(0), shuffle_(shuffle) {
    images_ = read_images(images_path);
    labels_ = read_labels(labels_path);
    num_samples_ = labels_.size();

    indices_.resize(num_samples_);
    std::iota(indices_.begin(), indices_.end(), 0);

    if (shuffle_) {
        std::mt19937 rng(42);
        std::shuffle(indices_.begin(), indices_.end(), rng);
    }
}

void MNISTLoader::reset() {
    cursor_ = 0;
    if (shuffle_) {
        std::mt19937 rng(std::random_device{}());
        std::shuffle(indices_.begin(), indices_.end(), rng);
    }
}

bool MNISTLoader::has_next() const { return cursor_ + batch_size_ <= num_samples_; }

Batch MNISTLoader::next() {
    if (!has_next()) throw std::runtime_error("No more batches");

    std::vector<float> img_data(batch_size_ * 784);
    std::vector<float> lbl_data(batch_size_);

    for (size_t i = 0; i < batch_size_; i++) {
        size_t idx = indices_[cursor_ + i];
        std::copy(images_.begin() + idx * 784, images_.begin() + idx * 784 + 784,
                  img_data.begin() + i * 784);
        lbl_data[i] = labels_[idx];
    }

    cursor_ += batch_size_;

    return {make_tensor({batch_size_, 784}, img_data), make_tensor({batch_size_}, lbl_data)};
}

}  // namespace mlf