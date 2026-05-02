#include "mlframework/model_io.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace mlf {

static constexpr char MAGIC[4] = {'M', 'L', 'F', '1'};
static constexpr uint32_t VERSION = 2;

static void write_bytes(std::ofstream& f, const void* src, size_t n) {
    f.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(n));
}

static void read_bytes(std::ifstream& f, void* dst, size_t n) {
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
}

void save_model(const MLP& model, const ModelConfig& cfg, const std::string& path) {
    std::cout << "Saving model to " << path << "... ";

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open file for writing: " + path);

    // --- header ---
    write_bytes(f, MAGIC, 4);
    write_bytes(f, &VERSION, sizeof(VERSION));

    // --- architecture ---
    write_bytes(f, &cfg.input_size, sizeof(size_t));

    size_t num_hidden = cfg.hidden_sizes.size();
    write_bytes(f, &num_hidden, sizeof(size_t));
    for (size_t h : cfg.hidden_sizes) {
        write_bytes(f, &h, sizeof(size_t));
    }

    write_bytes(f, &cfg.output_size, sizeof(size_t));
    write_bytes(f, &cfg.dropout_p, sizeof(float));
    uint8_t bn = cfg.use_batchnorm ? 1 : 0;
    write_bytes(f, &bn, sizeof(uint8_t));

    // --- tensors ---
    auto params = model.parameters();
    size_t num_tensors = params.size();
    write_bytes(f, &num_tensors, sizeof(size_t));

    size_t total_params = 0;
    for (auto& p : params) {
        // bring to CPU if in CUDA - data vector is empty on device
        TensorPtr p_cpu = (p->device == Device::CUDA) ? to(p, Device::CPU) : p;

        size_t rank = p_cpu->shape.size();
        write_bytes(f, &rank, sizeof(size_t));
        for (size_t dim : p_cpu->shape) {
            write_bytes(f, &dim, sizeof(size_t));
        }
        write_bytes(f, p_cpu->data.data(), p_cpu->numel() * sizeof(float));
        total_params += p_cpu->numel();
    }

    f.flush();
    size_t file_bytes = static_cast<size_t>(f.tellp());
    std::cout << "saved (" << file_bytes / 1024 << " KB, " << num_tensors << " tensors, "
              << total_params << " parameters)\n";

    // --- running stats ---
    Module::RunningStats stats;
    model.collect_running_stats(stats);

    size_t num_blocks = stats.size();
    write_bytes(f, &num_blocks, sizeof(size_t));

    for (auto& [mean, var] : stats) {
        size_t n = mean.size();
        write_bytes(f, &n, sizeof(size_t));
        write_bytes(f, mean.data(), n * sizeof(float));
        write_bytes(f, var.data(), n * sizeof(float));
    }
}

ModelConfig load_model(MLP& model, const std::string& path) {
    std::cout << "Loading model from " << path << "...\n";

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open file: " + path);

    // --- header ---
    char magic[4];
    read_bytes(f, magic, 4);
    if (std::memcmp(magic, MAGIC, 4) != 0) {
        throw std::runtime_error("invalid magic number (expected MLF1)");
    }

    uint32_t version = 0;
    read_bytes(f, &version, sizeof(version));
    if (version != VERSION) {
        throw std::runtime_error("unsupported version (got " + std::to_string(version) +
                                 ", expected " + std::to_string(VERSION) + ")");
    }

    // --- architecture ---
    ModelConfig cfg;
    read_bytes(f, &cfg.input_size, sizeof(size_t));

    size_t num_hidden = 0;
    read_bytes(f, &num_hidden, sizeof(size_t));
    cfg.hidden_sizes.resize(num_hidden);
    for (size_t& h : cfg.hidden_sizes) {
        read_bytes(f, &h, sizeof(size_t));
    }

    read_bytes(f, &cfg.output_size, sizeof(size_t));
    read_bytes(f, &cfg.dropout_p, sizeof(float));
    uint8_t bn = 0;
    read_bytes(f, &bn, sizeof(uint8_t));
    cfg.use_batchnorm = (bn != 0);

    // print architecture
    std::cout << "  Architecture : " << cfg.input_size << " -> [";
    for (size_t i = 0; i < cfg.hidden_sizes.size(); i++) {
        std::cout << cfg.hidden_sizes[i];
        if (i + 1 < cfg.hidden_sizes.size()) std::cout << ", ";
    }
    std::cout << "] -> " << cfg.output_size << "\n";
    std::cout << "  BatchNorm    : " << (cfg.use_batchnorm ? "yes" : "no") << "\n";
    std::cout << "  Dropout      : " << cfg.dropout_p << "\n";

    // reconstruct model with loaded architecture
    model =
        MLP(cfg.input_size, cfg.hidden_sizes, cfg.output_size, cfg.dropout_p, cfg.use_batchnorm);

    // --- tensors ---
    size_t num_tensors = 0;
    read_bytes(f, &num_tensors, sizeof(size_t));

    auto params = model.parameters();
    if (params.size() != num_tensors) {
        throw std::runtime_error("tensor count mismatch: expected " +
                                 std::to_string(params.size()) + ", got " +
                                 std::to_string(num_tensors));
    }

    size_t total_params = 0;
    for (size_t i = 0; i < num_tensors; i++) {
        size_t rank = 0;
        read_bytes(f, &rank, sizeof(size_t));

        std::vector<size_t> shape(rank);
        for (size_t& dim : shape) {
            read_bytes(f, &dim, sizeof(size_t));
        }

        if (shape != params[i]->shape) {
            std::string got, expected;
            for (size_t d : shape) got += std::to_string(d) + " ";
            for (size_t d : params[i]->shape) expected += std::to_string(d) + " ";
            throw std::runtime_error("tensor " + std::to_string(i) + " shape mismatch: expected [" +
                                     expected + "], got [" + got + "]");
        }

        read_bytes(f, params[i]->data.data(), params[i]->numel() * sizeof(float));
        total_params += params[i]->numel();
    }

    std::cout << "  Parameters   : " << total_params << "\n";

    // --- running stats ---
    size_t num_blocks = 0;
    read_bytes(f, &num_blocks, sizeof(size_t));

    Module::RunningStats stats(num_blocks);
    for (size_t i = 0; i < num_blocks; i++) {
        size_t n = 0;
        read_bytes(f, &n, sizeof(size_t));
        stats[i].first.resize(n);
        stats[i].second.resize(n);
        read_bytes(f, stats[i].first.data(), n * sizeof(float));
        read_bytes(f, stats[i].second.data(), n * sizeof(float));
    }

    size_t idx = 0;
    model.apply_running_stats(stats, idx);

    std::cout << "  BatchNorm blocks: " << num_blocks << "\n";
    std::cout << "  loaded.\n";

    return cfg;
}

}  // namespace mlf