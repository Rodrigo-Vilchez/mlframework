#include "mlframework/model_io.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace mlf {

static constexpr char MAGIC[4] = {'M', 'L', 'F', '1'};
static constexpr uint32_t VERSION = 5;

static void write_bytes(std::ostream& f, const void* src, size_t n) {
    f.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(n));
}

static void read_bytes(std::istream& f, void* dst, size_t n) {
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
}

// --- architecture helpers ---

static void write_mlp_architecture(std::ostream& f, const ModelConfig& cfg) {
    write_bytes(f, &cfg.input_size, sizeof(size_t));
    size_t num_hidden = cfg.hidden_sizes.size();
    write_bytes(f, &num_hidden, sizeof(size_t));
    for (size_t h : cfg.hidden_sizes) write_bytes(f, &h, sizeof(size_t));
    write_bytes(f, &cfg.output_size, sizeof(size_t));
    write_bytes(f, &cfg.dropout_p, sizeof(float));
    uint8_t bn = cfg.use_batchnorm ? 1 : 0;
    write_bytes(f, &bn, sizeof(uint8_t));
}

static void read_mlp_architecture(std::istream& f, ModelConfig& cfg) {
    read_bytes(f, &cfg.input_size, sizeof(size_t));
    size_t num_hidden = 0;
    read_bytes(f, &num_hidden, sizeof(size_t));
    cfg.hidden_sizes.resize(num_hidden);
    for (size_t& h : cfg.hidden_sizes) read_bytes(f, &h, sizeof(size_t));
    read_bytes(f, &cfg.output_size, sizeof(size_t));
    read_bytes(f, &cfg.dropout_p, sizeof(float));
    uint8_t bn = 0;
    read_bytes(f, &bn, sizeof(uint8_t));
    cfg.use_batchnorm = (bn != 0);
}

static void write_cnn_architecture(std::ostream& f, const CNNConfig& cfg) {
    write_bytes(f, &cfg.input_channels, sizeof(size_t));
    write_bytes(f, &cfg.input_size, sizeof(size_t));
    size_t num_conv = cfg.conv_layers.size();
    write_bytes(f, &num_conv, sizeof(size_t));
    for (const auto& blk : cfg.conv_layers) {
        write_bytes(f, &blk.out_channels, sizeof(size_t));
        write_bytes(f, &blk.kernel_size, sizeof(size_t));
        write_bytes(f, &blk.stride, sizeof(size_t));
        write_bytes(f, &blk.padding, sizeof(size_t));
        uint8_t pool = blk.pool ? 1 : 0;
        write_bytes(f, &pool, sizeof(uint8_t));
        write_bytes(f, &blk.pool_k, sizeof(size_t));
        write_bytes(f, &blk.pool_s, sizeof(size_t));
    }
    write_bytes(f, &cfg.hidden_dim, sizeof(size_t));
    write_bytes(f, &cfg.output_size, sizeof(size_t));
    write_bytes(f, &cfg.dropout_p, sizeof(float));
}

static void read_cnn_architecture(std::istream& f, CNNConfig& cfg) {
    read_bytes(f, &cfg.input_channels, sizeof(size_t));
    read_bytes(f, &cfg.input_size, sizeof(size_t));
    size_t num_conv = 0;
    read_bytes(f, &num_conv, sizeof(size_t));
    cfg.conv_layers.resize(num_conv);
    for (auto& blk : cfg.conv_layers) {
        read_bytes(f, &blk.out_channels, sizeof(size_t));
        read_bytes(f, &blk.kernel_size, sizeof(size_t));
        read_bytes(f, &blk.stride, sizeof(size_t));
        read_bytes(f, &blk.padding, sizeof(size_t));
        uint8_t pool = 0;
        read_bytes(f, &pool, sizeof(uint8_t));
        blk.pool = (pool != 0);
        read_bytes(f, &blk.pool_k, sizeof(size_t));
        read_bytes(f, &blk.pool_s, sizeof(size_t));
    }
    read_bytes(f, &cfg.hidden_dim, sizeof(size_t));
    read_bytes(f, &cfg.output_size, sizeof(size_t));
    read_bytes(f, &cfg.dropout_p, sizeof(float));
}

// skips architecture block without capturing values (used by load_optimizer)
static void skip_architecture(std::istream& f, ModelType type) {
    if (type == ModelType::MLP) {
        size_t input_size = 0, num_hidden = 0;
        read_bytes(f, &input_size, sizeof(size_t));
        read_bytes(f, &num_hidden, sizeof(size_t));
        f.seekg(static_cast<std::streamoff>(num_hidden * sizeof(size_t) + sizeof(size_t) +
                                            sizeof(float) + sizeof(uint8_t)),
                std::ios::cur);
    } else {
        size_t ic = 0, is = 0, num_conv = 0;
        read_bytes(f, &ic, sizeof(size_t));
        read_bytes(f, &is, sizeof(size_t));
        read_bytes(f, &num_conv, sizeof(size_t));
        // each ConvBlockConfig: 4*size_t + uint8_t + 2*size_t
        constexpr size_t blk_bytes = 4 * sizeof(size_t) + sizeof(uint8_t) + 2 * sizeof(size_t);
        f.seekg(
            static_cast<std::streamoff>(num_conv * blk_bytes + 2 * sizeof(size_t) + sizeof(float)),
            std::ios::cur);
    }
}

// --- header helpers ---

static void validate_header(std::ifstream& f, const std::string& path) {
    char magic[4];
    read_bytes(f, magic, 4);
    if (std::memcmp(magic, MAGIC, 4) != 0)
        throw std::runtime_error("invalid magic number (expected MLF1): " + path);
    uint32_t version = 0;
    read_bytes(f, &version, sizeof(version));
    if (version != VERSION)
        throw std::runtime_error("unsupported version (got " + std::to_string(version) +
                                 ", expected " + std::to_string(VERSION) + "): " + path);
}

// --- public API ---

ModelType peek_model_type(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    validate_header(f, path);
    uint32_t type_raw = 0;
    read_bytes(f, &type_raw, sizeof(type_raw));
    return static_cast<ModelType>(type_raw);
}

CNNConfig peek_cnn_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    validate_header(f, path);
    uint32_t type_raw = 0;
    read_bytes(f, &type_raw, sizeof(type_raw));
    if (static_cast<ModelType>(type_raw) != ModelType::CNN)
        throw std::runtime_error("peek_cnn_config: file does not contain a CNN model");
    CNNConfig cfg;
    read_cnn_architecture(f, cfg);
    return cfg;
}

void save_model(const Module& model, ModelType type, const ConfigVariant& config,
                const std::string& path) {
    std::cout << "Saving model to " << path << "... ";

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open file for writing: " + path);

    // header
    write_bytes(f, MAGIC, 4);
    write_bytes(f, &VERSION, sizeof(VERSION));
    uint32_t type_raw = static_cast<uint32_t>(type);
    write_bytes(f, &type_raw, sizeof(type_raw));

    // architecture
    if (std::holds_alternative<ModelConfig>(config))
        write_mlp_architecture(f, std::get<ModelConfig>(config));
    else
        write_cnn_architecture(f, std::get<CNNConfig>(config));

    // tensors
    auto params = model.parameters();
    size_t num_tensors = params.size();
    write_bytes(f, &num_tensors, sizeof(size_t));

    size_t total_params = 0;
    for (auto& p : params) {
        TensorPtr p_cpu = (p->device == Device::CUDA) ? to(p, Device::CPU) : p;
        size_t rank = p_cpu->shape.size();
        write_bytes(f, &rank, sizeof(size_t));
        for (size_t dim : p_cpu->shape) write_bytes(f, &dim, sizeof(size_t));
        write_bytes(f, p_cpu->data.data(), p_cpu->numel() * sizeof(float));
        total_params += p_cpu->numel();
    }

    // running stats
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

    f.flush();
    size_t file_bytes = static_cast<size_t>(f.tellp());
    std::cout << "saved (" << file_bytes / 1024 << " KB, " << num_tensors << " tensors, "
              << total_params << " parameters)\n";
}

ConfigVariant load_model(Module& model, ModelType type, const std::string& path) {
    std::cout << "Loading model from " << path << "...\n";

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);

    validate_header(f, path);

    uint32_t type_raw = 0;
    read_bytes(f, &type_raw, sizeof(type_raw));
    if (static_cast<ModelType>(type_raw) != type)
        throw std::runtime_error("model type mismatch in file");

    // read architecture for display only, the model was already constructed by caller
    ConfigVariant result_config;
    if (type == ModelType::MLP) {
        ModelConfig cfg;
        read_mlp_architecture(f, cfg);
        std::cout << "  Architecture : " << cfg.input_size << " -> [";
        for (size_t i = 0; i < cfg.hidden_sizes.size(); i++) {
            std::cout << cfg.hidden_sizes[i];
            if (i + 1 < cfg.hidden_sizes.size()) std::cout << ", ";
        }
        std::cout << "] -> " << cfg.output_size << "\n";
        std::cout << "  BatchNorm    : " << (cfg.use_batchnorm ? "yes" : "no") << "\n";
        std::cout << "  Dropout      : " << cfg.dropout_p << "\n";
        result_config = cfg;
    } else {
        CNNConfig cfg;
        read_cnn_architecture(f, cfg);
        std::cout << "  Architecture : CNN (" << cfg.input_channels << "ch, " << cfg.input_size
                  << "x" << cfg.input_size << ")\n";
        std::cout << "  Conv blocks  : " << cfg.conv_layers.size() << "\n";
        std::cout << "  Hidden dim   : " << cfg.hidden_dim << "\n";
        std::cout << "  Dropout      : " << cfg.dropout_p << "\n";
        result_config = cfg;
    }

    // tensors
    size_t num_tensors = 0;
    read_bytes(f, &num_tensors, sizeof(size_t));

    auto params = model.parameters();
    if (params.size() != num_tensors)
        throw std::runtime_error("tensor count mismatch: expected " +
                                 std::to_string(params.size()) + ", got " +
                                 std::to_string(num_tensors));

    size_t total_params = 0;
    for (size_t i = 0; i < num_tensors; i++) {
        size_t rank = 0;
        read_bytes(f, &rank, sizeof(size_t));
        std::vector<size_t> shape(rank);
        for (size_t& dim : shape) read_bytes(f, &dim, sizeof(size_t));
        if (shape != params[i]->shape)
            throw std::runtime_error("tensor " + std::to_string(i) + " shape mismatch");
        read_bytes(f, params[i]->data.data(), params[i]->numel() * sizeof(float));
        total_params += params[i]->numel();
    }

    // running stats
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

    std::cout << "  Parameters      : " << total_params << "\n";
    std::cout << "  BatchNorm blocks: " << num_blocks << "\n";
    std::cout << "  loaded.\n";
    return result_config;
}

// --- optimizer persistence ---

static constexpr uint8_t OPT_ADAM = 1;

void save_optimizer(const Module& model, const Adam& opt, const CosineAnnealingWR& scheduler,
                    const std::string& path) {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!f) throw std::runtime_error("cannot open for update: " + path);
    f.seekp(0, std::ios::end);

    uint8_t has_opt = 1;
    uint8_t opt_type = OPT_ADAM;
    write_bytes(f, &has_opt, sizeof(uint8_t));
    write_bytes(f, &opt_type, sizeof(uint8_t));

    size_t t = opt.step_count();
    write_bytes(f, &t, sizeof(size_t));

    size_t num_params = opt.num_params();
    write_bytes(f, &num_params, sizeof(size_t));

    std::vector<float> host_buf;
    for (size_t i = 0; i < num_params; i++) {
        size_t n = opt.param_numel(i);
        write_bytes(f, &n, sizeof(size_t));
        host_buf.resize(n);
        if (opt.on_cuda()) {
            cudaMemcpy(host_buf.data(), opt.cuda_m()[i].get(), n * sizeof(float),
                       cudaMemcpyDeviceToHost);
            write_bytes(f, host_buf.data(), n * sizeof(float));
            cudaMemcpy(host_buf.data(), opt.cuda_v()[i].get(), n * sizeof(float),
                       cudaMemcpyDeviceToHost);
            write_bytes(f, host_buf.data(), n * sizeof(float));
        } else {
            write_bytes(f, opt.cpu_m()[i].data(), n * sizeof(float));
            write_bytes(f, opt.cpu_v()[i].data(), n * sizeof(float));
        }
    }

    size_t t_cur = scheduler.t_cur();
    size_t T_cur = scheduler.T_cur();
    float lr_max = scheduler.lr_max();
    float lr_min = scheduler.lr_min();
    size_t T0 = scheduler.T0();
    float T_mult = scheduler.T_mult();
    write_bytes(f, &t_cur, sizeof(size_t));
    write_bytes(f, &T_cur, sizeof(size_t));
    write_bytes(f, &lr_max, sizeof(float));
    write_bytes(f, &lr_min, sizeof(float));
    write_bytes(f, &T0, sizeof(size_t));
    write_bytes(f, &T_mult, sizeof(float));

    f.flush();
    std::cout << "Optimizer state saved.\n";
}

void load_optimizer(Adam& opt, CosineAnnealingWR& scheduler, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);

    // skip header: magic(4) + version(4) + model_type(4)
    char magic_skip[4];
    read_bytes(f, magic_skip, 4);
    uint32_t ver_skip = 0;
    read_bytes(f, &ver_skip, sizeof(uint32_t));
    uint32_t type_skip = 0;
    read_bytes(f, &type_skip, sizeof(uint32_t));
    ModelType mt = static_cast<ModelType>(type_skip);

    // skip architecture (handles both MLP and CNN)
    skip_architecture(f, mt);

    // skip tensors
    size_t num_tensors = 0;
    read_bytes(f, &num_tensors, sizeof(size_t));
    for (size_t i = 0; i < num_tensors; i++) {
        size_t rank = 0;
        read_bytes(f, &rank, sizeof(size_t));
        size_t numel = 1;
        for (size_t d = 0; d < rank; d++) {
            size_t dim = 0;
            read_bytes(f, &dim, sizeof(size_t));
            numel *= dim;
        }
        f.seekg(static_cast<std::streamoff>(numel * sizeof(float)), std::ios::cur);
    }

    // skip running stats
    size_t num_blocks = 0;
    read_bytes(f, &num_blocks, sizeof(size_t));
    for (size_t i = 0; i < num_blocks; i++) {
        size_t n = 0;
        read_bytes(f, &n, sizeof(size_t));
        f.seekg(static_cast<std::streamoff>(2 * n * sizeof(float)), std::ios::cur);
    }

    // read optimizer block
    uint8_t has_opt = 0;
    read_bytes(f, &has_opt, sizeof(uint8_t));
    if (!has_opt)
        throw std::runtime_error(
            "model file has no optimizer state — was it saved with --save-opt?");

    uint8_t opt_type = 0;
    read_bytes(f, &opt_type, sizeof(uint8_t));
    if (opt_type != OPT_ADAM)
        throw std::runtime_error("optimizer type mismatch: expected Adam (1), got " +
                                 std::to_string(opt_type));

    size_t t = 0;
    read_bytes(f, &t, sizeof(size_t));

    size_t num_params = 0;
    read_bytes(f, &num_params, sizeof(size_t));
    if (num_params != opt.num_params())
        throw std::runtime_error("optimizer param count mismatch: expected " +
                                 std::to_string(opt.num_params()) + ", got " +
                                 std::to_string(num_params));

    std::vector<std::vector<float>> m(num_params), v(num_params);
    for (size_t i = 0; i < num_params; i++) {
        size_t n = 0;
        read_bytes(f, &n, sizeof(size_t));
        if (n != opt.param_numel(i))
            throw std::runtime_error("optimizer param " + std::to_string(i) + " numel mismatch");
        m[i].resize(n);
        v[i].resize(n);
        read_bytes(f, m[i].data(), n * sizeof(float));
        read_bytes(f, v[i].data(), n * sizeof(float));
    }
    opt.restore_state(t, std::move(m), std::move(v));

    size_t t_cur = 0, T_cur = 0, T0 = 0;
    float lr_max = 0, lr_min = 0, T_mult = 0;
    read_bytes(f, &t_cur, sizeof(size_t));
    read_bytes(f, &T_cur, sizeof(size_t));
    read_bytes(f, &lr_max, sizeof(float));
    read_bytes(f, &lr_min, sizeof(float));
    read_bytes(f, &T0, sizeof(size_t));
    read_bytes(f, &T_mult, sizeof(float));

    scheduler = CosineAnnealingWR(lr_max, lr_min, T0, T_mult);
    scheduler.restore_state(t_cur, T_cur);

    std::cout << "  Optimizer state loaded (Adam step=" << t << ", scheduler t_cur=" << t_cur << "/"
              << T_cur << ")\n";
}

}  // namespace mlf