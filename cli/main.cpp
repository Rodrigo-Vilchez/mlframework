#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>

#include "mlframework/cnn.hpp"
#include "mlframework/dataloader.hpp"
#include "mlframework/loss.hpp"
#include "mlframework/mlp.hpp"
#include "mlframework/model_io.hpp"
#include "mlframework/optimizer.hpp"
#include "mlframework/tensor.hpp"

using namespace mlf;

// --- uniform model interface ---

struct Model {
    enum class Type { MLP, CNN };
    Type type;
    std::unique_ptr<MLP> mlp;
    std::unique_ptr<CNN> cnn;

    Module& module() {
        return type == Type::MLP ? static_cast<Module&>(*mlp) : static_cast<Module&>(*cnn);
    }

    TensorPtr forward(TensorPtr x) { return module().forward(x); }

    std::vector<TensorPtr> parameters() const {
        return type == Type::MLP ? mlp->parameters() : cnn->parameters();
    }

    void to(Device device) { module().to(device); }
    void train() { module().train(); }
    void eval() { module().eval(); }
};

// --- helpers ---

static TensorPtr prepare_input(TensorPtr images, Model::Type type, Device device) {
    TensorPtr out = images;
    if (type == Model::Type::CNN) {
        size_t N = images->shape[0];
        out = reshape(out, {N, 1, 28, 28});
    }
    if (device == Device::CUDA) {
        out = to(out, Device::CUDA);
    }
    return out;
}

static float compute_accuracy(Model& model, IDXLoader& loader, Device active_device) {
    loader.reset();
    size_t correct = 0;
    size_t total = 0;

    EvalMode eval;  // deactivates dropout
    NoGrad ng;
    while (loader.has_next()) {
        Batch b = loader.next();
        auto input = prepare_input(b.images, model.type, active_device);
        auto logits = model.forward(input);

        // bring to CPU for argmax
        if (logits->device == Device::CUDA) logits = mlf::to(logits, Device::CPU);

        size_t B = logits->shape[0];
        size_t C = logits->shape[1];

        for (size_t i = 0; i < B; i++) {
            size_t pred = 0;
            float best = logits->data[i * C];
            for (size_t j = 1; j < C; j++) {
                if (logits->data[i * C + j] > best) {
                    best = logits->data[i * C + j];
                    pred = j;
                }
            }
            if (pred == static_cast<size_t>(b.labels->data[i])) correct++;
            total++;
        }
    }
    TrainMode train;  // restores training mode
    return static_cast<float>(correct) / static_cast<float>(total);
}

// --- main ---

int main(int argc, char* argv[]) {
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    std::string data_dir = "data/mnist";
    size_t epochs = 5;
    size_t batch_size = 64;
    float lr = 0.001F;
    std::string save_path;
    std::string load_path;
    bool eval_only = false;
    std::string model_type = "mlp";
    std::string device_str = "cpu";
    bool save_opt = false;
    bool load_opt = false;
    bool lr_explicitly_set = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        }
        if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoul(argv[++i]);
        }
        if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoul(argv[++i]);
        }
        if (arg == "--save" && i + 1 < argc) {
            save_path = (argv[++i]);
        }
        if (arg == "--load" && i + 1 < argc) {
            load_path = (argv[++i]);
        }
        if (arg == "--model" && i + 1 < argc) {
            model_type = argv[++i];
        }
        if (arg == "--eval-only") {
            eval_only = true;
        }
        if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        }
        if (arg == "--save-opt") {
            save_opt = true;
        }
        if (arg == "--load-opt") {
            load_opt = true;
        }
        if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
            lr_explicitly_set = true;
        }
    }

    if (eval_only && load_path.empty()) {
        std::cerr << "Error: --eval-only requires --load\n";
        return 1;
    }

    if (model_type != "mlp" && model_type != "cnn") {
        std::cerr << "Error: --model must be 'mlp' or 'cnn'\n";
        return 1;
    }

    std::cout << "Loading dataset from " << data_dir << "...\n";

    IDXLoader train_loader(data_dir, batch_size, true);
    IDXLoader test_loader(data_dir, batch_size, false);

    std::cout << "Train samples : " << train_loader.num_samples() << "\n";
    std::cout << "Test  samples : " << test_loader.num_samples() << "\n";
    std::cout << "Feature size  : " << train_loader.feature_size() << "\n\n";

    // build model
    Model model;
    if (model_type == "cnn") {
        model.type = Model::Type::CNN;
        model.cnn = std::make_unique<CNN>();
        std::cout << "Model: CNN (conv1→conv2→fc1→fc2)\n";
    } else {
        model.type = Model::Type::MLP;
        model.mlp = std::make_unique<MLP>(784, std::vector<size_t>{128, 64}, 10, 0.3F, true);
        std::cout << "Model: MLP (784→128→64→10)\n";
    }

    // load
    if (!load_path.empty()) {
        ModelType file_type = peek_model_type(load_path);

        if (file_type == ModelType::MLP && model.type != Model::Type::MLP)
            throw std::runtime_error("file contains MLP but --model cnn was specified");
        if (file_type == ModelType::CNN && model.type != Model::Type::CNN)
            throw std::runtime_error("file contains CNN but --model mlp was specified");

        if (file_type == ModelType::CNN) {
            CNNConfig cfg = peek_cnn_config(load_path);
            model.cnn = std::make_unique<CNN>(cfg);
        }

        load_model(model.module(), file_type, load_path);
    }

    if (device_str == "cuda") {
        std::cout << "Moving model to CUDA...\n";
        if (model.type == Model::Type::MLP)
            model.mlp->to(Device::CUDA);
        else
            model.cnn->to(Device::CUDA);
        std::cout << "Done.\n";
    }

    Device active_device = (device_str == "cuda") ? Device::CUDA : Device::CPU;

    if (eval_only) {
        float acc = compute_accuracy(model, test_loader, active_device);
        size_t correct = static_cast<size_t>(acc * test_loader.num_samples());
        std::cout << "Test accuracy: " << acc << " (" << correct << " / "
                  << test_loader.num_samples() << ")\n";
        return 0;
    }

    // SGD opt(model.parameters(), lr);
    Adam opt(model.parameters(), lr);
    CosineAnnealingWR scheduler(lr, 1e-5F, 5, 2.0F);  // T0=5, doubles each restart

    if (load_opt) {
        if (load_path.empty()) {
            std::cerr << "Error: --load-opt requires --load\n";
            return 1;
        }
        load_optimizer(opt, scheduler, load_path);
        // lr comes from restored scheduler, not --lr flag
        opt.set_lr(scheduler.get_lr());

        if (lr_explicitly_set) {
            std::cerr << "Warning: --lr " << lr << " ignored when --load-opt is active. "
                      << "Resuming with lr=" << scheduler.get_lr() << " (restored from file).\n";
        }

        std::cout << "  Resuming at lr=" << scheduler.get_lr() << "\n";
    }

    std::cout << "lr_max=" << scheduler.lr_max() << "  lr_min=" << scheduler.lr_min()
              << "  T0=" << scheduler.T0() << "  T_mult=" << scheduler.T_mult() << "\n\n";

    for (size_t epoch = 1; epoch <= epochs; epoch++) {
        float current_lr = scheduler.get_lr();
        opt.set_lr(current_lr);

        train_loader.reset();
        float epoch_loss = 0.0F;
        size_t steps = 0;

        while (train_loader.has_next()) {
            Batch b = train_loader.next();
            auto input = prepare_input(b.images, model.type, active_device);
            auto labels = (active_device == Device::CUDA) ? to(b.labels, Device::CUDA) : b.labels;
            opt.zero_grad();
            auto logits = model.forward(input);
            auto loss = cross_entropy(logits, labels);
            loss->backward();

            if (active_device == Device::CUDA) cudaDeviceSynchronize();

            opt.step();

            epoch_loss += loss->data[0];
            steps++;

            if (steps % 100 == 0) {
                std::cout << "  epoch " << epoch << "  step " << steps << "  loss "
                          << epoch_loss / static_cast<float>(steps) << "\r" << std::flush;
            }
        }

        float train_acc = compute_accuracy(model, train_loader, active_device);
        float test_acc = compute_accuracy(model, test_loader, active_device);

        std::cout << "Epoch " << epoch << "  lr " << current_lr << "  loss "
                  << epoch_loss / static_cast<float>(steps) << "  train_acc " << train_acc
                  << "  test_acc " << test_acc << "\n";

        scheduler.step();
    }

    // save
    if (!save_path.empty()) {
        size_t slash = save_path.rfind('/');
        if (slash != std::string::npos) {
            std::string cmd = "mkdir -p " + save_path.substr(0, slash);
            std::system(cmd.c_str());
        }
        if (model.type == Model::Type::MLP) {
            ModelConfig cfg{784, {128, 64}, 10, 0.3F, true};
            save_model(*model.mlp, ModelType::MLP, cfg, save_path);
        } else {
            save_model(*model.cnn, ModelType::CNN, model.cnn->config(), save_path);
        }
        if (save_opt) {
            save_optimizer(model.module(), opt, scheduler, save_path);
        }
    }

    return 0;
}