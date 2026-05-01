#include <iostream>
#include <string>

#include "mlframework/dataloader.hpp"
#include "mlframework/loss.hpp"
#include "mlframework/mlp.hpp"
#include "mlframework/model_io.hpp"
#include "mlframework/optimizer.hpp"

using namespace mlf;

static float compute_accuracy(MLP& model, MNISTLoader& loader) {
    loader.reset();
    size_t correct = 0;
    size_t total = 0;

    EvalMode eval;  // deactivates dropout
    NoGrad ng;
    while (loader.has_next()) {
        Batch b = loader.next();
        auto logits = model.forward(b.images);

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

// TODO(#1): remove warm-up once BatchNorm running stats are persisted
static void warmup_batchnorm(MLP& model, MNISTLoader& loader) {
    std::cout << "Running warm-up pass to reconstruct BatchNorm stats...\n";
    loader.reset();
    NoGrad ng;
    while (loader.has_next()) {
        Batch b = loader.next();
        model.forward(b.images);
    }
    std::cout << "Warm-up done.\n";
}

int main(int argc, char* argv[]) {
    std::string data_dir = "data/mnist";
    size_t epochs = 5;
    size_t batch_size = 64;
    float lr = 0.001F;
    std::string save_path;
    std::string load_path;
    bool eval_only = false;

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
        if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        }
        if (arg == "--save" && i + 1 < argc) {
            save_path = (argv[++i]);
        }
        if (arg == "--load" && i + 1 < argc) {
            load_path = (argv[++i]);
        }
        if (arg == "--eval-only") {
            eval_only = true;
        }
    }

    if (eval_only && load_path.empty()) {
        std::cerr << "Error: --eval-only requires --load\n";
        return 1;
    }

    std::cout << "Loading MNIST from " << data_dir << "...\n";

    MNISTLoader train_loader(data_dir + "/train-images-idx3-ubyte",
                             data_dir + "/train-labels-idx1-ubyte", batch_size, true);
    MNISTLoader test_loader(data_dir + "/t10k-images-idx3-ubyte",
                            data_dir + "/t10k-labels-idx1-ubyte", batch_size, false);

    std::cout << "Train samples : " << train_loader.num_samples() << "\n";
    std::cout << "Test  samples : " << test_loader.num_samples() << "\n\n";

    // default config - overwritten by --load
    ModelConfig cfg{784, {128, 64}, 10, 0.3F, true};  // 30% dropout, batch norm activated
    MLP model(cfg.input_size, cfg.hidden_sizes, cfg.output_size, cfg.dropout_p, cfg.use_batchnorm);

    if (!load_path.empty()) {
        cfg = load_model(model, load_path);
        warmup_batchnorm(model, train_loader);
    }

    if (eval_only) {
        float acc = compute_accuracy(model, test_loader);
        size_t correct = static_cast<size_t>(acc * test_loader.num_samples());
        std::cout << "Test accuracy: " << acc << " (" << correct << " / "
                  << test_loader.num_samples() << ")\n";
        return 0;
    }

    // SGD opt(model.parameters(), lr);
    Adam opt(model.parameters(), lr);
    CosineAnnealingWR scheduler(lr, 1e-5F, 5, 2.0F);  // T0=5, doubles each restart

    std::cout << "lr_max=" << lr << "  lr_min=1e-5  T0=5  T_mult=2\n\n";

    for (size_t epoch = 1; epoch <= epochs; epoch++) {
        float current_lr = scheduler.get_lr();
        opt.set_lr(current_lr);

        train_loader.reset();
        float epoch_loss = 0.0F;
        size_t steps = 0;

        while (train_loader.has_next()) {
            Batch b = train_loader.next();

            opt.zero_grad();
            auto logits = model.forward(b.images);
            auto loss = cross_entropy(logits, b.labels);
            loss->backward();
            opt.step();

            epoch_loss += loss->data[0];
            steps++;

            if (steps % 100 == 0) {
                std::cout << "  epoch " << epoch << "  step " << steps << "  loss "
                          << epoch_loss / static_cast<float>(steps) << "\r" << std::flush;
            }
        }

        float train_acc = compute_accuracy(model, train_loader);
        float test_acc = compute_accuracy(model, test_loader);

        std::cout << "Epoch " << epoch << "  lr " << current_lr << "  loss "
                  << epoch_loss / static_cast<float>(steps) << "  train_acc " << train_acc
                  << "  test_acc " << test_acc << "\n";

        scheduler.step();
    }

    if (!save_path.empty()) {
        // ensure directory exists
        size_t slash = save_path.rfind('/');
        if (slash != std::string::npos) {
            std::string dir = save_path.substr(0, slash);
            std::string cmd = "mkdir -p " + dir;
            std::system(cmd.c_str());
        }
        save_model(model, cfg, save_path);
    }

    return 0;
}