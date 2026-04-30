#include <iostream>
#include <string>

#include "mlframework/dataloader.hpp"
#include "mlframework/loss.hpp"
#include "mlframework/mlp.hpp"
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

int main(int argc, char* argv[]) {
    std::string data_dir = "data/mnist";
    size_t epochs = 5;
    size_t batch_size = 64;
    float lr = 0.01F;

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
    }

    std::cout << "Loading MNIST from " << data_dir << "...\n";

    MNISTLoader train_loader(data_dir + "/train-images-idx3-ubyte",
                             data_dir + "/train-labels-idx1-ubyte", batch_size, true);
    MNISTLoader test_loader(data_dir + "/t10k-images-idx3-ubyte",
                            data_dir + "/t10k-labels-idx1-ubyte", batch_size, false);

    std::cout << "Train samples : " << train_loader.num_samples() << "\n";
    std::cout << "Test  samples : " << test_loader.num_samples() << "\n\n";

    MLP model(784, {128, 64}, 10, 0.3F, true);  // 30% dropout, batch norm activated
    // SGD opt(model.parameters(), lr);
    Adam opt(model.parameters(), lr);

    for (size_t epoch = 1; epoch <= epochs; epoch++) {
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

        std::cout << "Epoch " << epoch << "  loss " << epoch_loss / static_cast<float>(steps)
                  << "  train_acc " << train_acc << "  test_acc " << test_acc << "\n";
    }

    return 0;
}