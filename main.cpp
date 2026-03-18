#include <iostream>
#include <vector>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "Dataset.h"
#include "Dataloader.h"
#include "Batch.h"
#include "Sequential.h"
#include "Tensor.h"
#include "Loss.h"
#include "Optimizer.h"
#include "EarlyStopping.h"
#include "Scheduler.h"
#include "Transformation.h"
#include "Trainer.h"
#include "Evaluation.h"


int main() {
    YAML::Node cfg = YAML::LoadFile("../config/config.yaml");

    // Load datasets
    Dataset train_ds(cfg["data"], true);
    Dataset test_ds(cfg["data"], false);

    float val_ratio = cfg["data"]["val_ratio"].as<float>();
    size_t N = train_ds.size();
    size_t val_size = static_cast<size_t>(N * val_ratio);
    size_t train_size = N - val_size;

    std::cout << "Dataset sizes:" << std::endl;
    std::cout << "  Train dataset total: " << N << std::endl;
    std::cout << "  Train split: " << train_size << std::endl;
    std::cout << "  Val split: " << val_size << std::endl;
    std::cout << "  Test dataset: " << test_ds.size() << std::endl;

    // Create dataset views
    DatasetView train_view{&train_ds, 0, train_size};
    DatasetView val_view{&train_ds, train_size, N};
    DatasetView test_view{&test_ds, 0, test_ds.size()};

    // dataloaders
    size_t batch_size = cfg["data"]["batch_size"].as<size_t>();
    bool shuffle = cfg["data"]["shuffle"].as<bool>();

    // dataloaders
    Dataloader train_loader(train_view, batch_size, shuffle);
    Dataloader val_loader(val_view, batch_size, false);
    Dataloader test_loader(test_view, batch_size, false);

    // Model
    Sequential model = Sequential::from_config(cfg);
    model.build(batch_size);

    // Optimizer
    SGD optimizer(
        cfg["train"]["optimizer"]["learning_rate"].as<float>(),
        cfg["train"]["optimizer"]["momentum"].as<float>(),
        cfg["train"]["optimizer"]["dampening"].as<float>(),
        cfg["train"]["optimizer"]["weight_decay"].as<float>(),
        cfg["train"]["optimizer"]["nesterov"].as<bool>()
    );

    // Train
    train_model(cfg, model, train_loader, val_loader, optimizer);

    // Final test
    EvalResult test_res = evaluate(model, test_loader);
    std::cout << "Final test: loss=" << test_res.loss
              << " acc=" << test_res.acc
              << " samples=" << test_res.total << "\n";

    return 0;
}

