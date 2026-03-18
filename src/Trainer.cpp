#include "Trainer.h"
#include <chrono>
#include <iostream>

void train_model(const YAML::Node& cfg,
                 Sequential& model,
                 Dataloader& train_loader,
                 Dataloader& val_loader,
                 SGD& optimizer) {

    int epochs = cfg["train"]["epochs"].as<int>();
    int patience = cfg["train"]["early_stopping"]["patience"].as<int>();

    Tensor grad_logits; // gradient tensor placeholder

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto t0 = std::chrono::high_resolution_clock::now();

        train_loader.reset();
        double loss_sum = 0.0;
        size_t total = 0;

        while (train_loader.has_next()) {
            Batch batch = train_loader.next();
            Tensor x(batch.images, nullptr, {batch.batch_size, batch.pixels});

            Tensor& logits = model.forward(x);

            float loss = ce_loss(logits, batch.labels, grad_logits);
            loss_sum += loss * batch.batch_size;
            total += batch.batch_size;

            model.backward(grad_logits);
            optimizer.step(model);
        }

        float train_loss = loss_sum / total;

        // Evaluate validation set
        EvalResult val_res = evaluate(model, val_loader);

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Epoch " << epoch
                  << " | train=" << train_loss
                  << " | val=" << val_res.loss
                  << " | acc=" << val_res.acc
                  << " | time=" << sec << "s\n";

        if (early_stopping(val_res.loss, patience))
            break;

        reduce_lr_on_plateau(
            val_res.loss, epoch, optimizer.lr,
            cfg["train"]["scheduler"]["reduction_factor"].as<float>(),
            cfg["train"]["scheduler"]["patience"].as<int>(),
            cfg["train"]["scheduler"]["minimal_lr"].as<float>(),
            cfg["train"]["scheduler"]["minimal_lr_change"].as<float>()
        );
    }
}
