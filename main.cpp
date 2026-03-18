#include <iostream>
#include <vector>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <filesystem>

#include "Sequential.h"
#include "Dataset.h"
/*
#include "Tensor.h"
#include "Loss.h"
#include "Optimizer.h"
#include "EarlyStopping.h"
#include "Scheduler.h"

struct EvalResult { float loss; float acc; size_t total; };
*/
/* EvalResult evaluate(Sequential& model, Dataloader& loader) {
   loader.reset();
    double loss_sum = 0.0;
    size_t total = 0;
    size_t correct = 0;

    while (loader.has_next()) {
        Batch batch = loader.next();
        Tensor logits = model.forward();

        Tensor dummy_grad;
        float loss = ce_loss(logits, batch.labels, dummy_grad);
        loss_sum += loss * batch.labels.size();

        size_t B = logits.get_shape()[0];
        size_t C = logits.get_shape()[1];

        for (size_t b = 0; b < B; b++) {
            float best_val = -1e30f;
            size_t best_idx = 0;
            for (size_t c = 0; c < C; c++) {
                float v = logits.data[b*C + c];
                if (v > best_val) { best_val = v; best_idx = c; }
            }
            if (best_idx == batch.labels[b]) correct++;
            total++;
        }
    }

    return {
        float(loss_sum / total),
        float(correct) / float(total),
        total
    };
return EvalResult();}
*/

struct Training{
    Training(const YAML::Node& cfg)
        :lr(cfg["optimizer"]["learning_rate"].as<float>()),
        weight_decay(cfg["optimizer"]["weight_decay"].as<float>()),
        momentum(cfg["optimizer"]["momentum"].as<float>()),
        dampening(cfg["optimizer"]["dampening"].as<float>()),
        nesterov(cfg["optimizer"]["nesterov"].as<bool>()),
        patience(cfg["early_stopping"]["patience"].as<int>()),
        scheduler_patience(cfg["scheduler"]["patience"].as<int>()),
        reduction_factor(cfg["scheduler"]["reduction_factor"].as<float>()),
        min_lr(cfg["scheduler"]["minimal_lr"].as<float>()),
        min_lr_change(cfg["scheduler"]["minimal_lr_change"].as<float>()),
        epochs(cfg["epochs"].as<int>())
        {}
    const float lr; /**< learning rate.*/
    const float weight_decay; /**< ??? */
    const float momentum; /**< ??? */
    const float dampening; /**< ??? */
    const float nesterov; /**< ??? */

    const float min_lr; /**< ??? */
    const float min_lr_change; /**< ??? */
    const float reduction_factor; /**< ??? */
    
    const size_t patience; /**< ??? */
    const size_t scheduler_patience; /**< ??? */
    const size_t epochs; /**< ??? */
    
};

int main()
{
    std::unique_ptr<YAML::Node> cfg = std::make_unique<YAML::Node>(YAML::LoadFile("../config/config.yaml"));

    auto t1 = std::chrono::high_resolution_clock::now();

    // dataset
    Dataset train_full((*cfg)["data"], true);
    
    Dataset test_ds((*cfg)["data"], false);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << (t2-t1).count() / 1000000 << " ms" << std::endl;
    
    Sequential model = Sequential::from_config(*cfg);
    
    Training train((*cfg)["train"]);

    for (size_t epoch{0}; epoch < train.epochs; epoch++){
        auto t0 = std::chrono::high_resolution_clock::now();        
        
        while(model.request_batch(train_full)){
            model.forward();
            //model.backward();
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Epoch " << epoch
                  << " | time=" << sec << "s\n";

    }

    return 1;

    

    /*
    // training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        auto t0 = std::chrono::high_resolution_clock::now();        
        Batch batch;
        double epoch_loss_sum = 0.0;
        size_t n_samples = 0;

        while (batch.next()) {
            batch = train_loader.next();

            Tensor logits = model.forward(batch.images);
            
            Tensor grad_logits;
            float loss = ce_loss(logits, batch.labels, grad_logits);
            epoch_loss_sum += loss * batch.labels.size();
            n_samples += batch.labels.size();

            model.backward(grad_logits);
            
            sgd(model, lr, momentum, dampening, weight_decay, nesterov);
        }

        float train_loss = epoch_loss_sum / n_samples;

        EvalResult val_res = evaluate(model, val_loader);

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Epoch " << epoch
                  << " | train=" << train_loss
                  << " | val=" << val_res.loss
                  << " | acc=" << val_res.acc
                  << " | time=" << sec << "s\n";

        if (early_stopping(val_res.loss, patience)) {
            break;
        }
        reduce_lr_on_plateau(val_res.loss, epoch, lr, reduction_factor, scheduler_patience, minimal_lr, minimal_lr_change);
    }
        

    // final test
    EvalResult test_res = evaluate(model, test_loader);
    std::cout << "Final test: loss=" << test_res.loss
              << " acc=" << test_res.acc
              << " samples=" << test_res.total << "\n";
    */
    return 0;
}
