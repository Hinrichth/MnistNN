#include "Evaluation.h"
#include "Transformation.h"
#include <execution>


EvalResult evaluate(Sequential& model, Dataloader& loader) {
    loader.reset();

    double loss_sum = 0.0;
    size_t total = 0;
    size_t correct = 0;

    // Don't create grad_dummy here - we'll create it inside the loop
    // Tensor grad_dummy;  // WRONG: Not initialized!
    while (loader.has_next()) {
        Batch batch{loader.next()};

        Tensor x(batch.images, nullptr, {batch.batch_size, batch.pixels});

        Tensor& logits = model.forward(x);

        size_t B = batch.batch_size;
        size_t C = logits.shape_[1];
        
        // Create properly sized gradient tensor
        Tensor grad_dummy({B, C});

        // loss
        float loss = ce_loss(logits, batch.labels, grad_dummy);
        loss_sum += loss * B;

        // accuracy
        const float* data = logits.data_;

        correct += std::accumulate(batch.labels, batch.labels + B, 0, [&](auto lhs, auto rhs){
            const float* max = std::max_element(std::execution::unseq, data, data + C);
            data += C;
            return lhs + (max - data + C == rhs); 
        });

        total += B;
    }

    return {
        static_cast<float>(loss_sum / total),
        static_cast<float>(correct) / static_cast<float>(total),
        total
    };
}