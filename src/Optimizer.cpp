#include <vector>
#include <Sequential.h>
#include <Optimizer.h>


void sgd(Sequential& model, 
         float lr,
         float momentum,
         float dampening,
         float weight_decay,
         bool nesterov
        ) {
            /*

    float current_update = 0.0;
    float current_b = 0.0;

    for (auto& layer_variant : model.layers) {
        std::visit([&](auto& layer) {
            using L = std::decay_t<decltype(layer)>;

            if constexpr (std::is_same_v<L, DenseLayer>) {
                // weights

                current_update = 0.0;
                current_b = 0.0;

                for (size_t i = 0; i < layer.weights.data.size(); i++) {

                    current_update = layer.weights.grad_[i];

                    if (weight_decay != 0) {
                        current_update -= weight_decay * layer.weights.data[i];
                    }
                    if (momentum != 0) {
                        if (i > 0) {
                            current_b = momentum * current_b + (1 - dampening) * current_update;
                        }
                        else {
                            current_b = current_update;
                        }

                        if (nesterov == true) {
                            current_update += momentum * current_b;
                        }
                        else {
                            current_update = current_b;
                        }
                    }

                    layer.weights.data[i] -= lr * current_update;
                    // layer.weights.grad_[i] = 0.0f;
                    
                }
                layer.weights.zero_grad();

                // bias, no weight decay

                current_update = 0.0;
                current_b = 0.0;

                for (size_t i = 0; i < layer.bias.data.size(); i++) {


                    current_update = layer.bias.grad_[i];

                    if (momentum != 0) {
                        if (i > 0) {
                            current_b = momentum * current_b + (1 - dampening) * current_update;
                        }
                        else {
                            current_b = current_update;
                        }

                        if (nesterov == true) {
                            current_update += momentum * current_b;
                        }
                        else {
                            current_update = current_b;
                        }
                    }

                    layer.bias.data[i] -= lr * current_update;
                    // layer.bias.grad_[i] = 0.0f;
                }
                layer.bias.zero_grad();
            }

        }, layer_variant);
    }
        */
}