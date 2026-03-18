#include "Optimizer.h"
#include <execution>

SGD::SGD(float lr_,
         float momentum_,
         float dampening_,
         float weight_decay_,
         bool nesterov_)
    : lr(lr_),
      momentum(momentum_),
      dampening(dampening_),
      weight_decay(weight_decay_),
      nesterov(nesterov_) {}

void SGD::step(Sequential& model) {
    std::for_each(model.layers.begin(), model.layers.end(), [&](auto& layer_variant){
        std::visit([&](auto& layer) {
            using L = std::decay_t<decltype(layer)>;

            if constexpr (std::is_same_v<L, DenseLayer>) {
                
                auto update_tensor = [&](Tensor& param, const std::string& name) {
                    float* w = param.data_;
                    float* g = param.grad_;
                    size_t n = param.size();
                    
                    float weight_sum = std::accumulate(w, w + n, 0);
                    float grad_sum = std::accumulate(g, g + n, 0);
                    
                    auto& v = velocity[&param];
                    
                    if (v.empty()) {
                        v.resize(n);
                    }

                    // Weight decay (L2 regularization)
                    std::transform(std::execution::unseq, w, w + n, g, g, [&](auto lhs, auto rhs){return rhs + weight_decay * lhs;});
                    
                    if (momentum){
                        
                        std::transform(std::execution::unseq, g, g+n, v.begin(), v.begin(), 
                            [&](float lhs, float rhs){
                                return momentum * rhs + (1.0f - dampening) * lhs;
                            });

                        if (nesterov){
                            std::transform(std::execution::unseq, g, g + n, v.begin(), g, 
                                [&](float lhs, float rhs){return lhs + momentum * rhs;}
                            );
                        }
                        
                        else{
                            std::transform(std::execution::unseq, v.begin(), v.end(), g, 
                                [&](float f){return f;}
                            );
                        } 
                    }

                    std::transform(std::execution::unseq, g, g + n, w, w, [&](auto lhs, auto rhs){return rhs - lr * lhs;});
                    
                    // Reset gradient (important!)
                    std::fill(std::execution::unseq, g, g + n, 0.0f);
                    
                    // unused !!!
                    /*
                    grad_sum = 0.0f;
                    for (size_t i = 0; i < n; ++i) {
                        grad_sum += std::abs(g[i]);
                    }
                        */
                };

                update_tensor(layer.weights, "weights");
                update_tensor(layer.bias, "bias");
            }
        }, layer_variant);
    });
}