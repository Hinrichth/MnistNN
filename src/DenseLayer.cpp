#include "DenseLayer.h"
#include "Tensor.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <execution>


DenseLayer::DenseLayer(size_t input_dimension,
                       size_t output_dimension, bool use_kaiming)
    : dim_in(input_dimension),
      dim_out(output_dimension),
      weights({dim_in, dim_out}),
      bias({dim_out})

{
    std::random_device rd;
    std::mt19937 urbg(rd());

    float const mu = 0.0f;
    float sigma;

    if (use_kaiming) {
        // Kaiming He initialization (ReLU)
        sigma = std::sqrt(2.0f / static_cast<float>(dim_in));
    } else {
        // Xavier initialization (sigmoid/tanh)
        sigma = std::sqrt(2.0f / static_cast<float>(dim_in + dim_out));
    }

    std::normal_distribution<float> norm(mu, sigma);

    std::generate_n(std::execution::unseq, weights.data_, weights.size(), [&](){return norm(urbg);});
    
    std::fill_n(std::execution::unseq, bias.data_, dim_out, 0.0f);
}

void DenseLayer::forward(const Tensor& input, Tensor& output)
{   

    input_cache = const_cast<Tensor*>(&input);

    size_t B = input.shape_[0]; // batch
    size_t In = weights.shape_[0];
    size_t Out = weights.shape_[1];

    // Dense Layer: out = W * input + b
    for (size_t b = 0; b < B; ++b) {
        for (size_t o = 0; o < Out; ++o) {
            
            output.data_[b*Out + o] = 
                std::inner_product(&input.data_[b*In], 
                                   &input.data_[b*In + In], 
                                   &weights.data_[o*Out],
                                   bias.data_[o]);
        }
    }
}

void DenseLayer::backward(const Tensor& grad_high, Tensor& grad_low) {
    
    // weights gradient
    // h = W x + b -> dL/dW = x * dL/dh

    // bias gradient
    // h = W x + b -> dL/db = 1 * dL/dh

    // compute gradients for next pass (for lower layers) 
    // h = W x + b -> dL/dx = W^T * dL/dh

    size_t B = grad_high.shape_[0];
    size_t In = weights.shape_[0];
    size_t Out = weights.shape_[1];
    
    
    float input_grad_sum = 
        std::accumulate(grad_high.grad_, grad_high.grad_ + B, 0.0f, 
            [](float& lhs, float& rhs){return lhs + std::abs(rhs);}
        );
    

    std::fill(std::execution::seq, grad_low.grad_, grad_low.grad_ + grad_low.size(), 0.0f);
    
    for (size_t b = 0; b < B; ++b) {
        for (size_t o = 0; o < Out; ++o) {
            float go = grad_high.grad_[b*Out + o];
            bias.grad_[o] += go;
            
            for (size_t i = 0; i < In; ++i) {
                weights.grad_[o*Out + i] += input_cache->data_[b*In + i] * go;
                grad_low.grad_[b*In + i] += weights.data_[o*Out + i] * go;
            }
        }
    }
    
    float weight_grad_sum = 
        std::accumulate(weights.grad_, weights.grad_ + weights.size_, 0.0f, 
            [](float& lhs, float& rhs){return lhs + std::abs(rhs);}
        );
    
    float bias_grad_sum = 
        std::accumulate(bias.grad_, bias.grad_ + bias.size_, 0.0f, 
            [](float& lhs, float& rhs){return lhs + std::abs(rhs);}
        );
    
}
