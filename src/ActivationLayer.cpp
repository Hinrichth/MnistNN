#include "ActivationLayer.h"
#include <cmath>
#include <algorithm>
#include <execution>


void ReLULayer::forward(const Tensor& input, Tensor& output){
    input_cache = const_cast<Tensor*>(&input);
    std::transform(std::execution::unseq, input.data_, input.data_ + input.size(), output.data_, [](const float lhs){
        return std::max(0.0f, lhs);
    });
}

void ReLULayer::backward(const Tensor& grad_high, Tensor& grad_low) {

    // Backpropagation through the relu activation:
    // dL/dx = dL/dy * 1(x > 0)
    std::transform(std::execution::unseq, grad_high.grad_, grad_high.grad_ + grad_high.size(), input_cache->data_, grad_low.grad_, [](const float grad, const float data){
        return grad * (data > 0);
    });
}

void SigmoidLayer::forward(const Tensor& input, Tensor& output){
    input_cache = const_cast<Tensor*>(&input);
    std::transform(std::execution::unseq, input.data_, input.data_ + input.size(), output.data_, [](const float lhs){
        return 1.0f / (1.0f + std::exp(- lhs));
    });
}

void SigmoidLayer::backward(const Tensor& grad_high, Tensor& grad_low) {

    // Backpropagation through the sigmoid activation:
    // dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))
    std::transform(std::execution::unseq, grad_high.grad_, grad_high.grad_ + grad_high.size(), input_cache->data_, grad_low.grad_, [](const float grad, const float data){
        float sig = 1.0f / (1.0f + std::exp(-grad));
        return grad * sig * (1 - sig);
    });
}
