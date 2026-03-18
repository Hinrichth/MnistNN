#include "DenseLayer.h"
#include "Tensor.h"
#include <random>
#include <algorithm>
#include <iostream>


DenseLayer::DenseLayer(Tensor* lhs, const size_t out_size, const size_t batch_size)
:l(lhs),
Layer(out_size, batch_size),
weights(out_size, lhs->data.rows()),
bias(out_size, batch_size)
{
    // Xavier initialization
    // Generate Normal distribution with mu = 0 and sigma = sqrt(2/(n_inputs + n_outputs))
    std::random_device rd;
    std::mt19937 urbg(rd());
    float const mu = 0.0f;
    float const sigma = sqrtf(2.0f / weights.size());
    auto norm = std::normal_distribution<float>{mu,sigma};
    
    // intialise the weights and bias
    auto rnd = [&]() -> float {return norm(urbg);};
    weights.NullaryExpr(out_size, lhs->data.rows(), rnd);
    bias.NullaryExpr(out_size, batch_size, rnd);
}

void DenseLayer::forward()
{   
    // Dense Layer: out = W * input + b
    r.data = weights * l->data + bias;
    //std::cout << r.data << std::endl;
}

void DenseLayer::backward() {

    // weights gradient
    // h = W x + b -> dL/dW = x * dL/dh
    weights += r.grad * l->data.transpose();
    
    // apply bias gradient
    // h = W x + b -> dL/db = 1 * dL/dh
    bias += r.grad;

    // compute gradients for next pass (for left layers) 
    // h = W x + b -> dL/dx = W^T * dL/dh
    l->grad = weights.transpose() * r.grad;
}
