#pragma once
#include "Tensor.h"


struct ReLULayer {
    Tensor* input_cache = nullptr;
    size_t out_dim(size_t in_dim) const { return in_dim; }

    void forward(const Tensor& input, Tensor& output);
    void backward(const Tensor& grad_high, Tensor& grad_low);
};

struct SigmoidLayer {
    Tensor* input_cache = nullptr;
    size_t out_dim(size_t in_dim) const { return in_dim; }
    
    void forward(const Tensor& input, Tensor& output);
    void backward(const Tensor& grad_high, Tensor& grad_low);
};
