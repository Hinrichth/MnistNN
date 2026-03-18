#pragma once
#include "Tensor.h"
#include <iostream>

struct DenseLayer {
    
    size_t dim_in;
    size_t dim_out;
    Tensor weights;
    Tensor bias;

    Tensor* input_cache = nullptr; //  for backprop

    DenseLayer(size_t input_dimension, size_t output_dimension, bool use_kaiming);

    size_t out_dim(size_t) const { return dim_out; }
    
    void forward(const Tensor& input, Tensor& output);
    void backward(const Tensor& grad_high, Tensor& grad_low);
};