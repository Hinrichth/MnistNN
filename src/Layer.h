#pragma once
#include "Tensor.h"

struct Layer{
    Layer(const size_t data_size, const size_t batch_size):r(data_size, batch_size){}
    Tensor r;
    virtual void forward() = 0;
    virtual void backward() = 0;
};