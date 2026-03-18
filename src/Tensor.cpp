// Tensor.cpp
#include "Tensor.h"
#include <cstring>
#include <stdexcept>

// Allocating constructor - owns both data and gradient
Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    size_ = 1;
    for (auto s : shape_) size_ *= s;
    
    // Allocate and initialize memory
    data_owner_ = std::make_unique<float[]>(size_);
    grad_owner_ = std::make_unique<float[]>(size_);
    
    data_ = data_owner_.get();
    grad_ = grad_owner_.get();
    
    // Zero initialize
    std::fill(data_, data_ + size_, 0.0f);
    std::fill(grad_, grad_ + size_, 0.0f);
}

// Non-owning constructor - uses external memory
Tensor::Tensor(float* data, float* grad, const std::vector<size_t>& shape) 
    : shape_(shape), data_(data), grad_(grad) 
{
    size_ = 1;
    for (auto s : shape_) size_ *= s;
    // unique_ptrs remain null - memory is externally managed
}