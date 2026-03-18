// Tensor.h
#pragma once
#include <vector>
#include <numeric>
#include <functional>
#include <memory>

struct Tensor {
private:
    std::unique_ptr<float[]> data_owner_;  // Owns data memory
    std::unique_ptr<float[]> grad_owner_;  // Owns gradient memory
    
public:
    float* data_ = nullptr;
    float* grad_ = nullptr;
    
    std::vector<size_t> shape_;
    size_t size_ = 0;

    // Constructors
    Tensor() = default;
    
    // Constructor that allocates both data and gradient
    Tensor(const std::vector<size_t>& shape);
    
    // Constructor that uses external memory (doesn't own it)
    Tensor(float* data, float* grad, const std::vector<size_t>& shape);
    
    // Copy operations disabled (tensors are move-only)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Move operations
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    
    // Destructor (automatically handled by unique_ptr)
    ~Tensor() = default;
    
    // Accessors
    size_t size() const { return size_; }
    std::vector<size_t> shape() const { return shape_; }
    
    // 1D access
    float& operator()(size_t idx) { return data_[idx]; }
    float operator()(size_t idx) const { return data_[idx]; }
    
    // 2D access
    float& operator()(size_t i, size_t j) { 
        return data_[i * shape_[1] + j]; 
    }
    float operator()(size_t i, size_t j) const { 
        return data_[i * shape_[1] + j]; 
    }
    
    // Gradient access helpers
    float& grad(size_t idx) { return grad_[idx]; }
    float grad(size_t idx) const { return grad_[idx]; }
    
    float& grad(size_t i, size_t j) { 
        return grad_[i * shape_[1] + j]; 
    }
    float grad(size_t i, size_t j) const { 
        return grad_[i * shape_[1] + j]; 
    }
};