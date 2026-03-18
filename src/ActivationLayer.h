#pragma once
#include "Tensor.h"
#include "Layer.h"

/**
 * @brief Functional layer in neural network architecture.
 * 
 * ActivationLayer  
 *  - "sigmoid": @f$  f(x) = \frac{1}{1 + e^{-x}} @f$ 
 */
class ActivationLayer: public Layer {
public:
    /**
     * @brief Constructor.
     * 
     * @param kind The proper forward and backward functors will be deduced from `kind`.
     * @param layer The Tensor to apply activation layer propagations to.
     */
    ActivationLayer(const std::string& kind, Tensor* lhs);

    /**
     * `forward()` applies an activation function elementwise to `layer`.
     */
    void forward();

    /**
     * @brief Backpropagation of `layer.grad`.
     * 
     * @warning Errors may occur, if `backward()` is called, before `forward()` has successfully finished.
     */
    void backward();
    
    Tensor* l;

private:
    std::function<float(float)> fwd_functor; /**< Elementwise functor for forward propagation.*/
    std::function<float(float, float)> bwd_functor; /**< Elementwise binary oparation for backpropagation. The second parameter should refer to the previously computed values `layer.data`*/
};
