#pragma once
#include "View.h"
#include "Tensor.h"
#include "Layer.h"

/**
 * @class 
 * @brief Functional layer in neural network architecture.
 * 
 * DenseLayer manages recources necessary for a transition between 2 layers.
 * Layers may have arbitrary sizes.
 * The transition involves a matrix multiplication and addition of a bias. 
 */
 struct DenseLayer: public Layer {
    
    m_float weights; /**< Eigen::MatrixXf holding weights. */
    m_float bias;    /**< Eigen::MatrixXf holding bias. */
    Tensor* l; /**< Data and gradient of left adjacent layer.*/
    
    /**
     * @brief Constructor!
     * 
     * Xavier initialisation is used to populate `weights` and `bias`.
     * 
     * @param batch_size Number of images in a batch.
     * @param rhs_size Size of right adjacent layer.
     * @param lhs Left adjacent layer.
     * */
    DenseLayer(Tensor* lhs, const size_t rhs_size, const size_t batch_size);

    /**
     * @brief Transition of `l.data` to `r.data`.
     * 
     * `forward()` effectively computes the matrix vector product @f$  l \cdot weights + bias @f$ and stores results in `r`.
     */
    void forward();
    /**
     * @brief Backpropagation of `r.grad`.
     * 
     * `backward()` computes gradients for `weights`, `bias` and `l` and applies them to `weights` and `bias`.
     */
    void backward();
};