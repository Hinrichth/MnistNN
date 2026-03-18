#pragma once
#include "View.h"
#include "Tensor.h"
#include "Dataset.h"
#include <iostream>
#include "Layer.h"

/**
 * @class 
 * @brief Functional layer in neural network architecture.
 * 
 * InputLayer holds the data for a neural network forward pass and receives result from the backward pass.
 */
 struct InputLayer: public Layer{

    /**
     * @brief Constructor!
     * 
     * @param batch_size Number of images in a batch.
     * @param data_size Size of image.
     */
    InputLayer(const size_t data_size, const size_t batch_size);

    /**
     * @brief Load evaluation data into InputLayer.
     * 
     * New data i sbeing requested from Dataset.
     * The data is copied from its initial location into `r.data`.
     */
    void setData(Dataset& set, size_t batch_size, v_ui8& labels){
        auto request = set.request_imgs(batch_size, labels);
        r.data = Eigen::Map<m_float>(request.first, batch_size, request.second);
    };

    /**
     * @brief No transformation.
     */
    void forward(){};
    /**
     * @brief No transformation.
     */
    void backward(){};
};