#pragma once
#include <vector>
#include <Eigen/Dense>
#include <numeric>  // for accumulate
#include <functional> // for multiplies
#include "View.h"

using m_float = Eigen::MatrixXf;
using v_float = Eigen::VectorXf;
/**
 * @class
 * @brief Data structure for neural network construction.
 * 
 * A Tensor contains two equal size Eigen::MatrixXf designated for evaluation data and its gradient respectively.
 * @note Since resizing Tensor is very costly, it is not allowed.
 */
class Tensor {
public:
    /**
     * Tensor members `data` and `grad` are initialized and resized to product of `shape`'s elements.
     * @param batch_size number of images processed in a batch.
     * @param data_size number of data points of each image.
     */
    Tensor(const size_t data_size, const size_t batch_size);

    m_float data; /**< Eigen::MatrixXf holding data of each element of a batch in a separate row.*/
    m_float grad; /**< Eigen::MatrixXf. Same size as `data`. Holds gradient for each element in `data`.*/
};