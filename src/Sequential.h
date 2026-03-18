#pragma once
#include "DenseLayer.h"
#include "ActivationLayer.h"
#include "InputLayer.h"
#include "Tensor.h"
#include <yaml-cpp/yaml.h>

#include <variant>
#include <vector>
#include "Dataset.h"

/**
 * @class
 * @brief Base network class. 
 * 
 * Sequential will initialize and manage all necessary recources for training and / or evaluating neural network data.
 * The network structure may be specified by code or through a yaml config file.
 */
struct Sequential {
    std::vector<Layer*> layers;
    v_ui8 labels;
    size_t batch_size;
    Sequential() = default;
    ~Sequential(){for (auto layer: layers) delete layer;}

    /**
     * 
     * 
     */
    bool request_batch(Dataset& set);

    /**
     * @brief Add the InputLayer.
     * 
     * Manages communication with a dataset to load input data.
     * Collects the results of backward passes.
     * @param n Size of input data.
     */
    Tensor* add_input(const size_t n);

    /**
     * @brief Add a dense functional layer.
     * 
     * A new DenseLayer is initialized and added to the back of the network.
     * Requires 2 hidden layers with respective sizes at the end of `hlayers`.
     * @param lhs Left hand side Tensor.
     * @param dim_rhs Size of new right hand side Tensor.
     */
    Tensor* add_dense(const size_t dim_rhs, Tensor* lhs);

    /**
     * @brief Add an activation layer.
     * 
     * A new ActivationLayer is initialized and added to the back of the network.
     * Requires 2 hidden layers with same size at the end of `hlayers`. 
     * @param kind type of activation function.
     * @param lhs Left hand side Tensor.
     */
    Tensor* add_activation(const std::string& kind, Tensor* lhs);

    /**
     * Coordinates forward pass of data in View through entire network. 
     * @see DenseLayer.forward()
     * @see ActivationLayer.forward()
     */
    void forward();

    /**
     * Coordinates backward pass of gradient stored inside the model writes results to `out`. 
     * @see DenseLayer.backward()
     * @see ActivationLayer.backward()
     */
    void backward();

    /**
     * @brief Initialization of Seqential from yaml config file.
     * 
     * @param cfg yaml file object holding all config details.
     */
    static Sequential from_config(const YAML::Node& cfg);
};
