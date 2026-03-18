#pragma once
#include "DenseLayer.h"
#include "ActivationLayer.h"
#include "Tensor.h"
#include <yaml-cpp/yaml.h>

#include <variant>
#include <vector>

using LayerVariant = std::variant<DenseLayer, SigmoidLayer, ReLULayer>;

struct Sequential {

    std::vector<LayerVariant> layers;

    // activation buffers (owned by Sequential)
    std::vector<Tensor> activations;
    std::vector<Tensor> gradients;

    size_t batch_size = 0;
    size_t input_dim = 0;

    static Sequential from_config(const YAML::Node& cfg);

    void add_dense(size_t in, size_t out, bool use_kaiming);
    void add_activation(const std::string& kind);

    void build(size_t B);

    Tensor& forward(const Tensor& input);
    void backward(const Tensor& grad_out);
};
