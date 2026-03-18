#include "Sequential.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include "debug.h"

bool Sequential::request_batch(Dataset& set){
    
    auto req = set.request_imgs(batch_size, labels);
    if (!req.first) return false;
    layers.front()->r.data = 
        Eigen::Map<m_float>(req.first, req.second, batch_size);
    
    //std::cout << labels.transpose() << std::endl;
    
    return true;
}

void Sequential::forward() {
    for(auto& layer: layers) layer->forward();

    v_float res(batch_size);
    for (size_t i{0}; i < batch_size; i++){
        layers.back()->r.data.col(i).maxCoeff(&res(i));
    }
}

void Sequential::backward() {
    for(auto& layer: layers) layer->backward();
}

Tensor* Sequential::add_dense(const size_t dim_rhs, Tensor* lhs) {
    layers.emplace_back(new DenseLayer(lhs, dim_rhs, batch_size));
    return &((DenseLayer*)layers.back())->r;
}

Tensor* Sequential::add_activation(const std::string& kind, Tensor* lhs) {
    layers.emplace_back(new ActivationLayer(kind, lhs));
    return lhs;
}

Tensor* Sequential::add_input(const size_t dim_rhs) {
    layers.emplace_back(new InputLayer(dim_rhs, batch_size));
    return &((InputLayer*)layers.back())->r;
}

Sequential Sequential::from_config(const YAML::Node& cfg) {
    Sequential model;
    size_t dim_lhs, dim_rhs;

    model.batch_size = cfg["data"]["batch_size"].as<size_t>();
    if (!model.batch_size) throw std::domain_error("Batchsize cannot be 0!");
    model.labels.resize(model.batch_size);

    const auto& layer_defs = cfg["model"]["layers"];

    if (layer_defs[0]["type"].as<std::string>() != "Input"){ 
        throw std::logic_error("Model class needs an Input layer");
    }
    dim_rhs = layer_defs[0]["out"].as<size_t>();
    
    Tensor* lhs = model.add_input(dim_rhs);

    for (const auto& L : layer_defs) {
        std::string type = L["type"].as<std::string>();
        
        dim_lhs = L["in"].as<size_t>();
        dim_rhs = L["out"].as<size_t>();

        if (type == "Dense") {
            lhs = model.add_dense(dim_rhs, lhs);
        }
        else if (type == "Activation") {
            std::string kind = L["kind"].as<std::string>();
            lhs = model.add_activation(kind, lhs);
        }
        else if (type == "Input") continue;
        else {
            throw std::runtime_error("Unknown layer type: " + type);
        }
    }

    return model;
}
