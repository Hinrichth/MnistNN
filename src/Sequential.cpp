#include "Sequential.h"
#include <stdexcept>

void Sequential::build(size_t B) {
    batch_size = B;

    activations.clear();
    gradients.clear();

    size_t cur_dim = input_dim;

    for (auto& L : layers) {
        size_t next_dim = std::visit(
            [&](auto& layer) {
                return layer.out_dim(cur_dim);
            }, L);

        activations.emplace_back(
            std::vector<size_t>{B, next_dim});
        gradients.emplace_back(
            std::vector<size_t>{B, cur_dim});

        cur_dim = next_dim;
    }
}

Tensor& Sequential::forward(const Tensor& input) {
    const Tensor* x = &input;

    for (size_t i = 0; i < layers.size(); ++i) {
        std::visit([&](auto& layer) {
            layer.forward(*x, activations[i]);
        }, layers[i]);

        x = &activations[i];
    }

    return activations.back();
}

void Sequential::backward(const Tensor& grad_out) {
    
    // unused

    /*
    float data_sum = std::accumulate(grad_out.data_, grad_out.data_ + grad_out.size(), 0, [](const float lhs, const float rhs){
        return lhs + std::abs(rhs);
    });

    float grad_sum = std::accumulate(grad_out.grad_, grad_out.grad_ + grad_out.size(), 0, [](const float lhs, const float rhs){
        return lhs + std::abs(rhs);
    });
    
    float sum = 0.0f;
    */

    const Tensor* g = &grad_out;

    for (int i = (int)layers.size() - 1; i >= 0; --i) {
        std::visit([&](auto& layer) {
            layer.backward(*g, gradients[i]);
        }, layers[i]);

        g = &gradients[i];
        
        // not used
        /*
        sum = std::accumulate(g->data_, g->data_ + g->size(), 0, [](const float lhs, const float rhs){
            return lhs + std::abs(rhs);
        });
        */

        
    }
}

void Sequential::add_dense(size_t in, size_t out, bool use_kaiming) {
    layers.emplace_back(DenseLayer(in, out, use_kaiming));
}

void Sequential::add_activation(const std::string& kind) {
    if (kind == "Sigmoid")
        layers.emplace_back(SigmoidLayer());
    else if (kind == "ReLU")
        layers.emplace_back(ReLULayer());
    else
        throw std::runtime_error("Unknown activation: " + kind);
}


Sequential Sequential::from_config(const YAML::Node& cfg) {
    size_t h = cfg["data"]["image_size"]["height"].as<size_t>();
    size_t w = cfg["data"]["image_size"]["width"].as<size_t>();
    size_t input_dim = h * w;

    Sequential model;
    model.input_dim = input_dim;

    const auto& layer_defs = cfg["model"]["layers"];
    for (size_t i = 0; i < layer_defs.size(); ++i) {
        std::string type = layer_defs[i]["type"].as<std::string>();

        if (type == "Dense") {
            size_t in  = layer_defs[i]["in"].as<size_t>();
            size_t out = layer_defs[i]["out"].as<size_t>();

            // Lookahead: check if the next layer is ReLU
            bool use_kaiming = false;
            if (i + 1 < layer_defs.size() && layer_defs[i+1]["type"].as<std::string>() == "Activation") {
                std::string act = layer_defs[i+1]["kind"].as<std::string>();
                if (act == "ReLU") use_kaiming = true; // if next Layer is ReLU, use Kaiming
            }
            model.add_dense(in, out, use_kaiming);
        }
        else if (type == "Activation") {
            std::string kind = layer_defs[i]["kind"].as<std::string>();
            model.add_activation(kind);
        }
        else {
            throw std::runtime_error("Unknown layer type: " + type);
        }
    }

    return model;
}
