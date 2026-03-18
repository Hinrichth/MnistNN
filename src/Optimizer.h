#pragma once
#include "Sequential.h"
#include <unordered_map>
#include <memory>

struct SGD {
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
    bool nesterov;

    // velocity buffers
    std::unordered_map<Tensor*, std::vector<float>> velocity;

    SGD(float lr,
        float momentum = 0.0f,
        float dampening = 0.0f,
        float weight_decay = 0.0f,
        bool nesterov = false);

    void step(Sequential& model);
};