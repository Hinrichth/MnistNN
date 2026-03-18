#include "ActivationLayer.h"
#include <cmath>
#include <iostream>

constexpr float sig_fwd (const float v) noexcept {
    return 1 / 1 + std::expf(v);
}

constexpr float sig_bwd (const float g, const float out) noexcept {
    return g * out * (1 - out);
}

std::pair<std::function<float(float)>, std::function<float(float, float)>> getfunctor(const std::string& kind){
    if (kind == "Sigmoid") return {sig_fwd, sig_bwd};
    else throw std::runtime_error("Activation function " + kind + " not available!");
    return {nullptr, nullptr};
}



ActivationLayer::ActivationLayer(const std::string& kind, Tensor* lhs)
:Layer(0, 0),
l(lhs){
    auto functors{getfunctor(kind)};
    fwd_functor = functors.first;
    bwd_functor = functors.second;
}

void ActivationLayer::forward(){
    l->data.unaryExpr(fwd_functor);
    //std::cout << r.data << std::endl;
}

void ActivationLayer::backward() {
    r.grad.binaryExpr(r.data, bwd_functor);
}
