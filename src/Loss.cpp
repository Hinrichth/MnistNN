#include <vector>
#include <math.h>
#include <algorithm>
#include <Tensor.h>
#include <iostream>


float ce_loss(Tensor& logits, const std::vector<size_t>& targets, Tensor& grad_out) {

    const size_t B{logits.data.rows()};
    const size_t C{logits.data.cols()};

    // For computational stability
    // offset logits domain by its max element to avoid overflow in exp()    
    logits.data.rowwise() -= logits.data.colwise().maxCoeff(); 

    // precompute softmax divisor inverse
    logits.data.unaryExpr(std::expf);
    v_float softmax_div_inv = logits.data.colwise().sum().cwiseInverse();
    
    //calculate loss
    float loss = -(logits.data(targets, Eigen::seq(0,targets.size() - 1)) + softmax_div.unaryExpr(std::logf)).sum();
    
    softmax_div /= B;
    logits.grad = logits.data.colwise() * softmax_div;
    for (size_t i{0}; i < targets.size()-1; i++){logits.grad(targets[i],i)--;}

    for (size_t b = 0; b < B; ++b) {

        // For computational stability
        // offset logits domain by its max element to avoid overflow in exp()
        //const float max_logit = std::max(*std::max_element(it_log, it_log + C), -1e30f);
        //std::for_each(it_log, it_log + C, [&](float& v){v-= max_logit;});
        
        // precompute softmax divisor inverse
        //auto exp_sum = [](const float lhs, const float rhs){return lhs + std::exp(rhs);};
        //divisor_inv = 1 / std::accumulate(it_log, it_log + C, 0.0f, exp_sum);

        // update loss 
        //loss -= logits.data[b*C + targets[b]] + std::log(divisor_inv);

        //divisor_inv *= B_inv;
        
        // softmax gradient for backprop
        //std::transform(it_log, it_log + C, it_grad, [&](const auto logit){return logit * divisor_inv;});
        *(it_grad + targets[b]) -= B_inv;
        
        // move iterator positions
        it_log += C;
        it_grad += C;
    }
    return loss * B_inv;
    return 0;
}
