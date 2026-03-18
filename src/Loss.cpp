#include <vector>
#include <cmath>
#include <algorithm>
#include "Tensor.h"
#include <cstdint>
#include <stdexcept>


float ce_loss(const Tensor& logits, 
              const uint8_t* targets,
              Tensor& grad_out) 
{
    auto shape = logits.shape();
    size_t B = shape[0];
    size_t C = shape[1];
    
    if (grad_out.shape() != shape) {
        grad_out = Tensor({B, C});
    } else {
        // Zero out gradient
        std::fill(grad_out.grad_, grad_out.grad_ + grad_out.size(), 0.0f);
    }
    
    float loss = 0.0f;

    for (size_t b = 0; b < B; ++b) {
        float max_logit = -1e30f;
        for (size_t c = 0; c < C; ++c) {
            max_logit = std::max(max_logit, logits(b, c));
        }

        float divisor = 0.0f;
        for (size_t c = 0; c < C; ++c) {
            divisor += std::exp(logits(b, c) - max_logit);
        }

        size_t y = static_cast<size_t>(targets[b]);
        float log_prob = (logits(b, y) - max_logit) - std::log(divisor);
        loss += -log_prob;
        
        for (size_t c = 0; c < C; ++c) {
            float soft = std::exp(logits(b, c) - max_logit) / divisor;
            grad_out.grad(b, c) = (soft - (c == y ? 1.0f : 0.0f)) / static_cast<float>(B);
        }
    }

    return loss / static_cast<float>(B);
}