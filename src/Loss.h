#pragma once
#include "Loss.h"
#include "Tensor.h"
#include <vector>

// Computes mean cross-entropy over batch.
// logits: (B, C)
// targets: vector of size B, integers in [0..C-1]
// Returns: scalar loss
// grad: gradient w.r.t logits (B, C)
float ce_loss(Tensor& logits,
              const std::vector<size_t>& labels,
              Tensor& grad);
