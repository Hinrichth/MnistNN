#pragma once
#include "Sequential.h"
#include "Dataloader.h"
#include "Batch.h"
#include "Tensor.h"
#include "Loss.h"

struct EvalResult {
    float loss;
    float acc;
    size_t total;
};

EvalResult evaluate(Sequential& model, Dataloader& loader);
