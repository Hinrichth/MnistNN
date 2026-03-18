#pragma once
#include "Sequential.h"

void sgd(Sequential& model, 
         float lr = 0.001,
         float momentum = 0,
         float dampening = 0,
         float weight_decay = 0,
         bool nesterov = false
        );
