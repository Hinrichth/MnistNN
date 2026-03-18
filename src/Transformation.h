#pragma once
#include "Tensor.h"
#include "Batch.h"
#include <vector>

struct Transformation {
    static Tensor ToTensor(const Batch& batch);
};
