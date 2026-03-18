#include <vector>
#include <numeric>
#include <functional>
#include "Tensor.h"


Tensor::Tensor(const size_t data_size, const size_t batch_size)
:data(data_size, batch_size),
 grad(data_size, batch_size){}

