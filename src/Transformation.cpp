#include "Transformation.h"
#include "Tensor.h"
#include <cstring>

Tensor Transformation::ToTensor(const Batch& batch) {
    // Create tensor with allocated memory
    Tensor result({batch.batch_size, batch.pixels});
    
    // Copy batch images into tensor data
    std::memcpy(result.data_, batch.images, 
                batch.batch_size * batch.pixels * sizeof(float));
    
    // Gradient is already allocated and zeroed by Tensor constructor
    
    return result;
}