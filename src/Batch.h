//
// Created by jan on 11/28/25.
//
#pragma once
#include <cstddef>
#include <cstdint>

struct Batch {
    float* images;   // pointer into Dataset
    const uint8_t* labels;
    size_t batch_size;
    size_t pixels;
};