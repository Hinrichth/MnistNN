#pragma once
#include <cstddef>
#include <random>
#include "Dataset.h"
#include "Batch.h"

struct Dataloader {
public:
    Dataloader(DatasetView view, 
               size_t batch_size,
               bool shuffle);

    Batch next(); // next iteration
    bool has_next() const;
    void reset(); // next epoch
    
private:
    DatasetView view_;
    size_t batch_size_;
    size_t cursor_;
    //bool shuffle_;

    //std::mt19937 rng_;  // shuffle indexing
};