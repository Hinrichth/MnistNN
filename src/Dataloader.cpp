//
// Created by jan on 11/28/25.
//

#include "Dataloader.h"
#include <algorithm>
#include "Transformation.h"
#include "Tensor.h"


Dataloader::Dataloader(DatasetView view,
                       size_t batch_size,
                       bool shuffle)
    : view_(view),
      batch_size_(batch_size),
      cursor_(0)//,
      //shuffle_(shuffle),
      //rng_(std::random_device{}())
{
    // Already done in Dataset::Dataset()
    /*
    if (shuffle_) {
        // shuffle data in-place
        Dataset& ds = *view_.dataset;
        size_t N = ds.size();

        for (size_t i = N - 1; i > 0; --i) {
            size_t j = rng_() % (i + 1);

            // swap images
            float* a = const_cast<float*>(ds.image_ptr(i));
            float* b = const_cast<float*>(ds.image_ptr(j));
            for (size_t p = 0; p < ds.pixels(); ++p)
                std::swap(a[p], b[p]);

            std::swap(ds.label(i), ds.label(j));
        }
    }
    */
}

bool Dataloader::has_next() const {
    // always drop last
    return (view_.size() - cursor_) >= batch_size_;
}

void Dataloader::reset() {
    cursor_ = 0;
}

Batch Dataloader::next() {
    size_t remaining = view_.size() - cursor_;
    
    // If remaining is less than batch_size_, skip this batch
    if (remaining < batch_size_) {
        // Return empty batch
        return Batch{nullptr, nullptr, 0, 0};
    }
    
    size_t bs = batch_size_;  // Always use full batch size
    size_t start = view_.begin + cursor_;
    cursor_ += bs;

    const std::vector<uint8_t>& all_labels = view_.dataset->labels();
    
    return Batch{
        view_.dataset->image_ptr(start),
        all_labels.data() + start,
        bs,
        view_.dataset->pixels()
    };
}
