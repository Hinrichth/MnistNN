#pragma once
#include <iostream>

// Reduces LR when validation loss plateaus.
// val_loss: validation loss of this epoch
// epoch: current epoch index (0-based)
// lr: learning rate (modified in-place)
void reduce_lr_on_plateau(float val_loss,
                       int epoch,
                       float& lr,
                       float factor = 0.1f,
                       int patience = 10,
                       float min_lr = 0.0f,
                       float eps = 1e-8f);
