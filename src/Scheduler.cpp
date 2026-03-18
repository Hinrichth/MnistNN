#include "Scheduler.h"
#include <cmath>   // std::fabs
#include <limits>  // std::numeric_limits

void reduce_lr_on_plateau(float val_loss,
                       int epoch,
                       float& lr,
                       float factor,
                       int patience,
                       float min_lr,
                       float eps)
{
    static int counter = 0;
    static float best_loss = std::numeric_limits<float>::infinity();

    // First epoch: initialize best loss
    if (epoch == 0) {
        best_loss = val_loss;
        counter = 0;
        return;
    }

    // Improvement?
    if (val_loss < best_loss - eps) {
        best_loss = val_loss;
        counter = 0;
        return;
    }

    // No improvement
    counter++;

    // Not enough patience yet
    if (counter < patience)
        return;

    // Time to reduce LR
    float new_lr = lr * factor;

    if (new_lr < min_lr + eps) {
        std::cout << "[LR Scheduler] LR NOT reduced: "
                  << "new LR " << new_lr << " < minimum " << min_lr << "\n";
    }
    else if (std::fabs(lr - new_lr) < eps) {
        std::cout << "[LR Scheduler] LR NOT reduced: "
                  << "change smaller than eps=" << eps << "\n";
    }
    else {
        std::cout << "[LR Scheduler] LR reduced from "
                  << lr << " → " << new_lr << "\n";
        lr = new_lr;
    }

    counter = 0;  // reset counter after plateau action
}
