#include <vector>
#include <iostream>
#include<EarlyStopping.h>


bool early_stopping(const float& val_loss, const int& epoch, const int& patience) {
    static int counter = 0;
    static float best = 1e30f;

    if (val_loss < best) {
        best = val_loss;
        counter = 0;
        return false;
    }
    counter++;

    bool should_stop = counter >= patience;
    if (should_stop) {
        std::cout << "Learning is stopped as we had " << patience << " epochs without an improvement" << "\n";
    }

    return counter >= patience;
}