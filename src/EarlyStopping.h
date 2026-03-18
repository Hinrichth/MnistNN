#pragma once

// Returns true if we should stop training.
bool early_stopping(const float& val_loss, const int& epoch, const int& patience=20);
