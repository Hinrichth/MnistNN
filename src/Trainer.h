#pragma once
#include "Sequential.h"
#include "Dataloader.h"
#include "Optimizer.h"
#include "Evaluation.h"
#include "EarlyStopping.h"
#include "Scheduler.h"
#include "Transformation.h"
#include "Tensor.h"
#include "Loss.h"
#include <yaml-cpp/yaml.h>

void train_model(const YAML::Node& cfg,
                 Sequential& model,
                 Dataloader& train_loader,
                 Dataloader& val_loader,
                 SGD& optimizer);
