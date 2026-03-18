## Name
Neural Network for Deep Learning in C++
A simple feedforward neural network implemented in C++ for handwritten digit classification using the MNIST dataset.

## Description
This project implements a basic deep learning pipeline in C++ from scratch. It supports loading datasets, converting images to tensors, building a Sequential neural network with Dense and Activation layers (linear and sigmoid), performing forward and backward passes, and training with mini-batch stochastic gradient descent.

# Features
+ Fully implemented Tensor class with gradient support.

+ DenseLayer and ActivationLayer for building neural networks.

+ Xavier and Kaiming (He) weight initialization depending on activation function

+ Sequential container for stacking layers.

+ Batch class for handling mini-batches.

+ Normalization and transformation functions for images.

+ Split of dataset into training and validation sets.

+ Mini-batch stochastic gradient descent (SGD) optimizer.

+ Scheduler and early stopping for training control.

+ Configuration via a YAML file (config.yaml) for flexible experiment setup.

## References
YAML parsing: https://github.com/jbeder/yaml-cpp.git

## Installation
1. Ensure you have C++17 or newer.
2. In your terminal:
cd project-path-in-your-local
mkdir build
cd build
cmake ..
cmake --build .
```

## Usage
This project is fully configured through a YAML file that defines the entire deep-learning pipeline:
dataset loading, preprocessing, model architecture, optimizer, scheduler, and early stopping.

To use the network, simply edit a configuration file (e.g., config.yaml) and run the executable.
No command-line flags are required — the YAML file controls everything.
### Dataset Configuration
Specify dataset paths and preprocessing settings in your YAML configuration:
```YAML
    data:
      dataset_dir: "../mnist"
      train_images: "emnist-digits-train-images-idx3-ubyte"
      train_labels: "emnist-digits-train-labels-idx1-ubyte"
      test_images: "emnist-digits-test-images-idx3-ubyte"
      test_labels: "emnist-digits-test-labels-idx1-ubyte"
    
      image_size:
        height: 28
        width: 28
    
      val_ratio: 0.00033
      batch_size: 64
      shuffle: true
```

The system automatically:
+ loads EMNIST/MNIST images
+ splits data into training and validation sets
+ converts images to tensors
+ normalizes pixel values
+ prepares shuffled mini-batches

### Model Architecture
Define your model in the YAML file using Sequential layer definitions:
```YAML
    model:
    layers:
        - type: Dense
          in: 784
          out: 300
          
        - type: Activation
          kind: Sigmoid # (ReLU or Sigmoid)
    
        - type: Dense
          in: 300
          out: 10
```

The project currently supports:

+ Dense layers
+ Sigmoid activation
+ ReLU Activation

Future extensions may include more activation functions and GPU support.

### Training Configuration
The training loop, optimizer, learning rate scheduling, and early stopping are configured in YAML:
```YAML
    train:
    epochs: 2
    
    optimizer:
        learning_rate: 0.01
        momentum: 0
        dampening: 0
        weight_decay: 0
        nesterov: false
    
    scheduler:
        patience: 10
        reduction_factor: 0.1
        minimal_lr: 0
        minimal_lr_change: 1e-08
    
    early_stopping:
        patience: 20
```
Supported training features:

+ Mini-batch SGD
+ Momentum
+ Early stopping

### Running the Program
Run the project using the main application in [`main.cpp`](./main.cpp).
If you followed installation step this can be done by: 
cd build
./TestNN

This will:
+ Load the dataset
+ Build the neural network defined in the YAML (Sequential with Dense and Activation layers). .
+ Train the network using the train() function with mini-batch SGD.
+ Apply early stopping
+ After training, evaluate the final model on the test dataset using the evaluate() function.
+ Evaluate accuracy on the validation and test set

## Support
Please go to "Issues" and create one if you have problems

## Future Work
+ Implement additional activation functions (Tanh, etc.).
+ Enable GPU training for acceleration.
+ Expand optimizer options (Adam, RMSProp, etc.).


## Authors and acknowledgment
Developer: Nikita Karpuks, Hinrich Thiele and Jan Wech

This project was implemented as part of the Advanced Programming course at TUM. The original assignment and project description can be found here https://gitlab.lrz.de/tum-i05/public/advprog-project-ideas/-/blob/master/deep-learning/deep-learning.md?ref_type=heads.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project status
Development is ongoing. All core functionalities (Tensor operations, Dense/Activation layers, Sequential model, DataLoader, training with SGD, scheduler, and early stopping) are implemented.
Future improvements may include:
+ Additional activation functions
+ GPU support for faster training
+ Enhanced optimizer implementations
