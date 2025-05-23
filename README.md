# Neural Networks Simulation

## Overview
This project demonstrates the implementation of a deep neural network from scratch using **NumPy**, without relying on high-level deep learning frameworks. The model is trained to classify handwritten digits from the **MNIST dataset**.

## Features
- Fully connected **3-layer neural network**
- **Feedforward and backpropagation** implementations
- Optimizers: **SGD, Momentum**
- **Xavier Initialization** for better weight scaling
- **Softmax activation** for classification tasks
- Command-line support for hyperparameter tuning

## Architecture
- **Input Layer:** 784 nodes (Flattened 28×28 images)
- **Hidden Layer:** 64 nodes with ReLU activation
- **Output Layer:** 10 nodes with Softmax activation

## Dataset
The **MNIST dataset** is used for training and testing:
- 60,000 training images
- 10,000 test images
- Grayscale images (pixel values: 0-255, normalized to 0-1)//
  




