# NMIST Neural Network

![Static Badge](https://img.shields.io/badge/made_in-France-red?labelColor=blue)
![Static Badge](https://img.shields.io/badge/language-Python-f7d54d?labelColor=4771a4)

This repository contains a severals neural network that can recognize handwritten digits from the MNIST dataset. Neural networks are either made from scratch using only the **CuPy** library, either build with the frameworks **PyTorch**.

> **Note**: The MNIST dataset is a dataset of 60,000 small square 28x28 pixel grayscale images of handwritten single digits between 0 and 9. The dataset also includes a test set of 10,000 images.<br><br>
> CuPy is an open-source array library accelerated with NVIDIA CUDA. It allows you to perform operations on a GPU. It is a drop-in replacement for NumPy.<br><br>
> TensorFlow was not used in this project because I couldn't find a way to make it work with my GPU.

## Summary

### 1. [Linear Classifier](#1---linear-classifier)

### 2. [Deep Neural Network](#2---deep-neural-network)

### 3. [Convolutional Neural Network](#3---convolutional-neural-network)

### 4. [AlexNet](4---alexnet)

### 5. [Performance Comparison](#5---performance-comparison)

## 1 - Linear Classifier

For this first neural network, we will directly plug the input layer to the output layer. This means that we will not have any hidden layers. The input layer of the network will have 784 neurons, each one representing the grayscale value of a pixel of the 28x28 image. Then, all the neurons of the input layer will be connected to all the neurons of the output layer. Finally, the output layer will have 10 neurons, each one representing a digit from 0 to 9.

The activation function used in this network is the ReLU function. The loss function is the mean squared error. The network will be trained using the gradient descent algorithm.

Here is the architecture of the network:

<p align="center">
    <img src="assets/svg/linear_classifier_architecture.svg" alt="Linear classifier architecture" style="width: 50%;"/>
</p>

> **Note**: Input layer: 784 neurons. Output layer: 10 neurons.

And here's what the loss and precision curves roughly look like for both versions of the network:

<br>
<p align="center">
    <img src="assets/images/loss_vs_epoch_linear_classifier.png" alt="Loss vs Epoch Linear Classifier" style="border-radius: 10px;"/>
</p>

## 2 - Deep Neural Network

<p align="center">
<!-- <img src="assets/svg/deep_neural_network_architecture.svg" alt="Single neuronne architecture" style="width:80%"/> -->
</p>

> **Note**: Input layer: 784 neurons, First layer: 64 neurons, Second layer: 32 neurons, Third layer: 16 neurons, Fourth layer: 8 neurons, Output layer: 10 neurons.

## 5 - Performance Comparison

### Here are tables comparing the performance of the different frameworks depending on the neural network used:

- Linear Classifier.

| Framework | Accuracy | Training Time | Epochs | Learning Rate | Device |
| :-------: | :------: | :-----------: | :----: | :-----------: | :----: |
|  Vanilla  |   ~88%   |     ~7.5s     |  100   |       1       |  GPU   |
|  Pytorch  |   ~88%   |    ~0.38s     |  100   |       1       |  GPU   |

- Deep Neural Network.

| Framework | Accuracy | Training Time | Epochs | Learning Rate | Device |
| :-------: | :------: | :-----------: | :----: | :-----------: | :----: |
|  Vanilla  |          |               |  100   |               |  GPU   |
|  Pytorch  |          |               |  100   |               |  GPU   |

- Convolutional Neural Network.

| Framework | Accuracy | Training Time | Epochs | Learning Rate | Device | 
| :-------: | :------: | :-----------: | :----: | :-----------: | :----: | 
|  Vanilla  |          |               |  100   |               |  GPU   | 
|  Pytorch  |          |               |  100   |               |  GPU   | 

- AlexNet.

| Framework | Accuracy | Training Time | Epochs | Learning Rate | Device |
| :-------: | :------: | :-----------: | :----: | :-----------: | :----: |
|  Vanilla  |          |               |  100   |               |  GPU   |
|  Pytorch  |          |               |  100   |               |  GPU   |

### Here are tables comparing the performance of the different neural networks depending on the framework used:

- Neural Network built from scratch using only **CuPy**.

|     Framework     | Accuracy | Training Time | Epochs | Learning Rate | Device |
| :---------------: | :------: | :-----------: | :----: | :-----------: | :----: |
| Linear Classifier |   ~88%   |     ~7.5s     |  100   |       1       |  GPU   |
|        DNN        |          |               |  100   |     0.01      |  GPU   |
|        CNN        |          |               |  100   |     0.01      |  GPU   |
|      AlexNet      |          |               |  100   |     0.01      |  GPU   |

- Neural Network built with **PyTorch**.

|  Neural Network   | Accuracy | Training Time | Epochs | Learning Rate | Device |
| :---------------: | :------: | :-----------: | :----: | :-----------: | :----: |
| Linear Classifier |   ~88%   |    ~0.38s     |  100   |       1       |  GPU   |
|        DNN        |          |               |  100   |     0.01      |  GPU   |
|        CNN        |          |               |  100   |     0.01      |  GPU   |
|      AlexNet      |          |               |  100   |     0.01      |  GPU   |

> **Note**: These values can change depending on the version of Python and your PC<br><br>
> For these benchmarks, I used Python 3.12.4 64-bit implemented with CPython on a Ryzen 5 3600, rtx 2060 with 2\*8GB of RAM clocked at 3600Hz on Windows 10.
