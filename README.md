# NMIST Neural Network

![Static Badge](https://img.shields.io/badge/made_in-France-red?labelColor=blue)
![Static Badge](https://img.shields.io/badge/language-Python-f7d54d?labelColor=4771a4)

This repository contains a severals neural network that can recognize handwritten digits from the MNIST dataset. Neural networks are either made from scratch using only the **CuPy** library, either build with frameworks like **PyTorch** or **TensorFlow**.

> **Note**: The MNIST dataset is a dataset of 60,000 small square 28x28 pixel grayscale images of handwritten single digits between 0 and 9. The dataset also includes a test set of 10,000 images.<br><br>
> CuPy is an open-source array library accelerated with NVIDIA CUDA. It allows you to perform operations on a GPU. It is a drop-in replacement for NumPy.

## Summary

### 1. [Simple Perceptron](#1---simple-perceptron)

### 2. [Deep Neural Network](#2---multi---layer-neural-network)

### 3. [Convolutional Neural Network](#3---convolutional-neural-network)

### 4. [AlexNet](4---alexnet)

### 5. [Performance Comparison](#5---performance-comparison)

## 1 - Simple Perceptron

For this first neural network, we will use a single neuron to classify the MNIST dataset. The input layer of the network will have 784 neurons, each one representing the grayscale value of a pixel of the 28x28 image. Then, all the neurons of the input layer will be connected to a single neuron in the hidden layer. Finally, the output layer will have 10 neurons, each one representing a digit from 0 to 9.

The activation function used in this network is the ReLU function. The loss function is the mean squared error. The network will be trained using the gradient descent algorithm.

Here is the architecture of the network:

<p align="center">
<img src="assets/svg/single_neuron_architecture.svg" alt="Single neuronne architecture" style="width:80%"/>
</p>

And here's what the loss and precision curves roughly look like for the 3 versions of the network:

<br>
<p align="center">
<img src="assets/images/loss_vs_epoch_single_neuron_scratch.png"/>
</p>

## 5 - Performance Comparison

### Here are tables comparing the performance of the different frameworks depending on the neural network used:

- Simple Perceptron.

| Framework  | Accuracy | Training Time | Epochs | Learning Rate | Device |
| ---------- | -------- | ------------- | ------ | ------------- | ------ |
| Vanilla    | ~90%     | ~10s          | 100    | 0.01          | GPU    |
| Pytorch    | ~90%     | ~0.35s        | 100    | 0.01          | GPU    |
| TensorFlow |          |               | 100    | 0.01          | CPU    |

- Deep Neural Network.

| Framework  | Accuracy | Training Time | Epochs | Learning Rate | Device |
| ---------- | -------- | ------------- | ------ | ------------- | ------ |
| Vanilla    |          |               | 100    | 0.01          | GPU    |
| Pytorch    |          |               | 100    | 0.01          | GPU    |
| TensorFlow |          |               | 100    | 0.01          | CPU    |

- Convolutional Neural Network.

| Framework  | Accuracy | Training Time | Epochs | Learning Rate | Device |
| ---------- | -------- | ------------- | ------ | ------------- | ------ |
| Vanilla    |          |               | 100    | 0.01          | GPU    |
| Pytorch    |          |               | 100    | 0.01          | GPU    |
| TensorFlow |          |               | 100    | 0.01          | CPU    |

- AlexNet.

| Framework  | Accuracy | Training Time | Epochs | Learning Rate | Device |
| ---------- | -------- | ------------- | ------ | ------------- | ------ |
| Vanilla    |          |               | 100    | 0.01          | GPU    |
| Pytorch    |          |               | 100    | 0.01          | GPU    |
| TensorFlow |          |               | 100    | 0.01          | CPU    |

### Here are tables comparing the performance of the different neural networks depending on the framework used:

- Neural Network built from scratch using only **CuPy**.

| Framework         | Accuracy | Training Time | Epochs | Learning Rate | Device |
| ----------------- | -------- | ------------- | ------ | ------------- | ------ |
| Simple Perceptron | ~90%     | ~10s          | 100    | 0.01          | GPU    |
| DNN               |          |               | 100    | 0.01          | GPU    |
| CNN               |          |               | 100    | 0.01          | GPU    |
| AlexNet           |          |               | 100    | 0.01          | GPU    |

- Neural Network built with **PyTorch**.

| Neural Network    | Accuracy | Training Time | Epochs | Learning Rate | Device |
| ----------------- | -------- | ------------- | ------ | ------------- | ------ |
| Simple Perceptron | ~90%     | ~0.35s        | 100    | 0.01          | GPU    |
| DNN               |          |               | 100    | 0.01          | GPU    |
| CNN               |          |               | 100    | 0.01          | GPU    |
| AlexNet           |          |               | 100    | 0.01          | GPU    |

- Neural Network built with **TensorFlow**.

| Neural Network    | Accuracy | Training Time | Epochs | Learning Rate | Device |
| ----------------- | -------- | ------------- | ------ | ------------- | ------ |
| Simple Perceptron |          |               | 100    | 0.01          | CPU    |
| DNN               |          |               | 100    | 0.01          | CPU    |
| CNN               |          |               | 100    | 0.01          | CPU    |
| AlexNet           |          |               | 100    | 0.01          | CPU    |

> **Note**: These values can change depending on the version of Python and your PC<br><br>
> For these benchmarks, I used Python 3.12.4 64-bit implemented with CPython on a ryzen 5 3600, rtx 2060 with 2\*8GB of RAM clocked at 3600Hz on Windows 10.
