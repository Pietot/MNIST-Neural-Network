from matplotlib import pyplot
from typing import Any

from numpy import typing as npt

import numpy as np

import keras


class NeuralNetwork:
    def __init__(self) -> None:
        self.vector_weight = np.random.rand(784, 10).T
        self.train_matrix, self.answer = load_train_mnist()
        self.test_matrix = load_test_mnist()
        self.bias = np.random.rand(1)

    def activation(self):
        weighted_sum = self.vector_weight.dot(self.train_matrix) + self.bias
        activation = np.maximum(0, weighted_sum)
        return activation

    def softmax(self):
        activation = self.activation()
        activation_max = np.max(activation, axis=1, keepdims=True)
        exp_activation = np.exp(activation - activation_max)
        return exp_activation / np.sum(exp_activation, axis=1, keepdims=True)

    def forward_propagation(self):
        return self.softmax()

    def log_loss(self):
        softmax = self.forward_propagation()
        print(softmax)
        size = self.train_matrix.shape[1]
        epsilon = 1e-15
        log_loss = (
            -1
            / size
            * np.sum(
                self.answer * np.log(softmax + epsilon)
                + (1 - self.answer) * np.log(1 - softmax + epsilon)
            )
        )
        return log_loss

    def train(self):
        loss = self.log_loss()
        print(loss)

    def test(self):
        pass


def load_train_mnist() -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    train_dataset = keras.datasets.mnist.load_data()[0]
    return train_dataset[0].reshape(60000, 784).T, train_dataset[1]


def load_test_mnist() -> npt.NDArray[np.uint8]:
    return keras.datasets.mnist.load_data()[1][0].reshape(10000, 784).T


if __name__ == "__main__":
    network = NeuralNetwork()
    network.train()
