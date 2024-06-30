from matplotlib import pyplot
from typing import Any

from numpy import typing as npt

import numpy as np

import keras


class NeuralNetwork:
    def __init__(self) -> None:
        self.vector_weight = np.random.rand(784)
        self.train_matrix, self.answer = load_train_mnist()
        self.test_matrix = load_test_mnist()
        self.bias = np.random.rand(1)
    
    def activation(self):
        weighted_sum = self.vector_weight.dot(self.train_matrix) + self.bias
        activation = np.maximum(0, weighted_sum)
        return activation
    
    def softmax(self):
        activation = self.activation()
        return np.exp(activation) / np.sum(np.exp(activation), axis=0)
    
    def forward_propagation(self):
        return self.softmax()
    
    def loss(self):
        return -np.log(self.forward_propagation())
    
    def train(self):
        pass
    
    def test(self):
        pass
        
    


def load_train_mnist() -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    train_dataset = keras.datasets.mnist.load_data()[0]
    return train_dataset[0].reshape(60000, 784), train_dataset[1]


def load_test_mnist() -> npt.NDArray[np.uint8]:
    return keras.datasets.mnist.load_data()[1][0].reshape(10000, 784)



if __name__ == "__main__":
    network = NeuralNetwork()