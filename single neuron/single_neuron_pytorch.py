import time
import torch
import torchvision
import pickle

from PIL import Image
from tqdm import tqdm
from typing import Any

import numpy as np


class NeuralNetwork:
    def __init__(self, nb_epoch: int = 100, learning_rate: float | int = 0.01) -> None:
        self.vector_weight = torch.rand(784, 10)
        self.train_matrix = load_train_mnist()
        # self.test_matrix = load_test_mnist()
        self.bias = torch.zeros(10, 1)
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.losses: list[torch.Tensor] = []
        self.accuracies: list[torch.Tensor] = []
        self.training_time: float = 0.0


def load_train_mnist() -> :
    """Load the training MNIST dataset

    Returns:
        : The training dataset
    """
    train_dataset = torchvision.datasets.MNIST(
        root="/",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    return train_dataset
    train_dataset = torch.reshape(train_dataset, (60000, 784))


if __name__ == "__main__":
    network = NeuralNetwork()
    network.train()
    print(network.training_time)
    network.test()
