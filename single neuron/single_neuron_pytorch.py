import time
import torch
import torchvision
import pickle

from torchvision.datasets.mnist import MNIST  # type: ignore
from torch.utils.data import DataLoader

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


def load_train_mnist() -> MNIST:
    """Load the training MNIST dataset

    Returns:
        The training dataset
    """
    train_dataset = torchvision.datasets.MNIST(
        root="C:\\Users\\Bapti\\OneDrive\\Bureau\\VS Code/Python\
\\Solo Project\\Medium Projects\\Autre\\MNIST Neural Network",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    return train_dataset


if __name__ == "__main__":
    labels_map = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }

    network = NeuralNetwork()
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(network.train_matrix), size=(1,)).item()
        img, label = network.train_matrix[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
