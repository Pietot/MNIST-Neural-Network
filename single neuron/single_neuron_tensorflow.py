import pickle
import time

import tensorflow as tf  # type: ignore
import tensorflow_datasets as tfds  # type: ignore
from matplotlib import pyplot as plt
from numpy import typing as npt
from PIL import Image
from tqdm import tqdm


class NeuralNetwork:
    """Neural Network class"""

    def __init__(self, nb_epoch: int = 100, learning_rate: float | int = 0.01) -> None:
        self.device = tf.config.list_physical_devices("GPU")  # type: ignore
        if not self.device:  # type: ignore
            self.device = tf.config.list_physical_devices("CPU")  # type: ignore
        self.vector_weight = tf.Variable(tf.random.normal((784, 10)))  # type: ignore
        self.train_matrix, self.answer = load_train_mnist()
        self.test_matrix, self.test_labels = load_test_mnist()
        self.bias = tf.Variable(tf.zeros((10, 1)))  # type: ignore
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.losses: list[tf.Tensor] = []
        self.training_time: float = 0.0


def load_train_mnist() -> tuple[tf.Tensor, tf.Tensor]:
    """Load the training MNIST dataset

    Returns:
        tuple[tf.Tensor, tf.Tensor]: The training dataset
    """
    (train_matrix, train_labels), _ = tfds.load(  # type: ignore
        "mnist",
        split=["train", "test"],
        data_dir="C:\\Users\\Bapti\\OneDrive\\Bureau\\VS Code/Python\
\\Solo Project\\Medium Projects\\Autre\\MNIST Neural Network",
        batch_size=-1,
        download=True,
        as_supervised=True,
    )
    train_matrix = tf.transpose(tf.reshape(train_matrix, (60000, 784)))  # type: ignore
    train_labels = tf.transpose(tf.one_hot(train_labels, 10))  # type: ignore
    return train_matrix, train_labels  # type: ignore


def load_test_mnist() -> tuple[tf.Tensor, tf.Tensor]:
    """Load the testing MNIST dataset

    Returns:
        tuple[tf.Tensor, tf.Tensor]: The testing dataset
    """
    _, (test_matrix, test_labels) = tfds.load(  # type: ignore
        "mnist",
        split=["train", "test"],
        data_dir="C:\\Users\\Bapti\\OneDrive\\Bureau\\VS Code/Python\
\\Solo Project\\Medium Projects\\Autre\\MNIST Neural Network",
        batch_size=-1,
        download=True,
        as_supervised=True,
    )
    test_matrix = tf.transpose(tf.reshape(test_matrix, (10000, 784)))  # type: ignore
    test_labels = tf.transpose(tf.one_hot(test_labels, 10))  # type: ignore
    return test_matrix, test_labels  # type: ignore


if __name__ == "__main__":
    x, y = load_test_mnist()
    print(x.shape, y.shape)
