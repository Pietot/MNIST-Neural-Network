import time
from matplotlib import pyplot as plt
from typing import Any

from numpy import typing as npt
from tqdm import tqdm

import numpy as np

import keras  # type: ignore


class NeuralNetwork:
    def __init__(self) -> None:
        self.vector_weight = np.random.rand(784, 10).T
        self.train_matrix, self.answer = load_train_mnist()
        self.test_matrix, self.test_labels = load_test_mnist()
        self.bias = np.zeros((10, 1))

    def activation(self, Z: npt.NDArray[np.float64]) -> np.ndarray[Any, np.dtype[Any]]:
        """Activation function using ReLU

        Args:
            Z (npt.NDArray[np.float64]): The input to the activation function

        Returns:
            np.ndarray[Any, np.dtype[Any]]: The output of the activation function
        """
        return np.maximum(0, Z)

    def softmax(self, A: npt.NDArray[np.float64]) -> Any:
        """Softmax function, formula: exp(A - max(A)) / sum(exp(A - max(A)))

        Args:
            A (npt.NDArray[np.float64]): The input to the softmax function

        Returns:
            Any: The output of the softmax function
        """
        exp_A = np.exp(A - np.max(A, axis=0, keepdims=True))
        return exp_A / np.sum(exp_A, axis=0, keepdims=True)

    def forward_propagation(self, matrix: npt.NDArray[np.uint8]) -> Any:
        """Forward propagation function, do the matrix multiplication,
            activation function and softmax function

        Args:
            matrix (npt.NDArray[np.uint8]): The input matrix

        Returns:
            Any: The output of the forward propagation
        """
        Z = self.vector_weight.dot(matrix) + self.bias
        A = self.activation(Z)
        return self.softmax(A)

    def log_loss(self, softmax: npt.NDArray[np.float64]) -> np.floating[Any]:
        """Log loss function, formula: -1 / size * sum(y * log(softmax) + (1 - y) * log(1 - softmax))

        Args:
            softmax (npt.NDArray[np.float64]): The output of the softmax function

        Returns:
            np.floating[Any]: The log loss
        """
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

    def gradient(
        self, losses: list[np.floating[Any]]
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], Any]:
        """Gradient function, calculate the gradient of the weights and bias

        Args:
            losses (list[np.floating[Any]]): The list of losses

        Returns:
            tuple[np.ndarray[Any, np.dtype[Any]], Any]: The gradient of the weights and bias
        """
        size = self.train_matrix.shape[1]
        predictions = self.forward_propagation(self.train_matrix)
        loss = self.log_loss(predictions)
        losses.append(loss)
        dw = 1 / size * self.train_matrix.dot(predictions.T - self.answer.T)
        db = 1 / size * np.sum(predictions - self.answer)
        return (dw, db)

    def update(self, losses: list[np.floating[Any]]) -> None:
        """Update function, update the weights and bias

        Args:
            losses (list[np.floating[Any]]): The list of losses
        """
        dw, db = self.gradient(losses)
        self.vector_weight -= 0.5 * dw.T
        self.bias -= 0.5 * db

    def train(self) -> None:
        """Train function, train the model and plot the losses"""
        start = time.time()
        losses: list[np.floating[Any]] = []
        for _ in tqdm(range(100)):
            self.update(losses)
        self.training_time = round(time.time() - start, 3)

        plt.plot(losses)
        plt.show()

    def test(self) -> None:
        """Test function, test the model and print the accuracy"""
        test_predictions = self.forward_propagation(self.test_matrix)
        test_predictions = np.argmax(test_predictions, axis=0)
        test_labels = np.argmax(self.test_labels, axis=0)
        accuracy = np.mean(test_predictions == test_labels)  # type: ignore
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


def load_train_mnist() -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Load the training MNIST dataset

    Returns:
        tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: The training dataset
    """
    train_dataset = keras.datasets.mnist.load_data()[0]
    return train_dataset[0].reshape(60000, 784).T, keras.utils.to_categorical(  # type: ignore
        train_dataset[1]
    ).T


def load_test_mnist() -> tuple[Any, Any]:
    """Load the testing MNIST dataset

    Returns:
        tuple[Any, Any]: The testing dataset
    """
    test_dataset = keras.datasets.mnist.load_data()[1]
    return test_dataset[0].reshape(10000, 784).T, keras.utils.to_categorical(  # type: ignore
        test_dataset[1]
    ).T


if __name__ == "__main__":
    network = NeuralNetwork()
    network.train()
    print(network.training_time)
    network.test()
