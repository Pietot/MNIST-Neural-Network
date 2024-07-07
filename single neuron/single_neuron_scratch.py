import time
import pickle
import keras  # type: ignore

from numpy import typing as npt

from PIL import Image
from tqdm import tqdm
from typing import Any

import numpy as np


class NeuralNetwork:
    def __init__(self, nb_epoch: int = 100, learning_rate: float | int = 0.01) -> None:
        self.vector_weight = np.random.rand(784, 10).T
        self.train_matrix, self.answer = load_train_mnist()
        self.test_matrix, self.test_labels = load_test_mnist()
        self.bias = np.zeros((10, 1), dtype=np.float64)
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.losses: list[np.floating[Any]] = []

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

    def gradient(self) -> tuple[np.ndarray[Any, np.dtype[Any]], Any]:
        """Gradient function, calculate the gradient of the weights and bias

        Returns:
            tuple[np.ndarray[Any, np.dtype[Any]], Any]: The gradient of the weights and bias
        """
        size = self.train_matrix.shape[1]
        predictions = self.forward_propagation(self.train_matrix)
        loss = self.log_loss(predictions)
        self.losses.append(loss)
        dw = 1 / size * self.train_matrix.dot(predictions.T - self.answer.T)
        db = 1 / size * np.sum(predictions - self.answer)
        return (dw, db)

    def update(self) -> None:
        """Update function, update the weights and bias"""
        dw, db = self.gradient()
        self.vector_weight -= self.learning_rate * dw.T
        self.bias -= self.learning_rate * db

    def train(self) -> tuple[np.ndarray[Any, np.dtype[Any]], Any]:
        """Train function, train the model and plot the losses"""
        start = time.time()
        for _ in tqdm(range(self.nb_epoch)):
            self.update()
        self.training_time = round(time.time() - start, 3)
        return (self.vector_weight, self.bias)

    def test(self) -> None:
        """Test function, test the model and print the accuracy"""
        test_predictions = self.forward_propagation(self.test_matrix)
        test_predictions = np.argmax(test_predictions, axis=0)
        test_labels = np.argmax(self.test_labels, axis=0)
        accuracy = np.mean(test_predictions == test_labels)  # type: ignore
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    def predict(self, fp: str) -> int:
        """Predict function, predict the digit in the image

        Args:
            fp (str): The path to the image

        Returns:
            Any: The prediction of the image
        """
        image = Image.open(fp).convert("L")
        image = np.asarray(image)
        if image.shape != (28, 28):
            raise ValueError("The image must be 28x28 pixels")
        image = image.reshape(784, 1)
        predictions = self.forward_propagation(image)
        return np.argmax(predictions, axis=0)[0]

    def save(self, path: str) -> None:
        """Save the model

        Args:
            path (str): The path to save the model
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)


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
