"""Neural Network with a single neuron built from scratch"""

import pickle
import time
from typing import Any

import keras  # type: ignore
import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt
from PIL import Image
from tqdm import tqdm


class NeuralNetwork:
    """Neural Network class"""

    def __init__(self, nb_epoch: int = 100, learning_rate: float | int = 0.01) -> None:
        self.vector_weight = np.random.rand(784, 10).T
        self.train_matrix, self.answer = load_train_mnist()
        self.test_matrix, self.test_labels = load_test_mnist()
        self.bias = np.zeros((10, 1), dtype=np.float64)
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.losses: list[np.floating[Any]] = []
        self.training_time: float = 0.0

    def activation(
        self, weighted_sum: npt.NDArray[np.float64]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Activation function using ReLU

        Args:
            weighted_sum (npt.NDArray[np.float64]): The input to the activation function

        Returns:
            np.ndarray[Any, np.dtype[Any]]: The output of the activation function
        """
        return np.maximum(0, weighted_sum)

    def softmax(self, activation: npt.NDArray[np.float64]) -> Any:
        """Softmax function, formula: exp(A - max(A)) / sum(exp(A - max(A)))

        Args:
            A (npt.NDArray[np.float64]): The input to the softmax function

        Returns:
            Any: The output of the softmax function
        """
        exp_activation = np.exp(activation - np.max(activation, axis=0, keepdims=True))
        return exp_activation / np.sum(exp_activation, axis=0, keepdims=True)

    def forward_propagation(self, matrix: npt.NDArray[np.uint8]) -> Any:
        """Forward propagation function, do the matrix multiplication,
            activation function and softmax function

        Args:
            matrix (npt.NDArray[np.uint8]): The input matrix

        Returns:
            Any: The output of the forward propagation
        """
        weighted_seum = self.vector_weight.dot(matrix) + self.bias
        activation = self.activation(weighted_seum)
        return self.softmax(activation)

    def log_loss(self, softmax: npt.NDArray[np.float64]) -> np.floating[Any]:
        """Log loss function, formula:
            -1 / size * sum(y * log(softmax) + (1 - y) * log(1 - softmax))

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

    def test(self) -> tuple[np.ndarray[Any, np.dtype[Any]], Any]:
        """Test function, test the model and print the accuracy"""
        test_predictions = self.forward_propagation(self.test_matrix)
        test_predictions = np.argmax(test_predictions, axis=0)
        test_labels = np.argmax(self.test_labels, axis=0)
        accuracy = np.mean(test_predictions == test_labels)  # type: ignore
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        failures = np.where(test_predictions != test_labels)[0]
        return failures, test_predictions

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

    def test_and_show_fails(self, number: int) -> None:
        """Show the failures of the model.

        Args:
            number (int): The number of failures to show.
        """
        failures, test_predictions = self.test()
        test_labels = np.argmax(self.test_labels, axis=0)

        if number > len(failures):
            number = len(failures)
            print(f"Only {number} failures found.")

        plt.figure(figsize=(10, 10))  # type: ignore
        for i in range(number):
            index = failures[i]
            image = self.test_matrix[:, index].reshape(28, 28)
            true_label = test_labels[index]
            predicted_label = test_predictions[index]

            plt.subplot(number // 3 + 1, 3, i + 1)  # type: ignore
            plt.imshow(image, cmap="gray")  # type: ignore
            plt.title(f"True: {true_label}, Pred: {predicted_label}")  # type: ignore
            plt.axis("off")  # type: ignore

        # Adjust layout to remove excess white space
        plt.subplots_adjust(
            hspace=0.5, wspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9
        )
        plt.show()  # type: ignore

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
    with open("single_neuron.pkl", "rb") as file:
        network = pickle.load(file)
    network.test_and_show_fails(20)
