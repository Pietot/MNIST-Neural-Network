"""Neural Network with entries directly connected to the outputs"""

import pickle
import time
from typing import Any

import cupy as cp  # type: ignore
import keras  # type: ignore
from matplotlib import pyplot as plt
from numpy import typing as npt
from PIL import Image
from tqdm import tqdm


class NeuralNetwork:
    """Neural Network class"""

    def __init__(self, nb_epoch: int = 100, learning_rate: float | int = 1) -> None:
        self.train_matrix, self.answer = load_train_mnist()
        self.test_matrix, self.test_labels = load_test_mnist()
        self.vector_weight = cp.random.rand(784, 10).T  # type: ignore
        self.bias = cp.zeros((10, 1), dtype=cp.float64)  # type: ignore
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.losses: list[cp.floating[Any]] = []
        self.training_time: float = 0.0

    def activation(  # type: ignore
        self, weighted_sum: npt.NDArray[cp.float64]
    ) -> cp.ndarray[Any, cp.dtype[Any]]:  # type: ignore
        """Activation function using ReLU, formula: max(0, weighted_sum)

        Args:
            weighted_sum (npt.NDArray[cp.float64]): The input to the activation function

        Returns:
            cp.ndarray[Any, cp.dtype[Any]]: The output of the activation function
        """
        return cp.maximum(0, weighted_sum)  # type: ignore

    def softmax(self, activation: npt.NDArray[cp.float64]) -> Any:
        """Softmax function, formula: exp(A - max(A)) / sum(exp(A - max(A)))

        Args:
            A (npt.NDArray[cp.float64]): The input to the softmax function

        Returns:
            Any: The output of the softmax function
        """
        max_actvation = cp.max(activation, axis=0, keepdims=True)  # type: ignore
        exp_activation = cp.exp(activation - max_actvation)  # type:ignore
        return exp_activation / cp.sum(exp_activation, axis=0, keepdims=True)  # type: ignore

    def forward_propagation(self, matrix: npt.NDArray[cp.uint8]) -> Any:
        """Forward propagation function, do the matrix multiplication,
            activation function and softmax function

        Args:
            matrix (npt.NDArray[cp.uint8]): The input matrix

        Returns:
            Any: The output of the forward propagation
        """
        weighted_sum = self.vector_weight.dot(matrix) + self.bias  # type: ignore
        activation = self.activation(weighted_sum)  # type: ignore
        return self.softmax(activation)  # type: ignore

    def log_loss(self, softmax: npt.NDArray[cp.float64]) -> cp.floating[Any]:
        """Log loss function implemented with CCE, formula:
            -1 / N * sum(y * log(softmax + epsilon))

        Args:
            softmax (npt.NDArray[cp.float64]): The output of the softmax function

        Returns:
            cp.floating[Any]: The log loss
        """
        epsilon = 1e-15
        size = self.train_matrix.shape[1]
        log_loss = (  # type: ignore
            -1
            / size
            * cp.sum(  # type: ignore
                self.answer * cp.log(softmax + epsilon)  # type: ignore
            )
        )
        return log_loss  # type: ignore

    def gradient(self) -> tuple[cp.ndarray[Any, cp.dtype[Any]], Any]:  # type: ignore
        """Gradient function, calculate the gradient of the weights and bias

        Returns:
            tuple[cp.ndarray[Any, cp.dtype[Any]], Any]: The gradient of the weights and bias
        """
        predictions = self.forward_propagation(self.train_matrix)
        loss = self.log_loss(predictions)
        self.losses.append(loss)
        size = self.train_matrix.shape[1]
        dw = 1 / size * self.train_matrix.dot(predictions.T - self.answer.T)
        db = 1 / size * cp.sum(predictions - self.answer)  # type: ignore
        return (dw, db)  # type: ignore

    def update(self) -> None:
        """Update function, update the weights and bias"""
        dw, db = self.gradient()  # type: ignore
        self.vector_weight -= self.learning_rate * dw.T  # type: ignore
        self.bias -= self.learning_rate * db  # type: ignore

    def train(self) -> tuple[cp.ndarray[Any, cp.dtype[Any]], Any]:  # type: ignore
        """Train function, train the model

        Returns:
            tuple[cp.ndarray[Any, cp.dtype[Any]], Any]: The weights and bias of the model
        """
        start = time.time()
        for _ in tqdm(range(self.nb_epoch)):
            self.update()
        self.training_time = round(time.time() - start, 3)
        return (self.vector_weight, self.bias)  # type: ignore

    def test(self) -> tuple[cp.ndarray[Any, cp.dtype[Any]], Any]:  # type: ignore
        """Test function, test the model and print the accuracy

        Returns:
            tuple[cp.ndarray[Any, cp.dtype[Any]], Any]: The failures and predictions of the model
        """
        test_predictions = self.forward_propagation(self.test_matrix)
        test_predictions = cp.argmax(test_predictions, axis=0)  # type: ignore
        test_labels = cp.argmax(self.test_labels, axis=0)  # type: ignore
        accuracy = cp.mean(test_predictions == test_labels)  # type: ignore
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        failures = cp.where(test_predictions != test_labels)[0]  # type: ignore
        return failures, test_predictions  # type: ignore

    def predict(self, image_path: str) -> int:
        """Predict function, predict the digit in the image

        Args:
            image_path (str): The path to the image

        Returns:
            Any: The prediction of the image
        """
        image = Image.open(image_path).convert("L")
        image = cp.asarray(image)  # type: ignore
        if image.shape != (28, 28):  # type: ignore
            raise ValueError("The image must be 28x28 pixels")
        image = image.reshape(784, 1) / 255  # type: ignore
        predictions = self.forward_propagation(image)  # type: ignore
        return cp.argmax(predictions, axis=0)[0]  # type: ignore

    def test_and_show_fails(self, number: int = 9) -> None:
        """Show the failures of the model

        Args:
            number (int): The number of failures to show
        """
        failures, test_predictions = self.test()  # type: ignore
        print(f"Number of failures: {len(failures)}")  # type: ignore
        test_labels = cp.argmax(self.test_labels, axis=0)  # type: ignore

        if number > len(failures):  # type: ignore
            number = len(failures)  # type: ignore
            print(f"Only {number} failures found.")

        plt.figure(figsize=(10, 10))  # type: ignore
        for i in range(number):
            index = failures[i]  # type: ignore
            image = cp.asnumpy(self.test_matrix[:, index].reshape(28, 28))  # type: ignore
            true_label = test_labels[index]
            predicted_label = test_predictions[index]

            plt.subplot(number // 3 + 1, 3, i + 1)  # type: ignore
            plt.imshow(image, cmap="gray")  # type: ignore
            plt.title(f"True: {true_label}, Pred: {predicted_label}")  # type: ignore
            plt.axis("off")  # type: ignore

        # Adjust layout to remove excess white space
        plt.subplots_adjust(hspace=0.5, wspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.show()  # type: ignore

    def show_loss(self) -> None:
        """Show the loss of the model"""
        plt.plot(self.losses)  # type: ignore
        plt.title("Loss")  # type: ignore
        plt.xlabel("Epoch")  # type: ignore
        plt.ylabel("Loss")  # type: ignore
        plt.show()  # type: ignore

    def save(self, path: str) -> None:
        """Save the model

        Args:
            path (str): The path to save the model
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)


def load_train_mnist() -> tuple[Any, Any]:
    """Load the training MNIST dataset

    Returns:
        tuple[Any, Any]: The training dataset
    """
    train_dataset = keras.datasets.mnist.load_data()[0]
    return cp.asarray(train_dataset[0].reshape(60000, 784).T) / 255, cp.asarray(  # type: ignore
        keras.utils.to_categorical(  # type: ignore
            train_dataset[1]
        ).T
    )


def load_test_mnist() -> tuple[Any, Any]:
    """Load the testing MNIST dataset

    Returns:
        tuple[Any, Any]: The testing dataset
    """
    test_dataset = keras.datasets.mnist.load_data()[1]
    return cp.asarray(test_dataset[0].reshape(10000, 784).T) / 255, cp.asarray(  # type: ignore
        keras.utils.to_categorical(  # type: ignore
            test_dataset[1]
        ).T
    )


if __name__ == "__main__":
    network = NeuralNetwork()
    network.train()
    network.test_and_show_fails()
