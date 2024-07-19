"""Neural Network with a single neuron built with PyTorch"""

import pickle
import time

import torch
import torchvision  # type: ignore
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.datasets.mnist import MNIST  # type: ignore
from tqdm import tqdm


class NeuralNetwork:
    """Neural Network class"""

    def __init__(self, nb_epoch: int = 100, learning_rate: float | int = 0.01) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vector_weight = torch.rand(784, 10).T.to(self.device)
        self.train_matrix = load_train_mnist(self.device)
        self.test_matrix = load_test_mnist(self.device)
        self.bias = torch.zeros(10, 1).to(self.device)
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.losses: list[torch.Tensor] = []
        self.training_time: float = 0.0

    def activation(self, weighted_sum: torch.Tensor) -> torch.Tensor:
        """Activation function using ReLU

        Args:
            weighted_sum (torch.Tensor): The input to the activation function

        Returns:
            torch.Tensor: The output of the activation function
        """
        return torch.relu(weighted_sum)

    def softmax(self, activation: torch.Tensor) -> torch.Tensor:
        """Softmax function, formula: exp(A - max(A)) / sum(exp(A - max(A)))

        Args:
            A (torch.Tensor): The input to the softmax function

        Returns:
            torch.Tensor: The output of the softmax function
        """
        exp_activation = torch.exp(
            activation - torch.max(activation, dim=0, keepdim=True).values
        )
        return exp_activation / torch.sum(exp_activation, dim=0, keepdim=True)

    def forward_propagation(self, matrix: torch.Tensor) -> torch.Tensor:
        """Forward propagation function, do the matrix multiplication,
            activation function and softmax function

        Args:
            matrix (torch.Tensor): The input matrix

        Returns:
            torch.Tensor: The output of the forward propagation
        """
        weighted_sum = torch.mm(self.vector_weight, matrix) + self.bias
        activation = self.activation(weighted_sum)
        return self.softmax(activation)

    def log_loss(self, softmax: torch.Tensor) -> torch.Tensor:
        """Log loss function, formula:
            -1 / N * sum(y * log(softmax) + (1 - y) * log(1 - softmax))

        Args:
            softmax (torch.Tensor): The output of the softmax function

        Returns:
            torch.Tensor: The output of the log loss function
        """
        size = self.train_matrix.data.shape[1]
        epsilon = 1e-15
        log_loss = (
            -1
            / size
            * torch.sum(
                self.train_matrix.targets * torch.log(softmax + epsilon)
                + (1 - self.train_matrix.targets) * torch.log(1 - softmax + epsilon)
            )
        )
        return log_loss

    def gradient(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradient function, calculate the gradient of the weights and bias

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The gradient of the weights and bias
        """
        size = self.train_matrix.data.shape[1]
        predictions = self.forward_propagation(self.train_matrix.data)
        loss = self.log_loss(predictions)
        self.losses.append(loss)
        dw = (
            1
            / size
            * torch.mm(
                self.train_matrix.data, (predictions.T - self.train_matrix.targets.T)
            )
        )
        db = 1 / size * torch.sum(predictions - self.train_matrix.targets)
        return (dw, db)

    def update(self) -> None:
        """Update function, update the weights and bias"""
        dw, db = self.gradient()
        self.vector_weight -= self.learning_rate * dw.T
        self.bias -= self.learning_rate * db

    def train(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Train function, train the model"""
        start = time.time()
        for _ in tqdm(range(self.nb_epoch)):
            self.update()
        self.training_time = round(time.time() - start, 3)
        return self.vector_weight, self.bias

    def test(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Test function, test the model and plot the accuracies"""
        test_predictions = self.forward_propagation(self.test_matrix.data)
        test_predictions = torch.argmax(test_predictions, dim=0)
        test_labels = torch.argmax(self.test_matrix.targets, dim=0)
        accuracy = torch.mean(torch.eq(test_predictions, test_labels).float())
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        failures = torch.where(test_predictions != test_labels)[0]
        return failures, test_predictions

    def predict(self, image_path: str) -> int:
        """Predict function, predict the class of an image

        Args:
            image_path (str): The path to the image

        Returns:
            int: The prediction
        """
        image = Image.open(image_path).convert("L")
        image = torchvision.transforms.ToTensor()(image)
        if image.shape != (28, 28):
            raise ValueError("The image must be 28x28 pixels")
        image = image.view(1, 784)
        prediction = self.forward_propagation(image)
        return int(torch.argmax(prediction).item())

    def test_and_show_fails(self, number: int) -> None:
        """Show the failures of the model.

        Args:
            number (int): The number of failures to show.
        """
        failures, test_predictions = self.test()
        print(f"Number of failures: {len(failures)}")
        test_labels = torch.argmax(self.test_matrix.targets, dim=1)

        if number > len(failures):
            number = len(failures)
            print(f"Only {number} failures found.")

        plt.figure(figsize=(10, 10))  # type: ignore
        for i in range(number):
            index = failures[i]
            image = self.test_matrix.data[index].reshape(28, 28)
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
        """Save the model to a file

        Args:
            path (str): The path to save the model
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)


def load_train_mnist(device: torch.device) -> MNIST:
    """Load the training MNIST dataset

    Returns:
        MNIST: The training dataset
    """
    train_dataset = torchvision.datasets.MNIST(
        root="C:\\Users\\Bapti\\OneDrive\\Bureau\\VS Code/Python\
\\Solo Project\\Medium Projects\\Autre\\MNIST Neural Network",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    train_dataset.data = train_dataset.data.view(60000, 784).T.float().to(device)
    train_dataset.targets = (
        torch.nn.functional.one_hot(  # pylint: disable=E1102
            train_dataset.targets, num_classes=10
        )
        .T.float()
        .to(device)
    )
    return train_dataset


def load_test_mnist(device: torch.device) -> MNIST:
    """Load the testing MNIST dataset

    Returns:
        MNIST: The testing dataset
    """
    test_dataset = torchvision.datasets.MNIST(
        root="C:\\Users\\Bapti\\OneDrive\\Bureau\\VS Code/Python\
\\Solo Project\\Medium Projects\\Autre\\MNIST Neural Network",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_dataset.data = test_dataset.data.view(10000, 784).T.float().to(device)
    test_dataset.targets = (
        torch.nn.functional.one_hot(  # pylint: disable=E1102
            test_dataset.targets, num_classes=10
        )
        .T.float()
        .to(device)
    )
    return test_dataset


if __name__ == "__main__":
    network = NeuralNetwork()
    network.train()
    print(f"Training time: {network.training_time} seconds")
    network.test()
