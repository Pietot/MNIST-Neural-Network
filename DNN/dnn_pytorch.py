"""Deep Neural Network (DNN) built with PyTorch with 2 hidden layers with 32 neurons each"""

import time
from typing import Any, List

import torch
import torchvision  # type: ignore
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor, cuda, nn, optim
from torchvision.datasets.mnist import MNIST  # type: ignore
from tqdm import tqdm


class DeepNeuralNet(nn.Module):
    """Deep Neural Network class using PyTorch nn.Module with 2 hidden layers"""

    def __init__(self) -> None:
        super().__init__()  # type: ignore
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward pass through the network

        Args:
            input_tensor: Input data tensor of shape

        Returns:
            Tensor: Output logits tensor of shape
        """
        first_hidden = self.relu(self.fc1(input_tensor))
        second_hidden = self.relu(self.fc2(first_hidden))
        return self.fc3(second_hidden)


class DeepNeuralNetwork:
    """Deep Neural Network class wrapper"""

    def __init__(self, nb_epoch: int = 500, learning_rate: float | int = 0.1) -> None:
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.train_matrix = load_train_mnist(self.device)
        self.test_matrix = load_test_mnist(self.device)

        self.model = DeepNeuralNet().to(self.device)

        # Initialize weights with He initialization (similar to the CuPy code)
        nn.init.kaiming_normal_(self.model.fc1.weight)
        nn.init.zeros_(self.model.fc1.bias)
        nn.init.kaiming_normal_(self.model.fc2.weight)
        nn.init.zeros_(self.model.fc2.bias)
        nn.init.kaiming_normal_(self.model.fc3.weight)
        nn.init.zeros_(self.model.fc3.bias)

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.nb_epoch = nb_epoch
        self.losses: List[torch.Tensor] = []
        self.training_time = 0.0

    def forward_propagation(self, matrix: torch.Tensor) -> torch.Tensor:
        """Forward propagation using the PyTorch model

        Args:
            matrix (torch.Tensor): The input matrix (784, batch_size)

        Returns:
            torch.Tensor: The output of the forward propagation
        """
        output = self.model(matrix.T)
        return nn.functional.softmax(output.T, dim=0)

    def train(self) -> dict[str, torch.Tensor]:
        """Train function using PyTorch's optimizer

        Returns:
            dict[str, torch.Tensor]: The state dict of the model
        """
        start = time.time()
        self.model.train()
        for _ in tqdm(range(self.nb_epoch)):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(self.train_matrix.data.T)
            targets = torch.argmax(self.train_matrix.targets, dim=0)
            loss = self.criterion(outputs, targets)
            self.losses.append(loss.detach().clone())

            # Backward pass
            loss.backward()
            self.optimizer.step()

        self.training_time = round(time.time() - start, 3)
        print(f"Training time: {self.training_time} seconds")
        self.show_loss()
        return self.model.state_dict()

    def test(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Test function, test the model and return failures

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The failures and predictions of the model
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.test_matrix.data.T)
            _, test_predictions = torch.max(outputs, 1)
            test_labels = torch.argmax(self.test_matrix.targets, dim=0)
            accuracy = torch.mean((test_predictions == test_labels).float())
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            failures = torch.where(test_predictions != test_labels)[0]
            return failures, test_predictions

    def predict(self, image_path: str) -> int:
        """Predict function, predict the class of an image

        Args:
            image_path (str): The path to the image

        Raises:
            ValueError: If the image is not 28x28 pixels

        Returns:
            int: The predicted class of the image
        """
        image = Image.open(image_path).convert("L")
        image = torchvision.transforms.ToTensor()(image).squeeze().to(self.device)
        if image.shape != (28, 28):
            raise ValueError("The image must be 28x28 pixels")

        image = image.view(1, 784) / 255
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            return int(torch.argmax(output).item())

    def test_and_show_fails(self, number: int = 9) -> None:
        """Show the failures of the model

        Args:
            number (int): The number of failures to show
        """
        failures, test_predictions = self.test()
        print(f"Number of failures: {len(failures)}")
        test_labels = torch.argmax(self.test_matrix.targets, dim=0)

        if number > len(failures):
            number = len(failures)
            print(f"Only {number} failures found.")

        plt.figure(figsize=(10, 10))  # type: ignore
        for i in range(number):
            index = failures[i]
            image = self.test_matrix.data[:, index].reshape(28, 28).to("cpu")
            true_label = test_labels[index]
            predicted_label = test_predictions[index]

            plt.subplot(number // 3 + 1, 3, i + 1)  # type: ignore
            plt.imshow(image, cmap="gray")  # type: ignore
            plt.title(f"True: {true_label}, Pred: {predicted_label}")  # type: ignore
            plt.axis("off")  # type: ignore

        plt.subplots_adjust(hspace=0.5, wspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.show()  # type: ignore

    def show_loss(self) -> None:
        """Show the loss of the model"""
        plt.plot([loss.item() for loss in self.losses])  # type: ignore
        plt.title("Loss")  # type: ignore
        plt.xlabel("Epoch")  # type: ignore
        plt.ylabel("Loss")  # type: ignore
        plt.show()  # type: ignore

    def save(self, filepath: str) -> None:
        """Save the model to a file using PyTorch's save mechanism

        Args:
            filepath (str): The filepath to save the model (extension .pt)
        """
        state: dict[str, dict[str, Any] | list[Tensor] | float | int] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
            "training_time": self.training_time,
            "nb_epoch": self.nb_epoch,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        torch.save(state, filepath)  # type: ignore

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "DeepNeuralNetwork":
        """Load a model from a file using PyTorch's load mechanism

        Args:
            path (str): The path to load the model from
            device (torch.device):
                Device to load the model on.
                If None, uses available device.

        Returns:
            DeepNeuralNetwork": The loaded model wrapper
        """
        if device is None:
            device = torch.device("cuda" if cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)  # type: ignore
        instance = cls(nb_epoch=checkpoint["nb_epoch"], learning_rate=checkpoint["learning_rate"])
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        instance.losses = checkpoint["losses"]
        instance.training_time = checkpoint["training_time"]
        return instance


def load_train_mnist(device: torch.device) -> MNIST:
    """Load the training MNIST dataset

    Args:
        device (torch.device): The device to load the dataset on

    Returns:
        MNIST: The training MNIST dataset
    """
    train_dataset = torchvision.datasets.MNIST(
        root="data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    train_dataset.data = train_dataset.data.view(60000, 784).T.float().to(device) / 255
    train_dataset.targets = (
        # pylint: disable-next=not-callable
        torch.nn.functional.one_hot(train_dataset.targets, num_classes=10).T.float().to(device)
    )
    return train_dataset


def load_test_mnist(device: torch.device) -> MNIST:
    """Load the test MNIST dataset

    Args:
        device (torch.device): The device to load the dataset on

    Returns:
        MNIST: The test MNIST dataset
    """
    test_dataset = torchvision.datasets.MNIST(
        root="data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_dataset.data = test_dataset.data.view(10000, 784).T.float().to(device) / 255
    test_dataset.targets = (
        # pylint: disable-next=not-callable
        torch.nn.functional.one_hot(test_dataset.targets, num_classes=10).T.float().to(device)
    )
    return test_dataset


if __name__ == "__main__":
    network = DeepNeuralNetwork(nb_epoch=500, learning_rate=0.1)
    network.train()
    network.test()
