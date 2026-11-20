import torch
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
from collections.abc import Mapping, Sequence

class SmallCNN(nn.Module):
    """Small CNN, without softmax, that takes 28*28 pictures in input, with 3 colours, and returns a class between 1 and 9"""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28×28 → 14×14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14×14 → 7×7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns, for each image in images, its class after going through the model

        Args:
            images (torch.Tensor): Contains all images that need to be put through the model

        Returns:
            torch.Tensor: A tensor of the class for each image
        """
        return self.network(images)


def train_client(
    global_model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Trains on a specific client, here an hospital, a copy of the global model.

    Args:
        global_model (nn.Module): The model which is copied by all clients at a specific step, and that copy is trained by each client.
        loader (DataLoader): Images and labels for the training
        epochs (int): Number of epochs in the training
        lr (float): Learning rate for the optimization
        device (torch.device): Device used for the training

    Returns:
        dict[str, torch.Tensor]: Contains all the weights for each layer after training.
    """
    model = deepcopy(global_model).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).reshape(-1).long()
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def weighted_average(
    states: Sequence[Mapping[str, torch.Tensor]], sizes: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Implements the Federated Averaging (FedAvg) aggregation algorithm.
    The function computes a weighted average of the model parameters, where
    each hospital's contribution is proportional to its training dataset size.
    The returned dictionary contains the aggregated parameters, ready to be
    loaded into the new global model.

    Args:
        states (Sequence[Mapping[str, torch.Tensor]]): 
            Contains the trained model parameters (state_dict) from each hospital.
            Each element corresponds to the locally trained model of one hospital.
        sizes (Sequence[int]): Contains the number of training images available at each hospital.

    Returns:
        dict[str, torch.Tensor]: Contains all the weights for each layer after training, averaged from each client,
            which are the weights that are going to be sent to the global model for update.
    """
    if not states or len(states) != len(sizes) or any(size <= 0 for size in sizes):
        raise ValueError(
            "There must be one model and a strictly positive dataset size per client."
        )
    total = sum(sizes)
    average = {}
    for name, first in states[0].items():
        if not first.is_floating_point():
            raise ValueError(f"Tensor {name} is not floating-point (e.g. BatchNorm).")
        average[name] = torch.zeros_like(first)
        for state, size in zip(states, sizes, strict=True):
            average[name].add_(state[name], alpha=size / total)
    return average


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluates the accuracy and the average of the loss sum of the model on the data in loader.

    Args:
        model (nn.Module): The model that is evaluated.
        loader (DataLoader): The data on which it is evaluated.
        device (torch.device): The device used for the evaluation.

    Returns:
        tuple[float, float]: Accuracy and average of the loss sum of the model.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct, total, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).reshape(-1).long()
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return correct / total, loss_sum / total