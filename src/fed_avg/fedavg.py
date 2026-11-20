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
        """Returns, for each image in images, its class after going through the model"""
        return self.network(images)

def train_client(
    global_model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Every hospital uses the same global model global_model at each step.

    loader contains all the images and labels for the training.
    epochs is the number of epochs in the training.
    lr is the learning rate of the training.
    device is the device used for the training.
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
    """
    Implements the Federated Averaging (FedAvg) aggregation algorithm.

    states contains the trained model parameters (state_dict) from each hospital.
    Each element corresponds to the locally trained model of one hospital.
    sizes contains the number of training images available at each hospital.
    The function computes a weighted average of the model parameters, where
    each hospital's contribution is proportional to its training dataset size.
    The returned dictionary contains the aggregated parameters, ready to be
    loaded into the new global model.

    The weights represent the number of images, not the number of mini-batches.
    Replacing this function will allow us to test other aggregation methods later.
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
    """
    Evaluates the accuracy and the average of the loss sum of the model on the data in loader.

    device is the device used for the evaluation.
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