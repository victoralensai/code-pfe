"""Le modèle, l'apprentissage local et la moyenne pondérée du serveur."""

from collections.abc import Mapping, Sequence
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader


class SmallCNN(nn.Module):
    """Images RGB 28×28 → neuf scores (logits), sans softmax final."""

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
        # Pas de BatchNorm : tous les éléments du state_dict sont des paramètres
        # flottants. Cela évite de devoir agréger aussi des compteurs/statistiques.

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def train_client(
    global_model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Chaque hôpital repart de la même copie du modèle global à chaque tour."""
    model = deepcopy(global_model).to(device)
    model.train()
    # SGD sans momentum : aucun état d'optimiseur à conserver entre les tours.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            # MedMNIST renvoie des labels [batch, 1], CrossEntropy attend [batch].
            labels = labels.to(device).reshape(-1).long()
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    # La copie CPU peut être transmise au serveur sans garder le graphe ni le GPU.
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def weighted_average(
    states: Sequence[Mapping[str, torch.Tensor]], sizes: Sequence[int]
) -> dict[str, torch.Tensor]:
    """FedAvg : w_(t+1) = somme_k [n_k / somme_j(n_j)] * w_(t+1,k).

    Les poids sont les nombres d'images, pas les nombres de mini-batches.
    Remplacer cette fonction permettra de tester d'autres agrégateurs plus tard.
    """
    if not states or len(states) != len(sizes) or any(size <= 0 for size in sizes):
        raise ValueError(
            "Il faut un modèle et un effectif strictement positif par client."
        )
    total = sum(sizes)
    average = {}
    for name, first in states[0].items():
        if not first.is_floating_point():
            raise ValueError(f"Le tenseur {name} n'est pas flottant (ex. BatchNorm).")
        average[name] = torch.zeros_like(first)
        for state, size in zip(states, sizes, strict=True):
            average[name].add_(state[name], alpha=size / total)
    return average


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Accuracy et perte moyennes par image sur le test, sans mise à jour."""
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
