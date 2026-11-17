"""Chargement de PathMNIST et répartition des images entre hôpitaux."""

from pathlib import Path

import numpy as np
from torch.utils.data import Subset


def load_pathmnist(root: Path, max_train: int | None, subset_seed: int = 12345):
    """Garder les splits officiels ; ne sous-échantillonner que le train.

    Le sous-ensemble éventuel est commun aux deux scénarios et aux trois graines.
    Le test reste toujours complet. Le split validation est réservé pour plus tard.
    """
    from medmnist import PathMNIST
    from torchvision import transforms

    root.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    train = PathMNIST(root=str(root), split="train", download=True, transform=transform)
    test = PathMNIST(root=str(root), split="test", download=True, transform=transform)
    indices = np.arange(len(train))
    if max_train is not None:
        if not 1 <= max_train <= len(train):
            raise ValueError(f"max_train doit être compris entre 1 et {len(train)}.")
        indices = np.sort(
            np.random.default_rng(subset_seed).choice(indices, max_train, replace=False)
        )
    labels = train.labels.reshape(-1)[indices]
    return Subset(train, indices.tolist()), test, labels, indices


def make_partitions(
    labels: np.ndarray,
    num_clients: int,
    split: str,
    seed: int,
    alpha: float = 0.5,
) -> list[np.ndarray]:
    """Retourner les indices locaux : chaque image appartient à un seul hôpital.

    IID : permutation uniforme, puis lots de tailles presque égales.
    Dirichlet : pour chaque classe, tirer ses proportions entre les hôpitaux.
    Ce second cas fait varier les proportions de classes ET les effectifs.
    """
    labels = np.asarray(labels).reshape(-1)
    if not 1 <= num_clients <= len(labels):
        raise ValueError("Il faut au moins une image par hôpital.")
    if split not in {"iid", "dirichlet"}:
        raise ValueError("split doit être 'iid' ou 'dirichlet'.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha doit être strictement positif et fini.")
    rng = np.random.default_rng(seed)
    if split == "iid":
        return list(np.array_split(rng.permutation(len(labels)), num_clients))

    # On retire toute la partition si un hôpital est vide, sans supprimer d'image.
    # Avec PathMNIST et alpha=0.5, cette précaution est rarement nécessaire.
    for _ in range(1000):
        clients: list[list[int]] = [[] for _ in range(num_clients)]
        for label in np.unique(labels):
            indices = rng.permutation(np.flatnonzero(labels == label))
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            sizes = rng.multinomial(len(indices), proportions)
            chunks = np.split(indices, np.cumsum(sizes)[:-1])
            for client, chunk in zip(clients, chunks, strict=True):
                client.extend(chunk.tolist())
        if all(clients):
            return [rng.permutation(client) for client in clients]
    raise RuntimeError("Hôpital vide après 1000 essais : augmenter le train ou alpha.")


def class_counts(labels: np.ndarray, partitions: list[np.ndarray]) -> np.ndarray:
    """Une ligne par hôpital, une colonne par classe PathMNIST (0 à 8)."""
    return np.stack(
        [np.bincount(labels[indices], minlength=9) for indices in partitions]
    )
