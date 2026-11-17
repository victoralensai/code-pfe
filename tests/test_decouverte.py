"""Vérifier la mécanique FL, sans téléchargement ni entraînement complet."""

import csv
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from code_pfe.decouverte.data import class_counts, make_partitions
from code_pfe.decouverte.fedavg import evaluate, train_client, weighted_average


@pytest.mark.parametrize("split", ["iid", "dirichlet"])
def test_partitions_cover_train_once_and_are_reproducible(split):
    labels = np.repeat(np.arange(9), 1000)
    partitions = make_partitions(labels, 10, split, seed=0)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(partitions)), np.arange(len(labels))
    )
    assert all(len(indices) > 0 for indices in partitions)
    for first, second in zip(
        partitions, make_partitions(labels, 10, split, 0), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    counts = class_counts(labels, partitions)
    np.testing.assert_array_equal(counts.sum(axis=0), np.bincount(labels))
    proportions = counts / counts.sum(axis=1, keepdims=True)
    # Un jeu équilibré assez grand distingue nettement les deux répartitions.
    if split == "iid":
        assert max(map(len, partitions)) - min(map(len, partitions)) <= 1
        assert proportions.std(axis=0).mean() < 0.02
    else:
        assert proportions.std(axis=0).mean() > 0.05


def test_fedavg_matches_manual_local_sgd_and_preserves_global_model():
    # Deux classes, un poids par classe, poids initiaux nuls.
    # Pour x=1,y=0 : gradient [-0.5, +0.5], donc poids [+0.1, -0.1].
    # Pour x=2,y=1 : gradient [+1.0, -1.0], donc poids [-0.2, +0.2].
    model = nn.Linear(1, 2, bias=False)
    nn.init.zeros_(model.weight)
    client_a = DataLoader(TensorDataset(torch.tensor([[1.0]]), torch.tensor([[0]])))
    client_b = DataLoader(
        TensorDataset(torch.full((3, 1), 2.0), torch.ones(3, 1).long()), batch_size=3
    )
    device = torch.device("cpu")
    first = train_client(model, client_a, 1, 0.2, device)
    second = train_client(model, client_b, 1, 0.2, device)
    torch.testing.assert_close(model.weight, torch.zeros(2, 1))
    torch.testing.assert_close(first["weight"], torch.tensor([[0.1], [-0.1]]))
    torch.testing.assert_close(second["weight"], torch.tensor([[-0.2], [0.2]]))
    result = weighted_average([first, second], [1, 3])
    torch.testing.assert_close(result["weight"], torch.tensor([[-0.125], [0.125]]))
    # L'agrégation ne doit pas modifier les modèles reçus.
    torch.testing.assert_close(first["weight"], torch.tensor([[0.1], [-0.1]]))


def test_evaluate_counts_images_including_incomplete_batch():
    logits = torch.tensor([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    labels = torch.tensor([[0], [1], [1]])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    accuracy, loss = evaluate(nn.Identity(), loader, torch.device("cpu"))
    assert accuracy == pytest.approx(2 / 3)
    assert loss == pytest.approx(
        nn.functional.cross_entropy(logits, labels.flatten()).item()
    )


@pytest.mark.parametrize("states,sizes", [([], []), ([{"w": torch.ones(1)}], [0])])
def test_aggregation_rejects_invalid_weights(states, sizes):
    with pytest.raises(ValueError):
        weighted_average(states, sizes)


def test_full_pipeline_three_seeds_with_synthetic_images(tmp_path, monkeypatch):
    from code_pfe.decouverte import experiment

    generator = torch.Generator().manual_seed(42)
    images = torch.rand(40, 3, 28, 28, generator=generator) * 2 - 1
    labels = torch.arange(40).remainder(9)
    train = TensorDataset(images, labels[:, None])
    test = TensorDataset(images[:9], labels[:9, None])
    monkeypatch.setattr(
        experiment,
        "load_pathmnist",
        lambda *args: (train, test, labels.numpy(), np.arange(40)),
    )
    output = tmp_path / "experiment"
    monkeypatch.setattr(
        sys, "argv", ["experiment", "--rounds", "1", "--output", str(output)]
    )
    experiment.main()
    with (output / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 2 * 2  # Trois graines × deux scénarios × tours 0 et 1.
    assert (output / "accuracy.png").stat().st_size > 0
    for seed in range(3):
        initial = [
            row for row in rows if int(row["seed"]) == seed and row["round"] == "0"
        ]
        assert initial[0]["test_loss"] == initial[1]["test_loss"]
        for split in ("iid", "dirichlet"):
            run_dir = output / f"{split}_seed_{seed}"
            assert (run_dir / "model.pt").exists()
            assert (run_dir / "partitions.png").stat().st_size > 0
            with np.load(run_dir / "partitions.npz") as partitions:
                assert len(partitions.files) == 10
                np.testing.assert_array_equal(
                    np.sort(np.concatenate(list(partitions.values()))), np.arange(40)
                )
