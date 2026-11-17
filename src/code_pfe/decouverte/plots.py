"""Figures sauvegardées sur disque, utilisables aussi sans interface graphique."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_accuracy(metrics_path: Path, output: Path, title: str = "PathMNIST — FedAvg"):
    """Courbes individuelles + moyenne ± écart-type (ce n'est pas un IC)."""
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split in zip(axes, ("iid", "dirichlet"), strict=True):
        split_rows = [row for row in rows if row["split"] == split]
        seeds = sorted({int(row["seed"]) for row in split_rows})
        curves, reference_rounds = [], None
        for seed in seeds:
            run = sorted(
                (row for row in split_rows if int(row["seed"]) == seed),
                key=lambda row: int(row["round"]),
            )
            rounds = np.array([int(row["round"]) for row in run])
            accuracy = 100 * np.array([float(row["test_accuracy"]) for row in run])
            ax.plot(rounds, accuracy, alpha=0.6, linewidth=1, label=f"Graine {seed}")
            if reference_rounds is None:
                reference_rounds = rounds
            elif not np.array_equal(rounds, reference_rounds):
                raise ValueError(
                    "Les graines doivent avoir les mêmes tours pour être comparées."
                )
            curves.append(accuracy)
        if not curves:
            raise ValueError(f"Aucune métrique pour {split}.")
        mean = np.mean(curves, axis=0)
        ax.plot(reference_rounds, mean, color="black", linewidth=2, label="Moyenne")
        if len(curves) > 1:
            std = np.std(curves, axis=0, ddof=1)
            ax.fill_between(
                reference_rounds,
                np.maximum(0, mean - std),
                np.minimum(100, mean + std),
                color="gray",
                alpha=0.2,
                label="± 1 écart-type",
            )
        ax.set(
            title="IID" if split == "iid" else "Non-IID (Dirichlet)",
            xlabel="Tours de communication",
            ylim=(0, 100),
        )
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy sur le test (%)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_partitions(counts: np.ndarray, output: Path, title: str):
    """Rendre visible l'hétérogénéité des classes entre hôpitaux."""
    proportions = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(proportions, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set(xticks=range(9), xlabel="Classe PathMNIST", title=title)
    ax.set_yticks(
        range(len(counts)),
        [f"Hôpital {i} (n={row.sum()})" for i, row in enumerate(counts)],
    )
    fig.colorbar(image, ax=ax, label="Proportion de la classe dans l'hôpital")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Retracer un CSV FedAvg sans réentraîner."
    )
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_accuracy(args.metrics, args.output or args.metrics.with_suffix(".png"))


if __name__ == "__main__":
    main()
