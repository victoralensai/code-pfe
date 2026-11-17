"""Comparer IID / Dirichlet sur trois graines : python -m code_pfe.decouverte.experiment."""

import argparse
import csv
import json
import os
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import class_counts, load_pathmnist, make_partitions
from .fedavg import SmallCNN, evaluate, train_client, weighted_average
from .plots import plot_accuracy, plot_partitions


def seed_everything(seed: int):
    """Reproductibilité sur une même machine et les mêmes versions logicielles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def run_scenario(
    train, test, labels, original_indices, split, seed, args, writer, handle
):
    # Réinitialiser AVANT de construire le modèle : même point de départ IID/non-IID.
    seed_everything(seed)
    device = torch.device(args.device)
    model = SmallCNN().to(device)
    partitions = make_partitions(labels, args.clients, split, seed, args.alpha)
    run_dir = args.output / f"{split}_seed_{seed}"
    run_dir.mkdir()
    counts = class_counts(labels, partitions)
    # Les indices sauvegardés pointent vers le train OFFICIEL, même en mode rapide.
    np.savez_compressed(
        run_dir / "partitions.npz",
        **{
            f"client_{i}": original_indices[indices]
            for i, indices in enumerate(partitions)
        },
    )
    (run_dir / "class_counts.json").write_text(
        json.dumps(counts.tolist(), indent=2) + "\n"
    )
    plot_partitions(counts, run_dir / "partitions.png", f"{split}, graine {seed}")

    # Un générateur propre à chaque client : le mélange des mini-batches ne dépend
    # pas de la taille des autres hôpitaux ni des appels au générateur du modèle.
    loaders = [
        DataLoader(
            Subset(train, indices.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed * args.clients + i),
            num_workers=0,
            drop_last=False,
        )
        for i, indices in enumerate(partitions)
    ]
    test_loader = DataLoader(
        test,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    sizes = [len(indices) for indices in partitions]
    start = perf_counter()
    for round_id in range(args.rounds + 1):
        if round_id > 0:
            # 1. Diffusion du modèle global ; 2. apprentissages locaux indépendants.
            states = [
                train_client(model, loader, args.local_epochs, args.lr, device)
                for loader in loaders
            ]
            # 3. Agrégation par le serveur ; le serveur ne reçoit pas les images.
            model.load_state_dict(weighted_average(states, sizes))
        # Tour 0 : le modèle non entraîné. Puis une évaluation par communication.
        accuracy, loss = evaluate(model, test_loader, device)
        writer.writerow(
            {
                "split": split,
                "seed": seed,
                "round": round_id,
                "test_accuracy": accuracy,
                "test_loss": loss,
                "elapsed_seconds": perf_counter() - start,
            }
        )
        handle.flush()  # Conserver les mesures même si un entraînement est interrompu.
        print(
            f"{split:9s} | graine {seed} | tour {round_id:3d}/{args.rounds} "
            f"| test accuracy {accuracy:.2%} | perte {loss:.4f}",
            flush=True,
        )
    torch.save(model.state_dict(), run_dir / "model.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/medmnist"))
    parser.add_argument("--output", type=Path, default=Path("outputs/decouverte"))
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--threads", type=int, default=2, help="Threads PyTorch sur CPU."
    )
    parser.add_argument(
        "--max-train", type=int, help="Sous-ensemble rapide ; test inchangé."
    )
    parser.add_argument("--subset-seed", type=int, default=12345)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    for name in ("clients", "rounds", "local_epochs", "batch_size", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"{name} doit être strictement positif.")
    for name in ("lr", "alpha"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} doit être strictement positif et fini.")
    if len(set(args.seeds)) != len(args.seeds) or any(
        not 0 <= s < 2**32 for s in args.seeds
    ):
        parser.error(
            "Les graines doivent être distinctes et comprises entre 0 et 2**32-1."
        )
    if not 0 <= args.subset_seed < 2**32:
        parser.error("subset-seed doit être compris entre 0 et 2**32-1.")
    if args.max_train is not None and args.max_train < args.clients:
        parser.error("max-train doit être au moins égal au nombre de clients.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponible : utiliser --device cpu.")
    # Ne pas écraser une expérience ou mélanger des mesures de configurations différentes.
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Le dossier de sortie doit être vide ou nouveau (--output).")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    train, test, labels, original_indices = load_pathmnist(
        args.data_dir, args.max_train, args.subset_seed
    )
    if len(train) < args.clients:
        parser.error("Il faut au moins une image d'entraînement par client.")
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    metadata = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("torch", "torchvision", "medmnist", "numpy", "matplotlib")
        },
        "git_commit": git.stdout.strip() or None,
        "git_dirty": bool(git_status.stdout.strip())
        if git_status.returncode == 0
        else None,
        "device": str(torch.cuda.get_device_name(0))
        if args.device == "cuda"
        else platform.machine(),
        "train_size": len(train),
        "test_size": len(test),
        "quick_run": args.max_train is not None,
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metrics_path = args.output / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "seed",
                "round",
                "test_accuracy",
                "test_loss",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for seed in args.seeds:
            for split in ("iid", "dirichlet"):
                run_scenario(
                    train,
                    test,
                    labels,
                    original_indices,
                    split,
                    seed,
                    args,
                    writer,
                    handle,
                )
    title = f"PathMNIST — FedAvg, Dirichlet α={args.alpha}"
    if args.max_train is not None:
        title += f" — essai réduit ({len(train)} images train)"
    plot_accuracy(metrics_path, args.output / "accuracy.png", title)
    print(f"Résultats : {args.output.resolve()}")


if __name__ == "__main__":
    main()
