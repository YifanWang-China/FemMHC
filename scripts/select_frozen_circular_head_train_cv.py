#!/usr/bin/env python
"""Select frozen circular-head hyperparameters using training participants only."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from femmhc import FemMHCJointModel, cyclic_phase_geometry_loss
from femmhc.data import McPhasesJointEmbeddingDataset


PHASE_TASK_ID = "mcphases/cycle_phase"
HEAD_FAMILIES = (
    "circular_fixed",
    "circular_permuted",
    "learnable_prototype",
    "bottleneck_softmax",
    "linear_matched",
    "linear_softmax",
)
REPRESENTATION_SOURCES = (
    "general",
    "cycle",
    "menstrual_domain",
    "cycle_task_route",
)


class CircularProjector(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        initialization_seed: int,
        prototype_order: tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 200_003)
            self.projector = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2),
            )
        base = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        self.register_buffer("prototypes", base[list(prototype_order)])
        self.supports_geometry = True

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vector = self.projector(values)
        direction = F.normalize(vector, dim=-1)
        logits = 4.0 * direction @ self.prototypes.to(direction.dtype).T
        return vector, logits


class LearnablePrototypeProjector(nn.Module):
    def __init__(self, hidden_dim: int, *, initialization_seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 200_003)
            self.projector = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2),
            )
        self.prototypes = nn.Parameter(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        )
        self.supports_geometry = False

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vector = self.projector(values)
        direction = F.normalize(vector, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return vector, 4.0 * direction @ prototypes.T


class BottleneckSoftmaxProjector(nn.Module):
    def __init__(self, hidden_dim: int, *, initialization_seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 200_003)
            self.projector = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2),
            )
            self.classifier = nn.Linear(2, 4)
        self.supports_geometry = False

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vector = self.projector(values)
        return vector, self.classifier(vector)


class LinearSoftmaxProjector(nn.Module):
    def __init__(self, hidden_dim: int, *, initialization_seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 200_003)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 4),
            )
        self.supports_geometry = False

    def forward(self, values: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, self.classifier(values)


class MatchedLinearProjector(nn.Module):
    """Direct four-class projection with 516 parameters at hidden_dim=128."""

    def __init__(self, hidden_dim: int, *, initialization_seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 200_003)
            self.classifier = nn.Linear(hidden_dim, 4)
        self.supports_geometry = False

    def forward(self, values: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, self.classifier(values)


def build_phase_head(
    hidden_dim: int,
    *,
    head_family: str,
    initialization_seed: int,
) -> nn.Module:
    if head_family == "circular_fixed":
        return CircularProjector(hidden_dim, initialization_seed=initialization_seed)
    if head_family == "circular_permuted":
        return CircularProjector(
            hidden_dim,
            initialization_seed=initialization_seed,
            prototype_order=(0, 2, 1, 3),
        )
    if head_family == "learnable_prototype":
        return LearnablePrototypeProjector(
            hidden_dim,
            initialization_seed=initialization_seed,
        )
    if head_family == "bottleneck_softmax":
        return BottleneckSoftmaxProjector(
            hidden_dim,
            initialization_seed=initialization_seed,
        )
    if head_family == "linear_softmax":
        return LinearSoftmaxProjector(
            hidden_dim,
            initialization_seed=initialization_seed,
        )
    if head_family == "linear_matched":
        return MatchedLinearProjector(
            hidden_dim,
            initialization_seed=initialization_seed,
        )
    raise ValueError(f"unknown head family: {head_family}")


@torch.no_grad()
def extract_phase_representations(
    artifact: dict[str, Any],
    dataset: McPhasesJointEmbeddingDataset,
    *,
    batch_size: int,
    device: torch.device,
    representation_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if representation_source not in REPRESENTATION_SOURCES:
        raise ValueError(f"unknown representation source: {representation_source}")
    architecture = str(artifact.get("architecture", ""))
    if not architecture.startswith("dual_path"):
        raise ValueError("base checkpoint must use a dual_path architecture")
    model = FemMHCJointModel(
        input_dim=int(artifact["input_dim"]),
        hidden_dim=int(artifact["hidden_dim"]),
        maximum_days=int(artifact["maximum_days"]),
        architecture=architecture,
        dropout=float(artifact.get("dropout", 0.0)),
        routing_initial_logit=float(artifact.get("routing_initial_logit", -2.0)),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model = model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    representations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    participants: list[np.ndarray] = []
    for batch in loader:
        embeddings = batch["daily_embeddings"].to(device, non_blocking=True)
        present = batch["day_present"].to(device, non_blocking=True)
        state = model.state_encoder(embeddings, present)
        if representation_source == "general":
            representation = state.shared_state
        elif representation_source == "cycle":
            representation = state.auxiliary["cycle_representation"]
        elif representation_source == "menstrual_domain":
            representation = state.domain_states["menstrual"]
        else:
            cache = model.task_heads._prepare_route_cache(state)
            representation, _ = model.task_heads._route(
                PHASE_TASK_ID,
                state,
                cache,
            )
        values = representation.float().cpu().numpy()
        target = batch["targets"][PHASE_TASK_ID].numpy()
        observed = np.isfinite(target) & (target >= 0)
        if observed.any():
            representations.append(values[observed])
            targets.append(target[observed].astype(np.int64))
            participant = np.asarray(batch["participant_id"], dtype=str)
            participants.append(participant[observed])
    if not representations:
        raise RuntimeError("training split contains no observed cycle-phase labels")
    return (
        np.concatenate(representations),
        np.concatenate(targets),
        np.concatenate(participants),
    )


def extract_cycle_representations(
    artifact: dict[str, Any],
    dataset: McPhasesJointEmbeddingDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for the CycleSSM representation."""

    return extract_phase_representations(
        artifact,
        dataset,
        batch_size=batch_size,
        device=device,
        representation_source="cycle",
    )


def fit_phase_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    participant_train: np.ndarray,
    *,
    hidden_dim: int,
    learning_rate: float,
    steps: int,
    batch_size: int,
    geometry_weight: float,
    seed: int,
    device: torch.device,
    head_family: str = "circular_fixed",
) -> nn.Module:
    model = build_phase_head(
        hidden_dim,
        head_family=head_family,
        initialization_seed=seed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    counts = Counter(participant_train.tolist())
    probabilities = np.asarray(
        [1.0 / counts[item] for item in participant_train],
        dtype=np.float64,
    )
    probabilities /= probabilities.sum()
    generator = np.random.default_rng(seed)
    x_tensor = torch.from_numpy(x_train.astype(np.float32, copy=False)).to(device)
    y_tensor = torch.from_numpy(y_train.astype(np.int64, copy=False)).to(device)
    for _ in range(steps):
        indices = generator.choice(
            len(y_train),
            size=min(batch_size, len(y_train)),
            replace=True,
            p=probabilities,
        )
        selected = torch.as_tensor(indices, dtype=torch.long, device=device)
        vector, logits = model(x_tensor[selected])
        target = y_tensor[selected]
        classification = F.cross_entropy(logits, target)
        if geometry_weight:
            if not bool(model.supports_geometry) or vector is None:
                raise ValueError(f"{head_family} does not support geometry loss")
            expected = model.prototypes[target]
            direction = 1.0 - (
                F.normalize(vector, dim=-1) * F.normalize(expected, dim=-1)
            ).sum(dim=-1)
            unit_norm = (vector.norm(dim=-1) - 1.0).square()
            geometry = direction.mean() + 0.1 * unit_norm.mean()
        else:
            geometry = classification.new_zeros(())
        loss = 0.5 * (classification + geometry_weight * geometry)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def predict_phase_head(
    model: nn.Module,
    values: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(values.astype(np.float32, copy=False)).to(device)
    _, logits = model(tensor)
    return logits.argmax(dim=-1).cpu().numpy()


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    participant_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    hidden_dim: int,
    learning_rate: float,
    steps: int,
    batch_size: int,
    geometry_weight: float,
    seed: int,
    device: torch.device,
    head_family: str = "circular_fixed",
) -> np.ndarray:
    model = fit_phase_head(
        x_train,
        y_train,
        participant_train,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        steps=steps,
        batch_size=batch_size,
        geometry_weight=geometry_weight,
        seed=seed,
        device=device,
        head_family=head_family,
    )
    with torch.inference_mode():
        return predict_phase_head(model, x_validation, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-family", choices=HEAD_FAMILIES, default="circular_fixed")
    parser.add_argument(
        "--representation-source",
        choices=REPRESENTATION_SOURCES,
        default="cycle",
    )
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument(
        "--mcphases-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"),
    )
    parser.add_argument("--learning-rate", action="append", type=float)
    parser.add_argument("--steps", action="append", type=int)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--geometry-weight", action="append", type=float)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    learning_rates = tuple(args.learning_rate or (0.003, 0.01))
    step_grid = tuple(args.steps or (75, 125, 225, 300))
    geometry_weights = tuple(args.geometry_weight or (0.0, 0.1, 0.25))
    if args.head_family not in {"circular_fixed", "circular_permuted"}:
        geometry_weights = (0.0,)
    if any(value <= 0 for value in learning_rates) or any(value <= 0 for value in step_grid):
        raise ValueError("learning rates and steps must be positive")
    if (
        args.inner_folds < 2
        or args.batch_size <= 0
        or any(value < 0 for value in geometry_weights)
    ):
        raise ValueError("invalid cross-validation or training settings")

    device = torch.device(args.device)
    artifact = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    dataset = McPhasesJointEmbeddingDataset(
        args.mcphases_dir,
        args.mcphases_embeddings,
        split="train",
        history_days=int(artifact["maximum_days"]),
        minimum_history_days=args.minimum_history_days,
    )
    x, y, participants = extract_phase_representations(
        artifact,
        dataset,
        batch_size=128,
        device=device,
        representation_source=args.representation_source,
    )
    unique_participants = np.unique(participants)
    folds = min(args.inner_folds, len(unique_participants))
    splitter = GroupKFold(n_splits=folds)
    split_indices = list(splitter.split(x, y, groups=participants))
    candidates: list[dict[str, Any]] = []
    for geometry_weight in geometry_weights:
        for learning_rate in learning_rates:
            for steps in step_grid:
                prediction = np.full(len(y), -1, dtype=np.int64)
                fold_scores: list[float] = []
                for fold_index, (train, validation) in enumerate(split_indices):
                    fold_prediction = fit_predict(
                        x[train],
                        y[train],
                        participants[train],
                        x[validation],
                        hidden_dim=int(artifact["hidden_dim"]),
                        learning_rate=learning_rate,
                        steps=steps,
                        batch_size=args.batch_size,
                        geometry_weight=geometry_weight,
                        seed=args.seed + fold_index * 101,
                        device=device,
                        head_family=args.head_family,
                    )
                    prediction[validation] = fold_prediction
                    fold_scores.append(
                        float(
                            f1_score(
                                y[validation],
                                fold_prediction,
                                labels=np.arange(4),
                                average="macro",
                                zero_division=0,
                            )
                        )
                    )
                pooled = float(
                    f1_score(
                        y,
                        prediction,
                        labels=np.arange(4),
                        average="macro",
                        zero_division=0,
                    )
                )
                candidates.append(
                    {
                        "geometry_weight": geometry_weight,
                        "head_family": args.head_family,
                        "learning_rate": learning_rate,
                        "steps": steps,
                        "pooled_macro_f1": pooled,
                        "fold_macro_f1": fold_scores,
                        "mean_fold_macro_f1": float(np.mean(fold_scores)),
                    }
                )
                print(json.dumps(candidates[-1]), flush=True)
    selected = max(
        candidates,
        key=lambda item: (
            item["pooled_macro_f1"],
            item["mean_fold_macro_f1"],
            -item["steps"],
            -item["learning_rate"],
            -item["geometry_weight"],
        ),
    )
    summary = {
        "format_version": 1,
        "selection_split": "train_only",
        "selection_protocol": "participant_group_kfold",
        "head_family": args.head_family,
        "representation_source": args.representation_source,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "participants": int(len(unique_participants)),
        "samples": int(len(y)),
        "inner_folds": folds,
        "geometry_weight_grid": geometry_weights,
        "trainable_parameters": int(
            sum(
                value.numel()
                for value in build_phase_head(
                    int(artifact["hidden_dim"]),
                    head_family=args.head_family,
                    initialization_seed=args.seed,
                ).parameters()
            )
        ),
        "seed": args.seed,
        "candidates": candidates,
        "selected": selected,
        "validation_participants_used": False,
        "test_participants_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
