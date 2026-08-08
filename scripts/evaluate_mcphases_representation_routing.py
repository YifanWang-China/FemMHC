#!/usr/bin/env python
"""Train-only representation selection for every mcPHASES downstream task."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from femmhc import FemMHCJointModel
from femmhc.data import McPhasesJointEmbeddingDataset
from femmhc.statistics import holm_adjust, paired_cluster_bootstrap


REPRESENTATION_SOURCES = (
    "general",
    "menstrual_domain",
    "task_route",
    "cycle",
)


@dataclass(frozen=True)
class StateTable:
    participants: np.ndarray
    targets: dict[str, np.ndarray]
    common_states: dict[str, np.ndarray]
    task_routes: dict[str, np.ndarray]

    def state_for(self, task_id: str, source: str) -> np.ndarray:
        if source == "task_route":
            return self.task_routes[task_id]
        return self.common_states[source]


@dataclass(frozen=True)
class ProbePrediction:
    hard: np.ndarray
    positive_probability: np.ndarray | None = None


def _load_model(artifact: dict[str, Any], device: torch.device) -> FemMHCJointModel:
    model = FemMHCJointModel(
        input_dim=int(artifact["input_dim"]),
        hidden_dim=int(artifact["hidden_dim"]),
        maximum_days=int(artifact["maximum_days"]),
        architecture=str(artifact["architecture"]),
        dropout=float(artifact.get("dropout", 0.0)),
        routing_initial_logit=float(artifact.get("routing_initial_logit", -2.0)),
    )
    model.load_state_dict(artifact["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def extract_state_table(
    artifact: dict[str, Any],
    dataset: McPhasesJointEmbeddingDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[StateTable, dict[str, Any]]:
    model = _load_model(artifact, device)
    task_specs = {
        task.task_id: task
        for task in model.task_heads.tasks
        if task.source == "mcphases"
    }
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    participants: list[np.ndarray] = []
    targets: dict[str, list[np.ndarray]] = {task_id: [] for task_id in task_specs}
    common: dict[str, list[np.ndarray]] = {
        "general": [],
        "cycle": [],
        "menstrual_domain": [],
    }
    routes: dict[str, list[np.ndarray]] = {task_id: [] for task_id in task_specs}
    for batch in loader:
        embeddings = batch["daily_embeddings"].to(device, non_blocking=True)
        present = batch["day_present"].to(device, non_blocking=True)
        states = model.state_encoder(embeddings, present)
        cache = model.task_heads._prepare_route_cache(states)
        common["general"].append(states.shared_state.float().cpu().numpy())
        common["cycle"].append(
            states.auxiliary["cycle_representation"].float().cpu().numpy()
        )
        common["menstrual_domain"].append(
            states.domain_states["menstrual"].float().cpu().numpy()
        )
        participants.append(np.asarray(batch["participant_id"], dtype=str))
        for task_id in task_specs:
            routed, _ = model.task_heads._route(task_id, states, cache)
            routes[task_id].append(routed.float().cpu().numpy())
            targets[task_id].append(batch["targets"][task_id].numpy())
    return (
        StateTable(
            participants=np.concatenate(participants),
            targets={key: np.concatenate(value) for key, value in targets.items()},
            common_states={key: np.concatenate(value) for key, value in common.items()},
            task_routes={key: np.concatenate(value) for key, value in routes.items()},
        ),
        task_specs,
    )


def observed_mask(kind: str, target: np.ndarray) -> np.ndarray:
    observed = np.isfinite(target)
    if kind != "regression":
        observed &= target >= 0
    return observed


def participant_weights(participants: np.ndarray) -> np.ndarray:
    counts = Counter(participants.tolist())
    values = np.asarray([1.0 / counts[item] for item in participants], dtype=np.float64)
    return values / values.mean()


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    participants: np.ndarray,
    x_evaluation: np.ndarray,
    *,
    kind: str,
    strength: float,
    seed: int,
) -> ProbePrediction:
    weights = participant_weights(participants)
    if kind == "regression":
        model = make_pipeline(StandardScaler(), Ridge(alpha=strength))
        model.fit(x_train, y_train, ridge__sample_weight=weights)
        prediction = model.predict(x_evaluation).astype(np.float64)
        return ProbePrediction(prediction)

    integer_target = y_train.astype(np.int64)
    unique = np.unique(integer_target)
    if len(unique) == 1:
        dummy = DummyClassifier(strategy="constant", constant=int(unique[0]))
        dummy.fit(x_train, integer_target, sample_weight=weights)
        hard = dummy.predict(x_evaluation).astype(np.int64)
        positive = np.full(len(hard), float(unique[0] == 1)) if kind == "binary" else None
        return ProbePrediction(hard, positive)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=strength,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    model.fit(x_train, integer_target, logisticregression__sample_weight=weights)
    hard = model.predict(x_evaluation).astype(np.int64)
    positive = None
    if kind == "binary":
        classes = model.named_steps["logisticregression"].classes_
        probabilities = model.predict_proba(x_evaluation)
        matches = np.flatnonzero(classes == 1)
        positive = (
            probabilities[:, matches[0]]
            if len(matches)
            else np.zeros(len(hard), dtype=np.float64)
        )
    return ProbePrediction(hard, positive)


def task_metric(
    target: np.ndarray,
    prediction: ProbePrediction,
    *,
    kind: str,
    classes: int | None,
    indices: np.ndarray | None = None,
) -> float | None:
    if indices is None:
        indices = np.arange(len(target))
    y = target[indices]
    hard = prediction.hard[indices]
    if kind == "binary":
        if prediction.positive_probability is None or len(np.unique(y)) < 2:
            return None
        return float(
            average_precision_score(y.astype(np.int64), prediction.positive_probability[indices])
        )
    if kind == "multiclass":
        return float(
            f1_score(
                y.astype(np.int64),
                hard.astype(np.int64),
                labels=np.arange(int(classes or 0)),
                average="macro",
                zero_division=0,
            )
        )
    return float(mean_absolute_error(y, hard))


def oriented(metric: float | None, *, kind: str) -> float:
    if metric is None or not np.isfinite(metric):
        return -np.inf
    return -float(metric) if kind in {"ordinal", "regression"} else float(metric)


def cross_validated_candidate(
    table: StateTable,
    task_id: str,
    task: Any,
    *,
    source: str,
    strength: float,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    target = table.targets[task_id]
    observed = observed_mask(task.kind, target)
    y = target[observed]
    participants = table.participants[observed]
    x = table.state_for(task_id, source)[observed]
    split_count = min(folds, len(np.unique(participants)))
    if split_count < 2:
        raise ValueError(f"{task_id} has fewer than two training participants")
    splitter = GroupKFold(n_splits=split_count)
    hard = np.empty(len(y), dtype=np.float64)
    positive = np.empty(len(y), dtype=np.float64) if task.kind == "binary" else None
    for fold, (train, held) in enumerate(
        splitter.split(x, y, groups=participants),
        start=1,
    ):
        prediction = fit_probe(
            x[train],
            y[train],
            participants[train],
            x[held],
            kind=task.kind,
            strength=strength,
            seed=seed + fold,
        )
        hard[held] = prediction.hard
        if positive is not None and prediction.positive_probability is not None:
            positive[held] = prediction.positive_probability
    prediction = ProbePrediction(hard, positive)
    metric = task_metric(
        y,
        prediction,
        kind=task.kind,
        classes=task.classes,
    )
    return {
        "representation_source": source,
        "strength": strength,
        "oof_primary_metric": metric,
        "oof_oriented_metric": oriented(metric, kind=task.kind),
    }


def fit_full_and_evaluate(
    train: StateTable,
    validation: StateTable,
    task_id: str,
    task: Any,
    *,
    source: str,
    strength: float,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, ProbePrediction]:
    train_target = train.targets[task_id]
    train_observed = observed_mask(task.kind, train_target)
    validation_target = validation.targets[task_id]
    validation_observed = observed_mask(task.kind, validation_target)
    if not bool(validation_observed.any()):
        empty = np.empty(0, dtype=np.float64)
        return (
            {
                "representation_source": source,
                "strength": strength,
                "validation_primary_metric": None,
                "validation_oriented_metric": None,
            },
            empty,
            np.empty(0, dtype=str),
            ProbePrediction(empty, empty if task.kind == "binary" else None),
        )
    prediction = fit_probe(
        train.state_for(task_id, source)[train_observed],
        train_target[train_observed],
        train.participants[train_observed],
        validation.state_for(task_id, source)[validation_observed],
        kind=task.kind,
        strength=strength,
        seed=seed,
    )
    target = validation_target[validation_observed]
    participants = validation.participants[validation_observed]
    metric = task_metric(
        target,
        prediction,
        kind=task.kind,
        classes=task.classes,
    )
    return (
        {
            "representation_source": source,
            "strength": strength,
            "validation_primary_metric": metric,
            "validation_oriented_metric": oriented(metric, kind=task.kind),
        },
        target,
        participants,
        prediction,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--strength", type=float, action="append")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument(
        "--mcphases-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    strengths = tuple(args.strength or (0.01, 0.1, 1.0, 10.0))
    if args.inner_folds < 2 or args.replicates <= 0 or any(value <= 0 for value in strengths):
        raise ValueError("cross-validation and probe settings must be positive")

    device = torch.device(args.device)
    artifact = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    dataset_arguments = {
        "processed_dir": args.mcphases_dir,
        "embeddings_path": args.mcphases_embeddings,
        "history_days": int(artifact["maximum_days"]),
        "minimum_history_days": args.minimum_history_days,
    }
    train_dataset = McPhasesJointEmbeddingDataset(split="train", **dataset_arguments)
    validation_dataset = McPhasesJointEmbeddingDataset(
        split="validation", **dataset_arguments
    )
    train, task_specs = extract_state_table(
        artifact,
        train_dataset,
        batch_size=args.batch_size,
        device=device,
    )
    validation, validation_specs = extract_state_table(
        artifact,
        validation_dataset,
        batch_size=args.batch_size,
        device=device,
    )
    if tuple(task_specs) != tuple(validation_specs):
        raise RuntimeError("train and validation task registries differ")

    fixed_manifest = None
    if args.selection_manifest is not None:
        fixed_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    bootstrap_p_values: dict[str, float] = {}
    for task_index, (task_id, task) in enumerate(task_specs.items()):
        target = train.targets[task_id]
        observed = observed_mask(task.kind, target)
        unique_participants = len(np.unique(train.participants[observed]))
        candidates: list[dict[str, Any]] = []
        if fixed_manifest is None:
            for source in REPRESENTATION_SOURCES:
                for strength in strengths:
                    candidates.append(
                        cross_validated_candidate(
                            train,
                            task_id,
                            task,
                            source=source,
                            strength=strength,
                            folds=args.inner_folds,
                            seed=args.seed + task_index * 100,
                        )
                    )
            best_by_source = {
                source: max(
                    (item for item in candidates if item["representation_source"] == source),
                    key=lambda item: float(item["oof_oriented_metric"]),
                )
                for source in REPRESENTATION_SOURCES
            }
            selected = max(
                (best_by_source[source] for source in REPRESENTATION_SOURCES),
                key=lambda item: float(item["oof_oriented_metric"]),
            )
        else:
            manifest_record = fixed_manifest["tasks"][task_id]
            best_by_source = manifest_record["selection"]["best_by_source"]
            selected = manifest_record["selection"]["selected"]

        validation_by_source: dict[str, dict[str, Any]] = {}
        predictions_by_source: dict[str, ProbePrediction] = {}
        validation_target = None
        validation_participants = None
        for source in REPRESENTATION_SOURCES:
            source_selection = best_by_source[source]
            result, target_values, participant_values, prediction = fit_full_and_evaluate(
                train,
                validation,
                task_id,
                task,
                source=source,
                strength=float(source_selection["strength"]),
                seed=args.seed + task_index * 100,
            )
            validation_by_source[source] = result
            predictions_by_source[source] = prediction
            if validation_target is None:
                validation_target = target_values
                validation_participants = participant_values
            elif not np.array_equal(validation_target, target_values):
                raise RuntimeError(f"validation targets changed across sources for {task_id}")

        selected_source = str(selected["representation_source"])
        general_result = validation_by_source["general"]
        selected_result = validation_by_source[selected_source]
        bootstrap_record = None
        if (
            selected_source != "general"
            and validation_target is not None
            and validation_participants is not None
            and len(validation_participants) > 0
        ):
            baseline_prediction = predictions_by_source["general"]
            candidate_prediction = predictions_by_source[selected_source]

            def score_pair(indices: np.ndarray) -> tuple[float | None, float | None]:
                return (
                    task_metric(
                        validation_target,
                        baseline_prediction,
                        kind=task.kind,
                        classes=task.classes,
                        indices=indices,
                    ),
                    task_metric(
                        validation_target,
                        candidate_prediction,
                        kind=task.kind,
                        classes=task.classes,
                        indices=indices,
                    ),
                )

            bootstrap = paired_cluster_bootstrap(
                validation_participants,
                score_pair,
                lower_is_better=task.kind in {"ordinal", "regression"},
                replicates=args.replicates,
                confidence=0.95,
                seed=args.seed + task_index,
                minimum_clusters=5,
            )
            bootstrap_record = {
                "estimate": bootstrap.estimate,
                "confidence_low": bootstrap.confidence_low,
                "confidence_high": bootstrap.confidence_high,
                "p_value_two_sided": bootstrap.p_value_two_sided,
                "valid_replicates": bootstrap.valid_replicates,
                "requested_replicates": bootstrap.requested_replicates,
                "clusters": bootstrap.clusters,
                "eligible": bootstrap.eligible,
                "reason": bootstrap.reason,
            }
            if bootstrap.eligible and bootstrap.p_value_two_sided is not None:
                bootstrap_p_values[task_id] = bootstrap.p_value_two_sided

        selected_oriented = selected_result["validation_oriented_metric"]
        general_oriented = general_result["validation_oriented_metric"]
        selected_delta = (
            None
            if selected_oriented is None or general_oriented is None
            else float(selected_oriented - general_oriented)
        )
        records[task_id] = {
            "domain": task.domain,
            "kind": task.kind,
            "classes": task.classes,
            "primary_metric": task.primary_metric,
            "train_samples": int(observed.sum()),
            "train_participants": unique_participants,
            "validation_samples": int(
                observed_mask(task.kind, validation.targets[task_id]).sum()
            ),
            "validation_participants": int(
                len(np.unique(validation_participants))
                if validation_participants is not None
                else 0
            ),
            "selection": {
                "candidates": candidates if fixed_manifest is None else None,
                "best_by_source": best_by_source,
                "selected": selected,
            },
            "validation_by_source": validation_by_source,
            "selected_validation_primary_metric": selected_result[
                "validation_primary_metric"
            ],
            "general_validation_primary_metric": general_result[
                "validation_primary_metric"
            ],
            "selected_oriented_delta_vs_general": selected_delta,
            "participant_bootstrap_selected_vs_general": bootstrap_record,
        }

    adjusted = holm_adjust(bootstrap_p_values)
    for task_id, value in adjusted.items():
        records[task_id]["participant_bootstrap_selected_vs_general"][
            "p_value_holm_selected_non_general_tasks"
        ] = value

    selected_counts = Counter(
        str(record["selection"]["selected"]["representation_source"])
        for record in records.values()
    )
    improved = sum(
        record["selected_oriented_delta_vs_general"] is not None
        and float(record["selected_oriented_delta_vs_general"]) > 0
        for record in records.values()
    )
    worsened = sum(
        record["selected_oriented_delta_vs_general"] is not None
        and float(record["selected_oriented_delta_vs_general"]) < 0
        for record in records.values()
    )
    evaluable = sum(
        record["selected_oriented_delta_vs_general"] is not None
        for record in records.values()
    )
    summary = {
        "format_version": 1,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "split": "validation",
        "selection_split": "train_only",
        "selection_manifest": (
            None
            if args.selection_manifest is None
            else str(args.selection_manifest.resolve())
        ),
        "representation_sources": REPRESENTATION_SOURCES,
        "strength_grid": strengths,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "test_used": False,
        "selected_source_counts": dict(selected_counts),
        "selected_vs_general_validation": {
            "improved_tasks": improved,
            "ties": evaluable - improved - worsened,
            "worsened_tasks": worsened,
            "not_evaluable_tasks": len(records) - evaluable,
        },
        "tasks": records,
        "limitations": [
            "Representation and regularization selection use training participants only.",
            "Validation contains only six participants for most daily tasks.",
            "The sealed test split is not used.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# mcPHASES任务级表示选择",
        "",
        "表示与正则化仅使用训练参与者选择；测试集未使用。",
        "",
        "| 任务 | 类型 | 训练选择表示 | 训练折外指标 | 验证指标 | 通用状态验证指标 | 定向变化 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for task_id, record in records.items():
        selected = record["selection"]["selected"]
        lines.append(
            "| {task} | {kind} | {source} | {oof:.4f} | {selected_metric} | "
            "{general_metric} | {delta} |".format(
                task=task_id,
                kind=record["kind"],
                source=selected["representation_source"],
                oof=float(selected["oof_primary_metric"]),
                selected_metric=(
                    "NA"
                    if record["selected_validation_primary_metric"] is None
                    else f"{float(record['selected_validation_primary_metric']):.4f}"
                ),
                general_metric=(
                    "NA"
                    if record["general_validation_primary_metric"] is None
                    else f"{float(record['general_validation_primary_metric']):.4f}"
                ),
                delta=(
                    "NA"
                    if record["selected_oriented_delta_vs_general"] is None
                    else f"{float(record['selected_oriented_delta_vs_general']):+.4f}"
                ),
            )
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "tasks": len(records),
                "selected_source_counts": dict(selected_counts),
                "selected_vs_general_validation": summary[
                    "selected_vs_general_validation"
                ],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
