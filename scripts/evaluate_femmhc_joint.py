#!/usr/bin/env python
"""Validation-first per-task evaluation for a FemMHC joint checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader

from femmhc import JOINT_TASKS, FemMHCJointModel, ProbabilisticOutput
from femmhc.data import (
    AffectiveJointEmbeddingDataset,
    HRVMentalJointEmbeddingDataset,
    McPhasesJointEmbeddingDataset,
    OpenMHCAuxiliaryEmbeddingDataset,
    PregnancyJointEmbeddingDataset,
)


def _datasets(args: argparse.Namespace) -> dict[str, Any]:
    split = args.split
    return {
        "openmhc": OpenMHCAuxiliaryEmbeddingDataset(
            args.openmhc_data_dir,
            args.openmhc_native_cache,
            args.openmhc_adapted_cache,
            split=split,
            history_days=args.openmhc_history_days,
        ),
        "mcphases": McPhasesJointEmbeddingDataset(
            args.mcphases_dir,
            args.mcphases_embeddings,
            split=split,
            history_days=args.maximum_days,
            minimum_history_days=args.minimum_history_days,
        ),
        "depress_fitbit": AffectiveJointEmbeddingDataset(
            "depress_fitbit",
            args.depress_dir,
            args.depress_embeddings,
            split=split,
            minimum_history_days=args.minimum_history_days,
        ),
        "inphrsym": AffectiveJointEmbeddingDataset(
            "inphrsym",
            args.inphrsym_dir,
            args.inphrsym_embeddings,
            split=split,
            minimum_history_days=args.minimum_history_days,
        ),
        "wearable_hrv_sleep": HRVMentalJointEmbeddingDataset(
            args.hrv_mental_dir,
            args.hrv_mental_embeddings,
            split=split,
        ),
        "pregnancy_ga_clock": PregnancyJointEmbeddingDataset(
            args.pregnancy_dir,
            args.pregnancy_embeddings,
            split=split,
        ),
    }


def _safe_correlation(function, target: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(target) < 3 or np.std(target) < 1e-12 or np.std(prediction) < 1e-12:
        return None
    value = float(function(target, prediction).statistic)
    return value if np.isfinite(value) else None


def _binary_ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if selected.any():
            result += selected.mean() * abs(
                float(target[selected].mean()) - float(probability[selected].mean())
            )
    return float(result)


def _metrics(
    *,
    kind: str,
    target: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray | None,
) -> dict[str, float | None]:
    if kind == "regression":
        return {
            "mae": float(mean_absolute_error(target, prediction)),
            "rmse": float(mean_squared_error(target, prediction) ** 0.5),
            "pearson_r": _safe_correlation(pearsonr, target, prediction),
            "spearman_r": _safe_correlation(spearmanr, target, prediction),
        }
    labels = target.astype(int)
    predicted_class = prediction.astype(int)
    if kind == "binary":
        positive = probabilities[:, 1]
        both = len(np.unique(labels)) == 2
        return {
            "auprc": float(average_precision_score(labels, positive)) if both else None,
            "auroc": float(roc_auc_score(labels, positive)) if both else None,
            "balanced_accuracy": (
                float(balanced_accuracy_score(labels, predicted_class)) if both else None
            ),
            "brier": float(np.mean((positive - labels) ** 2)),
            "ece": _binary_ece(labels, positive),
        }
    if kind == "multiclass":
        return {
            "macro_f1": float(f1_score(labels, predicted_class, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted_class)),
        }
    classes = np.arange(probabilities.shape[1], dtype=np.float64)
    expected = probabilities @ classes
    return {
        "mae": float(mean_absolute_error(labels, expected)),
        "macro_f1": float(f1_score(labels, predicted_class, average="macro")),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(labels, predicted_class, weights="quadratic")
        ),
        "spearman_r": _safe_correlation(spearmanr, labels, expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--openmhc-data-dir", type=Path, default=Path("datasets/openmhc-xs"))
    parser.add_argument("--openmhc-native-cache", type=Path, default=Path("artifacts/embeddings/openmhc-xs/openmhc-lsm2"))
    parser.add_argument("--openmhc-adapted-cache", type=Path, default=Path("artifacts/embeddings/openmhc-xs/femmhc-stage1-v4"))
    parser.add_argument("--openmhc-history-days", type=int, default=7)
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument("--mcphases-embeddings", type=Path, default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"))
    parser.add_argument("--depress-dir", type=Path, default=Path("processed/depress_fitbit"))
    parser.add_argument("--depress-embeddings", type=Path, default=Path("artifacts/embeddings/depress-fitbit-affective-dynamics-step100.npz"))
    parser.add_argument("--inphrsym-dir", type=Path, default=Path("processed/inphrsym"))
    parser.add_argument("--inphrsym-embeddings", type=Path, default=Path("artifacts/embeddings/inphrsym-affective-dynamics-step100.npz"))
    parser.add_argument("--hrv-mental-dir", type=Path, default=Path("processed/wearable_hrv_mental_female"))
    parser.add_argument("--hrv-mental-embeddings", type=Path, default=Path("artifacts/embeddings/hrv-mental-female/femmhc-stage1-seed42.npz"))
    parser.add_argument("--pregnancy-dir", type=Path, default=Path("processed/pregnancy_ga_clock_official"))
    parser.add_argument("--pregnancy-embeddings", type=Path, default=Path("artifacts/embeddings/pregnancy-ga-official/progression-v4-best.npz"))
    parser.add_argument("--maximum-days", type=int, default=60)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("test evaluation is locked; pass --allow-test only for the final run")

    device = torch.device(args.device)
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FemMHCJointModel(
        input_dim=int(artifact["input_dim"]),
        hidden_dim=int(artifact["hidden_dim"]),
        maximum_days=int(artifact["maximum_days"]),
        architecture=str(artifact.get("architecture", "full")),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model = model.to(device).eval()
    statistics = artifact["regression_target_statistics"]
    task_by_id = {task.task_id: task for task in JOINT_TASKS if task.trainable}
    collected: dict[str, dict[str, list[Any]]] = {}

    with torch.inference_mode():
        for cohort, dataset in _datasets(args).items():
            loader = DataLoader(
                dataset,
                batch_size=min(args.batch_size, len(dataset)),
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            for batch in loader:
                task_ids = tuple(batch["targets"])
                embeddings = batch["daily_embeddings"].to(device, non_blocking=True)
                present = batch["day_present"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model(embeddings, present, task_ids=task_ids)
                for task_id, target_tensor in batch["targets"].items():
                    task = task_by_id[task_id]
                    target = target_tensor.numpy()
                    observed = np.isfinite(target)
                    if task.kind != "regression":
                        observed &= target >= 0
                    if not observed.any():
                        continue
                    prediction_output = output.predictions[task_id]
                    record = collected.setdefault(
                        task_id,
                        {"target": [], "prediction": [], "probabilities": [], "participant": [], "cohort": cohort},
                    )
                    if isinstance(prediction_output, ProbabilisticOutput):
                        probabilities = prediction_output.probabilities.float().cpu().numpy()
                        prediction = probabilities.argmax(axis=1)
                        record["probabilities"].append(probabilities[observed])
                    else:
                        prediction = prediction_output.float().cpu().numpy()
                        if task_id in statistics:
                            prediction = prediction * statistics[task_id]["std"] + statistics[task_id]["mean"]
                    record["target"].append(target[observed])
                    record["prediction"].append(prediction[observed])
                    record["participant"].extend(
                        np.asarray(batch["participant_id"], dtype=object)[observed].tolist()
                    )

    rows = []
    detailed = {}
    for task_id, record in sorted(collected.items()):
        task = task_by_id[task_id]
        target = np.concatenate(record["target"])
        prediction = np.concatenate(record["prediction"])
        probabilities = (
            np.concatenate(record["probabilities"])
            if record["probabilities"]
            else None
        )
        metrics = _metrics(
            kind=task.kind,
            target=target,
            prediction=prediction,
            probabilities=probabilities,
        )
        detailed[task_id] = {
            "source": task.source,
            "domain": task.domain,
            "kind": task.kind,
            "primary_metric": task.primary_metric,
            "samples": int(len(target)),
            "participants": int(len(set(record["participant"]))),
            "metrics": metrics,
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "task_id": task_id,
                    "source": task.source,
                    "domain": task.domain,
                    "kind": task.kind,
                    "samples": len(target),
                    "participants": len(set(record["participant"])),
                    "metric": metric,
                    "value": value,
                    "is_primary": metric == task.primary_metric
                    or (task.primary_metric == "mae_weeks" and metric == "mae"),
                }
            )

    report = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(artifact["step"]),
        "split": args.split,
        "test_lock_explicitly_released": bool(args.allow_test),
        "evaluated_tasks": len(detailed),
        "tasks": detailed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "per_task_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "split": args.split,
                "evaluated_tasks": len(detailed),
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
