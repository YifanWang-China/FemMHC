#!/usr/bin/env python
"""Train-only temperature calibration for coherent FemMHC probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

from femmhc import FemMHCJointModel, JOINT_TASKS, ProbabilisticOutput
from femmhc.data import McPhasesJointEmbeddingDataset
from femmhc.statistics import holm_adjust, paired_cluster_bootstrap


ONSET_24H = "mcphases/menstrual_onset_24h"
ONSET_72H = "mcphases/menstrual_onset_72h"
PHASE = "mcphases/cycle_phase"


def _parse_checkpoint(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be SEED=PATH")
    raw_seed, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return int(raw_seed), path


def softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def categorical_nll(
    logits: np.ndarray,
    target: np.ndarray,
    temperature: float,
) -> float:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    labels = np.asarray(target, dtype=np.int64)
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, target: np.ndarray) -> float:
    result = minimize_scalar(
        lambda log_temperature: categorical_nll(
            logits,
            target,
            float(np.exp(log_temperature)),
        ),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def binary_ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if selected.any():
            value += float(selected.mean()) * abs(
                float(target[selected].mean()) - float(probability[selected].mean())
            )
    return float(value)


def multiclass_ece(target: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == target).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if selected.any():
            value += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(value)


def onset_probabilities(bin_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(bin_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("onset bin probabilities must have shape (samples, 3)")
    return values[:, 0], values[:, 0] + values[:, 1]


def onset_bins(target_24h: np.ndarray, target_72h: np.ndarray) -> np.ndarray:
    y24 = np.asarray(target_24h, dtype=np.int64)
    y72 = np.asarray(target_72h, dtype=np.int64)
    if np.any(y24 > y72):
        raise ValueError("observed onset labels violate 24h <= 72h nesting")
    return np.where(y24 == 1, 0, np.where(y72 == 1, 1, 2)).astype(np.int64)


def _binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    both = len(np.unique(target)) == 2
    return {
        "auprc": float(average_precision_score(target, probability)) if both else None,
        "auroc": float(roc_auc_score(target, probability)) if both else None,
        "brier": float(np.mean((probability - target) ** 2)),
        "ece": binary_ece(target, probability),
    }


def _phase_metrics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.arange(probabilities.shape[1])
    return {
        "macro_f1": float(
            f1_score(
                target,
                probabilities.argmax(axis=1),
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "brier": float(
            np.mean(
                np.sum(
                    (probabilities - np.eye(probabilities.shape[1])[target]) ** 2,
                    axis=1,
                )
            )
        ),
        "ece": multiclass_ece(target, probabilities),
        "nll": float(-np.mean(np.log(probabilities[np.arange(len(target)), target].clip(1e-12)))),
    }


def _load_model(artifact: dict[str, Any], device: torch.device) -> FemMHCJointModel:
    # Downstream mcPHASES experiments intentionally instantiate a six-task
    # head bank instead of the complete cross-cohort registry.  Reconstruct
    # that exact head bank from the checkpoint metadata; otherwise loading a
    # six-task checkpoint into the 70-task default model fails before
    # calibration can be evaluated.
    task_metadata = artifact.get("instantiated_tasks") or artifact.get("active_tasks")
    if task_metadata:
        task_ids = {
            item["task_id"] if isinstance(item, dict) else str(item)
            for item in task_metadata
        }
        tasks = tuple(task for task in JOINT_TASKS if task.task_id in task_ids)
    else:
        tasks = JOINT_TASKS
    model = FemMHCJointModel(
        input_dim=int(artifact["input_dim"]),
        hidden_dim=int(artifact["hidden_dim"]),
        tasks=tasks,
        maximum_days=int(artifact["maximum_days"]),
        architecture=str(artifact["architecture"]),
        dropout=float(artifact.get("dropout", 0.0)),
        routing_initial_logit=float(artifact.get("routing_initial_logit", -2.0)),
    )
    model.load_state_dict(artifact["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def collect_logits(
    model: FemMHCJointModel,
    dataset: McPhasesJointEmbeddingDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    values: dict[str, list[np.ndarray]] = {
        "onset_logits": [],
        "onset_24h": [],
        "onset_72h": [],
        "onset_participants": [],
        "phase_logits": [],
        "phase_target": [],
        "phase_participants": [],
    }
    for batch in loader:
        embeddings = batch["daily_embeddings"].to(device, non_blocking=True)
        present = batch["day_present"].to(device, non_blocking=True)
        output = model(
            embeddings,
            present,
            task_ids=(ONSET_24H, ONSET_72H, PHASE),
        )
        if output.nested_onset is None:
            raise RuntimeError("joint model did not return nested onset bins")
        participants = np.asarray(batch["participant_id"], dtype=str)
        y24 = batch["targets"][ONSET_24H].numpy()
        y72 = batch["targets"][ONSET_72H].numpy()
        onset_observed = (y24 >= 0) & (y72 >= 0)
        values["onset_logits"].append(
            output.nested_onset.bin_logits.float().cpu().numpy()[onset_observed]
        )
        values["onset_24h"].append(y24[onset_observed])
        values["onset_72h"].append(y72[onset_observed])
        values["onset_participants"].append(participants[onset_observed])

        phase_output = output.predictions[PHASE]
        if not isinstance(phase_output, ProbabilisticOutput):
            raise RuntimeError("phase output must be probabilistic")
        phase_target = batch["targets"][PHASE].numpy()
        phase_observed = phase_target >= 0
        values["phase_logits"].append(
            phase_output.logits.float().cpu().numpy()[phase_observed]
        )
        values["phase_target"].append(phase_target[phase_observed])
        values["phase_participants"].append(participants[phase_observed])
    return {key: np.concatenate(items) for key, items in values.items()}


def _bootstrap_metric(
    participants: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    calibrated: np.ndarray,
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    def score(indices: np.ndarray) -> tuple[float, float]:
        if metric == "brier":
            return (
                float(np.mean((baseline[indices] - target[indices]) ** 2)),
                float(np.mean((calibrated[indices] - target[indices]) ** 2)),
            )
        return (
            binary_ece(target[indices], baseline[indices]),
            binary_ece(target[indices], calibrated[indices]),
        )

    result = paired_cluster_bootstrap(
        participants,
        score,
        lower_is_better=True,
        replicates=replicates,
        seed=seed,
        minimum_clusters=5,
    )
    return {
        "estimate": result.estimate,
        "confidence_low": result.confidence_low,
        "confidence_high": result.confidence_high,
        "p_value_two_sided": result.p_value_two_sided,
        "clusters": result.clusters,
        "valid_replicates": result.valid_replicates,
        "eligible": result.eligible,
        "reason": result.reason,
    }


def evaluate_checkpoint(
    seed: int,
    path: Path,
    *,
    processed_dir: Path,
    embeddings_path: Path,
    batch_size: int,
    replicates: int,
    device: torch.device,
) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = _load_model(artifact, device)
    dataset_args = {
        "processed_dir": processed_dir,
        "embeddings_path": embeddings_path,
        "history_days": int(artifact["maximum_days"]),
        "minimum_history_days": 3,
    }
    train = collect_logits(
        model,
        McPhasesJointEmbeddingDataset(split="train", **dataset_args),
        batch_size=batch_size,
        device=device,
    )
    validation = collect_logits(
        model,
        McPhasesJointEmbeddingDataset(split="validation", **dataset_args),
        batch_size=batch_size,
        device=device,
    )

    train_onset_bins = onset_bins(train["onset_24h"], train["onset_72h"])
    onset_temperature = fit_temperature(train["onset_logits"], train_onset_bins)
    phase_temperature = fit_temperature(train["phase_logits"], train["phase_target"])
    onset_uncalibrated = softmax_temperature(validation["onset_logits"], 1.0)
    onset_calibrated = softmax_temperature(
        validation["onset_logits"], onset_temperature
    )
    phase_uncalibrated = softmax_temperature(validation["phase_logits"], 1.0)
    phase_calibrated = softmax_temperature(
        validation["phase_logits"], phase_temperature
    )
    base24, base72 = onset_probabilities(onset_uncalibrated)
    cal24, cal72 = onset_probabilities(onset_calibrated)

    bootstrap_records: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    for horizon, target, baseline, calibrated in (
        ("24h", validation["onset_24h"], base24, cal24),
        ("72h", validation["onset_72h"], base72, cal72),
    ):
        for metric_index, metric in enumerate(("brier", "ece")):
            key = f"{horizon}_{metric}"
            bootstrap_records[key] = _bootstrap_metric(
                validation["onset_participants"],
                target,
                baseline,
                calibrated,
                metric=metric,
                replicates=replicates,
                seed=seed * 100 + metric_index + (0 if horizon == "24h" else 10),
            )
            p_value = bootstrap_records[key]["p_value_two_sided"]
            if p_value is not None:
                p_values[key] = float(p_value)
    adjusted = holm_adjust(p_values)
    for key, value in adjusted.items():
        bootstrap_records[key]["p_value_holm"] = value

    return {
        "seed": seed,
        "checkpoint": str(path.resolve()),
        "selection_split": "train_only",
        "evaluation_split": "validation",
        "test_used": False,
        "onset_temperature": onset_temperature,
        "phase_temperature": phase_temperature,
        "train_samples": {
            "onset": int(len(train_onset_bins)),
            "phase": int(len(train["phase_target"])),
        },
        "validation_samples": {
            "onset": int(len(validation["onset_24h"])),
            "phase": int(len(validation["phase_target"])),
        },
        "onset": {
            "24h": {
                "uncalibrated": _binary_metrics(validation["onset_24h"], base24),
                "calibrated": _binary_metrics(validation["onset_24h"], cal24),
            },
            "72h": {
                "uncalibrated": _binary_metrics(validation["onset_72h"], base72),
                "calibrated": _binary_metrics(validation["onset_72h"], cal72),
            },
            "joint_nll_uncalibrated": categorical_nll(
                validation["onset_logits"],
                onset_bins(validation["onset_24h"], validation["onset_72h"]),
                1.0,
            ),
            "joint_nll_calibrated": categorical_nll(
                validation["onset_logits"],
                onset_bins(validation["onset_24h"], validation["onset_72h"]),
                onset_temperature,
            ),
            "uncalibrated_nesting_violations": int(np.count_nonzero(base24 > base72 + 1e-12)),
            "calibrated_nesting_violations": int(np.count_nonzero(cal24 > cal72 + 1e-12)),
            "bootstrap": bootstrap_records,
        },
        "phase": {
            "uncalibrated": _phase_metrics(validation["phase_target"], phase_uncalibrated),
            "calibrated": _phase_metrics(validation["phase_target"], phase_calibrated),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument(
        "--mcphases-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("batch size and bootstrap replicates must be positive")
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise ValueError("checkpoint seeds must be unique")
    device = torch.device(args.device)
    records = [
        evaluate_checkpoint(
            seed,
            path,
            processed_dir=args.mcphases_dir,
            embeddings_path=args.mcphases_embeddings,
            batch_size=args.batch_size,
            replicates=args.bootstrap_replicates,
            device=device,
        )
        for seed, path in sorted(checkpoints.items())
    ]
    summary = {
        "format_version": 1,
        "protocol": "train_temperature_validation_evaluation",
        "test_used": False,
        "seeds": [record["seed"] for record in records],
        "onset_temperature_mean": float(np.mean([record["onset_temperature"] for record in records])),
        "onset_temperature_sample_sd": float(
            np.std([record["onset_temperature"] for record in records], ddof=1)
            if len(records) > 1
            else 0.0
        ),
        "phase_temperature_mean": float(np.mean([record["phase_temperature"] for record in records])),
        "phase_temperature_sample_sd": float(
            np.std([record["phase_temperature"] for record in records], ddof=1)
            if len(records) > 1
            else 0.0
        ),
        "records": records,
    }
    for horizon in ("24h", "72h"):
        summary[f"onset_{horizon}"] = {}
        for metric in ("auprc", "auroc", "brier", "ece"):
            for calibration in ("uncalibrated", "calibrated"):
                values = [
                    record["onset"][horizon][calibration][metric]
                    for record in records
                    if record["onset"][horizon][calibration][metric] is not None
                ]
                summary[f"onset_{horizon}"][f"{metric}_{calibration}_mean"] = float(np.mean(values))
                summary[f"onset_{horizon}"][f"{metric}_{calibration}_sample_sd"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
    summary["nesting_violations_total"] = int(
        sum(
            record["onset"][key]
            for record in records
            for key in (
                "uncalibrated_nesting_violations",
                "calibrated_nesting_violations",
            )
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# FemMHC概率校准与月经开始一致性",
        "",
        "> 温度只在29名训练参与者上拟合，6名验证参与者只评估；测试集未使用。",
        "",
        f"- 月经开始温度：{summary['onset_temperature_mean']:.4f} ± {summary['onset_temperature_sample_sd']:.4f}",
        f"- 周期阶段温度：{summary['phase_temperature_mean']:.4f} ± {summary['phase_temperature_sample_sd']:.4f}",
        f"- 24小时Brier：{summary['onset_24h']['brier_uncalibrated_mean']:.4f} → {summary['onset_24h']['brier_calibrated_mean']:.4f}",
        f"- 72小时Brier：{summary['onset_72h']['brier_uncalibrated_mean']:.4f} → {summary['onset_72h']['brier_calibrated_mean']:.4f}",
        f"- 24小时ECE：{summary['onset_24h']['ece_uncalibrated_mean']:.4f} → {summary['onset_24h']['ece_calibrated_mean']:.4f}",
        f"- 72小时ECE：{summary['onset_72h']['ece_uncalibrated_mean']:.4f} → {summary['onset_72h']['ece_calibrated_mean']:.4f}",
        f"- 全部嵌套概率违反次数：{summary['nesting_violations_total']}",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
