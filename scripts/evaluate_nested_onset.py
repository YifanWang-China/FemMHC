"""Validation-calibrated evaluation for coherent 24 h / 72 h onset risks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from femmhc import McPhasesV2TaskHeads


def parse_model(value: str) -> tuple[str, Path, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("model must be NAME|CHECKPOINT|EMBEDDINGS")
    return parts[0], Path(parts[1]), Path(parts[2])


def expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            value += selected.mean() * abs(target[selected].mean() - probability[selected].mean())
    return float(value)


def onset_bins(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = (labels[:, 8] >= 0) & (labels[:, 9] >= 0)
    target = np.full(len(labels), -1, dtype=np.int64)
    target[observed] = np.where(
        labels[observed, 8] == 1,
        0,
        np.where(labels[observed, 9] == 1, 1, 2),
    )
    return target, observed


def load_bin_logits(checkpoint: Path, embedding: Path) -> tuple[np.ndarray, np.ndarray]:
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if artifact.get("task_head_version") != "v2":
        raise ValueError(f"nested onset evaluation requires a v2 checkpoint: {checkpoint}")
    heads = McPhasesV2TaskHeads(384).eval()
    heads.load_state_dict(artifact["task_heads_state_dict"])
    values = np.load(embedding)
    usable = np.isfinite(values).all(axis=1)
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 256):
            batch = torch.from_numpy(np.nan_to_num(values[start : start + 256])).float()
            _, onset = heads.forward_with_aux(batch)
            logits.append(onset.bin_logits.numpy())
    return np.concatenate(logits), usable


def fit_calibration(logits: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit scalar temperature and two free class-prior offsets on validation data."""

    def objective(parameters: np.ndarray) -> float:
        temperature = np.exp(parameters[0])
        bias = np.asarray([parameters[1], parameters[2], 0.0])
        scaled = (logits + bias) / temperature
        nll = np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(target)), target])
        return float(nll + 1e-4 * np.square(bias).sum())

    result = minimize(
        objective,
        x0=np.zeros(3, dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(np.log(0.05), np.log(20.0)), (-12.0, 12.0), (-12.0, 12.0)],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"temperature fitting failed: {result.message}")
    return float(np.exp(result.x[0])), np.asarray([result.x[1], result.x[2], 0.0])


def metric(metric_name: str, target: np.ndarray, probability: np.ndarray) -> float:
    if metric_name == "auprc":
        return float(average_precision_score(target, probability))
    if metric_name == "auroc":
        return float(roc_auc_score(target, probability))
    if metric_name == "brier":
        return float(brier_score_loss(target, probability))
    if metric_name == "ece_10":
        return expected_calibration_error(target, probability, bins=10)
    raise ValueError(metric_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=parse_model, required=True)
    parser.add_argument("--candidate", type=parse_model, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(args.processed_dir / "labels.npy")
    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_spec = json.loads(
        (args.processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    split_by_user = {user: split for split, users in split_spec.items() for user in users}
    sample_split = np.asarray([split_by_user[row["participant_id"]] for row in rows])
    participants = np.asarray([row["participant_id"] for row in rows])
    bins, bins_observed = onset_bins(labels)

    models = {args.baseline[0]: args.baseline, args.candidate[0]: args.candidate}
    if len(models) != 2:
        raise ValueError("baseline and candidate names must differ")
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    usable_by_model: dict[str, np.ndarray] = {}
    temperatures: dict[str, float] = {}
    calibration_biases: dict[str, list[float]] = {}
    for name, (_, checkpoint, embedding) in models.items():
        logits, usable = load_bin_logits(checkpoint, embedding)
        validation = (sample_split == "validation") & bins_observed & usable
        temperature, bias = fit_calibration(logits[validation], bins[validation])
        calibrated_bins = softmax((logits + bias) / temperature, axis=1)
        probabilities[name] = {
            "menstrual_onset_24h": calibrated_bins[:, 0],
            "menstrual_onset_72h": calibrated_bins[:, 0] + calibrated_bins[:, 1],
        }
        usable_by_model[name] = usable
        temperatures[name] = temperature
        calibration_biases[name] = bias.tolist()

    baseline_name = args.baseline[0]
    candidate_name = args.candidate[0]
    tasks = {
        "menstrual_onset_24h": ("24小时内月经开始", labels[:, 8]),
        "menstrual_onset_72h": ("72小时内月经开始", labels[:, 9]),
    }
    metric_names = ("auprc", "auroc", "brier", "ece_10")
    lower_is_better = {"brier", "ece_10"}
    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for task, (task_chinese, target) in tasks.items():
        mask = (
            (sample_split == "test")
            & (target >= 0)
            & usable_by_model[baseline_name]
            & usable_by_model[candidate_name]
        )
        y = target[mask]
        participant = participants[mask]
        unique_participants = np.unique(participant)
        groups = [np.flatnonzero(participant == value) for value in unique_participants]
        for model_name in (baseline_name, candidate_name):
            prediction = probabilities[model_name][task][mask]
            for metric_name in metric_names:
                metric_rows.append(
                    {
                        "model": model_name,
                        "task": task,
                        "task_chinese": task_chinese,
                        "metric": metric_name,
                        "value": metric(metric_name, y, prediction),
                        "temperature": temperatures[model_name],
                        "test_samples": int(mask.sum()),
                        "test_participants": int(len(unique_participants)),
                    }
                )
        for metric_name in metric_names:
            baseline_prediction = probabilities[baseline_name][task][mask]
            candidate_prediction = probabilities[candidate_name][task][mask]
            baseline_value = metric(metric_name, y, baseline_prediction)
            candidate_value = metric(metric_name, y, candidate_prediction)
            sign = -1.0 if metric_name in lower_is_better else 1.0
            point = 100.0 * sign * (candidate_value - baseline_value) / max(abs(baseline_value), 1e-12)
            draws: list[float] = []
            for _ in range(args.bootstrap_draws):
                choices = rng.integers(0, len(groups), size=len(groups))
                sampled = np.concatenate([groups[index] for index in choices])
                sampled_target = y[sampled]
                if metric_name in {"auprc", "auroc"} and np.unique(sampled_target).size < 2:
                    continue
                base_draw = metric(metric_name, sampled_target, baseline_prediction[sampled])
                candidate_draw = metric(metric_name, sampled_target, candidate_prediction[sampled])
                draws.append(
                    100.0 * sign * (candidate_draw - base_draw) / max(abs(base_draw), 1e-12)
                )
            draw_array = np.asarray(draws)
            bootstrap_rows.append(
                {
                    "task": task,
                    "task_chinese": task_chinese,
                    "metric": metric_name,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "relative_improvement_percent": point,
                    "ci95_low_percent": float(np.quantile(draw_array, 0.025)),
                    "ci95_high_percent": float(np.quantile(draw_array, 0.975)),
                    "bootstrap_probability_improved": float(np.mean(draw_array > 0)),
                    "valid_draws": int(len(draw_array)),
                }
            )

    pd.DataFrame(metric_rows).to_csv(
        args.output_dir / "calibrated_onset_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output_dir / "calibrated_onset_bootstrap.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "calibration_split": "validation participants only",
        "temperatures": temperatures,
        "class_biases": calibration_biases,
        "nested_probability_violations": 0,
    }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
