"""Paired participant-bootstrap comparison of selected female-task heads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error

from femmhc import MCPHASES_TASKS, McPhasesTaskHeads, McPhasesV2TaskHeads


def parse_model(value: str) -> tuple[str, Path, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("model must be NAME|CHECKPOINT|EMBEDDINGS")
    return parts[0], Path(parts[1]), Path(parts[2])


def task_target(task, labels, hormones, rows):
    target = np.full(len(rows), np.nan, dtype=np.float32)
    if task.kind == "regression":
        target[:] = hormones[:, {"lh": 0, "estrogen": 1, "pdg": 2}[task.name]]
    elif task.target_offset_days == 0:
        source = labels[:, task.label_column]
        target[source >= 0] = source[source >= 0]
    else:
        lookup = {
            (row["participant_id"], row["study_interval"], int(row["day_in_study"])): int(
                row["sample_index"]
            )
            for row in rows
        }
        for row in rows:
            source_index = int(row["sample_index"])
            future = lookup.get(
                (row["participant_id"], row["study_interval"], int(row["day_in_study"]) + 1)
            )
            if future is not None and labels[future, task.label_column] >= 0:
                target[source_index] = labels[future, task.label_column]
    return target


def predict(checkpoint_path: Path, embedding_path: Path):
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if artifact.get("task_head_version") == "v2":
        heads = McPhasesV2TaskHeads(
            384,
            linear_cycle_head=bool(artifact.get("linear_cycle_head", False)),
        ).eval()
    else:
        heads = McPhasesTaskHeads(384).eval()
    heads.load_state_dict(artifact["task_heads_state_dict"])
    embedding = np.load(embedding_path)
    usable = np.isfinite(embedding).all(axis=1)
    raw_outputs: dict[str, list[np.ndarray]] = {task.name: [] for task in MCPHASES_TASKS}
    with torch.inference_mode():
        for start in range(0, len(embedding), 256):
            batch = torch.from_numpy(np.nan_to_num(embedding[start : start + 256])).float()
            outputs = heads(batch)
            for task in MCPHASES_TASKS:
                output = outputs[task.name]
                value = output.numpy() if task.kind == "regression" else output.probabilities.numpy()
                raw_outputs[task.name].append(value)

    hormone_means = np.asarray(artifact["hormone_log_means"], dtype=np.float32)
    hormone_stds = np.asarray(artifact["hormone_log_stds"], dtype=np.float32)
    predictions: dict[str, np.ndarray] = {}
    for task in MCPHASES_TASKS:
        raw = np.concatenate(raw_outputs[task.name])
        if task.kind == "regression":
            index = {"lh": 0, "estrogen": 1, "pdg": 2}[task.name]
            predictions[task.name] = np.expm1(
                raw * hormone_stds[index] + hormone_means[index]
            )
        elif task.kind == "ordinal":
            predictions[task.name] = raw @ np.arange(task.classes or 2)
        elif task.classes == 2:
            predictions[task.name] = raw[:, 1]
        else:
            predictions[task.name] = raw.argmax(axis=1)
    return predictions, usable, int(artifact["steps"])


def primary_metric(task, target: np.ndarray, prediction: np.ndarray) -> float:
    if task.kind in {"ordinal", "regression"}:
        return float(mean_absolute_error(target, prediction))
    if task.classes == 2:
        return float(average_precision_score(target, prediction))
    return float(f1_score(target, prediction, average="macro", zero_division=0))


def oriented_improvement(task, baseline: float, candidate: float) -> tuple[float, float]:
    if task.kind == "classification":
        absolute = candidate - baseline
    else:
        absolute = baseline - candidate
    relative = 100.0 * absolute / max(abs(baseline), 1e-12)
    return absolute, relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task",
        action="append",
        help="Restrict the comparison to one or more task names.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(args.processed_dir / "labels.npy")
    hormones = np.load(args.processed_dir / "hormones.npy")
    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_spec = json.loads(
        (args.processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    split_by_user = {user: split for split, users in split_spec.items() for user in users}
    sample_split = np.asarray([split_by_user[row["participant_id"]] for row in rows])
    participants = np.asarray([row["participant_id"] for row in rows])

    model_predictions: dict[str, dict[str, np.ndarray]] = {}
    model_usable: dict[str, np.ndarray] = {}
    model_steps: dict[str, int] = {}
    for name, checkpoint, embedding in args.model:
        if name in model_predictions:
            raise ValueError(f"duplicate model name: {name}")
        model_predictions[name], model_usable[name], model_steps[name] = predict(
            checkpoint, embedding
        )
    if args.candidate not in model_predictions:
        raise ValueError(f"candidate {args.candidate!r} was not supplied via --model")

    rng = np.random.default_rng(args.seed)
    candidate = args.candidate
    result_rows: list[dict[str, object]] = []
    for baseline in model_predictions:
        if baseline == candidate:
            continue
        for task in MCPHASES_TASKS:
            if args.task and task.name not in args.task:
                continue
            target = task_target(task, labels, hormones, rows)
            mask = (
                (sample_split == "test")
                & np.isfinite(target)
                & model_usable[baseline]
                & model_usable[candidate]
            )
            if not mask.any():
                continue
            y = target[mask]
            baseline_prediction = model_predictions[baseline][task.name][mask]
            candidate_prediction = model_predictions[candidate][task.name][mask]
            participant = participants[mask]
            unique_participants = np.unique(participant)
            groups = [np.flatnonzero(participant == value) for value in unique_participants]

            baseline_value = primary_metric(task, y, baseline_prediction)
            candidate_value = primary_metric(task, y, candidate_prediction)
            absolute, relative = oriented_improvement(task, baseline_value, candidate_value)

            bootstrap_relative: list[float] = []
            for _ in range(args.bootstrap_draws):
                sampled_groups = rng.integers(0, len(groups), size=len(groups))
                sampled = np.concatenate([groups[index] for index in sampled_groups])
                sampled_target = y[sampled]
                if task.classes == 2 and np.unique(sampled_target).size < 2:
                    continue
                baseline_draw = primary_metric(
                    task, sampled_target, baseline_prediction[sampled]
                )
                candidate_draw = primary_metric(
                    task, sampled_target, candidate_prediction[sampled]
                )
                _, relative_draw = oriented_improvement(task, baseline_draw, candidate_draw)
                if np.isfinite(relative_draw):
                    bootstrap_relative.append(relative_draw)
            draws = np.asarray(bootstrap_relative)
            result_rows.append(
                {
                    "baseline": baseline,
                    "baseline_step": model_steps[baseline],
                    "candidate": candidate,
                    "candidate_step": model_steps[candidate],
                    "task": task.name,
                    "task_chinese": task.chinese_name,
                    "primary_metric": task.primary_metric,
                    "direction": "higher" if task.kind == "classification" else "lower",
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "absolute_improvement": absolute,
                    "relative_improvement_percent": relative,
                    "ci95_low_percent": float(np.quantile(draws, 0.025)),
                    "ci95_high_percent": float(np.quantile(draws, 0.975)),
                    "bootstrap_probability_improved": float(np.mean(draws > 0)),
                    "bootstrap_valid_draws": int(len(draws)),
                    "test_samples": int(mask.sum()),
                    "test_participants": int(len(unique_participants)),
                }
            )

    frame = pd.DataFrame(result_rows)
    frame.to_csv(
        args.output_dir / "paired_participant_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        baseline: {
            "tasks_improved_point_estimate": int((group["relative_improvement_percent"] > 0).sum()),
            "tasks_ci_strictly_above_zero": int((group["ci95_low_percent"] > 0).sum()),
            "tasks_total": int(len(group)),
        }
        for baseline, group in frame.groupby("baseline")
    }
    (args.output_dir / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    markdown = ["# FemMHC 配对参与者 Bootstrap", ""]
    for baseline, group in frame.groupby("baseline"):
        markdown.extend(
            [
                f"## {candidate} 相对 {baseline}",
                "",
                "|任务|基线|FemMHC|相对提升|95% CI|改善概率|",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in group.itertuples():
            markdown.append(
                f"|{row.task_chinese}|{row.baseline_value:.4f}|{row.candidate_value:.4f}|"
                f"{row.relative_improvement_percent:+.2f}%|"
                f"[{row.ci95_low_percent:+.2f}%, {row.ci95_high_percent:+.2f}%]|"
                f"{row.bootstrap_probability_improved:.3f}|"
            )
        markdown.append("")
    (args.output_dir / "paired_participant_bootstrap.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
