#!/usr/bin/env python
"""Paired participant-cluster bootstrap for two FemMHC joint checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate_femmhc_joint import _datasets, _metrics
from femmhc import JOINT_TASKS, FemMHCJointModel, ProbabilisticOutput
from femmhc.statistics import holm_adjust, paired_cluster_bootstrap


LOWER_IS_BETTER = {"mae", "mae_weeks", "rmse", "brier", "ece"}


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
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


def _load_model(path: Path, device: torch.device) -> tuple[FemMHCJointModel, dict[str, Any], dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = FemMHCJointModel(
        input_dim=int(artifact["input_dim"]),
        hidden_dim=int(artifact["hidden_dim"]),
        maximum_days=int(artifact["maximum_days"]),
        architecture=str(artifact.get("architecture", "full")),
    )
    model.load_state_dict(artifact["model_state_dict"])
    return model.to(device).eval(), artifact["regression_target_statistics"], artifact


def _prediction_arrays(
    prediction_output: ProbabilisticOutput | torch.Tensor,
    *,
    task_id: str,
    observed: np.ndarray,
    statistics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(prediction_output, ProbabilisticOutput):
        probabilities = prediction_output.probabilities.float().cpu().numpy()
        return probabilities.argmax(axis=1)[observed], probabilities[observed]
    prediction = prediction_output.float().cpu().numpy()
    if task_id in statistics:
        prediction = prediction * statistics[task_id]["std"] + statistics[task_id]["mean"]
    return prediction[observed], None


def _subset_summary(frame: pd.DataFrame) -> dict[str, int]:
    eligible = frame[frame["eligible"]]
    return {
        "tasks": int(len(frame)),
        "eligible_tasks": int(len(eligible)),
        "candidate_point_wins": int((frame["oriented_delta"] > 0.0).sum()),
        "candidate_point_losses": int((frame["oriented_delta"] < 0.0).sum()),
        "confidence_interval_above_zero": int((eligible["confidence_low"] > 0.0).sum()),
        "confidence_interval_below_zero": int((eligible["confidence_high"] < 0.0).sum()),
        "holm_significant_candidate_wins": int(
            ((eligible["oriented_delta"] > 0.0) & (eligible["holm_adjusted_p"] < 0.05)).sum()
        ),
        "holm_significant_candidate_losses": int(
            ((eligible["oriented_delta"] < 0.0) & (eligible["holm_adjusted_p"] < 0.05)).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-participants", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    _add_data_arguments(parser)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("test evaluation is locked; pass --allow-test only for a frozen final analysis")

    device = torch.device(args.device)
    baseline_model, baseline_statistics, baseline_artifact = _load_model(
        args.baseline_checkpoint, device
    )
    candidate_model, candidate_statistics, candidate_artifact = _load_model(
        args.candidate_checkpoint, device
    )
    task_by_id = {task.task_id: task for task in JOINT_TASKS if task.trainable}
    collected: dict[str, dict[str, Any]] = {}

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
                    baseline_output = baseline_model(embeddings, present, task_ids=task_ids)
                    candidate_output = candidate_model(embeddings, present, task_ids=task_ids)
                for task_id, target_tensor in batch["targets"].items():
                    task = task_by_id[task_id]
                    target = target_tensor.numpy()
                    observed = np.isfinite(target)
                    if task.kind != "regression":
                        observed &= target >= 0
                    if not observed.any():
                        continue
                    baseline_prediction, baseline_probabilities = _prediction_arrays(
                        baseline_output.predictions[task_id],
                        task_id=task_id,
                        observed=observed,
                        statistics=baseline_statistics,
                    )
                    candidate_prediction, candidate_probabilities = _prediction_arrays(
                        candidate_output.predictions[task_id],
                        task_id=task_id,
                        observed=observed,
                        statistics=candidate_statistics,
                    )
                    record = collected.setdefault(
                        task_id,
                        {
                            "target": [],
                            "participant": [],
                            "baseline_prediction": [],
                            "candidate_prediction": [],
                            "baseline_probabilities": [],
                            "candidate_probabilities": [],
                            "cohort": cohort,
                        },
                    )
                    record["target"].append(target[observed])
                    record["participant"].extend(
                        np.asarray(batch["participant_id"], dtype=str)[observed].tolist()
                    )
                    record["baseline_prediction"].append(baseline_prediction)
                    record["candidate_prediction"].append(candidate_prediction)
                    if baseline_probabilities is not None:
                        record["baseline_probabilities"].append(baseline_probabilities)
                    if candidate_probabilities is not None:
                        record["candidate_probabilities"].append(candidate_probabilities)

    rows: list[dict[str, Any]] = []
    for task_id, record in sorted(collected.items()):
        task = task_by_id[task_id]
        target = np.concatenate(record["target"])
        participant = np.asarray(record["participant"], dtype=str)
        baseline_prediction = np.concatenate(record["baseline_prediction"])
        candidate_prediction = np.concatenate(record["candidate_prediction"])
        baseline_probabilities = (
            np.concatenate(record["baseline_probabilities"])
            if record["baseline_probabilities"]
            else None
        )
        candidate_probabilities = (
            np.concatenate(record["candidate_probabilities"])
            if record["candidate_probabilities"]
            else None
        )
        metric_name = "mae" if task.primary_metric == "mae_weeks" else task.primary_metric

        def score_pair(indices: np.ndarray) -> tuple[float | None, float | None]:
            baseline_metrics = _metrics(
                kind=task.kind,
                target=target[indices],
                prediction=baseline_prediction[indices],
                probabilities=(
                    baseline_probabilities[indices]
                    if baseline_probabilities is not None
                    else None
                ),
            )
            candidate_metrics = _metrics(
                kind=task.kind,
                target=target[indices],
                prediction=candidate_prediction[indices],
                probabilities=(
                    candidate_probabilities[indices]
                    if candidate_probabilities is not None
                    else None
                ),
            )
            return baseline_metrics[metric_name], candidate_metrics[metric_name]

        result = paired_cluster_bootstrap(
            participant,
            score_pair,
            lower_is_better=task.primary_metric in LOWER_IS_BETTER,
            replicates=args.replicates,
            confidence=args.confidence,
            seed=args.seed + zlib.crc32(task_id.encode("utf-8")),
            minimum_clusters=args.minimum_participants,
        )
        baseline_value, candidate_value = score_pair(np.arange(len(target)))
        row = {
            "task_id": task_id,
            "source": task.source,
            "domain": task.domain,
            "kind": task.kind,
            "primary_metric": task.primary_metric,
            "samples": len(target),
            "participants": len(np.unique(participant)),
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
            "oriented_delta": result.estimate,
            **{key: value for key, value in asdict(result).items() if key != "estimate"},
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    adjusted = holm_adjust(
        {
            row.task_id: row.p_value_two_sided
            for row in frame.itertuples()
            if row.eligible and pd.notna(row.p_value_two_sided)
        }
    )
    frame["holm_adjusted_p"] = frame["task_id"].map(adjusted)
    summary = {
        "format_version": 1,
        "split": args.split,
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "baseline_step": int(baseline_artifact["step"]),
        "candidate_step": int(candidate_artifact["step"]),
        "resampling_unit": "participant",
        "paired": True,
        "replicates": args.replicates,
        "confidence": args.confidence,
        "minimum_participants": args.minimum_participants,
        "all_tasks": _subset_summary(frame),
        "openmhc_tasks": _subset_summary(frame[frame["source"] == "openmhc"]),
        "female_specific_tasks": _subset_summary(frame[frame["source"] != "openmhc"]),
        "domains": {
            domain: _subset_summary(group)
            for domain, group in frame.groupby("domain", sort=True)
        },
        "limitations": [
            "This is validation-set inference and is not a replacement for the locked final test.",
            "Tasks below the minimum participant threshold are reported descriptively only.",
            "Holm correction is applied across all eligible primary-task comparisons.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "paired_participant_bootstrap.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# FemMHC 配对参与者簇自助法",
        "",
        f"- 数据划分：{args.split}",
        f"- 基线：`{args.baseline_name}`",
        f"- 候选：`{args.candidate_name}`",
        f"- 重采样：按参与者配对重采样，共 {args.replicates} 次",
        f"- 纳入门槛：每项任务至少 {args.minimum_participants} 名参与者",
        "- 多重比较：对所有可推断主指标使用 Holm 校正",
        "",
        "| 范围 | 任务 | 可推断 | 候选点估计胜 | 95%区间全大于0 | Holm校正后显著胜 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("all_tasks", "全部任务"),
        ("openmhc_tasks", "OpenMHC任务"),
        ("female_specific_tasks", "女性特异任务"),
    ):
        item = summary[key]
        lines.append(
            f"| {label} | {item['tasks']} | {item['eligible_tasks']} | "
            f"{item['candidate_point_wins']} | {item['confidence_interval_above_zero']} | "
            f"{item['holm_significant_candidate_wins']} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
