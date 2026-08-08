#!/usr/bin/env python
"""Aggregate paired multi-seed FemMHC validation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"mae", "mae_weeks", "rmse", "brier", "ece"}


def _parse_run(value: str) -> tuple[str, int, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("run must be MODEL:SEED=OUTPUT_DIR")
    label, raw_path = value.split("=", 1)
    model, raw_seed = label.rsplit(":", 1)
    path = Path(raw_path)
    if not (path / "per_task_metrics.csv").is_file():
        raise argparse.ArgumentTypeError(
            f"missing per_task_metrics.csv under {path}"
        )
    return model, int(raw_seed), path


def _load(model: str, seed: int, directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "per_task_metrics.csv")
    frame = frame[frame["is_primary"].astype(str).str.lower() == "true"].copy()
    frame = frame[np.isfinite(frame["value"])].copy()
    frame["model"] = model
    frame["seed"] = seed
    frame["oriented_value"] = np.where(
        frame["metric"].isin(LOWER_IS_BETTER), -frame["value"], frame["value"]
    )
    return frame


def _subset_summary(paired: pd.DataFrame) -> dict[str, object]:
    if paired.empty:
        return {
            "tasks": 0,
            "candidate_mean_wins": 0,
            "ties": 0,
            "candidate_mean_losses": 0,
            "candidate_wins_at_least_two_seeds": 0,
            "candidate_wins_all_seeds": 0,
        }
    grouped = paired.groupby("task_id", sort=False)
    mean_delta = grouped["oriented_delta"].mean()
    seed_wins = grouped["oriented_delta"].apply(lambda x: int((x > 1e-12).sum()))
    return {
        "tasks": int(len(mean_delta)),
        "candidate_mean_wins": int((mean_delta > 1e-12).sum()),
        "ties": int((mean_delta.abs() <= 1e-12).sum()),
        "candidate_mean_losses": int((mean_delta < -1e-12).sum()),
        "candidate_wins_at_least_two_seeds": int((seed_wins >= 2).sum()),
        "candidate_wins_all_seeds": int((seed_wins == 3).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames = [_load(*run) for run in args.run]
    combined = pd.concat(frames, ignore_index=True)
    models = set(combined["model"])
    if {args.baseline, args.candidate} - models:
        raise ValueError("baseline and candidate must both occur in --run")

    duplicate = combined.duplicated(["model", "seed", "task_id"], keep=False)
    if duplicate.any():
        raise ValueError("each model/seed/task must contain one primary metric")
    seed_sets = combined.groupby("model")["seed"].apply(lambda x: set(x))
    if seed_sets[args.baseline] != seed_sets[args.candidate]:
        raise ValueError("baseline and candidate must use identical seeds")
    seeds = sorted(seed_sets[args.baseline])

    metadata = ["task_id", "source", "domain", "kind", "metric"]
    baseline = combined[combined["model"] == args.baseline][
        metadata + ["seed", "value", "oriented_value"]
    ].rename(
        columns={
            "value": "baseline_value",
            "oriented_value": "baseline_oriented",
        }
    )
    candidate = combined[combined["model"] == args.candidate][
        ["task_id", "seed", "value", "oriented_value"]
    ].rename(
        columns={
            "value": "candidate_value",
            "oriented_value": "candidate_oriented",
        }
    )
    paired = baseline.merge(candidate, on=["task_id", "seed"], how="inner")
    paired["oriented_delta"] = (
        paired["candidate_oriented"] - paired["baseline_oriented"]
    )
    complete = paired.groupby("task_id")["seed"].nunique()
    complete_ids = complete[complete == len(seeds)].index
    paired = paired[paired["task_id"].isin(complete_ids)].copy()

    per_task = (
        paired.groupby(metadata, as_index=False)
        .agg(
            baseline_mean=("baseline_value", "mean"),
            baseline_std=("baseline_value", "std"),
            candidate_mean=("candidate_value", "mean"),
            candidate_std=("candidate_value", "std"),
            oriented_delta_mean=("oriented_delta", "mean"),
            oriented_delta_std=("oriented_delta", "std"),
            candidate_seed_wins=(
                "oriented_delta",
                lambda x: int((x > 1e-12).sum()),
            ),
        )
        .sort_values(["source", "domain", "task_id"])
    )
    per_task["winner"] = np.select(
        [
            per_task["oriented_delta_mean"] > 1e-12,
            per_task["oriented_delta_mean"] < -1e-12,
        ],
        [args.candidate, args.baseline],
        default="tie",
    )

    summary: dict[str, object] = {
        "format_version": 1,
        "split": args.split,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "seeds": seeds,
        "all_tasks": _subset_summary(paired),
        "openmhc_tasks": _subset_summary(paired[paired["source"] == "openmhc"]),
        "female_specific_tasks": _subset_summary(
            paired[paired["source"] != "openmhc"]
        ),
        "domains": {
            domain: _subset_summary(group)
            for domain, group in paired.groupby("domain", sort=True)
        },
        "sources": {
            source: _subset_summary(group)
            for source, group in paired.groupby("source", sort=True)
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_seed_metrics.csv", index=False)
    per_task.to_csv(args.output_dir / "per_task_multiseed.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# FemMHC 三随机种子验证集汇总",
        "",
        f"- 基线：`{args.baseline}`",
        f"- 候选：`{args.candidate}`",
        f"- 随机种子：{', '.join(map(str, seeds))}",
        (
            "- 测试集：保持锁定"
            if args.split == "validation"
            else "- 数据划分：最终测试集（模型与超参数已冻结）"
        ),
        "",
        "| 范围 | 任务数 | 候选均值胜 | 平 | 候选均值负 | 至少2/3种子胜 | 3/3种子胜 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("all_tasks", "全部任务"),
        ("openmhc_tasks", "OpenMHC任务"),
        ("female_specific_tasks", "女性特异任务"),
    ):
        item = summary[key]
        lines.append(
            f"| {label} | {item['tasks']} | {item['candidate_mean_wins']} | "
            f"{item['ties']} | {item['candidate_mean_losses']} | "
            f"{item['candidate_wins_at_least_two_seeds']} | "
            f"{item['candidate_wins_all_seeds']} |"
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
