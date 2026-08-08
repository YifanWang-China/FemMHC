#!/usr/bin/env python
"""Compare train-selected mcPHASES state routing with random task routing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_result(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("seed result must be SEED=summary.json")
    raw_seed, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"missing result: {path}")
    return int(raw_seed), path


def normalized_utility(values: np.ndarray) -> np.ndarray:
    """Map within-task oriented metrics to [0, 1] without mixing units."""

    values = np.asarray(values, dtype=np.float64)
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        return np.full_like(values, 0.5)
    return (values - low) / (high - low)


def best_rank(values: np.ndarray, index: int) -> float:
    """Return descending rank with average ranks for exact ties."""

    values = np.asarray(values, dtype=np.float64)
    target = values[index]
    better = int(np.sum(values > target))
    tied = int(np.sum(np.isclose(values, target, rtol=0.0, atol=1e-12)))
    return 1.0 + better + (tied - 1) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-result",
        action="append",
        type=_parse_result,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.replicates <= 0:
        raise ValueError("replicates must be positive")

    loaded: dict[int, dict[str, Any]] = {}
    for seed, path in args.seed_result:
        if seed in loaded:
            raise ValueError(f"duplicate seed: {seed}")
        loaded[seed] = json.loads(path.read_text(encoding="utf-8"))
    seeds = sorted(loaded)
    reference = loaded[seeds[0]]
    sources = tuple(reference["representation_sources"])
    task_ids = tuple(reference["tasks"])

    records: list[dict[str, Any]] = []
    task_source_utilities: list[np.ndarray] = []
    selected_indices: list[int] = []
    for task_id in task_ids:
        selected_source = str(
            reference["tasks"][task_id]["selection"]["selected"][
                "representation_source"
            ]
        )
        if selected_source not in sources:
            raise ValueError(f"unknown selected source for {task_id}")
        per_seed: list[np.ndarray] = []
        evaluable = True
        for seed in seeds:
            task = loaded[seed]["tasks"][task_id]
            current_selected = str(
                task["selection"]["selected"]["representation_source"]
            )
            if current_selected != selected_source:
                raise ValueError(f"selection changed across seeds for {task_id}")
            values: list[float] = []
            for source in sources:
                value = task["validation_by_source"][source][
                    "validation_oriented_metric"
                ]
                if value is None or not np.isfinite(value):
                    evaluable = False
                    break
                values.append(float(value))
            if not evaluable:
                break
            per_seed.append(np.asarray(values, dtype=np.float64))
        if not evaluable:
            continue

        metrics = np.stack(per_seed)
        utilities = np.stack([normalized_utility(row) for row in metrics])
        mean_metrics = metrics.mean(axis=0)
        mean_utilities = utilities.mean(axis=0)
        selected_index = sources.index(selected_source)
        general_index = sources.index("general")
        records.append(
            {
                "task_id": task_id,
                "selected_source": selected_source,
                "selected_mean_utility": float(mean_utilities[selected_index]),
                "general_mean_utility": float(mean_utilities[general_index]),
                "selected_mean_oriented_metric": float(mean_metrics[selected_index]),
                "general_mean_oriented_metric": float(mean_metrics[general_index]),
                "selected_rank_of_four": best_rank(mean_metrics, selected_index),
            }
        )
        task_source_utilities.append(mean_utilities)
        selected_indices.append(selected_index)

    utilities = np.stack(task_source_utilities)
    selected = utilities[np.arange(len(records)), np.asarray(selected_indices)]
    observed = float(selected.mean())
    general = float(utilities[:, sources.index("general")].mean())
    random_expectation = float(utilities.mean())

    rng = np.random.default_rng(args.seed)
    random_indices = rng.integers(
        0,
        len(sources),
        size=(args.replicates, len(records)),
    )
    random_scores = utilities[np.arange(len(records))[None, :], random_indices].mean(
        axis=1
    )
    p_value = float(
        (1 + np.count_nonzero(random_scores >= observed)) / (args.replicates + 1)
    )
    positive_vs_general = int(sum(
        item["selected_mean_oriented_metric"] > item["general_mean_oriented_metric"]
        for item in records
    ))
    equal_to_general = int(sum(
        np.isclose(
            item["selected_mean_oriented_metric"],
            item["general_mean_oriented_metric"],
            rtol=0.0,
            atol=1e-12,
        )
        for item in records
    ))
    negative_vs_general = int(
        len(records) - positive_vs_general - equal_to_general
    )
    summary = {
        "format_version": 1,
        "split": "validation",
        "selection_split": "train_only",
        "test_used": False,
        "seeds": seeds,
        "sources": sources,
        "evaluable_tasks": len(records),
        "selected_mean_normalized_utility": observed,
        "general_mean_normalized_utility": general,
        "uniform_random_expected_utility": random_expectation,
        "selected_advantage_over_random_expectation": observed - random_expectation,
        "selected_advantage_over_general": observed - general,
        "random_routing_replicates": args.replicates,
        "random_routing_p_value": p_value,
        "positive_vs_general_tasks": positive_vs_general,
        "ties_vs_general_tasks": equal_to_general,
        "negative_vs_general_tasks": negative_vs_general,
        "mean_selected_rank_of_four": float(
            np.mean([item["selected_rank_of_four"] for item in records])
        ),
        "tasks": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "tasks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# mcPHASES任务级表示路由随机对照",
        "",
        "> 表示来源仅由训练参与者选择；随机对照在验证集上评估，测试集未使用。",
        "",
        f"- 可评估任务：{len(records)}",
        f"- 训练选择路由的平均归一化效用：{observed:.4f}",
        f"- 通用状态的平均归一化效用：{general:.4f}",
        f"- 均匀随机任务路由期望：{random_expectation:.4f}",
        f"- 训练选择相对随机期望：{observed-random_expectation:+.4f}",
        f"- 随机路由置换p值（{args.replicates:,}次）：{p_value:.6f}",
        f"- 相对通用状态：{positive_vs_general}胜/{equal_to_general}平/{negative_vs_general}负",
        f"- 训练选择来源在四种表示中的平均名次：{summary['mean_selected_rank_of_four']:.3f}",
        "",
        "| 任务 | 训练选择 | 选择效用 | 通用效用 | 四表示名次 |",
        "|---|---|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            f"| {item['task_id']} | {item['selected_source']} | "
            f"{item['selected_mean_utility']:.4f} | "
            f"{item['general_mean_utility']:.4f} | "
            f"{item['selected_rank_of_four']:.1f} |"
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
