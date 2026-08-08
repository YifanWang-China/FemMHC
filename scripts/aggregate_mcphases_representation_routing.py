#!/usr/bin/env python
"""Aggregate fixed train-selected mcPHASES representation routing across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def sample_sd(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-result",
        action="append",
        required=True,
        help="SEED=summary.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs: dict[int, dict[str, object]] = {}
    for raw in args.seed_result:
        seed_text, path_text = raw.split("=", 1)
        seed = int(seed_text)
        inputs[seed] = json.loads(Path(path_text).read_text(encoding="utf-8"))
    seeds = sorted(inputs)
    if len(seeds) < 2:
        raise ValueError("at least two seed results are required")
    task_ids = tuple(inputs[seeds[0]]["tasks"])
    if any(tuple(inputs[seed]["tasks"]) != task_ids for seed in seeds[1:]):
        raise ValueError("seed task registries differ")

    rows: list[dict[str, object]] = []
    for task_id in task_ids:
        records = [inputs[seed]["tasks"][task_id] for seed in seeds]
        selected_sources = {
            record["selection"]["selected"]["representation_source"]
            for record in records
        }
        if len(selected_sources) != 1:
            raise ValueError(f"selection changed across seeds for {task_id}")
        selected_source = str(selected_sources.pop())
        deltas = [
            record["selected_oriented_delta_vs_general"] for record in records
        ]
        evaluable_deltas = [float(value) for value in deltas if value is not None]
        selected_metrics = [
            float(record["selected_validation_primary_metric"])
            for record in records
            if record["selected_validation_primary_metric"] is not None
        ]
        general_metrics = [
            float(record["general_validation_primary_metric"])
            for record in records
            if record["general_validation_primary_metric"] is not None
        ]
        rows.append(
            {
                "task_id": task_id,
                "domain": records[0]["domain"],
                "kind": records[0]["kind"],
                "primary_metric": records[0]["primary_metric"],
                "selected_source": selected_source,
                "train_oof_metric": records[0]["selection"]["selected"][
                    "oof_primary_metric"
                ],
                "selected_validation_mean": (
                    mean(selected_metrics) if selected_metrics else None
                ),
                "selected_validation_sample_sd": (
                    sample_sd(selected_metrics) if selected_metrics else None
                ),
                "general_validation_mean": (
                    mean(general_metrics) if general_metrics else None
                ),
                "general_validation_sample_sd": (
                    sample_sd(general_metrics) if general_metrics else None
                ),
                "oriented_delta_by_seed": dict(zip(seeds, deltas, strict=True)),
                "mean_oriented_delta_vs_general": (
                    mean(evaluable_deltas) if evaluable_deltas else None
                ),
                "positive_seeds": sum(value > 0 for value in evaluable_deltas),
                "tie_seeds": sum(value == 0 for value in evaluable_deltas),
                "negative_seeds": sum(value < 0 for value in evaluable_deltas),
            }
        )

    evaluable = [row for row in rows if row["mean_oriented_delta_vs_general"] is not None]
    summary = {
        "format_version": 1,
        "split": "validation",
        "selection_split": "train_only_seed42",
        "seeds": seeds,
        "test_used": False,
        "tasks": rows,
        "aggregate": {
            "evaluable_tasks": len(evaluable),
            "positive_mean_delta": sum(
                float(row["mean_oriented_delta_vs_general"]) > 0 for row in evaluable
            ),
            "ties": sum(
                float(row["mean_oriented_delta_vs_general"]) == 0 for row in evaluable
            ),
            "negative_mean_delta": sum(
                float(row["mean_oriented_delta_vs_general"]) < 0 for row in evaluable
            ),
            "positive_all_seeds": sum(
                int(row["positive_seeds"]) == len(seeds) for row in evaluable
            ),
            "positive_at_least_two_seeds": sum(
                int(row["positive_seeds"]) >= 2 for row in evaluable
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    fields = [
        "task_id",
        "domain",
        "kind",
        "primary_metric",
        "selected_source",
        "train_oof_metric",
        "selected_validation_mean",
        "selected_validation_sample_sd",
        "general_validation_mean",
        "general_validation_sample_sd",
        "mean_oriented_delta_vs_general",
        "positive_seeds",
        "tie_seeds",
        "negative_seeds",
    ]
    with (args.output_dir / "tasks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# mcPHASES任务级表示路由三种子汇总",
        "",
        "表示来源和正则化由seed42训练参与者选择，并原样固定到三个正式主干；测试集未使用。",
        "",
        "| 任务 | 训练选择表示 | 验证均值 | 通用状态均值 | 定向差值 | 正/平/负种子 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["mean_oriented_delta_vs_general"] is None:
            selected_text = general_text = delta_text = "NA"
        else:
            selected_text = f"{float(row['selected_validation_mean']):.4f}"
            general_text = f"{float(row['general_validation_mean']):.4f}"
            delta_text = f"{float(row['mean_oriented_delta_vs_general']):+.4f}"
        lines.append(
            "| {task} | {source} | {selected} | {general} | {delta} | {positive}/{tie}/{negative} |".format(
                task=row["task_id"],
                source=row["selected_source"],
                selected=selected_text,
                general=general_text,
                delta=delta_text,
                positive=row["positive_seeds"],
                tie=row["tie_seeds"],
                negative=row["negative_seeds"],
            )
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
