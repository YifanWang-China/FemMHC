"""Aggregate multiple FemMHC stages under the matched OpenMHC protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_openmhc_32_tasks import (
    METRIC_NAMES_ZH,
    PRIMARY_METRICS,
    TASK_NAMES_ZH,
    TASK_TYPE_ZH,
    _markdown_table,
)


def _parse_metric(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("method label cannot be empty")
    return label.strip(), Path(path)


def _primary(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["metric"].isin(PRIMARY_METRICS)].copy()
    if len(frame) != frame["task"].nunique():
        raise ValueError(f"{path} does not contain one primary metric per task")
    return frame[["task", "task_type", "metric", "n_test", "value"]].rename(
        columns={"value": label}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="append", type=_parse_metric, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    labels = [label for label, _ in args.metrics]
    if len(labels) != len(set(labels)):
        raise ValueError("method labels must be unique")
    if args.baseline not in labels:
        raise ValueError("baseline label must be one of --metrics labels")

    merged: pd.DataFrame | None = None
    for label, path in args.metrics:
        current = _primary(path, label)
        if merged is None:
            merged = current
        else:
            merged = merged.merge(
                current[["task", label]],
                on="task",
                validate="one_to_one",
            )
    assert merged is not None
    merged.insert(1, "任务中文", merged["task"].map(TASK_NAMES_ZH))
    merged["task_type"] = merged["task_type"].map(TASK_TYPE_ZH)
    merged["metric"] = merged["metric"].map(METRIC_NAMES_ZH)
    merged = merged.rename(
        columns={
            "task": "任务代码",
            "task_type": "任务类型",
            "metric": "主要指标",
            "n_test": "测试人数",
        }
    )

    summary: dict[str, object] = {
        "format_version": 1,
        "baseline": args.baseline,
        "task_list_size": len(merged),
        "methods": {},
    }
    baseline_values = merged[args.baseline]
    for label in labels:
        scorable = np.isfinite(baseline_values) & np.isfinite(merged[label])
        delta = merged.loc[scorable, label] - baseline_values[scorable]
        summary["methods"][label] = {
            "evaluable_tasks": int(scorable.sum()),
            "mean_primary_metric": float(merged.loc[scorable, label].mean()),
            "mean_delta_vs_baseline": float(delta.mean()),
            "median_delta_vs_baseline": float(delta.median()),
            "wins_vs_baseline": int((delta > 0).sum()),
            "losses_vs_baseline": int((delta < 0).sum()),
            "ties_vs_baseline": int((delta == 0).sum()),
        }
        if label != args.baseline:
            merged[f"{label}相对{args.baseline}绝对变化"] = merged[label] - baseline_values

    for previous, current in zip(labels, labels[1:]):
        merged[f"{current}相对{previous}阶段变化"] = merged[current] - merged[previous]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "三阶段32项消融.csv", index=False)
    (args.output_dir / "三阶段32项消融汇总.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    display = merged[["任务中文", "任务类型", "主要指标", "测试人数", *labels]].copy()
    for label in labels:
        display[label] = display[label].map(
            lambda value: "不可评估" if not np.isfinite(value) else f"{value:.4f}"
        )
    lines = [
        "# OpenMHC 与 FemMHC 各阶段的 32 项消融",
        "",
        _markdown_table(display),
        "",
        "## 汇总",
        "",
    ]
    for label in labels:
        item = summary["methods"][label]
        lines.append(
            f"- {label}：{item['evaluable_tasks']} 项可评估，平均主要指标 "
            f"{item['mean_primary_metric']:.4f}；相对 {args.baseline} 为 "
            f"{item['wins_vs_baseline']} 胜/{item['losses_vs_baseline']} 负/"
            f"{item['ties_vs_baseline']} 平。"
        )
    (args.output_dir / "三阶段32项消融.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
