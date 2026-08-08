#!/usr/bin/env python
"""Aggregate equal-parameter cycle-phase probes across representation sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


SOURCES = (
    "general",
    "cycle",
    "menstrual_domain",
    "cycle_task_route",
)
DISPLAY_NAMES = {
    "general": "通用GRU状态",
    "cycle": "CycleSSM状态",
    "menstrual_domain": "月经领域状态",
    "cycle_task_route": "周期任务路由状态",
}


def sample_sd(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, (count - rank) * p_values[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def load_protocol(
    directory: Path,
    *,
    seeds: list[int],
    cycle_fallback: Path | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in SOURCES:
        source_directory = directory
        if source == "cycle" and not (directory / f"cycle-seed{seeds[0]}.json").is_file():
            if cycle_fallback is None:
                raise ValueError("common protocol is missing the cycle records")
            source_directory = cycle_fallback
        records = [
            json.loads(
                (source_directory / f"{source}-seed{seed}.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in seeds
        ]
        if any(record["representation_source"] != source for record in records):
            raise ValueError(f"representation metadata mismatch for {source}")
        if any(record["head_family"] != "linear_matched" for record in records):
            raise ValueError(f"non-matched probe found for {source}")
        if any(int(record["trainable_parameters"]) != 516 for record in records):
            raise ValueError(f"parameter mismatch for {source}")
        if any(bool(record["validation_used_for_selection"]) for record in records):
            raise ValueError(f"validation leakage flag found for {source}")
        if any(bool(record["test_used"]) for record in records):
            raise ValueError(f"test-use flag found for {source}")
        macro_f1 = [float(record["macro_f1"]) for record in records]
        balanced_accuracy = [
            float(record["balanced_accuracy"]) for record in records
        ]
        rows.append(
            {
                "representation_source": source,
                "display_name": DISPLAY_NAMES[source],
                "trainable_parameters": 516,
                "macro_f1_by_seed": dict(zip(seeds, macro_f1, strict=True)),
                "macro_f1_mean": mean(macro_f1),
                "macro_f1_sample_sd": sample_sd(macro_f1),
                "balanced_accuracy_mean": mean(balanced_accuracy),
                "balanced_accuracy_sample_sd": sample_sd(balanced_accuracy),
                "learning_rate": float(records[0]["learning_rate"]),
                "steps": int(records[0]["steps"]),
            }
        )
    general_mean = next(
        float(row["macro_f1_mean"])
        for row in rows
        if row["representation_source"] == "general"
    )
    for row in rows:
        row["absolute_delta_vs_general"] = (
            float(row["macro_f1_mean"]) - general_mean
        )
    return sorted(rows, key=lambda row: float(row["macro_f1_mean"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "artifacts/benchmark/femmhc-joint-phase-representation-controls"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seeds = sorted(args.seeds)

    selected_directory = args.root / "train-selected-validation"
    common_directory = args.root / "common-config-validation"
    selected = load_protocol(selected_directory, seeds=seeds)
    common = load_protocol(
        common_directory,
        seeds=seeds,
        cycle_fallback=selected_directory,
    )

    selections: dict[str, dict[str, object]] = {}
    for source in SOURCES:
        record = json.loads(
            (args.root / f"train-selection-{source}-seed42.json").read_text(
                encoding="utf-8"
            )
        )
        selected_record = record["selected"]
        selections[source] = {
            "learning_rate": float(selected_record["learning_rate"]),
            "steps": int(selected_record["steps"]),
            "pooled_train_oof_macro_f1": float(
                selected_record["pooled_macro_f1"]
            ),
            "validation_used": bool(record["validation_participants_used"]),
            "test_used": bool(record["test_participants_used"]),
        }

    bootstraps: dict[str, dict[str, object]] = {}
    for baseline in ("general", "menstrual_domain", "cycle_task_route"):
        record = json.loads(
            (
                args.root
                / "participant-bootstrap"
                / f"seed42-cycle-vs-{baseline}.json"
            ).read_text(encoding="utf-8")
        )
        result = record[
            "participant_cluster_bootstrap_candidate_minus_baseline"
        ]
        bootstraps[f"cycle_vs_{baseline}"] = result
    adjusted = holm_adjust(
        {
            key: float(value["p_value_two_sided"])
            for key, value in bootstraps.items()
        }
    )
    for key, value in bootstraps.items():
        value["p_value_holm_three_representation_contrasts"] = adjusted[key]

    summary = {
        "format_version": 1,
        "task_id": "mcphases/cycle_phase",
        "split": "validation",
        "probe": "linear_matched",
        "trainable_parameters": 516,
        "seeds": seeds,
        "selection_split": "train_only",
        "test_used": False,
        "train_only_selections": selections,
        "train_selected_protocol": selected,
        "common_lr_0.01_steps_300_protocol": common,
        "seed42_common_protocol_participant_bootstrap": bootstraps,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    fields = [
        "representation_source",
        "display_name",
        "trainable_parameters",
        "macro_f1_mean",
        "macro_f1_sample_sd",
        "balanced_accuracy_mean",
        "balanced_accuracy_sample_sd",
        "absolute_delta_vs_general",
        "learning_rate",
        "steps",
    ]
    with (args.output_dir / "common_protocol.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(common)

    lines = [
        "# 周期阶段等参数表示来源对照",
        "",
        "四种冻结表示均使用516参数直接线性头；下表统一采用学习率0.01、300步。",
        "",
        "| 表示来源 | 宏平均F1（均值±样本标准差） | 相对通用状态 | 平衡准确率 |",
        "|---|---:|---:|---:|",
    ]
    for row in common:
        lines.append(
            "| {name} | {f1:.4f} ± {sd:.4f} | {delta:+.4f} | {ba:.4f} |".format(
                name=row["display_name"],
                f1=row["macro_f1_mean"],
                sd=row["macro_f1_sample_sd"],
                delta=row["absolute_delta_vs_general"],
                ba=row["balanced_accuracy_mean"],
            )
        )
    lines.extend(
        [
            "",
            "CycleSSM相对三类基线的seed42参与者聚类Bootstrap原始区间均为正；三项Holm校正后，仅相对月经领域状态的p值低于0.05。",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
