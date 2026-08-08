#!/usr/bin/env python
"""Aggregate matched nested-CV reports across FemMHC model seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.reports) < 2:
        raise ValueError("aggregation needs at least two seed reports")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    reference = reports[0]
    for report in reports[1:]:
        for field in ("cohort", "protocol", "outer_folds", "inner_folds", "fold_seed"):
            if report[field] != reference[field]:
                raise ValueError(f"reports disagree on {field}")
        if [item["task"] for item in report["comparisons"]] != [
            item["task"] for item in reference["comparisons"]
        ]:
            raise ValueError("reports contain different tasks or task order")

    summaries = []
    for comparison in reference["comparisons"]:
        task = comparison["task"]
        kind = comparison["kind"]
        metric = "auprc" if kind == "classification" else "mae"
        native_values = []
        adapted_values = []
        improvements = []
        confidence_interval_positive = 0
        seed_rows = []
        for path, report in zip(args.reports, reports):
            native = next(
                item
                for item in report["results"]
                if item["task"] == task and item["representation"] == "native_openmhc"
            )
            adapted = next(
                item
                for item in report["results"]
                if item["task"] == task and item["representation"] != "native_openmhc"
            )
            paired = next(
                item for item in report["comparisons"] if item["task"] == task
            )["participant_paired_bootstrap"]
            native_value = float(native["embedding_probe"][metric])
            adapted_value = float(adapted["embedding_probe"][metric])
            improvement = (
                adapted_value - native_value
                if kind == "classification"
                else native_value - adapted_value
            )
            native_values.append(native_value)
            adapted_values.append(adapted_value)
            improvements.append(improvement)
            confidence_interval_positive += int(
                float(paired["improvement_ci_low"]) > 0
            )
            seed_rows.append(
                {
                    "report": str(path),
                    "adapted_representation": adapted["representation"],
                    "native": native_value,
                    "adapted": adapted_value,
                    "absolute_improvement": improvement,
                    "bootstrap_ci_low": float(paired["improvement_ci_low"]),
                    "bootstrap_ci_high": float(paired["improvement_ci_high"]),
                    "bootstrap_probability_improved": float(
                        paired["probability_improved"]
                    ),
                }
            )
        native_mean = float(np.mean(native_values))
        adapted_mean = float(np.mean(adapted_values))
        absolute_mean = float(np.mean(improvements))
        relative_improvement = (
            100.0 * absolute_mean / native_mean if native_mean != 0 else None
        )
        summaries.append(
            {
                "task": task,
                "kind": kind,
                "primary_metric": metric,
                "native_mean": native_mean,
                "native_std": float(np.std(native_values, ddof=1)),
                "adapted_mean": adapted_mean,
                "adapted_std": float(np.std(adapted_values, ddof=1)),
                "absolute_improvement_mean": absolute_mean,
                "absolute_improvement_std": float(np.std(improvements, ddof=1)),
                "relative_improvement_percent": relative_improvement,
                "improved_seeds": int(np.sum(np.asarray(improvements) > 0)),
                "confidence_interval_positive_seeds": confidence_interval_positive,
                "seeds": seed_rows,
            }
        )

    output = {
        "format_version": 1,
        "cohort": reference["cohort"],
        "protocol": reference["protocol"],
        "outer_folds": reference["outer_folds"],
        "inner_folds": reference["inner_folds"],
        "fold_seed": reference["fold_seed"],
        "model_seed_reports": [str(path) for path in args.reports],
        "task_summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
