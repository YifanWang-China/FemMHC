"""Aggregate OpenMHC 32-task retention diagnostics across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_openmhc_32_tasks import PRIMARY_METRICS, TASK_NAMES_ZH


def parse_result(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("result must be SEED=PATH")
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def primary(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["metric"].isin(PRIMARY_METRICS)].copy()
    return frame[["task", "task_type", "metric", "n_test", "value"]].rename(
        columns={"value": value_name}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--result", action="append", type=parse_result, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = primary(args.baseline, "openmhc")
    seeds = [seed for seed, _ in args.result]
    seed_frames: list[pd.DataFrame] = []
    seed_summary: dict[str, object] = {}
    for seed, path in args.result:
        candidate = primary(path, "femmhc")
        merged = baseline.merge(candidate[["task", "femmhc"]], on="task")
        merged.insert(0, "seed", seed)
        merged["delta"] = merged["femmhc"] - merged["openmhc"]
        seed_frames.append(merged)
        scorable = np.isfinite(merged["openmhc"]) & np.isfinite(merged["femmhc"])
        delta = merged.loc[scorable, "delta"]
        seed_summary[str(seed)] = {
            "evaluable_tasks": int(scorable.sum()),
            "openmhc_mean": float(merged.loc[scorable, "openmhc"].mean()),
            "femmhc_mean": float(merged.loc[scorable, "femmhc"].mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
            "wins": int((delta > 0).sum()),
            "losses": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
        }

    combined = pd.concat(seed_frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for task, group in combined.groupby("task", sort=False):
        first = group.iloc[0]
        scorable = np.isfinite(group["openmhc"]) & np.isfinite(group["femmhc"])
        values = group.loc[scorable, "femmhc"]
        deltas = group.loc[scorable, "delta"]
        rows.append(
            {
                "task": task,
                "task_chinese": TASK_NAMES_ZH.get(task, task),
                "task_type": first["task_type"],
                "metric": first["metric"],
                "n_test": int(first["n_test"]),
                "openmhc": float(first["openmhc"]),
                "femmhc_mean": float(values.mean()) if len(values) else np.nan,
                "femmhc_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "mean_delta": float(deltas.mean()) if len(deltas) else np.nan,
                "improved_seeds": int((deltas > 0).sum()),
                "evaluable_seeds": int(len(deltas)),
            }
        )
    task_frame = pd.DataFrame(rows)
    scorable_tasks = task_frame["mean_delta"].notna()
    report = {
        "format_version": 1,
        "seeds": seeds,
        "seed_count": len(seeds),
        "task_list_size": len(task_frame),
        "evaluable_tasks": int(scorable_tasks.sum()),
        "per_seed": seed_summary,
        "femmhc_mean_across_seeds": float(
            np.mean([item["femmhc_mean"] for item in seed_summary.values()])
        ),
        "femmhc_std_across_seeds": float(
            np.std(
                [item["femmhc_mean"] for item in seed_summary.values()], ddof=1
            )
        ),
        "mean_delta_across_seeds": float(
            np.mean([item["mean_delta"] for item in seed_summary.values()])
        ),
        "tasks_improved_on_mean": int((task_frame["mean_delta"] > 0).sum()),
        "tasks_worse_on_mean": int((task_frame["mean_delta"] < 0).sum()),
        "tasks_improved_all_seeds": int(
            task_frame["improved_seeds"].eq(task_frame["evaluable_seeds"])
            .where(task_frame["evaluable_seeds"] > 0, False)
            .sum()
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_frame.to_csv(
        args.output_dir / "OpenMHC_32项三种子逐项结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "OpenMHC_32项三种子汇总.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
