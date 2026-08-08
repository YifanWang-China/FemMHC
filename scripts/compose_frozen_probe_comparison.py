#!/usr/bin/env python
"""Compose already-evaluated frozen-probe rows into a matched comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_source(value: str) -> tuple[Path, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be CSV_PATH=MODEL_NAME")
    path, model = value.rsplit("=", 1)
    return Path(path), model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=parse_source, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected: list[pd.DataFrame] = []
    task_sets: list[set[str]] = []
    for path, model in args.source:
        frame = pd.read_csv(path)
        subset = frame[frame["model"].astype(str).eq(model)].copy()
        if subset.empty:
            raise ValueError(f"model {model!r} is absent from {path}")
        selected.append(subset)
        primary = subset["is_primary"].astype(str).str.lower().eq("true")
        task_sets.append(set(subset.loc[primary, "task"]))
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError("sources do not contain the same primary task set")
    output = pd.concat(selected, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        {
            "output": str(args.output),
            "models": output["model"].drop_duplicates().tolist(),
            "primary_tasks": len(task_sets[0]),
        }
    )


if __name__ == "__main__":
    main()
