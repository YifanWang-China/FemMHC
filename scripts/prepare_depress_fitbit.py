#!/usr/bin/env python
"""Prepare female DEPRESS Fitbit streams and questionnaire windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from femmhc.data import prepare_depress_fitbit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fitbit-root", type=Path)
    parser.add_argument("--history-days", type=int, default=28)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--minimum-observed-minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_depress_fitbit(
        args.source_dir,
        args.output_dir,
        fitbit_root=args.fitbit_root,
        history_days=args.history_days,
        minimum_history_days=args.minimum_history_days,
        minimum_observed_minutes=args.minimum_observed_minutes,
        seed=args.seed,
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
