"""Prepare female minute-level NHANES activity and sleep-wear channels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from femmhc.data import prepare_nhanes_female


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-age", type=float, default=12.0)
    parser.add_argument("--minimum-valid-minutes", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_nhanes_female(
        args.source_dir,
        args.output_dir,
        minimum_age=args.minimum_age,
        minimum_valid_minutes=args.minimum_valid_minutes,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
