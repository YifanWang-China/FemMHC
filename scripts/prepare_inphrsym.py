#!/usr/bin/env python
"""Prepare female inPHRsym daily streams and next-day affective targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from femmhc.data import prepare_inphrsym


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-observed-minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_inphrsym(
        args.source_dir,
        args.output_dir,
        minimum_observed_minutes=args.minimum_observed_minutes,
        seed=args.seed,
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
