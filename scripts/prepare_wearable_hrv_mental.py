"""Prepare the female subset of Wearable HRV and Sleep."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from femmhc.data import prepare_wearable_hrv_mental


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-windows-per-day", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_wearable_hrv_mental(
        args.source_dir,
        args.output_dir,
        minimum_windows_per_day=args.minimum_windows_per_day,
        seed=args.seed,
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
