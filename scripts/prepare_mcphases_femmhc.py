"""Prepare the restricted mcPHASES archive for FemMHC experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from femmhc.data import prepare_mcphases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Build labels, daily context, schema and participant splits without minute arrays.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard generated sensor progress and rebuild every sensor channel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_mcphases(
        args.archive,
        args.output_dir,
        include_sensors=not args.labels_only,
        seed=args.seed,
        resume=not args.no_resume,
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
