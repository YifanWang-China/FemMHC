from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from femmhc.data import (
    fit_pregnancy_ga_normalization,
    prepare_pregnancy_ga_processed_pickle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the official processed pregnancy pickle once."
    )
    parser.add_argument("pickle", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_pregnancy_ga_processed_pickle(
        args.pickle,
        args.output_dir,
        seed=args.seed,
    )
    normalization = fit_pregnancy_ga_normalization(args.output_dir)
    print(
        json.dumps(
            {"summary": asdict(summary), "normalization": normalization},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
