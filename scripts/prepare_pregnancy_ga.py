from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from femmhc.data import (
    fit_pregnancy_ga_normalization,
    prepare_pregnancy_ga_clock,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the pregnancy gestational-age actigraphy cohort."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = prepare_pregnancy_ga_clock(
        args.archive,
        args.output_dir,
        days=args.days,
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
