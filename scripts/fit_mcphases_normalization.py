"""Fit leakage-safe mcPHASES normalization statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from femmhc.data import fit_mcphases_normalization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    args = parser.parse_args()
    result = fit_mcphases_normalization(args.processed_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
