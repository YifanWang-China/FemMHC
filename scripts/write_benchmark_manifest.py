from __future__ import annotations

import argparse
from pathlib import Path

from femmhc.benchmark import write_benchmark_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark"))
    args = parser.parse_args()
    paths = write_benchmark_manifest(args.output_dir)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
