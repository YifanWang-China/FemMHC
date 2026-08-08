"""Build the FemMHC local dataset inventory and lightweight schema report."""

from __future__ import annotations

import argparse
from pathlib import Path

from femmhc.data.inventory import profile_catalog, write_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--curated-root", type=Path, default=Path("数据集")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/data_profile")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = profile_catalog(args.data_root, args.curated_root)
    write_inventory(
        profiles,
        args.output_dir / "dataset_inventory.json",
        args.output_dir / "dataset_inventory.md",
    )
    ready = sum(profile.readiness == "ready" for profile in profiles)
    partial = sum(profile.readiness == "partial" for profile in profiles)
    print(
        f"profiled={len(profiles)} ready={ready} partial={partial} "
        f"output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
