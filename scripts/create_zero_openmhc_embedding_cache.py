#!/usr/bin/env python
"""Create a zero second-view cache aligned to a native OpenMHC cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    embeddings = np.load(args.native_cache / "embeddings.npy", mmap_mode="r")
    user_ids = np.load(args.native_cache / "user_ids.npy", allow_pickle=True)
    dates = np.load(args.native_cache / "dates.npy", allow_pickle=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", np.zeros(embeddings.shape, dtype=np.float32))
    np.save(args.output_dir / "user_ids.npy", user_ids)
    np.save(args.output_dir / "dates.npy", dates)
    report = {
        "format_version": 1,
        "role": "zero_second_view_control",
        "native_cache": str(args.native_cache.resolve()),
        "shape": list(embeddings.shape),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
