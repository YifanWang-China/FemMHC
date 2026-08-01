"""Crash-safe checkpoint helpers for long FemMHC runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.random.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])


def save_training_checkpoint(path: Path, artifact: dict[str, Any]) -> None:
    """Atomically update a checkpoint and a small human-readable status file."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(artifact)
    artifact["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(path)
    report = {
        key: value
        for key, value in artifact.items()
        if not key.endswith("state_dict")
        and not key.endswith("random_state")
        and key != "cuda_random_state"
    }
    path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


__all__ = ["capture_rng_state", "restore_rng_state", "save_training_checkpoint"]
