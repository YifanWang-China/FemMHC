"""Cache OpenMHC or FemMHC daily embeddings for the 32-task benchmark.

The cache layout is intentionally identical to OpenMHC's bundled LSM2 method:
``embeddings.npy``, ``user_ids.npy`` and ``dates.npy``.  Per-participant shards
make the expensive minute-level extraction resumable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from downstream_evaluation.data.loader import DataLoader
from downstream_evaluation.data.provider import lookup_filename
from downstream_evaluation.data.splits import load_split_file
from downstream_evaluation.models.lsm2.model import _build_transforms
from femmhc import FemMHCEncoder, OPENMHC_SENSOR_DESCRIPTORS, SensorBatch
from openmhc._constants import BENCHMARK_TASKS
from openmhc._evaluate import _DatasetPaths
from openmhc.models.lsm2.modules import LSM2Module
from openmhc.models.lsm2.utils import create_inherited_mask


def _eligible_days(data_dir: Path) -> dict[str, list[str]]:
    """Return every day needed by at least one official benchmark task."""

    paths = _DatasetPaths.from_root(str(data_dir))
    split_users = load_split_file(paths.splits_file)
    cohort_users: set[str] = set()
    for users in split_users.values():
        cohort_users.update(str(user) for user in users)

    lookup_path = (
        Path(paths.root)
        / "processed"
        / lookup_filename("daily", full_history=False)
    )
    lookup = pd.read_parquet(
        lookup_path,
        columns=["user_id", "date", *BENCHMARK_TASKS],
    )
    valid_any = np.zeros(len(lookup), dtype=bool)
    for task in BENCHMARK_TASKS:
        values = lookup[task].to_numpy()
        if np.issubdtype(values.dtype, np.floating):
            valid = ~(np.isnan(values) | (values == -1.0))
        else:
            valid = values != -1
        valid_any |= valid

    eligible: dict[str, list[str]] = defaultdict(list)
    selected = lookup.loc[valid_any, ["user_id", "date"]]
    for user_id, date in zip(
        selected["user_id"].astype(str),
        selected["date"].astype(str),
    ):
        if user_id in cohort_users:
            eligible[user_id].append(date[:10])
    return {
        user_id: sorted(set(dates))
        for user_id, dates in sorted(eligible.items())
    }


def _shard_path(shard_dir: Path, user_id: str) -> Path:
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:20]
    return shard_dir / f"{digest}.npz"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_encoder(args: argparse.Namespace, device: torch.device):
    source = LSM2Module.load_from_checkpoint(
        str(args.checkpoint),
        map_location="cpu",
    )
    if args.model == "openmhc":
        model = source.model
        checkpoint_stage = "openmhc_lsm2"
        checkpoint_step = 0
    else:
        if args.femmhc_checkpoint is None:
            raise ValueError("--femmhc-checkpoint is required for --model femmhc")
        artifact = torch.load(
            args.femmhc_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model = FemMHCEncoder(
            source.model,
            freeze_backbone=True,
            internal_adapter_rank=int(artifact.get("internal_adapter_rank", 0)),
            internal_adapter_layers=int(artifact.get("internal_adapter_layers", 0)),
        )
        model.load_state_dict(artifact["student_state_dict"])
        checkpoint_stage = str(artifact.get("stage", "unknown"))
        checkpoint_step = int(artifact.get("steps", 0))
    del source
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, checkpoint_stage, checkpoint_step


def _transform_days(values: np.ndarray, transforms: list) -> torch.Tensor:
    days = []
    for value in values:
        day = torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32)
        for transform in transforms:
            day = transform(day)
        days.append(day)
    return torch.stack(days)


def _encode_openmhc(
    model: torch.nn.Module,
    values: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    values = values.to(device, non_blocking=True)
    inherited = create_inherited_mask(values, patch_size=model.patch_size)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        latent, mask = model.forward_encoder_dense(values, inherited)
    observed = mask == 0
    usable = observed.any(dim=1)
    denominator = observed.sum(dim=1, keepdim=True).clamp_min(1)
    pooled = (
        (latent.float() * observed.unsqueeze(-1)).sum(dim=1)
        / denominator.float()
    )
    return pooled.cpu().numpy().astype(np.float32), usable.cpu().numpy()


def _encode_femmhc(
    model: FemMHCEncoder,
    values: torch.Tensor,
    device: torch.device,
    *,
    branch: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = values.to(device, non_blocking=True)
    batch = SensorBatch(values, OPENMHC_SENSOR_DESCRIPTORS)
    # Eligible OpenMHC days should contain usable patches.  Filter defensively
    # so one pathological day cannot abort a resumable multi-hour extraction.
    from femmhc import build_patch_missing_mask

    missing = build_patch_missing_mask(
        batch,
        patch_size=model.patch_size,
        min_observed_fraction=model.min_observed_fraction,
    )
    usable = (~missing).any(dim=1)
    output = np.full((len(values), model.embed_dim), np.nan, dtype=np.float32)
    if bool(usable.any()):
        selected = SensorBatch(values[usable], OPENMHC_SENSOR_DESCRIPTORS)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            pooled = (
                model.forward_native(selected).pooled
                if branch == "native"
                else model(selected).pooled
            )
        output[usable.cpu().numpy()] = pooled.float().cpu().numpy()
    return output, usable.cpu().numpy()


def _consolidate(
    output_dir: Path,
    eligible: dict[str, list[str]],
) -> tuple[int, int]:
    shard_dir = output_dir / "participant_shards"
    embeddings: list[np.ndarray] = []
    user_ids: list[str] = []
    dates: list[str] = []
    skipped_days = 0
    for user_id in sorted(eligible):
        shard_path = _shard_path(shard_dir, user_id)
        if not shard_path.is_file():
            raise RuntimeError(f"missing participant shard: {user_id}")
        with np.load(shard_path, allow_pickle=False) as shard:
            shard_embeddings = shard["embeddings"].astype(np.float32)
            shard_dates = shard["dates"].astype(str).tolist()
            skipped_days += int(shard["skipped_days"].item())
        embeddings.append(shard_embeddings)
        user_ids.extend([user_id] * len(shard_embeddings))
        dates.extend(shard_dates)

    matrix = np.concatenate(embeddings, axis=0)
    np.save(output_dir / "embeddings.npy", matrix)
    np.save(output_dir / "user_ids.npy", np.asarray(user_ids, dtype=object))
    np.save(output_dir / "dates.npy", np.asarray(dates, dtype=object))
    return len(matrix), skipped_days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("openmhc", "femmhc"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--femmhc-checkpoint", type=Path)
    parser.add_argument(
        "--femmhc-branch",
        choices=("adapted", "native"),
        default="adapted",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    device = torch.device(args.device)
    eligible = _eligible_days(args.data_dir)
    if args.max_users is not None:
        eligible = dict(list(eligible.items())[: args.max_users])
    expected_days = sum(len(dates) for dates in eligible.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "participant_shards"
    shard_dir.mkdir(exist_ok=True)

    extraction_config = {
        "format_version": 1,
        "model": args.model,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": _file_sha256(args.checkpoint),
        "femmhc_checkpoint": (
            str(args.femmhc_checkpoint.resolve())
            if args.femmhc_checkpoint is not None
            else None
        ),
        "femmhc_checkpoint_sha256": (
            _file_sha256(args.femmhc_checkpoint)
            if args.femmhc_checkpoint is not None
            else None
        ),
        "femmhc_branch": args.femmhc_branch if args.model == "femmhc" else None,
        "data_dir": str(args.data_dir.resolve()),
        "benchmark_tasks": list(BENCHMARK_TASKS),
        "precision": "bfloat16" if device.type == "cuda" else "float32",
    }
    config_path = args.output_dir / "extraction_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != extraction_config and any(shard_dir.iterdir()) and not args.force:
            raise ValueError(
                "embedding cache configuration changed; choose a new output directory "
                "or pass --force to rebuild every participant shard"
            )
    config_path.write_text(
        json.dumps(extraction_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    model, checkpoint_stage, checkpoint_step = _load_encoder(args, device)
    paths = _DatasetPaths.from_root(str(args.data_dir))
    transforms = _build_transforms(
        Path(paths.daily_hf).parent / "normalization_stats.json"
    )
    loader = DataLoader(str(args.data_dir), resolution="minute")

    started = time.perf_counter()
    resumed_users = 0
    encoded_users = 0
    for user_index, (user_id, requested_dates) in enumerate(eligible.items(), start=1):
        shard_path = _shard_path(shard_dir, user_id)
        if shard_path.is_file() and not args.force:
            resumed_users += 1
            continue
        values, used_dates = loader.participant_minute(user_id, requested_dates)
        participant_embeddings: list[np.ndarray] = []
        participant_dates: list[str] = []
        skipped = 0
        for start in range(0, len(values), args.batch_size):
            stop = min(start + args.batch_size, len(values))
            transformed = _transform_days(values[start:stop], transforms)
            if args.model == "openmhc":
                pooled, usable = _encode_openmhc(model, transformed, device)
            else:
                pooled, usable = _encode_femmhc(
                    model,
                    transformed,
                    device,
                    branch=args.femmhc_branch,
                )
            participant_embeddings.append(pooled[usable])
            participant_dates.extend(
                date for date, keep in zip(used_dates[start:stop], usable) if keep
            )
            skipped += int((~usable).sum())
        if participant_embeddings:
            matrix = np.concatenate(participant_embeddings, axis=0)
        else:
            matrix = np.empty((0, 384), dtype=np.float32)
        np.savez_compressed(
            shard_path,
            embeddings=matrix,
            dates=np.asarray(participant_dates, dtype="U10"),
            skipped_days=np.asarray(skipped, dtype=np.int64),
        )
        encoded_users += 1
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "event": "participant",
                    "participant": user_index,
                    "participants": len(eligible),
                    "days": len(matrix),
                    "skipped_days": skipped,
                    "elapsed_seconds": elapsed,
                }
            ),
            flush=True,
        )

    encoded_days, skipped_days = _consolidate(args.output_dir, eligible)
    report = {
        "format_version": 1,
        "model": args.model,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "femmhc_checkpoint": (
            str(args.femmhc_checkpoint.resolve())
            if args.femmhc_checkpoint is not None
            else None
        ),
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_step": checkpoint_step,
        "femmhc_branch": args.femmhc_branch if args.model == "femmhc" else None,
        "data_dir": str(args.data_dir.resolve()),
        "benchmark_tasks": len(BENCHMARK_TASKS),
        "eligible_participants": len(eligible),
        "expected_days": expected_days,
        "encoded_days": encoded_days,
        "skipped_days": skipped_days,
        "resumed_participants": resumed_users,
        "encoded_participants_this_run": encoded_users,
        "embedding_dimension": int(model.embed_dim),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
