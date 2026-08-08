"""Locked multi-cohort female continual pretraining for the FemMHC adapter."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from femmhc import (
    AFFECTIVE_DAILY_SENSOR_DESCRIPTORS,
    MCPHASES_SENSOR_DESCRIPTORS,
    PREGNANCY_GA_SENSOR_DESCRIPTORS,
    WEARABLE_HRV_MENTAL_SENSOR_DESCRIPTORS,
    PatchReconstructionHead,
    SensorBatch,
    build_femmhc_encoder_from_artifact,
    drop_sensor_channels,
    mask_sensor_patches,
    masked_patch_reconstruction_loss,
    preservation_distance,
    sensor_set_consistency_loss,
)
from femmhc.checkpointing import capture_rng_state, save_training_checkpoint
from femmhc.data import (
    DEPRESSFitbitDailyDataset,
    InPHRSymDailyDataset,
    McPhasesDataset,
    PregnancyGADailyDataset,
    WearableHRVMentalDailyDataset,
)
from femmhc.multicohort import FemaleCohort, square_root_sampling_probabilities
from openmhc.models.lsm2.modules import LSM2Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--female-init", type=Path, required=True)
    parser.add_argument("--mcphases-dir", type=Path, required=True)
    parser.add_argument("--depress-dir", type=Path, required=True)
    parser.add_argument("--inphrsym-dir", type=Path, required=True)
    parser.add_argument("--hrv-dir", type=Path, required=True)
    parser.add_argument("--pregnancy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument("--female-preservation-weight", type=float, default=2.0)
    parser.add_argument("--native-preservation-weight", type=float, default=0.25)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--sensor-drop-probability", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _usable_rows(values: torch.Tensor, patch_size: int) -> torch.Tensor:
    patches = values.reshape(
        values.shape[0], values.shape[1], -1, patch_size
    )
    return torch.isfinite(patches).float().mean(dim=-1).ge(0.5).any(dim=(1, 2))


def _next_batch(
    loaders: list[DataLoader],
    iterators: list[object],
    cohort_index: int,
) -> dict[str, object]:
    try:
        return next(iterators[cohort_index])  # type: ignore[arg-type,return-value]
    except StopIteration:
        iterators[cohort_index] = iter(loaders[cohort_index])
        return next(iterators[cohort_index])  # type: ignore[arg-type,return-value]


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("steps, batch size, and save interval must be positive")
    if min(
        args.reconstruction_weight,
        args.consistency_weight,
        args.female_preservation_weight,
        args.native_preservation_weight,
    ) < 0:
        raise ValueError("objective weights must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    initialization = torch.load(args.female_init, map_location="cpu", weights_only=False)
    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    student = build_femmhc_encoder_from_artifact(
        source.model,
        initialization,
        freeze_backbone=True,
    ).to(device).train()
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    del source

    reconstruction_head = PatchReconstructionHead(
        student.embed_dim, student.patch_size
    ).to(device).train()
    if "reconstruction_head_state_dict" in initialization:
        reconstruction_head.load_state_dict(
            initialization["reconstruction_head_state_dict"]
        )

    cohorts = [
        FemaleCohort(
            "mcphases",
            McPhasesDataset(args.mcphases_dir, split="train", normalize=True),
            MCPHASES_SENSOR_DESCRIPTORS,
        ),
        FemaleCohort(
            "depress_fitbit",
            DEPRESSFitbitDailyDataset(args.depress_dir, split="train", normalize=True),
            AFFECTIVE_DAILY_SENSOR_DESCRIPTORS,
        ),
        FemaleCohort(
            "inphrsym",
            InPHRSymDailyDataset(args.inphrsym_dir, split="train", normalize=True),
            AFFECTIVE_DAILY_SENSOR_DESCRIPTORS,
        ),
        FemaleCohort(
            "wearable_hrv_mental",
            WearableHRVMentalDailyDataset(args.hrv_dir, split="train", normalize=True),
            WEARABLE_HRV_MENTAL_SENSOR_DESCRIPTORS,
        ),
        FemaleCohort(
            "pregnancy_ga_clock",
            PregnancyGADailyDataset(args.pregnancy_dir, split="train", normalize=True),
            PREGNANCY_GA_SENSOR_DESCRIPTORS,
        ),
    ]
    probabilities = square_root_sampling_probabilities(cohorts)
    loader_generators: list[torch.Generator] = []
    loaders: list[DataLoader] = []
    for index, cohort in enumerate(cohorts):
        generator = torch.Generator().manual_seed(args.seed + 1009 * index)
        loader_generators.append(generator)
        loaders.append(
            DataLoader(
                cohort.dataset,  # type: ignore[arg-type]
                batch_size=args.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
        )
    iterators: list[object] = [iter(loader) for loader in loaders]
    cohort_rng = np.random.default_rng(args.seed)

    trainable = [
        parameter
        for module in (student, reconstruction_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainable_names = [
        name for name, parameter in student.named_parameters() if parameter.requires_grad
    ]
    trainable_encoder_parameters = sum(
        parameter.numel() for parameter in student.parameters() if parameter.requires_grad
    )
    total_encoder_parameters = sum(parameter.numel() for parameter in student.parameters())

    history: list[dict[str, object]] = []
    cohort_counts = {cohort.name: 0 for cohort in cohorts}
    skipped_empty_batches = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def save(status: str, step: int) -> None:
        artifact = {
            "format_version": 1,
            "model": "FemMHC",
            "stage": "multicohort_female_continual_pretraining",
            "status": status,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "femmhc_initialization": str(args.female_init.resolve()),
            "internal_adapter_rank": int(initialization.get("internal_adapter_rank", 0)),
            "internal_adapter_layers": int(initialization.get("internal_adapter_layers", 0)),
            "seed": args.seed,
            "steps": step,
            "max_steps": args.max_steps,
            "checkpoint_selection": "final_step",
            "training_split_only": True,
            "test_split_used": False,
            "cohort_sizes": {cohort.name: len(cohort) for cohort in cohorts},
            "cohort_sampling_probabilities": {
                cohort.name: probability
                for cohort, probability in zip(cohorts, probabilities)
            },
            "cohort_batch_counts": cohort_counts,
            "skipped_empty_batches": skipped_empty_batches,
            "objective_weights": {
                "reconstruction": args.reconstruction_weight,
                "consistency": args.consistency_weight,
                "female_preservation": args.female_preservation_weight,
                "native_preservation": args.native_preservation_weight,
            },
            "trainable_encoder_parameters": trainable_encoder_parameters,
            "total_encoder_parameters": total_encoder_parameters,
            "trainable_encoder_fraction": (
                trainable_encoder_parameters / total_encoder_parameters
            ),
            "trainable_parameter_names": trainable_names,
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else None
            ),
            "student_state_dict": student.state_dict(),
            "reconstruction_head_state_dict": reconstruction_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **capture_rng_state(),
        }
        save_training_checkpoint(args.output, artifact)

    print(
        json.dumps(
            {
                "event": "locked_protocol_started",
                "device": str(device),
                "cohort_sizes": {cohort.name: len(cohort) for cohort in cohorts},
                "sampling_probabilities": {
                    cohort.name: round(probability, 6)
                    for cohort, probability in zip(cohorts, probabilities)
                },
                "trainable_encoder_parameters": trainable_encoder_parameters,
                "trainable_encoder_fraction": trainable_encoder_parameters
                / total_encoder_parameters,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    step = 0
    while step < args.max_steps:
        cohort_index = int(cohort_rng.choice(len(cohorts), p=probabilities))
        item = _next_batch(loaders, iterators, cohort_index)
        values = item["sensor_values"]  # type: ignore[index]
        present = item["channel_present"]  # type: ignore[index]
        if not isinstance(values, torch.Tensor) or not isinstance(present, torch.Tensor):
            raise TypeError("daily datasets must return tensor sensor values and masks")
        usable = _usable_rows(values, student.patch_size)
        if not bool(usable.any()):
            skipped_empty_batches += 1
            continue
        values = values[usable].to(device, non_blocking=True)
        present = present[usable].to(device, non_blocking=True)
        batch = SensorBatch(values, cohorts[cohort_index].descriptors, present)
        masked, artificial_mask = mask_sensor_patches(
            batch,
            patch_size=student.patch_size,
            mask_probability=args.mask_probability,
        )
        subset = drop_sensor_channels(
            batch,
            drop_probability=args.sensor_drop_probability,
            patch_size=student.patch_size,
            min_observed_fraction=student.min_observed_fraction,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            masked_output = student(masked)
            subset_output = student(subset)
            with torch.no_grad():
                anchor = teacher.forward_dual(batch)
            reconstruction = masked_patch_reconstruction_loss(
                reconstruction_head(masked_output.latent),
                batch.values,
                artificial_mask,
                patch_size=student.patch_size,
            )
            consistency = sensor_set_consistency_loss(
                masked_output.pooled,
                subset_output.pooled,
            )
            female_preservation = 0.5 * (
                preservation_distance(
                    masked_output.pooled, anchor.adapted.pooled
                ).mean()
                + preservation_distance(
                    subset_output.pooled, anchor.adapted.pooled
                ).mean()
            )
            if bool(anchor.native_available.any()):
                available = anchor.native_available
                native_preservation = 0.5 * (
                    preservation_distance(
                        masked_output.pooled[available],
                        anchor.native_pooled[available],
                    ).mean()
                    + preservation_distance(
                        subset_output.pooled[available],
                        anchor.native_pooled[available],
                    ).mean()
                )
            else:
                native_preservation = reconstruction.new_zeros(())
            total = (
                args.reconstruction_weight * reconstruction
                + args.consistency_weight * consistency
                + args.female_preservation_weight * female_preservation
                + args.native_preservation_weight * native_preservation
            )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        total.backward()
        gradient_norm = clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        step += 1
        cohort_counts[cohorts[cohort_index].name] += 1
        record = {
            "step": step,
            "cohort": cohorts[cohort_index].name,
            "batch_samples": int(values.shape[0]),
            "total": float(total.detach().float().cpu()),
            "reconstruction": float(reconstruction.detach().float().cpu()),
            "consistency": float(consistency.detach().float().cpu()),
            "female_preservation": float(
                female_preservation.detach().float().cpu()
            ),
            "native_preservation": float(
                native_preservation.detach().float().cpu()
            ),
            "gradient_norm": float(torch.as_tensor(gradient_norm).float().cpu()),
        }
        history.append(record)
        if step == 1 or step % 25 == 0 or step == args.max_steps:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if step % args.save_every == 0 and step < args.max_steps:
            save("running", step)

    save("complete", step)
    summary = {
        "event": "training_complete",
        "output": str(args.output.resolve()),
        "steps": step,
        "cohort_batch_counts": cohort_counts,
        "elapsed_seconds": time.perf_counter() - started,
        "final_loss": history[-1]["total"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
