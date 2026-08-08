"""Self-supervised and specialization objectives used by FemMHC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .sensors import SensorBatch


def sensor_set_consistency_loss(
    full_embedding: torch.Tensor,
    subset_embedding: torch.Tensor,
) -> torch.Tensor:
    """Cosine distance between two sensor-set views of the same recording."""

    if full_embedding.shape != subset_embedding.shape:
        raise ValueError("sensor-set views must have identical embedding shapes")
    distance = 1.0 - F.cosine_similarity(full_embedding, subset_embedding, dim=-1)
    return distance.clamp_min(0.0).mean()


def preservation_distance(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
) -> torch.Tensor:
    """Per-example cosine distance from the frozen OpenMHC representation."""

    if student_embedding.shape != teacher_embedding.shape:
        raise ValueError("student and teacher embeddings must have identical shapes")
    return (
        1.0
        - F.cosine_similarity(
            student_embedding,
            teacher_embedding.detach(),
            dim=-1,
        )
    ).clamp_min(0.0)


def preservation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
) -> torch.Tensor:
    """Mean representation-preservation loss used for optimization."""

    return preservation_distance(
        student_embedding,
        teacher_embedding,
    ).mean()


class TemporalOrderHead(nn.Module):
    """Predict whether the second representation occurs after the first."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError("temporal pair embeddings must have identical shapes")
        features = torch.cat([first, second, second - first], dim=-1)
        return self.network(features).squeeze(-1)


def temporal_order_loss(
    head: TemporalOrderHead,
    first: torch.Tensor,
    second: torch.Tensor,
    second_is_later: torch.Tensor,
) -> torch.Tensor:
    logits = head(first, second)
    target = second_is_later.to(device=logits.device, dtype=logits.dtype)
    if target.shape != logits.shape:
        raise ValueError(f"expected temporal labels shaped {tuple(logits.shape)}")
    return F.binary_cross_entropy_with_logits(logits, target)


class PhysiologyChangeHead(nn.Module):
    """Predict within-person day-to-day changes instead of participant identity.

    For each sensor channel the target contains the change in daily mean and
    daily standard deviation.  The head therefore rewards state-sensitive
    representations without declaring adjacent days from the same person to be
    interchangeable positives.
    """

    def __init__(
        self,
        embed_dim: int,
        channels: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or channels <= 0:
            raise ValueError("embed_dim and channels must be positive")
        self.channels = int(channels)
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * 2),
        )

    def forward(self, earlier: torch.Tensor, later: torch.Tensor) -> torch.Tensor:
        if earlier.shape != later.shape or earlier.ndim != 2:
            raise ValueError("daily embeddings must share shape (batch, embed_dim)")
        features = torch.cat([earlier, later, later - earlier], dim=-1)
        return self.network(features)


def daily_sensor_statistics(values: torch.Tensor) -> torch.Tensor:
    """Return finite-value mean and standard deviation for every daily channel."""

    if values.ndim != 3:
        raise ValueError("values must have shape (batch, channels, samples)")
    finite = torch.isfinite(values)
    count = finite.sum(dim=-1).clamp_min(1)
    clean = torch.where(finite, values, torch.zeros_like(values))
    mean = clean.sum(dim=-1) / count
    centered = torch.where(finite, values - mean.unsqueeze(-1), torch.zeros_like(values))
    variance = centered.square().sum(dim=-1) / count
    standard_deviation = variance.clamp_min(0.0).sqrt()
    return torch.cat([mean, standard_deviation], dim=-1)


def physiology_change_loss(
    head: PhysiologyChangeHead,
    earlier_embedding: torch.Tensor,
    later_embedding: torch.Tensor,
    earlier_values: torch.Tensor,
    later_values: torch.Tensor,
) -> torch.Tensor:
    """Robustly regress observed daily-statistic changes from representation pairs."""

    prediction = head(earlier_embedding, later_embedding)
    with torch.no_grad():
        target = daily_sensor_statistics(later_values) - daily_sensor_statistics(
            earlier_values
        )
        target = target.clamp(min=-5.0, max=5.0)
    if prediction.shape != target.shape:
        raise ValueError(
            f"change prediction {tuple(prediction.shape)} != target {tuple(target.shape)}"
        )
    return F.smooth_l1_loss(prediction, target)


def adjacent_day_contrastive_loss(
    earlier: torch.Tensor,
    later: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE for adjacent days from the same participant."""

    if earlier.shape != later.shape or earlier.ndim != 2:
        raise ValueError("adjacent embeddings must share shape (batch, embed_dim)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if earlier.shape[0] < 2:
        return earlier.sum() * 0.0
    first = F.normalize(earlier, dim=-1)
    second = F.normalize(later, dim=-1)
    logits = first @ second.transpose(0, 1) / temperature
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.transpose(0, 1), target)
    )


def masked_patch_reconstruction_loss(
    prediction: torch.Tensor,
    target_values: torch.Tensor,
    artificial_patch_mask: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    """MSE on artificially hidden patches that existed in the source data."""

    if target_values.ndim != 3:
        raise ValueError("target_values must have shape (B,C,L)")
    batch, channels, minutes = target_values.shape
    if minutes % patch_size:
        raise ValueError("target length must be divisible by patch_size")
    target = target_values.reshape(batch, channels, minutes // patch_size, patch_size)
    target = target.reshape(batch, -1, patch_size)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match {tuple(target.shape)}"
        )
    if artificial_patch_mask.shape != target.shape[:2]:
        raise ValueError("artificial_patch_mask has the wrong shape")
    originally_observed = torch.isfinite(target).all(dim=-1)
    selected = artificial_patch_mask.bool() & originally_observed
    if not bool(selected.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[selected], target[selected])


def drop_sensor_channels(
    batch: SensorBatch,
    *,
    drop_probability: float = 0.35,
    generator: torch.Generator | None = None,
    patch_size: int | None = None,
    min_observed_fraction: float = 0.5,
) -> SensorBatch:
    """Create a sensor-subset view while retaining at least one channel per sample."""

    batch.validate()
    if not 0 <= drop_probability < 1:
        raise ValueError("drop_probability must be in [0, 1)")
    present = batch.present_mask().clone()
    if patch_size is not None:
        if batch.values.shape[-1] % patch_size:
            raise ValueError("sensor length must be divisible by patch_size")
        patches = batch.values.reshape(
            batch.values.shape[0],
            batch.values.shape[1],
            -1,
            patch_size,
        )
        usable = torch.isfinite(patches).float().mean(dim=-1)
        usable = usable.ge(min_observed_fraction).any(dim=-1)
        present = present & usable
    random_values = torch.rand(
        present.shape,
        device=present.device,
        generator=generator,
    )
    retained = present & (random_values >= drop_probability)
    for row in range(retained.shape[0]):
        if not bool(retained[row].any()):
            candidates = present[row].nonzero(as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            retained[row, candidates[0]] = True
    values = batch.values.clone()
    values = values.masked_fill(~retained.unsqueeze(-1), torch.nan)
    return SensorBatch(values, batch.descriptors, retained)


def mask_sensor_patches(
    batch: SensorBatch,
    *,
    patch_size: int,
    mask_probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[SensorBatch, torch.Tensor]:
    """Hide fully observed patches and return the exact reconstruction target mask."""

    batch.validate()
    if not 0 < mask_probability < 1:
        raise ValueError("mask_probability must be in (0, 1)")
    batch_size, channels, minutes = batch.values.shape
    if minutes % patch_size:
        raise ValueError("sensor length must be divisible by patch_size")
    patches = batch.values.reshape(
        batch_size,
        channels,
        minutes // patch_size,
        patch_size,
    )
    eligible = torch.isfinite(patches).all(dim=-1)
    eligible = eligible & batch.present_mask().unsqueeze(-1)
    selected = eligible & (
        torch.rand(eligible.shape, device=eligible.device, generator=generator)
        < mask_probability
    )
    for row in range(batch_size):
        eligible_count = int(eligible[row].sum())
        if eligible_count > 1 and not bool(selected[row].any()):
            first = eligible[row].nonzero(as_tuple=False)[0]
            selected[row, first[0], first[1]] = True
        if eligible_count > 0 and int(selected[row].sum()) >= eligible_count:
            retained = selected[row].nonzero(as_tuple=False)[0]
            selected[row, retained[0], retained[1]] = False
    masked_patches = patches.clone().masked_fill(selected.unsqueeze(-1), torch.nan)
    return (
        SensorBatch(
            masked_patches.reshape_as(batch.values),
            batch.descriptors,
            batch.channel_present,
        ),
        selected.reshape(batch_size, -1),
    )


@dataclass(frozen=True)
class FemMHCLosses:
    total: torch.Tensor
    reconstruction: torch.Tensor
    sensor_consistency: torch.Tensor
    preservation: torch.Tensor
    trajectory: torch.Tensor


def combine_losses(
    *,
    reconstruction: torch.Tensor,
    sensor_consistency: torch.Tensor,
    preservation: torch.Tensor,
    trajectory: torch.Tensor,
    consistency_weight: float = 1.0,
    preservation_weight: float = 1.0,
    trajectory_weight: float = 0.5,
) -> FemMHCLosses:
    total = (
        reconstruction
        + consistency_weight * sensor_consistency
        + preservation_weight * preservation
        + trajectory_weight * trajectory
    )
    return FemMHCLosses(
        total=total,
        reconstruction=reconstruction,
        sensor_consistency=sensor_consistency,
        preservation=preservation,
        trajectory=trajectory,
    )
