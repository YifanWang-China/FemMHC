"""OpenMHC-initialized variable-sensor encoder for FemMHC."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from openmhc.models.lsm2.positional import get_1d_sincos_pos_embed_from_grid

from .sensors import SensorBatch, SensorMetadataEncoder


def build_patch_missing_mask(
    batch: SensorBatch,
    *,
    patch_size: int,
    min_observed_fraction: float,
) -> torch.Tensor:
    """Return a boolean ``(B, C*P)`` mask for missing or padded patches."""

    batch.validate()
    if not 0 < min_observed_fraction <= 1:
        raise ValueError("min_observed_fraction must be in (0, 1]")
    values = batch.values
    batch_size, channels, minutes = values.shape
    if minutes % patch_size:
        raise ValueError(f"{minutes=} must be divisible by {patch_size=}")
    patches = values.reshape(batch_size, channels, minutes // patch_size, patch_size)
    observed = torch.isfinite(patches).float().mean(dim=-1)
    missing = observed < min_observed_fraction
    missing = missing | ~batch.present_mask().unsqueeze(-1)
    return missing.reshape(batch_size, -1)


def _time_position_embedding(
    *,
    channels: int,
    patches_per_channel: int,
    embed_dim: int,
) -> torch.Tensor:
    time_positions = np.tile(
        np.arange(patches_per_channel, dtype=np.float32),
        channels,
    )
    values = get_1d_sincos_pos_embed_from_grid(embed_dim, time_positions)
    return torch.from_numpy(values).float().unsqueeze(0)


OPENMHC_SENSOR_ANCHORS: dict[str, int] = {
    # OpenMHC channel indices: watch_steps=3, watch_hr=5, sleep_asleep=7.
    # Novel female-wearable channels intentionally have no numeric-channel anchor.
    "steps": 3,
    "heart_rate": 5,
    "sleep_state": 7,
    "iphone_steps": 0,
    "iphone_distance": 1,
    "iphone_flights": 2,
    "watch_steps": 3,
    "watch_distance": 4,
    "watch_hr": 5,
    "watch_energy": 6,
    "sleep_asleep": 7,
    "sleep_inbed": 8,
    "workout_walking": 9,
    "workout_cycling": 10,
    "workout_running": 11,
    "workout_other": 12,
    "workout_mixed_cardio": 13,
    "workout_strength": 14,
    "workout_elliptical": 15,
    "workout_hiit": 16,
    "workout_functional": 17,
    "workout_yoga": 18,
}


class LowRankAdapter(nn.Module):
    """A residual bottleneck initialized as an identity mapping."""

    def __init__(self, embed_dim: int, rank: int, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("adapter rank must be positive")
        self.normalizer = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(rank, embed_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.activation(self.down(self.normalizer(x)))))


class PhysiologicalRegimeAdapterBank(nn.Module):
    """Softly route tokens across unlabeled physiological-regime experts."""

    def __init__(
        self,
        embed_dim: int,
        *,
        n_experts: int = 3,
        rank: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        self.experts = nn.ModuleList(
            LowRankAdapter(embed_dim, rank, dropout) for _ in range(n_experts)
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_experts),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.gate(pooled), dim=-1)
        deltas = torch.stack([expert.delta(tokens) for expert in self.experts], dim=1)
        mixed = torch.sum(deltas * weights[:, :, None, None], dim=1)
        return tokens + mixed, weights


@dataclass(frozen=True)
class FemMHCOutput:
    pooled: torch.Tensor
    latent: torch.Tensor
    patch_missing_mask: torch.Tensor
    adapter_weights: torch.Tensor


class FemMHCEncoder(nn.Module):
    """Variable-sensor encoder initialized from an OpenMHC LSM2 checkpoint.

    The OpenMHC patch projection is shared across channels.  FemMHC reuses it
    and the pretrained Transformer, replaces device-specific channel indices
    with semantic sensor descriptors, and adds a small gated adapter bank.
    """

    def __init__(
        self,
        pretrained_lsm2: nn.Module,
        *,
        min_observed_fraction: float = 0.5,
        adapter_rank: int = 32,
        n_adapter_experts: int = 3,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.seq_length = int(pretrained_lsm2.seq_length)
        self.patch_size = int(pretrained_lsm2.patch_size)
        self.embed_dim = int(pretrained_lsm2.embed_dim)
        self.patches_per_channel = self.seq_length // self.patch_size
        self.min_observed_fraction = float(min_observed_fraction)

        self.patch_projection = copy.deepcopy(pretrained_lsm2.patch_embed.proj)
        self.encoder = copy.deepcopy(pretrained_lsm2.encoder)
        if self.embed_dim % 2:
            raise ValueError("OpenMHC positional transfer requires an even embedding dimension")
        self.position_dim = self.embed_dim // 2
        self.sensor_metadata = SensorMetadataEncoder(self.position_dim)
        # Start close to OpenMHC's original channel position for shared sensors,
        # while allowing semantics to replace the arbitrary device channel ID.
        self.anchor_mix_logit = nn.Parameter(torch.tensor(-2.0))
        self.adapters = PhysiologicalRegimeAdapterBank(
            self.embed_dim,
            n_experts=n_adapter_experts,
            rank=adapter_rank,
            dropout=0.1,
        )
        self.set_backbone_trainable(not freeze_backbone)

    def _channel_position_embedding(self, batch: SensorBatch) -> torch.Tensor:
        semantic = self.sensor_metadata(batch.descriptors)
        anchor_indices = [OPENMHC_SENSOR_ANCHORS.get(item.name) for item in batch.descriptors]
        known = torch.tensor(
            [item is not None for item in anchor_indices],
            device=semantic.device,
            dtype=torch.bool,
        )
        if not bool(known.any()):
            return semantic
        grid = np.asarray([item or 0 for item in anchor_indices], dtype=np.float32)
        anchors = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(self.position_dim, grid)
        ).to(device=semantic.device, dtype=semantic.dtype)
        mixture = torch.sigmoid(self.anchor_mix_logit)
        blended = anchors + mixture * (semantic - anchors)
        return torch.where(known.unsqueeze(-1), blended, semantic)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.patch_projection.parameters():
            parameter.requires_grad = trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def _patch_tokens(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, channels, minutes = values.shape
        clean = torch.nan_to_num(values, nan=0.0)
        tokens = self.patch_projection(clean.reshape(batch_size * channels, 1, minutes))
        tokens = tokens.reshape(
            batch_size,
            channels,
            self.embed_dim,
            self.patches_per_channel,
        )
        return tokens.permute(0, 1, 3, 2).reshape(batch_size, -1, self.embed_dim)

    def forward(self, batch: SensorBatch) -> FemMHCOutput:
        batch.validate()
        values = batch.values
        batch_size, channels, minutes = values.shape
        if minutes != self.seq_length:
            raise ValueError(f"expected {self.seq_length} samples per day, got {minutes}")

        missing = build_patch_missing_mask(
            batch,
            patch_size=self.patch_size,
            min_observed_fraction=self.min_observed_fraction,
        )
        empty = (~missing).sum(dim=1) == 0
        if bool(empty.any()):
            indices = empty.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"samples have no usable sensor patches: {indices}")

        tokens = self._patch_tokens(values)
        sensor_embed = self._channel_position_embedding(batch)
        sensor_embed = sensor_embed[:, None, :].expand(
            channels,
            self.patches_per_channel,
            self.position_dim,
        )
        time_embed = _time_position_embedding(
            channels=channels,
            patches_per_channel=self.patches_per_channel,
            embed_dim=self.position_dim,
        ).to(device=tokens.device, dtype=tokens.dtype)
        position_embed = torch.cat(
            [sensor_embed.reshape(1, -1, self.position_dim), time_embed],
            dim=-1,
        )
        tokens = tokens + position_embed

        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            tokens.shape[1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        attention_mask.masked_fill_(missing[:, None, None, :], float("-inf"))
        latent = self.encoder(tokens, attn_mask=attention_mask)

        observed = (~missing).to(latent.dtype).unsqueeze(-1)
        pooled = (latent * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0)
        latent, adapter_weights = self.adapters(latent, pooled)
        pooled = (latent * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0)
        return FemMHCOutput(
            pooled=pooled,
            latent=latent,
            patch_missing_mask=missing,
            adapter_weights=adapter_weights,
        )


class PatchReconstructionHead(nn.Module):
    """Decode each FemMHC token back to one normalized sensor patch."""

    def __init__(self, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, patch_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projection(self.normalizer(latent))


# Backward-compatible name for early experiment checkpoints.
LifecycleAdapterBank = PhysiologicalRegimeAdapterBank
