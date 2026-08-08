"""Sensor metadata and batching primitives for FemMHC.

FemMHC treats a wearable recording as a *set* of named sensor streams rather
than a fixed device-specific channel vector.  A batch may still be represented
as a dense ``(batch, channels, time)`` tensor, but ``channel_present`` marks
padding or deliberately dropped sensors and every channel carries semantic
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import torch
from torch import nn


MODALITIES: tuple[str, ...] = (
    "unknown",
    "activity",
    "heart_rate",
    "heart_rate_variability",
    "temperature",
    "oxygen",
    "sleep",
    "light",
    "electrodermal_activity",
    "respiration",
)

UNITS: tuple[str, ...] = (
    "unknown",
    "count",
    "bpm",
    "millisecond",
    "celsius",
    "relative",
    "state",
    "lux",
    "minute",
    "meter",
    "flight",
    "kilocalorie",
    "binary",
)

BODY_SITES: tuple[str, ...] = (
    "unknown",
    "wrist",
    "finger",
    "hip",
    "chest",
    "ankle",
    "phone",
)


@dataclass(frozen=True)
class SensorDescriptor:
    """Device-independent metadata for one sensor stream."""

    name: str
    modality: str = "unknown"
    unit: str = "unknown"
    body_site: str = "unknown"
    sampling_rate_hz: float = 1.0 / 60.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("sensor name must be non-empty")
        if not math.isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be finite and positive")


@dataclass(frozen=True)
class SensorBatch:
    """A padded sensor-set batch.

    ``descriptors`` describes the shared channel axis.  Individual examples
    can omit channels through ``channel_present`` without changing the tensor
    shape.
    """

    values: torch.Tensor
    descriptors: tuple[SensorDescriptor, ...]
    channel_present: torch.Tensor | None = None

    def validate(self) -> "SensorBatch":
        if self.values.ndim != 3:
            raise ValueError(f"expected values shaped (B,C,L), got {tuple(self.values.shape)}")
        batch, channels, _ = self.values.shape
        if len(self.descriptors) != channels:
            raise ValueError(
                f"received {len(self.descriptors)} descriptors for {channels} channels"
            )
        if self.channel_present is not None and self.channel_present.shape != (batch, channels):
            raise ValueError(
                "channel_present must have shape "
                f"{(batch, channels)}, got {tuple(self.channel_present.shape)}"
            )
        return self

    def present_mask(self) -> torch.Tensor:
        self.validate()
        if self.channel_present is None:
            return torch.ones(
                self.values.shape[:2],
                dtype=torch.bool,
                device=self.values.device,
            )
        return self.channel_present.to(device=self.values.device, dtype=torch.bool)


def _category_index(value: str, vocabulary: tuple[str, ...]) -> int:
    try:
        return vocabulary.index(value)
    except ValueError:
        return 0


def _stable_name_bucket(name: str, n_buckets: int) -> int:
    digest = hashlib.sha256(name.strip().lower().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % n_buckets


def _sampling_rate_bucket(rate_hz: float, n_buckets: int) -> int:
    # Wearable sampling rates span several orders of magnitude.  Bucket in
    # log2 space around one sample per minute and reserve bucket zero for
    # future unknown/malformed metadata.
    center = math.log2(rate_hz * 60.0)
    shifted = int(round(center)) + n_buckets // 2
    return min(max(shifted, 1), n_buckets - 1)


class SensorMetadataEncoder(nn.Module):
    """Encode sensor semantics into the OpenMHC token dimension."""

    def __init__(
        self,
        embed_dim: int,
        *,
        name_buckets: int = 512,
        sampling_rate_buckets: int = 32,
    ) -> None:
        super().__init__()
        if name_buckets < 2 or sampling_rate_buckets < 2:
            raise ValueError("metadata embedding tables need at least two buckets")
        self.name_buckets = int(name_buckets)
        self.sampling_rate_buckets = int(sampling_rate_buckets)
        self.name_embedding = nn.Embedding(name_buckets, embed_dim)
        self.modality_embedding = nn.Embedding(len(MODALITIES), embed_dim)
        self.unit_embedding = nn.Embedding(len(UNITS), embed_dim)
        self.body_site_embedding = nn.Embedding(len(BODY_SITES), embed_dim)
        self.sampling_rate_embedding = nn.Embedding(sampling_rate_buckets, embed_dim)
        self.normalizer = nn.LayerNorm(embed_dim)

        for embedding in (
            self.name_embedding,
            self.modality_embedding,
            self.unit_embedding,
            self.body_site_embedding,
            self.sampling_rate_embedding,
        ):
            nn.init.normal_(embedding.weight, std=0.02)

    def descriptor_ids(
        self,
        descriptors: Iterable[SensorDescriptor],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        descriptors = tuple(descriptors)
        names = [_stable_name_bucket(d.name, self.name_buckets) for d in descriptors]
        modalities = [_category_index(d.modality, MODALITIES) for d in descriptors]
        units = [_category_index(d.unit, UNITS) for d in descriptors]
        body_sites = [_category_index(d.body_site, BODY_SITES) for d in descriptors]
        rates = [
            _sampling_rate_bucket(d.sampling_rate_hz, self.sampling_rate_buckets)
            for d in descriptors
        ]
        return tuple(
            torch.tensor(values, dtype=torch.long, device=device)
            for values in (names, modalities, units, body_sites, rates)
        )

    def forward(self, descriptors: Iterable[SensorDescriptor]) -> torch.Tensor:
        descriptors = tuple(descriptors)
        device = self.name_embedding.weight.device
        name, modality, unit, body_site, rate = self.descriptor_ids(
            descriptors,
            device=device,
        )
        encoded = (
            self.name_embedding(name)
            + self.modality_embedding(modality)
            + self.unit_embedding(unit)
            + self.body_site_embedding(body_site)
            + self.sampling_rate_embedding(rate)
        )
        return self.normalizer(encoded)


MCPHASES_SENSOR_DESCRIPTORS: tuple[SensorDescriptor, ...] = (
    SensorDescriptor("steps", "activity", "count", "wrist", 1.0 / 60.0),
    SensorDescriptor("heart_rate", "heart_rate", "bpm", "wrist", 1.0 / 60.0),
    SensorDescriptor(
        "hrv_rmssd",
        "heart_rate_variability",
        "millisecond",
        "wrist",
        1.0 / 300.0,
    ),
    SensorDescriptor(
        "wrist_temperature",
        "temperature",
        "relative",
        "wrist",
        1.0 / 60.0,
    ),
    SensorDescriptor(
        "oxygen_variation",
        "oxygen",
        "relative",
        "wrist",
        1.0 / 60.0,
    ),
    SensorDescriptor("sleep_state", "sleep", "state", "wrist", 1.0 / 60.0),
)


OPENMHC_SENSOR_DESCRIPTORS: tuple[SensorDescriptor, ...] = (
    SensorDescriptor("iphone_steps", "activity", "count", "phone", 1.0 / 60.0),
    SensorDescriptor("iphone_distance", "activity", "meter", "phone", 1.0 / 60.0),
    SensorDescriptor("iphone_flights", "activity", "flight", "phone", 1.0 / 60.0),
    SensorDescriptor("watch_steps", "activity", "count", "wrist", 1.0 / 60.0),
    SensorDescriptor("watch_distance", "activity", "meter", "wrist", 1.0 / 60.0),
    SensorDescriptor("watch_hr", "heart_rate", "bpm", "wrist", 1.0 / 60.0),
    SensorDescriptor("watch_energy", "activity", "kilocalorie", "wrist", 1.0 / 60.0),
    SensorDescriptor("sleep_asleep", "sleep", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("sleep_inbed", "sleep", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_walking", "activity", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_cycling", "activity", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_running", "activity", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_other", "activity", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_mixed_cardio", "activity", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_strength", "activity", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_elliptical", "activity", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_hiit", "activity", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_functional", "activity", "binary", "wrist", 1.0 / 60.0),
    SensorDescriptor("workout_yoga", "activity", "binary", "wrist", 1.0 / 60.0),
)


PREGNANCY_GA_SENSOR_DESCRIPTORS: tuple[SensorDescriptor, ...] = (
    SensorDescriptor(
        "wrist_actigraphy",
        "activity",
        "count",
        "wrist",
        1.0 / 60.0,
    ),
    SensorDescriptor(
        "ambient_light",
        "light",
        "lux",
        "wrist",
        1.0 / 60.0,
    ),
)


WEARABLE_HRV_MENTAL_SENSOR_DESCRIPTORS: tuple[SensorDescriptor, ...] = (
    SensorDescriptor("steps", "activity", "count", "wrist", 1.0 / 60.0),
    SensorDescriptor("heart_rate", "heart_rate", "bpm", "wrist", 1.0 / 60.0),
    SensorDescriptor(
        "hrv_rmssd",
        "heart_rate_variability",
        "millisecond",
        "wrist",
        1.0 / 300.0,
    ),
    SensorDescriptor("ambient_light", "light", "lux", "wrist", 1.0 / 300.0),
)


AFFECTIVE_DAILY_SENSOR_DESCRIPTORS: tuple[SensorDescriptor, ...] = (
    SensorDescriptor("steps", "activity", "count", "wrist", 1.0 / 60.0),
    SensorDescriptor("heart_rate", "heart_rate", "bpm", "wrist", 1.0 / 60.0),
    SensorDescriptor("sleep_state", "sleep", "binary", "wrist", 1.0 / 60.0),
)


NHANES_FEMALE_SENSOR_DESCRIPTORS: tuple[SensorDescriptor, ...] = (
    SensorDescriptor(
        "wrist_activity_log10_mims",
        "activity",
        "relative",
        "wrist",
        1.0 / 60.0,
    ),
    SensorDescriptor("sleep_wear", "sleep", "binary", "wrist", 1.0 / 60.0),
)
