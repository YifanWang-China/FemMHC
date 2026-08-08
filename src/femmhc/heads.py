"""Generic probing heads for the FemWear benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ProbabilisticOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor


class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        *,
        hidden_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("classification requires at least two classes")
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.register_buffer("calibration_temperature", torch.tensor(1.0))

    def forward(self, embedding: torch.Tensor) -> ProbabilisticOutput:
        logits = self.network(embedding)
        probabilities = torch.softmax(
            logits / self.calibration_temperature.clamp_min(1e-6),
            dim=-1,
        )
        return ProbabilisticOutput(logits, probabilities)


class LinearClassificationHead(nn.Module):
    """A normalized linear probe that pushes supervision into the encoder."""

    def __init__(self, embed_dim: int, n_classes: int) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("classification requires at least two classes")
        self.network = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, n_classes))
        self.register_buffer("calibration_temperature", torch.tensor(1.0))

    def forward(self, embedding: torch.Tensor) -> ProbabilisticOutput:
        logits = self.network(embedding)
        probabilities = torch.softmax(
            logits / self.calibration_temperature.clamp_min(1e-6),
            dim=-1,
        )
        return ProbabilisticOutput(logits, probabilities)


class OrdinalHead(nn.Module):
    """Cumulative-link ordinal head with guaranteed ordered thresholds."""

    def __init__(self, embed_dim: int, n_classes: int) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("ordinal prediction requires at least two classes")
        self.n_classes = int(n_classes)
        self.score = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.first_threshold = nn.Parameter(torch.tensor(-1.0))
        self.threshold_deltas = nn.Parameter(torch.zeros(n_classes - 2))

    def thresholds(self) -> torch.Tensor:
        if self.threshold_deltas.numel() == 0:
            return self.first_threshold.reshape(1)
        positive_deltas = torch.nn.functional.softplus(self.threshold_deltas)
        return torch.cat(
            [
                self.first_threshold.reshape(1),
                self.first_threshold + torch.cumsum(positive_deltas, dim=0),
            ]
        )

    def forward(self, embedding: torch.Tensor) -> ProbabilisticOutput:
        score = self.score(embedding)
        logits = self.thresholds().unsqueeze(0) - score
        cumulative = torch.sigmoid(logits)
        zeros = torch.zeros_like(cumulative[:, :1])
        ones = torch.ones_like(cumulative[:, :1])
        probabilities = torch.cat([cumulative, ones], dim=1) - torch.cat(
            [zeros, cumulative],
            dim=1,
        )
        return ProbabilisticOutput(logits, probabilities.clamp_min(0.0))


class RegressionHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding).squeeze(-1)
