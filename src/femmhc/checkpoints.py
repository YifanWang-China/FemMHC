"""Checkpoint-aware construction helpers for FemMHC encoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .model import FemMHCEncoder


def build_femmhc_encoder_from_artifact(
    pretrained_lsm2: nn.Module,
    artifact: Mapping[str, Any] | None = None,
    *,
    freeze_backbone: bool = True,
) -> FemMHCEncoder:
    """Build an encoder with the architecture recorded in a checkpoint.

    Older cache scripts constructed the default encoder before loading a
    checkpoint.  That fails for checkpoints containing internal Transformer
    adapters because those modules must exist before ``load_state_dict``.
    Centralizing construction keeps all cohort cache paths consistent.
    """

    checkpoint = dict(artifact or {})
    model = FemMHCEncoder(
        pretrained_lsm2,
        internal_adapter_rank=int(checkpoint.get("internal_adapter_rank", 0)),
        internal_adapter_layers=int(checkpoint.get("internal_adapter_layers", 0)),
        history_conditioned_internal_adapters=bool(
            checkpoint.get("history_conditioned_internal_adapters", False)
        ),
        history_context_dim=int(checkpoint.get("history_context_dim", 96)),
        history_maximum_days=int(checkpoint.get("history_maximum_days", 60)),
        history_cycle_modes=int(checkpoint.get("history_cycle_modes", 8)),
        freeze_backbone=freeze_backbone,
    )
    if artifact is None:
        return model

    state = checkpoint.get("student_state_dict", checkpoint.get("model_state_dict"))
    if state is None:
        raise ValueError(
            "FemMHC checkpoint has no student_state_dict/model_state_dict"
        )
    model.load_state_dict(state)
    return model
