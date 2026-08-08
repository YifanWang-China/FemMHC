"""Native OpenMHC teacher encoding for representation-preserving transfer."""

from __future__ import annotations

import torch

from openmhc.models.lsm2.utils import create_inherited_mask


def pool_native_openmhc(
    model: torch.nn.Module,
    values: torch.Tensor,
) -> torch.Tensor:
    """Dense-encode normalized 19-channel days and pool observed patches."""

    if values.ndim != 3 or values.shape[1] != int(model.in_channels):
        raise ValueError(
            f"expected (B,{int(model.in_channels)},L) OpenMHC values, got {tuple(values.shape)}"
        )
    inherited = create_inherited_mask(values, patch_size=int(model.patch_size))
    latent, mask = model.forward_encoder_dense(values, inherited)
    observed = mask == 0
    empty = ~observed.any(dim=1)
    if bool(empty.any()):
        indices = empty.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"native OpenMHC teacher received empty samples: {indices}")
    weights = observed.to(latent.dtype).unsqueeze(-1)
    return (latent * weights).sum(dim=1) / weights.sum(dim=1)


__all__ = ["pool_native_openmhc"]
