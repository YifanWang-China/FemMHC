from __future__ import annotations

import unittest

import torch

from openmhc.models.lsm2.vit1d import LSM2ViT1D
from w3m import RingLSM2Encoder, RingWorldModelHeads


def tiny_openmhc_encoder() -> LSM2ViT1D:
    return LSM2ViT1D(
        seq_length=20,
        patch_size=10,
        in_channels=19,
        embed_dim=32,
        depth=1,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
    )


class RingLSM2EncoderTests(unittest.TestCase):
    def test_forward_and_adapter_gradient(self) -> None:
        model = RingLSM2Encoder(tiny_openmhc_encoder(), freeze_backbone=True)
        values = torch.randn(2, 6, 20)
        values[0, 3, :10] = torch.nan

        pooled, latent, mask = model(values)
        self.assertEqual(tuple(pooled.shape), (2, 32))
        self.assertEqual(tuple(latent.shape), (2, 12, 32))
        self.assertEqual(tuple(mask.shape), (2, 12))
        self.assertEqual(int(mask.sum()), 1)

        pooled.square().mean().backward()
        self.assertIsNotNone(model.channel_delta.grad)
        self.assertGreater(float(model.channel_delta.grad.abs().sum()), 0)
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))

    def test_all_missing_sample_is_rejected(self) -> None:
        model = RingLSM2Encoder(tiny_openmhc_encoder())
        with self.assertRaisesRegex(ValueError, "no usable ring patches"):
            model(torch.full((1, 6, 20), torch.nan))

    def test_probabilistic_head_shapes(self) -> None:
        heads = RingWorldModelHeads(embed_dim=32)
        output = heads(torch.randn(4, 32))
        self.assertEqual(tuple(output.state_logits.shape), (4, 3))
        self.assertEqual(tuple(output.future_mean.shape), (4, 3, 6))
        self.assertEqual(tuple(output.future_log_scale.shape), (4, 3, 6))


if __name__ == "__main__":
    unittest.main()
