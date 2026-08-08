from __future__ import annotations

import unittest

import numpy as np
import torch

from femmhc import (
    FemMHCEncoder,
    MCPHASES_SENSOR_DESCRIPTORS,
    OPENMHC_SENSOR_DESCRIPTORS,
    OrdinalHead,
    MCPHASES_TASKS,
    McPhasesTaskHeads,
    McPhasesV2TaskHeads,
    PatchReconstructionHead,
    SensorBatch,
    TemporalOrderHead,
    build_femmhc_encoder_from_artifact,
    drop_sensor_channels,
    cyclic_phase_loss,
    mask_sensor_patches,
    masked_patch_reconstruction_loss,
    preservation_loss,
    preservation_distance,
    pool_native_openmhc,
    nested_onset_loss,
    sensor_set_consistency_loss,
    temporal_order_loss,
)
from openmhc.models.lsm2.vit1d import LSM2ViT1D
from femmhc.data.openmhc_xs import OpenMHCFemaleDataset, preprocess_openmhc_day


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


class FemMHCEncoderTests(unittest.TestCase):
    def test_checkpoint_helper_restores_internal_adapter_architecture(self) -> None:
        original = FemMHCEncoder(
            tiny_openmhc_encoder(),
            freeze_backbone=True,
            internal_adapter_rank=4,
            internal_adapter_layers=1,
        )
        artifact = {
            "internal_adapter_rank": 4,
            "internal_adapter_layers": 1,
            "student_state_dict": original.state_dict(),
        }
        restored = build_femmhc_encoder_from_artifact(
            tiny_openmhc_encoder(),
            artifact,
        )

        self.assertEqual(restored.internal_adapter_rank, 4)
        self.assertEqual(restored.internal_adapter_indices, (0,))
        for name, expected in original.state_dict().items():
            self.assertTrue(torch.equal(expected, restored.state_dict()[name]), name)

    def test_variable_sensor_forward_and_adapter_gradient(self) -> None:
        model = FemMHCEncoder(
            tiny_openmhc_encoder(),
            adapter_rank=8,
            freeze_backbone=True,
        )
        descriptors = MCPHASES_SENSOR_DESCRIPTORS[:4]
        values = torch.randn(3, 4, 20)
        values[0, 2, :10] = torch.nan
        channel_present = torch.ones(3, 4, dtype=torch.bool)
        channel_present[1, 3] = False

        output = model(SensorBatch(values, descriptors, channel_present))
        self.assertEqual(tuple(output.pooled.shape), (3, 32))
        self.assertEqual(tuple(output.latent.shape), (3, 8, 32))
        self.assertEqual(tuple(output.patch_missing_mask.shape), (3, 8))
        self.assertEqual(tuple(output.adapter_weights.shape), (3, 3))
        self.assertTrue(torch.allclose(output.adapter_weights.sum(-1), torch.ones(3)))

        output.pooled.square().mean().backward()
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertGreater(
            sum(
                float(p.grad.abs().sum())
                for p in model.sensor_metadata.parameters()
                if p.grad is not None
            ),
            0.0,
        )

    def test_internal_transformer_adapter_is_zero_initialized_and_trainable(self) -> None:
        teacher = tiny_openmhc_encoder().eval()
        model = FemMHCEncoder(
            teacher,
            adapter_rank=8,
            freeze_backbone=True,
            internal_adapter_rank=4,
            internal_adapter_layers=1,
        ).eval()
        baseline = FemMHCEncoder(
            teacher,
            adapter_rank=8,
            freeze_backbone=True,
        ).eval()
        baseline.load_state_dict(model.state_dict(), strict=False)
        values = torch.randn(2, 19, 20)
        batch = SensorBatch(values, OPENMHC_SENSOR_DESCRIPTORS)
        with torch.inference_mode():
            reference = baseline(batch).pooled
            adapted = model(batch).pooled
        # The internal residual branch is zero-initialized, so adding it must
        # preserve the pretrained path at initialization (up to numerical
        # precision) while still exposing trainable adapter parameters.
        self.assertTrue(torch.allclose(reference, adapted, atol=1e-5, rtol=1e-5))
        self.assertEqual(model.internal_adapter_indices, (0,))
        self.assertGreater(
            sum(p.numel() for p in model.internal_adapters.parameters()), 0
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.encoder.parameters())
        )
        model.train()
        loss = model(batch).pooled.square().mean()
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and float(parameter.grad.abs().sum()) > 0.0
                for parameter in model.internal_adapters.parameters()
            )
        )

    def test_history_conditioned_internal_adapter_is_static_at_start_and_trainable(self) -> None:
        model = FemMHCEncoder(
            tiny_openmhc_encoder(),
            adapter_rank=8,
            freeze_backbone=True,
            internal_adapter_rank=4,
            internal_adapter_layers=1,
            history_conditioned_internal_adapters=True,
            history_context_dim=32,
            history_maximum_days=4,
        ).eval()
        batch = SensorBatch(torch.randn(2, 19, 20), OPENMHC_SENSOR_DESCRIPTORS)
        history = torch.randn(2, 4, 32)
        present = torch.tensor([[True, True, True, True], [False, True, True, True]])
        with torch.inference_mode():
            cold_start = model(batch).pooled
            initialized = model(
                batch,
                history_embeddings=history,
                history_present=present,
            )
        # The context gate begins at one and the adapter residual begins at
        # zero, preserving the exact static adaptation at initialization.
        self.assertTrue(torch.allclose(cold_start, initialized.pooled, atol=1e-5, rtol=1e-5))
        self.assertIsNotNone(initialized.history_context)
        self.assertEqual(tuple(initialized.history_context.shape), (2, 32))

        adapter = model.internal_adapters["0"]
        torch.nn.init.normal_(adapter.up.weight, std=0.03)
        torch.nn.init.normal_(adapter.history_gate[-1].weight, std=0.03)
        model.train()
        output = model(
            batch,
            history_embeddings=history,
            history_present=present,
        )
        output.pooled.square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
                for parameter in model.history_encoder.parameters()
            )
        )

    def test_history_conditioned_adapter_rejects_malformed_history(self) -> None:
        model = FemMHCEncoder(
            tiny_openmhc_encoder(),
            adapter_rank=8,
            freeze_backbone=True,
            internal_adapter_rank=4,
            internal_adapter_layers=1,
            history_conditioned_internal_adapters=True,
            history_context_dim=32,
            history_maximum_days=4,
        )
        batch = SensorBatch(torch.randn(2, 19, 20), OPENMHC_SENSOR_DESCRIPTORS)
        with self.assertRaisesRegex(ValueError, "history_embeddings and history_present"):
            model(batch, history_embeddings=torch.randn(2, 4, 32))
        with self.assertRaisesRegex(ValueError, "history exceeds"):
            model(
                batch,
                history_embeddings=torch.randn(2, 5, 32),
                history_present=torch.ones(2, 5, dtype=torch.bool),
            )

    def test_all_missing_sample_is_rejected(self) -> None:
        model = FemMHCEncoder(tiny_openmhc_encoder(), adapter_rank=8)
        batch = SensorBatch(
            torch.full((1, 2, 20), torch.nan),
            MCPHASES_SENSOR_DESCRIPTORS[:2],
        )
        with self.assertRaisesRegex(ValueError, "no usable sensor patches"):
            model(batch)

    def test_sensor_dropout_retains_one_channel(self) -> None:
        batch = SensorBatch(
            torch.randn(8, 4, 20),
            MCPHASES_SENSOR_DESCRIPTORS[:4],
        )
        dropped = drop_sensor_channels(batch, drop_probability=0.99)
        self.assertTrue(bool(dropped.present_mask().any(dim=1).all()))

    def test_patch_mask_only_selects_complete_source_patches(self) -> None:
        values = torch.randn(2, 3, 20)
        values[0, 1, :5] = torch.nan
        batch = SensorBatch(values, MCPHASES_SENSOR_DESCRIPTORS[:3])
        masked, selected = mask_sensor_patches(
            batch,
            patch_size=10,
            mask_probability=0.5,
        )
        source_complete = torch.isfinite(values.reshape(2, 3, 2, 10)).all(dim=-1)
        self.assertTrue(bool((selected.reshape_as(source_complete) <= source_complete).all()))
        self.assertTrue(bool(torch.isnan(masked.values).any()))

    def test_native_openmhc_teacher_pooling(self) -> None:
        teacher = tiny_openmhc_encoder().eval()
        values = torch.randn(2, 19, 20)
        values[0, 5, :5] = torch.nan
        with torch.inference_mode():
            pooled = pool_native_openmhc(teacher, values)
        self.assertEqual(tuple(pooled.shape), (2, 32))
        self.assertTrue(bool(torch.isfinite(pooled).all()))
        with self.assertRaisesRegex(ValueError, "empty samples"):
            pool_native_openmhc(teacher, torch.full((1, 19, 20), torch.nan))

    def test_known_sensor_initialization_matches_native_openmhc(self) -> None:
        teacher = tiny_openmhc_encoder().eval()
        student = FemMHCEncoder(teacher, adapter_rank=8).eval()
        values = torch.randn(2, 19, 20)
        # Exercise OpenMHC's asymmetric inherited-mask rule: ordinary source
        # channels reject a partially missing patch, while HR channel 5 keeps
        # it unless the whole patch is missing.
        values[0, 0, 2] = torch.nan
        values[0, 5, 2] = torch.nan
        values[1, 5, :10] = torch.nan
        with torch.inference_mode():
            native = pool_native_openmhc(teacher, values)
            adapted = student(
                SensorBatch(values, OPENMHC_SENSOR_DESCRIPTORS)
            ).pooled
        similarity = torch.nn.functional.cosine_similarity(native, adapted, dim=-1)
        self.assertTrue(bool((similarity > 0.999).all()))
        with torch.inference_mode():
            native_branch = student.forward_native(
                SensorBatch(values, OPENMHC_SENSOR_DESCRIPTORS)
            ).pooled
        self.assertTrue(torch.allclose(native, native_branch, atol=1e-5, rtol=1e-5))

    def test_native_branch_restores_source_channel_order(self) -> None:
        teacher = tiny_openmhc_encoder().eval()
        student = FemMHCEncoder(teacher, adapter_rank=8).eval()
        values = torch.randn(2, 19, 20)
        selected_indices = (5, 3, 7)
        selected_descriptors = tuple(
            OPENMHC_SENSOR_DESCRIPTORS[index] for index in selected_indices
        )
        selected_values = values[:, selected_indices]
        dense_values = torch.full_like(values, torch.nan)
        dense_values[:, selected_indices] = selected_values
        with torch.inference_mode():
            expected = pool_native_openmhc(teacher, dense_values)
            actual = student.forward_native(
                SensorBatch(selected_values, selected_descriptors)
            ).pooled
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))

    def test_dual_branch_concatenates_native_and_adapted_views(self) -> None:
        teacher = tiny_openmhc_encoder().eval()
        student = FemMHCEncoder(teacher, adapter_rank=8).eval()
        values = torch.randn(2, 19, 20)
        batch = SensorBatch(values, OPENMHC_SENSOR_DESCRIPTORS)
        with torch.inference_mode():
            dual = student.forward_dual(batch)
            native = student.forward_native(batch).pooled
            adapted = student(batch).pooled
        self.assertEqual(tuple(dual.pooled.shape), (2, 64))
        self.assertTrue(bool(dual.native_available.all()))
        self.assertTrue(torch.allclose(dual.native_pooled, native, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(dual.pooled[:, :32], native, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(dual.pooled[:, 32:], adapted, atol=1e-5, rtol=1e-5))

    def test_dual_branch_supports_novel_only_sensor_sets(self) -> None:
        student = FemMHCEncoder(tiny_openmhc_encoder(), adapter_rank=8).eval()
        batch = SensorBatch(
            torch.randn(2, 1, 20),
            MCPHASES_SENSOR_DESCRIPTORS[2:3],
        )
        with torch.inference_mode():
            dual = student.forward_dual(batch)
        self.assertFalse(bool(dual.native_available.any()))
        self.assertTrue(bool((dual.native_pooled == 0).all()))
        self.assertTrue(bool(torch.isfinite(dual.pooled).all()))

    def test_native_branch_rejects_novel_sensor(self) -> None:
        student = FemMHCEncoder(tiny_openmhc_encoder(), adapter_rank=8).eval()
        with self.assertRaisesRegex(ValueError, "registered source sensors"):
            student.forward_native(
                SensorBatch(torch.randn(1, 1, 20), MCPHASES_SENSOR_DESCRIPTORS[2:3])
            )

    def test_openmhc_preprocessing_restores_zero_filled_missingness(self) -> None:
        values = np.zeros((19, 20), dtype=np.float32)
        values[5, 0] = 60.0
        processed = preprocess_openmhc_day(
            values,
            means=np.zeros(19, dtype=np.float32),
            stds=np.ones(19, dtype=np.float32),
            normalization_channels=np.arange(7),
        )
        self.assertTrue(np.isnan(processed[0]).all())
        self.assertEqual(processed[5, 0], 60.0)
        self.assertTrue(np.isnan(processed[5, 1:]).all())
        self.assertTrue(np.isfinite(processed[2]).all())
        self.assertTrue(np.isnan(processed[7]).all())

    def test_openmhc_validation_indices_are_participant_balanced(self) -> None:
        dataset = OpenMHCFemaleDataset.__new__(OpenMHCFemaleDataset)
        dataset.indices = list(range(30))
        dataset.participant_ids = ["a"] * 15 + ["b"] * 10 + ["c"] * 5
        selected = dataset.balanced_indices(12, seed=42)
        selected_participants = [dataset.participant_ids[index] for index in selected]
        counts = {item: selected_participants.count(item) for item in set(selected_participants)}
        self.assertEqual(len(selected), 12)
        self.assertEqual(set(selected_participants), {"a", "b", "c"})
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(selected, dataset.balanced_indices(12, seed=42))


class FemMHCObjectiveTests(unittest.TestCase):
    def test_specialization_losses_are_finite_and_differentiable(self) -> None:
        first = torch.randn(4, 32, requires_grad=True)
        second = torch.randn(4, 32, requires_grad=True)
        consistency = sensor_set_consistency_loss(first, second)
        preservation = preservation_loss(first, second)
        distances = preservation_distance(first, second)
        self.assertEqual(tuple(distances.shape), (4,))
        self.assertTrue(torch.allclose(preservation, distances.mean()))
        order = temporal_order_loss(
            TemporalOrderHead(32),
            first,
            second,
            torch.tensor([0.0, 1.0, 1.0, 0.0]),
        )
        loss = consistency + preservation + order
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(first.grad)

    def test_patch_reconstruction_selects_only_artificial_mask(self) -> None:
        values = torch.arange(40, dtype=torch.float32).reshape(1, 2, 20)
        prediction = torch.zeros(1, 4, 10, requires_grad=True)
        mask = torch.tensor([[False, True, False, True]])
        loss = masked_patch_reconstruction_loss(
            prediction,
            values,
            mask,
            patch_size=10,
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_ordinal_probabilities_are_valid(self) -> None:
        output = OrdinalHead(32, 6)(torch.randn(5, 32))
        self.assertEqual(tuple(output.probabilities.shape), (5, 6))
        self.assertTrue(bool((output.probabilities >= 0).all()))
        self.assertTrue(
            torch.allclose(
                output.probabilities.sum(dim=-1),
                torch.ones(5),
                atol=1e-5,
            )
        )

    def test_female_task_registry_has_explicit_horizons(self) -> None:
        self.assertEqual(len(MCPHASES_TASKS), 13)
        next_day = {task.name for task in MCPHASES_TASKS if task.target_offset_days == 1}
        self.assertIn("cramps", next_day)
        self.assertIn("mood_swing", next_day)
        outputs = McPhasesTaskHeads(32)(torch.randn(2, 32))
        self.assertEqual(set(outputs), {task.name for task in MCPHASES_TASKS})

    def test_v2_onset_probabilities_are_nested(self) -> None:
        heads = McPhasesV2TaskHeads(32)
        outputs, onset = heads.forward_with_aux(torch.randn(5, 32))
        probability_24h = outputs["menstrual_onset_24h"].probabilities[:, 1]
        probability_72h = outputs["menstrual_onset_72h"].probabilities[:, 1]
        self.assertTrue(bool((probability_72h >= probability_24h).all()))
        self.assertTrue(
            torch.allclose(onset.bin_probabilities.sum(dim=-1), torch.ones(5), atol=1e-6)
        )
        loss = nested_onset_loss(
            onset,
            torch.tensor([1, 0, 0, -1, 0]),
            torch.tensor([1, 1, 0, -1, 0]),
        )
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_cyclic_phase_loss_is_finite(self) -> None:
        output = McPhasesV2TaskHeads(32)(torch.randn(4, 32))["cycle_phase"]
        loss = cyclic_phase_loss(output, torch.tensor([0, 1, 2, 3]))
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_v2_linear_cycle_head_has_no_hidden_projection(self) -> None:
        heads = McPhasesV2TaskHeads(32, linear_cycle_head=True)
        cycle = heads.heads["cycle_phase"]
        linear_layers = [
            module for module in cycle.modules() if isinstance(module, torch.nn.Linear)
        ]

        self.assertEqual(len(linear_layers), 1)
        self.assertEqual(linear_layers[0].in_features, 32)
        self.assertEqual(linear_layers[0].out_features, 4)


if __name__ == "__main__":
    unittest.main()
