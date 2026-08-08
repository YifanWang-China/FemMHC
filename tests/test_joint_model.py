from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from femmhc import (
    FEMALE_HEALTH_TASKS,
    HEALTH_DOMAINS,
    OPENMHC_AUXILIARY_TASKS,
    OPENMHC_PHYSIOLOGY_AUXILIARY_TASKS,
    FemMHCJointModel,
    cyclic_phase_geometry_loss,
    partial_multitask_loss,
    validate_joint_tasks,
)
from femmhc.data import load_aligned_embeddings


class JointTaskRegistryTests(unittest.TestCase):
    def test_complete_openmhc_registry_and_known_xs_gaps(self) -> None:
        validate_joint_tasks()
        self.assertEqual(len(OPENMHC_AUXILIARY_TASKS), 32)
        unavailable = [task for task in OPENMHC_AUXILIARY_TASKS if not task.trainable]
        self.assertEqual(len(unavailable), 4)
        self.assertEqual(len(OPENMHC_PHYSIOLOGY_AUXILIARY_TASKS), 7)
        self.assertTrue(
            any(task.domain == "autonomic" for task in OPENMHC_PHYSIOLOGY_AUXILIARY_TASKS)
        )
        self.assertTrue(any(task.source == "mcphases" for task in FEMALE_HEALTH_TASKS))
        self.assertTrue(any(task.source == "pregnancy_ga_clock" for task in FEMALE_HEALTH_TASKS))

    def test_single_view_cache_is_aligned_to_adapted_half(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npy"
            np.save(path, np.ones((2, 3), dtype=np.float32))
            aligned = load_aligned_embeddings(path, output_dim=6)
        np.testing.assert_array_equal(aligned[:, :3], np.zeros((2, 3)))
        np.testing.assert_array_equal(aligned[:, 3:], np.ones((2, 3)))


class FemMHCJointModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            graph_heads=4,
            dropout=0.0,
        )

    def test_joint_forward_has_factorized_states_and_normalized_graph(self) -> None:
        embeddings = torch.randn(3, 7, 32)
        present = torch.ones(3, 7, dtype=torch.bool)
        present[0, :2] = False
        task_ids = (
            "mcphases/menstrual_onset_24h",
            "mcphases/menstrual_onset_72h",
            "mcphases/fatigue",
            "depress/negative_affect",
            "openmhc/sleep_duration",
        )
        output = self.model(embeddings, present, task_ids=task_ids)
        self.assertEqual(set(output.states.domain_states), set(HEALTH_DOMAINS))
        self.assertEqual(tuple(output.states.shared_state.shape), (3, 32))
        self.assertEqual(tuple(output.states.relation_attention.shape), (3, 4, 8, 8))
        torch.testing.assert_close(
            output.states.relation_attention.sum(dim=-1),
            torch.ones(3, 4, 8),
        )
        self.assertEqual(set(output.predictions), set(task_ids))

    def test_onset_horizons_are_nested(self) -> None:
        output = self.model(
            torch.randn(5, 4, 32),
            task_ids=(
                "mcphases/menstrual_onset_24h",
                "mcphases/menstrual_onset_72h",
            ),
        )
        risk_24 = output.predictions[
            "mcphases/menstrual_onset_24h"
        ].probabilities[:, 1]
        risk_72 = output.predictions[
            "mcphases/menstrual_onset_72h"
        ].probabilities[:, 1]
        self.assertTrue(bool((risk_72 >= risk_24).all()))

    def test_last_day_shared_ignores_earlier_history(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="last_day_shared",
            initialization_seed=42,
        ).eval()
        first = torch.randn(2, 5, 32)
        second = first.clone()
        second[:, :-1] = torch.randn_like(second[:, :-1]) * 20.0
        present = torch.ones(2, 5, dtype=torch.bool)
        task_ids = ("mcphases/fatigue",)
        with torch.inference_mode():
            first_output = model(first, present, task_ids=task_ids)
            second_output = model(second, present, task_ids=task_ids)
        torch.testing.assert_close(
            first_output.predictions["mcphases/fatigue"].logits,
            second_output.predictions["mcphases/fatigue"].logits,
        )

    def test_three_ablation_architectures_have_comparable_outputs(self) -> None:
        parameter_counts = {}
        embeddings = torch.randn(2, 6, 32)
        for architecture in (
            "shared_backbone",
            "factorized_no_graph",
            "full",
            "gated_graph",
        ):
            model = FemMHCJointModel(
                input_dim=32,
                hidden_dim=32,
                maximum_days=14,
                cycle_modes=4,
                graph_heads=4,
                dropout=0.0,
                architecture=architecture,
            )
            output = model(
                embeddings,
                task_ids=("mcphases/fatigue", "openmhc/sleep_duration"),
            )
            self.assertEqual(set(output.states.domain_states), set(HEALTH_DOMAINS))
            self.assertEqual(tuple(output.states.shared_state.shape), (2, 32))
            self.assertEqual(
                tuple(output.states.relation_attention.shape), (2, 4, 8, 8)
            )
            if architecture in {"shared_backbone", "factorized_no_graph"}:
                expected = torch.eye(8).reshape(1, 1, 8, 8).expand(2, 4, -1, -1)
                torch.testing.assert_close(output.states.relation_attention, expected)
            parameter_counts[architecture] = sum(
                parameter.numel() for parameter in model.parameters()
            )
        self.assertLess(
            parameter_counts["shared_backbone"],
            parameter_counts["factorized_no_graph"],
        )
        self.assertLess(
            parameter_counts["factorized_no_graph"], parameter_counts["full"]
        )
        self.assertEqual(
            parameter_counts["full"] + len(HEALTH_DOMAINS),
            parameter_counts["gated_graph"],
        )

    def test_gated_graph_starts_as_identity_and_receives_gradient(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            graph_heads=4,
            dropout=0.0,
            architecture="gated_graph",
        )
        scale = model.state_encoder.graph_residual_scale
        torch.testing.assert_close(scale, torch.zeros_like(scale))
        output = model(torch.randn(4, 5, 32), task_ids=("mcphases/fatigue",))
        output.predictions["mcphases/fatigue"].logits.sum().backward()
        self.assertIsNotNone(scale.grad)
        self.assertGreater(float(scale.grad.abs().sum()), 0.0)

    def test_task_conditioned_router_is_normalized_and_task_specific(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            graph_heads=4,
            dropout=0.0,
            architecture="task_router",
            initialization_seed=42,
        )
        task_ids = ("mcphases/fatigue", "openmhc/sleep_duration")
        output = model(torch.randn(3, 5, 32), task_ids=task_ids)
        self.assertEqual(set(output.routing_attention), set(task_ids))
        for task_id, attention in output.routing_attention.items():
            self.assertEqual(tuple(attention.shape), (3, len(HEALTH_DOMAINS)))
            torch.testing.assert_close(
                attention.sum(dim=-1), torch.ones(3), rtol=1e-5, atol=1e-6
            )
            own_domain = "sleep_recovery"
            own_index = HEALTH_DOMAINS.index(own_domain)
            self.assertTrue(
                bool(
                    (
                        attention[:, own_index]
                        >= attention.max(dim=-1).values - 1e-6
                    ).all()
                )
            )

    def test_dual_path_router_keeps_general_and_female_states(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_router",
            initialization_seed=42,
            routing_initial_logit=-2.0,
        )
        output = model(
            torch.randn(2, 5, 32),
            task_ids=("mcphases/fatigue", "openmhc/sleep_duration"),
        )
        self.assertEqual(tuple(output.states.shared_state.shape), (2, 32))
        self.assertEqual(set(output.states.domain_states), set(HEALTH_DOMAINS))
        self.assertEqual(len(output.routing_attention), 2)
        self.assertEqual(model.task_heads.routing_base, "shared")

    def test_dual_view_residual_router_preserves_native_and_falls_back(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=24,
            maximum_days=14,
            dropout=0.0,
            architecture="dual_view_residual_router",
            initialization_seed=42,
            routing_initial_logit=-2.0,
        )
        embeddings = torch.randn(2, 5, 32)
        embeddings[1, :, :16] = 0.0
        output = model(
            embeddings,
            task_ids=("mcphases/fatigue", "openmhc/sleep_duration"),
        )
        available = output.states.auxiliary["native_view_available"]
        torch.testing.assert_close(available, torch.tensor([True, False]))
        native = output.states.auxiliary["native_representation"]
        female = output.states.auxiliary["female_representation"]
        torch.testing.assert_close(output.states.shared_state[0], native[0])
        torch.testing.assert_close(output.states.shared_state[1], female[1])
        for state in output.states.domain_states.values():
            torch.testing.assert_close(state, output.states.shared_state)
        self.assertEqual(set(output.routing_attention), {
            "mcphases/fatigue",
            "openmhc/sleep_duration",
        })

        loss = output.predictions["mcphases/fatigue"].logits.square().mean()
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.state_encoder.female_temporal.parameters()
            )
        )

    def test_dual_path_mechanism_ablations_are_parameter_matched(self) -> None:
        architectures = (
            "dual_path_router",
            "dual_path_no_cycle",
            "dual_path_own_domain",
            "dual_path_fixed_gate",
            "dual_path_timescale_router",
            "dual_path_source_aware",
            "dual_path_cycle_aware",
            "dual_path_cycle_direct",
            "dual_path_task_selected",
            "dual_path_task_selected_soft",
        )
        models = [
            FemMHCJointModel(
                input_dim=32,
                hidden_dim=32,
                maximum_days=14,
                cycle_modes=4,
                dropout=0.0,
                architecture=architecture,
                initialization_seed=42,
                routing_initial_logit=-2.0,
            )
            for architecture in architectures
        ]
        counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
        self.assertEqual(len(set(counts)), 1)
        reference_keys = tuple(models[0].state_dict())
        for model in models[1:]:
            self.assertEqual(tuple(model.state_dict()), reference_keys)
        phase_model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_phase_geometry",
            initialization_seed=42,
            routing_initial_logit=-2.0,
        )
        phase_count = sum(parameter.numel() for parameter in phase_model.parameters())
        self.assertEqual(phase_count - counts[0], 130)

    def test_own_domain_ablation_uses_one_hot_route(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_own_domain",
            initialization_seed=42,
            routing_initial_logit=-2.0,
        )
        task_domains = {
            "mcphases/cramps": "menstrual",
            "openmhc/sleep_duration": "sleep_recovery",
        }
        output = model(torch.randn(3, 5, 32), task_ids=tuple(task_domains))
        for task_id, domain in task_domains.items():
            expected = torch.zeros(3, len(HEALTH_DOMAINS))
            expected[:, HEALTH_DOMAINS.index(domain)] = 1.0
            torch.testing.assert_close(output.routing_attention[task_id], expected)

    def test_timescale_router_separates_fast_and_slow_health_states(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_timescale_router",
            initialization_seed=42,
            routing_initial_logit=-2.0,
        )
        output = model(
            torch.randn(3, 5, 32),
            task_ids=("mcphases/cramps", "openmhc/bmi_values"),
        )
        fast_attention = output.routing_attention["mcphases/cramps"]
        for domain in ("cardiometabolic", "life_stage", "context"):
            torch.testing.assert_close(
                fast_attention[:, HEALTH_DOMAINS.index(domain)], torch.zeros(3)
            )
        slow_attention = output.routing_attention["openmhc/bmi_values"]
        expected = torch.zeros_like(slow_attention)
        expected[:, HEALTH_DOMAINS.index("cardiometabolic")] = 1.0
        torch.testing.assert_close(slow_attention, expected)

    def test_source_aware_router_uses_general_for_openmhc_and_domain_for_female(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_source_aware",
            initialization_seed=42,
            routing_initial_logit=-100.0,
        )
        states = model.state_encoder(torch.randn(3, 5, 32))
        cache = model.task_heads._prepare_route_cache(states)
        female, _ = model.task_heads._route("mcphases/cramps", states, cache)
        general, _ = model.task_heads._route(
            "openmhc/sleep_duration", states, cache
        )
        torch.testing.assert_close(
            female, states.domain_states["menstrual"], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            general, states.shared_state, rtol=1e-5, atol=1e-6
        )

    def test_cycle_aware_router_uses_domain_only_for_cycle_cohort(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_cycle_aware",
            initialization_seed=42,
            routing_initial_logit=-100.0,
        )
        states = model.state_encoder(torch.randn(3, 5, 32))
        cache = model.task_heads._prepare_route_cache(states)
        cycle_state, _ = model.task_heads._route("mcphases/cramps", states, cache)
        affective_state, _ = model.task_heads._route(
            "depress/negative_affect", states, cache
        )
        torch.testing.assert_close(
            cycle_state, states.domain_states["menstrual"], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            affective_state, states.shared_state, rtol=1e-5, atol=1e-6
        )

    def test_cycle_direct_router_bypasses_domain_projection_for_menstrual_tasks(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_cycle_direct",
            initialization_seed=42,
            routing_initial_logit=-100.0,
        )
        states = model.state_encoder(torch.randn(3, 5, 32))
        cache = model.task_heads._prepare_route_cache(states)
        menstrual, _ = model.task_heads._route("mcphases/cramps", states, cache)
        affective, _ = model.task_heads._route(
            "depress/negative_affect", states, cache
        )
        torch.testing.assert_close(
            menstrual,
            states.auxiliary["cycle_representation"],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            affective,
            states.shared_state,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_task_selected_router_uses_train_only_representation_mapping(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_task_selected",
            initialization_seed=42,
            routing_initial_logit=-100.0,
        )
        states = model.state_encoder(torch.randn(3, 5, 32))
        cache = model.task_heads._prepare_route_cache(states)
        expected = {
            "mcphases/cycle_phase": states.auxiliary["cycle_representation"],
            "mcphases/cramps": states.domain_states["menstrual"],
            "mcphases/mood_swing": states.domain_states["menstrual"],
            "mcphases/fatigue": states.shared_state,
        }
        for task_id, representation in expected.items():
            routed, attention = model.task_heads._route(task_id, states, cache)
            torch.testing.assert_close(routed, representation)
            self.assertIsNone(attention)

        task_route, attention = model.task_heads._route(
            "mcphases/perceived_stress", states, cache
        )
        torch.testing.assert_close(
            task_route, states.shared_state, rtol=1e-5, atol=1e-6
        )
        self.assertIsNotNone(attention)

    def test_soft_task_selected_router_keeps_residual_route(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_task_selected_soft",
            initialization_seed=42,
            routing_initial_logit=-100.0,
        )
        states = model.state_encoder(torch.randn(3, 5, 32))
        cache = model.task_heads._prepare_route_cache(states)
        expected = {
            "mcphases/cycle_phase": states.auxiliary["cycle_representation"],
            "mcphases/cramps": states.domain_states["menstrual"],
            "mcphases/fatigue": states.shared_state,
        }
        for task_id, representation in expected.items():
            routed, attention = model.task_heads._route(task_id, states, cache)
            torch.testing.assert_close(routed, representation, rtol=1e-5, atol=1e-6)
            self.assertIsNotNone(attention)

        general_task, attention = model.task_heads._route(
            "openmhc/sleep_duration", states, cache
        )
        torch.testing.assert_close(
            general_task, states.shared_state, rtol=1e-5, atol=1e-6
        )
        self.assertIsNotNone(attention)

    def test_phase_geometry_closes_the_four_phase_cycle(self) -> None:
        correct = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            requires_grad=True,
        )
        target = torch.tensor([0, 1, 2, 3])
        correct_loss = cyclic_phase_geometry_loss(correct, target)
        wrong_loss = cyclic_phase_geometry_loss(correct.roll(1, dims=0), target)
        self.assertLess(float(correct_loss.detach()), float(wrong_loss.detach()))
        correct_loss.backward()
        self.assertIsNotNone(correct.grad)

    def test_phase_geometry_supervises_cycle_representation_directly(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_phase_geometry",
            initialization_seed=42,
            routing_initial_logit=-2.0,
        )
        output = model(
            torch.randn(4, 5, 32),
            task_ids=("mcphases/cycle_phase",),
        )
        self.assertEqual(tuple(output.cycle_phase_geometry.shape), (4, 2))
        losses = partial_multitask_loss(
            output,
            {"mcphases/cycle_phase": torch.tensor([0, 1, 2, 3])},
            phase_geometry_weight=0.25,
        )
        self.assertIn("mcphases/cycle_phase_geometry", losses.per_task)
        losses.total.backward()
        self.assertIsNotNone(model.cycle_phase_projector[1].weight.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.state_encoder.cycle_temporal.parameters()
            )
        )

    def test_circular_phase_head_uses_ordered_unit_circle_prototypes(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_circular_phase_head",
            initialization_seed=42,
            routing_initial_logit=-2.0,
        )
        phase_vectors = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        circular = model._circular_phase_output(phase_vectors)
        torch.testing.assert_close(
            circular.probabilities.sum(dim=-1), torch.ones(4)
        )
        torch.testing.assert_close(
            circular.probabilities.argmax(dim=-1), torch.arange(4)
        )
        output = model(
            torch.randn(4, 5, 32),
            task_ids=("mcphases/cycle_phase",),
        )
        self.assertEqual(
            tuple(output.predictions["mcphases/cycle_phase"].logits.shape),
            (4, 4),
        )

    def test_no_cycle_and_fixed_gate_cut_only_targeted_gradients(self) -> None:
        no_cycle = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_no_cycle",
            initialization_seed=42,
            routing_initial_logit=-2.0,
        )
        prediction = no_cycle(
            torch.randn(3, 5, 32), task_ids=("mcphases/cramps",)
        ).predictions["mcphases/cramps"]
        prediction.logits.sum().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in no_cycle.state_encoder.cycle_temporal.parameters())
        )

        fixed_gate = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_fixed_gate",
            initialization_seed=42,
            routing_initial_logit=-2.0,
        )
        prediction = fixed_gate(
            torch.randn(3, 5, 32), task_ids=("mcphases/cramps",)
        ).predictions["mcphases/cramps"]
        prediction.logits.sum().backward()
        self.assertIsNone(fixed_gate.task_heads.routing_gate_logits.grad)
        self.assertIsNotNone(fixed_gate.task_heads.routing_queries.grad)

    def test_task_router_reuses_domain_key_value_computation(self) -> None:
        model = FemMHCJointModel(
            input_dim=32,
            hidden_dim=32,
            maximum_days=14,
            cycle_modes=4,
            dropout=0.0,
            architecture="dual_path_router",
            initialization_seed=42,
        )
        calls = {"key": 0, "value": 0}
        hooks = [
            model.task_heads.routing_key.register_forward_hook(
                lambda *_: calls.__setitem__("key", calls["key"] + 1)
            ),
            model.task_heads.routing_value.register_forward_hook(
                lambda *_: calls.__setitem__("value", calls["value"] + 1)
            ),
        ]
        model(
            torch.randn(2, 5, 32),
            task_ids=(
                "mcphases/fatigue",
                "depress/negative_affect",
                "openmhc/sleep_duration",
            ),
        )
        for hook in hooks:
            hook.remove()
        self.assertEqual(calls, {"key": 1, "value": 1})

    def test_standard_mixture_baselines_emit_normalized_task_gates(self) -> None:
        embeddings = torch.randn(3, 5, 32)
        task_ids = ("mcphases/fatigue", "openmhc/sleep_duration")
        expected_width = {"mmoe": 8, "ple": 3}
        for architecture, width in expected_width.items():
            model = FemMHCJointModel(
                input_dim=32,
                hidden_dim=32,
                maximum_days=14,
                dropout=0.0,
                architecture=architecture,
                initialization_seed=42,
            )
            output = model(embeddings, task_ids=task_ids)
            self.assertEqual(set(output.routing_attention), set(task_ids))
            for attention in output.routing_attention.values():
                self.assertEqual(tuple(attention.shape), (3, width))
                torch.testing.assert_close(
                    attention.sum(dim=-1),
                    torch.ones(3),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_partial_labels_skip_absent_tasks_and_missing_values(self) -> None:
        task_ids = (
            "mcphases/menstrual_onset_24h",
            "mcphases/menstrual_onset_72h",
            "mcphases/fatigue",
            "depress/negative_affect",
            "openmhc/sleep_duration",
        )
        output = self.model(torch.randn(4, 5, 32), task_ids=task_ids)
        targets = {
            "mcphases/menstrual_onset_24h": torch.tensor([1, 0, -1, 0]),
            "mcphases/menstrual_onset_72h": torch.tensor([1, 1, -1, 0]),
            "mcphases/fatigue": torch.tensor([2, 3, -1, 1]),
            "depress/negative_affect": torch.tensor([1.0, float("nan"), 3.0, 2.0]),
            # No sleep-duration labels in this cohort batch.
        }
        losses = partial_multitask_loss(output, targets)
        self.assertTrue(bool(torch.isfinite(losses.total)))
        self.assertIn("mcphases/menstrual_onset_nested", losses.per_task)
        self.assertNotIn("openmhc/sleep_duration", losses.per_task)
        self.assertEqual(losses.observed_counts["mcphases/menstrual_onset_nested"], 3)
        losses.total.backward()
        self.assertIsNotNone(self.model.state_encoder.graph.query.weight.grad)

    def test_temporal_paths_do_not_read_future_days(self) -> None:
        self.model.eval()
        prefix = torch.randn(2, 3, 32)
        first = torch.cat([prefix, torch.zeros(2, 2, 32)], dim=1)
        second = torch.cat([prefix, torch.randn(2, 2, 32) * 100.0], dim=1)
        with torch.inference_mode():
            first_output = self.model(first, task_ids=("mcphases/fatigue",))
            second_output = self.model(second, task_ids=("mcphases/fatigue",))
        torch.testing.assert_close(
            first_output.states.general_sequence[:, :3],
            second_output.states.general_sequence[:, :3],
        )
        torch.testing.assert_close(
            first_output.states.cycle_sequence[:, :3],
            second_output.states.cycle_sequence[:, :3],
        )


if __name__ == "__main__":
    unittest.main()
