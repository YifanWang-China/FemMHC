#!/usr/bin/env python
"""One-step smoke test for the partially labelled FemMHC joint model."""

from __future__ import annotations

import argparse
import json

import torch

from femmhc import FemMHCJointModel, partial_multitask_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42)
    model = FemMHCJointModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        maximum_days=max(args.days, 60),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    embeddings = torch.randn(
        args.batch_size, args.days, args.input_dim, device=device
    )
    task_ids = (
        "mcphases/menstrual_onset_24h",
        "mcphases/menstrual_onset_72h",
        "mcphases/fatigue",
        "depress/negative_affect",
        "openmhc/sleep_duration",
        "openmhc/vigorous_activity",
    )
    output = model(embeddings, task_ids=task_ids)
    missing_class = torch.full((args.batch_size,), -1, device=device, dtype=torch.long)
    missing_class[::2] = torch.randint(0, 4, (len(missing_class[::2]),), device=device)
    targets = {
        "mcphases/menstrual_onset_24h": torch.randint(0, 2, (args.batch_size,), device=device),
        "mcphases/menstrual_onset_72h": torch.ones(args.batch_size, device=device, dtype=torch.long),
        "mcphases/fatigue": torch.randint(0, 6, (args.batch_size,), device=device),
        "depress/negative_affect": torch.randn(args.batch_size, device=device),
        "openmhc/sleep_duration": missing_class,
        # This cohort batch intentionally has no vigorous-activity labels.
    }
    targets["mcphases/menstrual_onset_72h"] = torch.maximum(
        targets["mcphases/menstrual_onset_24h"],
        targets["mcphases/menstrual_onset_72h"],
    )
    losses = partial_multitask_loss(output, targets)
    optimizer.zero_grad(set_to_none=True)
    losses.total.backward()
    optimizer.step()
    report = {
        "status": "ok",
        "device": str(device),
        "joint_loss": float(losses.total.detach().cpu()),
        "active_tasks": sorted(losses.per_task),
        "active_domains": sorted(losses.per_domain),
        "observed_counts": dict(losses.observed_counts),
        "shared_state_shape": list(output.states.shared_state.shape),
        "relation_attention_shape": list(output.states.relation_attention.shape),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
