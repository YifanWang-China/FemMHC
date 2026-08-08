#!/usr/bin/env python
"""Summarize task-conditioned health-state routing on a sealed data split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bootstrap_femmhc_joint_pair import _add_data_arguments, _load_model
from evaluate_femmhc_joint import _datasets
from femmhc import HEALTH_DOMAINS, JOINT_TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    _add_data_arguments(parser)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("test evaluation is locked; pass --allow-test only for a frozen final analysis")

    device = torch.device(args.device)
    model, _, artifact = _load_model(args.checkpoint, device)
    if not model.task_heads.task_conditioned_routing:
        raise ValueError("checkpoint does not contain a task-conditioned router")
    if model.task_heads.routing_gate_logits is None:
        raise RuntimeError("routing gate logits are missing")
    task_by_id = {task.task_id: task for task in JOINT_TASKS if task.trainable}
    collected: dict[str, dict[str, Any]] = {}

    with torch.inference_mode():
        for cohort, dataset in _datasets(args).items():
            loader = DataLoader(
                dataset,
                batch_size=min(args.batch_size, len(dataset)),
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            for batch in loader:
                task_ids = tuple(batch["targets"])
                embeddings = batch["daily_embeddings"].to(device, non_blocking=True)
                present = batch["day_present"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model(embeddings, present, task_ids=task_ids)
                for task_id, target_tensor in batch["targets"].items():
                    attention = output.routing_attention.get(task_id)
                    if attention is None:
                        continue
                    task = task_by_id[task_id]
                    target = target_tensor.numpy()
                    observed = np.isfinite(target)
                    if task.kind != "regression":
                        observed &= target >= 0
                    if not observed.any():
                        continue
                    values = attention.float().cpu().numpy()[observed]
                    entropy = -np.sum(values * np.log(np.clip(values, 1e-12, 1.0)), axis=1)
                    record = collected.setdefault(
                        task_id,
                        {
                            "sum": np.zeros(len(HEALTH_DOMAINS), dtype=np.float64),
                            "sum_squares": np.zeros(len(HEALTH_DOMAINS), dtype=np.float64),
                            "entropy_sum": 0.0,
                            "count": 0,
                            "participants": set(),
                            "cohort": cohort,
                        },
                    )
                    record["sum"] += values.sum(axis=0)
                    record["sum_squares"] += np.square(values).sum(axis=0)
                    record["entropy_sum"] += float(entropy.sum())
                    record["count"] += len(values)
                    record["participants"].update(
                        np.asarray(batch["participant_id"], dtype=str)[observed].tolist()
                    )

    rows: list[dict[str, Any]] = []
    onset_24h = model.task_heads.ONSET_24H
    onset_72h = model.task_heads.ONSET_72H
    for task_id, record in sorted(collected.items()):
        task = task_by_id[task_id]
        count = int(record["count"])
        mean = record["sum"] / count
        variance = np.maximum(record["sum_squares"] / count - np.square(mean), 0.0)
        effective_route_task = onset_24h if task_id == onset_72h else task_id
        gate_index = model.task_heads.routing_index[effective_route_task]
        gate = float(
            torch.sigmoid(model.task_heads.routing_gate_logits[gate_index]).item()
        )
        own_index = HEALTH_DOMAINS.index(task.domain)
        order = np.argsort(-mean)
        row: dict[str, Any] = {
            "task_id": task_id,
            "effective_route_task": effective_route_task,
            "source": task.source,
            "domain": task.domain,
            "samples": count,
            "participants": len(record["participants"]),
            "routing_gate": gate,
            "own_domain_attention": float(mean[own_index]),
            "cross_domain_attention": float(1.0 - mean[own_index]),
            "normalized_entropy": float(
                record["entropy_sum"] / count / math.log(len(HEALTH_DOMAINS))
            ),
            "top_domain": HEALTH_DOMAINS[int(order[0])],
            "second_domain": HEALTH_DOMAINS[int(order[1])],
            "top_attention": float(mean[order[0]]),
        }
        for index, domain in enumerate(HEALTH_DOMAINS):
            row[f"attention_{domain}"] = float(mean[index])
            row[f"attention_sd_{domain}"] = float(np.sqrt(variance[index]))
        rows.append(row)

    task_frame = pd.DataFrame(rows)
    domain_rows = []
    for domain, group in task_frame.groupby("domain", sort=True):
        row = {
            "domain": domain,
            "tasks": len(group),
            "mean_routing_gate": float(group["routing_gate"].mean()),
            "mean_own_domain_attention": float(group["own_domain_attention"].mean()),
            "mean_cross_domain_attention": float(group["cross_domain_attention"].mean()),
            "mean_normalized_entropy": float(group["normalized_entropy"].mean()),
        }
        for routed_domain in HEALTH_DOMAINS:
            row[f"attention_{routed_domain}"] = float(
                group[f"attention_{routed_domain}"].mean()
            )
        domain_rows.append(row)
    domain_frame = pd.DataFrame(domain_rows)
    cross_domain = task_frame[task_frame["top_domain"] != task_frame["domain"]]
    summary = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(artifact["step"]),
        "architecture": str(artifact.get("architecture", "unknown")),
        "split": args.split,
        "tasks": int(len(task_frame)),
        "health_domains": list(HEALTH_DOMAINS),
        "routing_gate": {
            "mean": float(task_frame["routing_gate"].mean()),
            "median": float(task_frame["routing_gate"].median()),
            "minimum": float(task_frame["routing_gate"].min()),
            "maximum": float(task_frame["routing_gate"].max()),
        },
        "mean_own_domain_attention": float(task_frame["own_domain_attention"].mean()),
        "mean_cross_domain_attention": float(task_frame["cross_domain_attention"].mean()),
        "mean_normalized_entropy": float(task_frame["normalized_entropy"].mean()),
        "tasks_with_cross_domain_top_route": int(len(cross_domain)),
        "interpretation": (
            "Attention describes routing allocation, while the gate describes how much "
            "the routed state changes the shared state. Neither is a causal attribution."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_frame.to_csv(args.output_dir / "task_routing.csv", index=False)
    domain_frame.to_csv(args.output_dir / "domain_routing.csv", index=False)
    (args.output_dir / "routing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# FemMHC 任务路由解释性分析",
        "",
        f"- 数据划分：{args.split}",
        f"- 分析任务：{len(task_frame)}",
        f"- 平均路由门控：{summary['routing_gate']['mean']:.3f}",
        f"- 平均本域注意力：{summary['mean_own_domain_attention']:.3f}",
        f"- 平均跨域注意力：{summary['mean_cross_domain_attention']:.3f}",
        f"- 跨域成为最高权重的任务：{len(cross_domain)}",
        "",
        "| 任务域 | 任务数 | 平均门控 | 本域注意力 | 跨域注意力 | 归一化熵 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in domain_frame.itertuples():
        lines.append(
            f"| {row.domain} | {row.tasks} | {row.mean_routing_gate:.3f} | "
            f"{row.mean_own_domain_attention:.3f} | {row.mean_cross_domain_attention:.3f} | "
            f"{row.mean_normalized_entropy:.3f} |"
        )
    lines.extend(
        [
            "",
            "注意力表示模型在健康状态之间的分配，门控表示该路由对共享状态的实际改变量；二者均不是因果归因。",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
