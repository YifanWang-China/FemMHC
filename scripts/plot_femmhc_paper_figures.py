#!/usr/bin/env python
"""Create reproducible manuscript figures from frozen FemMHC artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
NAVY = "#111827"
TEAL = "#22A7B5"
BLUE = "#3973D5"
CORAL = "#EF635B"
VIOLET = "#8B5CF6"
AMBER = "#E9AD3F"
PALE = "#F4F7FA"
GRID = "#DDE4EC"
GRAY = "#8A97A8"


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _box(ax, xy, width, height, text, *, facecolor, edgecolor=None, fontsize=9, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor or facecolor,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NAVY,
        weight=weight,
    )
    return patch


def _arrow(ax, start, end, *, color=GRAY, width=1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.2, 6.88, "General + female-specific wearable cohorts", fontsize=14, weight="bold", color=NAVY)
    cohorts = [
        "OpenMHC XS",
        "mcPHASES",
        "DEPRESS Fitbit",
        "inPHRsym",
        "HRV + sleep",
        "Pregnancy clock",
    ]
    for index, cohort in enumerate(cohorts):
        col, row = index % 2, index // 2
        x = 0.25 + col * 1.65
        y = 5.75 - row * 1.23
        _box(ax, (x, y), 1.45, 0.78, cohort, facecolor="#EEF7F8", edgecolor="#A7DDE2", fontsize=8.5, weight="bold")
        xs = np.linspace(x + 0.13, x + 1.32, 30)
        ys = y + 0.16 + 0.045 * np.sin(np.linspace(0, 4 * np.pi, 30) + index)
        ax.plot(xs, ys, color=TEAL, linewidth=1.2)
    ax.text(1.72, 2.02, "longitudinal days + partial labels", ha="center", fontsize=9, color=GRAY)

    _arrow(ax, (3.55, 4.25), (4.15, 4.25), color=TEAL, width=2.0)
    _box(ax, (4.18, 3.42), 1.62, 1.7, "OpenMHC\ndaily encoder", facecolor="#DFF3F5", edgecolor=TEAL, fontsize=12, weight="bold")
    ax.text(4.99, 3.68, "cached daily tokens", ha="center", fontsize=8, color=GRAY)

    _arrow(ax, (5.84, 4.25), (6.35, 4.25), color=BLUE, width=2.0)
    ax.text(6.34, 6.88, "FemMHC dual-path representation", fontsize=14, weight="bold", color=NAVY)
    _box(ax, (6.42, 4.65), 3.55, 1.05, "General causal temporal state", facecolor="#E8F0FC", edgecolor=BLUE, fontsize=11, weight="bold")
    _box(ax, (6.42, 2.34), 3.55, 1.67, "", facecolor="#FFF0EE", edgecolor=CORAL, fontsize=11, weight="bold")
    ax.text(6.68, 3.76, "Female temporal state", fontsize=11, weight="bold", color=NAVY, va="center")
    _box(ax, (6.66, 2.57), 1.05, 0.68, "CycleSSM", facecolor="#F0E9FE", edgecolor=VIOLET, fontsize=9, weight="bold")
    domains = ["Cycle", "Sleep", "Affect", "ANS", "Activity", "Cardio", "Life stage", "Context"]
    for index, domain in enumerate(domains):
        col, row = index % 4, index // 4
        _box(
            ax,
            (7.92 + col * 0.49, 3.14 - row * 0.55),
            0.44,
            0.42,
            domain,
            facecolor="#FFFFFF",
            edgecolor="#F1AAA3",
            fontsize=5.6,
        )
    _arrow(ax, (8.2, 4.62), (9.95, 4.18), color=BLUE)
    _arrow(ax, (8.2, 4.03), (9.95, 4.18), color=CORAL)
    _box(ax, (9.92, 3.72), 0.95, 0.92, "Task\nfusion", facecolor="#FFF8E8", edgecolor=AMBER, fontsize=9, weight="bold")

    _arrow(ax, (10.9, 4.18), (11.38, 4.18), color=AMBER, width=2.0)
    ax.text(11.4, 6.88, "Unified women’s health tasks", fontsize=14, weight="bold", color=NAVY)
    task_labels = [
        "Menstrual onset",
        "Cycle symptoms",
        "Mood + stress",
        "Sleep + recovery",
        "Autonomic state",
        "Activity load",
        "Cardiometabolic",
        "Pregnancy stage",
    ]
    task_colors = ["#FCE8F1", "#FCE8F1", "#F0E9FE", "#E8F0FC", "#E8F5F3", "#E8F5F3", "#FFF4E2", "#FFF0EE"]
    for index, label in enumerate(task_labels):
        col, row = index % 2, index // 2
        _box(ax, (11.45 + col * 2.02, 5.7 - row * 0.88), 1.82, 0.61, label, facecolor=task_colors[index], edgecolor=GRID, fontsize=8.5, weight="bold")
    _box(ax, (11.48, 1.78), 3.82, 0.76, "69 evaluated tasks  •  6 cohorts  •  44–19 vs MMoE (val.)", facecolor="#EEF7F8", edgecolor=TEAL, fontsize=9.7, weight="bold")

    _box(ax, (0.25, 0.32), 15.05, 0.82, "", facecolor=PALE, edgecolor=GRID, fontsize=10, weight="bold")
    ax.text(0.58, 0.72, "Participant-disjoint protocol", fontsize=10, weight="bold", color=NAVY, va="center")
    _box(ax, (4.48, 0.49), 1.35, 0.46, "Train", facecolor="#E8F5F3", edgecolor=TEAL, fontsize=9, weight="bold")
    _box(ax, (6.02, 0.49), 1.55, 0.46, "Validation", facecolor="#FFF5DD", edgecolor=AMBER, fontsize=9, weight="bold")
    _box(ax, (7.76, 0.49), 1.72, 0.46, "Locked test", facecolor="#F0F2F5", edgecolor=GRAY, fontsize=9, weight="bold")
    ax.text(9.77, 0.72, "model selection uses validation only", fontsize=8.5, color=GRAY, va="center")

    fig.tight_layout(pad=0.5)
    fig.savefig(FIGURES / "Figure1_FemMHC_graphical_abstract.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure1_FemMHC_graphical_abstract.svg", bbox_inches="tight")
    plt.close(fig)


def results_figure() -> None:
    dual_shared = _json("artifacts/benchmark/femmhc-joint-formal/dual-vs-shared-multiseed-validation/summary.json")
    mmoe_shared = _json("artifacts/benchmark/femmhc-joint-formal/mmoe-vs-shared-multiseed-validation/summary.json")
    dual_mmoe = _json("artifacts/benchmark/femmhc-joint-formal/dual-vs-mmoe-multiseed-validation/summary.json")
    bootstrap = _json("artifacts/benchmark/femmhc-joint-formal/dual-vs-mmoe-seed42-participant-bootstrap-validation/summary.json")
    routing = pd.read_csv(ROOT / "artifacts/benchmark/femmhc-joint-formal/dual-path-router-seed42-routing-validation/domain_routing.csv")

    fig = plt.figure(figsize=(15.5, 11.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    comparisons = [
        ("FemMHC vs Shared", dual_shared),
        ("MMoE vs Shared", mmoe_shared),
        ("FemMHC vs MMoE", dual_mmoe),
    ]
    y_positions = np.arange(6)[::-1]
    labels, wins, losses, colors = [], [], [], []
    for name, summary in comparisons:
        for scope_key, scope_label in (("all_tasks", "All"), ("female_specific_tasks", "Female")):
            item = summary[scope_key]
            labels.append(f"{name} — {scope_label}")
            wins.append(item["candidate_mean_wins"])
            losses.append(item["candidate_mean_losses"])
            colors.append(CORAL if name.startswith("FemMHC") else TEAL)
    ax.barh(y_positions, wins, color=colors, edgecolor="none", label="Candidate wins")
    ax.barh(y_positions, losses, left=wins, color="#DDE3EA", edgecolor="none", label="Candidate losses")
    for y, win, loss in zip(y_positions, wins, losses):
        ax.text(win / 2, y, str(win), ha="center", va="center", color="white", weight="bold", fontsize=9)
        ax.text(win + loss / 2, y, str(loss), ha="center", va="center", color=NAVY, fontsize=9)
    ax.set_yticks(y_positions, labels, fontsize=8.5)
    ax.set_xlabel("Primary tasks (three-seed mean)")
    ax.set_title("A  Pairwise primary-task wins", loc="left", weight="bold", fontsize=13)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[0, 1])
    domain_order = ["menstrual", "sleep_recovery", "affect_stress", "autonomic", "activity_load", "cardiometabolic", "life_stage", "context"]
    domain_labels = ["Menstrual", "Sleep", "Affect", "Autonomic", "Activity", "Cardiometabolic", "Life stage", "Context"]
    domain_wins = [dual_mmoe["domains"][domain]["candidate_mean_wins"] for domain in domain_order]
    domain_losses = [dual_mmoe["domains"][domain]["candidate_mean_losses"] for domain in domain_order]
    yy = np.arange(len(domain_order))[::-1]
    ax.barh(yy, domain_wins, color=CORAL, label="FemMHC wins")
    ax.barh(yy, [-value for value in domain_losses], color="#AEB8C5", label="MMoE wins")
    ax.axvline(0, color=NAVY, linewidth=0.8)
    for y, win, loss in zip(yy, domain_wins, domain_losses):
        ax.text(win + 0.15, y, str(win), va="center", fontsize=8, color=CORAL, weight="bold")
        ax.text(-loss - 0.15, y, str(loss), va="center", ha="right", fontsize=8, color=NAVY)
    ax.set_yticks(yy, domain_labels, fontsize=9)
    ax.set_xlabel("Task wins (left: MMoE, right: FemMHC)")
    ax.set_title("B  FemMHC vs MMoE by health domain", loc="left", weight="bold", fontsize=13)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 0])
    evidence_labels = ["Eligible tasks", "95% CI > 0", "95% CI < 0", "Holm-adjusted wins"]
    evidence_values = [
        bootstrap["all_tasks"]["eligible_tasks"],
        bootstrap["all_tasks"]["confidence_interval_above_zero"],
        bootstrap["all_tasks"]["confidence_interval_below_zero"],
        bootstrap["all_tasks"]["holm_significant_candidate_wins"],
    ]
    bars = ax.barh(np.arange(4)[::-1], evidence_values, color=[BLUE, CORAL, "#AEB8C5", AMBER])
    ax.set_yticks(np.arange(4)[::-1], evidence_labels, fontsize=9)
    ax.set_xlim(0, 45)
    for bar, value in zip(bars, evidence_values):
        ax.text(value + 0.7, bar.get_y() + bar.get_height() / 2, str(value), va="center", weight="bold", color=NAVY)
    ax.set_xlabel("Number of primary tasks")
    ax.set_title("C  Participant-cluster bootstrap evidence", loc="left", weight="bold", fontsize=13)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 1])
    matrix = routing.set_index("domain")[[f"attention_{domain}" for domain in domain_order]].reindex(domain_order).to_numpy()
    cmap = LinearSegmentedColormap.from_list("femmhc", ["#F4F7FA", "#C9E8EB", TEAL, BLUE])
    image = ax.imshow(matrix, vmin=0.0, vmax=0.55, cmap=cmap, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[row, col] > 0.32 else NAVY)
    ax.set_xticks(np.arange(8), ["Cycle", "Sleep", "Affect", "ANS", "Activity", "Cardio", "Life", "Context"], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(8), domain_labels, fontsize=8.5)
    ax.set_xlabel("Routed health state")
    ax.set_ylabel("Task family")
    ax.set_title("D  Routing allocation (own-domain prior included)", loc="left", weight="bold", fontsize=13)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean attention", fontsize=8)

    fig.suptitle("FemMHC validation evidence after architecture freeze", x=0.055, y=0.985, ha="left", fontsize=17, weight="bold", color=NAVY)
    fig.text(0.055, 0.953, "Three seeds for task-level comparisons; participant-cluster uncertainty for seed 42", ha="left", fontsize=10, color=GRAY)
    fig.savefig(FIGURES / "Figure3_FemMHC_validation_results.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure3_FemMHC_validation_results.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    graphical_abstract()
    results_figure()
    print(
        json.dumps(
            {
                "status": "complete",
                "figures": [
                    str(FIGURES / "Figure1_FemMHC_graphical_abstract.png"),
                    str(FIGURES / "Figure3_FemMHC_validation_results.png"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
