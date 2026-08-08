"""Create publication-ready vector figures for the FemMHC arXiv manuscript.

All numerical panels read the locked source-data tables in
``paper/femmhc_arxiv/source_data``. The script exports PDF/SVG for the paper and
450-dpi PNG files for rapid review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY = "#111827"
TEAL = "#1596A7"
TEAL_LIGHT = "#DFF3F5"
AMBER = "#D99A21"
AMBER_LIGHT = "#FFF2D5"
CORAL = "#E85D5D"
CORAL_LIGHT = "#FCE8E8"
BLUE = "#3974D5"
BLUE_LIGHT = "#E7EFFC"
GRAY = "#6B7280"
LIGHT = "#E5EAF1"
PALE = "#F7F9FC"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": NAVY,
            "axes.linewidth": 0.7,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def save_all(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def box(ax, xy, width, height, text, *, face, edge, size=8.5, weight="semibold"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        weight=weight,
        color=NAVY,
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, color=GRAY, width=1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def graphical_abstract(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.95, "Heterogeneous longitudinal cohorts", weight="bold", fontsize=10)
    cohorts = [
        ("OpenMHC\nretention", 0.02, 0.72),
        ("mcPHASES\ncycle + symptoms", 0.17, 0.72),
        ("DEPRESS Fitbit\nmood scales", 0.02, 0.49),
        ("inPHRsym\nnext-day affect", 0.17, 0.49),
        ("HRV + sleep", 0.02, 0.26),
        ("Pregnancy\nactivity", 0.17, 0.26),
    ]
    for label, x, y in cohorts:
        box(ax, (x, y), 0.13, 0.15, label, face=PALE, edge=TEAL, size=7.2)
    ax.text(0.02, 0.11, "Partial labels  •  missing days  •  participant-disjoint evaluation", fontsize=7.2, color=GRAY)

    box(ax, (0.38, 0.56), 0.19, 0.26, "Frozen OpenMHC\ndaily encoder\n21.54M parameters", face=BLUE_LIGHT, edge=BLUE, size=9)
    ax.text(0.475, 0.84, "99% frozen", ha="center", fontsize=7.5, color=BLUE, weight="bold")
    box(ax, (0.41, 0.31), 0.13, 0.13, "Female adapter\n239k parameters\n(1.11%)", face=TEAL_LIGHT, edge=TEAL, size=7.6)
    box(ax, (0.60, 0.43), 0.13, 0.20, "Causal history\n+ task-family\nheads", face=AMBER_LIGHT, edge=AMBER, size=8)
    arrow(ax, (0.31, 0.58), (0.38, 0.66), TEAL)
    arrow(ax, (0.475, 0.56), (0.475, 0.44), TEAL)
    arrow(ax, (0.54, 0.375), (0.60, 0.50), AMBER)

    ax.text(0.77, 0.95, "Selective transfer", weight="bold", fontsize=10)
    outputs = [
        ("Cycle phase", 0.77, 0.72, TEAL_LIGHT, TEAL, "+8.15%*"),
        ("Cramps", 0.89, 0.72, TEAL_LIGHT, TEAL, "+9.32%*"),
        ("Sleep\nsymptoms", 0.77, 0.49, TEAL_LIGHT, TEAL, "+9.43%*"),
        ("24 h onset", 0.89, 0.49, CORAL_LIGHT, CORAL, "−3.40%*"),
        ("HRV + sleep", 0.77, 0.26, TEAL_LIGHT, TEAL, "+2.01%†"),
        ("Pregnancy", 0.89, 0.26, CORAL_LIGHT, CORAL, "−1.51%†"),
    ]
    for label, x, y, face, edge, value in outputs:
        box(ax, (x, y), 0.10, 0.15, f"{label}\n{value}", face=face, edge=edge, size=7.2)
    arrow(ax, (0.73, 0.53), (0.77, 0.79), TEAL)
    arrow(ax, (0.73, 0.53), (0.77, 0.565), TEAL)
    arrow(ax, (0.73, 0.53), (0.77, 0.335), TEAL)
    ax.text(0.77, 0.11, "* fixed split; † continued vs. static adapter", fontsize=6.7, color=GRAY)
    ax.text(0.60, 0.16, "No universal advantage\nover GRU or MMoE", ha="center", fontsize=7.5, color=CORAL, weight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.03)
    save_all(fig, output_dir, "figure1_graphical_abstract")


def core_results(source_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(source_dir / "core_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharex=True)
    panels = [
        ("Fixed participant split (3 seeds)", "Fixed participant split"),
        ("Nested leave-one-participant-out (42 women)", "Nested leave-one-participant-out"),
    ]
    for panel, (title, protocol) in zip(axes, panels):
        sub = df[df.protocol == protocol].copy().iloc[::-1]
        y = np.arange(len(sub))
        vals = sub.relative_improvement_percent.to_numpy()
        colors = [TEAL if v >= 0 else CORAL for v in vals]
        panel.axvline(0, color=NAVY, lw=0.8)
        panel.hlines(y, 0, vals, color=colors, lw=2.2)
        panel.scatter(vals, y, c=colors, s=32, zorder=3, edgecolor="white", linewidth=0.6)
        for yi, value in zip(y, vals):
            panel.text(value + 0.35, yi, f"{value:+.2f}%", va="center", ha="left", fontsize=7.2, weight="semibold")
        panel.set_yticks(y, sub.task)
        panel.set_title(title, loc="left", weight="bold")
        panel.set_xlim(-5.2, 10.8)
        panel.grid(axis="x", color=LIGHT, linewidth=0.7)
        panel.spines[["top", "right", "left"]].set_visible(False)
        panel.tick_params(axis="y", length=0)
        panel.set_xlabel("Relative improvement (higher is better)")
    axes[0].text(-0.16, 1.08, "a", transform=axes[0].transAxes, fontsize=11, weight="bold")
    axes[1].text(-0.16, 1.08, "b", transform=axes[1].transAxes, fontsize=11, weight="bold")
    fig.tight_layout(w_pad=1.5)
    save_all(fig, output_dir, "figure2_menstrual_transfer")


def architecture_and_domains(source_dir: Path, output_dir: Path) -> None:
    arch = pd.read_csv(source_dir / "architecture_results.csv")
    domain = pd.read_csv(source_dir / "domain_transfer.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))

    a = axes[0]
    order = ["Latest-day shared MLP", "FemMHC dual path", "MMoE (8 experts)", "Shared GRU adapter"]
    sub = arch.set_index("model").loc[order].reset_index()
    y = np.arange(len(sub))
    colors = [GRAY, TEAL, AMBER, BLUE]
    for yi, mean, sd, color in zip(y, sub.validation_loss_mean, sub.validation_loss_sd, colors):
        a.errorbar(mean, yi, xerr=sd, fmt="none", ecolor=color, elinewidth=2.0, capsize=3)
        a.scatter(mean, yi, c=color, s=38, zorder=3, edgecolor="white", linewidth=0.6)
    a.set_yticks(y, ["Latest-day MLP", "FemMHC", "MMoE", "Shared GRU"])
    a.invert_yaxis()
    a.set_xlabel("Validation loss (mean ± SD; lower is better)")
    a.set_title("Capacity-matched temporal comparison", loc="left", weight="bold")
    a.grid(axis="x", color=LIGHT, linewidth=0.7)
    a.spines[["top", "right", "left"]].set_visible(False)
    a.tick_params(axis="y", length=0)
    a.text(-0.16, 1.08, "a", transform=a.transAxes, fontsize=11, weight="bold")

    b = axes[1]
    vals = domain.relative_improvement_percent.to_numpy()[::-1]
    labels = domain.domain.to_numpy()[::-1]
    y = np.arange(len(vals))
    colors2 = [TEAL if v >= 0 else CORAL for v in vals]
    b.axvline(0, color=NAVY, lw=0.8)
    b.barh(y, vals, color=colors2, height=0.55)
    for yi, value in zip(y, vals):
        if value >= 0:
            b.text(value + 0.08, yi, f"{value:+.2f}%", va="center", ha="left", fontsize=7.3, weight="semibold")
        else:
            b.text(value + 0.08, yi, f"{value:+.2f}%", va="center", ha="left", fontsize=7.3, weight="semibold", color="white")
    b.set_yticks(y, labels)
    b.set_xlim(-2.0, 2.5)
    b.set_xlabel("Continued vs. static adapter improvement")
    b.set_title("Domain-dependent continued pretraining", loc="left", weight="bold")
    b.grid(axis="x", color=LIGHT, linewidth=0.7)
    b.spines[["top", "right", "left"]].set_visible(False)
    b.tick_params(axis="y", length=0)
    b.text(-0.16, 1.08, "b", transform=b.transAxes, fontsize=11, weight="bold")
    fig.tight_layout(w_pad=2.2)
    save_all(fig, output_dir, "figure3_capacity_and_transfer")


def reliability(source_dir: Path, output_dir: Path) -> None:
    cal = pd.read_csv(source_dir / "calibration.csv")
    rob = pd.read_csv(source_dir / "robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))

    a = axes[0]
    x = np.arange(len(cal))
    width = 0.34
    a.bar(x - width / 2, cal.brier_reduction_percent, width, color=BLUE, label="Brier reduction")
    a.bar(x + width / 2, cal.ece_reduction_percent, width, color=TEAL, label="ECE reduction")
    for xx, value in zip(x - width / 2, cal.brier_reduction_percent):
        a.text(xx, value + 2, f"{value:.1f}%", ha="center", fontsize=7.2, weight="semibold")
    for xx, value in zip(x + width / 2, cal.ece_reduction_percent):
        a.text(xx, value + 2, f"{value:.1f}%", ha="center", fontsize=7.2, weight="semibold")
    a.set_xticks(x, ["24 h onset", "72 h onset"])
    a.set_ylim(0, 100)
    a.set_ylabel("Relative calibration-error reduction")
    a.set_title("Train-only probability calibration", loc="left", weight="bold")
    a.legend(frameon=False, loc="upper left")
    a.grid(axis="y", color=LIGHT, linewidth=0.7)
    a.spines[["top", "right"]].set_visible(False)
    a.text(-0.16, 1.08, "a", transform=a.transAxes, fontsize=11, weight="bold")

    b = axes[1]
    styles = {
        "Random deletion": (TEAL, "o"),
        "Contiguous deletion": (AMBER, "s"),
        "Latest-day deletion": (CORAL, "D"),
    }
    for name, group in rob.groupby("missingness_type", sort=False):
        color, marker = styles[name]
        xvals = np.arange(1, len(group) + 1)
        b.plot(xvals, group.relative_change_percent, color=color, marker=marker, linewidth=1.8, markersize=4.5, label=name)
        value = float(group.relative_change_percent.iloc[-1])
        b.text(3.0, value + (0.75 if value >= 0 else -0.9), f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=6.8)
    b.axhline(0, color=NAVY, lw=0.8)
    b.set_xticks([1, 2, 3], ["Low", "Moderate", "High"])
    b.set_ylim(-21, 5)
    b.set_ylabel("Median relative performance change")
    b.set_title("Recent history is the critical failure mode", loc="left", weight="bold")
    b.legend(frameon=False, loc="lower left")
    b.grid(axis="y", color=LIGHT, linewidth=0.7)
    b.spines[["top", "right"]].set_visible(False)
    b.text(-0.16, 1.08, "b", transform=b.transAxes, fontsize=11, weight="bold")
    fig.tight_layout(w_pad=2.0)
    save_all(fig, output_dir, "figure4_calibration_robustness")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=Path("paper/femmhc_arxiv"))
    args = parser.parse_args()
    root = args.paper_root.resolve()
    configure_style()
    graphical_abstract(root / "figures")
    core_results(root / "source_data", root / "figures")
    architecture_and_domains(root / "source_data", root / "figures")
    reliability(root / "source_data", root / "figures")


if __name__ == "__main__":
    main()
