#!/usr/bin/env python
"""Re-render FemMHC manuscript figures with a Nature-style layout.

The script reads only the locked source-data tables in ``paper/femmhc_arxiv``.
It keeps the numerical values unchanged, uses a colour-vision-safe palette and
places every annotation with explicit margins so labels do not collide with
marks, axes or neighbouring panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "femmhc_arxiv"
SOURCE = PAPER / "source_data"
FIGURES = PAPER / "figures"

# Okabe-Ito inspired, colour-vision-safe palette.
INK = "#1F2937"
FEMMHC = "#0072B2"
POSITIVE = "#009E73"
NEGATIVE = "#D55E00"
AMBER = "#E69F00"
BASELINE = "#6B7280"
GRID = "#D9E2EC"
LIGHT = "#F3F6F9"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def save(fig: mpl.figure.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, letter: str) -> None:
    ax.text(
        -0.08,
        1.06,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def clean_axes(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(length=3, width=0.7)
    ax.set_axisbelow(True)


def value_label(ax: mpl.axes.Axes, x: float, y: float, text: str, *, xlim: tuple[float, float]) -> None:
    """Place a value label outside a point/bar with a guaranteed visual gap."""
    span = xlim[1] - xlim[0]
    pad = 0.018 * span
    if abs(x) < 0.45:
        # Keep near-zero annotations away from the zero spine and marker.
        side = 1 if x >= 0 else -1
        ax.text(x + side * 0.75, y, text, ha="left" if side > 0 else "right", va="center", fontsize=7.4, fontweight="bold")
    elif x >= 0:
        ax.text(x + pad, y, text, ha="left", va="center", fontsize=7.4, fontweight="bold")
    else:
        # Keep negative annotations inside the plotting region.  A label far
        # to the left of the point can collide with long categorical labels.
        ax.text(
            x,
            y + 0.20,
            text,
            ha="center",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        )


def fig2_menstrual() -> None:
    df = pd.read_csv(SOURCE / "core_results.csv")
    protocols = [
        ("Fixed participant split\n(3 seeds)", "Fixed participant split"),
        ("Nested leave-one-participant-out\n(42 women)", "Nested leave-one-participant-out"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True)
    xlim = (-5.7, 11.0)
    for ax, (title, protocol) in zip(axes, protocols):
        sub = df[df["protocol"] == protocol].copy()
        sub = sub.iloc[::-1]
        y = np.arange(len(sub))
        ax.axvline(0, color=INK, linewidth=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, linewidth=0.65)
        for yi, val, task in zip(y, sub["relative_improvement_percent"], sub["task"]):
            color = POSITIVE if val >= 0 else NEGATIVE
            ax.hlines(yi, 0, val, color=color, linewidth=2.6, zorder=2)
            ax.scatter([val], [yi], s=34, color=color, edgecolor="white", linewidth=0.7, zorder=3)
            value_label(ax, float(val), yi, f"{val:+.2f}%", xlim=xlim)
        ax.set_yticks(y, sub["task"])
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.55, len(sub) - 0.45)
        ax.set_xlabel("Relative change in task metric (%)")
        ax.set_title(title, loc="left", fontsize=9.0, fontweight="bold", pad=8)
        clean_axes(ax)
        ax.tick_params(axis="y", length=0, pad=3)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.22, top=0.79, wspace=0.60)
    save(fig, "figure2_menstrual_transfer")


def fig3_capacity() -> None:
    arch = pd.read_csv(SOURCE / "architecture_results.csv")
    domain = pd.read_csv(SOURCE / "domain_transfer.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))

    # Panel a: matched-capacity validation loss.
    ax = axes[0]
    order = ["Latest-day shared MLP", "FemMHC dual path", "MMoE (8 experts)", "Shared GRU adapter"]
    labels = ["Latest-day MLP", "FemMHC", "MMoE", "Shared GRU"]
    sub = arch.set_index("model").loc[order].reset_index()
    y = np.arange(len(sub))[::-1]
    colors = [BASELINE, FEMMHC, AMBER, "#56B4E9"]
    for yi, mean, sd, color in zip(y, sub["validation_loss_mean"], sub["validation_loss_sd"], colors):
        ax.errorbar(mean, yi, xerr=sd, fmt="none", ecolor=color, elinewidth=2.0, capsize=3, zorder=2)
        ax.scatter(mean, yi, s=38, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        # Put the numeric value beyond the error-bar cap so it never sits on
        # top of the uncertainty mark.
        ax.text(mean + sd + 0.00045, yi, f"{mean:.4f}", ha="left", va="center", fontsize=7.2, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.758, 0.798)
    ax.set_xlabel("Validation loss (mean ± s.d.; lower is better)")
    ax.set_title("Capacity-matched temporal comparison", loc="left", fontsize=9.0, fontweight="bold", pad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    clean_axes(ax)
    ax.tick_params(axis="y", length=0, pad=3)

    # Panel b: domain transfer with labels placed according to bar direction.
    ax = axes[1]
    values = domain["relative_improvement_percent"].to_numpy()
    labels = ["Menstrual", "Affective", "HRV and sleep", "Pregnancy"]
    y = np.arange(len(values))[::-1]
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in values]
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.barh(y, values, color=colors, height=0.52, edgecolor="none")
    for yi, val in zip(y, values):
        if val >= 0:
            ax.text(val + 0.08, yi, f"+{val:.2f}%", ha="left", va="center", fontsize=7.4, fontweight="bold")
        else:
            ax.text(val - 0.08, yi, f"{val:.2f}%", ha="right", va="center", fontsize=7.4, fontweight="bold", color=NEGATIVE)
    ax.set_yticks(y, labels)
    ax.set_xlim(-2.75, 2.65)
    ax.set_xlabel("Continued vs. static adapter (% change)")
    ax.set_title("Domain-dependent continued pretraining", loc="left", fontsize=9.0, fontweight="bold", pad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    clean_axes(ax)
    ax.tick_params(axis="y", length=0, pad=3)

    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.22, top=0.79, wspace=0.68)
    save(fig, "figure3_capacity_and_transfer")


def fig4_reliability() -> None:
    cal = pd.read_csv(SOURCE / "calibration.csv")
    rob = pd.read_csv(SOURCE / "robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))

    # Panel a: paired calibration reductions.
    ax = axes[0]
    x = np.arange(len(cal))
    width = 0.32
    bars1 = ax.bar(x - width / 2, cal["brier_reduction_percent"], width, color=FEMMHC)
    bars2 = ax.bar(x + width / 2, cal["ece_reduction_percent"], width, color=POSITIVE)
    for bars, metric in ((bars1, "Brier"), (bars2, "ECE")):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.0,
                f"{metric}\n{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
                linespacing=1.0,
            )
    ax.set_xticks(x, ["24 h onset", "72 h onset"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Relative calibration-error reduction (%)")
    ax.set_title("Train-only probability calibration", loc="left", fontsize=9.0, fontweight="bold", pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    clean_axes(ax)

    # Panel b: robustness curves; labels are anchored at the final point with offsets.
    ax = axes[1]
    styles = {
        "Random deletion": (FEMMHC, "o"),
        "Contiguous deletion": (AMBER, "s"),
        "Latest-day deletion": (NEGATIVE, "D"),
    }
    xvals = np.arange(1, 4)
    offsets = {"Random deletion": -1.2, "Contiguous deletion": -0.2, "Latest-day deletion": -1.5}
    for name, group in rob.groupby("missingness_type", sort=False):
        color, marker = styles[name]
        yvals = group["relative_change_percent"].to_numpy()
        ax.plot(xvals, yvals, color=color, marker=marker, linewidth=1.8, markersize=4.5, label=name, zorder=3)
        final = float(yvals[-1])
        ax.text(3.0 + 0.07, final + offsets[name], f"{final:+.1f}%", ha="left", va="center", fontsize=7.3, fontweight="bold", color=color)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(xvals, ["Low", "Moderate", "High"])
    ax.set_ylim(-21.5, 5.5)
    ax.set_xlim(0.8, 3.85)
    ax.set_ylabel("Median relative performance change (%)")
    ax.set_title("Recent history is the critical failure mode", loc="left", fontsize=9.0, fontweight="bold", pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02), borderaxespad=0, handlelength=1.5)
    clean_axes(ax)

    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.78, wspace=0.52)
    save(fig, "figure4_calibration_robustness")


def main() -> None:
    configure()
    fig2_menstrual()
    fig3_capacity()
    fig4_reliability()
    print("Wrote Nature-style figures to", FIGURES)


if __name__ == "__main__":
    main()
