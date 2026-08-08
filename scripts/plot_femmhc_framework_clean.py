"""Create a deliberately minimal FemMHC manuscript architecture figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


NAVY = "#111827"
GRAY = "#667085"
MID = "#AEB7C6"
LIGHT = "#E5EAF1"
PALE = "#F8FAFC"
WHITE = "#FFFFFF"
BLUE = "#4F7EDB"
BLUE_LIGHT = "#EAF0FC"
TEAL = "#22A7B5"
TEAL_DARK = "#137E8B"
TEAL_LIGHT = "#E4F6F8"
AMBER = "#E9AD3F"
AMBER_LIGHT = "#FFF5DD"
CORAL = "#EF635B"
CORAL_LIGHT = "#FDE9E7"


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "femmhc_arxiv" / "figures"


def rounded(ax, x, y, w, h, fc=WHITE, ec=LIGHT, lw=1.0, r=0.8, z=1):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


def label(ax, x, y, s, size=8, weight="normal", color=NAVY, ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z)


def arrow(ax, x1, y1, x2, y2, color=TEAL, lw=1.8, scale=11, z=4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=scale,
        linewidth=lw, color=color, shrinkA=1, shrinkB=1, zorder=z,
    ))


def panel_heading(ax, x, letter, title, subtitle):
    ax.add_patch(Circle((x + 1.25, 34.8), 0.95, facecolor=NAVY, edgecolor="none", zorder=4))
    label(ax, x + 1.25, 34.8, letter, 8, "bold", WHITE)
    label(ax, x + 2.8, 35.2, title, 10.2, "bold", NAVY, "left")
    label(ax, x + 2.8, 33.5, subtitle, 6.5, "normal", GRAY, "left")


def person(ax, x, y):
    ax.add_patch(Circle((x, y + 3.3), 0.95, facecolor="#B7796F", edgecolor="none", zorder=4))
    rounded(ax, x - 1.25, y - 0.2, 2.5, 3.0, fc="#B7796F", ec="#B7796F", r=0.65, z=4)
    ax.plot([x - 0.8, x - 1.9], [y + 1.7, y + 0.4], color="#B7796F", lw=2.8, solid_capstyle="round", zorder=4)
    ax.plot([x + 0.8, x + 1.9], [y + 1.7, y + 0.4], color="#B7796F", lw=2.8, solid_capstyle="round", zorder=4)
    ax.plot([x + 1.48, x + 1.78], [y + 0.82, y + 0.55], color=TEAL_DARK, lw=3.4, solid_capstyle="round", zorder=5)


def lock(ax, x, y, scale=1.0):
    ax.add_patch(Rectangle((x, y), 1.35 * scale, 1.15 * scale, facecolor=BLUE, edgecolor="none", zorder=5))
    ax.add_patch(Arc((x + 0.675 * scale, y + 1.1 * scale), 0.9 * scale, 1.1 * scale, theta1=0, theta2=180, color=BLUE, lw=1.5 * scale, zorder=5))


def sensor_pill(ax, x, y, s, color, light):
    rounded(ax, x, y, 4.4, 2.15, fc=light, ec=color, lw=0.9, r=0.55, z=2)
    label(ax, x + 2.2, y + 1.08, s, 6.1, "bold", color)


def input_panel(ax):
    person(ax, 8.1, 23.3)
    label(ax, 8.1, 21.6, "longitudinal participant", 6.4, "semibold", GRAY)

    specs = [
        ("ACT", TEAL_DARK, TEAL_LIGHT),
        ("HR", CORAL, CORAL_LIGHT),
        ("HRV", BLUE, BLUE_LIGHT),
        ("SLEEP", AMBER, AMBER_LIGHT),
        ("TEMP", CORAL, CORAL_LIGHT),
        ("SpO₂", BLUE, BLUE_LIGHT),
    ]
    for i, (name, color, light) in enumerate(specs):
        row, col = divmod(i, 3)
        sensor_pill(ax, 2.1 + col * 5.25, 15.5 - row * 2.8, name, color, light)

    label(ax, 2.2, 9.3, "past days", 6.2, "bold", GRAY, "left")
    ax.plot([2.2, 16.2], [7.0, 7.0], color=LIGHT, lw=1.4, zorder=1)
    colors = [TEAL, BLUE, AMBER, WHITE, CORAL, TEAL, BLUE]
    for i, color in enumerate(colors):
        edge = MID if color == WHITE else WHITE
        ax.add_patch(Circle((2.2 + i * 2.33, 7.0), 0.42, facecolor=color, edgecolor=edge, linewidth=0.9, zorder=3))
    label(ax, 2.2, 5.5, "t−6", 5.7, "normal", GRAY)
    label(ax, 16.2, 5.5, "t", 5.7, "normal", GRAY)
    rounded(ax, 2.1, 1.8, 14.3, 2.8, fc=PALE, ec=LIGHT, lw=0.8, r=0.5)
    label(ax, 9.25, 3.55, "multi-cohort · partial labels", 5.5, "semibold", GRAY)
    label(ax, 9.25, 2.65, "missing days retained", 5.3, "normal", GRAY)


def backbone_panel(ax):
    label(ax, 23.7, 29.8, "semantic alignment", 6.4, "bold", GRAY, "left")
    token_colors = [TEAL, CORAL, BLUE, AMBER, TEAL]
    for i, color in enumerate(token_colors):
        rounded(ax, 23.7 + i * 2.35, 26.9, 1.75, 1.75, fc=color, ec=color, lw=0, r=0.35, z=3)
    arrow(ax, 35.3, 27.8, 37.0, 27.8, TEAL, 1.4, 8)

    rounded(ax, 37.2, 25.5, 7.5, 4.6, fc=BLUE_LIGHT, ec=BLUE, lw=1.1, r=0.75)
    label(ax, 40.95, 28.4, "LSM2", 8.0, "bold", BLUE)
    label(ax, 40.95, 26.8, "patch projection", 5.8, "normal", GRAY)
    arrow(ax, 40.95, 25.35, 40.95, 24.1, BLUE, 1.4, 8)

    layer_y = [5.9, 9.8, 13.7, 17.6, 21.5]
    for i, y in enumerate(layer_y):
        rounded(ax, 29.0, y, 16.5, 2.8, fc=BLUE_LIGHT, ec=BLUE, lw=1.0, r=0.55)
        lock(ax, 30.2, y + 0.72, 0.52)
        label(ax, 37.2, y + 1.4, f"Transformer block {i + 1}", 6.2, "semibold", BLUE)
    rounded(ax, 23.7, 7.7, 4.3, 14.4, fc=BLUE_LIGHT, ec=BLUE, lw=1.0, r=0.75)
    lock(ax, 25.2, 15.4, 0.9)
    label(ax, 25.85, 12.4, "frozen", 6.5, "bold", BLUE)


def specialization_panel(ax):
    rounded(ax, 51.0, 24.0, 19.4, 6.8, fc=TEAL_LIGHT, ec=TEAL, lw=1.3, r=0.9)
    label(ax, 60.7, 28.4, "Female residual adapters", 8.2, "bold", TEAL_DARK)
    label(ax, 60.7, 26.0, "239k trainable parameters · 1.11%", 6.5, "semibold", TEAL_DARK)
    arrow(ax, 45.8, 23.0, 50.8, 27.2, TEAL, 1.6, 10)
    arrow(ax, 69.0, 24.0, 70.0, 20.1, TEAL, 1.2, 8)

    label(ax, 51.0, 20.8, "causal personal history", 6.4, "bold", GRAY, "left")
    for i in range(6):
        fc = TEAL if i == 5 else WHITE
        ec = TEAL if i == 5 else MID
        rounded(ax, 51.0 + i * 3.1, 17.0, 2.35, 2.45, fc=fc, ec=ec, lw=0.9, r=0.5)
        label(ax, 52.18 + i * 3.1, 18.22, f"t−{5-i}" if i < 5 else "t", 5.4, "normal", WHITE if i == 5 else GRAY)
        if i < 5:
            arrow(ax, 53.5 + i * 3.1, 18.22, 53.9 + i * 3.1, 18.22, MID, 0.8, 5)

    rounded(ax, 51.0, 8.0, 19.4, 6.0, fc=PALE, ec=LIGHT, lw=0.9, r=0.75)
    label(ax, 52.0, 12.7, "soft-routed physiological states", 6.2, "bold", NAVY, "left")
    for i, (name, color, light) in enumerate([("E1", TEAL, TEAL_LIGHT), ("E2", AMBER, AMBER_LIGHT), ("E3", CORAL, CORAL_LIGHT)]):
        rounded(ax, 52.0 + i * 4.35, 9.1, 3.6, 2.4, fc=light, ec=color, lw=1.0, r=0.5)
        label(ax, 53.8 + i * 4.35, 10.3, name, 6.2, "bold", color)
    arrow(ax, 64.5, 10.3, 66.3, 10.3, TEAL, 1.2, 7)
    rounded(ax, 66.5, 9.1, 2.8, 2.4, fc=TEAL, ec=TEAL, lw=0, r=0.5)
    label(ax, 67.9, 10.3, "zᵢ,ₜ", 6.5, "bold", WHITE)

    arrow(ax, 68.1, 11.7, 70.0, 17.1, TEAL, 1.2, 8)
    ax.add_patch(Circle((70.2, 18.8), 1.18, facecolor=TEAL, edgecolor=WHITE, linewidth=1.0, zorder=5))
    label(ax, 70.2, 18.8, "zᵢ,ₜ", 6.4, "bold", WHITE)

    rounded(ax, 53.3, 2.5, 14.8, 2.4, fc=AMBER_LIGHT, ec=AMBER, lw=1.0, r=0.55)
    label(ax, 60.7, 3.7, "masked causal multitask learning", 5.6, "bold", "#A46B09")


def task_card(ax, y, title, detail, color, light):
    rounded(ax, 77.0, y, 18.9, 7.3, fc=WHITE, ec=color, lw=1.15, r=0.8)
    ax.add_patch(Rectangle((77.0, y), 0.9, 7.3, facecolor=color, edgecolor="none", zorder=3))
    label(ax, 79.0, y + 5.0, title, 7.5, "bold", NAVY, "left")
    label(ax, 79.0, y + 2.6, detail, 5.9, "normal", GRAY, "left")


def outputs_panel(ax):
    task_card(ax, 23.0, "Cycle & menstruation", "phase · onset risk · cramps", TEAL, TEAL_LIGHT)
    task_card(ax, 14.1, "Daily wellbeing", "mood · sleep · fatigue", CORAL, CORAL_LIGHT)
    task_card(ax, 5.2, "Physiology & life stage", "HRV/sleep · pregnancy · activity", BLUE, BLUE_LIGHT)
    arrow(ax, 71.3, 18.8, 76.7, 26.7, TEAL, 1.4, 9)
    arrow(ax, 71.3, 18.8, 76.7, 17.8, CORAL, 1.4, 9)
    arrow(ax, 71.3, 18.8, 76.7, 8.9, BLUE, 1.4, 9)
    rounded(ax, 79.1, 1.8, 14.7, 2.3, fc=PALE, ec=LIGHT, lw=0.8, r=0.5)
    label(ax, 86.45, 2.95, "probability · score · class", 5.8, "bold", GRAY)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(14.5, 5.9), facecolor=WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    for x, w in [(0.8, 18.0), (20.5, 27.5), (49.0, 23.0), (74.0, 24.8)]:
        rounded(ax, x, 0.6, w, 36.5, fc=WHITE, ec=LIGHT, lw=1.0, r=0.9, z=0)
    panel_heading(ax, 0.8, "a", "Wearable signals", "multimodal longitudinal observations")
    panel_heading(ax, 20.5, "b", "OpenMHC backbone", "general wearable representation")
    panel_heading(ax, 49.0, "c", "Female specialization", "small adapters + causal personal context")
    panel_heading(ax, 74.0, "d", "Female-health tasks", "shared representation, task-specific outputs")

    arrow(ax, 18.95, 18.8, 20.25, 18.8, TEAL, 2.0, 11)
    arrow(ax, 48.15, 18.8, 48.75, 18.8, TEAL, 2.0, 11)

    input_panel(ax)
    backbone_panel(ax)
    specialization_panel(ax)
    outputs_panel(ax)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(OUT / f"figure1_femmhc_framework_clean.{suffix}", bbox_inches="tight", facecolor=WHITE, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
