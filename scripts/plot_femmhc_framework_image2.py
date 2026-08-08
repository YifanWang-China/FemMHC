"""Draw the Image2/BioRender-style FemMHC framework figure.

The figure is intentionally built from vector primitives so every label and
shape remains editable in the SVG/PDF exports.  It follows the concept brief
used for the attempted Image2 generation, while keeping scientific text exact.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


NAVY = "#111827"
TEAL = "#22A7B5"
TEAL_DARK = "#137E8B"
TEAL_LIGHT = "#E4F6F8"
AMBER = "#E9AD3F"
AMBER_LIGHT = "#FFF5DD"
CORAL = "#EF635B"
CORAL_LIGHT = "#FDE9E7"
BLUE = "#4F7EDB"
BLUE_LIGHT = "#EAF0FC"
GRAY = "#667085"
MID = "#A9B2C2"
LIGHT = "#E5EAF1"
PALE = "#F8FAFC"
WHITE = "#FFFFFF"


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "femmhc_arxiv" / "figures"


def rounded(ax, x, y, w, h, *, fc=WHITE, ec=LIGHT, lw=1.0, radius=0.8, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, *, size=8, weight="normal", color=NAVY, ha="center", va="center", z=5, **kwargs):
    return ax.text(x, y, value, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kwargs)


def arrow(ax, x1, y1, x2, y2, *, color=NAVY, lw=1.4, scale=10, style="-|>", z=4, connection=None):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        connectionstyle=connection,
        shrinkA=1,
        shrinkB=1,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel(ax, x, y, w, h, letter, title, subtitle):
    rounded(ax, x, y, w, h, fc=WHITE, ec=LIGHT, lw=1.0, radius=1.0, z=0)
    ax.add_patch(Circle((x + 2.0, y + h - 2.2), 1.05, facecolor=NAVY, edgecolor="none", zorder=3))
    text(ax, x + 2.0, y + h - 2.2, letter, size=9, weight="bold", color=WHITE)
    text(ax, x + 3.7, y + h - 1.75, title, size=10.5, weight="bold", ha="left")
    text(ax, x + 3.7, y + h - 3.25, subtitle, size=6.7, color=GRAY, ha="left")


def person(ax, x, y, color, wearable=True, scale=1.0):
    ax.add_patch(Circle((x, y + 3.0 * scale), 0.82 * scale, facecolor=color, edgecolor="none", zorder=3))
    rounded(ax, x - 1.05 * scale, y - 0.2 * scale, 2.1 * scale, 2.7 * scale, fc=color, ec=color, radius=0.55 * scale, z=3)
    ax.plot([x - 0.7 * scale, x - 1.65 * scale], [y + 1.6 * scale, y + 0.35 * scale], color=color, lw=2.2 * scale, solid_capstyle="round", zorder=3)
    ax.plot([x + 0.7 * scale, x + 1.65 * scale], [y + 1.6 * scale, y + 0.35 * scale], color=color, lw=2.2 * scale, solid_capstyle="round", zorder=3)
    if wearable:
        ax.plot([x + 1.27 * scale, x + 1.56 * scale], [y + 0.72 * scale, y + 0.44 * scale], color=TEAL_DARK, lw=3.1 * scale, solid_capstyle="round", zorder=4)


def chip(ax, x, y, w, label, *, fc=PALE, ec=LIGHT, color=NAVY, size=6.5):
    rounded(ax, x, y, w, 2.1, fc=fc, ec=ec, lw=0.9, radius=0.55, z=2)
    text(ax, x + w / 2, y + 1.05, label, size=size, weight="semibold", color=color)


def timeline(ax, x, y, colors, missing=()):
    ax.plot([x, x + 15.5], [y, y], color=LIGHT, lw=1.2, zorder=1)
    for idx in range(7):
        xx = x + idx * 2.5
        if idx in missing:
            ax.add_patch(Circle((xx, y), 0.33, facecolor=WHITE, edgecolor=MID, linewidth=0.9, zorder=3))
        else:
            ax.add_patch(Circle((xx, y), 0.40, facecolor=colors[idx % len(colors)], edgecolor=WHITE, linewidth=0.7, zorder=3))


def lock(ax, x, y, scale=1.0):
    ax.add_patch(Rectangle((x, y), 1.35 * scale, 1.15 * scale, facecolor=BLUE, edgecolor="none", zorder=5))
    ax.add_patch(Arc((x + 0.675 * scale, y + 1.15 * scale), 0.9 * scale, 1.2 * scale, theta1=0, theta2=180, color=BLUE, lw=1.5 * scale, zorder=5))


def matrix_mask(ax, x, y):
    values = [
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ]
    for row, row_values in enumerate(values):
        for col, value in enumerate(row_values):
            ax.add_patch(
                Rectangle(
                    (x + col * 0.78, y - row * 0.78),
                    0.58,
                    0.58,
                    facecolor=TEAL if value else WHITE,
                    edgecolor=TEAL if value else MID,
                    linewidth=0.7,
                    zorder=4,
                )
            )


def draw_cohorts(ax):
    # Participant diversity and wearable context.
    person(ax, 5.0, 38.9, "#8D6E63", scale=0.78)
    person(ax, 10.1, 38.9, "#D6A47A", scale=0.78)
    person(ax, 15.2, 38.9, "#6C7A89", scale=0.78)
    person(ax, 20.1, 38.9, "#B7796F", scale=0.78)
    for x, label in zip([2.7, 7.8, 12.9, 17.7], ["mcPHASES", "DEPRESS", "inPHRsym", "Pregnancy"]):
        chip(ax, x, 35.0, 4.5, label, fc=PALE, ec=LIGHT, size=5.5)

    sensor_labels = [
        ("ACT", TEAL_LIGHT, TEAL_DARK),
        ("HR", CORAL_LIGHT, CORAL),
        ("HRV", BLUE_LIGHT, BLUE),
        ("SLEEP", AMBER_LIGHT, AMBER),
        ("TEMP", CORAL_LIGHT, CORAL),
        ("SpO2", BLUE_LIGHT, BLUE),
    ]
    for idx, (label, fc, ec) in enumerate(sensor_labels):
        row, col = divmod(idx, 3)
        chip(ax, 3.0 + col * 6.3, 29.9 - row * 2.7, 5.2, label, fc=fc, ec=ec, color=ec, size=6.2)

    text(ax, 3.0, 24.1, "participant-day timelines", size=6.5, weight="semibold", color=GRAY, ha="left")
    timeline(ax, 3.3, 21.8, [TEAL, BLUE, AMBER], missing=(3,))
    timeline(ax, 3.3, 19.1, [CORAL, TEAL, BLUE], missing=(1, 5))
    timeline(ax, 3.3, 16.4, [AMBER, TEAL, CORAL], missing=(4,))
    text(ax, 3.2, 14.6, "t−6", size=5.8, color=GRAY)
    text(ax, 18.2, 14.6, "t", size=5.8, color=GRAY)

    rounded(ax, 3.0, 6.6, 18.2, 6.2, fc=PALE, ec=LIGHT, radius=0.8)
    matrix_mask(ax, 4.2, 10.9)
    text(ax, 8.3, 10.7, "partial-label registry", size=6.7, weight="bold", ha="left")
    text(ax, 8.3, 8.8, "unobserved ≠ negative", size=6.2, color=CORAL, ha="left")
    text(ax, 8.3, 7.4, "missing days retained", size=6.2, color=GRAY, ha="left")


def draw_encoder(ax):
    # Semantic alignment and patch projection.
    text(ax, 27.0, 44.5, "semantic sensor alignment", size=6.7, weight="bold", ha="left")
    token_colors = [TEAL, CORAL, BLUE, AMBER, CORAL, BLUE]
    for idx, color in enumerate(token_colors):
        ax.add_patch(FancyBboxPatch((27.0 + idx * 2.05, 41.4), 1.55, 1.55, boxstyle="round,pad=0.02,rounding_size=0.25", facecolor=color, edgecolor="none", zorder=3))
    text(ax, 33.0, 40.0, "aligned daily patches", size=5.9, color=GRAY)
    arrow(ax, 39.6, 42.2, 42.1, 42.2, color=TEAL, lw=1.6)

    rounded(ax, 42.3, 38.9, 8.0, 6.6, fc=BLUE_LIGHT, ec=BLUE, lw=1.2, radius=0.75)
    text(ax, 46.3, 43.2, "LSM2", size=8.3, weight="bold", color=BLUE)
    text(ax, 46.3, 41.5, "patch projection", size=6.0, color=GRAY)
    lock(ax, 48.1, 39.6, scale=0.65)
    arrow(ax, 50.5, 42.2, 52.1, 42.2, color=BLUE, lw=1.6)

    # Frozen Transformer stack with adapters in the last two blocks.
    layer_y = [22.5, 26.2, 29.9, 33.6, 37.3]
    for idx, yy in enumerate(layer_y):
        active = idx >= 3
        rounded(ax, 36.1, yy, 14.6, 2.7, fc=BLUE_LIGHT if not active else WHITE, ec=BLUE, lw=1.0, radius=0.55)
        text(ax, 41.6, yy + 1.35, f"Transformer {idx + 1}", size=6.3, weight="semibold", color=BLUE, ha="center")
        lock(ax, 37.0, yy + 0.68, scale=0.48)
        if active:
            rounded(ax, 47.5, yy + 0.33, 2.5, 2.0, fc=TEAL_LIGHT, ec=TEAL, lw=1.0, radius=0.4, z=4)
            text(ax, 48.75, yy + 1.34, "LoRA", size=5.5, weight="bold", color=TEAL_DARK)

    rounded(ax, 28.0, 31.0, 6.6, 7.0, fc=BLUE_LIGHT, ec=BLUE, lw=1.0, radius=0.8)
    lock(ax, 30.6, 34.5, scale=0.85)
    text(ax, 31.3, 32.4, "99% frozen", size=6.8, weight="bold", color=BLUE)
    arrow(ax, 34.7, 34.5, 36.0, 34.5, color=BLUE)

    rounded(ax, 27.0, 23.0, 7.6, 5.9, fc=TEAL_LIGHT, ec=TEAL, lw=1.2, radius=0.8)
    text(ax, 30.8, 26.8, "female adapters", size=6.8, weight="bold", color=TEAL_DARK)
    text(ax, 30.8, 24.8, "239k · 1.11%", size=7.4, weight="bold", color=TEAL_DARK)
    arrow(ax, 34.7, 25.9, 36.0, 26.8, color=TEAL)

    # Causal history and soft-routed expert bank.
    text(ax, 27.0, 19.4, "causal longitudinal context", size=6.7, weight="bold", ha="left")
    for idx in range(6):
        color = TEAL if idx == 5 else LIGHT
        ec = TEAL if idx == 5 else MID
        rounded(ax, 27.0 + idx * 3.0, 15.9, 2.2, 2.2, fc=color if idx == 5 else WHITE, ec=ec, lw=0.8, radius=0.45)
        text(ax, 28.1 + idx * 3.0, 17.0, f"t−{5-idx}" if idx < 5 else "t", size=5.2, color=WHITE if idx == 5 else GRAY)
        if idx < 5:
            arrow(ax, 29.3 + idx * 3.0, 17.0, 29.9 + idx * 3.0, 17.0, color=MID, lw=0.8, scale=6)
    rounded(ax, 46.0, 15.2, 7.7, 3.7, fc=AMBER_LIGHT, ec=AMBER, lw=1.1, radius=0.65)
    text(ax, 49.85, 17.1, "causal memory", size=6.4, weight="bold", color="#A46B09")
    arrow(ax, 44.4, 17.0, 45.9, 17.0, color=AMBER, lw=1.4)

    rounded(ax, 27.0, 7.0, 26.7, 6.2, fc=PALE, ec=LIGHT, lw=0.9, radius=0.8)
    text(ax, 28.0, 11.7, "soft-routed physiological experts", size=6.4, weight="bold", ha="left")
    expert_specs = [("E1", TEAL_LIGHT, TEAL), ("E2", AMBER_LIGHT, AMBER), ("E3", CORAL_LIGHT, CORAL)]
    for idx, (label, fc, ec) in enumerate(expert_specs):
        rounded(ax, 28.1 + idx * 5.3, 8.2, 4.2, 2.5, fc=fc, ec=ec, lw=1.0, radius=0.5)
        text(ax, 30.2 + idx * 5.3, 9.45, label, size=6.8, weight="bold", color=ec)
    rounded(ax, 45.0, 8.2, 6.8, 2.5, fc=WHITE, ec=TEAL, lw=1.1, radius=0.5)
    text(ax, 48.4, 9.45, "daily zᵢ,ₜ", size=6.6, weight="bold", color=TEAL_DARK)
    for idx, (_, _, ec) in enumerate(expert_specs):
        arrow(ax, 32.5 + idx * 5.3, 9.45, 44.8, 9.45, color=ec, lw=0.8, scale=6, connection="arc3,rad=0.10")


def task_card(ax, x, y, w, title, detail, color, light):
    rounded(ax, x, y, w, 5.0, fc=WHITE, ec=color, lw=1.05, radius=0.65)
    ax.add_patch(Rectangle((x, y), 0.85, 5.0, facecolor=color, edgecolor="none", zorder=3))
    text(ax, x + 1.5, y + 3.35, title, size=6.6, weight="bold", color=NAVY, ha="left")
    text(ax, x + 1.5, y + 1.65, detail, size=5.6, color=GRAY, ha="left")


def draw_tasks(ax):
    rounded(ax, 59.1, 38.9, 4.8, 5.4, fc=TEAL_LIGHT, ec=TEAL, lw=1.2, radius=0.8)
    text(ax, 61.5, 42.0, "shared", size=6.2, weight="bold", color=TEAL_DARK)
    text(ax, 61.5, 40.2, "zᵢ,ₜ", size=8.2, weight="bold", color=TEAL_DARK)
    arrow(ax, 61.5, 38.8, 61.5, 35.8, color=TEAL, lw=1.6)
    ax.plot([61.5, 61.5], [12.1, 35.8], color=TEAL, lw=1.5, zorder=2)

    cards = [
        ("Menstrual", "phase · onset 24/72 h", TEAL, TEAL_LIGHT),
        ("Symptoms", "cramps · mood · sleep", CORAL, CORAL_LIGHT),
        ("Autonomic", "HRV · sleep state", BLUE, BLUE_LIGHT),
        ("Pregnancy", "gestational activity", AMBER, AMBER_LIGHT),
        ("General health", "activity · cardio retention", TEAL_DARK, TEAL_LIGHT),
    ]
    y_values = [33.0, 27.1, 21.2, 15.3, 9.4]
    for (title_value, detail, color, light), yy in zip(cards, y_values):
        arrow(ax, 61.5, yy + 2.5, 64.1, yy + 2.5, color=color, lw=1.0, scale=7)
        task_card(ax, 64.2, yy, 12.4, title_value, detail, color, light)

    rounded(ax, 59.1, 5.6, 17.5, 2.8, fc=PALE, ec=LIGHT, lw=0.9, radius=0.55)
    matrix_mask(ax, 60.0, 7.35)
    text(ax, 64.2, 7.0, "masked multitask loss", size=6.0, weight="bold", ha="left")
    text(ax, 64.2, 6.1, "only observed labels", size=5.3, color=GRAY, ha="left")

    # Coherent nested risks.
    rounded(ax, 66.0, 40.2, 10.6, 3.3, fc=PALE, ec=LIGHT, lw=0.8, radius=0.55)
    text(ax, 67.0, 42.5, "calibrated onset risk", size=5.6, weight="bold", ha="left")
    ax.add_patch(Rectangle((67.0, 41.1), 7.6, 0.45, facecolor=LIGHT, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((67.0, 41.1), 4.8, 0.45, facecolor=AMBER, edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((67.0, 40.5), 7.6, 0.35, facecolor=TEAL, edgecolor="none", zorder=4))
    text(ax, 75.2, 41.35, "24 h", size=5.0, color=AMBER, ha="left")
    text(ax, 75.2, 40.65, "72 h", size=5.0, color=TEAL_DARK, ha="left")


def eval_row(ax, y, icon_color, title, subtitle):
    ax.add_patch(Circle((82.5, y + 1.7), 1.25, facecolor=icon_color, edgecolor="none", zorder=3))
    text(ax, 82.5, y + 1.7, "✓", size=8, weight="bold", color=WHITE)
    text(ax, 84.4, y + 2.25, title, size=6.5, weight="bold", ha="left")
    text(ax, 84.4, y + 0.95, subtitle, size=5.4, color=GRAY, ha="left")


def draw_evaluation(ax):
    eval_row(ax, 38.1, TEAL, "Participant-disjoint", "train · validation · test")
    eval_row(ax, 32.2, BLUE, "Nested LOSO", "42 held-out participants")
    eval_row(ax, 26.3, AMBER, "Capacity matched", "MLP · GRU · MMoE")
    eval_row(ax, 20.4, TEAL_DARK, "Probability quality", "calibration · nesting")
    eval_row(ax, 14.5, CORAL, "Missing-history audit", "random · block · recent")

    rounded(ax, 81.2, 6.2, 15.8, 6.5, fc=PALE, ec=LIGHT, lw=1.0, radius=0.9)
    text(ax, 82.2, 11.1, "SELECTIVE TRANSFER", size=7.2, weight="bold", color=NAVY, ha="left")
    chip(ax, 82.2, 7.8, 4.0, "cycle ↑", fc=TEAL_LIGHT, ec=TEAL, color=TEAL_DARK, size=5.7)
    chip(ax, 86.6, 7.8, 4.0, "HRV ↑", fc=TEAL_LIGHT, ec=TEAL, color=TEAL_DARK, size=5.7)
    chip(ax, 91.0, 7.8, 4.8, "pregnancy ↓", fc=CORAL_LIGHT, ec=CORAL, color=CORAL, size=5.5)
    text(ax, 89.1, 6.8, "not universal superiority", size=5.6, weight="semibold", color=CORAL)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "text.color": NAVY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.axis("off")

    text(ax, 2.0, 54.2, "FemMHC", size=20, weight="bold", color=NAVY, ha="left")
    text(ax, 14.2, 54.2, "Parameter-efficient specialization of OpenMHC for longitudinal female wearable health", size=11.5, weight="semibold", color=GRAY, ha="left")
    ax.plot([2.0, 98.0], [51.7, 51.7], color=LIGHT, lw=1.2)

    panel(ax, 1.5, 2.7, 22.1, 47.5, "A", "Female wearable cohorts", "heterogeneous sensors · partial labels · missing days")
    panel(ax, 24.8, 2.7, 31.1, 47.5, "B", "OpenMHC specialization", "frozen foundation encoder + 1.11% trainable parameters")
    panel(ax, 57.1, 2.7, 21.5, 47.5, "C", "Causal task families", "one representation · multiple female-health objectives")
    panel(ax, 79.8, 2.7, 18.7, 47.5, "D", "Evidence boundary", "strict evaluation of gains, failures, and reliability")

    arrow(ax, 23.8, 26.5, 24.6, 26.5, color=TEAL, lw=2.0, scale=12)
    arrow(ax, 56.1, 26.5, 56.9, 26.5, color=TEAL, lw=2.0, scale=12)
    arrow(ax, 78.8, 26.5, 79.6, 26.5, color=TEAL, lw=2.0, scale=12)

    draw_cohorts(ax)
    draw_encoder(ax)
    draw_tasks(ax)
    draw_evaluation(ax)

    text(ax, 50.0, 1.1, "Only past observations enter each prediction · absent labels are masked · transfer is evaluated per task and cohort", size=6.5, color=GRAY)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(OUT / f"figure1_femmhc_framework_image2.{suffix}", bbox_inches="tight", facecolor=WHITE, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
