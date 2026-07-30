from __future__ import annotations

from pathlib import Path
import hashlib
import math
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent

# Palette adapted from the user's examples. No matplotlib default colors are used.
COL = {
    "repress": "#225B91",
    "repress_mid": "#4884AF",
    "repress_light": "#B3CEDE",
    "shape": "#FC9262",
    "shape_light": "#FED8C6",
    "afhc": "#EAA5C3",
    "afhc_light": "#F3D5E2",
    "static": "#44935B",
    "static_light": "#B3D7AE",
    "initial": "#8A929B",
    "initial_light": "#D8DCE0",
    "oracle": "#D5A43C",
    "oracle_light": "#FEDE8E",
    "ink": "#20262D",
    "muted": "#66727D",
    "grid": "#DCE2E7",
    "panel": "#F7F9FB",
    "teal": "#4F9DA6",
    "purple": "#7D6AA5",
    "red": "#B85450",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Nimbus Roman No9 L",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "font.size": 8.8,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.8,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.unicode_minus": False,
})


def lighten(color: str, amount: float = 0.65) -> tuple[float, float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    mixed = rgb + (1.0 - rgb) * amount
    return (*mixed, 1.0)


def darken(color: str, amount: float = 0.18) -> tuple[float, float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - amount)
    return (*mixed, 1.0)


def style_axis(ax, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.7, color=COL["muted"])
    if grid:
        ax.grid(axis=grid, color=COL["grid"], linewidth=0.65, alpha=0.8)
        ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.085,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=COL["ink"],
    )


def save_figure(fig, stem: str, dpi: int = 420) -> None:
    fig.savefig(
        ROOT / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    fig.savefig(
        ROOT / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)


def gradient_bar(
    ax,
    x: float,
    height: float,
    width: float,
    color: str,
    bottom: float = 0.0,
    zorder: int = 3,
    edge_lw: float = 0.65,
) -> Rectangle:
    """Draw a restrained vertical gradient clipped to a bar rectangle."""
    low = min(bottom, bottom + height)
    high = max(bottom, bottom + height)
    if high - low < 1e-10:
        rect = Rectangle(
            (x - width / 2.0, bottom),
            width,
            0.0,
            facecolor="none",
            edgecolor=darken(color, 0.12),
            linewidth=edge_lw,
            zorder=zorder + 1,
        )
        ax.add_patch(rect)
        return rect
    rect = Rectangle(
        (x - width / 2.0, low),
        width,
        max(high - low, 1e-12),
        facecolor="none",
        edgecolor=darken(color, 0.12),
        linewidth=edge_lw,
        zorder=zorder + 1,
    )
    ax.add_patch(rect)
    n = 256
    top_rgba = np.asarray(to_rgba(color))
    bottom_rgba = np.asarray(lighten(color, 0.82))
    colors = np.linspace(bottom_rgba, top_rgba, n).reshape(n, 1, 4)
    if height < 0:
        colors = colors[::-1]
    image = ax.imshow(
        colors,
        extent=(x - width / 2.0, x + width / 2.0, low, high),
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=zorder,
        clip_on=True,
    )
    image.set_clip_path(rect)
    return rect


def add_bar_value(ax, x: float, value: float, fmt: str = "{:.2f}", offset: float = 0.012) -> None:
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    if value >= 0:
        y = value + offset * span
        va = "bottom"
    else:
        y = value - offset * span
        va = "top"
    ax.text(
        x,
        y,
        fmt.format(value),
        ha="center",
        va=va,
        fontsize=7.2,
        color=COL["ink"],
    )


# -----------------------------------------------------------------------------
# Figure 1: architecture
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.25, 3.75))
ax.set_xlim(0, 12.6)
ax.set_ylim(0, 6.2)
ax.axis("off")


def architecture_box(x, y, w, h, title, lines, face, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.045,rounding_size=0.12",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.18,
        y + h - 0.28,
        title,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=COL["ink"],
    )
    ax.text(
        x + 0.18,
        y + h - 0.72,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=7.55,
        color="#34424D",
        linespacing=1.25,
    )


def architecture_arrow(x1, y1, x2, y2, text=None, rad=0.0):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.05,
        color="#56636D",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if text:
        ax.text(
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0 + 0.10,
            text,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#56636D",
        )

architecture_box(
    0.25,
    3.55,
    2.15,
    1.75,
    "Observed workload",
    ["Point and range queries", "Aggregates and thresholds", "Past events only"],
    lighten(COL["repress"], 0.88),
    COL["repress_mid"],
)
architecture_box(
    3.05,
    3.42,
    2.55,
    2.0,
    "Query-shape monitor",
    ["100-query short window", "300-query long window", "Measured cost profiles"],
    lighten(COL["static"], 0.88),
    COL["static"],
)
architecture_box(
    6.25,
    3.42,
    2.75,
    2.0,
    "Reconfiguration policy",
    ["Window agreement", "Three confirmations", "Gain and break-even tests", "Cooldown and write budget"],
    lighten(COL["shape"], 0.86),
    COL["shape"],
)
architecture_box(
    9.65,
    3.42,
    2.65,
    2.0,
    "Single-active store",
    ["Versioned payloads", "SQLite WAL catalog", "Atomic publication", "Bounded migration log"],
    lighten(COL["purple"], 0.87),
    COL["purple"],
)

architecture_box(
    3.05,
    0.48,
    2.55,
    1.65,
    "Representation service",
    ["Whole LZ4 and Zstandard", "LZ4 blocks: 64, 256, 1024", "Lossless query execution"],
    "#F6F8FA",
    "#87939C",
)
architecture_box(
    6.25,
    0.48,
    2.75,
    1.65,
    "Migration transaction",
    ["Intent and staging", "Encode and validate", "Publish and commit", "Cleanup or rollback"],
    "#F6F8FA",
    "#87939C",
)
architecture_box(
    9.65,
    0.48,
    2.65,
    1.65,
    "Dependability controls",
    ["Checksum verification", "Targeted startup recovery", "Offline fsck", "Thread-local codec state"],
    "#F6F8FA",
    "#87939C",
)

architecture_arrow(2.40, 4.42, 3.05, 4.42)
architecture_arrow(5.60, 4.42, 6.25, 4.42)
architecture_arrow(9.00, 4.42, 9.65, 4.42)
architecture_arrow(4.32, 2.13, 4.32, 3.42, "cost and codecs")
architecture_arrow(7.62, 2.13, 7.62, 3.42, "transaction")
architecture_arrow(10.98, 2.13, 10.98, 3.42, "recovery")
architecture_arrow(10.25, 3.42, 8.98, 3.42, "commit feedback", rad=0.24)

ax.text(
    0.25,
    5.92,
    "Runtime control path",
    fontsize=10.5,
    fontweight="bold",
    color=COL["ink"],
)
ax.plot([0.25, 12.3], [5.72, 5.72], color="#CAD1D8", lw=0.8)
save_figure(fig, "Figure_1_architecture")


# -----------------------------------------------------------------------------
# Figure 2: continuous trace construction
# -----------------------------------------------------------------------------
def stable_seed(*parts) -> int:
    token = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little")


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def category_probabilities(trace_name: str, n_events: int = 640000) -> np.ndarray:
    seed = 20260728
    rng = np.random.default_rng(stable_seed(seed, trace_name, "category"))
    x = np.linspace(0.0, 1.0, n_events, dtype=np.float64)
    if trace_name == "R1_A_gradual_periodic":
        sigmoid = 1.0 / (1.0 + np.exp(-(x - 0.50) / 0.12))
        point = np.clip(
            0.05 + 0.65 * sigmoid + 0.045 * np.sin(2.0 * np.pi * 9.0 * x),
            0.03,
            0.76,
        )
        aggregate = np.clip(
            0.16 + 0.035 * np.sin(2.0 * np.pi * 4.0 * x + 1.1),
            0.08,
            0.25,
        )
        scan = np.maximum(1.0 - point - aggregate, 0.08)
        p = np.stack([point, scan, aggregate], axis=1)
        return p / p.sum(axis=1, keepdims=True)

    step = 100
    knots = math.ceil(n_events / step) + 1
    logits = np.zeros((knots, 3), dtype=np.float64)
    logits[0] = np.asarray([-0.4, 0.9, -0.2])
    center = np.asarray([-0.15, 0.50, -0.20])
    for i in range(1, knots):
        logits[i] = 0.985 * logits[i - 1] + 0.015 * center + rng.normal(0.0, 0.075, 3)
        logits[i, 0] += 0.08 * math.sin(2.0 * math.pi * i / 170.0)
        logits[i, 2] += 0.06 * math.sin(2.0 * math.pi * i / 113.0 + 0.7)

    if trace_name == "R1_C_noisy_bursts":
        positions = np.arange(knots, dtype=np.float64)
        for _ in range(24):
            center_i = int(rng.integers(10, knots - 10))
            width = float(rng.uniform(2.0, 12.0))
            category = int(rng.integers(0, 3))
            amplitude = float(rng.uniform(0.65, 1.60))
            logits[:, category] += amplitude * np.exp(
                -0.5 * ((positions - center_i) / width) ** 2
            )
    knot_positions = np.arange(knots) * step
    event_positions = np.arange(n_events)
    interpolated = np.column_stack(
        [np.interp(event_positions, knot_positions, logits[:, j]) for j in range(3)]
    )
    return softmax(interpolated)

trace_order = [
    "R1_A_gradual_periodic",
    "R1_B_mean_reverting",
    "R1_C_noisy_bursts",
]
trace_titles = [
    "Gradual and periodic",
    "Mean reverting",
    "Noisy bursts",
]
fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), sharey=True)
for index, (ax, trace, title) in enumerate(zip(axes, trace_order, trace_titles)):
    probabilities = category_probabilities(trace)
    block = 1000
    usable = (len(probabilities) // block) * block
    coarse = probabilities[:usable].reshape(-1, block, 3).mean(axis=1)
    coarse = pd.DataFrame(coarse).rolling(9, center=True, min_periods=1).mean().to_numpy()
    x = np.linspace(0.0, 1.0, len(coarse))
    ax.plot(x, coarse[:, 0], color=COL["repress"], label="Point")
    ax.plot(x, coarse[:, 1], color=COL["shape"], label="Scan")
    ax.plot(x, coarse[:, 2], color=COL["static"], label="Aggregate")
    ax.fill_between(x, 0, coarse[:, 0], color=COL["repress_light"], alpha=0.16)
    ax.set_title(title, pad=5, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Normalized query time")
    style_axis(ax)
    panel_label(ax, f"({chr(97 + index)})")
axes[0].set_ylabel("Latent query-category probability")
axes[-1].legend(
    handles=[
        Line2D([0], [0], color=COL["repress"], lw=2, label="Point"),
        Line2D([0], [0], color=COL["shape"], lw=2, label="Scan"),
        Line2D([0], [0], color=COL["static"], lw=2, label="Aggregate"),
    ],
    frameon=False,
    loc="upper right",
)
fig.tight_layout(w_pad=1.2)
save_figure(fig, "Figure_2_continuous_traces")


# -----------------------------------------------------------------------------
# Figure 3: strong baseline trade-off
# -----------------------------------------------------------------------------
s1 = pd.read_csv(ROOT / "s1_holdout_global_comparison_summary.csv")
trace_keys = ["shape_only", "frequency_only", "joint_orthogonal", "brief_burst", "stable_scan"]
trace_labels = ["Shape", "Frequency", "Joint", "Brief", "Stable"]
fig = plt.figure(figsize=(7.25, 5.45))
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
for panel_index, dataset in enumerate(["PEMS08", "BDG2"]):
    ax = fig.add_subplot(gs[0, panel_index])
    x = np.arange(len(trace_keys), dtype=float)
    widths = 0.32
    for size_index, (size, color) in enumerate(
        [(8192, COL["repress_light"]), (16384, COL["repress"])]
    ):
        values = []
        for trace in trace_keys:
            row = s1[
                (s1.dataset == dataset)
                & (s1.trace == trace)
                & (s1.chunk_size == size)
            ].iloc[0]
            values.append(100.0 * (row.geometric_mean_ratio_repressdb_over_shape_ewma - 1.0))
        offsets = x + (size_index - 0.5) * widths
        for xpos, value in zip(offsets, values):
            gradient_bar(ax, xpos, value, widths * 0.86, color)
        for xpos, value in zip(offsets, values):
            add_bar_value(ax, xpos, value, "{:.1f}", offset=0.018)
    ax.axhline(0, color="#69747D", lw=0.9)
    ax.set_xticks(x, trace_labels, rotation=18)
    ax.set_title(dataset, fontweight="bold")
    ax.set_ylim(-4.0, 39.5)
    ax.set_ylabel("Cost relative to Shape-EWMA-BE (%)")
    ax.set_xlim(-0.5, 4.5)
    style_axis(ax)
    panel_label(ax, f"({chr(97 + panel_index)})")
    if panel_index == 1:
        ax.legend(
            handles=[
                Rectangle((0, 0), 1, 1, facecolor=COL["repress_light"], edgecolor=darken(COL["repress_light"]), label="8K"),
                Rectangle((0, 0), 1, 1, facecolor=COL["repress"], edgecolor=darken(COL["repress"]), label="16K"),
            ],
            frameon=False,
            ncol=2,
            loc="upper left",
        )

holdout_counts = pd.DataFrame(
    {
        "dataset": ["PEMS08", "PEMS08", "BDG2", "BDG2"],
        "policy": ["Shape-EWMA-BE", "RePressDB", "Shape-EWMA-BE", "RePressDB"],
        "migrations": [532, 342, 517, 309],
        "rewrite_mb": [18.314880, 12.580190, 19.137950, 11.852998],
    }
)
for panel_index, metric in enumerate(["migrations", "rewrite_mb"], start=2):
    ax = fig.add_subplot(gs[1, panel_index - 2])
    x = np.arange(2, dtype=float)
    width = 0.32
    for policy_index, (policy, color) in enumerate(
        [("Shape-EWMA-BE", COL["shape"]), ("RePressDB", COL["repress"])]
    ):
        sub = holdout_counts[holdout_counts.policy == policy]
        values = [float(sub[sub.dataset == ds][metric].iloc[0]) for ds in ["PEMS08", "BDG2"]]
        offsets = x + (policy_index - 0.5) * width
        for xpos, value in zip(offsets, values):
            gradient_bar(ax, xpos, value, width * 0.86, color)
            add_bar_value(ax, xpos, value, "{:.0f}" if metric == "migrations" else "{:.1f}")
    ax.set_xticks(x, ["PEMS08", "BDG2"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Migration count" if metric == "migrations" else "Rewritten payload (MB)")
    ax.set_ylim(0, 590 if metric == "migrations" else 21.5)
    style_axis(ax)
    panel_label(ax, f"({chr(97 + panel_index)})")
fig.legend(
    handles=[
        Rectangle((0, 0), 1, 1, facecolor=COL["shape"], edgecolor=darken(COL["shape"]), label="Shape-EWMA-BE"),
        Rectangle((0, 0), 1, 1, facecolor=COL["repress"], edgecolor=darken(COL["repress"]), label="RePressDB"),
    ],
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
)
fig.subplots_adjust(
    left=0.105, right=0.985, top=0.955, bottom=0.145,
    hspace=0.53, wspace=0.27,
)
save_figure(fig, "Figure_3_strong_baseline_tradeoff")


# -----------------------------------------------------------------------------
# Figure 4: R1 normalized total cost
# -----------------------------------------------------------------------------
r1g = pd.read_csv(ROOT / "r1_model_global_results.csv")
trace_order = ["R1_A_gradual_periodic", "R1_B_mean_reverting", "R1_C_noisy_bursts"]
trace_labels = ["Gradual and periodic", "Mean reverting", "Noisy bursts"]
policy_specs = [
    ("best_static", "Best static", COL["static"]),
    ("afhc_d_frozen", "AF-HC-D", COL["afhc"]),
    ("shape_ewma_be_frozen", "Shape-EWMA-BE", COL["shape"]),
    ("repressdb_frozen", "RePressDB", COL["repress"]),
]
fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.65), sharey=True)
for row_index, size in enumerate([8192, 16384]):
    for col_index, (trace, trace_label) in enumerate(zip(trace_order, trace_labels)):
        ax = axes[row_index, col_index]
        values = []
        initial = r1g[
            (r1g.trace == trace)
            & (r1g.chunk_size == size)
            & (r1g.policy == "initial_best")
        ].iloc[0].total_time_ns
        for policy, _, _ in policy_specs:
            row = r1g[
                (r1g.trace == trace)
                & (r1g.chunk_size == size)
                & (r1g.policy == policy)
            ].iloc[0]
            values.append(float(row.total_time_ns / initial))
        x = np.arange(4, dtype=float)
        for xpos, value, (_, _, color) in zip(x, values, policy_specs):
            gradient_bar(ax, xpos, value - 0.82, 0.56, color, bottom=0.82)
            ax.text(
                xpos,
                value + 0.009,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=COL["ink"],
            )
        ax.axhline(1.0, color="#68727B", lw=0.85, ls=(0, (4, 3)))
        ax.set_xticks(x, ["Static", "AF-HC", "EWMA", "RePress"], rotation=0)
        ax.set_xlim(-0.5, 3.5)
        ax.tick_params(axis="x", labelsize=6.6)
        if row_index == 0:
            ax.set_title(trace_label, fontweight="bold")
        if col_index == 0:
            ax.set_ylabel(f"{size // 1024}K\nTotal cost / initial-best")
        ax.set_ylim(0.82, 1.18)
        style_axis(ax)
        panel_label(ax, f"({chr(97 + row_index * 3 + col_index)})")
fig.legend(
    handles=[
        Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=darken(color), label=label)
        for _, label, color in policy_specs
    ],
    frameon=False,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
)
fig.subplots_adjust(
    left=0.085, right=0.99, top=0.94, bottom=0.16,
    hspace=0.44, wspace=0.25,
)
save_figure(fig, "Figure_4_continuous_total_cost")


# -----------------------------------------------------------------------------
# Figure 5: migration risk
# -----------------------------------------------------------------------------
r1m = pd.read_csv(ROOT / "r1_migration_records.csv")
policy_order = ["afhc_d_frozen", "shape_ewma_be_frozen", "repressdb_frozen"]
policy_labels = ["AF-HC-D", "Shape-EWMA-BE", "RePressDB"]
policy_colors = [COL["afhc"], COL["shape"], COL["repress"]]
summary_rows = []
for policy in policy_order:
    sub = r1m[r1m.policy == policy]
    assessable = sub[sub.stale_assessable.astype(bool)]
    summary_rows.append(
        {
            "migrations": len(sub),
            "rewrite_mb": sub.rewrite_bytes.sum() / 1e6,
            "unamortized": 100.0 * sub.unamortized.astype(bool).mean(),
            "stale": 100.0 * assessable.stale.astype(bool).mean(),
        }
    )
metrics = [
    ("migrations", "Migrations", "{:.0f}"),
    ("rewrite_mb", "Rewritten payload (MB)", "{:.1f}"),
    ("unamortized", "Unamortized migrations (%)", "{:.1f}"),
    ("stale", "Stale migrations (%)", "{:.1f}"),
]
fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.85))
for panel_index, (ax, (key, ylabel, fmt)) in enumerate(zip(axes.flat, metrics)):
    values = [row[key] for row in summary_rows]
    x = np.arange(3, dtype=float)
    upper = max(values) * 1.22 if max(values) > 0 else 1.0
    ax.set_ylim(0, upper)
    for xpos, value, color in zip(x, values, policy_colors):
        gradient_bar(ax, xpos, value, 0.56, color)
        add_bar_value(ax, xpos, value, fmt, offset=0.018)
    ax.set_xticks(x, policy_labels, rotation=13)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel(ylabel)
    style_axis(ax)
    panel_label(ax, f"({chr(97 + panel_index)})")
fig.subplots_adjust(
    left=0.105, right=0.985, top=0.955, bottom=0.145,
    hspace=0.46, wspace=0.30,
)
save_figure(fig, "Figure_5_migration_risk")


# -----------------------------------------------------------------------------
# Figure 6: cumulative cost and migration paths
# -----------------------------------------------------------------------------
r1c = pd.read_csv(ROOT / "r1_cumulative_cost_curves.csv")
selected = r1c[
    (r1c.trace == "R1_C_noisy_bursts")
    & (r1c.chunk_size == 16384)
    & r1c.policy.isin(
        ["best_static", "initial_best", "afhc_d_frozen", "shape_ewma_be_frozen", "repressdb_frozen"]
    )
].copy()
line_specs = [
    ("best_static", "Best static", COL["static"], "-"),
    ("initial_best", "Initial-best", COL["initial"], "--"),
    ("afhc_d_frozen", "AF-HC-D", COL["afhc"], "-"),
    ("shape_ewma_be_frozen", "Shape-EWMA-BE", COL["shape"], "-"),
    ("repressdb_frozen", "RePressDB", COL["repress"], "-"),
]
fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05))
ax = axes[0]
for policy, label, color, linestyle in line_specs:
    sub = selected[selected.policy == policy].sort_values("global_event_index")
    ax.plot(
        sub.global_event_index / 1000.0,
        sub.cumulative_total_time_ns / 1e9,
        color=color,
        linestyle=linestyle,
        label=label,
        linewidth=1.65 if policy == "repressdb_frozen" else 1.35,
    )
ax.set_xlabel("Processed queries (thousands)")
ax.set_ylabel("Cumulative measured cost (s)")
style_axis(ax)
panel_label(ax, "(a)")
ax.legend(frameon=False, loc="upper left", ncol=2)

ax = axes[1]
for policy, label, color in [
    ("afhc_d_frozen", "AF-HC-D", COL["afhc"]),
    ("shape_ewma_be_frozen", "Shape-EWMA-BE", COL["shape"]),
    ("repressdb_frozen", "RePressDB", COL["repress"]),
]:
    sub = selected[selected.policy == policy].sort_values("global_event_index")
    ax.step(
        sub.global_event_index / 1000.0,
        sub.cumulative_migrations,
        where="post",
        color=color,
        label=label,
    )
ax.set_xlabel("Processed queries (thousands)")
ax.set_ylabel("Cumulative migrations")
style_axis(ax)
panel_label(ax, "(b)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(w_pad=1.4)
save_figure(fig, "Figure_6_cumulative_behavior")


# -----------------------------------------------------------------------------
# Figure 7: correctness, scale, and startup
# -----------------------------------------------------------------------------
with open(ROOT / "b1_assessment.json", "r", encoding="utf-8") as handle:
    b1 = json.load(handle)
with open(ROOT / "b2_assessment.json", "r", encoding="utf-8") as handle:
    b2 = json.load(handle)
scale = pd.read_csv(ROOT / "b3_scale_build_summary.csv")
startup = pd.read_csv(ROOT / "b3_startup_records.csv")

operation_labels = [
    "B1 file queries",
    "B1 migrations",
    "B1 recovery trials",
    "B2 file queries",
    "B2 migrations",
]
operation_counts = [100000, 1000, 600, 1000000, 1000]

fig = plt.figure(figsize=(7.25, 3.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.38)
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(operation_labels))[::-1]
for ypos, count in zip(y, operation_counts):
    gradient_bar(ax, ypos, math.log10(count), 0.53, COL["static"], bottom=0.0)
# Rotate coordinate convention for horizontal appearance by recreating with barh-style rectangles.
ax.clear()
style_axis(ax, grid="x")
ax.set_xscale("log")
for ypos, count in zip(y, operation_counts):
    ax.barh(
        ypos,
        count,
        height=0.56,
        color=lighten(COL["static"], 0.25),
        edgecolor=darken(COL["static"], 0.1),
        linewidth=0.7,
    )
    ax.text(count * 1.05, ypos, f"{count:,} passed", va="center", fontsize=7.0, clip_on=False)
ax.set_yticks(y, operation_labels)
ax.set_xlabel("Verified operations (log scale)")
ax.set_xlim(300, 3_200_000)
panel_label(ax, "(a)")

ax = fig.add_subplot(gs[0, 1])
scale_order = ["logical_1GiB", "logical_5GiB", "logical_10GiB"]
labels = ["1 GiB", "5 GiB", "10 GiB"]
clean = []
dirty = []
for name in scale_order:
    clean.append(
        startup[(startup.scale_name == name) & (startup.startup_type == "clean")].startup_ns.median() / 1e9
    )
    dirty.append(
        startup[(startup.scale_name == name) & (startup.startup_type == "dirty")].startup_ns.median() / 1e9
    )
x = np.arange(3, dtype=float)
width = 0.32
for xpos, value in zip(x - width / 2, clean):
    gradient_bar(ax, xpos, value, width * 0.86, COL["repress_light"])
    add_bar_value(ax, xpos, value, "{:.2f}")
for xpos, value in zip(x + width / 2, dirty):
    gradient_bar(ax, xpos, value, width * 0.86, COL["repress"])
    add_bar_value(ax, xpos, value, "{:.2f}")
ax.set_xticks(x, labels)
ax.set_ylabel("Startup time (s)")
ax.set_ylim(0, 7.1)
style_axis(ax)
ax.legend(
    handles=[
        Rectangle((0, 0), 1, 1, facecolor=COL["repress_light"], edgecolor=darken(COL["repress_light"]), label="Clean"),
        Rectangle((0, 0), 1, 1, facecolor=COL["repress"], edgecolor=darken(COL["repress"]), label="One dirty chunk"),
    ],
    frameon=False,
    loc="upper left",
)
panel_label(ax, "(b)")

ax = fig.add_subplot(gs[0, 2])
scale_lookup = scale.set_index("scale_name")
chunk_counts = [scale_lookup.loc[name, "logical_chunk_count"] for name in scale_order]
catalog_bpc = [
    scale_lookup.loc[name, "base_catalog_bytes"] / scale_lookup.loc[name, "logical_chunk_count"]
    for name in scale_order
]
ax.plot(
    chunk_counts,
    catalog_bpc,
    color=COL["purple"],
    marker="o",
    markerfacecolor="white",
    markeredgewidth=1.3,
)
for xpos, value in zip(chunk_counts, catalog_bpc):
    ax.text(xpos, value + 0.42, f"{value:.1f}", ha="center", va="bottom", fontsize=7.2)
ax.set_xscale("log")
ax.set_xticks(chunk_counts, ["32,768", "163,840", "327,680"])
ax.set_xlabel("Logical chunks")
ax.set_ylabel("Catalog bytes / chunk")
ax.set_ylim(295, 305)
style_axis(ax)
panel_label(ax, "(c)")
fig.subplots_adjust(
    left=0.105, right=0.985, top=0.93, bottom=0.17, wspace=0.42
)
save_figure(fig, "Figure_7_dependability_and_scale")


# -----------------------------------------------------------------------------
# Figure 8: interference and actual audit
# -----------------------------------------------------------------------------
b3c = pd.read_csv(ROOT / "b3c_balanced_interference_summary.csv")
audit = pd.read_csv(ROOT / "r1_actual_audit_summary.csv")

fig = plt.figure(figsize=(7.25, 4.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.44, wspace=0.33)

order_scales = ["logical_1GiB", "logical_5GiB", "logical_10GiB"]
scale_labels = {"logical_1GiB": "1 GiB", "logical_5GiB": "5 GiB", "logical_10GiB": "10 GiB"}
rows = []
for scale_name in order_scales:
    for threads in [1, 4, 8]:
        row = b3c[(b3c.scale_name == scale_name) & (b3c.thread_count == threads)].iloc[0]
        rows.append((scale_name, threads, row))

y = np.arange(len(rows))[::-1]
labels = [f"{scale_labels[s]} / {t} thread{'s' if t > 1 else ''}" for s, t, _ in rows]

ax = fig.add_subplot(gs[0, 0])
values = np.asarray([row.throughput_ratio_geometric_mean for _, _, row in rows])
low = np.asarray([row.throughput_ratio_ci_low for _, _, row in rows])
high = np.asarray([row.throughput_ratio_ci_high for _, _, row in rows])
ax.errorbar(
    values,
    y,
    xerr=[values - low, high - values],
    fmt="o",
    color=COL["repress"],
    ecolor=COL["repress_mid"],
    capsize=2.5,
    markersize=4.5,
)
ax.axvline(1.0, color="#6D777F", ls=(0, (4, 3)), lw=0.9)
ax.set_yticks(y, labels)
ax.set_xlabel("Background / foreground throughput")
ax.set_xlim(0.95, 1.005)
style_axis(ax, grid="x")
panel_label(ax, "(a)")

ax = fig.add_subplot(gs[0, 1])
values = np.asarray([row.p99_amplification_geometric_mean for _, _, row in rows])
low = np.asarray([row.p99_amplification_ci_low for _, _, row in rows])
high = np.asarray([row.p99_amplification_ci_high for _, _, row in rows])
ax.errorbar(
    values,
    y,
    xerr=[values - low, high - values],
    fmt="s",
    color=COL["shape"],
    ecolor=darken(COL["shape"], 0.05),
    capsize=2.5,
    markersize=4.2,
)
ax.axvline(1.0, color="#6D777F", ls=(0, (4, 3)), lw=0.9)
ax.set_yticks(y, [""] * len(y))
ax.set_xlabel("Background / foreground p99 latency")
ax.set_xlim(0.99, 1.16)
style_axis(ax, grid="x")
panel_label(ax, "(b)")

ax = fig.add_subplot(gs[1, :])
policy_specs = [
    ("initial_best", "Initial-best", COL["initial"]),
    ("shape_ewma_be_frozen", "Shape-EWMA-BE", COL["shape"]),
    ("repressdb_frozen", "RePressDB", COL["repress"]),
]
x = np.arange(2, dtype=float)
width = 0.24
for policy_index, (policy, label, color) in enumerate(policy_specs):
    values = [
        float(
            audit[(audit.policy == policy) & (audit.chunk_size == size)].total_time_ns.iloc[0]
            / 1e9
        )
        for size in [8192, 16384]
    ]
    offsets = x + (policy_index - 1) * width
    for xpos, value in zip(offsets, values):
        gradient_bar(ax, xpos, value, width * 0.86, color)
        add_bar_value(ax, xpos, value, "{:.3f}", offset=0.018)
ax.set_xticks(x, ["8K", "16K"])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("Audit total time (s)")
ax.set_ylim(0, 7.25)
style_axis(ax)
ax.legend(
    handles=[
        Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=darken(color), label=label)
        for _, label, color in policy_specs
    ],
    frameon=False,
    ncol=3,
    loc="upper left",
)
panel_label(ax, "(c)")
fig.subplots_adjust(
    left=0.105, right=0.985, top=0.95, bottom=0.13,
    hspace=0.46, wspace=0.34,
)
save_figure(fig, "Figure_8_interference_and_audit")


# -----------------------------------------------------------------------------
# Graphical abstract (separate submission item)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.45, 4.15))
ax.set_xlim(0, 13.28)
ax.set_ylim(0, 5.31)
ax.axis("off")

# Header and three-stage narrative
ax.text(
    0.28,
    4.93,
    "RePressDB",
    fontsize=18,
    fontweight="bold",
    color=COL["repress"],
)
ax.text(
    2.62,
    4.94,
    "dependable self-adaptive storage under workload drift",
    fontsize=11.5,
    color=COL["ink"],
)
ax.plot([0.28, 13.0], [4.65, 4.65], color="#D3D9DE", lw=0.9)

stages = [
    (0.35, "Observe", ["Point, scan, and aggregate mix", "Short and long evidence windows"], COL["repress"]),
    (4.55, "Decide", ["Agreement and confirmation", "Break-even and write budget"], COL["shape"]),
    (8.75, "Reconfigure safely", ["Single active version", "Atomic commit and recovery"], COL["static"]),
]
for x0, title, lines, color in stages:
    patch = FancyBboxPatch(
        (x0, 2.15),
        3.55,
        1.75,
        boxstyle="round,pad=0.05,rounding_size=0.14",
        facecolor=lighten(color, 0.87),
        edgecolor=color,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x0 + 0.22, 3.57, title, fontsize=11.2, fontweight="bold", color=COL["ink"])
    ax.text(x0 + 0.22, 3.12, "\n".join(lines), fontsize=9.0, color="#34424D", va="top", linespacing=1.35)
for x1, x2 in [(3.90, 4.55), (8.10, 8.75)]:
    ax.add_patch(
        FancyArrowPatch(
            (x1, 3.02),
            (x2, 3.02),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.15,
            color="#5F6B74",
        )
    )

metrics = [
    (0.55, "3.0-7.1%", "lower total cost\nacross six continuous replays", COL["repress"]),
    (4.65, "69%", "fewer migrations than\nthe strong single-EWMA baseline", COL["shape"]),
    (8.80, "600/600", "controlled recovery trials\ncompleted correctly", COL["static"]),
]
for x0, number, text, color in metrics:
    ax.text(x0, 1.37, number, fontsize=16.5, fontweight="bold", color=color)
    ax.text(x0, 0.88, text, fontsize=8.8, color=COL["ink"], va="top", linespacing=1.25)

save_figure(fig, "graphical_abstract", dpi=360)

print("Generated eight manuscript figures and one graphical abstract.")
