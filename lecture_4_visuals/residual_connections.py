"""
Lecture 4 Visual: Residual Connections — The Skip That Changes Everything

Shows:
1. Plain block vs Residual block — side-by-side architecture diagrams
2. Signal propagation across 8 layers: without vs with residual connections
3. Gradient flow: the "highway" that prevents vanishing gradients
4. The "small correction" insight — f(x) learns the delta, not the whole thing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ── Style ───────────────────────────────────────────────
BG      = "#1e1e2e"
TEXT    = "#cdd6f4"
SUBTLE  = "#6c7086"
MEDIUM  = "#a6adc8"
C_POS   = "#a6e3a1"
C_NEG   = "#f38ba8"
C_BOUND = "#f9e2af"
C_RULE  = "#fab387"
C_I     = "#89b4fa"
C_B     = "#cba6f7"
C_DIM   = "#585b70"
C_NODE  = "#89dceb"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ═════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════
def draw_box(ax, xy, w, h, label, color, fontsize=10, lw=1.8, alpha=0.15):
    """Draw a rounded box with centered label."""
    bx, by = xy
    ax.add_patch(patches.FancyBboxPatch(
        (bx, by), w, h, boxstyle="round,pad=0.06",
        facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw))
    ax.text(bx + w / 2, by + h / 2, label, fontsize=fontsize,
            ha="center", va="center", color=color, fontweight="bold")


def draw_arrow(ax, start, end, color, lw=2, style="-|>", shrinkA=0, shrinkB=0,
               connectionstyle=None):
    """Draw an arrow between two points."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=shrinkA, shrinkB=shrinkB,
                                connectionstyle=connectionstyle))


def draw_circle(ax, center, radius, label, color, fontsize=9):
    """Draw a circle node with label."""
    cx, cy = center
    circle = plt.Circle((cx, cy), radius, facecolor=BG, edgecolor=color,
                         linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(cx, cy, label, fontsize=fontsize, ha="center", va="center",
            color=color, fontweight="bold", zorder=6)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 34))
fig.suptitle("Residual Connections: The Skip That Changes Everything",
             fontsize=18, fontweight="bold", y=0.995, color=C_BOUND)

gs = GridSpec(5, 2, figure=fig,
              height_ratios=[0.6, 3.5, 3.2, 3.2, 1.4],
              hspace=0.18, wspace=0.28,
              top=0.965, bottom=0.015, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 0: Intro
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-0.2, 3)

ax0.text(8, 2.4,
    "Without residual connections, each layer must learn the ENTIRE transformation.",
    fontsize=13, ha="center", color=TEXT)
ax0.text(8, 1.4,
    "\"Like rebuilding a house from scratch every time you want to repaint a wall.\"",
    fontsize=12, ha="center", color=MEDIUM, fontstyle="italic")
ax0.text(8, 0.3,
    "With residual connections, each layer only learns a SMALL CORRECTION on top of what's already there.",
    fontsize=13, ha="center", color=C_I, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: Plain block — x → f(x)
# ─────────────────────────────────────────────────────────
ax_plain = fig.add_subplot(gs[1, 0])
ax_plain.axis("off")
ax_plain.set_xlim(0, 14); ax_plain.set_ylim(-1, 11)
ax_plain.set_title("Plain Block:  x → f(x)",
                   fontsize=14, fontweight="bold", color=C_NEG, pad=10)

# Input label
ax_plain.text(7, 10.2, "x  (input)", fontsize=13, ha="center",
              color=C_I, fontweight="bold")

# Arrow from input down to layer box
draw_arrow(ax_plain, (7, 9.8), (7, 8.3), C_I, lw=2.5)

# Layer box (tall)
draw_box(ax_plain, (3, 5.5), 8, 2.8, "", C_NEG, lw=2)
ax_plain.text(7, 7.3, "Layer Block", fontsize=12, ha="center",
              color=C_NEG, fontweight="bold")
ax_plain.text(7, 6.5, "weights → matmul → ReLU", fontsize=10,
              ha="center", color=MEDIUM)

# Arrow from layer box down to output
draw_arrow(ax_plain, (7, 5.5), (7, 3.8), C_NEG, lw=2.5)

# Output label
ax_plain.text(7, 3.2, "f(x)  (output)", fontsize=13, ha="center",
              color=C_NEG, fontweight="bold")

# Explanation below
ax_plain.add_patch(patches.FancyBboxPatch(
    (1.5, -0.5), 11, 2.5, boxstyle="round,pad=0.1",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1.2))
ax_plain.text(7, 1.5, "The layer must learn the COMPLETE\n"
              "transformation from scratch.", fontsize=10.5,
              ha="center", color=C_NEG, fontweight="bold")
ax_plain.text(7, 0.1, "If f is close to identity, the layer must\n"
              "carefully learn w ≈ I — hard to get right.",
              fontsize=9.5, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: Residual block — x → x + f(x)
# ─────────────────────────────────────────────────────────
ax_res = fig.add_subplot(gs[1, 1])
ax_res.axis("off")
ax_res.set_xlim(0, 14); ax_res.set_ylim(-1, 11)
ax_res.set_title("Residual Block:  x → x + f(x)",
                fontsize=14, fontweight="bold", color=C_POS, pad=10)

# Input label
ax_res.text(5, 10.2, "x  (input)", fontsize=13, ha="center",
            color=C_I, fontweight="bold")

# Arrow splitting point
draw_arrow(ax_res, (5, 9.8), (5, 8.3), C_I, lw=2.5)

# Layer box (centered left-ish)
draw_box(ax_res, (1.5, 5.5), 7, 2.8, "", C_B, lw=2)
ax_res.text(5, 7.3, "Layer Block", fontsize=12, ha="center",
            color=C_B, fontweight="bold")
ax_res.text(5, 6.5, "weights → matmul → ReLU", fontsize=10,
            ha="center", color=MEDIUM)

# Arrow from input into layer box
draw_arrow(ax_res, (5, 8.3), (5, 8.3), C_I, lw=0)  # placeholder

# Arrow from layer box down to addition circle
draw_arrow(ax_res, (5, 5.5), (5, 4.5), C_B, lw=2)

# === THE SKIP CONNECTION (the key visual) ===
# Curved arrow from input level, going to the right, then down, then to the add node
# Draw it as a path: right from split point, down, then left to the add node
skip_x = 11.5
ax_res.annotate("", xy=(5.5, 4.0), xytext=(5.5, 9.5),
                arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=3.5,
                                connectionstyle="arc3,rad=-0.5",
                                linestyle="-"))

# Label on the skip connection
ax_res.text(11.2, 6.9, "SKIP\nCONNECTION",
            fontsize=12, ha="center", color=C_POS, fontweight="bold",
            bbox=dict(facecolor=BG, edgecolor=C_POS, pad=6, alpha=0.95,
                      boxstyle="round,pad=0.3"))
ax_res.text(11.2, 5.8, "x passes through\nunchanged",
            fontsize=9.5, ha="center", color=C_POS)

# Addition circle
draw_circle(ax_res, (5, 3.8), 0.5, "+", C_BOUND, fontsize=16)

# Arrow from add circle to output
draw_arrow(ax_res, (5, 3.3), (5, 2.3), C_POS, lw=2.5)

# Output label
ax_res.text(5, 1.7, "x + f(x)  (output)", fontsize=13, ha="center",
            color=C_POS, fontweight="bold")

# Small labels on the two inputs to +
ax_res.text(3.8, 4.7, "f(x)", fontsize=10, ha="center", color=C_B,
            fontweight="bold")
ax_res.text(6.6, 4.7, "x", fontsize=10, ha="center", color=C_POS,
            fontweight="bold")

# Explanation below
ax_res.add_patch(patches.FancyBboxPatch(
    (0.5, -0.5), 12.5, 2.5, boxstyle="round,pad=0.1",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.2))
ax_res.text(6.75, 1.5, "The layer only needs to learn a SMALL\n"
            "CORRECTION f(x) on top of x.",
            fontsize=10.5, ha="center", color=C_POS, fontweight="bold")
ax_res.text(6.75, 0.1, "If the best thing to do is \"almost nothing,\"\n"
            "f(x) ≈ 0 is trivial to learn!",
            fontsize=9.5, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 2: Signal propagation — 8 layers comparison
# ─────────────────────────────────────────────────────────
ax_sig = fig.add_subplot(gs[2, :])
ax_sig.axis("off")
ax_sig.set_xlim(0, 16); ax_sig.set_ylim(-2.5, 7.5)
ax_sig.set_title("Signal Propagation Through 8 Layers",
                fontsize=14, fontweight="bold", color=C_BOUND, pad=10)

n_layers = 8
w = 0.7  # each layer multiplies signal by this

# ── Top row: Plain network ──
y_plain = 6.0
ax_sig.text(0.3, y_plain + 0.8, "Plain Network    x → f(f(f(...))) = w⁸·x",
            fontsize=11, color=C_NEG, fontweight="bold")

plain_signal = 1.0
for i in range(n_layers + 1):
    cx = 1.2 + i * 1.65
    # Signal bar (height proportional to signal strength)
    bar_h = plain_signal * 2.5
    bar_w = 0.6
    ax_sig.add_patch(patches.FancyBboxPatch(
        (cx - bar_w/2, y_plain - bar_h), bar_w, bar_h,
        boxstyle="round,pad=0.02",
        facecolor=C_NEG, alpha=0.3 + 0.5 * plain_signal,
        edgecolor=C_NEG, linewidth=1))
    # Signal magnitude label
    ax_sig.text(cx, y_plain - bar_h - 0.3, f"{plain_signal:.2f}",
                fontsize=8, ha="center", color=C_NEG, fontweight="bold")
    # Layer label
    if i == 0:
        ax_sig.text(cx, y_plain + 0.35, "x", fontsize=9, ha="center",
                    color=TEXT, fontweight="bold")
    else:
        ax_sig.text(cx, y_plain + 0.35, f"L{i}", fontsize=8, ha="center",
                    color=SUBTLE)

    # Arrow to next
    if i < n_layers:
        ax_sig.annotate("", xy=(cx + 0.75, y_plain - 0.3),
                        xytext=(cx + 0.4, y_plain - 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=1))
        # Multiply label
        ax_sig.text(cx + 0.82, y_plain - 0.05, f"×{w}", fontsize=7,
                    ha="center", color=SUBTLE)
    plain_signal *= w

# Result annotation
final_plain = 0.7 ** 8
ax_sig.text(15.2, y_plain - 1.0, f"Signal = {final_plain:.3f}\n(5.7% left!)",
            fontsize=10, ha="center", color=C_NEG, fontweight="bold",
            bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=5, alpha=0.9))

# ── Bottom row: Residual network ──
y_res = 1.5
ax_sig.text(0.3, y_res + 0.8, "Residual Network    x → x + f(x) each step → (1+w)⁸·x",
            fontsize=11, color=C_POS, fontweight="bold")

# With residual, effective multiplier is (1 + w) per layer
# but we want to show: even if w is small/negative (shrinking), the +1 stabilizes
# Use w_eff = 0.7 for the correction, so total multiplier is 1 + (0.7 - 1) = 0.7?
# No — the point is: in plain net, multiplier is w. In residual, multiplier is (1+w).
# So if the learned transformation produces a factor of w=0.7 on x,
# plain: output = 0.7x
# residual: output = x + 0.7x = 1.7x? No that grows.
# Let's reframe: the layer's correction f(x) ≈ w·x where w is small.
# Plain net layer: output = f(x) = w·x (the layer IS the whole thing)
# Residual layer: output = x + f(x) = x + w·x = (1+w)x
# If w = -0.3 (the correction slightly shrinks):
# Plain: multiplier = -0.3 (destroys signal immediately)
# Residual: multiplier = 0.7 (okay, still reduces but much more gentle)
#
# Better: show f(x) = w·x as the transformation.
# Plain: each layer does x → w·x, so after L layers: w^L · x
# Residual: each layer does x → (1+w)·x, so after L layers: (1+w)^L · x
# The notes say: w=0.7 plain → (0.7)^8 = 0.058
# Residual with w as the correction factor. If correction w = -0.05 (small):
# (1 + (-0.05))^8 = 0.95^8 = 0.663 — much more preserved.
# Let's use correction w = -0.05 for residual to make the contrast clear.

w_corr = -0.05  # small correction per layer
res_signal = 1.0
for i in range(n_layers + 1):
    cx = 1.2 + i * 1.65
    bar_h = res_signal * 2.5
    bar_w = 0.6
    ax_sig.add_patch(patches.FancyBboxPatch(
        (cx - bar_w/2, y_res - bar_h), bar_w, bar_h,
        boxstyle="round,pad=0.02",
        facecolor=C_POS, alpha=0.3 + 0.5 * min(res_signal, 1),
        edgecolor=C_POS, linewidth=1))
    ax_sig.text(cx, y_res - bar_h - 0.3, f"{res_signal:.2f}",
                fontsize=8, ha="center", color=C_POS, fontweight="bold")
    if i == 0:
        ax_sig.text(cx, y_res + 0.35, "x", fontsize=9, ha="center",
                    color=TEXT, fontweight="bold")
    else:
        ax_sig.text(cx, y_res + 0.35, f"L{i}", fontsize=8, ha="center",
                    color=SUBTLE)

    if i < n_layers:
        ax_sig.annotate("", xy=(cx + 0.75, y_res - 0.3),
                        xytext=(cx + 0.4, y_res - 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=1))
        ax_sig.text(cx + 0.82, y_res - 0.05, f"×{1+w_corr}", fontsize=7,
                    ha="center", color=SUBTLE)
    res_signal *= (1 + w_corr)

final_res = (1 + w_corr) ** 8
ax_sig.text(15.2, y_res - 1.0, f"Signal = {final_res:.3f}\n({final_res*100:.0f}% left!)",
            fontsize=10, ha="center", color=C_POS, fontweight="bold",
            bbox=dict(facecolor=BG, edgecolor=C_POS, pad=5, alpha=0.9))

# Divider
ax_sig.plot([0.5, 15.5], [y_res + 1.3, y_res + 1.3], color=C_DIM,
            lw=0.8, ls="--", alpha=0.4)

# Central comparison
ax_sig.add_patch(patches.FancyBboxPatch(
    (4.5, -2.3), 7, 1.5, boxstyle="round,pad=0.1",
    facecolor=C_BOUND, alpha=0.06, edgecolor=C_BOUND, linewidth=1.2))
ax_sig.text(8, -1.1, "Plain: each layer multiplies by w → signal = w⁸",
            fontsize=9.5, ha="center", color=C_NEG, fontweight="bold")
ax_sig.text(8, -1.8, "Residual: each layer multiplies by (1+w) → signal ≈ stays near 1",
            fontsize=9.5, ha="center", color=C_POS, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 3: Gradient flow — the highway
# ─────────────────────────────────────────────────────────
ax_grad = fig.add_subplot(gs[3, :])
ax_grad.axis("off")
ax_grad.set_xlim(0, 16); ax_grad.set_ylim(-2.5, 8)
ax_grad.set_title("Gradient Flow: Why Residual = Express Highway for Gradients",
                 fontsize=14, fontweight="bold", color=C_B, pad=10)

# ── Top: Plain network gradient ──
y_pg = 6.5
ax_grad.text(0.2, y_pg + 0.6, "Plain Network — gradients must pass through every layer",
             fontsize=11, color=C_NEG, fontweight="bold")

block_w = 1.3
gap = 0.4
n_blocks = 5
start_x = 1.5

for i in range(n_blocks):
    bx = start_x + i * (block_w + gap)
    # Layer box
    ax_grad.add_patch(patches.FancyBboxPatch(
        (bx, y_pg - 0.7), block_w, 1.0, boxstyle="round,pad=0.04",
        facecolor=C_NEG, alpha=0.1 + 0.05 * i, edgecolor=C_NEG, linewidth=1.5))
    ax_grad.text(bx + block_w / 2, y_pg - 0.2, f"L{i+1}",
                 fontsize=9, ha="center", color=C_NEG, fontweight="bold")
    # Forward arrows
    if i < n_blocks - 1:
        ax_grad.annotate("", xy=(bx + block_w + gap - 0.05, y_pg - 0.05),
                         xytext=(bx + block_w + 0.05, y_pg - 0.05),
                         arrowprops=dict(arrowstyle="-|>", color=C_I,
                                         lw=1.5, alpha=0.4))

# Loss box
loss_x = start_x + n_blocks * (block_w + gap)
ax_grad.add_patch(patches.FancyBboxPatch(
    (loss_x, y_pg - 0.7), 1.3, 1.0, boxstyle="round,pad=0.04",
    facecolor=C_BOUND, alpha=0.15, edgecolor=C_BOUND, linewidth=1.5))
ax_grad.text(loss_x + 0.65, y_pg - 0.2, "Loss",
             fontsize=9, ha="center", color=C_BOUND, fontweight="bold")

# Backward gradient arrows (below the boxes) — getting dimmer
for i in range(n_blocks - 1, -1, -1):
    bx = start_x + i * (block_w + gap)
    alpha_val = 0.2 + 0.6 * ((i + 1) / n_blocks)
    lw_val = 1 + 1.5 * ((i + 1) / n_blocks)
    if i < n_blocks - 1:
        next_bx = start_x + (i + 1) * (block_w + gap)
        ax_grad.annotate("", xy=(bx + block_w + 0.05, y_pg - 0.9),
                         xytext=(next_bx - 0.05, y_pg - 0.9),
                         arrowprops=dict(arrowstyle="-|>", color=C_NEG,
                                         lw=lw_val, alpha=alpha_val))
    else:
        ax_grad.annotate("", xy=(bx + block_w + 0.05, y_pg - 0.9),
                         xytext=(loss_x - 0.05, y_pg - 0.9),
                         arrowprops=dict(arrowstyle="-|>", color=C_NEG,
                                         lw=lw_val, alpha=alpha_val))

# First layer arrow (dimmest)
bx0 = start_x
ax_grad.annotate("", xy=(bx0 - 0.4, y_pg - 0.9),
                 xytext=(bx0 - 0.05, y_pg - 0.9),
                 arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=0.8,
                                 alpha=0.15))

# Gradient labels
ax_grad.text(loss_x + 1.8, y_pg - 0.2, "← gradient gets\n    weaker and\n    weaker",
             fontsize=9, color=C_NEG, fontweight="bold")

ax_grad.text(start_x - 1.0, y_pg - 0.5, "almost\nnothing\narrives",
             fontsize=8.5, color=C_NEG, fontweight="bold", ha="center",
             alpha=0.5)

# Divider
ax_grad.plot([0.5, 15.5], [y_pg - 1.8, y_pg - 1.8], color=C_DIM,
             lw=0.8, ls="--", alpha=0.4)

# ── Bottom: Residual network gradient ──
y_rg = 3.0
ax_grad.text(0.2, y_rg + 1.8, "Residual Network — gradients have a direct highway",
             fontsize=11, color=C_POS, fontweight="bold")

for i in range(n_blocks):
    bx = start_x + i * (block_w + gap)
    # Layer box
    ax_grad.add_patch(patches.FancyBboxPatch(
        (bx, y_rg - 0.2), block_w, 1.0, boxstyle="round,pad=0.04",
        facecolor=C_B, alpha=0.1 + 0.03 * i, edgecolor=C_B, linewidth=1.5))
    ax_grad.text(bx + block_w / 2, y_rg + 0.3, f"L{i+1}",
                 fontsize=9, ha="center", color=C_B, fontweight="bold")

    # "+" circle after each block
    plus_x = bx + block_w + gap / 2
    circle = plt.Circle((plus_x, y_rg + 0.3), 0.15,
                         facecolor=BG, edgecolor=C_BOUND, linewidth=1.5, zorder=5)
    ax_grad.add_patch(circle)
    ax_grad.text(plus_x, y_rg + 0.3, "+", fontsize=8, ha="center",
                 va="center", color=C_BOUND, fontweight="bold", zorder=6)

# Loss box
ax_grad.add_patch(patches.FancyBboxPatch(
    (loss_x, y_rg - 0.2), 1.3, 1.0, boxstyle="round,pad=0.04",
    facecolor=C_BOUND, alpha=0.15, edgecolor=C_BOUND, linewidth=1.5))
ax_grad.text(loss_x + 0.65, y_rg + 0.3, "Loss",
             fontsize=9, ha="center", color=C_BOUND, fontweight="bold")

# THE HIGHWAY — a big bold arrow going straight from Loss to start, ABOVE the boxes
highway_y = y_rg + 1.5
ax_grad.annotate("",
    xy=(start_x - 0.3, highway_y),
    xytext=(loss_x + 1.3, highway_y),
    arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=4, alpha=0.8))
ax_grad.text((start_x + loss_x) / 2 + 0.5, highway_y + 0.25,
             "GRADIENT HIGHWAY  (skip connections carry gradient directly)",
             fontsize=9.5, ha="center", color=C_POS, fontweight="bold",
             bbox=dict(facecolor=BG, edgecolor=C_POS, pad=4, alpha=0.9))

# Also show gradients flowing through the blocks (secondary path)
for i in range(n_blocks - 1, -1, -1):
    bx = start_x + i * (block_w + gap)
    # Small arrows below blocks
    if i < n_blocks - 1:
        next_bx = start_x + (i + 1) * (block_w + gap)
        ax_grad.annotate("", xy=(bx + block_w + 0.05, y_rg - 0.45),
                         xytext=(next_bx - 0.05, y_rg - 0.45),
                         arrowprops=dict(arrowstyle="-|>", color=C_B,
                                         lw=1.2, alpha=0.35))

# Label the secondary path
ax_grad.text((start_x + loss_x) / 2 + 0.5, y_rg - 0.75,
             "(gradients also flow through layers — but they don't NEED to)",
             fontsize=8.5, ha="center", color=C_B, alpha=0.6)

# Key insight box
ax_grad.add_patch(patches.FancyBboxPatch(
    (2.5, -2.3), 11, 1.5, boxstyle="round,pad=0.1",
    facecolor=C_BOUND, alpha=0.06, edgecolor=C_BOUND, linewidth=1.2))
ax_grad.text(8, -1.1, "∂Loss/∂x  always includes a term that goes directly from output to input.",
             fontsize=10, ha="center", color=C_BOUND, fontweight="bold")
ax_grad.text(8, -1.85, "Even if the layer gradients are tiny, the skip path keeps the total gradient alive.",
             fontsize=9.5, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 4: Key Takeaway
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4, :])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-3, 3.5)

ax4.add_patch(patches.FancyBboxPatch(
    (1.0, -2.5), 14, 5.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax4.text(8, 2.5, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("Residual connection:  output = x + f(x)  instead of  output = f(x)", True),
    ("The input x gets a \"free pass\" — it always arrives at the output unchanged.", False),
    ("f(x) only needs to learn the CORRECTION — not rebuild everything.", True),
    ("Signal stays strong: multiplying by (1+w) instead of w prevents decay.", False),
    ("Gradients get a highway: even if layers have tiny gradients, the skip path survives.", True),
    ("This is why modern deep networks (ResNets, Transformers) all use residual connections.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.5 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_4_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "residual_connections.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
