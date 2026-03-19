"""
Lecture 4 Visual: Layer Normalization — What Activations Are & Why We Normalize

Shows:
1. What "activations" actually are — the numbers living inside each neuron
2. A concrete numeric example: raw activations → compute mean/std → normalize
3. What goes wrong WITHOUT normalization (drift toward extremes across layers)
4. What happens WITH normalization (activations stay healthy)
5. The learned scale (γ) and shift (β) — the network can adjust after normalizing
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


def style_ax(ax):
    ax.tick_params(colors=SUBTLE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_DIM)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 42))
fig.suptitle("Layer Normalization: What Activations Are & Why We Normalize",
             fontsize=18, fontweight="bold", y=0.997, color=C_BOUND)

gs = GridSpec(6, 2, figure=fig,
              height_ratios=[0.5, 3.0, 3.5, 3.5, 2.8, 1.2],
              hspace=0.18, wspace=0.28,
              top=0.968, bottom=0.012, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 0: Intro
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-0.2, 3)

ax0.text(8, 2.4,
    "\"Activations\" are just the numbers that live inside each neuron after computation.",
    fontsize=13, ha="center", color=TEXT)
ax0.text(8, 1.4,
    "Without control, these numbers can drift toward zero or infinity — killing learning.",
    fontsize=12, ha="center", color=MEDIUM, fontstyle="italic")
ax0.text(8, 0.3,
    "Layer normalization recenters and rescales them so they stay in a healthy range.",
    fontsize=13, ha="center", color=C_I, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: What IS an activation?
# ─────────────────────────────────────────────────────────
ax_what = fig.add_subplot(gs[1, 0])
ax_what.axis("off")
ax_what.set_xlim(0, 14); ax_what.set_ylim(-1, 11)
ax_what.set_title("What Is an \"Activation\"?",
                  fontsize=14, fontweight="bold", color=C_I, pad=10)

# Show a mini network: input → layer → hidden neurons with numbers
# Input
ax_what.text(1.5, 9.5, "Input x", fontsize=11, ha="center",
             color=C_I, fontweight="bold")
ax_what.add_patch(patches.FancyBboxPatch(
    (0.3, 8.5), 2.4, 0.8, boxstyle="round,pad=0.05",
    facecolor=C_I, alpha=0.1, edgecolor=C_I, linewidth=1.5))
ax_what.text(1.5, 8.9, "[0.5, −1.2, 0.8]", fontsize=9, ha="center",
             color=C_I, family="monospace")

# Arrow
ax_what.annotate("", xy=(5.5, 8.9), xytext=(3.0, 8.9),
                 arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=2))
ax_what.text(4.25, 9.4, "W·x + b", fontsize=9, ha="center", color=MEDIUM)
ax_what.text(4.25, 9.85, "then ReLU", fontsize=8.5, ha="center", color=MEDIUM)

# Hidden layer — THE ACTIVATIONS
ax_what.text(9, 9.5, "Hidden layer h", fontsize=11, ha="center",
             color=C_POS, fontweight="bold")

# Draw 4 neurons with their activation values
neuron_vals = [2.1, 0.0, 5.7, 0.3]
neuron_labels = ["h₁", "h₂", "h₃", "h₄"]
neuron_colors = [C_POS, C_NEG, C_RULE, C_POS]
neuron_y_positions = [7.8, 6.2, 4.6, 3.0]

for j, (val, label, color, ny) in enumerate(
        zip(neuron_vals, neuron_labels, neuron_colors, neuron_y_positions)):
    # Neuron circle
    circle = plt.Circle((7, ny), 0.55, facecolor=BG, edgecolor=color,
                         linewidth=2, zorder=5)
    ax_what.add_patch(circle)
    ax_what.text(7, ny, label, fontsize=10, ha="center", va="center",
                 color=color, fontweight="bold", zorder=6)

    # Value label
    ax_what.text(9, ny, f"= {val}", fontsize=12, ha="left", va="center",
                 color=color, fontweight="bold", family="monospace")

    # Explanation
    if val == 0.0:
        ax_what.text(11.2, ny, "← dead (ReLU killed it)",
                     fontsize=8.5, ha="left", va="center", color=C_NEG)
    elif val == 5.7:
        ax_what.text(11.2, ny, "← very active!",
                     fontsize=8.5, ha="left", va="center", color=C_RULE)
    else:
        ax_what.text(11.2, ny, "← moderately active",
                     fontsize=8.5, ha="left", va="center", color=SUBTLE)

# Brace / label for the activation vector
ax_what.annotate("", xy=(5.8, 3.0), xytext=(5.8, 7.8),
                 arrowprops=dict(arrowstyle="<->", color=C_BOUND, lw=1.5))
ax_what.text(5.0, 5.4, "activation\nvector\nh = [2.1, 0, 5.7, 0.3]",
             fontsize=8.5, ha="center", color=C_BOUND, fontweight="bold")

# Summary box
ax_what.add_patch(patches.FancyBboxPatch(
    (0.5, -0.5), 13, 1.8, boxstyle="round,pad=0.1",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1))
ax_what.text(7, 0.9, "An \"activation\" = the output number of a neuron",
             fontsize=10, ha="center", color=C_I, fontweight="bold")
ax_what.text(7, 0.1, "The \"activation vector\" = all those numbers collected together: h = [h₁, h₂, h₃, h₄]",
             fontsize=9, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: The problem — activations drift
# ─────────────────────────────────────────────────────────
ax_prob = fig.add_subplot(gs[1, 1])
ax_prob.axis("off")
ax_prob.set_xlim(0, 14); ax_prob.set_ylim(-1, 11)
ax_prob.set_title("The Problem: Activations Drift Across Layers",
                  fontsize=14, fontweight="bold", color=C_NEG, pad=10)

# Show activation values growing/shrinking across 4 layers
layer_names = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]
layer_y = [9.0, 6.8, 4.6, 2.4]

# Simulated activations that get progressively worse
layer_activations = [
    [1.2, -0.5, 0.8, 0.3],       # healthy
    [3.8, -2.1, 4.5, -0.1],      # getting bigger
    [15.2, -8.7, 22.1, 0.01],    # too big, one too small
    [142.0, -89.0, 310.0, 0.0],  # exploded / dead
]
health_colors = [C_POS, C_BOUND, C_NEG, C_NEG]
health_labels = ["✓ Healthy range", "⚠ Getting stretched", "✗ Way too extreme",
                 "✗ Exploded + dead"]

for i, (name, y, acts, hc, hl) in enumerate(
        zip(layer_names, layer_y, layer_activations, health_colors, health_labels)):
    # Layer label
    ax_prob.text(0.3, y + 0.4, name, fontsize=10, color=hc, fontweight="bold")

    # Draw the 4 activation values as bars
    max_abs = max(abs(a) for a in acts)
    bar_scale = 3.5 / max(max_abs, 1)  # normalize bar width
    for j, a in enumerate(acts):
        bx = 3.5
        by = y - 0.15 + j * 0.35
        bw = abs(a) * bar_scale
        color = C_POS if a > 0 else C_NEG
        alpha = 0.4 if abs(a) < 50 else 0.7

        if a >= 0:
            ax_prob.add_patch(patches.Rectangle(
                (bx, by), bw, 0.25, facecolor=color, alpha=alpha))
        else:
            ax_prob.add_patch(patches.Rectangle(
                (bx - bw, by), bw, 0.25, facecolor=color, alpha=alpha))

    # Numeric values
    vals_str = ", ".join(f"{a:>6.1f}" if abs(a) < 100 else f"{a:>6.0f}"
                         for a in acts)
    ax_prob.text(8.5, y + 0.1, f"[{vals_str}]",
                 fontsize=7.5, ha="left", va="center", color=hc,
                 family="monospace")

    # Health label
    ax_prob.text(8.5, y - 0.4, hl, fontsize=8.5, ha="left", color=hc,
                 fontweight="bold")

# Problem explanation
ax_prob.add_patch(patches.FancyBboxPatch(
    (0.5, -0.5), 13, 1.8, boxstyle="round,pad=0.1",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1))
ax_prob.text(7, 0.9, "Without normalization, activations compound each layer.",
             fontsize=10, ha="center", color=C_NEG, fontweight="bold")
ax_prob.text(7, 0.1, "Big values → exploding gradients.  Tiny values → vanishing gradients.",
             fontsize=9, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 2: The computation — step by step with real numbers
# ─────────────────────────────────────────────────────────
ax_comp = fig.add_subplot(gs[2, :])
ax_comp.axis("off")
ax_comp.set_xlim(0, 16); ax_comp.set_ylim(-1.5, 8.5)
ax_comp.set_title("Layer Norm: Step-by-Step with Real Numbers",
                  fontsize=14, fontweight="bold", color=C_BOUND, pad=10)

# Use a concrete example: h = [2.0, -1.0, 5.0, 0.0]
raw = np.array([2.0, -1.0, 5.0, 0.0])
mu = raw.mean()       # 1.5
sigma = raw.std()     # ~2.06
normed = (raw - mu) / sigma

# ── Step 0: Raw activations ──
y0 = 7.5
ax_comp.text(1.0, y0, "START:", fontsize=11, color=C_I, fontweight="bold")
ax_comp.text(4.0, y0, "Raw activation vector  h = [2.0,  −1.0,  5.0,  0.0]",
             fontsize=11, color=C_I, fontweight="bold", family="monospace")

# Draw bars for raw values
bar_y = 5.3
bar_h = 1.6
bar_w = 1.2
bar_gap = 0.3
bar_start = 2.0
colors_raw = [C_POS, C_NEG, C_RULE, SUBTLE]

for j, (v, c) in enumerate(zip(raw, colors_raw)):
    bx = bar_start + j * (bar_w + bar_gap)
    # Bar from 0 up/down to value
    zero_y = bar_y + bar_h / 2
    bar_top = zero_y + v / 5.0 * (bar_h / 2)
    if v >= 0:
        ax_comp.add_patch(patches.FancyBboxPatch(
            (bx, zero_y), bar_w, bar_top - zero_y,
            boxstyle="round,pad=0.02",
            facecolor=c, alpha=0.4, edgecolor=c, linewidth=1.5))
    else:
        ax_comp.add_patch(patches.FancyBboxPatch(
            (bx, bar_top), bar_w, zero_y - bar_top,
            boxstyle="round,pad=0.02",
            facecolor=c, alpha=0.4, edgecolor=c, linewidth=1.5))
    ax_comp.text(bx + bar_w / 2, bar_top + (0.2 if v >= 0 else -0.35),
                 f"{v:.1f}", fontsize=10, ha="center", color=c,
                 fontweight="bold")

# Zero line
ax_comp.plot([bar_start - 0.3, bar_start + 4 * (bar_w + bar_gap)],
             [bar_y + bar_h / 2] * 2, color=C_DIM, lw=1, ls="--", alpha=0.5)
ax_comp.text(bar_start - 0.5, bar_y + bar_h / 2, "0", fontsize=8,
             ha="center", color=SUBTLE)

# Label
ax_comp.text(bar_start + 2 * (bar_w + bar_gap) - 0.15, bar_y - 0.1,
             "scattered, uncentered", fontsize=9, ha="center",
             color=MEDIUM, fontstyle="italic")

# ── Arrow to step 1 ──
ax_comp.annotate("", xy=(8.3, bar_y + bar_h / 2),
                 xytext=(7.5, bar_y + bar_h / 2),
                 arrowprops=dict(arrowstyle="-|>", color=C_BOUND, lw=2.5))

# ── Step 1+2: Compute mean and std ──
step_x = 8.5
ax_comp.add_patch(patches.FancyBboxPatch(
    (step_x, bar_y - 0.3), 3.8, 2.5, boxstyle="round,pad=0.08",
    facecolor=C_BOUND, alpha=0.06, edgecolor=C_BOUND, linewidth=1.5))

ax_comp.text(step_x + 1.9, bar_y + 1.85, "Step 1: Compute mean",
             fontsize=10, ha="center", color=C_BOUND, fontweight="bold")
ax_comp.text(step_x + 1.9, bar_y + 1.25,
             f"μ = (2 + (−1) + 5 + 0) / 4 = {mu:.1f}",
             fontsize=9, ha="center", color=TEXT, family="monospace")

ax_comp.text(step_x + 1.9, bar_y + 0.65, "Step 2: Compute std dev",
             fontsize=10, ha="center", color=C_BOUND, fontweight="bold")
ax_comp.text(step_x + 1.9, bar_y + 0.05,
             f"σ = √(avg of squared deviations) = {sigma:.2f}",
             fontsize=9, ha="center", color=TEXT, family="monospace")

# ── Arrow to step 3 ──
ax_comp.annotate("", xy=(14.0, bar_y + bar_h / 2),
                 xytext=(12.5, bar_y + bar_h / 2),
                 arrowprops=dict(arrowstyle="-|>", color=C_BOUND, lw=2.5))

# ── Step 3+4: Apply ──
ax_comp.add_patch(patches.FancyBboxPatch(
    (13.0, bar_y - 0.3), 2.8, 2.5, boxstyle="round,pad=0.08",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.5))
ax_comp.text(14.4, bar_y + 1.85, "Step 3-4: Normalize",
             fontsize=10, ha="center", color=C_POS, fontweight="bold")
ax_comp.text(14.4, bar_y + 1.2, "ĥᵢ = (hᵢ − μ) / σ",
             fontsize=10, ha="center", color=C_POS, fontweight="bold",
             family="monospace")
for j, (v, n) in enumerate(zip(raw, normed)):
    ax_comp.text(14.4, bar_y + 0.55 - j * 0.5,
                 f"({v:>4.1f} − {mu:.1f}) / {sigma:.2f} = {n:>5.2f}",
                 fontsize=8, ha="center", color=TEXT, family="monospace")

# ── Result: Normalized bars ──
bar2_y = 0.8
bar2_h = 2.5
ax_comp.text(1.0, bar2_y + bar2_h + 0.4, "RESULT:",
             fontsize=11, color=C_POS, fontweight="bold")
normed_str = ", ".join(f"{n:>5.2f}" for n in normed)
ax_comp.text(4.0, bar2_y + bar2_h + 0.4,
             f"Normalized  ĥ = [{normed_str}]",
             fontsize=11, color=C_POS, fontweight="bold", family="monospace")

for j, (n, c) in enumerate(zip(normed, colors_raw)):
    bx = bar_start + j * (bar_w + bar_gap)
    zero_y = bar2_y + bar2_h / 2
    bar_top = zero_y + n / 2.5 * (bar2_h / 2)
    if n >= 0:
        ax_comp.add_patch(patches.FancyBboxPatch(
            (bx, zero_y), bar_w, bar_top - zero_y,
            boxstyle="round,pad=0.02",
            facecolor=c, alpha=0.5, edgecolor=c, linewidth=1.5))
    else:
        ax_comp.add_patch(patches.FancyBboxPatch(
            (bx, bar_top), bar_w, zero_y - bar_top,
            boxstyle="round,pad=0.02",
            facecolor=c, alpha=0.5, edgecolor=c, linewidth=1.5))
    ax_comp.text(bx + bar_w / 2, bar_top + (0.2 if n >= 0 else -0.35),
                 f"{n:.2f}", fontsize=10, ha="center", color=c,
                 fontweight="bold")

# Zero line
ax_comp.plot([bar_start - 0.3, bar_start + 4 * (bar_w + bar_gap)],
             [bar2_y + bar2_h / 2] * 2, color=C_DIM, lw=1, ls="--",
             alpha=0.5)
ax_comp.text(bar_start - 0.5, bar2_y + bar2_h / 2, "0", fontsize=8,
             ha="center", color=SUBTLE)
ax_comp.text(bar_start + 2 * (bar_w + bar_gap) - 0.15, bar2_y - 0.3,
             "centered around 0, spread ≈ 1", fontsize=9, ha="center",
             color=C_POS, fontweight="bold", fontstyle="italic")

# Properties annotation
props_x = 8.5
ax_comp.add_patch(patches.FancyBboxPatch(
    (props_x, bar2_y - 0.5), 7, 3.0, boxstyle="round,pad=0.1",
    facecolor=C_POS, alpha=0.04, edgecolor=C_POS, linewidth=1))
ax_comp.text(props_x + 3.5, bar2_y + 2.1, "After normalization:",
             fontsize=10, ha="center", color=C_POS, fontweight="bold")
ax_comp.text(props_x + 3.5, bar2_y + 1.4, "• Mean = 0  (centered)",
             fontsize=9.5, ha="center", color=TEXT)
ax_comp.text(props_x + 3.5, bar2_y + 0.7, "• Std dev = 1  (controlled spread)",
             fontsize=9.5, ha="center", color=TEXT)
ax_comp.text(props_x + 3.5, bar2_y + 0.0, "• No value is extreme — all in a safe range",
             fontsize=9.5, ha="center", color=TEXT)


# ─────────────────────────────────────────────────────────
#  ROW 3, LEFT: Without norm — layer-by-layer activation distributions
# ─────────────────────────────────────────────────────────
ax_no = fig.add_subplot(gs[3, 0])
ax_no.set_title("Without Layer Norm — Activations Across 6 Layers",
               fontsize=12, fontweight="bold", color=C_NEG, pad=10)

np.random.seed(42)
n_neurons = 200
n_layers_vis = 6

# Simulate activations drifting: each layer multiplies & adds noise
# Start with healthy activations, let them grow
activations_no_norm = []
h = np.random.randn(n_neurons) * 0.5 + 0.5  # start healthy
for L in range(n_layers_vis):
    activations_no_norm.append(h.copy())
    # Simulate: W·h + b → ReLU, with slightly expansive weights
    W_scale = 1.15  # each layer amplifies a bit
    h = np.maximum(0, h * W_scale + np.random.randn(n_neurons) * 0.3
                   + 0.1 * L)

# Plot as violin-style distributions
positions = list(range(1, n_layers_vis + 1))
parts = ax_no.violinplot(activations_no_norm, positions=positions,
                         showmeans=True, showmedians=False)
for pc in parts['bodies']:
    pc.set_facecolor(C_NEG)
    pc.set_alpha(0.3)
    pc.set_edgecolor(C_NEG)
parts['cmeans'].set_color(C_BOUND)
parts['cmins'].set_color(C_DIM)
parts['cmaxes'].set_color(C_DIM)
parts['cbars'].set_color(C_DIM)

# Show the spread growing
for i, acts in enumerate(activations_no_norm):
    spread = acts.max() - acts.min()
    ax_no.text(i + 1, acts.max() + 0.8, f"spread\n{spread:.1f}",
               fontsize=7.5, ha="center", color=C_NEG, fontweight="bold")

ax_no.set_xlabel("Layer", fontsize=11, color=C_I)
ax_no.set_ylabel("Activation values", fontsize=11, color=C_I)
ax_no.set_xticks(positions)
ax_no.set_xticklabels([f"L{i}" for i in positions])

# Danger zone shading
ylims = ax_no.get_ylim()
ax_no.axhspan(ylims[0], 0, alpha=0.03, color=C_NEG)
ax_no.text(n_layers_vis + 0.3, ylims[0] + 0.5, "dead\nzone",
           fontsize=8, color=C_NEG, alpha=0.6)

style_ax(ax_no)


# ─────────────────────────────────────────────────────────
#  ROW 3, RIGHT: With norm — healthy distributions
# ─────────────────────────────────────────────────────────
ax_yes = fig.add_subplot(gs[3, 1])
ax_yes.set_title("With Layer Norm — Activations Across 6 Layers",
                fontsize=12, fontweight="bold", color=C_POS, pad=10)

activations_with_norm = []
h = np.random.randn(n_neurons) * 0.5 + 0.5
for L in range(n_layers_vis):
    # Apply layer norm: subtract mean, divide by std
    h_normed = (h - h.mean()) / (h.std() + 1e-8)
    activations_with_norm.append(h_normed.copy())
    # Same forward pass
    h = np.maximum(0, h_normed * 1.15 + np.random.randn(n_neurons) * 0.3
                   + 0.1 * L)

parts2 = ax_yes.violinplot(activations_with_norm, positions=positions,
                           showmeans=True, showmedians=False)
for pc in parts2['bodies']:
    pc.set_facecolor(C_POS)
    pc.set_alpha(0.3)
    pc.set_edgecolor(C_POS)
parts2['cmeans'].set_color(C_BOUND)
parts2['cmins'].set_color(C_DIM)
parts2['cmaxes'].set_color(C_DIM)
parts2['cbars'].set_color(C_DIM)

for i, acts in enumerate(activations_with_norm):
    spread = acts.max() - acts.min()
    ax_yes.text(i + 1, acts.max() + 0.3, f"spread\n{spread:.1f}",
                fontsize=7.5, ha="center", color=C_POS, fontweight="bold")

ax_yes.set_xlabel("Layer", fontsize=11, color=C_I)
ax_yes.set_ylabel("Activation values", fontsize=11, color=C_I)
ax_yes.set_xticks(positions)
ax_yes.set_xticklabels([f"L{i}" for i in positions])

# Healthy zone shading
ax_yes.axhspan(-2, 2, alpha=0.03, color=C_POS)
ax_yes.text(n_layers_vis + 0.3, 0, "healthy\nzone",
            fontsize=8, color=C_POS, alpha=0.6)

style_ax(ax_yes)


# ─────────────────────────────────────────────────────────
#  ROW 4: Learned scale (γ) and shift (β) — full picture
# ─────────────────────────────────────────────────────────
ax_full = fig.add_subplot(gs[4, :])
ax_full.axis("off")
ax_full.set_xlim(0, 16); ax_full.set_ylim(-1, 8)
ax_full.set_title("The Full Picture: Normalize, Then Let the Network Adjust",
                 fontsize=14, fontweight="bold", color=C_B, pad=10)

# Pipeline: h → subtract mean → divide by std → multiply by γ → add β → output
pipe_y = 5.5
box_w = 2.2
box_h = 1.5
boxes = [
    ("Raw\nactivations\nh", C_NEG, 0.5),
    ("Subtract\nmean\nh − μ", C_BOUND, 3.3),
    ("Divide by\nstd dev\n(h−μ)/σ", C_BOUND, 6.1),
    ("Scale by γ\n(learned)", C_B, 8.9),
    ("Shift by β\n(learned)", C_B, 11.7),
]

for label, color, bx in boxes:
    ax_full.add_patch(patches.FancyBboxPatch(
        (bx, pipe_y - 0.2), box_w, box_h, boxstyle="round,pad=0.06",
        facecolor=color, alpha=0.1, edgecolor=color, linewidth=1.8))
    ax_full.text(bx + box_w / 2, pipe_y + box_h / 2 - 0.2, label,
                 fontsize=9, ha="center", va="center", color=color,
                 fontweight="bold")

# Arrows between boxes
for i in range(len(boxes) - 1):
    x1 = boxes[i][2] + box_w
    x2 = boxes[i + 1][2]
    ax_full.annotate("", xy=(x2, pipe_y + box_h / 2 - 0.2),
                     xytext=(x1, pipe_y + box_h / 2 - 0.2),
                     arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=2))

# Output
ax_full.text(14.5, pipe_y + box_h / 2 - 0.2, "→",
             fontsize=16, ha="center", va="center", color=C_POS)
ax_full.add_patch(patches.FancyBboxPatch(
    (14.8, pipe_y - 0.2), 1.0, box_h, boxstyle="round,pad=0.06",
    facecolor=C_POS, alpha=0.15, edgecolor=C_POS, linewidth=1.8))
ax_full.text(15.3, pipe_y + box_h / 2 - 0.2, "ĥ",
             fontsize=14, ha="center", va="center", color=C_POS,
             fontweight="bold")

# Labels: "fixed math" vs "learned parameters"
ax_full.annotate("", xy=(3.3, pipe_y + box_h + 0.2),
                 xytext=(8.3 + box_w, pipe_y + box_h + 0.2),
                 arrowprops=dict(arrowstyle="<->", color=C_BOUND, lw=1.5))
ax_full.text(5.85, pipe_y + box_h + 0.55,
             "Fixed math (no learnable parameters)",
             fontsize=9, ha="center", color=C_BOUND, fontweight="bold")

ax_full.annotate("", xy=(8.9, pipe_y + box_h + 0.2),
                 xytext=(11.7 + box_w, pipe_y + box_h + 0.2),
                 arrowprops=dict(arrowstyle="<->", color=C_B, lw=1.5))
ax_full.text(11.4, pipe_y + box_h + 0.55,
             "Learned by the network",
             fontsize=9, ha="center", color=C_B, fontweight="bold")

# Formula
ax_full.text(8, 3.5,
             "Full formula:   ĥ  =  γ · (h − μ) / σ  +  β",
             fontsize=13, ha="center", color=C_BOUND, fontweight="bold",
             family="monospace",
             bbox=dict(facecolor=BG, edgecolor=C_BOUND, pad=8, alpha=0.9))

# Why γ and β explanation
ax_full.add_patch(patches.FancyBboxPatch(
    (1.0, 0.0), 14, 2.8, boxstyle="round,pad=0.1",
    facecolor=C_B, alpha=0.05, edgecolor=C_B, linewidth=1.2))

ax_full.text(8, 2.3, "Why do we need γ and β?", fontsize=11,
             ha="center", color=C_B, fontweight="bold")
ax_full.text(8, 1.6,
             "Without them, every layer's activations are FORCED to mean=0, std=1.",
             fontsize=10, ha="center", color=TEXT)
ax_full.text(8, 0.9,
             "But sometimes the network WANTS activations shifted or scaled differently.",
             fontsize=10, ha="center", color=TEXT)
ax_full.text(8, 0.2,
             "γ and β let the network learn: \"normalize first, then adjust to whatever range works best.\"",
             fontsize=10, ha="center", color=C_B, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 5: Key Takeaway
# ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[5, :])
ax5.axis("off")
ax5.set_xlim(0, 16); ax5.set_ylim(-3, 3.5)

ax5.add_patch(patches.FancyBboxPatch(
    (1.0, -2.5), 14, 5.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax5.text(8, 2.5, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("\"Activations\" = the output numbers of each neuron. Together they form a vector.", True),
    ("Without normalization, these values drift to extremes layer by layer.", False),
    ("Layer norm recenters (mean→0) and rescales (std→1) every activation vector.", True),
    ("Then γ (scale) and β (shift) let the network fine-tune the range it actually needs.", False),
    ("Result: stable training, even in very deep networks.", True),
]
for i, (txt, bold) in enumerate(takeaways):
    ax5.text(8, 1.5 - i * 0.7, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_4_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "layer_normalization.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
