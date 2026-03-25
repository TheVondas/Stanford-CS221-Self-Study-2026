"""
Lecture 4 Visual: Initialization — Why Starting Weights Matter

Shows:
1. What initialization IS — picking the starting numbers for all weights
2. The dot-product problem: why more inputs = bigger outputs (with real numbers)
3. Too large vs too small vs just right — activations across 10 layers
4. Xavier scaling: 1/√d_in — the fix, with a concrete d_in=100 example
5. Key takeaway
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
fig = plt.figure(figsize=(16, 40))
fig.suptitle("Initialization: Why Starting Weights Matter",
             fontsize=18, fontweight="bold", y=0.997, color=C_BOUND)

gs = GridSpec(6, 2, figure=fig,
              height_ratios=[0.5, 3.2, 3.5, 3.5, 3.0, 1.2],
              hspace=0.18, wspace=0.28,
              top=0.968, bottom=0.012, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 0: Intro
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-0.2, 3)

ax0.text(8, 2.4,
    "Before training starts, every weight in the network must be given an initial value.",
    fontsize=13, ha="center", color=TEXT)
ax0.text(8, 1.4,
    "Pick them too large → activations explode.  Too small → activations vanish.",
    fontsize=12, ha="center", color=MEDIUM, fontstyle="italic")
ax0.text(8, 0.3,
    "Xavier initialization scales weights by 1/√d_in so the signal stays balanced from layer to layer.",
    fontsize=13, ha="center", color=C_I, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: What IS initialization?
# ─────────────────────────────────────────────────────────
ax_what = fig.add_subplot(gs[1, 0])
ax_what.axis("off")
ax_what.set_xlim(0, 14); ax_what.set_ylim(-1, 11)
ax_what.set_title("What Is Initialization?",
                  fontsize=14, fontweight="bold", color=C_I, pad=10)

# Show a mini weight matrix with random numbers
ax_what.text(7, 9.8, "A neural network starts with RANDOM weights.",
             fontsize=10.5, ha="center", color=TEXT)

# Weight matrix visualization
mat_x, mat_y = 2.5, 5.8
mat_w, mat_h = 9, 3.5
ax_what.add_patch(patches.FancyBboxPatch(
    (mat_x, mat_y), mat_w, mat_h, boxstyle="round,pad=0.08",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))
ax_what.text(mat_x + mat_w / 2, mat_y + mat_h + 0.2,
             "Weight matrix W  (before any training)",
             fontsize=10, ha="center", color=C_I, fontweight="bold")

# Fill with random-looking numbers
np.random.seed(7)
sample_w = np.random.randn(3, 4) * 0.5
for r in range(3):
    for c in range(4):
        val = sample_w[r, c]
        color = C_POS if val >= 0 else C_NEG
        cx = mat_x + 1.1 + c * 2.1
        cy = mat_y + mat_h - 0.7 - r * 1.1
        ax_what.text(cx, cy, f"{val:+.2f}", fontsize=11, ha="center",
                     va="center", color=color, fontweight="bold",
                     family="monospace")

# The question
ax_what.text(7, 5.0, "The question:", fontsize=11, ha="center",
             color=C_BOUND, fontweight="bold")
ax_what.text(7, 4.0, "HOW BIG should these random numbers be?",
             fontsize=12, ha="center", color=C_BOUND, fontweight="bold")

# Three scenarios
scenarios = [
    ("Too large (e.g. ±5)", C_NEG, "→ activations explode layer by layer"),
    ("Too small (e.g. ±0.001)", C_B, "→ activations shrink to nothing"),
    ("Just right (Xavier)", C_POS, "→ activations stay balanced!"),
]
for i, (label, color, desc) in enumerate(scenarios):
    y = 2.5 - i * 1.0
    ax_what.add_patch(patches.FancyBboxPatch(
        (1, y - 0.3), 12, 0.8, boxstyle="round,pad=0.04",
        facecolor=color, alpha=0.08, edgecolor=color, linewidth=1))
    ax_what.text(5, y + 0.1, label, fontsize=10, ha="center",
                 color=color, fontweight="bold")
    ax_what.text(10, y + 0.1, desc, fontsize=9.5, ha="center",
                 color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: The dot product problem
# ─────────────────────────────────────────────────────────
ax_dot = fig.add_subplot(gs[1, 1])
ax_dot.axis("off")
ax_dot.set_xlim(0, 14); ax_dot.set_ylim(-1, 11)
ax_dot.set_title("The Core Problem: Dot Products Grow With Dimension",
                 fontsize=13, fontweight="bold", color=C_NEG, pad=10)

# Show: one neuron computes z = w₁x₁ + w₂x₂ + ... + w_d x_d
ax_dot.text(7, 9.8, "Each neuron computes a dot product:",
            fontsize=10.5, ha="center", color=TEXT)
ax_dot.text(7, 8.9, "z = w₁·x₁ + w₂·x₂ + w₃·x₃ + ... + w_d·x_d",
            fontsize=12, ha="center", color=C_I, fontweight="bold",
            family="monospace")

# Small example: d_in = 4
ax_dot.add_patch(patches.FancyBboxPatch(
    (0.5, 5.5), 13, 3.0, boxstyle="round,pad=0.1",
    facecolor=C_POS, alpha=0.04, edgecolor=C_POS, linewidth=1.2))
ax_dot.text(7, 8.1, "Example with d_in = 4 inputs, weights ~ ±1:",
            fontsize=10, ha="center", color=C_POS, fontweight="bold")

ax_dot.text(7, 7.2,
    "z  =  (+0.8)(+0.5) + (−0.6)(+1.1) + (+0.9)(−0.3) + (+0.4)(+0.7)",
    fontsize=9.5, ha="center", color=TEXT, family="monospace")
ax_dot.text(7, 6.4,
    "=  0.40   +   (−0.66)   +   (−0.27)   +   0.28",
    fontsize=9.5, ha="center", color=TEXT, family="monospace")
ax_dot.text(7, 5.7,
    "=  −0.25    ← 4 terms, result stays small",
    fontsize=10, ha="center", color=C_POS, fontweight="bold",
    family="monospace")

# Large example: d_in = 1000
ax_dot.add_patch(patches.FancyBboxPatch(
    (0.5, 2.5), 13, 2.5, boxstyle="round,pad=0.1",
    facecolor=C_NEG, alpha=0.04, edgecolor=C_NEG, linewidth=1.2))
ax_dot.text(7, 4.6, "But with d_in = 1000 inputs, same ±1 weights:",
            fontsize=10, ha="center", color=C_NEG, fontweight="bold")
ax_dot.text(7, 3.8,
    "z  =  w₁x₁ + w₂x₂ + ... + w₁₀₀₀x₁₀₀₀",
    fontsize=10, ha="center", color=TEXT, family="monospace")
ax_dot.text(7, 3.1,
    "1000 random ±1 terms add up  →  |z| ≈ √1000 ≈ 31.6  ← HUGE!",
    fontsize=10, ha="center", color=C_NEG, fontweight="bold",
    family="monospace")

# The insight
ax_dot.add_patch(patches.FancyBboxPatch(
    (1.0, -0.2), 12, 2.3, boxstyle="round,pad=0.1",
    facecolor=C_BOUND, alpha=0.06, edgecolor=C_BOUND, linewidth=1.2))
ax_dot.text(7, 1.5, "The math: if each w·x has variance σ²,",
            fontsize=10, ha="center", color=C_BOUND, fontweight="bold")
ax_dot.text(7, 0.7,
    "then the sum of d terms has variance d·σ²  →  magnitude grows as √d",
    fontsize=10, ha="center", color=C_BOUND, fontweight="bold")
ax_dot.text(7, 0.05, "More inputs = bigger output.  This compounds layer after layer!",
            fontsize=9.5, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 2: Bar chart — output magnitude vs d_in
# ─────────────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[2, 0])
ax_bar.set_title("Output Magnitude Grows With # of Inputs",
                fontsize=12, fontweight="bold", color=C_NEG, pad=10)

dims = [4, 16, 64, 256, 1024]
np.random.seed(42)
n_trials = 500

# For UNSCALED weights (w ~ N(0,1)):
magnitudes_unscaled = []
for d in dims:
    outputs = []
    for _ in range(n_trials):
        w = np.random.randn(d)
        x = np.random.randn(d)
        z = np.dot(w, x)
        outputs.append(abs(z))
    magnitudes_unscaled.append(np.mean(outputs))

# For XAVIER weights (w ~ N(0, 1/d)):
magnitudes_xavier = []
for d in dims:
    outputs = []
    for _ in range(n_trials):
        w = np.random.randn(d) / np.sqrt(d)
        x = np.random.randn(d)
        z = np.dot(w, x)
        outputs.append(abs(z))
    magnitudes_xavier.append(np.mean(outputs))

x_pos = np.arange(len(dims))
width = 0.35

bars1 = ax_bar.bar(x_pos - width/2, magnitudes_unscaled, width,
                   color=C_NEG, alpha=0.6, edgecolor=C_NEG, linewidth=1.5,
                   label="Unscaled: w ~ N(0, 1)")
bars2 = ax_bar.bar(x_pos + width/2, magnitudes_xavier, width,
                   color=C_POS, alpha=0.6, edgecolor=C_POS, linewidth=1.5,
                   label="Xavier: w ~ N(0, 1/d)")

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}", fontsize=8, ha="center", color=C_NEG,
                fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}", fontsize=8, ha="center", color=C_POS,
                fontweight="bold")

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels([f"d={d}" for d in dims])
ax_bar.set_xlabel("Number of inputs (d_in)", fontsize=11, color=C_I)
ax_bar.set_ylabel("Average |output|", fontsize=11, color=C_I)
ax_bar.legend(fontsize=9, facecolor=BG, edgecolor=C_DIM, labelcolor=TEXT)

# Reference line at 1.0
ax_bar.axhline(y=0.8, color=C_POS, lw=1, ls="--", alpha=0.3)
ax_bar.text(len(dims) - 0.5, 1.2, "ideal ≈ 0.8",
            fontsize=8, color=C_POS, alpha=0.6)

style_ax(ax_bar)


# ─────────────────────────────────────────────────────────
#  ROW 2, RIGHT: The Xavier fix — concrete example
# ─────────────────────────────────────────────────────────
ax_fix = fig.add_subplot(gs[2, 1])
ax_fix.axis("off")
ax_fix.set_xlim(0, 14); ax_fix.set_ylim(-1, 11)
ax_fix.set_title("The Xavier Fix: Scale by 1/√d_in",
                fontsize=13, fontweight="bold", color=C_POS, pad=10)

# Concrete example with d_in = 100
ax_fix.add_patch(patches.FancyBboxPatch(
    (0.5, 7.0), 13, 3.5, boxstyle="round,pad=0.1",
    facecolor=C_NEG, alpha=0.05, edgecolor=C_NEG, linewidth=1.2))

ax_fix.text(7, 10.0, "Problem: d_in = 100 inputs",
            fontsize=11, ha="center", color=C_NEG, fontweight="bold")
ax_fix.text(7, 9.1,
    "If weights w ~ N(0, 1):  each w·x has variance ≈ 1",
    fontsize=10, ha="center", color=TEXT)
ax_fix.text(7, 8.3,
    "Sum of 100 terms → variance = 100 → std dev = √100 = 10",
    fontsize=10, ha="center", color=C_NEG, fontweight="bold")
ax_fix.text(7, 7.5,
    "Output is ~10× too large!  And this happens EVERY LAYER.",
    fontsize=10, ha="center", color=C_NEG, fontweight="bold")

# Arrow
ax_fix.annotate("", xy=(7, 6.7), xytext=(7, 7.0),
                arrowprops=dict(arrowstyle="-|>", color=C_BOUND, lw=2.5))
ax_fix.text(7, 6.4, "Xavier solution:", fontsize=11, ha="center",
            color=C_BOUND, fontweight="bold")

# Solution
ax_fix.add_patch(patches.FancyBboxPatch(
    (0.5, 3.2), 13, 3.0, boxstyle="round,pad=0.1",
    facecolor=C_POS, alpha=0.05, edgecolor=C_POS, linewidth=1.2))

ax_fix.text(7, 5.7,
    "Scale each weight by 1/√d_in = 1/√100 = 1/10 = 0.1",
    fontsize=10.5, ha="center", color=C_POS, fontweight="bold")
ax_fix.text(7, 4.9,
    "Now w ~ N(0, 1/100):  each w·x has variance ≈ 1/100",
    fontsize=10, ha="center", color=TEXT)
ax_fix.text(7, 4.1,
    "Sum of 100 terms → variance = 100 × (1/100) = 1 → std dev = 1",
    fontsize=10, ha="center", color=C_POS, fontweight="bold")
ax_fix.text(7, 3.4,
    "Output magnitude ≈ 1.  Perfectly balanced!",
    fontsize=10.5, ha="center", color=C_POS, fontweight="bold")

# The formula
ax_fix.text(7, 2.2,
    "Xavier:    w  ~  N(0,  1/d_in)",
    fontsize=14, ha="center", color=C_BOUND, fontweight="bold",
    family="monospace",
    bbox=dict(facecolor=BG, edgecolor=C_BOUND, pad=8, alpha=0.9))

# Intuition
ax_fix.text(7, 0.8,
    "Intuition: more inputs → each weight contributes less",
    fontsize=10.5, ha="center", color=MEDIUM, fontstyle="italic")
ax_fix.text(7, 0.1,
    "so the total stays at the same scale regardless of layer width.",
    fontsize=10.5, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 3: Activation distributions across 10 layers — 3 scenarios
# ─────────────────────────────────────────────────────────
ax_layers = fig.add_subplot(gs[3, :])
ax_layers.set_title("Activations Across 10 Layers: Three Initialization Strategies",
                   fontsize=14, fontweight="bold", color=C_BOUND, pad=10)

np.random.seed(123)
n_neurons = 256
n_layers = 10
d_in = 256

# Run 3 scenarios
scenarios = {
    "Too Large (σ=2.0)":   2.0,
    "Too Small (σ=0.01)":  0.01,
    "Xavier (σ=1/√256≈0.0625)": 1.0 / np.sqrt(d_in),
}
colors_scen = [C_NEG, C_B, C_POS]
offsets = [-0.25, 0, 0.25]

# Collect stats: mean |activation| per layer for each scenario
for idx, ((name, sigma), color, offset) in enumerate(
        zip(scenarios.items(), colors_scen, offsets)):
    mean_abs = []
    h = np.random.randn(n_neurons)  # same starting input for all
    for L in range(n_layers):
        W = np.random.randn(n_neurons, n_neurons) * sigma
        z = W @ h
        h = np.maximum(0, z)  # ReLU
        mean_abs.append(np.mean(np.abs(h)) if np.mean(np.abs(h)) > 0 else 1e-30)

    x_pos = np.arange(1, n_layers + 1) + offset
    bars = ax_layers.bar(x_pos, mean_abs, width=0.22, color=color,
                         alpha=0.6, edgecolor=color, linewidth=1,
                         label=name)

    # Value labels (only if not too extreme)
    for bar in bars:
        bh = bar.get_height()
        if bh > 1e10:
            label_text = f"{bh:.0e}"
        elif bh < 1e-5:
            label_text = f"{bh:.0e}"
        else:
            label_text = f"{bh:.1f}"
        # Position label
        y_label = min(bh + bh * 0.05, ax_layers.get_ylim()[1] * 0.95
                      if ax_layers.get_ylim()[1] > 0 else bh + 1)
        ax_layers.text(bar.get_x() + bar.get_width()/2,
                       bh * 1.05 + 0.1,
                       label_text, fontsize=6.5, ha="center", color=color,
                       fontweight="bold", rotation=45)

ax_layers.set_xlabel("Layer", fontsize=11, color=C_I)
ax_layers.set_ylabel("Mean |activation|  (log scale)", fontsize=11, color=C_I)
ax_layers.set_xticks(range(1, n_layers + 1))
ax_layers.set_xticklabels([f"L{i}" for i in range(1, n_layers + 1)])
ax_layers.set_yscale("log")
ax_layers.legend(fontsize=9, facecolor=BG, edgecolor=C_DIM, labelcolor=TEXT,
                loc="upper left")

# Healthy zone band
ax_layers.axhspan(0.1, 10, alpha=0.04, color=C_POS)
ax_layers.text(n_layers + 0.3, 1.0, "healthy\nzone",
               fontsize=9, color=C_POS, fontweight="bold", alpha=0.7)

style_ax(ax_layers)


# ─────────────────────────────────────────────────────────
#  ROW 4: How Xavier keeps variance = 1 at every layer
# ─────────────────────────────────────────────────────────
ax_chain = fig.add_subplot(gs[4, :])
ax_chain.axis("off")
ax_chain.set_xlim(0, 16); ax_chain.set_ylim(-1, 8.5)
ax_chain.set_title("Why Xavier Works: Variance Stays = 1 at Every Layer",
                  fontsize=14, fontweight="bold", color=C_POS, pad=10)

# Show the chain: layer by layer, variance in vs variance out
chain_y = 6.0
box_w = 1.8
box_h = 1.2

labels = ["Input", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Output"]
n_boxes = len(labels)
total_w = n_boxes * box_w + (n_boxes - 1) * 0.6
start_x = (16 - total_w) / 2

for i, label in enumerate(labels):
    bx = start_x + i * (box_w + 0.6)
    is_layer = i > 0 and i < n_boxes - 1
    color = C_B if is_layer else C_I

    ax_chain.add_patch(patches.FancyBboxPatch(
        (bx, chain_y), box_w, box_h, boxstyle="round,pad=0.06",
        facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5))
    ax_chain.text(bx + box_w / 2, chain_y + box_h / 2 + 0.15,
                  label, fontsize=9, ha="center", va="center",
                  color=color, fontweight="bold")

    # Variance label below each box
    ax_chain.text(bx + box_w / 2, chain_y - 0.35,
                  "Var ≈ 1", fontsize=9, ha="center",
                  color=C_POS, fontweight="bold")

    # Arrow to next
    if i < n_boxes - 1:
        ax_chain.annotate("",
            xy=(bx + box_w + 0.55, chain_y + box_h / 2),
            xytext=(bx + box_w + 0.05, chain_y + box_h / 2),
            arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=2))

        # "×d × (1/d) = ×1" annotation above arrows for layer boxes
        if i < n_boxes - 2:
            ax_chain.text(bx + box_w + 0.3, chain_y + box_h + 0.25,
                          "×1", fontsize=8.5, ha="center",
                          color=C_POS, fontweight="bold")

# The math explanation below
math_y = 3.5
ax_chain.add_patch(patches.FancyBboxPatch(
    (1.0, math_y - 0.5), 14, 2.8, boxstyle="round,pad=0.1",
    facecolor=C_BOUND, alpha=0.05, edgecolor=C_BOUND, linewidth=1.2))

ax_chain.text(8, math_y + 1.8, "The math (simplified):", fontsize=11,
              ha="center", color=C_BOUND, fontweight="bold")

lines = [
    ("One neuron:    z = w₁x₁ + w₂x₂ + ... + w_d x_d", TEXT),
    ("Var(z) = d_in × Var(w) × Var(x)", TEXT),
    ("If Var(w) = 1/d_in:    Var(z) = d_in × (1/d_in) × Var(x) = 1 × Var(x)", C_POS),
    ("So output variance = input variance.  Signal preserved!", C_POS),
]
for i, (line, color) in enumerate(lines):
    ax_chain.text(8, math_y + 1.1 - i * 0.6, line, fontsize=9.5,
                  ha="center", color=color, family="monospace",
                  fontweight="bold" if color == C_POS else "normal")

# Contrast: without Xavier
contrast_y = 0.5
ax_chain.add_patch(patches.FancyBboxPatch(
    (1.0, contrast_y - 0.5), 6.5, 2.0, boxstyle="round,pad=0.1",
    facecolor=C_NEG, alpha=0.05, edgecolor=C_NEG, linewidth=1))
ax_chain.text(4.25, contrast_y + 1.1,
              "Without Xavier (Var(w) = 1):", fontsize=9.5,
              ha="center", color=C_NEG, fontweight="bold")
ax_chain.text(4.25, contrast_y + 0.4,
              "Var(z) = d_in × Var(x)", fontsize=9.5,
              ha="center", color=C_NEG, family="monospace")
ax_chain.text(4.25, contrast_y - 0.2,
              "Each layer multiplies variance by d!", fontsize=9,
              ha="center", color=C_NEG)

ax_chain.add_patch(patches.FancyBboxPatch(
    (8.5, contrast_y - 0.5), 6.5, 2.0, boxstyle="round,pad=0.1",
    facecolor=C_POS, alpha=0.05, edgecolor=C_POS, linewidth=1))
ax_chain.text(11.75, contrast_y + 1.1,
              "With Xavier (Var(w) = 1/d_in):", fontsize=9.5,
              ha="center", color=C_POS, fontweight="bold")
ax_chain.text(11.75, contrast_y + 0.4,
              "Var(z) = 1 × Var(x)", fontsize=9.5,
              ha="center", color=C_POS, family="monospace")
ax_chain.text(11.75, contrast_y - 0.2,
              "Each layer preserves variance!", fontsize=9,
              ha="center", color=C_POS)


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
    ("Initialization = choosing the starting random weights before training begins.", True),
    ("Dot products grow with dimension: d inputs with ±1 weights → output ≈ √d.", False),
    ("Xavier fix: scale weights by 1/√d_in so each layer's output variance ≈ input variance.", True),
    ("Too large → exploding activations.  Too small → vanishing activations.", False),
    ("With Xavier, the signal stays balanced from first layer to last — training can begin.", True),
]
for i, (txt, bold) in enumerate(takeaways):
    ax5.text(8, 1.5 - i * 0.7, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_4_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "initialization.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
