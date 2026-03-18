"""
Lecture 3 Visual: Multiclass Classification — One Weight Vector Per Class

Shows:
1. Binary vs multiclass side-by-side: one weight vector vs K weight vectors
2. The weight matrix: how K separate weight vectors stack into a matrix
3. A concrete 3-class example with actual numbers: input → logits → prediction
4. How each class "competes" by computing its own score
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec

# ── Style ────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_POS = "#a6e3a1"        # green
C_NEG = "#f38ba8"         # pink
C_BOUNDARY = "#f9e2af"    # yellow
C_RULE = "#fab387"        # orange
C_I = "#89b4fa"           # blue
C_B = "#cba6f7"           # purple
C_DIM = "#585b70"

# One color per class for the 3-class example
C_CAT = "#f38ba8"         # pink
C_DOG = "#89b4fa"         # blue
C_BIRD = "#a6e3a1"        # green

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 24))
fig.suptitle("Multiclass Classification: One Weight Vector Per Class",
             fontsize=17, fontweight="bold", y=0.995, color=C_BOUNDARY)

gs = GridSpec(3, 2, figure=fig,
              height_ratios=[2.5, 3.5, 2.0],
              hspace=0.22, wspace=0.25,
              top=0.97, bottom=0.03, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: Binary — one weight vector
# ─────────────────────────────────────────────────────────
ax_bin = fig.add_subplot(gs[0, 0])
ax_bin.axis("off")
ax_bin.set_xlim(0, 10)
ax_bin.set_ylim(-1, 8)

ax_bin.text(5, 7.5, "Binary Classification", fontsize=14,
            ha="center", fontweight="bold", color=C_RULE)

# One weight vector
ax_bin.add_patch(patches.FancyBboxPatch(
    (0.5, 3.5), 9, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))

ax_bin.text(5, 6.5, "ONE score, ONE weight vector", fontsize=11,
            ha="center", fontweight="bold", color=C_I)

ax_bin.text(5, 5.5, "z  =  w · x  +  b", fontsize=14,
            ha="center", fontfamily="monospace", fontweight="bold",
            color=C_BOUNDARY)

ax_bin.text(5, 4.6, "w  =  [ w₁,  w₂,  …,  wₙ ]", fontsize=11,
            ha="center", fontfamily="monospace", color=MEDIUM)
ax_bin.text(5, 4.0, "one vector,  one bias,  one score",
            fontsize=10, ha="center", color=SUBTLE)

# Decision
ax_bin.text(5, 2.5, "z > 0  →  +1", fontsize=11,
            ha="center", fontfamily="monospace", color=C_POS)
ax_bin.text(5, 1.8, "z < 0  →  −1", fontsize=11,
            ha="center", fontfamily="monospace", color=C_NEG)

ax_bin.text(5, 0.5, "Two classes, one fence between them.",
            fontsize=10, ha="center", color=SUBTLE, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: Multiclass — K weight vectors
# ─────────────────────────────────────────────────────────
ax_multi = fig.add_subplot(gs[0, 1])
ax_multi.axis("off")
ax_multi.set_xlim(0, 10)
ax_multi.set_ylim(-1, 8)

ax_multi.text(5, 7.5, "Multiclass Classification (K classes)",
              fontsize=14, ha="center", fontweight="bold", color=C_RULE)

ax_multi.add_patch(patches.FancyBboxPatch(
    (0.5, 2.0), 9, 5.0, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_multi.text(5, 6.5, "K scores, K weight vectors", fontsize=11,
              ha="center", fontweight="bold", color=C_B)

# Each class gets its own line
classes = [
    ("z₀  =  w₀ · x  +  b₀", "class 0", C_CAT),
    ("z₁  =  w₁ · x  +  b₁", "class 1", C_DOG),
    ("z₂  =  w₂ · x  +  b₂", "class 2", C_BIRD),
    ("...", "", SUBTLE),
    ("zₖ₋₁ = wₖ₋₁ · x + bₖ₋₁", "class K−1", C_BOUNDARY),
]

for i, (eq, label, color) in enumerate(classes):
    y = 5.5 - i * 0.85
    ax_multi.text(1.5, y, eq, fontsize=11,
                  fontfamily="monospace", color=color, fontweight="bold")
    if label:
        ax_multi.text(8.5, y, label, fontsize=9.5, color=color)

# Decision
ax_multi.text(5, 1.2, "predict = class with largest zc",
              fontsize=11, ha="center", fontfamily="monospace",
              color=C_BOUNDARY, fontweight="bold")

ax_multi.text(5, 0.3, "K classes, each competing with its own score.",
              fontsize=10, ha="center", color=SUBTLE, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 2: Concrete 3-class example with the weight matrix
# ─────────────────────────────────────────────────────────
ax_ex = fig.add_subplot(gs[1, :])
ax_ex.axis("off")
ax_ex.set_xlim(0, 16)
ax_ex.set_ylim(-8, 7)

ax_ex.text(8, 6.5, "Concrete Example:  3 Classes,  2 Features",
           fontsize=14, ha="center", fontweight="bold", color=C_RULE)

ax_ex.text(8, 5.6,
           "Classes:  cat, dog, bird          Input features:  x = [ x₁, x₂ ]",
           fontsize=11, ha="center", color=MEDIUM)

# ── The weight matrix ──
ax_ex.add_patch(patches.FancyBboxPatch(
    (0.3, 0.5), 7.0, 4.5, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))

ax_ex.text(3.8, 4.5, "The Weight Matrix W", fontsize=12,
           ha="center", fontweight="bold", color=C_I)

ax_ex.text(3.8, 3.7,
           "Each ROW is one class's weight vector:",
           fontsize=10, ha="center", color=TEXT)

# Matrix display
# W = [[2, -1], [0, 3], [-1, 1]]  b = [1, -1, 0]
mat_x = 1.2
ax_ex.text(mat_x, 2.7, "W  =", fontsize=13,
           fontfamily="monospace", color=C_I, fontweight="bold")

# Big bracket
bracket_x = mat_x + 1.6
ax_ex.text(bracket_x, 2.7, "⎡", fontsize=20, fontfamily="monospace",
           color=C_DIM)
ax_ex.text(bracket_x, 1.8, "⎢", fontsize=20, fontfamily="monospace",
           color=C_DIM)
ax_ex.text(bracket_x, 0.9, "⎣", fontsize=20, fontfamily="monospace",
           color=C_DIM)

ax_ex.text(bracket_x + 0.5, 2.8, " 2   −1", fontsize=12,
           fontfamily="monospace", color=C_CAT, fontweight="bold")
ax_ex.text(bracket_x + 0.5, 1.9, " 0    3", fontsize=12,
           fontfamily="monospace", color=C_DOG, fontweight="bold")
ax_ex.text(bracket_x + 0.5, 1.0, "−1    1", fontsize=12,
           fontfamily="monospace", color=C_BIRD, fontweight="bold")

close_x = bracket_x + 3.0
ax_ex.text(close_x, 2.7, "⎤", fontsize=20, fontfamily="monospace",
           color=C_DIM)
ax_ex.text(close_x, 1.8, "⎥", fontsize=20, fontfamily="monospace",
           color=C_DIM)
ax_ex.text(close_x, 0.9, "⎦", fontsize=20, fontfamily="monospace",
           color=C_DIM)

# Row labels
ax_ex.text(close_x + 0.6, 2.8, "← w_cat", fontsize=10,
           fontfamily="monospace", color=C_CAT)
ax_ex.text(close_x + 0.6, 1.9, "← w_dog", fontsize=10,
           fontfamily="monospace", color=C_DOG)
ax_ex.text(close_x + 0.6, 1.0, "← w_bird", fontsize=10,
           fontfamily="monospace", color=C_BIRD)

# Bias vector
ax_ex.text(1.2, 0.1, "b = [ 1,  −1,  0 ]", fontsize=11,
           fontfamily="monospace", color=MEDIUM)

# ── The computation ──
ax_ex.add_patch(patches.FancyBboxPatch(
    (8.0, 0.5), 7.7, 4.5, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.5))

ax_ex.text(11.85, 4.5, "Computing Scores for  x = [3, 2]",
           fontsize=12, ha="center", fontweight="bold", color=C_BOUNDARY)

# Each class computation
comps = [
    ("cat:", "w_cat · x + b₀", "[ 2,−1]·[3,2] + 1",
     "= 6 − 2 + 1", "= 5", C_CAT),
    ("dog:", "w_dog · x + b₁", "[ 0, 3]·[3,2] − 1",
     "= 0 + 6 − 1", "= 5", C_DOG),
    ("bird:", "w_bird · x + b₂", "[−1, 1]·[3,2] + 0",
     "= −3 + 2 + 0", "= −1", C_BIRD),
]

for i, (cls, formula, nums, mid, result, color) in enumerate(comps):
    y = 3.5 - i * 1.2
    ax_ex.text(8.5, y, cls, fontsize=11, color=color, fontweight="bold")
    ax_ex.text(10.0, y, nums, fontsize=10, fontfamily="monospace",
               color=MEDIUM)
    ax_ex.text(13.5, y, mid, fontsize=10, fontfamily="monospace",
               color=MEDIUM)
    ax_ex.text(15.0, y, result, fontsize=12, fontfamily="monospace",
               color=color, fontweight="bold")

# Result
ax_ex.add_patch(patches.FancyBboxPatch(
    (0.3, -3.5), 15.4, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_ex.text(8, -0.5, "The Logits (Scores)", fontsize=12,
           ha="center", fontweight="bold", color=C_B)

# Score bar chart
bar_data = [("cat", 5, C_CAT), ("dog", 5, C_DOG), ("bird", -1, C_BIRD)]
bar_x_positions = [3, 8, 13]
max_score = 5
bar_width = 2.5
bar_base = -3.0

for (cls, score, color), bx in zip(bar_data, bar_x_positions):
    bar_h = (score / max_score) * 2.0
    if score > 0:
        ax_ex.add_patch(patches.FancyBboxPatch(
            (bx - bar_width/2, bar_base), bar_width, bar_h,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5))
    else:
        ax_ex.add_patch(patches.FancyBboxPatch(
            (bx - bar_width/2, bar_base + bar_h), bar_width, -bar_h,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.15, edgecolor=color,
            linewidth=1.5, linestyle="--"))

    ax_ex.text(bx, bar_base + bar_h + 0.15 if score > 0 else bar_base + bar_h - 0.35,
               f"z = {score}", fontsize=12, ha="center",
               fontfamily="monospace", color=color, fontweight="bold")
    ax_ex.text(bx, -3.3, cls, fontsize=11, ha="center",
               color=color, fontweight="bold")

# Baseline
ax_ex.plot([1, 15], [bar_base, bar_base], "--",
           color=SUBTLE, lw=1, alpha=0.4)
ax_ex.text(15.2, bar_base, "0", fontsize=9, color=SUBTLE)

# Tie note
ax_ex.text(8, -4.3,
           "cat and dog are tied at z = 5  →  both are equally likely.    "
           "bird scores lowest at z = −1.",
           fontsize=10, ha="center", color=MEDIUM)

# ── Key insight ──
ax_ex.text(8, -5.3,
           "Next step: feed these logits into softmax to get probabilities  "
           "→  then cross-entropy loss for training.",
           fontsize=10, ha="center", color=C_B, fontweight="bold")

# ── What the matrix really is ──
ax_ex.text(8, -6.5,
           "The weight matrix W is just K weight vectors stacked as rows. "
           "Each row asks: \"how much does x look like my class?\"",
           fontsize=10.5, ha="center", color=C_BOUNDARY, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                     edgecolor=C_BOUNDARY, alpha=0.85))


# ─────────────────────────────────────────────────────────
#  ROW 3: The mental model
# ─────────────────────────────────────────────────────────
ax_model = fig.add_subplot(gs[2, :])
ax_model.axis("off")
ax_model.set_xlim(0, 16)
ax_model.set_ylim(-4, 5)

ax_model.text(8, 4.5, "The Mental Model", fontsize=14,
              ha="center", fontweight="bold", color=C_RULE)

# Analogy
ax_model.add_patch(patches.FancyBboxPatch(
    (0.5, 0.5), 15, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_model.text(8, 3.5, "Analogy:  K judges scoring a contestant",
              fontsize=12, ha="center", fontweight="bold", color=C_B)

ax_model.text(8, 2.6,
              "Each class has its own \"judge\" — a weight vector wc and bias bc.",
              fontsize=10.5, ha="center", color=MEDIUM)
ax_model.text(8, 1.9,
              "Each judge looks at the same input x and computes "
              "a score:  zc = wc · x + bc",
              fontsize=10.5, ha="center", color=MEDIUM)
ax_model.text(8, 1.2,
              "The judge who gives the highest score wins — "
              "that class is the prediction.",
              fontsize=10.5, ha="center", color=C_B, fontweight="bold")

# Summary line
ax_model.add_patch(patches.FancyBboxPatch(
    (1.5, -3.0), 13, 2.8, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.10, edgecolor=C_RULE, linewidth=1.5))

summaries = [
    ("Binary:  1 weight vector → 1 score → threshold at 0", True),
    ("Multiclass:  K weight vectors → K scores → pick the largest", True),
    ("The weight matrix W (K × n) is just those K vectors stacked as rows.  "
     "Each row is one class's detector.", False),
]
for i, (txt, bold) in enumerate(summaries):
    ax_model.text(8, -0.7 - i * 0.7, txt, fontsize=10, ha="center",
                  color=C_RULE,
                  fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_3_visuals/multiclass_weights.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
