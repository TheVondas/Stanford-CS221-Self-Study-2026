"""
Lecture 3 Visual: Basic Linear Classification

Shows:
1. Four data points — two labeled +1, two labeled −1
2. The decision boundary: the line where w · x + b = 0
3. Annotations connecting the score function z = w · x + b to the boundary
4. Shaded regions showing which side is predicted +1 vs −1
5. A summary panel explaining the core idea
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec

# ── Style (matching existing visuals) ────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_POS = "#a6e3a1"        # green: +1 class
C_NEG = "#f38ba8"         # pink: −1 class
C_BOUNDARY = "#f9e2af"    # yellow: decision boundary
C_RULE = "#fab387"        # orange: insight boxes
C_I = "#89b4fa"           # blue: math / annotations
C_B = "#cba6f7"           # purple
C_DIM = "#585b70"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Data ─────────────────────────────────────────────────
# Four simple points in 2D, linearly separable
points = np.array([
    [1.0, 3.0],   # +1
    [2.0, 4.5],   # +1
    [4.0, 1.5],   # −1
    [5.0, 2.5],   # −1
])
labels = np.array([+1, +1, -1, -1])

# ── Classifier parameters (chosen to separate cleanly) ───
# Decision boundary: w · x + b = 0
# We want a line that separates left-upper (+1) from right-lower (−1)
# Line: x1 + x2 - 5.5 = 0  →  w = [1, 1], b = -5.5
w = np.array([1.0, 1.0])
b = -5.5


def score(x):
    """Compute z = w · x + b"""
    return np.dot(w, x) + b


def insight_box(ax, x, y, lines, w_box=5.5, line_h=0.45):
    h = len(lines) * line_h + 0.25
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w_box, h, boxstyle="round,pad=0.12",
        facecolor=C_RULE, alpha=0.10, edgecolor=C_RULE, linewidth=1.5))
    for i, (txt, bold) in enumerate(lines):
        yy = y + h - 0.28 - i * line_h
        ax.text(x + w_box / 2, yy, txt, ha="center", fontsize=9.5,
                color=C_RULE, fontweight="bold" if bold else "normal")


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 28))
fig.suptitle("Linear Classification: The Core Idea",
             fontsize=17, fontweight="bold", y=0.995, color=C_BOUNDARY)

gs = GridSpec(4, 2, figure=fig,
              height_ratios=[3.0, 1.8, 2.2, 1.2],
              hspace=0.25, wspace=0.25,
              top=0.97, bottom=0.02, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: The classification plot
# ─────────────────────────────────────────────────────────
ax_main = fig.add_subplot(gs[0, 0])

# Shaded regions
x_fill = np.linspace(-0.5, 7, 300)
# Decision boundary line: x1 + x2 = 5.5  →  x2 = 5.5 - x1
boundary_y = 5.5 - x_fill

# +1 region (above the line)
ax_main.fill_between(x_fill, boundary_y, 7,
                     alpha=0.08, color=C_POS, label="_nolegend_")
# −1 region (below the line)
ax_main.fill_between(x_fill, -0.5, boundary_y,
                     alpha=0.08, color=C_NEG, label="_nolegend_")

# Decision boundary line
x_line = np.linspace(-0.5, 7, 100)
y_line = 5.5 - x_line
ax_main.plot(x_line, y_line, color=C_BOUNDARY, lw=2.5, linestyle="--",
             label="Decision boundary", zorder=3)

# Region labels
ax_main.text(0.8, 5.8, "Predict  +1", fontsize=13, fontweight="bold",
             color=C_POS, alpha=0.7)
ax_main.text(0.8, 5.2, "w · x + b > 0", fontsize=10,
             color=C_POS, alpha=0.5, fontfamily="monospace")
ax_main.text(5.0, 0.3, "Predict  −1", fontsize=13, fontweight="bold",
             color=C_NEG, alpha=0.7)
ax_main.text(5.0, -0.3, "w · x + b < 0", fontsize=10,
             color=C_NEG, alpha=0.5, fontfamily="monospace")

# Plot points
for i, (pt, lbl) in enumerate(zip(points, labels)):
    color = C_POS if lbl == +1 else C_NEG
    marker = "^" if lbl == +1 else "v"
    label_str = f"y = +1" if lbl == +1 else f"y = −1"
    # Only add legend label for first of each class
    show_label = label_str if i in [0, 2] else "_nolegend_"
    ax_main.scatter(pt[0], pt[1], c=color, s=180, marker=marker,
                    edgecolors="white", linewidths=1.5, zorder=5,
                    label=show_label)

    # Score annotation
    z = score(pt)
    sign_str = "+" if z > 0 else ""
    ax_main.annotate(
        f"({pt[0]:.0f}, {pt[1]:.0f})\nz = {sign_str}{z:.1f}",
        xy=(pt[0], pt[1]),
        xytext=(pt[0] + (1.2 if lbl == -1 else -1.8),
                pt[1] + (0.8 if i % 2 == 0 else -0.8)),
        fontsize=9, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=color, alpha=0.8),
        zorder=6)

# Boundary label
ax_main.annotate(
    "w · x + b = 0\n(decision boundary)",
    xy=(2.5, 3.0), xytext=(4.5, 5.2),
    fontsize=10, color=C_BOUNDARY, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_BOUNDARY, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
              edgecolor=C_BOUNDARY, alpha=0.8),
    zorder=6)

ax_main.set_xlim(-0.5, 7)
ax_main.set_ylim(-0.5, 7)
ax_main.set_xlabel("$x_1$", fontsize=12, color=C_I)
ax_main.set_ylabel("$x_2$", fontsize=12, color=C_I)
ax_main.set_title("Four Points, One Boundary",
                  fontsize=14, fontweight="bold", color=C_RULE, pad=12)
ax_main.legend(fontsize=9.5, facecolor=BG, edgecolor=C_DIM,
               labelcolor=TEXT, loc="lower left")
ax_main.tick_params(colors=SUBTLE, labelsize=9)
ax_main.set_aspect("equal")
for spine in ax_main.spines.values():
    spine.set_color(C_DIM)


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: How the score function works
# ─────────────────────────────────────────────────────────
ax_how = fig.add_subplot(gs[0, 1])
ax_how.axis("off")
ax_how.set_xlim(0, 10)
ax_how.set_ylim(-1, 12)

ax_how.text(5, 11.5, "How It Works", fontsize=14,
            ha="center", fontweight="bold", color=C_RULE)

# Step 1: The score function
ax_how.add_patch(patches.FancyBboxPatch(
    (0.5, 8.5), 9, 2.5, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))

ax_how.text(5, 10.5, "Step 1:  Compute the score (logit)",
            fontsize=12, ha="center", fontweight="bold", color=C_I)
ax_how.text(5, 9.7, "z  =  w · x  +  b",
            fontsize=14, ha="center", fontfamily="monospace",
            fontweight="bold", color=C_BOUNDARY)
ax_how.text(5, 9.0,
            "w = [1, 1]     b = −5.5",
            fontsize=10.5, ha="center", fontfamily="monospace",
            color=MEDIUM)

# Step 2: Check the sign
ax_how.add_patch(patches.FancyBboxPatch(
    (0.5, 5.5), 9, 2.5, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.5))

ax_how.text(5, 7.5, "Step 2:  Check the sign of z",
            fontsize=12, ha="center", fontweight="bold", color=C_BOUNDARY)
ax_how.text(5, 6.7, "z > 0   →   predict +1",
            fontsize=11, ha="center", fontfamily="monospace", color=C_POS)
ax_how.text(5, 6.05, "z < 0   →   predict −1",
            fontsize=11, ha="center", fontfamily="monospace", color=C_NEG)

# Concrete examples
ax_how.add_patch(patches.FancyBboxPatch(
    (0.5, 0.2), 9, 4.8, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_how.text(5, 4.5, "Concrete Examples from the Plot",
            fontsize=12, ha="center", fontweight="bold", color=C_B)

examples = [
    (points[0], labels[0], C_POS),
    (points[1], labels[1], C_POS),
    (points[2], labels[2], C_NEG),
    (points[3], labels[3], C_NEG),
]

for idx, (pt, lbl, color) in enumerate(examples):
    z = score(pt)
    sign_str = "+" if z > 0 else ""
    pred = "+1" if z > 0 else "−1"
    correct = "✓" if (z > 0 and lbl == 1) or (z < 0 and lbl == -1) else "✗"
    lbl_str = "+1" if lbl == 1 else "−1"

    y_pos = 3.6 - idx * 0.8
    ax_how.text(1.0, y_pos,
                f"x = ({pt[0]:.0f}, {pt[1]:.0f})",
                fontsize=10, fontfamily="monospace", color=MEDIUM)
    ax_how.text(4.0, y_pos,
                f"z = {sign_str}{z:.1f}",
                fontsize=10, fontfamily="monospace", color=color,
                fontweight="bold")
    ax_how.text(6.2, y_pos,
                f"predict {pred}",
                fontsize=10, fontfamily="monospace", color=color)
    ax_how.text(8.3, y_pos,
                f"true: {lbl_str}  {correct}",
                fontsize=10, fontfamily="monospace", color=color,
                fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 2: What IS the decision boundary?
# ─────────────────────────────────────────────────────────
ax_explain = fig.add_subplot(gs[1, :])
ax_explain.axis("off")
ax_explain.set_xlim(0, 16)
ax_explain.set_ylim(-3, 4)

ax_explain.text(8, 3.5,
                "What Is the Decision Boundary?",
                fontsize=14, ha="center", fontweight="bold", color=C_BOUNDARY)

# Left box: the equation
ax_explain.add_patch(patches.FancyBboxPatch(
    (0.5, -0.3), 7, 3.2, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.5))

ax_explain.text(4, 2.5, "The boundary is the set of all x where:",
                fontsize=11, ha="center", color=TEXT)
ax_explain.text(4, 1.6, "w · x  +  b  =  0",
                fontsize=14, ha="center", fontfamily="monospace",
                fontweight="bold", color=C_BOUNDARY)
ax_explain.text(4, 0.7,
                "In our example:  x₁ + x₂ − 5.5 = 0",
                fontsize=11, ha="center", fontfamily="monospace", color=MEDIUM)
ax_explain.text(4, 0.1,
                "which rearranges to:  x₂ = 5.5 − x₁",
                fontsize=11, ha="center", fontfamily="monospace", color=MEDIUM)

# Right box: intuition
ax_explain.add_patch(patches.FancyBboxPatch(
    (8.5, -0.3), 7, 3.2, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_explain.text(12, 2.5, "Intuition: a fence between two fields",
                fontsize=11, ha="center", fontweight="bold", color=C_B)
ax_explain.text(12, 1.7,
                "The boundary is a fence that separates",
                fontsize=10.5, ha="center", color=MEDIUM)
ax_explain.text(12, 1.1,
                "the +1 field from the −1 field.",
                fontsize=10.5, ha="center", color=MEDIUM)
ax_explain.text(12, 0.3,
                "Points on the fence are exactly undecided.",
                fontsize=10.5, ha="center", color=C_B, fontweight="bold")

# Dimension note
ax_explain.text(8, -1.5,
                "In 2D → the boundary is a line.     "
                "In 3D → a plane.     "
                "In higher dimensions → a hyperplane.",
                fontsize=10, ha="center", color=SUBTLE, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 3: The role of w and b
# ─────────────────────────────────────────────────────────
ax_wb = fig.add_subplot(gs[2, :])
ax_wb.axis("off")
ax_wb.set_xlim(0, 16)
ax_wb.set_ylim(-4, 5)

ax_wb.text(8, 4.5, "What Do w and b Control?",
           fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# w box
ax_wb.add_patch(patches.FancyBboxPatch(
    (0.5, 0.8), 7, 3.2, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.5))

ax_wb.text(4, 3.5, "w  (weight vector)", fontsize=13,
           ha="center", fontweight="bold", color=C_POS)
ax_wb.text(4, 2.6, "Controls the TILT of the boundary",
           fontsize=11, ha="center", color=TEXT)
ax_wb.text(4, 1.8,
           "w is perpendicular to the boundary line.",
           fontsize=10, ha="center", color=MEDIUM)
ax_wb.text(4, 1.2,
           "It points toward the +1 side.",
           fontsize=10, ha="center", color=C_POS, fontweight="bold")

# b box
ax_wb.add_patch(patches.FancyBboxPatch(
    (8.5, 0.8), 7, 3.2, boxstyle="round,pad=0.15",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1.5))

ax_wb.text(12, 3.5, "b  (bias)", fontsize=13,
           ha="center", fontweight="bold", color=C_NEG)
ax_wb.text(12, 2.6, "Controls the POSITION of the boundary",
           fontsize=11, ha="center", color=TEXT)
ax_wb.text(12, 1.8,
           "Shifts the line toward or away from",
           fontsize=10, ha="center", color=MEDIUM)
ax_wb.text(12, 1.2,
           "the origin without changing its angle.",
           fontsize=10, ha="center", color=MEDIUM)

# Together
ax_wb.text(8, -0.2,
           "Together: w sets the angle, b slides the fence along the w direction",
           fontsize=11, ha="center", color=C_BOUNDARY, fontweight="bold")

# The three questions
ax_wb.add_patch(patches.FancyBboxPatch(
    (1.0, -3.5), 14, 2.5, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.06, edgecolor=C_RULE, linewidth=1.5))

ax_wb.text(8, -1.5, "The Classification Learning Problem",
           fontsize=12, ha="center", fontweight="bold", color=C_RULE)
ax_wb.text(8, -2.15,
           "Given training points with labels, find w and b such that the boundary",
           fontsize=10.5, ha="center", color=MEDIUM)
ax_wb.text(8, -2.7,
           "separates the +1 points from the −1 points as well as possible.",
           fontsize=10.5, ha="center", color=MEDIUM)
ax_wb.text(8, -3.3,
           "That is the entire goal of linear classification.",
           fontsize=10.5, ha="center", color=C_RULE, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 4: Summary
# ─────────────────────────────────────────────────────────
ax_sum = fig.add_subplot(gs[3, :])
ax_sum.axis("off")
ax_sum.set_xlim(0, 16)
ax_sum.set_ylim(-3, 3)

ax_sum.text(8, 2.5, "Summary", fontsize=14, ha="center",
            fontweight="bold", color=C_RULE)

insight_box(ax_sum, 2.0, -2.5, [
    ("A linear classifier computes a score:  z = w · x + b", True),
    ("If z > 0 → predict +1.    If z < 0 → predict −1.", True),
    ("The decision boundary is the line/plane where z = 0 — "
     "the classifier is exactly undecided.", False),
    ("Learning = finding w and b that place the boundary so it "
     "correctly separates the training data.", False),
], w_box=12, line_h=0.55)


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_3_visuals/linear_classification_basics.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
