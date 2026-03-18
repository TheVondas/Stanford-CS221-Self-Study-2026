"""
Lecture 2 Visual: einsum("i ->", v) — Sum All Entries of a Vector

Shows WHY the notation works:
- The input has one axis labeled 'i'
- The output has NO axes (right of -> is empty)
- Any axis that disappears from the output is summed over
- No surviving axes → scalar result
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Style (matches lecture 1 visuals) ─────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_I = "#89b4fa"       # blue:   i axis
C_OUT = "#f9e2af"     # yellow: output / result
C_RULE = "#fab387"    # orange: rule highlights
C_K = "#f38ba8"       # pink:   contracted / summed
C_DIM = "#585b70"     # dimmed

CELL = 0.55
GAP = 0.06
S = CELL + GAP

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Drawing helpers ───────────────────────────────────────

def dc(ax, x, y, val, col, alpha=0.7, fs=13):
    """Draw a single cell."""
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), CELL, CELL, boxstyle="round,pad=0.02",
        facecolor=col, alpha=alpha, edgecolor="white", linewidth=1))
    ax.text(x + CELL / 2, y + CELL / 2, str(val), ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#1e1e2e")


def dvh(ax, vals, x0, y0, col):
    """Horizontal vector."""
    for i, v in enumerate(vals):
        dc(ax, x0 + i * S, y0, v, col)


def arr_h(ax, x0, w, y, col, label, above=True):
    """Horizontal arrow with label."""
    ax.annotate("", xy=(x0 + w + 0.08, y), xytext=(x0 - 0.08, y),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    dy = 0.18 if above else -0.22
    va = "bottom" if above else "top"
    ax.text(x0 + w / 2, y + dy, label, ha="center", fontsize=10,
            color=col, fontweight="bold", va=va)


def insight_box(ax, x, y, lines, w=5.5, line_h=0.45):
    """Rounded box with multi-line text."""
    h = len(lines) * line_h + 0.25
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=C_RULE, alpha=0.10, edgecolor=C_RULE, linewidth=1.5))
    for i, (txt, bold) in enumerate(lines):
        yy = y + h - 0.28 - i * line_h
        ax.text(x + w / 2, yy, txt, ha="center", fontsize=9.5,
                color=C_RULE, fontweight="bold" if bold else "normal")


# ═══════════════════════════════════════════════════════════
#  FIGURE — 3 sections stacked vertically
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(14, 16),
                         gridspec_kw={"height_ratios": [1, 1.6, 1.4],
                                      "hspace": 0.15})
fig.suptitle('einsum("i  →  ") — Sum All Entries of a Vector',
             fontsize=16, fontweight="bold", y=0.97, color=C_OUT)

for a in axes:
    a.axis("off")


# ───────────────────────────────────────────────────────────
#  SECTION 1: The Notation — What Each Part Means
# ───────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 14)
ax.set_ylim(-1.5, 4.5)

ax.text(0.5, 4.0, "Step 1 — Read the Notation", fontsize=14,
        fontweight="bold", color=C_RULE)

# The notation string, broken into labeled parts
notation_y = 2.8
ax.text(2.0, notation_y, 'einsum("', fontsize=14, fontfamily="monospace",
        color=MEDIUM)
ax.text(4.35, notation_y, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(4.85, notation_y, '  ->  ', fontsize=14, fontfamily="monospace",
        color=TEXT)
ax.text(6.8, notation_y, '  ', fontsize=14, fontfamily="monospace",
        color=C_K)
ax.text(7.0, notation_y, '", v)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# Annotation arrows from notation parts
# Arrow from 'i' down to explanation
ax.annotate("input has one axis\nlabeled  i",
            xy=(4.55, notation_y - 0.2), xytext=(2.0, notation_y - 1.4),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5,
                            connectionstyle="arc3,rad=0.2"))

# Arrow from '->' down to explanation
ax.annotate("separates\ninput → output",
            xy=(5.85, notation_y - 0.2), xytext=(5.85, notation_y - 1.4),
            fontsize=10, color=TEXT, ha="center",
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.5))

# Arrow from empty output to explanation
ax.annotate("output has NO axes\n(empty = scalar)",
            xy=(6.9, notation_y - 0.2), xytext=(9.5, notation_y - 1.4),
            fontsize=10, color=C_K, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_K, lw=1.5,
                            connectionstyle="arc3,rad=-0.2"))

# Highlight the empty space after ->
ax.add_patch(patches.FancyBboxPatch(
    (6.65, notation_y - 0.15), 0.5, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_K, alpha=0.15, edgecolor=C_K, linewidth=1.5,
    linestyle="--"))
ax.text(6.9, notation_y + 0.55, "nothing here!", fontsize=9,
        color=C_K, ha="center", fontstyle="italic")


# ───────────────────────────────────────────────────────────
#  SECTION 2: The Mechanics — What Happens Step by Step
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-3.5, 5.5)

ax.text(0.5, 5.0, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

# The vector
v = [3, 7, 2, 5, 1]
vx, vy = 1.0, 3.5
dvh(ax, v, vx, vy, C_I)
ax.text(vx + 2 * S, vy + CELL + 0.2, "v", fontsize=13,
        fontweight="bold", ha="center")
arr_h(ax, vx, 5 * S - GAP, vy - 0.35, C_I, "axis  i  (5 elements)",
      above=False)

# Label each element with its index
for idx, val in enumerate(v):
    ax.text(vx + idx * S + CELL / 2, vy + CELL + 0.55,
            f"i={idx}", fontsize=8, color=C_I, ha="center",
            fontfamily="monospace")

# The key question
ax.text(0.5, 2.3, "Q:  axis  i  is in the input.  Is it in the output?",
        fontsize=12, color=TEXT, fontweight="bold")
ax.text(0.5, 1.7, "A:  No — the right side of  →  is empty.",
        fontsize=12, color=C_K, fontweight="bold")
ax.text(0.5, 1.1, "∴   axis  i  is SUMMED OVER  (it disappears)",
        fontsize=12, color=C_K, fontweight="bold")

# Show the summation visually: all cells flowing into one
# Draw curved arrows from each cell to the result
result_x, result_y = 10.0, 3.5
dc(ax, result_x, result_y, 18, C_OUT, fs=14)
ax.text(result_x + CELL / 2, result_y + CELL + 0.2, "scalar",
        fontsize=10, ha="center", color=C_OUT, fontweight="bold")

for idx in range(5):
    x_start = vx + idx * S + CELL
    y_start = vy + CELL / 2
    rad = -0.15 - idx * 0.05
    ax.annotate("",
                xy=(result_x - 0.05, result_y + CELL / 2),
                xytext=(x_start + 0.05, y_start),
                arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.2,
                                alpha=0.6,
                                connectionstyle=f"arc3,rad={rad}"))

# Plus signs between arrows
ax.text(7.5, 4.6, "all summed together", fontsize=10,
        color=C_OUT, ha="center", fontstyle="italic")

# Computation trace
ty = -0.2
ax.text(0.5, ty, "The computation:", fontsize=12, fontweight="bold",
        color=MEDIUM)

# Step by step addition
ax.text(0.5, ty - 0.7, "output", fontsize=11, fontfamily="monospace",
        color=C_OUT, fontweight="bold")
ax.text(2.3, ty - 0.7, "=  Σ  v[i]", fontsize=11, fontfamily="monospace",
        color=TEXT)
ax.text(5.5, ty - 0.7, "(sum over the vanished axis  i )",
        fontsize=10, color=C_K, fontstyle="italic")

ax.text(2.3, ty - 1.3, "=  v[0] + v[1] + v[2] + v[3] + v[4]",
        fontsize=11, fontfamily="monospace", color=MEDIUM)

ax.text(2.3, ty - 1.9, "=   3   +   7   +   2   +   5   +   1",
        fontsize=11, fontfamily="monospace", color=MEDIUM)

ax.text(2.3, ty - 2.5, "=  18", fontsize=13, fontfamily="monospace",
        color=C_OUT, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: The Rule — Why This Works
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 14)
ax.set_ylim(-2.5, 5)

ax.text(0.5, 4.5, "Step 3 — The General Rule", fontsize=14,
        fontweight="bold", color=C_RULE)

# Side-by-side comparison: what's on the left vs right of ->
left_x = 0.8
# Input side
ax.add_patch(patches.FancyBboxPatch(
    (left_x, 1.5), 5.2, 2.5, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.08, edgecolor=C_I, linewidth=1.5))
ax.text(left_x + 2.6, 3.7, "LEFT of  →  (input axes)", fontsize=11,
        ha="center", fontweight="bold", color=C_I)
ax.text(left_x + 2.6, 3.1, "i", fontsize=24, ha="center",
        fontweight="bold", color=C_I, fontfamily="monospace")
ax.text(left_x + 2.6, 2.3, "\"the vector has one axis\"",
        fontsize=10, ha="center", color=C_I, fontstyle="italic")
ax.text(left_x + 2.6, 1.8, "this tells einsum the input shape",
        fontsize=9, ha="center", color=MEDIUM)

# Output side
right_x = 7.5
ax.add_patch(patches.FancyBboxPatch(
    (right_x, 1.5), 5.2, 2.5, boxstyle="round,pad=0.15",
    facecolor=C_K, alpha=0.08, edgecolor=C_K, linewidth=1.5))
ax.text(right_x + 2.6, 3.7, "RIGHT of  →  (output axes)", fontsize=11,
        ha="center", fontweight="bold", color=C_K)
ax.text(right_x + 2.6, 3.1, "(empty)", fontsize=18, ha="center",
        fontweight="bold", color=C_K, fontfamily="monospace")
ax.text(right_x + 2.6, 2.3, "\"the output has no axes\"",
        fontsize=10, ha="center", color=C_K, fontstyle="italic")
ax.text(right_x + 2.6, 1.8, "= a scalar (just one number)",
        fontsize=9, ha="center", color=MEDIUM)

# The deduction arrow
ax.annotate("", xy=(6.9, 2.75), xytext=(6.1, 2.75),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

# The conclusion
insight_box(ax, 1.5, -1.8, [
    ("i  is on the LEFT  but not on the RIGHT", True),
    ("→  axis  i  must be eliminated", True),
    ("→  the only way to eliminate an axis is to SUM over it", False),
    ("→  result: one number = the sum of all entries", False),
], w=11, line_h=0.5)

ax.text(7.0, -0.1, "Missing axis  =  summed axis", fontsize=13,
        ha="center", fontweight="bold", color=C_OUT)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_vector_sum.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
