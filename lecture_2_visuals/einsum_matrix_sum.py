"""
Lecture 2 Visual: einsum("i j ->", M) — Sum All Entries of a Matrix

Shows WHY the notation works:
- The input has two axes: i (rows) and j (columns)
- The output has NO axes (right of -> is empty)
- Both i and j disappear → both are summed over
- Result is a scalar: every entry accumulated into one number
- Natural extension of vector sum (i ->) to two dimensions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Style ─────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_I = "#89b4fa"       # blue:   i axis (rows)
C_J = "#a6e3a1"       # green:  j axis (columns)
C_OUT = "#f9e2af"     # yellow: output / result
C_RULE = "#fab387"    # orange: rule highlights
C_K = "#f38ba8"       # pink:   contracted / summed
C_DIM = "#585b70"     # dimmed
C_WARN = "#f5c2e7"    # light pink: callouts

CELL = 0.55
GAP = 0.06
S = CELL + GAP

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Drawing helpers ───────────────────────────────────────

def dc(ax, x, y, val, col, alpha=0.7, fs=13):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), CELL, CELL, boxstyle="round,pad=0.02",
        facecolor=col, alpha=alpha, edgecolor="white", linewidth=1))
    ax.text(x + CELL / 2, y + CELL / 2, str(val), ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#1e1e2e")


def dm(ax, data, x0, y0, col=None, cc=None):
    R, C_ = data.shape
    for r in range(R):
        for c in range(C_):
            color = col if cc is None else cc[r][c]
            dc(ax, x0 + c * S, y0 - r * S, data[r, c], color)


def arr_h(ax, x0, w, y, col, label, above=True):
    ax.annotate("", xy=(x0 + w + 0.08, y), xytext=(x0 - 0.08, y),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    dy = 0.18 if above else -0.22
    va = "bottom" if above else "top"
    ax.text(x0 + w / 2, y + dy, label, ha="center", fontsize=10,
            color=col, fontweight="bold", va=va)


def arr_v(ax, x, y_top, h, col, label):
    ax.annotate("", xy=(x, y_top - h - 0.08), xytext=(x, y_top + 0.08),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax.text(x - 0.22, y_top - h / 2, label, ha="center", va="center",
            fontsize=10, color=col, fontweight="bold", rotation=90)


def insight_box(ax, x, y, lines, w=5.5, line_h=0.45):
    h = len(lines) * line_h + 0.25
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=C_RULE, alpha=0.10, edgecolor=C_RULE, linewidth=1.5))
    for i, (txt, bold) in enumerate(lines):
        yy = y + h - 0.28 - i * line_h
        ax.text(x + w / 2, yy, txt, ha="center", fontsize=9.5,
                color=C_RULE, fontweight="bold" if bold else "normal")


# ═══════════════════════════════════════════════════════════
#  FIGURE — 3 sections
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(14, 20),
                         gridspec_kw={"height_ratios": [1, 2.4, 1.4],
                                      "hspace": 0.12})
fig.suptitle('einsum("i j  →  ") — Sum All Entries of a Matrix',
             fontsize=16, fontweight="bold", y=0.97, color=C_OUT)

for a in axes:
    a.axis("off")


# ───────────────────────────────────────────────────────────
#  SECTION 1: Read the Notation
# ───────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 14)
ax.set_ylim(-1.5, 4.5)

ax.text(0.5, 4.0, "Step 1 — Read the Notation", fontsize=14,
        fontweight="bold", color=C_RULE)

ny = 2.8
ax.text(1.2, ny, 'einsum("', fontsize=14, fontfamily="monospace",
        color=MEDIUM)
ax.text(3.55, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(4.0, ny, ' ', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(4.15, ny, 'j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(4.7, ny, '  ->  ', fontsize=14, fontfamily="monospace",
        color=TEXT)
ax.text(7.0, ny, '", M)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# Highlight empty output
ax.add_patch(patches.FancyBboxPatch(
    (6.55, ny - 0.15), 0.55, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_K, alpha=0.15, edgecolor=C_K, linewidth=1.5,
    linestyle="--"))
ax.text(6.82, ny + 0.55, "nothing here!", fontsize=9,
        color=C_K, ha="center", fontweight="bold", fontstyle="italic")

# Annotation: i
ax.annotate("rows\n(axis  i )",
            xy=(3.7, ny - 0.2), xytext=(1.8, ny - 1.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5,
                            connectionstyle="arc3,rad=0.2"))

# Annotation: j
ax.annotate("columns\n(axis  j )",
            xy=(4.3, ny - 0.2), xytext=(4.3, ny - 1.5),
            fontsize=10, color=C_J, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.5))

# Annotation: empty output
ax.annotate("output has NO axes\n= scalar",
            xy=(6.82, ny - 0.2), xytext=(9.5, ny - 1.5),
            fontsize=10, color=C_K, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_K, lw=1.5,
                            connectionstyle="arc3,rad=-0.2"))


# ───────────────────────────────────────────────────────────
#  SECTION 2: What Happens
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-5.5, 7)

ax.text(0.5, 6.5, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

M = np.array([[1, 4, 2],
              [5, 3, 6],
              [2, 1, 3]])
total = int(M.sum())  # 27

# ── Draw the matrix ──
mx, my = 1.5, 4.5
dm(ax, M, mx, my, C_K)
ax.text(mx + 1 * S, my + CELL + 0.55, "M", fontsize=13,
        fontweight="bold", ha="center")
arr_h(ax, mx, 3 * S - GAP, my + CELL + 0.15, C_J, "j  (columns)")
arr_v(ax, mx - 0.45, my + CELL, 3 * S - GAP, C_I, "i  (rows)")

# Index labels
for idx in range(3):
    ax.text(mx + idx * S + CELL / 2, my + CELL + 0.9,
            f"j={idx}", fontsize=8, color=C_J, ha="center",
            fontfamily="monospace")
    ax.text(mx - 0.85, my - idx * S + CELL / 2,
            f"i={idx}", fontsize=8, color=C_I, ha="center",
            fontfamily="monospace", va="center")

# ── Curved arrows from every cell to the result ──
result_x, result_y = 8.5, my - 1 * S
dc(ax, result_x, result_y, total, C_OUT, fs=16)
ax.text(result_x + CELL / 2, result_y + CELL + 0.25, "scalar",
        fontsize=11, ha="center", color=C_OUT, fontweight="bold")

for r in range(3):
    for c in range(3):
        cx = mx + c * S + CELL
        cy = my - r * S + CELL / 2
        # Vary the arc radius so arrows don't overlap
        rad = -0.05 - r * 0.08 - c * 0.04
        ax.annotate("",
                    xy=(result_x - 0.05, result_y + CELL / 2),
                    xytext=(cx + 0.05, cy),
                    arrowprops=dict(arrowstyle="->", color=C_OUT,
                                    lw=1.0, alpha=0.45,
                                    connectionstyle=f"arc3,rad={rad}"))

ax.text((mx + 3 * S + result_x) / 2 + 0.5, my + CELL + 0.5,
        "ALL 9 entries\nflow into one sum",
        fontsize=10, color=C_OUT, ha="center", fontweight="bold")

# ── The algorithm ──
ax.text(0.5, 1.8, "The algorithm:", fontsize=12,
        fontweight="bold", color=TEXT)

py = 1.1
ax.text(1.2, py,
        "output = 0", fontsize=10.5, fontfamily="monospace", color=SUBTLE)
ax.text(1.2, py - 0.55,
        "for each  i :", fontsize=10.5, fontfamily="monospace", color=C_I)
ax.text(1.2, py - 1.1,
        "    for each  j :", fontsize=10.5, fontfamily="monospace", color=C_J)
ax.text(1.2, py - 1.65,
        "        output  +=  M[i, j]", fontsize=12, fontfamily="monospace",
        color=TEXT, fontweight="bold")

# Annotate: only one input → no × needed, just +=
ax.annotate("one input → nothing to multiply\njust accumulate with  +=",
            xy=(6.5, py - 1.65 + 0.1), xytext=(7.5, py - 0.5),
            fontsize=10, color=C_OUT, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.3,
                            connectionstyle="arc3,rad=-0.2"))

# ── Key point: both axes vanish ──
ax.text(0.5, py - 2.7,
        "Both  i  and  j  are missing from the output  →  both summed over",
        fontsize=11, fontweight="bold", color=C_K)
ax.text(0.5, py - 3.3,
        "Two nested loops, each accumulating  →  every cell visited  →  total sum",
        fontsize=10.5, color=MEDIUM)

# ── Computation trace ──
ty = py - 4.2
ax.text(0.5, ty, "The full computation:", fontsize=12,
        fontweight="bold", color=MEDIUM)

ax.text(0.5, ty - 0.6, "output", fontsize=11, fontfamily="monospace",
        color=C_OUT, fontweight="bold")
ax.text(2.3, ty - 0.6, "=  Σ_i  Σ_j  M[i, j]", fontsize=11,
        fontfamily="monospace", color=TEXT)

ax.text(2.3, ty - 1.2,
        "=  (1 + 4 + 2)  +  (5 + 3 + 6)  +  (2 + 1 + 3)",
        fontsize=10.5, fontfamily="monospace", color=MEDIUM)

# Color-code the row sums
ax.text(2.3, ty - 1.8,
        "=      7       +      14      +      6",
        fontsize=10.5, fontfamily="monospace", color=MEDIUM)
ax.text(3.05, ty - 2.15, "row 0", fontsize=8, color=C_I,
        fontfamily="monospace")
ax.text(6.3, ty - 2.15, "row 1", fontsize=8, color=C_I,
        fontfamily="monospace")
ax.text(9.3, ty - 2.15, "row 2", fontsize=8, color=C_I,
        fontfamily="monospace")

ax.text(2.3, ty - 2.6, "=  27", fontsize=14, fontfamily="monospace",
        color=C_OUT, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: The General Rule + Connection to Vector Sum
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 14)
ax.set_ylim(-3.5, 5.5)

ax.text(0.5, 5.0, "Step 3 — The General Rule", fontsize=14,
        fontweight="bold", color=C_RULE)

# Three-column layout
c1x = 0.5
ax.add_patch(patches.FancyBboxPatch(
    (c1x, 2.2), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.08, edgecolor=C_I, linewidth=1.5))
ax.text(c1x + 1.9, 4.2, "INPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_I)
ax.text(c1x + 1.9, 3.55, "i  j", fontsize=22, ha="center",
        fontweight="bold", color=C_I, fontfamily="monospace")
ax.text(c1x + 1.9, 2.8, "matrix has two axes",
        fontsize=9.5, ha="center", color=C_I)
ax.text(c1x + 1.9, 2.4, "i = rows,  j = columns",
        fontsize=9.5, ha="center", color=MEDIUM)

ax.annotate("", xy=(4.8, 3.4), xytext=(4.4, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c2x = 5.0
ax.add_patch(patches.FancyBboxPatch(
    (c2x, 2.2), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_K, alpha=0.08, edgecolor=C_K, linewidth=1.5))
ax.text(c2x + 1.9, 4.2, "OUTPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_K)
ax.text(c2x + 1.9, 3.55, "(empty)", fontsize=18, ha="center",
        fontweight="bold", color=C_K, fontfamily="monospace")
ax.text(c2x + 1.9, 2.8, "BOTH  i  and  j  missing",
        fontsize=9.5, ha="center", color=C_K, fontweight="bold")
ax.text(c2x + 1.9, 2.4, "= both axes SUMMED OVER",
        fontsize=9.5, ha="center", color=C_K)

ax.annotate("", xy=(9.3, 3.4), xytext=(8.9, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c3x = 9.5
ax.add_patch(patches.FancyBboxPatch(
    (c3x, 2.2), 4.0, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax.text(c3x + 2.0, 4.2, "RESULT", fontsize=11,
        ha="center", fontweight="bold", color=C_RULE)
ax.text(c3x + 2.0, 3.55, "Σ_i Σ_j M[i,j]", fontsize=13, ha="center",
        fontweight="bold", color=C_RULE, fontfamily="monospace")
ax.text(c3x + 2.0, 2.8, "sum every entry",
        fontsize=9.5, ha="center", color=C_RULE)
ax.text(c3x + 2.0, 2.4, "output is a scalar",
        fontsize=9.5, ha="center", color=MEDIUM)

# ── Connection to vector sum ──
ax.text(7.0, 1.5, "Same Pattern as Vector Sum — Just More Axes to Remove",
        fontsize=11.5, ha="center", fontweight="bold", color=C_OUT)

# Vector sum row
ax.text(1.0, 0.6, "i  →     ", fontsize=12, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(4.0, 0.6, "1 axis vanishes", fontsize=10, color=C_I)
ax.text(7.5, 0.6, "→  Σ_i  v[i]", fontsize=10, fontfamily="monospace",
        color=MEDIUM)
ax.text(11.0, 0.6, "→  scalar", fontsize=10, color=C_OUT,
        fontweight="bold")

# Matrix sum row
ax.text(1.0, -0.2, "i j →   ", fontsize=12, fontfamily="monospace",
        color=C_K, fontweight="bold")
ax.text(4.0, -0.2, "2 axes vanish", fontsize=10, color=C_K)
ax.text(7.5, -0.2, "→  Σ_i Σ_j  M[i,j]", fontsize=10,
        fontfamily="monospace", color=MEDIUM)
ax.text(11.0, -0.2, "→  scalar", fontsize=10, color=C_OUT,
        fontweight="bold")

ax.plot([1.0, 13.0], [0.2, 0.2], "-", color=SUBTLE, alpha=0.3, lw=1)

# Bottom insight
insight_box(ax, 1.0, -2.8, [
    ("Every axis missing from the output gets its own summation loop", True),
    ("1 axis missing = 1 sum (vector sum)     2 axes missing = 2 sums (matrix sum)", False),
    ("The rule scales to any number of dimensions:  i j k →   sums a 3D tensor to a scalar", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_matrix_sum.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
