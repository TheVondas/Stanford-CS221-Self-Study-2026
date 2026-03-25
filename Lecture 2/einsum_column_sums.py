"""
Lecture 2 Visual: einsum("i j -> j", M) — Column Sums

Shows WHY the notation works:
- The input has two axes: i (rows) and j (columns)
- The output keeps j but drops i
- i disappears → i is summed over → rows collapse
- j survives → one entry per column → result is a vector
- Mirror of row sums: swap which label survives, swap the operation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Style ─────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_I = "#89b4fa"       # blue:   i axis (rows) — SUMMED
C_J = "#a6e3a1"       # green:  j axis (columns) — KEPT
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


def dvh(ax, vals, x0, y0, col):
    for i, v in enumerate(vals):
        dc(ax, x0 + i * S, y0, v, col)


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
fig, axes = plt.subplots(3, 1, figsize=(14, 22),
                         gridspec_kw={"height_ratios": [1, 2.6, 1.6],
                                      "hspace": 0.12})
fig.suptitle('einsum("i j  →  j") — Column Sums',
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
ax.text(6.85, ny, 'j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(7.3, ny, '", M)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# Highlight output 'j'
ax.add_patch(patches.FancyBboxPatch(
    (6.7, ny - 0.15), 0.5, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_J, alpha=0.15, edgecolor=C_J, linewidth=1.5,
    linestyle="--"))

# Annotation: i on input side
ax.annotate("rows (axis  i )\nonly on LEFT side",
            xy=(3.7, ny - 0.2), xytext=(1.3, ny - 1.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5,
                            connectionstyle="arc3,rad=0.2"))

# Annotation: j on input side
ax.annotate("columns (axis  j )\npresent on BOTH sides",
            xy=(4.3, ny - 0.2), xytext=(4.3, ny - 1.5),
            fontsize=10, color=C_J, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.5))

# Annotation: output j
ax.annotate("j  KEPT → one output\nentry per column",
            xy=(6.95, ny - 0.2), xytext=(9.5, ny - 0.5),
            fontsize=10, color=C_J, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.5,
                            connectionstyle="arc3,rad=-0.15"))

# i missing callout
ax.text(9.5, ny - 1.4, "i  MISSING → summed over\n(rows collapse)",
        fontsize=10, color=C_I, fontweight="bold", ha="center")


# ───────────────────────────────────────────────────────────
#  SECTION 2: What Happens — column-by-column summation
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-5.5, 7)

ax.text(0.5, 6.5, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

M = np.array([[1, 4, 2],
              [5, 3, 6],
              [2, 1, 3]])
col_sums = M.sum(axis=0)  # [8, 8, 11]

# ── Draw the matrix with columns color-coded ──
mx, my = 2.5, 4.5
col_colors = [
    ["#7ec89e", "#8fd4af", "#74c296"],  # col 0, col 1, col 2
    ["#7ec89e", "#8fd4af", "#74c296"],
    ["#7ec89e", "#8fd4af", "#74c296"],
]
dm(ax, M, mx, my, cc=col_colors)

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

# ── Result vector (horizontal, below the matrix) ──
result_y = my - 3 * S - 1.2
dvh(ax, col_sums.tolist(), mx, result_y, C_OUT)
ax.text(mx - 1.0, result_y + CELL / 2, "result", fontsize=11,
        fontweight="bold", ha="center", va="center", color=C_OUT)
arr_h(ax, mx, 3 * S - GAP, result_y - 0.35, C_J, "j  (kept)",
      above=False)

# ── Arrows: each column flows down into its result entry ──
for c in range(3):
    col_x = mx + c * S + CELL / 2
    # "+" signs between matrix cells in this column
    for r in range(2):
        py_plus = my - r * S - (GAP / 2)
        ax.text(col_x, py_plus, "+", fontsize=11, color=C_OUT,
                ha="center", va="center", fontweight="bold")
    # Arrow from bottom of column to result cell
    y_start = my - 2 * S - 0.15
    y_end = result_y + CELL + 0.1
    ax.annotate("",
                xy=(col_x, y_end),
                xytext=(col_x, y_start),
                arrowprops=dict(arrowstyle="->", color=C_OUT, lw=2,
                                alpha=0.7))

# Labels on result
for idx in range(3):
    ax.text(mx + idx * S + CELL / 2, result_y - 0.65,
            f"= {int(col_sums[idx])}", fontsize=12, color=C_OUT,
            fontweight="bold", ha="center")

# ── Visual: i axis collapses ──
bracket_x = mx + 3 * S + 0.3
ax.annotate("", xy=(bracket_x, my + CELL),
            xytext=(bracket_x, my - 2 * S),
            arrowprops=dict(arrowstyle="<->", color=C_I, lw=1.5))
ax.text(bracket_x + 0.3, my - 0.5 * S,
        "i  axis\ncollapses\n(summed\naway)",
        fontsize=9, color=C_I, fontweight="bold", va="center")

# ── The algorithm ──
algo_x = 7.5
ax.text(algo_x, 4.5, "The algorithm:", fontsize=12,
        fontweight="bold", color=TEXT)

py = 3.8
ax.text(algo_x + 0.3, py,
        "for each  j :", fontsize=10.5, fontfamily="monospace", color=C_J)
ax.text(algo_x + 0.3, py - 0.55,
        "    output[j] = 0", fontsize=10.5, fontfamily="monospace",
        color=SUBTLE)
ax.text(algo_x + 0.3, py - 1.1,
        "    for each  i :", fontsize=10.5, fontfamily="monospace",
        color=C_I)
ax.text(algo_x + 0.3, py - 1.65,
        "        output[j]  +=  M[i, j]",
        fontsize=11, fontfamily="monospace",
        color=TEXT, fontweight="bold")

# Annotate
ax.text(algo_x + 0.3, py - 2.5,
        "j  loop is outer → one slot per column",
        fontsize=9.5, color=C_J, fontweight="bold")
ax.text(algo_x + 0.3, py - 3.0,
        "i  loop sums DOWN each column",
        fontsize=9.5, color=C_I, fontweight="bold")

# ── Computation trace ──
ty = -1.5
ax.text(0.5, ty, "Column by column:", fontsize=12,
        fontweight="bold", color=MEDIUM)

traces = [
    ("j=0:", "M[0,0] + M[1,0] + M[2,0]", "1 + 5 + 2", "=  8"),
    ("j=1:", "M[0,1] + M[1,1] + M[2,1]", "4 + 3 + 1", "=  8"),
    ("j=2:", "M[0,2] + M[1,2] + M[2,2]", "2 + 6 + 3", "= 11"),
]
for idx, (jlabel, expr, nums, res) in enumerate(traces):
    y = ty - 0.6 - idx * 0.55
    ax.text(1.0, y, jlabel, fontsize=10, fontfamily="monospace",
            color=C_J, fontweight="bold")
    ax.text(2.3, y, expr, fontsize=10, fontfamily="monospace", color=MEDIUM)
    ax.text(7.5, y, nums, fontsize=10, fontfamily="monospace", color=MEDIUM)
    ax.text(10.0, y, res, fontsize=11, fontfamily="monospace",
            color=C_OUT, fontweight="bold")

ax.text(1.0, ty - 2.4,
        "result  =  [8, 8, 11]     ← one number per column  (axis  j  survived)",
        fontsize=11, fontfamily="monospace", color=C_OUT, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: The General Rule + Mirror Comparison
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
    facecolor=C_J, alpha=0.08, edgecolor=C_J, linewidth=1.5))
ax.text(c1x + 1.9, 4.2, "INPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_J)
ax.text(c1x + 1.9, 3.55, "i  j", fontsize=22, ha="center",
        fontweight="bold", color=C_J, fontfamily="monospace")
ax.text(c1x + 1.9, 2.8, "matrix has two axes",
        fontsize=9.5, ha="center", color=MEDIUM)
ax.text(c1x + 1.9, 2.4, "i = rows,  j = columns",
        fontsize=9.5, ha="center", color=MEDIUM)

ax.annotate("", xy=(4.8, 3.4), xytext=(4.4, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c2x = 5.0
ax.add_patch(patches.FancyBboxPatch(
    (c2x, 2.2), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_OUT, alpha=0.08, edgecolor=C_OUT, linewidth=1.5))
ax.text(c2x + 1.9, 4.2, "OUTPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_OUT)
ax.text(c2x + 1.9, 3.55, "j", fontsize=22, ha="center",
        fontweight="bold", color=C_J, fontfamily="monospace")
ax.text(c2x + 1.9, 2.8, "j  KEPT  →  columns survive",
        fontsize=9.5, ha="center", color=C_J, fontweight="bold")
ax.text(c2x + 1.9, 2.4, "i  GONE  →  rows summed",
        fontsize=9.5, ha="center", color=C_I, fontweight="bold")

ax.annotate("", xy=(9.3, 3.4), xytext=(8.9, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c3x = 9.5
ax.add_patch(patches.FancyBboxPatch(
    (c3x, 2.2), 4.0, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax.text(c3x + 2.0, 4.2, "RESULT", fontsize=11,
        ha="center", fontweight="bold", color=C_RULE)
ax.text(c3x + 2.0, 3.55, "Σ_i  M[i, j]", fontsize=13, ha="center",
        fontweight="bold", color=C_RULE, fontfamily="monospace")
ax.text(c3x + 2.0, 2.8, "sum down rows per column",
        fontsize=9.5, ha="center", color=C_RULE)
ax.text(c3x + 2.0, 2.4, "output is a vector (one axis)",
        fontsize=9.5, ha="center", color=MEDIUM)

# ── Mirror comparison with row sums ──
ax.text(7.0, 1.5, "Row Sums vs Column Sums: Mirror Images",
        fontsize=11.5, ha="center", fontweight="bold", color=C_OUT)

# Row sums
ax.text(1.0, 0.65, "i j → i", fontsize=12, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(4.0, 0.65, "keep  i  (rows)", fontsize=10, color=C_I,
        fontweight="bold")
ax.text(7.5, 0.65, "sum out  j  (columns)", fontsize=10, color=C_J)
ax.text(11.5, 0.65, "→  row sums", fontsize=10, color=C_I,
        fontweight="bold")

# Column sums
ax.text(1.0, 0.0, "i j → j", fontsize=12, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(4.0, 0.0, "keep  j  (columns)", fontsize=10, color=C_J,
        fontweight="bold")
ax.text(7.5, 0.0, "sum out  i  (rows)", fontsize=10, color=C_I)
ax.text(11.5, 0.0, "→  col sums", fontsize=10, color=C_J,
        fontweight="bold")

ax.plot([1.0, 13.0], [0.35, 0.35], "-", color=SUBTLE, alpha=0.3, lw=1)

# Bottom insight
insight_box(ax, 1.0, -2.8, [
    ("Row sums and column sums are the SAME operation", True),
    ("The only difference: which letter you write after the arrow", True),
    ("The surviving label selects the axis that stays;  the vanishing label selects the axis that collapses", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_column_sums.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
