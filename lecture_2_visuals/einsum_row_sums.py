"""
Lecture 2 Visual: einsum("i j -> i", M) — Row Sums

Shows WHY the notation works:
- The input has two axes: i (rows) and j (columns)
- The output keeps i but drops j
- j disappears → j is summed over → columns collapse
- i survives → one entry per row → result is a vector
- Contrast with matrix sum (both vanish) and column sums (i vanishes)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Style ─────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_I = "#89b4fa"       # blue:   i axis (rows) — KEPT
C_J = "#a6e3a1"       # green:  j axis (columns) — SUMMED
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


def dvv(ax, vals, x0, y0, col):
    """Vertical vector; y0 = bottom of TOP cell, grows downward."""
    for i, v in enumerate(vals):
        dc(ax, x0, y0 - i * S, v, col)


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
fig.suptitle('einsum("i j  →  i") — Row Sums',
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
ax.text(6.85, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(7.3, ny, '", M)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# Highlight output 'i'
ax.add_patch(patches.FancyBboxPatch(
    (6.7, ny - 0.15), 0.5, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_I, alpha=0.15, edgecolor=C_I, linewidth=1.5,
    linestyle="--"))

# Annotation: i on input side
ax.annotate("rows (axis  i )\npresent on BOTH sides",
            xy=(3.7, ny - 0.2), xytext=(1.3, ny - 1.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5,
                            connectionstyle="arc3,rad=0.2"))

# Annotation: j on input side
ax.annotate("columns (axis  j )\nonly on LEFT side",
            xy=(4.3, ny - 0.2), xytext=(4.3, ny - 1.5),
            fontsize=10, color=C_J, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.5))

# Annotation: output i
ax.annotate("i  KEPT → one output\nentry per row",
            xy=(6.95, ny - 0.2), xytext=(9.5, ny - 0.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5,
                            connectionstyle="arc3,rad=-0.15"))

# j missing callout
ax.text(9.5, ny - 1.4, "j  MISSING → summed over\n(columns collapse)",
        fontsize=10, color=C_J, fontweight="bold", ha="center")


# ───────────────────────────────────────────────────────────
#  SECTION 2: What Happens — row-by-row summation
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-5.5, 7)

ax.text(0.5, 6.5, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

M = np.array([[1, 4, 2],
              [5, 3, 6],
              [2, 1, 3]])
row_sums = M.sum(axis=1)  # [7, 14, 6]

# ── Draw the matrix with rows color-coded ──
mx, my = 1.5, 4.5
# Each row gets a slightly different shade to show grouping
row_colors = [
    ["#7aa8e8", "#7aa8e8", "#7aa8e8"],  # row 0: blue tint
    ["#6b9be0", "#6b9be0", "#6b9be0"],  # row 1: slightly different blue
    ["#5c8ed8", "#5c8ed8", "#5c8ed8"],  # row 2: another blue
]
dm(ax, M, mx, my, cc=row_colors)

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

# ── Result vector (vertical, to the right) ──
rv_x = 8.0
dvv(ax, row_sums.tolist(), rv_x, my, C_OUT)
ax.text(rv_x + CELL / 2, my + CELL + 0.55, "result", fontsize=11,
        fontweight="bold", ha="center", color=C_OUT)
arr_v(ax, rv_x - 0.45, my + CELL, 3 * S - GAP, C_I, "i  (kept)")

# ── Arrows: each row flows into its result entry ──
for r in range(3):
    row_y = my - r * S + CELL / 2
    # Arrow from end of row to result cell
    x_start = mx + 3 * S - GAP + 0.15
    x_end = rv_x - 0.1
    ax.annotate("",
                xy=(x_end, row_y),
                xytext=(x_start, row_y),
                arrowprops=dict(arrowstyle="->", color=C_OUT, lw=2,
                                alpha=0.7))
    # "+" signs between matrix cells in this row
    for c in range(2):
        px = mx + c * S + CELL + (GAP / 2)
        ax.text(px, row_y, "+", fontsize=11, color=C_OUT,
                ha="center", va="center", fontweight="bold")

# Labels on result
for idx in range(3):
    ax.text(rv_x + CELL + 0.3, my - idx * S + CELL / 2,
            f"= {int(row_sums[idx])}", fontsize=12, color=C_OUT,
            fontweight="bold", va="center")

# ── Visual: j axis collapses, i axis stays ──
# Draw a bracket showing j collapsing
bracket_y = my - 3 * S + CELL - 0.6
ax.annotate("", xy=(mx + 3 * S - GAP, bracket_y),
            xytext=(mx, bracket_y),
            arrowprops=dict(arrowstyle="<->", color=C_J, lw=1.5))
ax.text(mx + 1.5 * S - GAP / 2, bracket_y - 0.3,
        "j  axis collapses (summed away)",
        fontsize=10, color=C_J, ha="center", fontweight="bold")

# ── The algorithm ──
ax.text(0.5, 0.8, "The algorithm:", fontsize=12,
        fontweight="bold", color=TEXT)

py = 0.1
ax.text(1.2, py,
        "for each  i :", fontsize=10.5, fontfamily="monospace", color=C_I)
ax.text(1.2, py - 0.55,
        "    output[i] = 0", fontsize=10.5, fontfamily="monospace",
        color=SUBTLE)
ax.text(1.2, py - 1.1,
        "    for each  j :", fontsize=10.5, fontfamily="monospace", color=C_J)
ax.text(1.2, py - 1.65,
        "        output[i]  +=  M[i, j]", fontsize=12,
        fontfamily="monospace", color=TEXT, fontweight="bold")

# Annotate
ax.annotate("i  loop is the outer loop\n→ one output slot per row",
            xy=(3.7, py + 0.1), xytext=(7.5, py + 0.3),
            fontsize=10, color=C_I, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.3,
                            connectionstyle="arc3,rad=-0.15"))

ax.annotate("j  loop sums WITHIN\neach row",
            xy=(4.5, py - 1.1 + 0.1), xytext=(7.5, py - 1.1),
            fontsize=10, color=C_J, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.3,
                            connectionstyle="arc3,rad=-0.15"))

# ── Computation trace ──
ty = py - 3.0
ax.text(0.5, ty, "Row by row:", fontsize=12,
        fontweight="bold", color=MEDIUM)

traces = [
    ("i=0:", "M[0,0] + M[0,1] + M[0,2]", "1 + 4 + 2", "=  7"),
    ("i=1:", "M[1,0] + M[1,1] + M[1,2]", "5 + 3 + 6", "= 14"),
    ("i=2:", "M[2,0] + M[2,1] + M[2,2]", "2 + 1 + 3", "=  6"),
]
for idx, (ilabel, expr, nums, res) in enumerate(traces):
    y = ty - 0.6 - idx * 0.55
    ax.text(1.0, y, ilabel, fontsize=10, fontfamily="monospace",
            color=C_I, fontweight="bold")
    ax.text(2.3, y, expr, fontsize=10, fontfamily="monospace", color=MEDIUM)
    ax.text(7.5, y, nums, fontsize=10, fontfamily="monospace", color=MEDIUM)
    ax.text(10.0, y, res, fontsize=11, fontfamily="monospace",
            color=C_OUT, fontweight="bold")

ax.text(1.0, ty - 2.4,
        "result  =  [7, 14, 6]     ← one number per row  (axis  i  survived)",
        fontsize=11, fontfamily="monospace", color=C_OUT, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: The General Rule + Contrast
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
ax.text(c2x + 1.9, 3.55, "i", fontsize=22, ha="center",
        fontweight="bold", color=C_I, fontfamily="monospace")
ax.text(c2x + 1.9, 2.8, "i  KEPT  →  rows survive",
        fontsize=9.5, ha="center", color=C_I, fontweight="bold")
ax.text(c2x + 1.9, 2.4, "j  GONE  →  columns summed",
        fontsize=9.5, ha="center", color=C_J, fontweight="bold")

ax.annotate("", xy=(9.3, 3.4), xytext=(8.9, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c3x = 9.5
ax.add_patch(patches.FancyBboxPatch(
    (c3x, 2.2), 4.0, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax.text(c3x + 2.0, 4.2, "RESULT", fontsize=11,
        ha="center", fontweight="bold", color=C_RULE)
ax.text(c3x + 2.0, 3.55, "Σ_j  M[i, j]", fontsize=13, ha="center",
        fontweight="bold", color=C_RULE, fontfamily="monospace")
ax.text(c3x + 2.0, 2.8, "sum across columns per row",
        fontsize=9.5, ha="center", color=C_RULE)
ax.text(c3x + 2.0, 2.4, "output is a vector (one axis)",
        fontsize=9.5, ha="center", color=MEDIUM)

# ── Selective summation comparison ──
ax.text(7.0, 1.5, "Which Axis Survives Determines the Operation",
        fontsize=11.5, ha="center", fontweight="bold", color=C_OUT)

# Matrix sum
ax.text(1.0, 0.7, "i j →     ", fontsize=12, fontfamily="monospace",
        color=C_K, fontweight="bold")
ax.text(4.5, 0.7, "both vanish", fontsize=10, color=C_K)
ax.text(8.0, 0.7, "→  total sum (scalar)", fontsize=10,
        color=C_K, fontweight="bold")

# Row sums
ax.text(1.0, 0.05, "i j → i  ", fontsize=12, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(4.5, 0.05, "j  vanishes", fontsize=10, color=C_J)
ax.text(8.0, 0.05, "→  row sums (vector)", fontsize=10,
        color=C_I, fontweight="bold")

# Column sums
ax.text(1.0, -0.6, "i j → j  ", fontsize=12, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(4.5, -0.6, "i  vanishes", fontsize=10, color=C_I)
ax.text(8.0, -0.6, "→  column sums (vector)", fontsize=10,
        color=C_J, fontweight="bold")

ax.plot([1.0, 13.0], [0.42, 0.42], "-", color=SUBTLE, alpha=0.3, lw=1)
ax.plot([1.0, 13.0], [-0.25, -0.25], "-", color=SUBTLE, alpha=0.3, lw=1)

# Bottom insight
insight_box(ax, 1.0, -2.8, [
    ("The output label is a SELECTOR: it tells einsum which axis to preserve", True),
    ("i j → i   keeps rows, sums columns  =  row sums", False),
    ("i j → j   keeps columns, sums rows  =  column sums  (next visual!)", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_row_sums.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
