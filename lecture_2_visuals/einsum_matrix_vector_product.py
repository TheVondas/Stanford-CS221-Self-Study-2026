"""
Lecture 2 Visual: einsum("i j, j -> i", M, v) — Matrix-Vector Product

Shows WHY the notation works:
- Matrix has axes i (rows) and j (columns)
- Vector has axis j
- j appears in BOTH inputs → elements PAIRED (column-to-element alignment)
- j is NOT in output → SUMMED OVER (dot product along each row)
- i is in output → KEPT (one result entry per row)
- Each output entry = dot product of one matrix row with the vector
- Combines all three einsum mechanics: pairing, multiplication, summation
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
C_J = "#a6e3a1"       # green:  j axis (columns / vector) — SUMMED
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


def dvh(ax, vals, x0, y0, col):
    for i, v in enumerate(vals):
        dc(ax, x0 + i * S, y0, v, col)


def dvv(ax, vals, x0, y0, col):
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
#  FIGURE — 4 sections
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(14, 32),
                         gridspec_kw={"height_ratios": [1, 2.8, 2.2, 1.6],
                                      "hspace": 0.10})
fig.suptitle('einsum("i j, j  →  i") — Matrix-Vector Product',
             fontsize=16, fontweight="bold", y=0.98, color=C_OUT)

for a in axes:
    a.axis("off")


# ───────────────────────────────────────────────────────────
#  SECTION 1: Read the Notation
# ───────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 14)
ax.set_ylim(-2, 4.5)

ax.text(0.5, 4.0, "Step 1 — Read the Notation", fontsize=14,
        fontweight="bold", color=C_RULE)

ny = 2.8
ax.text(0.6, ny, 'einsum("', fontsize=14, fontfamily="monospace",
        color=MEDIUM)
ax.text(2.95, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(3.35, ny, ' ', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(3.5, ny, 'j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(3.85, ny, ',', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(4.2, ny, ' j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(4.85, ny, '  ->  ', fontsize=14, fontfamily="monospace",
        color=TEXT)
ax.text(6.85, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(7.3, ny, '", M, v)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# Highlight output 'i'
ax.add_patch(patches.FancyBboxPatch(
    (6.7, ny - 0.15), 0.5, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_I, alpha=0.15, edgecolor=C_I, linewidth=1.5,
    linestyle="--"))

# Annotation: i in matrix
ax.annotate("matrix rows\n(axis  i )",
            xy=(3.1, ny - 0.2), xytext=(1.0, ny - 1.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5,
                            connectionstyle="arc3,rad=0.2"))

# Annotation: j in matrix
ax.annotate("matrix cols\n(axis  j )",
            xy=(3.65, ny - 0.2), xytext=(3.3, ny - 1.5),
            fontsize=10, color=C_J, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.5))

# Annotation: j in vector — bracket to show same j
ax.annotate("", xy=(4.5, ny + 0.5), xytext=(3.65, ny + 0.5),
            arrowprops=dict(arrowstyle="<->", color=C_K, lw=1.5,
                            connectionstyle="arc3,rad=-0.4"))
ax.text(4.07, ny + 1.05, "same  j  = PAIRED", fontsize=9, color=C_K,
        ha="center", fontweight="bold", fontstyle="italic")

# Annotation: j in vector
ax.annotate("vector\n(axis  j )",
            xy=(4.5, ny - 0.2), xytext=(5.5, ny - 1.5),
            fontsize=10, color=C_J, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.5))

# Annotation: output i
ax.annotate("i  KEPT\nj  GONE (summed)",
            xy=(6.95, ny - 0.2), xytext=(9.5, ny - 1.0),
            fontsize=10, color=C_OUT, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.5,
                            connectionstyle="arc3,rad=-0.15"))


# ───────────────────────────────────────────────────────────
#  SECTION 2: What Happens — row-by-row dot products
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-6.5, 7.5)

ax.text(0.5, 7.0, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

M = np.array([[2, 1, 3],
              [4, 0, 2],
              [1, 5, 1]])
v = np.array([3, 2, 1])
result = M @ v  # [11, 14, 14]

# ── Draw the matrix ──
mx, my = 1.0, 5.0

# Color rows to show each row participates independently
row0_c = "#7aa8e8"
row1_c = "#6b9be0"
row2_c = "#5c8ed8"
cc = [[row0_c]*3, [row1_c]*3, [row2_c]*3]
dm(ax, M, mx, my, cc=cc)

ax.text(mx + 1 * S, my + CELL + 0.55, "M  (3×3)", fontsize=12,
        fontweight="bold", ha="center")
arr_h(ax, mx, 3 * S - GAP, my + CELL + 0.15, C_J, "j  (columns)")
arr_v(ax, mx - 0.45, my + CELL, 3 * S - GAP, C_I, "i  (rows)")

# Index labels
for idx in range(3):
    ax.text(mx + idx * S + CELL / 2, my + CELL + 0.9,
            f"j={idx}", fontsize=8, color=C_J, ha="center",
            fontfamily="monospace")
for idx in range(3):
    ax.text(mx - 0.85, my - idx * S + CELL / 2,
            f"i={idx}", fontsize=8, color=C_I, ha="center",
            fontfamily="monospace", va="center")

# ── Draw the vector (horizontal, above matrix, aligned to columns) ──
vx, vy = mx, my + CELL + 1.6
dvh(ax, v.tolist(), vx, vy, C_J)
ax.text(vx + 1 * S, vy + CELL + 0.3, "v", fontsize=12,
        fontweight="bold", ha="center", color=C_J)
arr_h(ax, vx, 3 * S - GAP, vy + CELL + 0.15, C_J, "j")

# Dashed lines connecting vector elements to matrix columns
for c in range(3):
    xc = mx + c * S + CELL / 2
    ax.plot([xc, xc], [vy - 0.05, my + CELL + 0.05],
            "--", color=SUBTLE, alpha=0.5, lw=1)

ax.text(mx + 3 * S + 0.5, (vy + my + CELL) / 2,
        "j  aligns\ncolumns ↔ vector",
        fontsize=9.5, color=C_K, fontweight="bold",
        va="center", fontstyle="italic")

# ── Result vector (vertical, to the right) ──
rv_x = 8.0
dvv(ax, result.tolist(), rv_x, my, C_OUT)
ax.text(rv_x + CELL / 2, my + CELL + 0.55, "result", fontsize=11,
        fontweight="bold", ha="center", color=C_OUT)
arr_v(ax, rv_x - 0.45, my + CELL, 3 * S - GAP, C_I, "i  (kept)")

# ── Arrows: each row dot-products with v → one result entry ──
for r in range(3):
    row_y = my - r * S + CELL / 2
    x_start = mx + 3 * S - GAP + 0.15
    x_end = rv_x - 0.1
    ax.annotate("",
                xy=(x_end, row_y),
                xytext=(x_start, row_y),
                arrowprops=dict(arrowstyle="->", color=C_OUT, lw=2,
                                alpha=0.7))

# Label: dot product
ax.text((mx + 3 * S + rv_x) / 2, my + CELL / 2 + 0.55,
        "dot product\nof each row\nwith  v",
        fontsize=9.5, ha="center", color=C_OUT, fontweight="bold")

# ── Detailed trace: highlight row 0 ──
ty = 0.5
ax.text(0.5, ty + 0.7, "Tracing each row:", fontsize=12,
        fontweight="bold", color=MEDIUM)

traces = [
    ("i=0:", "M[0,:] · v", "[2,1,3] · [3,2,1]", "2×3 + 1×2 + 3×1", "= 11"),
    ("i=1:", "M[1,:] · v", "[4,0,2] · [3,2,1]", "4×3 + 0×2 + 2×1", "= 14"),
    ("i=2:", "M[2,:] · v", "[1,5,1] · [3,2,1]", "1×3 + 5×2 + 1×1", "= 14"),
]

for idx, (ilabel, desc, vecs, calc, res) in enumerate(traces):
    y = ty - idx * 0.65
    ax.text(0.5, y, ilabel, fontsize=10, fontfamily="monospace",
            color=C_I, fontweight="bold")
    ax.text(1.5, y, desc, fontsize=10, fontfamily="monospace", color=MEDIUM)
    ax.text(4.0, y, vecs, fontsize=9.5, fontfamily="monospace", color=MEDIUM)
    ax.text(7.5, y, calc, fontsize=9.5, fontfamily="monospace", color=MEDIUM)
    ax.text(11.5, y, res, fontsize=11, fontfamily="monospace",
            color=C_OUT, fontweight="bold")

# Show the Σ_j explicitly
ax.text(0.5, ty - 2.3,
        "Each output[i]  =  Σ_j  M[i, j] × v[j]",
        fontsize=12, fontfamily="monospace", color=C_OUT, fontweight="bold")
ax.text(0.5, ty - 2.9,
        "= dot product of row  i  with vector  v",
        fontsize=11, color=MEDIUM)

# Result
ax.text(0.5, ty - 3.7,
        "result  =  [11, 14, 14]     ← one dot product per row",
        fontsize=11, fontfamily="monospace", color=C_OUT, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: WHY this pattern — the anatomy of Mv
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 14)
ax.set_ylim(-4.5, 6)

ax.text(0.5, 5.5, "Step 3 — Anatomy: Why Each Label Does What It Does",
        fontsize=14, fontweight="bold", color=C_WARN)

# Three roles
roles = [
    ("j", "in M and v", "SHARED label → pairing",
     "Aligns matrix columns with vector elements.\n"
     "This is WHY the vector must have the same\n"
     "length as the matrix has columns.",
     C_J),
    ("j", "NOT in output", "VANISHES → summed",
     "After pairing and multiplying, the products\n"
     "along  j  are accumulated.  This is the\n"
     "\"dot product\" part of Mv.",
     C_K),
    ("i", "only in M, IN output", "SURVIVES → one entry per row",
     "The  i  axis is not shared with  v , so\n"
     "it doesn't pair with anything.  It just\n"
     "indexes which row we're dot-producting.",
     C_I),
]

for idx, (label, where, role, explanation, col) in enumerate(roles):
    bx = 0.5
    by = 4.2 - idx * 2.5
    bw = 13.0
    bh = 2.0

    ax.add_patch(patches.FancyBboxPatch(
        (bx, by), bw, bh, boxstyle="round,pad=0.12",
        facecolor=col, alpha=0.06, edgecolor=col, linewidth=1.5))

    ax.text(bx + 0.4, by + bh - 0.35, label, fontsize=20,
            fontweight="bold", color=col, fontfamily="monospace")
    ax.text(bx + 1.2, by + bh - 0.35, where, fontsize=11,
            color=col, fontweight="bold")
    ax.text(bx + 5.5, by + bh - 0.35, role, fontsize=11,
            color=col, fontweight="bold")
    ax.text(bx + 0.4, by + 0.2, explanation, fontsize=9.5,
            color=MEDIUM, va="bottom")

# ── The pseudocode ──
ax.text(7.5, -2.5, "The algorithm:", fontsize=11,
        fontweight="bold", color=TEXT)
ax.text(7.5, -3.0, "for each  i :", fontsize=10,
        fontfamily="monospace", color=C_I)
ax.text(7.5, -3.45, "    output[i] = 0", fontsize=10,
        fontfamily="monospace", color=SUBTLE)
ax.text(7.5, -3.9, "    for each  j :", fontsize=10,
        fontfamily="monospace", color=C_J)
ax.text(7.5, -4.35, "        output[i]  +=  M[i,j] × v[j]",
        fontsize=10, fontfamily="monospace", color=TEXT,
        fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 4: The General Rule + Connections
# ───────────────────────────────────────────────────────────
ax = axes[3]
ax.set_xlim(0, 14)
ax.set_ylim(-3.5, 5.5)

ax.text(0.5, 5.0, "Step 4 — The General Rule", fontsize=14,
        fontweight="bold", color=C_RULE)

# Three-column layout
c1x = 0.5
ax.add_patch(patches.FancyBboxPatch(
    (c1x, 2.2), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_J, alpha=0.08, edgecolor=C_J, linewidth=1.5))
ax.text(c1x + 1.9, 4.2, "INPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_J)
ax.text(c1x + 1.9, 3.55, "i j ,  j", fontsize=20, ha="center",
        fontweight="bold", color=C_J, fontfamily="monospace")
ax.text(c1x + 1.9, 2.8, "j  shared → paired (×)",
        fontsize=9.5, ha="center", color=C_K, fontweight="bold")
ax.text(c1x + 1.9, 2.4, "i  only in matrix",
        fontsize=9.5, ha="center", color=C_I)

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
ax.text(c2x + 1.9, 2.8, "i  KEPT  →  one entry per row",
        fontsize=9.5, ha="center", color=C_I, fontweight="bold")
ax.text(c2x + 1.9, 2.4, "j  GONE  →  summed (dot product)",
        fontsize=9.5, ha="center", color=C_K, fontweight="bold")

ax.annotate("", xy=(9.3, 3.4), xytext=(8.9, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c3x = 9.5
ax.add_patch(patches.FancyBboxPatch(
    (c3x, 2.2), 4.0, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax.text(c3x + 2.0, 4.2, "RESULT", fontsize=11,
        ha="center", fontweight="bold", color=C_RULE)
ax.text(c3x + 2.0, 3.55, "Σ_j M[i,j]×v[j]", fontsize=12.5, ha="center",
        fontweight="bold", color=C_RULE, fontfamily="monospace")
ax.text(c3x + 2.0, 2.8, "dot product per row",
        fontsize=9.5, ha="center", color=C_RULE)
ax.text(c3x + 2.0, 2.4, "output is a vector (axis  i )",
        fontsize=9.5, ha="center", color=MEDIUM)

# ── Building blocks ──
ax.text(7.0, 1.5, "Matrix-Vector Product = All Three Einsum Mechanics Together",
        fontsize=11, ha="center", fontweight="bold", color=C_OUT)

ax.text(1.0, 0.7, "1.  Shared label  j", fontsize=11,
        color=C_J, fontweight="bold")
ax.text(5.5, 0.7, "→  PAIR  matrix columns with vector entries  (×)",
        fontsize=10, color=MEDIUM)

ax.text(1.0, 0.1, "2.  j  absent from output", fontsize=11,
        color=C_K, fontweight="bold")
ax.text(5.5, 0.1, "→  SUM  over  j  (collapse paired products into one number)",
        fontsize=10, color=MEDIUM)

ax.text(1.0, -0.5, "3.  i  present in output", fontsize=11,
        color=C_I, fontweight="bold")
ax.text(5.5, -0.5, "→  KEEP  one such number per row  (result is a vector)",
        fontsize=10, color=MEDIUM)

# Bottom insight
insight_box(ax, 1.0, -2.8, [
    ("Mv is a row of dot products:  each row paired with v, summed over j, indexed by i", True),
    ("This is why  j  must match: matrix columns = vector length (the shared axis)", True),
    ("The same pattern scales: i j, j k -> i k  is matrix-matrix multiplication", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_matrix_vector_product.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
