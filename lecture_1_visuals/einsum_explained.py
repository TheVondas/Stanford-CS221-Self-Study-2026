"""
Lecture 1 Visual: Understanding Einsum Notation
Progressive examples building from dot product to attention scores.

Key question answered: WHY do we label axes the way we do?
- Same label = same dimension, elements paired position-by-position
- Different label = independent dimensions, all combinations
- In output = kept;  NOT in output = summed over
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

# Axis colors — consistent throughout
C_I = "#89b4fa"       # blue:   i / rows / seq1
C_J = "#a6e3a1"       # green:  j / cols / seq2
C_K = "#f38ba8"       # pink:   k / hidden (often contracted)
C_B = "#cba6f7"       # purple: batch
C_OUT = "#f9e2af"     # yellow: output / result
C_RULE = "#fab387"    # orange: rule highlights
C_DIM = "#585b70"     # dimmed cells (not highlighted)

CELL = 0.55
GAP = 0.06
S = CELL + GAP        # stride between cells

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Drawing helpers ────────────────────────────────────────

def dc(ax, x, y, val, col, alpha=0.7, fs=11):
    """Draw a single cell at bottom-left (x, y)."""
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), CELL, CELL, boxstyle="round,pad=0.02",
        facecolor=col, alpha=alpha, edgecolor="white", linewidth=1))
    ax.text(x + CELL / 2, y + CELL / 2, str(val), ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#1e1e2e")


def dvh(ax, vals, x0, y0, col):
    """Horizontal vector; y0 = bottom of cells."""
    for i, v in enumerate(vals):
        dc(ax, x0 + i * S, y0, v, col)


def dvv(ax, vals, x0, y0, col):
    """Vertical vector; y0 = bottom of TOP cell, grows downward."""
    for i, v in enumerate(vals):
        dc(ax, x0, y0 - i * S, v, col)


def dm(ax, data, x0, y0, col=None, cc=None):
    """Matrix; y0 = bottom of top-left cell."""
    R, C_ = data.shape
    for r in range(R):
        for c in range(C_):
            color = col if cc is None else cc[r][c]
            dc(ax, x0 + c * S, y0 - r * S, data[r, c], color)


def arr_h(ax, x0, w, y, col, label, above=True):
    """Horizontal arrow with label."""
    ax.annotate("", xy=(x0 + w + 0.08, y), xytext=(x0 - 0.08, y),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    dy = 0.18 if above else -0.22
    va = "bottom" if above else "top"
    ax.text(x0 + w / 2, y + dy, label, ha="center", fontsize=10,
            color=col, fontweight="bold", va=va)


def arr_v(ax, x, y_top, h, col, label):
    """Vertical arrow with label (y_top = top of arrow region)."""
    ax.annotate("", xy=(x, y_top - h - 0.08), xytext=(x, y_top + 0.08),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax.text(x - 0.22, y_top - h / 2, label, ha="center", va="center",
            fontsize=10, color=col, fontweight="bold", rotation=90)


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
#  FIGURE
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 28))
fig.suptitle("Understanding Einsum: How Axis Labels Determine the Computation",
             fontsize=17, fontweight="bold", y=0.995, color=TEXT)

gs = GridSpec(5, 1, figure=fig,
              height_ratios=[0.55, 1, 1.3, 1.7, 1.9],
              hspace=0.28, top=0.97, bottom=0.01, left=0.03, right=0.97)


# ───────────────────────────────────────────────────────────
#  ROW 0 — THE THREE RULES
# ───────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
ax.axis("off")
ax.set_xlim(0, 18)
ax.set_ylim(0, 3)

ax.text(9, 2.55, "Three Rules of Einsum Labeling", ha="center",
        fontsize=14, fontweight="bold", color=C_OUT)

rules = [
    ("①  Same label on two axes", "→  elements are PAIRED  (position 0↔0, 1↔1, …)"),
    ("②  Label appears in output", "→  that axis is KEPT  in the result"),
    ("③  Shared label NOT in output", "→  that axis is SUMMED OVER  (contracted)"),
]
for i, (prefix, body) in enumerate(rules):
    y = 1.8 - i * 0.65
    ax.text(1.0, y, prefix, fontsize=11.5, fontweight="bold", color=C_RULE)
    ax.text(7.2, y, body, fontsize=11, color=TEXT, va="center")


# ───────────────────────────────────────────────────────────
#  ROW 1 — DOT PRODUCT:  i, i  ->   (scalar)
# ───────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1])
ax.axis("off")
ax.set_xlim(0, 18)
ax.set_ylim(-1.2, 4)

ax.text(0.5, 3.5, "Example 1: Vector Dot Product", fontsize=13,
        fontweight="bold", color=C_I)
ax.text(0.5, 2.7, 'einsum("i, i ->", a, b)', fontsize=12,
        fontfamily="monospace", color=C_OUT)
ax.text(6.5, 2.7, "i  not in output  →  summed  →  scalar result",
        fontsize=10, color=MEDIUM)

# -- vectors --
a = [1, 2, 3]
b = [4, 5, 6]
ax0_x = 1.0
bx0 = ax0_x + 3 * S + 0.55

dvh(ax, a, ax0_x, 1.4, C_I)
ax.text(ax0_x + 1.5 * S - GAP / 2, 1.4 + CELL + 0.12,
        "a", ha="center", fontsize=11, fontweight="bold")
arr_h(ax, ax0_x, 3 * S - GAP, 1.4 - 0.3, C_I, "i", above=False)

dvh(ax, b, bx0, 1.4, C_I)
ax.text(bx0 + 1.5 * S - GAP / 2, 1.4 + CELL + 0.12,
        "b", ha="center", fontsize=11, fontweight="bold")
arr_h(ax, bx0, 3 * S - GAP, 1.4 - 0.3, C_I, "i", above=False)

# pairing brackets
for idx in range(3):
    xa = ax0_x + idx * S + CELL / 2
    xb = bx0 + idx * S + CELL / 2
    ymid = 1.4 + CELL + 0.02
    ax.annotate("", xy=(xb, ymid), xytext=(xa, ymid),
                arrowprops=dict(arrowstyle="<->", color=C_K, lw=1.2,
                                connectionstyle="arc3,rad=-0.3"))

ax.text((ax0_x + bx0 + 3 * S) / 2, 1.4 + CELL + 0.45,
        "same label i → paired", ha="center", fontsize=9, color=C_K,
        fontstyle="italic")

# arrow to result
res_x = bx0 + 3 * S + 0.7
ax.annotate("", xy=(res_x, 1.4 + CELL / 2),
            xytext=(res_x - 0.65, 1.4 + CELL / 2),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

dc(ax, res_x + 0.15, 1.4, 32, C_OUT)
ax.text(res_x + 0.15 + CELL / 2, 1.4 + CELL + 0.12,
        "scalar", ha="center", fontsize=10, color=MEDIUM)

# trace
ax.text(1.0, 0.3,
        "= 1×4  +  2×5  +  3×6  =  4 + 10 + 18  =  32",
        fontsize=11, color=MEDIUM, fontfamily="monospace")

# insight
insight_box(ax, 11.5, -0.3, [
    ("Same label  i  →  elements aligned", True),
    ("i  absent from output  →  summed", True),
    ("All dimensions gone  →  scalar", False),
], w=6)


# ───────────────────────────────────────────────────────────
#  ROW 2 — OUTER PRODUCT:  i, j  ->  ij   (matrix)
# ───────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2])
ax.axis("off")
ax.set_xlim(0, 18)
ax.set_ylim(-2.8, 5)

ax.text(0.5, 4.5, "Example 2: Outer Product  (contrast with dot product!)",
        fontsize=13, fontweight="bold", color=C_J)
ax.text(0.5, 3.7, 'einsum("i, j -> ij", a, b)', fontsize=12,
        fontfamily="monospace", color=C_OUT)
ax.text(6.5, 3.7, "i  and  j  both in output  →  both kept  →  matrix",
        fontsize=10, color=MEDIUM)

# vector a — vertical
a2 = [1, 2, 3]
avx, avy = 1.0, 2.5
dvv(ax, a2, avx, avy, C_I)
arr_v(ax, avx - 0.35, avy + CELL, 3 * S - GAP, C_I, "i")

# vector b — horizontal (above result)
b2 = [4, 5]
bvx, bvy = 3.5, 3.5
dvh(ax, b2, bvx, bvy, C_J)
arr_h(ax, bvx, 2 * S - GAP, bvy + CELL + 0.1, C_J, "j")

# result matrix
result_op = np.array([[4, 5], [8, 10], [12, 15]])
rx, ry = 3.5, 2.5
dm(ax, result_op, rx, ry, C_OUT)
arr_h(ax, rx, 2 * S - GAP, ry - 3 * S + CELL - 0.2, C_J, "j", above=False)
arr_v(ax, rx - 0.35, ry + CELL, 3 * S - GAP, C_I, "i")
ax.text(rx + S - GAP / 2, ry + CELL + 0.15,
        "result (i × j)", ha="center", fontsize=10, color=C_OUT, fontweight="bold")

# connecting lines: a[i] → result row i
for idx in range(3):
    ya = avy - idx * S + CELL / 2
    ax.plot([avx + CELL + 0.05, rx - 0.08], [ya, ya],
            "--", color=SUBTLE, alpha=0.5, lw=1)

# connecting lines: b[j] → result col j
for idx in range(2):
    xb = bvx + idx * S + CELL / 2
    ax.plot([xb, xb], [bvy - 0.02, ry + CELL + 0.02],
            "--", color=SUBTLE, alpha=0.5, lw=1)

# computation hint
ax.text(1.0, -0.5,
        "output[i, j]  =  a[i] × b[j]     ←  every combination, no summation",
        fontsize=11, color=MEDIUM, fontfamily="monospace")

# insight
insight_box(ax, 10, 0.6, [
    ("Different labels  i, j  →  NOT paired", True),
    ("Both appear in output  →  both KEPT", True),
    ("No shared label removed  →  nothing summed", False),
    ("Result shape = (len i) × (len j)", False),
], w=7.2)

# contrast callout
ax.text(10.3, -0.5,
        "Compare: same label (dot) sums → scalar",
        fontsize=10, color=C_K, fontstyle="italic")
ax.text(10.3, -1.0,
        "         diff labels (outer) keeps → matrix",
        fontsize=10, color=C_J, fontstyle="italic")


# ───────────────────────────────────────────────────────────
#  ROW 3 — MATRIX MULTIPLY:  ik, kj  ->  ij
# ───────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[3])
ax.axis("off")
ax.set_xlim(0, 18)
ax.set_ylim(-4, 5.5)

ax.text(0.5, 5, "Example 3: Matrix Multiplication  (combines both rules)",
        fontsize=13, fontweight="bold", color=C_OUT)
ax.text(0.5, 4.2, 'einsum("ik, kj -> ij", A, B)', fontsize=12,
        fontfamily="monospace", color=C_OUT)

# matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6, 7], [8, 9, 10]])
C_ = A @ B  # [[21,24,27],[47,54,61]]

# highlight: row 0 of A (i=0), col 0 of B (j=0) → C[0,0]
A_cc = [[C_I, C_I],
        [C_DIM, C_DIM]]
B_cc = [[C_J, C_DIM, C_DIM],
        [C_J, C_DIM, C_DIM]]
C_cc = [[C_OUT, C_DIM, C_DIM],
        [C_DIM, C_DIM, C_DIM]]

ax_x, ay = 1.0, 3.0
dm(ax, A, ax_x, ay, cc=A_cc)
ax.text(ax_x + S - GAP / 2, ay + CELL + 0.55, "A", ha="center",
        fontsize=12, fontweight="bold")
arr_h(ax, ax_x, 2 * S - GAP, ay + CELL + 0.15, C_K, "k")
arr_v(ax, ax_x - 0.4, ay + CELL, 2 * S - GAP, C_I, "i")

bx_x = 4.5
dm(ax, B, bx_x, ay, cc=B_cc)
ax.text(bx_x + 1.5 * S - GAP / 2, ay + CELL + 0.55, "B", ha="center",
        fontsize=12, fontweight="bold")
arr_h(ax, bx_x, 3 * S - GAP, ay + CELL + 0.15, C_J, "j")
arr_v(ax, bx_x - 0.4, ay + CELL, 2 * S - GAP, C_K, "k")

# ── show k matched ──
k_label_A_x = ax_x + 2 * S - GAP + 0.15   # right edge of A's k arrow
k_label_B_y_top = ay + CELL                # top of B's k arrow
ax.annotate(
    "same k!", xy=(bx_x - 0.15, ay + 0.6),
    xytext=(ax_x + 2 * S + 0.1, ay + 0.6),
    fontsize=9, color=C_K, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_K, lw=1.2,
                    connectionstyle="arc3,rad=0.15"))

# arrow to result
eq_x = bx_x + 3 * S + 0.3
ax.text(eq_x, ay - S / 2 + CELL / 2, "=", fontsize=16, ha="center",
        va="center", color=TEXT)

cx_x = eq_x + 0.6
dm(ax, C_, cx_x, ay, cc=C_cc)
ax.text(cx_x + 1.5 * S - GAP / 2, ay + CELL + 0.55, "C", ha="center",
        fontsize=12, fontweight="bold")
arr_h(ax, cx_x, 3 * S - GAP, ay + CELL + 0.15, C_J, "j  (kept)")
arr_v(ax, cx_x - 0.45, ay + CELL, 2 * S - GAP, C_I, "i  (kept)")

# ── computation trace for C[0,0] ──
ty = 0.9
ax.text(1.0, ty + 0.6, "Computing C[0, 0]:  sum over k", fontsize=11.5,
        fontweight="bold", color=C_OUT)

# show row 0 of A
ax.text(1.0, ty, "Row 0 of A (i=0):", fontsize=9.5, color=C_I, fontweight="bold")
dvh(ax, [1, 2], 4.2, ty - 0.35, C_I)

ax.text(4.2 + 2 * S + 0.15, ty - 0.35 + CELL / 2, "·", fontsize=18,
        ha="center", va="center", color=TEXT)

# show col 0 of B (horizontal for visual alignment)
ax.text(4.2 + 2 * S + 0.5, ty, "Col 0 of B (j=0):", fontsize=9.5,
        color=C_J, fontweight="bold")
dvh(ax, [5, 8], 4.2 + 2 * S + 0.5, ty - 0.35, C_J)

ax.text(4.2 + 4 * S + 1.1, ty - 0.35 + CELL / 2,
        "=  [1×5, 2×8]  =  [5, 16]", fontsize=10,
        ha="left", va="center", color=MEDIUM, fontfamily="monospace")

# sum line
ax.text(4.2 + 4 * S + 1.1, ty - 0.35 + CELL / 2 - 0.5,
        "Σ  =  5 + 16  =  21  ✓", fontsize=11,
        ha="left", va="center", color=C_OUT, fontweight="bold")

# key rules
ax.text(1.0, -1.2,
        "k  in both A and B  but  NOT in output  →  k is summed over  (Σ_k)",
        fontsize=10.5, fontweight="bold", color=C_K)
ax.text(1.0, -1.8,
        "i  in A  and in output  →  kept       j  in B  and in output  →  kept",
        fontsize=10.5, fontweight="bold", color=C_J)
ax.text(1.0, -2.5,
        "Shape:  A(2×2) · B(2×3) → C(2×3)     inner dim k=2 matched & contracted",
        fontsize=10, color=MEDIUM, fontfamily="monospace")

# visual legend for highlighting
ax.text(1.0, -3.3, "Highlighted: ", fontsize=9, color=MEDIUM)
dc(ax, 3.0, -3.55, " ", C_I, fs=8)
ax.text(3.7, -3.3, "= row i=0 of A", fontsize=9, color=C_I)
dc(ax, 6.5, -3.55, " ", C_J, fs=8)
ax.text(7.2, -3.3, "= col j=0 of B", fontsize=9, color=C_J)
dc(ax, 10.0, -3.55, " ", C_OUT, fs=8)
ax.text(10.7, -3.3, "= result C[0,0]", fontsize=9, color=C_OUT)
dc(ax, 13.5, -3.55, " ", C_DIM, fs=8)
ax.text(14.2, -3.3, "= not involved in this element", fontsize=9, color=SUBTLE)


# ───────────────────────────────────────────────────────────
#  ROW 4 — ATTENTION SCORES:  b s h,  b t h  ->  b s t
# ───────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[4])
ax.axis("off")
ax.set_xlim(0, 18)
ax.set_ylim(-5.5, 6)

ax.text(0.5, 5.5, "Example 4: Attention Scores  (from lecture)",
        fontsize=13, fontweight="bold", color=C_B)
ax.text(0.5, 4.7,
        'einsum("b s h, b t h -> b s t", Q, K)',
        fontsize=12, fontfamily="monospace", color=C_OUT)
ax.text(0.5, 4.0,
        "= dot product of every query position with every key position, per batch",
        fontsize=10, color=MEDIUM)

# Q tensor (showing batch=1 slice as 2D: s=2, h=3)
Q = np.array([[1, 0, 1],
              [0, 1, 0]])
qx, qy = 1.0, 2.8
dm(ax, Q, qx, qy, C_I)
ax.text(qx + 1.5 * S - GAP / 2, qy + CELL + 0.55,
        "Q  (batch 0)", ha="center", fontsize=11, fontweight="bold")
arr_h(ax, qx, 3 * S - GAP, qy + CELL + 0.15, C_K, "h (hidden)")
arr_v(ax, qx - 0.45, qy + CELL, 2 * S - GAP, C_I, "s (seq1)")

# K tensor (showing batch=1 slice: t=2, h=3)
K = np.array([[1, 1, 0],
              [0, 0, 1]])
kx, ky = 5.5, 2.8
dm(ax, K, kx, ky, C_J)
ax.text(kx + 1.5 * S - GAP / 2, ky + CELL + 0.55,
        "K  (batch 0)", ha="center", fontsize=11, fontweight="bold")
arr_h(ax, kx, 3 * S - GAP, ky + CELL + 0.15, C_K, "h (hidden)")
arr_v(ax, kx - 0.45, ky + CELL, 2 * S - GAP, C_J, "t (seq2)")

# ── show h matched ──
ax.annotate(
    "same h!", xy=(kx + 0.2, qy + CELL + 0.15),
    xytext=(qx + 3 * S, qy + CELL + 0.15),
    fontsize=9, color=C_K, fontweight="bold", ha="left",
    arrowprops=dict(arrowstyle="<->", color=C_K, lw=1.3,
                    connectionstyle="arc3,rad=-0.2"))

# arrow
eq_x2 = kx + 3 * S + 0.5
ax.text(eq_x2, qy - S / 2 + CELL / 2, "=", fontsize=16,
        ha="center", va="center", color=TEXT)

# result: attention scores
scores = Q @ K.T   # [[1,1],[1,0]]
sx, sy = eq_x2 + 0.7, 2.8
dm(ax, scores, sx, sy, C_OUT)
ax.text(sx + S - GAP / 2, sy + CELL + 0.55,
        "Scores", ha="center", fontsize=11, fontweight="bold", color=C_OUT)
arr_h(ax, sx, 2 * S - GAP, sy + CELL + 0.15, C_J, "t (seq2)")
arr_v(ax, sx - 0.45, sy + CELL, 2 * S - GAP, C_I, "s (seq1)")

# ── axis-by-axis explanation ──
ey = 0.8
ax.text(1.0, ey + 0.6, "Why each label is what it is:",
        fontsize=12, fontweight="bold", color=C_OUT)

explanations = [
    ("b", "batch",   "Same in Q and K,  IN output", "→ passes through unchanged", C_B),
    ("h", "hidden",  "Same in Q and K,  NOT in output", "→ SUMMED  (the dot product)", C_K),
    ("s", "seq1",    "Only in Q,  IN output", "→ KEPT  (one row per query pos)", C_I),
    ("t", "seq2",    "Only in K,  IN output", "→ KEPT  (one col per key pos)", C_J),
]
for idx, (label, meaning, rule, effect, col) in enumerate(explanations):
    y = ey - idx * 0.65
    ax.text(1.3, y, label, fontsize=13, fontweight="bold",
            fontfamily="monospace", color=col)
    ax.text(2.0, y, f"({meaning})", fontsize=10, color=col)
    ax.text(4.2, y, rule, fontsize=9.5, color=MEDIUM)
    ax.text(11.5, y, effect, fontsize=9.5, color=col, fontweight="bold")

# ── computation trace ──
ty2 = -1.2
ax.text(1.0, ty2, "Tracing one element — Scores[s=0, t=1]:",
        fontsize=11, fontweight="bold", color=C_OUT)
ax.text(1.0, ty2 - 0.6,
        "= Σ_h  Q[b=0, s=0, h] × K[b=0, t=1, h]",
        fontsize=10.5, color=MEDIUM, fontfamily="monospace")
ax.text(1.0, ty2 - 1.1,
        "= Q[0,0,0]·K[0,1,0] + Q[0,0,1]·K[0,1,1] + Q[0,0,2]·K[0,1,2]",
        fontsize=10, color=MEDIUM, fontfamily="monospace")
ax.text(1.0, ty2 - 1.6,
        "=     1 × 0      +     0 × 0      +     1 × 1      =  1",
        fontsize=10.5, color=C_OUT, fontfamily="monospace", fontweight="bold")

# ── semantic meaning ──
ax.text(1.0, ty2 - 2.5,
        "Each score[s, t] = how much query position s should attend to key position t",
        fontsize=10.5, color=C_OUT, fontstyle="italic")
ax.text(1.0, ty2 - 3.1,
        "The hidden dimension h is the \"language\" of the dot product — it gets consumed (summed)\n"
        "to produce a single similarity number for each (query, key) pair.",
        fontsize=9.5, color=MEDIUM)

# ── shape summary ──
ax.text(1.0, ty2 - 4.0,
        "Shape:  Q(1,2,3) · K(1,2,3) → Scores(1,2,2)     "
        "h=3 contracted, b=1 kept, s=2 kept, t=2 kept",
        fontsize=10, color=MEDIUM, fontfamily="monospace")


# ── Save ───────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_1_visuals/einsum_explained.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved to {out}")
