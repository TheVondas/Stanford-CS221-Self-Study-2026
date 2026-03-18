"""
Lecture 2 Visual: einsum("i, j -> i j", a, b) — Outer Product

Shows WHY the notation works:
- The two inputs have DIFFERENT labels 'i' and 'j' → NOT paired
- Both labels appear in the output → both axes KEPT
- No label disappears → nothing is summed
- Result is a matrix: every combination of a[i] × b[j]
- Contrast with dot product: different labels = expansion, same labels = contraction
- Reinforces the two hardcoded operations: × across inputs, += for missing axes
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Style ─────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_I = "#89b4fa"       # blue:   i axis
C_J = "#a6e3a1"       # green:  j axis
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


def dvh(ax, vals, x0, y0, col):
    for i, v in enumerate(vals):
        dc(ax, x0 + i * S, y0, v, col)


def dvv(ax, vals, x0, y0, col):
    """Vertical vector; y0 = bottom of TOP cell, grows downward."""
    for i, v in enumerate(vals):
        dc(ax, x0, y0 - i * S, v, col)


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
#  FIGURE — 4 sections
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(14, 30),
                         gridspec_kw={"height_ratios": [1, 2.6, 1.8, 1.4],
                                      "hspace": 0.12})
fig.suptitle('einsum("i, j  →  i j") — Outer Product',
             fontsize=16, fontweight="bold", y=0.98, color=C_OUT)

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
ax.text(3.95, ny, ',', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(4.4, ny, ' j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(5.2, ny, '  ->  ', fontsize=14, fontfamily="monospace",
        color=TEXT)
ax.text(7.2, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(7.55, ny, ' ', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(7.7, ny, 'j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(8.1, ny, '", a, b)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# First input label 'i'
ax.annotate("first input\nhas axis  i",
            xy=(3.7, ny - 0.2), xytext=(1.8, ny - 1.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5,
                            connectionstyle="arc3,rad=0.2"))

# Second input label 'j'
ax.annotate("second input\nhas axis  j",
            xy=(4.75, ny - 0.2), xytext=(4.75, ny - 1.5),
            fontsize=10, color=C_J, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.5))

# Key: different labels!
ax.annotate("", xy=(4.55, ny + 0.5), xytext=(3.75, ny + 0.5),
            arrowprops=dict(arrowstyle="<->", color=C_WARN, lw=1.5,
                            connectionstyle="arc3,rad=-0.4"))
ax.text(4.15, ny + 1.1, "DIFFERENT labels!", fontsize=9, color=C_WARN,
        ha="center", fontweight="bold", fontstyle="italic")

# Output labels
ax.add_patch(patches.FancyBboxPatch(
    (7.0, ny - 0.15), 1.0, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_OUT, alpha=0.15, edgecolor=C_OUT, linewidth=1.5,
    linestyle="--"))
ax.annotate("BOTH  i  and  j  kept\n→ output is a matrix (2D)",
            xy=(7.5, ny - 0.2), xytext=(10.5, ny - 1.5),
            fontsize=10, color=C_OUT, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.5,
                            connectionstyle="arc3,rad=-0.2"))


# ───────────────────────────────────────────────────────────
#  SECTION 2: What Happens — the grid of all combinations
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-6, 7.5)

ax.text(0.5, 7.0, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

a = [3, 7, 2]
b = [4, 1, 5]
result = np.array([[ai * bj for bj in b] for ai in a])
# [[12, 3, 15],
#  [28, 7, 35],
#  [ 8, 2, 10]]

# ── Vector a (vertical, left of result) ──
avx, avy = 1.0, 4.8
dvv(ax, a, avx, avy, C_I)
ax.text(avx + CELL / 2, avy + CELL + 0.55, "a", fontsize=13,
        fontweight="bold", ha="center", color=C_I)
arr_v(ax, avx - 0.4, avy + CELL, 3 * S - GAP, C_I, "i")

# Index labels for a
for idx, val in enumerate(a):
    ax.text(avx - 0.7, avy - idx * S + CELL / 2,
            f"i={idx}", fontsize=9, color=C_I, ha="center",
            fontfamily="monospace", va="center")

# ── Vector b (horizontal, above result) ──
bvx, bvy = 3.5, 6.0
dvh(ax, b, bvx, bvy, C_J)
ax.text(bvx + 1 * S, bvy + CELL + 0.55, "b", fontsize=13,
        fontweight="bold", ha="center", color=C_J)
arr_h(ax, bvx, 3 * S - GAP, bvy + CELL + 0.15, C_J, "j")

# Index labels for b
for idx, val in enumerate(b):
    ax.text(bvx + idx * S + CELL / 2, bvy - 0.3,
            f"j={idx}", fontsize=9, color=C_J, ha="center",
            fontfamily="monospace")

# ── Result matrix ──
rx, ry = 3.5, 4.8
dm(ax, result, rx, ry, C_OUT)
arr_h(ax, rx, 3 * S - GAP, ry - 3 * S + CELL - 0.3, C_J,
      "j  (kept)", above=False)
arr_v(ax, rx - 0.4, ry + CELL, 3 * S - GAP, C_I, "i  (kept)")
ax.text(rx + 1 * S, ry + CELL + 0.15, "result  (i × j)",
        fontsize=11, ha="center", color=C_OUT, fontweight="bold")

# ── Dashed lines connecting a → rows, b → columns ──
for idx in range(3):
    ya = avy - idx * S + CELL / 2
    ax.plot([avx + CELL + 0.05, rx - 0.08], [ya, ya],
            "--", color=SUBTLE, alpha=0.5, lw=1)

for idx in range(3):
    xb = bvx + idx * S + CELL / 2
    ax.plot([xb, xb], [bvy - 0.4, ry + CELL + 0.02],
            "--", color=SUBTLE, alpha=0.5, lw=1)

# ── Highlight one cell: result[1,2] = a[1] × b[2] = 7 × 5 = 35 ──
hi_r, hi_c = 1, 2  # i=1, j=2
hx = rx + hi_c * S
hy = ry - hi_r * S

# Highlight box around the cell
ax.add_patch(patches.FancyBboxPatch(
    (hx - 0.04, hy - 0.04), CELL + 0.08, CELL + 0.08,
    boxstyle="round,pad=0.02",
    facecolor="none", edgecolor=C_WARN, linewidth=2.5))

# Arrow from a[1]
ax.annotate("",
            xy=(rx - 0.1, avy - hi_r * S + CELL / 2),
            xytext=(avx + CELL + 0.1, avy - hi_r * S + CELL / 2),
            arrowprops=dict(arrowstyle="<-", color=C_WARN, lw=1.8))

# Arrow from b[2]
ax.annotate("",
            xy=(bvx + hi_c * S + CELL / 2, ry + CELL + 0.05),
            xytext=(bvx + hi_c * S + CELL / 2, bvy - 0.45),
            arrowprops=dict(arrowstyle="<-", color=C_WARN, lw=1.8))

# Annotation for highlighted cell
ax.annotate(
    "result[i=1, j=2]\n= a[1] × b[2]\n= 7 × 5 = 35",
    xy=(hx + CELL + 0.1, hy + CELL / 2),
    xytext=(hx + 1.8, hy + 1.2),
    fontsize=10.5, color=C_WARN, fontweight="bold", ha="left",
    arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5,
                    connectionstyle="arc3,rad=-0.2"))

# ── The key insight: every combination ──
ax.text(0.5, 1.5, "The algorithm  (same two hardcoded ops):",
        fontsize=12, fontweight="bold", color=TEXT)

py = 0.8
ax.text(1.2, py,
        "for each  i :", fontsize=11, fontfamily="monospace", color=C_I)
ax.text(1.2, py - 0.55,
        "    for each  j :", fontsize=11, fontfamily="monospace", color=C_J)
ax.text(1.2, py - 1.1,
        "        output[i, j]  +=  a[i]  ×  b[j]",
        fontsize=12, fontfamily="monospace", color=TEXT, fontweight="bold")

# Annotate the ops
ax.annotate("always  ×",
            xy=(8.3, py - 1.1 + 0.1), xytext=(10.5, py - 0.6),
            fontsize=10, color=C_K, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_K, lw=1.3,
                            connectionstyle="arc3,rad=-0.15"))

ax.text(0.5, py - 2.0,
        "No axis disappears  →  no summation  →  the  +=  just fires once per (i, j)",
        fontsize=10.5, color=C_OUT, fontweight="bold")
ax.text(0.5, py - 2.6,
        "Every combination of  i  and  j  produces one output entry",
        fontsize=10.5, color=MEDIUM)

# Full expansion
ax.text(0.5, py - 3.5, "Full expansion:", fontsize=11,
        fontweight="bold", color=MEDIUM)

traces = [
    ("i=0:", "3×4=12", "3×1= 3", "3×5=15"),
    ("i=1:", "7×4=28", "7×1= 7", "7×5=35"),
    ("i=2:", "2×4= 8", "2×1= 2", "2×5=10"),
]
for idx, (ilabel, j0, j1, j2) in enumerate(traces):
    y = py - 4.1 - idx * 0.5
    ax.text(1.0, y, ilabel, fontsize=10, fontfamily="monospace",
            color=C_I, fontweight="bold")
    ax.text(2.5, y, f"j=0: {j0}", fontsize=10, fontfamily="monospace",
            color=MEDIUM)
    ax.text(5.5, y, f"j=1: {j1}", fontsize=10, fontfamily="monospace",
            color=MEDIUM)
    ax.text(8.5, y, f"j=2: {j2}", fontsize=10, fontfamily="monospace",
            color=MEDIUM)


# ───────────────────────────────────────────────────────────
#  SECTION 3: WHY different labels = expansion
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 14)
ax.set_ylim(-4, 6)

ax.text(0.5, 5.5, "Step 3 — WHY Different Labels Create Expansion",
        fontsize=14, fontweight="bold", color=C_WARN)

# ── Same label vs different label comparison ──
ax.text(0.5, 4.5, "The label is the index variable in the loop.",
        fontsize=12, color=MEDIUM, fontweight="bold")
ax.text(0.5, 3.9, "Same label = same loop variable = elements forced into lockstep.",
        fontsize=11, color=C_I)
ax.text(0.5, 3.3, "Different labels = different loop variables = every combination explored.",
        fontsize=11, color=C_J)

# Side-by-side: same label loop vs different label loop
# Left: same label
sl_x, sl_y = 0.5, 0.8
ax.add_patch(patches.FancyBboxPatch(
    (sl_x, sl_y), 6.0, 2.1, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.08, edgecolor=C_I, linewidth=1.5))
ax.text(sl_x + 3.0, sl_y + 1.75, "Same label:   i, i", fontsize=12,
        ha="center", fontweight="bold", color=C_I, fontfamily="monospace")
ax.text(sl_x + 0.3, sl_y + 1.15,
        "for each  i :", fontsize=10, fontfamily="monospace", color=C_I)
ax.text(sl_x + 0.3, sl_y + 0.65,
        "    a[i] × b[i]", fontsize=11, fontfamily="monospace",
        color=TEXT, fontweight="bold")
ax.text(sl_x + 3.0, sl_y + 0.15, "ONE loop → N products",
        fontsize=9.5, ha="center", color=C_I, fontweight="bold")

# Arrow between
ax.text(6.8, sl_y + 1.0, "vs", fontsize=14, ha="center",
        color=SUBTLE, fontweight="bold")

# Right: different labels
dl_x, dl_y = 7.5, 0.8
ax.add_patch(patches.FancyBboxPatch(
    (dl_x, dl_y), 6.0, 2.1, boxstyle="round,pad=0.15",
    facecolor=C_J, alpha=0.08, edgecolor=C_J, linewidth=1.5))
ax.text(dl_x + 3.0, dl_y + 1.75, "Different labels:   i, j", fontsize=12,
        ha="center", fontweight="bold", color=C_J, fontfamily="monospace")
ax.text(dl_x + 0.3, dl_y + 1.15,
        "for each  i :", fontsize=10, fontfamily="monospace", color=C_I)
ax.text(dl_x + 0.3, dl_y + 0.65,
        "    for each  j :", fontsize=10, fontfamily="monospace", color=C_J)
ax.text(dl_x + 0.3, dl_y + 0.15,
        "        a[i] × b[j]", fontsize=11, fontfamily="monospace",
        color=TEXT, fontweight="bold")

# What they produce
ax.text(sl_x + 3.0, sl_y - 0.35, "→ vector (N entries)",
        fontsize=10, ha="center", color=C_I, fontweight="bold")
ax.text(dl_x + 3.0, dl_y - 0.35, "→ matrix (N × M entries)",
        fontsize=10, ha="center", color=C_J, fontweight="bold")

# ── Numeric example comparison ──
cy = -1.3
ax.text(0.5, cy, "With  a = [3, 7, 2]  and  b = [4, 1, 5]:", fontsize=11,
        fontweight="bold", color=MEDIUM)

# Same label result
ax.text(0.5, cy - 0.7, "i, i → i", fontsize=12, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(3.5, cy - 0.7, "[3×4, 7×1, 2×5]  =  [12, 7, 10]",
        fontsize=11, fontfamily="monospace", color=MEDIUM)
ax.text(10.5, cy - 0.7, "3 products",
        fontsize=10, color=C_I, fontweight="bold")

# Different label result
ax.text(0.5, cy - 1.5, "i, j → i j", fontsize=12, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(3.5, cy - 1.3, "[[3×4, 3×1, 3×5],",
        fontsize=10, fontfamily="monospace", color=MEDIUM)
ax.text(3.5, cy - 1.7, " [7×4, 7×1, 7×5],",
        fontsize=10, fontfamily="monospace", color=MEDIUM)
ax.text(3.5, cy - 2.1, " [2×4, 2×1, 2×5]]",
        fontsize=10, fontfamily="monospace", color=MEDIUM)
ax.text(10.5, cy - 1.7, "9 products\n(3 × 3)",
        fontsize=10, color=C_J, fontweight="bold", va="center")

# Bottom insight
insight_box(ax, 1.0, cy - 3.2, [
    ("Same label  =  lockstep (diagonal of the grid)", True),
    ("Different labels  =  full grid (every combination)", True),
    ("Outer product gives you the COMPLETE multiplication table", False),
], w=12, line_h=0.5)


# ───────────────────────────────────────────────────────────
#  SECTION 4: The General Rule + Contrast
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
ax.text(c1x + 1.9, 3.55, "i , j", fontsize=22, ha="center",
        fontweight="bold", color=C_J, fontfamily="monospace")
# Color i and j differently in explanation
ax.text(c1x + 1.9, 2.8, "DIFFERENT labels",
        fontsize=9.5, ha="center", color=C_WARN, fontweight="bold")
ax.text(c1x + 1.9, 2.4, "= NOT paired (independent loops)",
        fontsize=9.5, ha="center", color=MEDIUM)

ax.annotate("", xy=(4.8, 3.4), xytext=(4.4, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c2x = 5.0
ax.add_patch(patches.FancyBboxPatch(
    (c2x, 2.2), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_OUT, alpha=0.08, edgecolor=C_OUT, linewidth=1.5))
ax.text(c2x + 1.9, 4.2, "OUTPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_OUT)
ax.text(c2x + 1.9, 3.55, "i  j", fontsize=22, ha="center",
        fontweight="bold", color=C_OUT, fontfamily="monospace")
ax.text(c2x + 1.9, 2.8, "BOTH  i  and  j  in output",
        fontsize=9.5, ha="center", color=C_OUT, fontweight="bold")
ax.text(c2x + 1.9, 2.4, "= both axes KEPT, nothing summed",
        fontsize=9.5, ha="center", color=MEDIUM)

ax.annotate("", xy=(9.3, 3.4), xytext=(8.9, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c3x = 9.5
ax.add_patch(patches.FancyBboxPatch(
    (c3x, 2.2), 4.0, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax.text(c3x + 2.0, 4.2, "RESULT", fontsize=11,
        ha="center", fontweight="bold", color=C_RULE)
ax.text(c3x + 2.0, 3.55, "out[i,j] = a[i]×b[j]", fontsize=12.5,
        ha="center", fontweight="bold", color=C_RULE, fontfamily="monospace")
ax.text(c3x + 2.0, 2.8, "every combination (×)",
        fontsize=9.5, ha="center", color=C_RULE)
ax.text(c3x + 2.0, 2.4, "output is a matrix (2D)",
        fontsize=9.5, ha="center", color=MEDIUM)

# ── Three-way comparison ──
ax.text(7.0, 1.5, "The Spectrum: Same Labels Contract, Different Labels Expand",
        fontsize=11.5, ha="center", fontweight="bold", color=C_OUT)

# Dot product
ax.text(1.0, 0.6, "i, i →   ", fontsize=12, fontfamily="monospace",
        color=C_K, fontweight="bold")
ax.text(4.5, 0.6, "same  i , vanishes", fontsize=10, color=C_K)
ax.text(8.5, 0.6, "→  contract to scalar", fontsize=10,
        color=C_K, fontweight="bold")
dc(ax, 12.5, 0.45, 41, C_K, fs=9)

# Elementwise
ax.text(1.0, -0.1, "i, i → i ", fontsize=12, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(4.5, -0.1, "same  i , survives", fontsize=10, color=C_I)
ax.text(8.5, -0.1, "→  stay at vector", fontsize=10,
        color=C_I, fontweight="bold")
for idx, v in enumerate([12, 7, 10]):
    dc(ax, 12.0 + idx * (0.4 + GAP), -0.25, v, C_I, fs=8)

# Outer product
ax.text(1.0, -0.8, "i, j → i j", fontsize=12, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(4.5, -0.8, "diff labels, both survive", fontsize=10, color=C_J)
ax.text(8.5, -0.8, "→  expand to matrix", fontsize=10,
        color=C_J, fontweight="bold")
# tiny matrix hint
for r in range(2):
    for c in range(2):
        dc(ax, 12.0 + c * (0.4 + GAP), -0.95 - r * (0.4 + GAP),
           "", C_J, fs=6, alpha=0.5)

ax.plot([1.0, 13.5], [0.35, 0.35], "-", color=SUBTLE, alpha=0.3, lw=1)
ax.plot([1.0, 13.5], [-0.35, -0.35], "-", color=SUBTLE, alpha=0.3, lw=1)

# Bottom insight
insight_box(ax, 1.0, -2.8, [
    ("Labels are loop variables.  Same label = one loop.  Different labels = nested loops.", True),
    ("More independent loops = more combinations = larger output.", False),
    ("Outer product is the multiplication table of two vectors.", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_outer_product.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
