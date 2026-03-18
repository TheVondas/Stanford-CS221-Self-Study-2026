"""
Lecture 2 Visual: einsum("i, i -> i", a, b) — Elementwise Product

Shows WHY the notation works:
- Both inputs share the SAME label 'i' → elements are paired by position
- 'i' appears in the output → the axis is KEPT (not summed)
- Result: a new vector where each element = a[i] * b[i]
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
C_J = "#a6e3a1"       # green:  second input (same axis, different tensor)
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
fig, axes = plt.subplots(3, 1, figsize=(14, 18),
                         gridspec_kw={"height_ratios": [1, 2, 1.4],
                                      "hspace": 0.15})
fig.suptitle('einsum("i, i  →  i") — Elementwise Product',
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
ny = 2.8
ax.text(1.2, ny, 'einsum("', fontsize=14, fontfamily="monospace",
        color=MEDIUM)
ax.text(3.55, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(3.95, ny, ',', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(4.3, ny, ' i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(5.1, ny, '  ->  ', fontsize=14, fontfamily="monospace",
        color=TEXT)
ax.text(7.1, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_OUT, fontweight="bold")
ax.text(7.55, ny, '", a, b)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# Annotation arrows
# Arrow from first 'i'
ax.annotate("first input\nhas axis  i",
            xy=(3.7, ny - 0.2), xytext=(1.8, ny - 1.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5,
                            connectionstyle="arc3,rad=0.2"))

# Arrow from second 'i'
ax.annotate("second input\nalso has axis  i",
            xy=(4.65, ny - 0.2), xytext=(4.65, ny - 1.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5))

# Arrow connecting same labels
ax.annotate("", xy=(4.55, ny + 0.55), xytext=(3.75, ny + 0.55),
            arrowprops=dict(arrowstyle="<->", color=C_K, lw=1.5,
                            connectionstyle="arc3,rad=-0.4"))
ax.text(4.15, ny + 1.1, "same label!", fontsize=9, color=C_K,
        ha="center", fontweight="bold", fontstyle="italic")

# Arrow from output 'i'
ax.annotate("output KEEPS\naxis  i",
            xy=(7.3, ny - 0.2), xytext=(9.8, ny - 1.5),
            fontsize=10, color=C_OUT, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.5,
                            connectionstyle="arc3,rad=-0.2"))

# Highlight the output i
ax.add_patch(patches.FancyBboxPatch(
    (6.95, ny - 0.15), 0.5, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_OUT, alpha=0.15, edgecolor=C_OUT, linewidth=1.5,
    linestyle="--"))


# ───────────────────────────────────────────────────────────
#  SECTION 2: The Mechanics — What Happens Step by Step
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-4.5, 7)

ax.text(0.5, 6.5, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

# ── Two input vectors ──
a = [3, 7, 2, 5]
b = [4, 1, 6, 2]
result = [a_i * b_i for a_i, b_i in zip(a, b)]  # [12, 7, 12, 10]

vx = 1.5
ay_pos = 5.0
by_pos = 3.2

# Vector a
dvh(ax, a, vx, ay_pos, C_I)
ax.text(vx - 0.6, ay_pos + CELL / 2, "a", fontsize=13,
        fontweight="bold", ha="center", va="center", color=C_I)
arr_h(ax, vx, 4 * S - GAP, ay_pos + CELL + 0.15, C_I, "axis  i")

# Vector b
dvh(ax, b, vx, by_pos, C_J)
ax.text(vx - 0.6, by_pos + CELL / 2, "b", fontsize=13,
        fontweight="bold", ha="center", va="center", color=C_J)
arr_h(ax, vx, 4 * S - GAP, by_pos - 0.35, C_I, "axis  i", above=False)

# Index labels above a
for idx in range(4):
    cx = vx + idx * S + CELL / 2
    ax.text(cx, ay_pos + CELL + 0.6, f"i={idx}", fontsize=9,
            color=C_I, ha="center", fontfamily="monospace")

# ── Pairing brackets: show same i connects a[i] to b[i] ──
for idx in range(4):
    xa = vx + idx * S + CELL / 2
    # Vertical dashed line connecting a[i] to b[i]
    ax.plot([xa, xa], [ay_pos - 0.05, by_pos + CELL + 0.05],
            "--", color=C_K, alpha=0.5, lw=1.5)
    # Small "×" symbol between them
    mid_y = (ay_pos + by_pos + CELL) / 2
    ax.text(xa + 0.18, mid_y, "×", fontsize=12, color=C_K,
            ha="center", va="center", fontweight="bold")

ax.text(vx + 4 * S + 0.3, (ay_pos + by_pos + CELL) / 2,
        "same  i  →  paired\nposition by position",
        fontsize=10, color=C_K, va="center", fontstyle="italic")

# ── Result vector ──
ry = 1.2
rx = vx

# Arrow from pairing zone down to result
ax.annotate("", xy=(vx + 2 * S - GAP / 2, ry + CELL + 0.15),
            xytext=(vx + 2 * S - GAP / 2, by_pos - 0.5),
            arrowprops=dict(arrowstyle="->", color=C_OUT, lw=2))
ax.text(vx + 2 * S - GAP / 2 + 1.5, (by_pos - 0.5 + ry + CELL + 0.15) / 2,
        "multiply each pair\nkeep axis  i", fontsize=10,
        color=C_OUT, ha="center", va="center", fontweight="bold")

dvh(ax, result, rx, ry, C_OUT)
ax.text(rx - 0.8, ry + CELL / 2, "out", fontsize=13,
        fontweight="bold", ha="center", va="center", color=C_OUT)
arr_h(ax, rx, 4 * S - GAP, ry - 0.35, C_OUT, "axis  i  (kept!)",
      above=False)

# ── Per-element breakdown ──
ty = -0.8
ax.text(0.5, ty, "Element by element:", fontsize=12, fontweight="bold",
        color=MEDIUM)

traces = [
    ("i=0:", "a[0] × b[0]", f"  3 × 4", f"= {result[0]}"),
    ("i=1:", "a[1] × b[1]", f"  7 × 1", f"= {result[1]}"),
    ("i=2:", "a[2] × b[2]", f"  2 × 6", f"= {result[2]}"),
    ("i=3:", "a[3] × b[3]", f"  5 × 2", f"= {result[3]}"),
]

for idx, (ilabel, expr, nums, res) in enumerate(traces):
    y = ty - 0.6 - idx * 0.55
    ax.text(1.0, y, ilabel, fontsize=11, fontfamily="monospace",
            color=C_I, fontweight="bold")
    ax.text(2.3, y, expr, fontsize=11, fontfamily="monospace", color=MEDIUM)
    ax.text(5.5, y, nums, fontsize=11, fontfamily="monospace", color=MEDIUM)
    ax.text(7.8, y, res, fontsize=11, fontfamily="monospace",
            color=C_OUT, fontweight="bold")

# Result summary
ax.text(1.0, ty - 3.2,
        "output  =  [12, 7, 12, 10]     ← same length as inputs (axis  i  survived)",
        fontsize=11, fontfamily="monospace", color=C_OUT, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: The Rule — Why This Works
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 14)
ax.set_ylim(-3, 5)

ax.text(0.5, 4.5, "Step 3 — The General Rule", fontsize=14,
        fontweight="bold", color=C_RULE)

# Three-column layout: input labels → decision → outcome

# Column 1: What labels are in the input?
c1x = 0.5
ax.add_patch(patches.FancyBboxPatch(
    (c1x, 1.8), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.08, edgecolor=C_I, linewidth=1.5))
ax.text(c1x + 1.9, 3.8, "INPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_I)
ax.text(c1x + 1.9, 3.15, "i , i", fontsize=22, ha="center",
        fontweight="bold", color=C_I, fontfamily="monospace")
ax.text(c1x + 1.9, 2.4, "same label on both inputs",
        fontsize=9.5, ha="center", color=C_I)
ax.text(c1x + 1.9, 2.0, "→ elements PAIRED by position",
        fontsize=9.5, ha="center", color=C_K, fontweight="bold")

# Arrow
ax.annotate("", xy=(4.8, 3.0), xytext=(4.4, 3.0),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

# Column 2: What's in the output?
c2x = 5.0
ax.add_patch(patches.FancyBboxPatch(
    (c2x, 1.8), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_OUT, alpha=0.08, edgecolor=C_OUT, linewidth=1.5))
ax.text(c2x + 1.9, 3.8, "OUTPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_OUT)
ax.text(c2x + 1.9, 3.15, "i", fontsize=22, ha="center",
        fontweight="bold", color=C_OUT, fontfamily="monospace")
ax.text(c2x + 1.9, 2.4, "i  is in the output",
        fontsize=9.5, ha="center", color=C_OUT)
ax.text(c2x + 1.9, 2.0, "→ axis is KEPT (not summed)",
        fontsize=9.5, ha="center", color=C_OUT, fontweight="bold")

# Arrow
ax.annotate("", xy=(9.3, 3.0), xytext=(8.9, 3.0),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

# Column 3: What's the result?
c3x = 9.5
ax.add_patch(patches.FancyBboxPatch(
    (c3x, 1.8), 4.0, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax.text(c3x + 2.0, 3.8, "RESULT", fontsize=11,
        ha="center", fontweight="bold", color=C_RULE)
ax.text(c3x + 2.0, 3.15, "out[i] = a[i] × b[i]", fontsize=13, ha="center",
        fontweight="bold", color=C_RULE, fontfamily="monospace")
ax.text(c3x + 2.0, 2.4, "multiply at each position",
        fontsize=9.5, ha="center", color=C_RULE)
ax.text(c3x + 2.0, 2.0, "output is a vector (same shape)",
        fontsize=9.5, ha="center", color=MEDIUM)

# Contrast box at the bottom
insight_box(ax, 1.0, -2.5, [
    ("Compare with dot product:   i, i  →       (no output labels)", True),
    ("There, i  disappears → summed → scalar", False),
    ("Here,  i  survives   → kept   → vector", False),
    ("Same pairing, different fate for the axis!", False),
], w=12, line_h=0.5)

ax.text(7.0, 0.8, "Same label  =  paired.    In output  =  kept.    Not in output  =  summed.",
        fontsize=11.5, ha="center", fontweight="bold", color=C_OUT)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_elementwise_product.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
