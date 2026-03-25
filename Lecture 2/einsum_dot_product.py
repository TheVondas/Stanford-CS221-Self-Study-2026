"""
Lecture 2 Visual: einsum("i, i ->", a, b) — Dot Product

Shows WHY the notation works:
- Both inputs share the SAME label 'i' → elements are paired by position
- 'i' does NOT appear in the output → the axis is SUMMED OVER
- No surviving axes → scalar result
- Contrast with elementwise product: same pairing, but i vanishes here
- WHY multiply and sum? Because those are einsum's two hardcoded operations.
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
C_J = "#a6e3a1"       # green:  second input
C_OUT = "#f9e2af"     # yellow: output / result
C_RULE = "#fab387"    # orange: rule highlights
C_K = "#f38ba8"       # pink:   contracted / summed
C_DIM = "#585b70"     # dimmed
C_WARN = "#f5c2e7"    # light pink: "why not X?" callouts

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


def warn_box(ax, x, y, lines, w=5.5, line_h=0.45):
    """Rounded box for 'why not X?' explanations."""
    h = len(lines) * line_h + 0.25
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=C_WARN, alpha=0.08, edgecolor=C_WARN, linewidth=1.5))
    for i, (txt, bold, col) in enumerate(lines):
        yy = y + h - 0.28 - i * line_h
        ax.text(x + w / 2, yy, txt, ha="center", fontsize=9.5,
                color=col, fontweight="bold" if bold else "normal")


# ═══════════════════════════════════════════════════════════
#  FIGURE — 4 sections stacked vertically
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(14, 28),
                         gridspec_kw={"height_ratios": [1, 2.2, 2.0, 1.6],
                                      "hspace": 0.12})
fig.suptitle('einsum("i, i  →  ") — Dot Product',
             fontsize=16, fontweight="bold", y=0.98, color=C_OUT)

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

# The notation string
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
ax.text(7.3, ny, '", a, b)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# Highlight the empty space after ->
ax.add_patch(patches.FancyBboxPatch(
    (6.85, ny - 0.15), 0.55, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_K, alpha=0.15, edgecolor=C_K, linewidth=1.5,
    linestyle="--"))
ax.text(7.12, ny + 0.55, "nothing here!", fontsize=9,
        color=C_K, ha="center", fontweight="bold", fontstyle="italic")

# First 'i'
ax.annotate("first input\nhas axis  i",
            xy=(3.7, ny - 0.2), xytext=(1.8, ny - 1.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5,
                            connectionstyle="arc3,rad=0.2"))

# Second 'i'
ax.annotate("second input\nalso has axis  i",
            xy=(4.65, ny - 0.2), xytext=(4.65, ny - 1.5),
            fontsize=10, color=C_I, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.5))

# Same label bracket
ax.annotate("", xy=(4.55, ny + 0.5), xytext=(3.75, ny + 0.5),
            arrowprops=dict(arrowstyle="<->", color=C_K, lw=1.5,
                            connectionstyle="arc3,rad=-0.4"))
ax.text(4.15, ny + 1.05, "same label = paired", fontsize=9, color=C_K,
        ha="center", fontweight="bold", fontstyle="italic")

# Empty output
ax.annotate("output has NO axes\n= scalar (one number)",
            xy=(7.12, ny - 0.2), xytext=(10.0, ny - 1.5),
            fontsize=10, color=C_K, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_K, lw=1.5,
                            connectionstyle="arc3,rad=-0.2"))


# ───────────────────────────────────────────────────────────
#  SECTION 2: The Mechanics — What Happens Step by Step
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-5.5, 7.5)

ax.text(0.5, 7.0, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

# ── Two input vectors ──
a = [3, 7, 2, 5]
b = [4, 1, 6, 2]
products = [a_i * b_i for a_i, b_i in zip(a, b)]  # [12, 7, 12, 10]
dot = sum(products)  # 41

vx = 1.5
ay_pos = 5.5
by_pos = 3.7

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

# ── Step A: Pairing — same i connects a[i] to b[i] ──
for idx in range(4):
    xa = vx + idx * S + CELL / 2
    ax.plot([xa, xa], [ay_pos - 0.05, by_pos + CELL + 0.05],
            "--", color=C_K, alpha=0.5, lw=1.5)
    mid_y = (ay_pos + by_pos + CELL) / 2
    ax.text(xa + 0.18, mid_y, "×", fontsize=12, color=C_K,
            ha="center", va="center", fontweight="bold")

ax.text(vx + 4 * S + 0.3, (ay_pos + by_pos + CELL) / 2,
        "same  i  →  paired\nmultiply position\nby position",
        fontsize=10, color=C_K, va="center", fontstyle="italic")

# ── Step B: Intermediate products ──
prod_y = 2.0
ax.text(0.5, prod_y + 0.7, "Intermediate products  (one per i):",
        fontsize=11, fontweight="bold", color=MEDIUM)
dvh(ax, products, vx, prod_y, C_K)

for idx in range(4):
    cx = vx + idx * S + CELL / 2
    ax.text(cx, prod_y - 0.35, f"{a[idx]}×{b[idx]}", fontsize=8,
            color=MEDIUM, ha="center", fontfamily="monospace")

for idx in range(4):
    xa = vx + idx * S + CELL / 2
    ax.annotate("",
                xy=(xa, prod_y + CELL + 0.05),
                xytext=(xa, by_pos - 0.15),
                arrowprops=dict(arrowstyle="->", color=C_K, lw=1.2,
                                alpha=0.5))

# ── Step C: Sum all products → scalar ──
ax.text(0.5, 0.6, "axis  i  is NOT in output  →  sum over  i  →  collapse to scalar",
        fontsize=11, fontweight="bold", color=C_K)

result_x = 10.0
result_y = prod_y
dc(ax, result_x, result_y, dot, C_OUT, fs=14)
ax.text(result_x + CELL / 2, result_y + CELL + 0.2, "scalar",
        fontsize=10, ha="center", color=C_OUT, fontweight="bold")

for idx in range(4):
    x_start = vx + idx * S + CELL
    y_start = prod_y + CELL / 2
    rad = -0.1 - idx * 0.06
    ax.annotate("",
                xy=(result_x - 0.05, result_y + CELL / 2),
                xytext=(x_start + 0.05, y_start),
                arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.3,
                                alpha=0.6,
                                connectionstyle=f"arc3,rad={rad}"))

ax.text((vx + 4 * S + result_x) / 2, prod_y + CELL + 0.6,
        "Σ  sum them all", fontsize=11,
        color=C_OUT, ha="center", fontweight="bold")

# ── Computation trace ──
ty = -1.2
ax.text(0.5, ty, "The full computation:", fontsize=12, fontweight="bold",
        color=MEDIUM)

ax.text(0.5, ty - 0.65, "output", fontsize=11, fontfamily="monospace",
        color=C_OUT, fontweight="bold")
ax.text(2.3, ty - 0.65, "=  Σ_i  a[i] × b[i]", fontsize=11,
        fontfamily="monospace", color=TEXT)
ax.text(7.0, ty - 0.65, "(pair by  i ,  then sum out  i )",
        fontsize=10, color=C_K, fontstyle="italic")

ax.text(2.3, ty - 1.3, "=  a[0]×b[0]  +  a[1]×b[1]  +  a[2]×b[2]  +  a[3]×b[3]",
        fontsize=11, fontfamily="monospace", color=MEDIUM)

ax.text(2.3, ty - 1.95, "=    3 × 4    +    7 × 1    +    2 × 6    +    5 × 2",
        fontsize=11, fontfamily="monospace", color=MEDIUM)

ax.text(2.3, ty - 2.6, "=     12      +      7      +     12      +     10",
        fontsize=11, fontfamily="monospace", color=MEDIUM)

ax.text(2.3, ty - 3.3, "=  41", fontsize=14, fontfamily="monospace",
        color=C_OUT, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: WHY multiply and sum? The two hardcoded ops
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 14)
ax.set_ylim(-4.5, 6)

ax.text(0.5, 5.5, 'Step 3 — But WHY "multiply then sum"?', fontsize=14,
        fontweight="bold", color=C_WARN)
ax.text(0.5, 4.8,
        "There are many ways to turn two vectors into a scalar.",
        fontsize=11, color=MEDIUM)
ax.text(0.5, 4.3,
        "Why does einsum pick this particular one?",
        fontsize=11, color=MEDIUM)

# ── The answer: two hardcoded operations ──
ax.text(0.5, 3.3, "Answer:  einsum has exactly TWO hardcoded operations.",
        fontsize=13, fontweight="bold", color=C_OUT)
ax.text(0.5, 2.7, "They are not inferred. They are the definition of einsum.",
        fontsize=10.5, color=MEDIUM)

# Operation 1 box
op1_x, op1_y = 0.5, 0.8
ax.add_patch(patches.FancyBboxPatch(
    (op1_x, op1_y), 6.0, 1.5, boxstyle="round,pad=0.15",
    facecolor=C_K, alpha=0.10, edgecolor=C_K, linewidth=2))
ax.text(op1_x + 0.3, op1_y + 1.1, "Operation 1", fontsize=10,
        color=C_K, fontweight="bold")
ax.text(op1_x + 3.0, op1_y + 1.1, "MULTIPLY", fontsize=16,
        color=C_K, fontweight="bold", ha="center",
        fontfamily="monospace")
ax.text(op1_x + 3.0, op1_y + 0.4,
        "Multiple inputs?  Entries are always MULTIPLIED together.",
        fontsize=9.5, ha="center", color=MEDIUM)
ax.text(op1_x + 3.0, op1_y + 0.05,
        "Never added, never averaged, never max'd — always  ×",
        fontsize=9, ha="center", color=C_K, fontstyle="italic")

# Operation 2 box
op2_x, op2_y = 7.2, 0.8
ax.add_patch(patches.FancyBboxPatch(
    (op2_x, op2_y), 6.3, 1.5, boxstyle="round,pad=0.15",
    facecolor=C_OUT, alpha=0.10, edgecolor=C_OUT, linewidth=2))
ax.text(op2_x + 0.3, op2_y + 1.1, "Operation 2", fontsize=10,
        color=C_OUT, fontweight="bold")
ax.text(op2_x + 3.15, op2_y + 1.1, "SUM  ( += )", fontsize=16,
        color=C_OUT, fontweight="bold", ha="center",
        fontfamily="monospace")
ax.text(op2_x + 3.15, op2_y + 0.4,
        "Missing axis?  Those products are always SUMMED.",
        fontsize=9.5, ha="center", color=MEDIUM)
ax.text(op2_x + 3.15, op2_y + 0.05,
        "Never concatenated, never kept separate — always  +",
        fontsize=9, ha="center", color=C_OUT, fontstyle="italic")

# The algorithm in pseudocode
ax.text(0.5, -0.3, "The einsum algorithm (for every pattern, always):",
        fontsize=12, fontweight="bold", color=TEXT)

pseudo_x = 1.2
py = -0.9
ax.text(pseudo_x, py,
        "output = 0                              # start at zero",
        fontsize=10.5, fontfamily="monospace", color=SUBTLE)
ax.text(pseudo_x, py - 0.55,
        "for each assignment of index  i :",
        fontsize=10.5, fontfamily="monospace", color=C_I)
ax.text(pseudo_x, py - 1.1,
        "    output  +=  a[i]  ×  b[i]",
        fontsize=12, fontfamily="monospace", color=TEXT, fontweight="bold")

# Annotate the two operations in the pseudocode
ax.annotate("always  ×\n(op 1)",
            xy=(7.8, py - 1.1 + 0.1), xytext=(9.5, py - 0.5),
            fontsize=10, color=C_K, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_K, lw=1.3,
                            connectionstyle="arc3,rad=-0.2"))

ax.annotate("always  +=\n(op 2)",
            xy=(3.6, py - 1.1 + 0.1), xytext=(3.6, py - 0.2),
            fontsize=10, color=C_OUT, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.3,
                            connectionstyle="arc3,rad=0.2"))

# ── "What about other operations?" callout ──
warn_box(ax, 0.5, -3.8, [
    ('"Couldn\'t I sum each vector separately and add them?"', True, C_WARN),
    ("Yes — but that is a DIFFERENT computation.  einsum cannot express it.", False, MEDIUM),
    ("einsum never adds entries from different tensors.  It only  ×  across tensors and  +  within axes.", False, MEDIUM),
    ("The notation doesn't choose an operation — it IS the operation:   ×  then  +=", True, C_OUT),
], w=13, line_h=0.55)


# ───────────────────────────────────────────────────────────
#  SECTION 4: The Rule + Contrast with Elementwise
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
    facecolor=C_I, alpha=0.08, edgecolor=C_I, linewidth=1.5))
ax.text(c1x + 1.9, 4.2, "INPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_I)
ax.text(c1x + 1.9, 3.55, "i , i", fontsize=22, ha="center",
        fontweight="bold", color=C_I, fontfamily="monospace")
ax.text(c1x + 1.9, 2.8, "same label on both inputs",
        fontsize=9.5, ha="center", color=C_I)
ax.text(c1x + 1.9, 2.4, "= elements PAIRED by position",
        fontsize=9.5, ha="center", color=C_K, fontweight="bold")

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
ax.text(c2x + 1.9, 2.8, "i  is NOT in the output",
        fontsize=9.5, ha="center", color=C_K)
ax.text(c2x + 1.9, 2.4, "= axis  i  is SUMMED OVER",
        fontsize=9.5, ha="center", color=C_K, fontweight="bold")

ax.annotate("", xy=(9.3, 3.4), xytext=(8.9, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c3x = 9.5
ax.add_patch(patches.FancyBboxPatch(
    (c3x, 2.2), 4.0, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax.text(c3x + 2.0, 4.2, "RESULT", fontsize=11,
        ha="center", fontweight="bold", color=C_RULE)
ax.text(c3x + 2.0, 3.55, "Σ_i  a[i] × b[i]", fontsize=13, ha="center",
        fontweight="bold", color=C_RULE, fontfamily="monospace")
ax.text(c3x + 2.0, 2.8, "multiply (hardcoded) + sum (hardcoded)",
        fontsize=9, ha="center", color=C_RULE)
ax.text(c3x + 2.0, 2.4, "output is a scalar (no axes)",
        fontsize=9.5, ha="center", color=MEDIUM)

# ── Side-by-side contrast with elementwise ──
ax.text(7.0, 1.5, "The Only Difference from Elementwise Product",
        fontsize=12, ha="center", fontweight="bold", color=C_OUT)

# Elementwise row
ew_y = 0.5
ax.text(1.0, ew_y, "i, i → i", fontsize=13, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(4.5, ew_y, "i  in output", fontsize=10, color=C_J,
        fontweight="bold")
ax.text(7.5, ew_y, "→  axis kept", fontsize=10, color=C_J)
ax.text(10.5, ew_y, "→  vector", fontsize=11, color=C_J,
        fontweight="bold")
for idx, v in enumerate([12, 7, 12, 10]):
    dc(ax, 12.0 + idx * (0.4 + GAP), ew_y - 0.12, v, C_J, fs=8)

# Dot product row
dp_y = -0.3
ax.text(1.0, dp_y, "i, i →  ", fontsize=13, fontfamily="monospace",
        color=C_K, fontweight="bold")
ax.text(4.5, dp_y, "i  NOT in output", fontsize=10, color=C_K,
        fontweight="bold")
ax.text(7.5, dp_y, "→  axis summed", fontsize=10, color=C_K)
ax.text(10.5, dp_y, "→  scalar", fontsize=11, color=C_OUT,
        fontweight="bold")
dc(ax, 12.4, dp_y - 0.12, 41, C_OUT, fs=9)

ax.plot([1.0, 13.5], [0.1, 0.1], "-", color=SUBTLE, alpha=0.4, lw=1)

# Bottom insight
insight_box(ax, 1.0, -2.8, [
    ("Dot product = elementwise product + sum", True),
    ("The ONLY difference is whether  i  appears after the arrow", True),
    ("Present → keep the axis (vector)       Absent → sum it away (scalar)", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_dot_product.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
