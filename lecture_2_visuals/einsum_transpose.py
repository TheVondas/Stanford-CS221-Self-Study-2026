"""
Lecture 2 Visual: einsum("i j -> j i", M) — Transpose

Shows WHY the notation works:
- The input has axes i (rows) and j (columns)
- The output has BOTH axes — nothing vanishes, nothing is summed
- But the ORDER is swapped: j comes first, i comes second
- Rows become columns, columns become rows
- This is the simplest einsum that rearranges without computing
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
fig, axes = plt.subplots(3, 1, figsize=(14, 22),
                         gridspec_kw={"height_ratios": [1, 2.8, 1.6],
                                      "hspace": 0.12})
fig.suptitle('einsum("i j  →  j i") — Transpose',
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
ax.text(1.0, ny, 'einsum("', fontsize=14, fontfamily="monospace",
        color=MEDIUM)
ax.text(3.35, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(3.8, ny, ' ', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(3.95, ny, 'j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(4.5, ny, '  ->  ', fontsize=14, fontfamily="monospace",
        color=TEXT)
ax.text(6.6, ny, 'j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(7.0, ny, ' ', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(7.15, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(7.6, ny, '", M)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# Highlight: both labels survive — but SWAPPED
ax.add_patch(patches.FancyBboxPatch(
    (6.4, ny - 0.15), 1.15, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_OUT, alpha=0.15, edgecolor=C_OUT, linewidth=1.5,
    linestyle="--"))

# Crossing arrows showing the swap
# i goes from position 1 (input) to position 2 (output)
ax.annotate("",
            xy=(7.3, ny + 0.65), xytext=(3.5, ny + 0.65),
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.8,
                            connectionstyle="arc3,rad=-0.35"))
ax.text(5.4, ny + 1.35, "i :  axis 0 → axis 1", fontsize=9,
        color=C_I, ha="center", fontweight="bold")

# j goes from position 2 (input) to position 1 (output)
ax.annotate("",
            xy=(6.75, ny + 0.55), xytext=(4.1, ny + 0.55),
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.8,
                            connectionstyle="arc3,rad=0.35"))
ax.text(5.4, ny - 0.75, "j :  axis 1 → axis 0", fontsize=9,
        color=C_J, ha="center", fontweight="bold")

# Key callout
ax.text(10.0, ny + 0.3, "Both labels KEPT", fontsize=11,
        color=C_OUT, fontweight="bold")
ax.text(10.0, ny - 0.25, "→ nothing summed", fontsize=11,
        color=C_OUT)
ax.text(10.0, ny - 0.8, "Order SWAPPED", fontsize=11,
        color=C_WARN, fontweight="bold")
ax.text(10.0, ny - 1.35, "→ axes rearranged", fontsize=11,
        color=C_WARN)


# ───────────────────────────────────────────────────────────
#  SECTION 2: What Happens — visual swap
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-6.5, 7)

ax.text(0.5, 6.5, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

M = np.array([[1, 4, 2],
              [5, 3, 6]])
Mt = M.T  # [[1, 5], [4, 3], [2, 6]]

# ── Input matrix (2×3) ──
mx, my = 1.0, 4.5

# Color each cell uniquely so you can track it through the transpose
cell_colors_in = [
    ["#89b4fa", "#a6e3a1", "#f9e2af"],   # row 0
    ["#cba6f7", "#f38ba8", "#94e2d5"],    # row 1
]
dm(ax, M, mx, my, cc=cell_colors_in)

ax.text(mx + 1 * S, my + CELL + 0.55, "M  (2 × 3)", fontsize=12,
        fontweight="bold", ha="center")
arr_h(ax, mx, 3 * S - GAP, my + CELL + 0.15, C_J, "j  (3 columns)")
arr_v(ax, mx - 0.45, my + CELL, 2 * S - GAP, C_I, "i  (2 rows)")

# Index labels
for idx in range(3):
    ax.text(mx + idx * S + CELL / 2, my + CELL + 0.9,
            f"j={idx}", fontsize=8, color=C_J, ha="center",
            fontfamily="monospace")
for idx in range(2):
    ax.text(mx - 0.85, my - idx * S + CELL / 2,
            f"i={idx}", fontsize=8, color=C_I, ha="center",
            fontfamily="monospace", va="center")

# ── Arrow showing transformation ──
arrow_x = mx + 3 * S + 0.8
ax.text(arrow_x + 0.3, my - 0.5 * S + CELL / 2, "→", fontsize=28,
        ha="center", va="center", color=TEXT)
ax.text(arrow_x + 0.3, my - 0.5 * S + CELL / 2 - 0.55,
        "swap\naxes", fontsize=10, ha="center", color=C_WARN,
        fontweight="bold")

# ── Output matrix (3×2) — transposed ──
ox = arrow_x + 1.5
oy = my + S  # one row taller, so center vertically

# Same unique colors but rearranged
cell_colors_out = [
    ["#89b4fa", "#cba6f7"],   # was col 0: M[0,0], M[1,0]
    ["#a6e3a1", "#f38ba8"],   # was col 1: M[0,1], M[1,1]
    ["#f9e2af", "#94e2d5"],   # was col 2: M[0,2], M[1,2]
]
dm(ax, Mt, ox, oy, cc=cell_colors_out)

ax.text(ox + 0.5 * S, oy + CELL + 0.55, "M^T  (3 × 2)", fontsize=12,
        fontweight="bold", ha="center")
# NOTE: axes are swapped — j is now rows, i is now columns
arr_h(ax, ox, 2 * S - GAP, oy + CELL + 0.15, C_I, "i  (now columns)")
arr_v(ax, ox - 0.45, oy + CELL, 3 * S - GAP, C_J, "j  (now rows)")

# Index labels on output
for idx in range(2):
    ax.text(ox + idx * S + CELL / 2, oy + CELL + 0.9,
            f"i={idx}", fontsize=8, color=C_I, ha="center",
            fontfamily="monospace")
for idx in range(3):
    ax.text(ox - 0.85, oy - idx * S + CELL / 2,
            f"j={idx}", fontsize=8, color=C_J, ha="center",
            fontfamily="monospace", va="center")

# ── Trace specific cells to show the mapping ──
# Draw lines connecting matching colors
connections = [
    # (input_row, input_col, output_row, output_col, color)
    (0, 0, 0, 0, "#89b4fa"),
    (0, 1, 1, 0, "#a6e3a1"),
    (0, 2, 2, 0, "#f9e2af"),
    (1, 0, 0, 1, "#cba6f7"),
    (1, 1, 1, 1, "#f38ba8"),
    (1, 2, 2, 1, "#94e2d5"),
]

# Show a few key mappings with dashed lines
for ir, ic, or_, oc, col in connections:
    x1 = mx + ic * S + CELL
    y1 = my - ir * S + CELL / 2
    x2 = ox
    y2 = oy - or_ * S + CELL / 2
    ax.plot([x1 + 0.02, x2 - 0.02], [y1, y2],
            "--", color=col, alpha=0.35, lw=1.2)

# ── The algorithm ──
ax.text(0.5, 1.5, "The algorithm:", fontsize=12,
        fontweight="bold", color=TEXT)

py = 0.8
ax.text(1.2, py,
        "for each  i :", fontsize=10.5, fontfamily="monospace", color=C_I)
ax.text(1.2, py - 0.55,
        "    for each  j :", fontsize=10.5, fontfamily="monospace", color=C_J)
ax.text(1.2, py - 1.1,
        "        output[j, i]  =  M[i, j]", fontsize=12,
        fontfamily="monospace", color=TEXT, fontweight="bold")

# Annotate the swap in indexing
ax.annotate("input:  [i, j]\noutput: [j, i]\nindices swapped!",
            xy=(4.2, py - 1.1 + 0.1), xytext=(8.0, py - 0.3),
            fontsize=10.5, color=C_WARN, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.3,
                            connectionstyle="arc3,rad=-0.15"))

# Key point
ax.text(0.5, py - 2.1,
        "No axis vanishes  →  no summation  →  pure rearrangement",
        fontsize=11, fontweight="bold", color=C_OUT)
ax.text(0.5, py - 2.7,
        "Every value is copied exactly once — nothing is combined",
        fontsize=10.5, color=MEDIUM)

# ── Element mapping table ──
ty = py - 3.8
ax.text(0.5, ty, "Element by element:", fontsize=12,
        fontweight="bold", color=MEDIUM)

mappings = [
    ("M[0,0] = 1", "→  out[0,0] = 1", "#89b4fa"),
    ("M[0,1] = 4", "→  out[1,0] = 4", "#a6e3a1"),
    ("M[0,2] = 2", "→  out[2,0] = 2", "#f9e2af"),
    ("M[1,0] = 5", "→  out[0,1] = 5", "#cba6f7"),
    ("M[1,1] = 3", "→  out[1,1] = 3", "#f38ba8"),
    ("M[1,2] = 6", "→  out[2,1] = 6", "#94e2d5"),
]

for idx, (src, dst, col) in enumerate(mappings):
    col_offset = 0 if idx < 3 else 6.5
    row = idx % 3
    y = ty - 0.55 - row * 0.5
    # Color swatch
    dc(ax, 0.5 + col_offset, y - 0.12, "", col, fs=6, alpha=0.7)
    ax.text(1.2 + col_offset, y, src, fontsize=10,
            fontfamily="monospace", color=MEDIUM)
    ax.text(4.0 + col_offset, y, dst, fontsize=10,
            fontfamily="monospace", color=C_OUT, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: The General Rule + Key Insight
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
ax.text(c1x + 1.9, 2.8, "rows first, columns second",
        fontsize=9.5, ha="center", color=MEDIUM)
ax.text(c1x + 1.9, 2.4, "shape: (rows × cols)",
        fontsize=9.5, ha="center", color=MEDIUM)

ax.annotate("", xy=(4.8, 3.4), xytext=(4.4, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c2x = 5.0
ax.add_patch(patches.FancyBboxPatch(
    (c2x, 2.2), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_OUT, alpha=0.08, edgecolor=C_OUT, linewidth=1.5))
ax.text(c2x + 1.9, 4.2, "OUTPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_OUT)
ax.text(c2x + 1.9, 3.55, "j  i", fontsize=22, ha="center",
        fontweight="bold", color=C_OUT, fontfamily="monospace")
ax.text(c2x + 1.9, 2.8, "BOTH kept, order SWAPPED",
        fontsize=9.5, ha="center", color=C_WARN, fontweight="bold")
ax.text(c2x + 1.9, 2.4, "shape: (cols × rows)",
        fontsize=9.5, ha="center", color=MEDIUM)

ax.annotate("", xy=(9.3, 3.4), xytext=(8.9, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c3x = 9.5
ax.add_patch(patches.FancyBboxPatch(
    (c3x, 2.2), 4.0, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax.text(c3x + 2.0, 4.2, "RESULT", fontsize=11,
        ha="center", fontweight="bold", color=C_RULE)
ax.text(c3x + 2.0, 3.55, "out[j,i] = M[i,j]", fontsize=13, ha="center",
        fontweight="bold", color=C_RULE, fontfamily="monospace")
ax.text(c3x + 2.0, 2.8, "pure rearrangement",
        fontsize=9.5, ha="center", color=C_RULE)
ax.text(c3x + 2.0, 2.4, "no summation, no multiplication",
        fontsize=9.5, ha="center", color=MEDIUM)

# ── What the output labels control ──
ax.text(7.0, 1.5, "The Output Labels Control THREE Things",
        fontsize=11.5, ha="center", fontweight="bold", color=C_OUT)

ax.text(1.0, 0.7, "1.  Which axes survive",
        fontsize=11, color=TEXT, fontweight="bold")
ax.text(5.0, 0.7, "(present = kept,  absent = summed)",
        fontsize=10, color=MEDIUM)

ax.text(1.0, 0.1, "2.  What ORDER they appear in",
        fontsize=11, color=C_WARN, fontweight="bold")
ax.text(5.0, 0.1, "(i j = original,  j i = transposed)",
        fontsize=10, color=MEDIUM)

ax.text(1.0, -0.5, "3.  The SHAPE of the result",
        fontsize=11, color=C_OUT, fontweight="bold")
ax.text(5.0, -0.5, "(label order = axis order = shape)",
        fontsize=10, color=MEDIUM)

# Bottom insight
insight_box(ax, 1.0, -2.8, [
    ("Transpose is the only common einsum with NO arithmetic at all", True),
    ("Same labels in, same labels out — just reordered", True),
    ("This is why einops/einsum notation makes transposes so readable", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_transpose.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
