"""
Lecture 6 Visual: The Grid Search Problem

Simple reference diagram of the 5x5 grid used in the lecture.
S at (0,0), E at (4,4), walls as blocked cells, optimal path highlighted.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Style ────────────────────────────────────────────────
BG      = "#1e1e2e"
TEXT    = "#cdd6f4"
SUBTLE  = "#6c7086"
MEDIUM  = "#a6adc8"
C_POS   = "#a6e3a1"
C_NEG   = "#f38ba8"
C_BOUND = "#f9e2af"
C_I     = "#89b4fa"
C_NODE  = "#89dceb"
C_DIM   = "#585b70"
C_WALL  = "#45475a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})

# ═════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 8))
ax.axis("off")

# Grid definition
grid = [
    "S....",
    "###.#",
    ".....",
    ".####",
    "....E",
]
rows, cols = len(grid), len(grid[0])

# Optimal path: 14 steps
optimal_path = [
    (0,0), (0,1), (0,2), (0,3),
    (1,3),
    (2,3), (2,2), (2,1), (2,0),
    (3,0),
    (4,0), (4,1), (4,2), (4,3), (4,4),
]
path_set = set(optimal_path)

cell = 1.2  # cell size
pad = 0.5
total_w = cols * cell
total_h = rows * cell

# Center the grid
ox = (7 - total_w) / 2
oy = 0.8

ax.set_xlim(0, 7)
ax.set_ylim(-0.5, total_h + oy + 2.0)

ax.text(7 / 2, total_h + oy + 1.5, "Grid Search Problem", fontsize=15,
        ha="center", fontweight="bold", color=C_BOUND)
ax.text(7 / 2, total_h + oy + 0.9,
        "actions: up, down, left, right    cost: 1 each",
        fontsize=10, ha="center", color=MEDIUM)

# Column labels
for c in range(cols):
    ax.text(ox + c * cell + cell / 2, total_h + oy + 0.2, str(c),
            fontsize=10, ha="center", va="center", color=SUBTLE)

# Draw grid
for r in range(rows):
    # Row label
    ax.text(ox - 0.3, total_h + oy - r * cell - cell / 2, str(r),
            fontsize=10, ha="center", va="center", color=SUBTLE)

    for c in range(cols):
        x = ox + c * cell
        y = total_h + oy - (r + 1) * cell
        ch = grid[r][c]

        # Cell background
        if ch == "#":
            fc = C_WALL
            fa = 0.6
        elif (r, c) in path_set:
            fc = C_POS
            fa = 0.12
        else:
            fc = TEXT
            fa = 0.04

        rect = patches.FancyBboxPatch(
            (x + 0.04, y + 0.04), cell - 0.08, cell - 0.08,
            boxstyle="round,pad=0.05",
            facecolor=fc, alpha=fa, edgecolor=SUBTLE,
            linewidth=0.8, zorder=2)
        ax.add_patch(rect)

        # Cell label
        if ch == "S":
            ax.text(x + cell / 2, y + cell / 2, "S", fontsize=16,
                    ha="center", va="center", fontweight="bold",
                    color=C_POS, zorder=5)
        elif ch == "E":
            ax.text(x + cell / 2, y + cell / 2, "E", fontsize=16,
                    ha="center", va="center", fontweight="bold",
                    color=C_BOUND, zorder=5)
        elif ch == "#":
            ax.text(x + cell / 2, y + cell / 2, "#", fontsize=14,
                    ha="center", va="center", fontweight="bold",
                    color=C_NEG, alpha=0.7, zorder=5)

# Draw optimal path arrows
for i in range(len(optimal_path) - 1):
    r1, c1 = optimal_path[i]
    r2, c2 = optimal_path[i + 1]
    x1 = ox + c1 * cell + cell / 2
    y1 = total_h + oy - r1 * cell - cell / 2
    x2 = ox + c2 * cell + cell / 2
    y2 = total_h + oy - r2 * cell - cell / 2
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=2.0,
                                alpha=0.6, shrinkA=8, shrinkB=8),
                zorder=4)

# Legend
ax.text(7 / 2, -0.1,
        "Optimal path:  S → ... → E   (cost 14)",
        fontsize=11, ha="center", fontfamily="monospace",
        color=C_POS, fontweight="bold")

# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/Lecture 6/grid_example.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
