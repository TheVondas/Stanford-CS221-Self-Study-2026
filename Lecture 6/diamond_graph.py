"""
Lecture 6 Visual: The Diamond Graph

Simple reference diagram of the diamond graph used throughout Lecture 6.
A↔B (1), A↔C (100), B↔C (1), B↔D (100), C↔D (1).  Start: A, End: D.
"""

import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Style ────────────────────────────────────────────────
BG      = "#1e1e2e"
TEXT    = "#cdd6f4"
SUBTLE  = "#6c7086"
MEDIUM  = "#a6adc8"
C_POS   = "#a6e3a1"
C_BOUND = "#f9e2af"
C_I     = "#89b4fa"
C_NODE  = "#89dceb"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})

# ═════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")
ax.set_xlim(-3, 3); ax.set_ylim(-3.2, 3.2)
ax.set_aspect("equal")

ax.text(0, 2.9, "The Diamond Graph", fontsize=14, ha="center",
        fontweight="bold", color=C_BOUND)

# Node positions — diamond shape
#       B
#      / \
#     A   D
#      \ /
#       C
positions = {"A": (-1.8, 0), "B": (0, 1.5), "C": (0, -1.5), "D": (1.8, 0)}
colors    = {"A": C_POS, "B": C_NODE, "C": C_NODE, "D": C_BOUND}

# Draw nodes
for lbl, (x, y) in positions.items():
    circle = plt.Circle((x, y), 0.38, facecolor=colors[lbl], alpha=0.14,
                         edgecolor=colors[lbl], linewidth=2.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, lbl, ha="center", va="center", fontsize=16,
            fontweight="bold", color=colors[lbl], zorder=6)

# START / END labels
ax.text(-1.8, -0.6, "start", fontsize=9, ha="center", color=C_POS,
        fontweight="bold")
ax.text(1.8, -0.6, "end", fontsize=9, ha="center", color=C_BOUND,
        fontweight="bold")

# Edges with costs
edges = [
    ("A", "B", "1"),
    ("A", "C", "100"),
    ("B", "C", "1"),
    ("B", "D", "100"),
    ("C", "D", "1"),
]

for src, dst, cost in edges:
    x1, y1 = positions[src]
    x2, y2 = positions[dst]
    # Draw a simple line (bidirectional implied)
    ax.plot([x1, x2], [y1, y2], color=C_I, lw=2.0, alpha=0.45, zorder=2)
    # Cost label at midpoint
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx, my, cost, fontsize=11, ha="center", va="center",
            color=C_BOUND, fontweight="bold",
            bbox=dict(facecolor=BG, edgecolor="none", pad=2))

# Best path note
ax.text(0, -2.6, "Best path:  A → B → C → D   (cost 3)",
        fontsize=11, ha="center", fontfamily="monospace",
        color=C_POS, fontweight="bold")

# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/Lecture 6/diamond_graph.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
