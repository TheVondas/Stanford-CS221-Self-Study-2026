"""
Lecture 5 Visual: Cycles Break Naive Exhaustive Search

Simple diagram: A → B → C → A cycle showing infinite recursion.
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
C_NEG   = "#f38ba8"
C_BOUND = "#f9e2af"
C_DIM   = "#585b70"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})

# ═════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 5.5))
ax.axis("off")
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.8)
ax.set_aspect("equal")

ax.text(0, 2.5, "Cycles Break Naive Search",
        fontsize=13, ha="center", fontweight="bold", color=C_BOUND)

# Triangle: A top, B bottom-left, C bottom-right
cx, cy, r = 0, 0.3, 1.3
angles = [90, 210, 330]
labels = ["A", "B", "C"]
positions = []
for angle, label in zip(angles, labels):
    rad = math.radians(angle)
    nx, ny = cx + r * math.cos(rad), cy + r * math.sin(rad)
    positions.append((nx, ny))
    circle = plt.Circle((nx, ny), 0.35, facecolor=C_NEG, alpha=0.12,
                         edgecolor=C_NEG, linewidth=2.5, zorder=5)
    ax.add_patch(circle)
    ax.text(nx, ny, label, ha="center", va="center", fontsize=15,
            fontweight="bold", color=C_NEG, zorder=6)

# Curved arrows A→B→C→A
for i in range(3):
    x1, y1 = positions[i]
    x2, y2 = positions[(i + 1) % 3]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=2.5,
                                connectionstyle="arc3,rad=-0.2",
                                shrinkA=18, shrinkB=18, alpha=0.8),
                zorder=3)

# Caption
ax.text(0, -2.0, "A -> B -> C -> A -> B -> C -> ...",
        fontsize=10, ha="center", fontfamily="monospace", color=C_NEG,
        fontweight="bold")
ax.text(0, -2.4, "recursion never terminates",
        fontsize=9.5, ha="center", color=MEDIUM, fontstyle="italic")

# ── Save ─────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_5_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "cycle_problem.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
