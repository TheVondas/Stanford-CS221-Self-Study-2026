"""
Lecture 1 Visual: Scalar vs Vector vs Matrix vs Tensor
Illustrates the progression from rank-0 to rank-3 tensors,
showing what axes represent for each.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "text.color": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4",
    "xtick.color": "#6c7086",
    "ytick.color": "#6c7086",
    "font.family": "sans-serif",
    "font.size": 11,
})

ACCENT = "#89b4fa"
ACCENT2 = "#a6e3a1"
ACCENT3 = "#f9e2af"
ACCENT4 = "#cba6f7"
GRID_COLOR = "#45475a"
CELL_ALPHA = 0.7


fig = plt.figure(figsize=(18, 5))
fig.suptitle(
    "Scalar → Vector → Matrix → Tensor (Rank 0 → 3)",
    fontsize=16,
    fontweight="bold",
    color="#cdd6f4",
    y=0.97,
)

# ═══════════════════════════════════════════════════════════
# 1) SCALAR — rank 0, shape ()
# ═══════════════════════════════════════════════════════════
ax0 = fig.add_subplot(1, 4, 1)
ax0.set_xlim(-1, 1)
ax0.set_ylim(-1, 1)
ax0.set_aspect("equal")
ax0.axis("off")

# Draw a single box with a number
rect = patches.FancyBboxPatch(
    (-0.3, -0.3), 0.6, 0.6,
    boxstyle="round,pad=0.05",
    facecolor=ACCENT, alpha=CELL_ALPHA,
    edgecolor="white", linewidth=2,
)
ax0.add_patch(rect)
ax0.text(0, 0, "7", ha="center", va="center", fontsize=22, fontweight="bold", color="#1e1e2e")

ax0.set_title("Scalar (Rank 0)", fontsize=13, fontweight="bold", color=ACCENT, pad=12)
ax0.text(0, -0.65, "shape: ()", ha="center", fontsize=10, color="#a6adc8")
ax0.text(0, -0.85, "0 axes — a single number", ha="center", fontsize=9, color="#6c7086")

# ═══════════════════════════════════════════════════════════
# 2) VECTOR — rank 1, shape (D,)
# ═══════════════════════════════════════════════════════════
ax1 = fig.add_subplot(1, 4, 2)
ax1.set_xlim(-0.5, 5.5)
ax1.set_ylim(-2, 1.5)
ax1.set_aspect("equal")
ax1.axis("off")

vec_vals = [3, 1, 4, 1, 5]
for i, v in enumerate(vec_vals):
    rect = patches.FancyBboxPatch(
        (i, -0.4), 0.85, 0.8,
        boxstyle="round,pad=0.04",
        facecolor=ACCENT2, alpha=CELL_ALPHA,
        edgecolor="white", linewidth=1.5,
    )
    ax1.add_patch(rect)
    ax1.text(i + 0.42, 0, str(v), ha="center", va="center",
             fontsize=16, fontweight="bold", color="#1e1e2e")

# Axis label
ax1.annotate(
    "", xy=(5.2, -0.7), xytext=(-0.2, -0.7),
    arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=2),
)
ax1.text(2.5, -1.15, "Axis 0: elements (D=5)", ha="center", fontsize=10, color=ACCENT2)

ax1.set_title("Vector (Rank 1)", fontsize=13, fontweight="bold", color=ACCENT2, pad=12)
ax1.text(2.5, -1.6, "shape: (5,)", ha="center", fontsize=10, color="#a6adc8")
ax1.text(2.5, -1.95, "1 axis — e.g. features, tokens", ha="center", fontsize=9, color="#6c7086")

# ═══════════════════════════════════════════════════════════
# 3) MATRIX — rank 2, shape (N, D)
# ═══════════════════════════════════════════════════════════
ax2 = fig.add_subplot(1, 4, 3)
ax2.set_xlim(-1.5, 5)
ax2.set_ylim(-2.5, 4)
ax2.set_aspect("equal")
ax2.axis("off")

mat = np.array([[2, 7, 1], [8, 3, 5], [4, 6, 9]])
rows, cols = mat.shape
for r in range(rows):
    for c in range(cols):
        rect = patches.FancyBboxPatch(
            (c * 1.2, (rows - 1 - r) * 1.0), 1.0, 0.8,
            boxstyle="round,pad=0.04",
            facecolor=ACCENT3, alpha=CELL_ALPHA,
            edgecolor="white", linewidth=1.5,
        )
        ax2.add_patch(rect)
        ax2.text(c * 1.2 + 0.5, (rows - 1 - r) * 1.0 + 0.4,
                 str(mat[r, c]), ha="center", va="center",
                 fontsize=14, fontweight="bold", color="#1e1e2e")

# Axis 1 arrow (columns → features)
ax2.annotate(
    "", xy=(3.8, -0.5), xytext=(-0.2, -0.5),
    arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=2),
)
ax2.text(1.8, -0.95, "Axis 1: features (D=3)", ha="center", fontsize=10, color=ACCENT3)

# Axis 0 arrow (rows → examples)
ax2.annotate(
    "", xy=(-0.8, -0.2), xytext=(-0.8, 2.8),
    arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=2),
)
ax2.text(-1.1, 1.3, "Axis 0:\nexamples\n(N=3)", ha="center", fontsize=9, color=ACCENT3)

ax2.set_title("Matrix (Rank 2)", fontsize=13, fontweight="bold", color=ACCENT3, pad=12)
ax2.text(1.8, -1.6, "shape: (3, 3)", ha="center", fontsize=10, color="#a6adc8")
ax2.text(1.8, -2.0, "2 axes — e.g. examples × features", ha="center", fontsize=9, color="#6c7086")

# ═══════════════════════════════════════════════════════════
# 4) TENSOR (rank 3) — shape (B, N, D)
# ═══════════════════════════════════════════════════════════
ax3 = fig.add_subplot(1, 4, 4, projection="3d")
ax3.set_facecolor("#1e1e2e")
ax3.grid(False)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])

# Hide the 3D pane backgrounds
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor("#1e1e2e")
ax3.yaxis.pane.set_edgecolor("#1e1e2e")
ax3.zaxis.pane.set_edgecolor("#1e1e2e")

# Draw stacked cubes to form a 2×3×4 block
B, N, D = 2, 3, 4
cube_size = 0.85
gap = 0.15

def draw_cube(ax, x, y, z, size, color, alpha=0.55):
    """Draw a single cube at position (x, y, z)."""
    s = size
    vertices = [
        [x, y, z], [x+s, y, z], [x+s, y+s, z], [x, y+s, z],       # bottom
        [x, y, z+s], [x+s, y, z+s], [x+s, y+s, z+s], [x, y+s, z+s] # top
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]
    collection = Poly3DCollection(
        faces, alpha=alpha, facecolor=color,
        edgecolor="white", linewidth=0.5,
    )
    ax.add_collection3d(collection)

for b in range(B):
    for n in range(N):
        for d in range(D):
            draw_cube(
                ax3,
                d * (cube_size + gap),
                n * (cube_size + gap),
                b * (cube_size + gap),
                cube_size,
                ACCENT4,
                alpha=0.45,
            )

# Set viewing limits
total_d = D * (cube_size + gap)
total_n = N * (cube_size + gap)
total_b = B * (cube_size + gap)
ax3.set_xlim(0, total_d)
ax3.set_ylim(0, total_n)
ax3.set_zlim(0, total_b)

ax3.view_init(elev=22, azim=-55)

# Axis labels using text3D
ax3.text(total_d / 2, -0.8, -0.3, "Axis 2: features (D=4)",
         color=ACCENT4, fontsize=8, ha="center")
ax3.text(-0.8, total_n / 2, -0.3, "Axis 1: examples (N=3)",
         color=ACCENT4, fontsize=8, ha="center")
ax3.text(-0.6, -0.6, total_b / 2, "Axis 0:\nbatch\n(B=2)",
         color=ACCENT4, fontsize=8, ha="center")

ax3.set_title("Tensor (Rank 3)", fontsize=13, fontweight="bold", color=ACCENT4, pad=0)
ax3.text2D(0.5, 0.02, "shape: (2, 3, 4)", transform=ax3.transAxes,
           ha="center", fontsize=10, color="#a6adc8")
ax3.text2D(0.5, -0.03, "3 axes — e.g. batch × examples × features",
           transform=ax3.transAxes, ha="center", fontsize=9, color="#6c7086")

# ── Save ───────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.02, 1, 0.93])
out_path = "/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026/lecture_1_visuals/tensor_vs_matrix_vector_scalar.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved to {out_path}")
