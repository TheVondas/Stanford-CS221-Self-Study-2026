"""
Lecture 4 Visual: Nonlinear Feature Maps — From Straight Lines to Circular Boundaries

Shows:
1. Original 2D space where no straight line can separate circularly-distributed classes
2. The same 2D space after applying φ(x) — a circular boundary emerges
3. 3D feature space showing how φ(x) lifts points onto a paraboloid,
   where a flat plane cleanly separates them
4. Step-by-step math connecting φ(x), w, and the circular boundary
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ── Style (matching lecture_3_visuals) ──────────────────
BG      = "#1e1e2e"
TEXT    = "#cdd6f4"
SUBTLE  = "#6c7086"
MEDIUM  = "#a6adc8"
C_POS   = "#a6e3a1"     # green
C_NEG   = "#f38ba8"     # pink/red
C_BOUND = "#f9e2af"     # yellow
C_RULE  = "#fab387"     # orange
C_I     = "#89b4fa"     # blue
C_B     = "#cba6f7"     # purple
C_DIM   = "#585b70"     # dim gray
C_NODE  = "#89dceb"     # teal

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Data ────────────────────────────────────────────────
np.random.seed(42)
n = 300
x1 = np.random.uniform(-1.0, 3.0, n)
x2 = np.random.uniform(-1.0, 3.0, n)

# Circle boundary: (x1 - 1)^2 + (x2 - 1)^2 = 2
dist_sq = (x1 - 1)**2 + (x2 - 1)**2
labels = np.where(dist_sq < 2, 1, -1)
pos = labels == 1
neg = labels == -1

# Feature map: φ(x) = (x1, x2, x1² + x2²)
phi_3 = x1**2 + x2**2

# Circle for plotting
theta = np.linspace(0, 2 * np.pi, 200)
circ_x = 1 + np.sqrt(2) * np.cos(theta)
circ_y = 1 + np.sqrt(2) * np.sin(theta)


# ── Helper ──────────────────────────────────────────────
def style_2d(ax):
    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-1.5, 3.5)
    ax.set_xlabel("$x_1$", fontsize=12, color=C_I)
    ax.set_ylabel("$x_2$", fontsize=12, color=C_I)
    ax.tick_params(colors=SUBTLE, labelsize=9)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color(C_DIM)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 30))
fig.suptitle("Nonlinear Feature Maps:\nFrom Straight Lines to Circular Boundaries",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(5, 2, figure=fig,
              height_ratios=[0.8, 3.2, 4.2, 3.2, 1.5],
              hspace=0.22, wspace=0.28,
              top=0.955, bottom=0.02, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 0: Intro
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-0.5, 3)

ax0.text(8, 2.4,
    "A linear classifier divides space with straight boundaries.",
    fontsize=13, ha="center", color=TEXT)
ax0.text(8, 1.5,
    "But many real patterns need curved boundaries (circles, rings, blobs).",
    fontsize=13, ha="center", color=TEXT)
ax0.text(8, 0.5,
    "The trick: transform the input with a nonlinear feature map  φ(x),  then",
    fontsize=13, ha="center", color=C_I, fontweight="bold")
ax0.text(8, -0.2,
    "apply a plain linear classifier in the new space.",
    fontsize=13, ha="center", color=C_I, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: Linear boundary fails
# ─────────────────────────────────────────────────────────
ax_lin = fig.add_subplot(gs[1, 0])

ax_lin.scatter(x1[pos], x2[pos], c=C_POS, s=20, alpha=0.7,
               edgecolors="none", label="+1 (inside circle)", zorder=3)
ax_lin.scatter(x1[neg], x2[neg], c=C_NEG, s=20, alpha=0.7,
               edgecolors="none", label="−1 (outside circle)", zorder=3)

# Two failed linear attempts
line_x = np.linspace(-1.5, 3.5, 100)
ax_lin.plot(line_x, 2.0 - line_x, color=C_BOUND, lw=2, ls="--",
            zorder=4, label="Linear attempts", alpha=0.7)
ax_lin.plot(np.full(100, 1.0), np.linspace(-1.5, 3.5, 100),
            color=C_BOUND, lw=2, ls=":", zorder=4, alpha=0.5)

# Shade to show one attempt's prediction regions
ax_lin.fill_between(line_x, 2.0 - line_x, 3.5,
                    alpha=0.04, color=C_POS)
ax_lin.fill_between(line_x, -1.5, 2.0 - line_x,
                    alpha=0.04, color=C_NEG)

style_2d(ax_lin)
ax_lin.set_title("Original Space — Linear Boundary Fails",
                 fontsize=13, fontweight="bold", color=C_NEG, pad=10)
ax_lin.legend(fontsize=8.5, facecolor=BG, edgecolor=C_DIM,
              labelcolor=TEXT, loc="upper right")

ax_lin.text(1.0, -1.1,
    "No straight line can separate\ngreen from pink!",
    fontsize=10, ha="center", color=C_NEG, fontweight="bold",
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=4, alpha=0.9))


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: Feature map gives circular boundary
# ─────────────────────────────────────────────────────────
ax_nl = fig.add_subplot(gs[1, 1])

ax_nl.scatter(x1[pos], x2[pos], c=C_POS, s=20, alpha=0.7,
              edgecolors="none", label="+1 (inside)", zorder=3)
ax_nl.scatter(x1[neg], x2[neg], c=C_NEG, s=20, alpha=0.7,
              edgecolors="none", label="−1 (outside)", zorder=3)

# Circular boundary
ax_nl.plot(circ_x, circ_y, color=C_BOUND, lw=2.5, zorder=4,
           label="w · φ(x) = 0  →  circle!")

# Center + radius annotation
ax_nl.plot(1, 1, '+', color=C_BOUND, markersize=12, mew=2.5, zorder=5)
r_angle = np.pi / 4
ax_nl.annotate("",
    xy=(1 + np.sqrt(2)*np.cos(r_angle), 1 + np.sqrt(2)*np.sin(r_angle)),
    xytext=(1, 1),
    arrowprops=dict(arrowstyle="<->", color=C_BOUND, lw=1.5))
ax_nl.text(1.55, 2.05, "r = √2", fontsize=10, color=C_BOUND,
           fontweight="bold")
ax_nl.annotate("center\n(1, 1)", xy=(1, 1), xytext=(-0.3, -0.7),
    fontsize=9, color=C_BOUND, ha="center",
    arrowprops=dict(arrowstyle="->", color=C_BOUND, lw=1),
    bbox=dict(facecolor=BG, edgecolor=C_BOUND, pad=3, alpha=0.9))

# Shade regions
ax_nl.fill(circ_x, circ_y, alpha=0.06, color=C_POS)

style_2d(ax_nl)
ax_nl.set_title("After Feature Map — Circular Boundary Works!",
                fontsize=13, fontweight="bold", color=C_POS, pad=10)
ax_nl.legend(fontsize=8.5, facecolor=BG, edgecolor=C_DIM,
             labelcolor=TEXT, loc="upper right")

ax_nl.text(1.0, -1.1,
    "A linear classifier in φ-space\nproduces a circle in x-space!",
    fontsize=10, ha="center", color=C_POS, fontweight="bold",
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=4, alpha=0.9))


# ─────────────────────────────────────────────────────────
#  ROW 2: 3D Feature Space
# ─────────────────────────────────────────────────────────
ax3d = fig.add_subplot(gs[2, :], projection="3d")
ax3d.set_facecolor(BG)

# Style panes / grid
for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor(C_DIM)
    axis._axinfo["grid"]["color"] = (0.35, 0.36, 0.44, 0.3)
ax3d.tick_params(axis="x", colors=SUBTLE, labelsize=8)
ax3d.tick_params(axis="y", colors=SUBTLE, labelsize=8)
ax3d.tick_params(axis="z", colors=SUBTLE, labelsize=8)

# Scatter lifted points
ax3d.scatter(x1[pos], x2[pos], phi_3[pos], c=C_POS, s=15, alpha=0.6,
             edgecolors="none", label="+1  (below plane)", depthshade=False)
ax3d.scatter(x1[neg], x2[neg], phi_3[neg], c=C_NEG, s=15, alpha=0.6,
             edgecolors="none", label="−1  (above plane)", depthshade=False)

# Separating plane: z = 2x₁ + 2x₂
gx, gy = np.meshgrid(np.linspace(-1, 3, 25), np.linspace(-1, 3, 25))
gz_plane = 2 * gx + 2 * gy
ax3d.plot_surface(gx, gy, gz_plane, alpha=0.18,
                  color=C_BOUND, edgecolor=C_BOUND, linewidth=0.15,
                  label="_nolegend_")

# Paraboloid wireframe (very faint, to show the surface shape)
gz_parab = gx**2 + gy**2
ax3d.plot_wireframe(gx, gy, gz_parab, alpha=0.05,
                    color=C_NODE, linewidth=0.3)

# A few representative "lifting lines" to show the mapping
rng = np.random.default_rng(7)
rep = rng.choice(np.where(pos)[0], 5, replace=False)
rep = np.concatenate([rep, rng.choice(np.where(neg)[0], 5, replace=False)])
for i in rep:
    ax3d.plot([x1[i], x1[i]], [x2[i], x2[i]], [0, phi_3[i]],
              color=C_DIM, lw=0.6, ls="--", alpha=0.5)

# Draw the circle at z = 0 (shadow / projection)
ax3d.plot(circ_x, circ_y, zs=0, zdir="z",
          color=C_BOUND, lw=1.2, ls="--", alpha=0.35)

# Labels
ax3d.set_xlabel("$x_1$", fontsize=11, color=C_I, labelpad=8)
ax3d.set_ylabel("$x_2$", fontsize=11, color=C_I, labelpad=8)
ax3d.set_zlabel("$\\phi_3 = x_1^2 + x_2^2$", fontsize=11, color=C_I,
                labelpad=8)
ax3d.set_zlim(0, 20)

ax3d.set_title(
    "Feature Space:  φ(x) = (x₁, x₂, x₁² + x₂²)\n"
    "Points are lifted onto a paraboloid — a flat plane now separates them",
    fontsize=13, fontweight="bold", color=C_B, pad=15)
ax3d.view_init(elev=22, azim=-55)
ax3d.legend(fontsize=9, facecolor=BG, edgecolor=C_DIM, labelcolor=TEXT,
            loc="upper left")

# Manual annotation for the plane
ax3d.text2D(0.72, 0.38,
    "Separating plane\nz = 2x₁ + 2x₂",
    transform=ax3d.transAxes, fontsize=10, color=C_BOUND,
    fontweight="bold", ha="center",
    bbox=dict(facecolor=BG, edgecolor=C_BOUND, pad=4, alpha=0.85))


# ─────────────────────────────────────────────────────────
#  ROW 3: Math breakdown
# ─────────────────────────────────────────────────────────
ax_m = fig.add_subplot(gs[3, :])
ax_m.axis("off")
ax_m.set_xlim(0, 16); ax_m.set_ylim(-3.5, 6)
ax_m.set_title("  The Mathematics: How a Linear Classifier Becomes Circular",
               fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# ── Step 1 ──
ax_m.add_patch(patches.FancyBboxPatch(
    (0.3, 3.5), 7.2, 2.0, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))
ax_m.text(3.9, 5.1, "Step 1 — Define the feature map",
          fontsize=11, ha="center", fontweight="bold", color=C_I)
ax_m.text(3.9, 4.3, "φ(x)  =  ( x₁ ,  x₂ ,  x₁² + x₂² )",
          fontsize=12, ha="center", fontfamily="monospace", color=C_BOUND)
ax_m.text(3.9, 3.75,
    "Add a 3rd feature: squared distance from the origin",
    fontsize=9.5, ha="center", color=MEDIUM, fontstyle="italic")

# ── Step 2 ──
ax_m.add_patch(patches.FancyBboxPatch(
    (8.5, 3.5), 7.2, 2.0, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))
ax_m.text(12.1, 5.1, "Step 2 — Linear classifier in φ-space",
          fontsize=11, ha="center", fontweight="bold", color=C_B)
ax_m.text(12.1, 4.3, "f(x)  =  w · φ(x)      w = (−2, −2, 1)",
          fontsize=12, ha="center", fontfamily="monospace", color=C_BOUND)
ax_m.text(12.1, 3.75,
    "Just a dot product — a completely linear operation",
    fontsize=9.5, ha="center", color=MEDIUM, fontstyle="italic")

# ── Step 3 ──
ax_m.add_patch(patches.FancyBboxPatch(
    (0.3, 0.3), 15.4, 2.8, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.06, edgecolor=C_RULE, linewidth=1.5))
ax_m.text(8, 2.7,
    "Step 3 — Expand the decision boundary  w · φ(x) = 0",
    fontsize=11, ha="center", fontweight="bold", color=C_RULE)

ax_m.text(8, 1.95,
    "(−2)(x₁) + (−2)(x₂) + (1)(x₁² + x₂²)  =  0",
    fontsize=11.5, ha="center", fontfamily="monospace", color=C_BOUND)
ax_m.text(8, 1.25,
    "x₁² + x₂² − 2x₁ − 2x₂  =  0",
    fontsize=11.5, ha="center", fontfamily="monospace", color=C_BOUND)
ax_m.text(8, 0.55,
    "(x₁ − 1)²  +  (x₂ − 1)²  =  2        ← a circle!",
    fontsize=12, ha="center", fontfamily="monospace",
    color=C_POS, fontweight="bold")

# Bottom summary
ax_m.text(8, -0.8,
    "Linear in feature space   =   Circular in original space",
    fontsize=14, ha="center", color=C_BOUND, fontweight="bold")
ax_m.text(8, -1.7,
    "The classifier is still just a dot product. The feature map did all the heavy lifting.",
    fontsize=11, ha="center", color=MEDIUM, fontstyle="italic")

# Arrow connecting step 1 + 2 → step 3
ax_m.annotate("", xy=(8, 3.1), xytext=(8, 3.45),
    arrowprops=dict(arrowstyle="->", color=C_DIM, lw=1.5))


# ─────────────────────────────────────────────────────────
#  ROW 4: Key Takeaway
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4, :])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-3, 3)

ax4.add_patch(patches.FancyBboxPatch(
    (1.0, -2.2), 14, 4.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax4.text(8, 1.8, "Key Insight", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("A nonlinear feature map lets a linear classifier produce curved boundaries.", True),
    ("The classifier is still linear — it just operates in a richer space.", True),
    ("But who designs the feature map?  That is the question deep learning answers:", False),
    ("learn the feature map automatically from data.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.0 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_4_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "nonlinear_feature_map.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
