"""
Lecture 4 Visual: Dead Neurons — When ReLU's Zero Region Becomes a Trap

Shows:
1. ReLU with its zero-gradient region clearly highlighted as a danger zone
2. A healthy neuron vs a dead neuron: forward + backward signal comparison
3. How a dead neuron gets stuck: the vicious cycle (zero output → zero gradient
   → zero update → still zero output)
4. Near-zero gradients in sigmoid — the same problem in a different form
5. Common remedies (Leaky ReLU, proper initialization)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ── Style ───────────────────────────────────────────────
BG      = "#1e1e2e"
TEXT    = "#cdd6f4"
SUBTLE  = "#6c7086"
MEDIUM  = "#a6adc8"
C_POS   = "#a6e3a1"
C_NEG   = "#f38ba8"
C_BOUND = "#f9e2af"
C_RULE  = "#fab387"
C_I     = "#89b4fa"
C_B     = "#cba6f7"
C_DIM   = "#585b70"
C_NODE  = "#89dceb"
C_WARN  = "#f9e2af"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


def style_ax(ax):
    ax.tick_params(colors=SUBTLE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_DIM)


# ── Data ────────────────────────────────────────────────
z = np.linspace(-6, 6, 600)
relu_z = np.maximum(0, z)
sigmoid_z = 1 / (1 + np.exp(-z))
sigmoid_grad = sigmoid_z * (1 - sigmoid_z)
leaky_relu_z = np.where(z > 0, z, 0.01 * z)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 34))
fig.suptitle("Dead Neurons:\nWhen ReLU's Zero Region Becomes a Trap",
             fontsize=18, fontweight="bold", y=0.997, color=C_BOUND)

gs = GridSpec(6, 2, figure=fig,
              height_ratios=[0.7, 3.2, 3.5, 3.2, 3.2, 1.5],
              hspace=0.22, wspace=0.28,
              top=0.96, bottom=0.02, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 0: Intro
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-0.2, 2.8)

ax0.text(8, 2.2,
    "ReLU(z) = 0 for all z ≤ 0,  and in that region the gradient is also 0.",
    fontsize=13, ha="center", color=TEXT)
ax0.text(8, 1.2,
    "If a neuron always receives negative input, it always outputs zero",
    fontsize=13, ha="center", color=C_NEG, fontweight="bold")
ax0.text(8, 0.3,
    "AND gets no learning signal to fix itself.  It is permanently dead.",
    fontsize=13, ha="center", color=C_NEG, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1: ReLU with danger zone + gradient overlay
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, :])

# ReLU
ax1.plot(z, relu_z, color=C_I, lw=3, zorder=4, label="ReLU(z)")

# Danger zone shading
ax1.fill_between(z[z <= 0], -1.5, 6.5, alpha=0.08, color=C_NEG, zorder=1)
ax1.fill_between(z[z > 0], -1.5, 6.5, alpha=0.03, color=C_POS, zorder=1)

# Gradient overlay (scaled for visibility)
grad = np.where(z > 0, 1.0, 0.0)
ax1.fill_between(z[z <= 0], 0, grad[z <= 0], alpha=0.3, color=C_NEG,
                 step="mid", label="Gradient = 0  (DEAD ZONE)")
ax1.fill_between(z[z > 0], 0, grad[z > 0], alpha=0.15, color=C_POS,
                 step="mid", label="Gradient = 1  (alive)")

# Annotations
ax1.annotate("DEAD ZONE\n\nOutput = 0\nGradient = 0\nNo update possible",
    xy=(-3, 0), xytext=(-4.2, 4),
    fontsize=11, color=C_NEG, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=2),
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=6, alpha=0.92),
    zorder=6)

ax1.annotate("ACTIVE ZONE\n\nOutput = z\nGradient = 1\nLearning works normally",
    xy=(4, 4), xytext=(3.5, 5.2),
    fontsize=11, color=C_POS, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=2),
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=6, alpha=0.92),
    zorder=6)

# Vertical line at z=0
ax1.axvline(x=0, color=C_BOUND, lw=1.5, ls="--", alpha=0.6)
ax1.text(0.15, -1.1, "z = 0", fontsize=10, color=C_BOUND, fontweight="bold")

ax1.set_xlim(-6, 6); ax1.set_ylim(-1.5, 6.5)
ax1.set_xlabel("z  (weighted input to neuron)", fontsize=12, color=C_I)
ax1.set_ylabel("output / gradient", fontsize=12, color=C_I)
ax1.set_title("ReLU: The Zero Region Is a Gradient Graveyard",
              fontsize=14, fontweight="bold", color=C_NEG, pad=10)
ax1.legend(fontsize=10, facecolor=BG, edgecolor=C_DIM,
           labelcolor=TEXT, loc="upper left")
ax1.axhline(y=0, color=C_DIM, lw=0.8, alpha=0.4)
style_ax(ax1)


# ─────────────────────────────────────────────────────────
#  ROW 2: Healthy neuron vs Dead neuron (diagram)
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2, :])
ax2.axis("off")
ax2.set_xlim(0, 16); ax2.set_ylim(-2, 7)
ax2.set_title("  Healthy Neuron vs Dead Neuron",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# ---- Healthy neuron (top) ----
hy = 4.5  # vertical center

# Input arrow
ax2.annotate("", xy=(3.5, hy), xytext=(1, hy),
    arrowprops=dict(arrowstyle="-|>", color=C_I, lw=2.5))
ax2.text(2.25, hy + 0.35, "input z = 2.5", fontsize=10, ha="center",
         color=C_I, fontweight="bold")

# Neuron circle
circle_h = plt.Circle((4.8, hy), 1.0, facecolor=C_POS, alpha=0.15,
                       edgecolor=C_POS, linewidth=2.5)
ax2.add_patch(circle_h)
ax2.text(4.8, hy + 0.2, "ReLU", fontsize=11, ha="center", va="center",
         color=C_POS, fontweight="bold")
ax2.text(4.8, hy - 0.3, "z > 0", fontsize=9, ha="center", va="center",
         color=C_POS)

# Output arrow
ax2.annotate("", xy=(8.5, hy), xytext=(6.0, hy),
    arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=2.5))
ax2.text(7.25, hy + 0.35, "output = 2.5", fontsize=10, ha="center",
         color=C_POS, fontweight="bold")

# Gradient arrow (backward)
ax2.annotate("", xy=(3.5, hy - 0.7), xytext=(6.0, hy - 0.7),
    arrowprops=dict(arrowstyle="-|>", color=C_BOUND, lw=2, ls="--"))
ax2.text(4.75, hy - 1.1, "gradient = 1  (passes through!)",
         fontsize=9, ha="center", color=C_BOUND, fontweight="bold")

# Status
ax2.add_patch(patches.FancyBboxPatch(
    (9, hy - 0.7), 5.5, 1.5, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.08, edgecolor=C_POS, linewidth=1.5))
ax2.text(11.75, hy + 0.3, "HEALTHY", fontsize=13, ha="center",
         fontweight="bold", color=C_POS)
ax2.text(11.75, hy - 0.3, "Produces output, receives gradient,",
         fontsize=9.5, ha="center", color=MEDIUM)
ax2.text(11.75, hy - 0.7, "weights can update → neuron can learn",
         fontsize=9.5, ha="center", color=MEDIUM)

# ---- Dead neuron (bottom) ----
dy = 1.0  # vertical center

# Input arrow
ax2.annotate("", xy=(3.5, dy), xytext=(1, dy),
    arrowprops=dict(arrowstyle="-|>", color=C_I, lw=2.5))
ax2.text(2.25, dy + 0.35, "input z = −3.0", fontsize=10, ha="center",
         color=C_I, fontweight="bold")

# Neuron circle (dead — dimmed/red)
circle_d = plt.Circle((4.8, dy), 1.0, facecolor=C_NEG, alpha=0.12,
                       edgecolor=C_NEG, linewidth=2.5, linestyle="--")
ax2.add_patch(circle_d)
ax2.text(4.8, dy + 0.2, "ReLU", fontsize=11, ha="center", va="center",
         color=C_NEG, fontweight="bold")
ax2.text(4.8, dy - 0.3, "z ≤ 0", fontsize=9, ha="center", va="center",
         color=C_NEG)

# Output arrow (faded)
ax2.annotate("", xy=(8.5, dy), xytext=(6.0, dy),
    arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=2.5, ls="--"))
ax2.text(7.25, dy + 0.35, "output = 0", fontsize=10, ha="center",
         color=C_DIM, fontweight="bold")

# Gradient arrow (blocked)
ax2.plot([3.5, 6.0], [dy - 0.7, dy - 0.7], color=C_DIM, lw=2, ls="--",
         alpha=0.4)
ax2.plot([4.55, 4.95], [dy - 0.5, dy - 0.9], color=C_NEG, lw=3)
ax2.plot([4.55, 4.95], [dy - 0.9, dy - 0.5], color=C_NEG, lw=3)
ax2.text(4.75, dy - 1.25, "gradient = 0  (BLOCKED!)",
         fontsize=9, ha="center", color=C_NEG, fontweight="bold")

# Status
ax2.add_patch(patches.FancyBboxPatch(
    (9, dy - 0.7), 5.5, 1.5, boxstyle="round,pad=0.15",
    facecolor=C_NEG, alpha=0.08, edgecolor=C_NEG, linewidth=1.5))
ax2.text(11.75, dy + 0.3, "DEAD", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)
ax2.text(11.75, dy - 0.3, "Outputs zero, receives no gradient,",
         fontsize=9.5, ha="center", color=MEDIUM)
ax2.text(11.75, dy - 0.7, "weights never update → permanently stuck",
         fontsize=9.5, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 3: The vicious cycle + sigmoid comparison
# ─────────────────────────────────────────────────────────

# ---- LEFT: The vicious cycle ----
ax3l = fig.add_subplot(gs[3, 0])
ax3l.axis("off")
ax3l.set_xlim(-1, 11); ax3l.set_ylim(-1.5, 9)
ax3l.set_title("The Vicious Cycle of a Dead Neuron",
               fontsize=13, fontweight="bold", color=C_NEG, pad=10)

# Cycle boxes
cycle = [
    (5, 7.5, "Neuron receives\nnegative input  z < 0",       C_I),
    (5, 5.2, "ReLU outputs 0\n(contributes nothing)",         C_NEG),
    (5, 2.9, "Gradient = 0\n(no learning signal)",            C_NEG),
    (5, 0.6, "Weights don't change\n(no update)",             C_DIM),
]
for cx, cy, txt, col in cycle:
    ax3l.add_patch(patches.FancyBboxPatch(
        (cx - 3.5, cy - 0.7), 7, 1.5, boxstyle="round,pad=0.12",
        facecolor=col, alpha=0.08, edgecolor=col, linewidth=1.5))
    ax3l.text(cx, cy, txt, fontsize=10, ha="center", va="center",
              color=col, fontweight="bold")

# Down arrows
for y_top, y_bot in [(6.7, 6.0), (4.4, 3.7), (2.1, 1.4)]:
    ax3l.annotate("", xy=(5, y_bot), xytext=(5, y_top),
        arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=2))

# Loop-back arrow (right side)
ax3l.annotate("", xy=(9.2, 7.5), xytext=(9.2, 0.6),
    arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=2.5,
                    connectionstyle="arc3,rad=-0.5"))
ax3l.text(10.2, 4.0, "STUCK\nFOREVER", fontsize=10, ha="center",
          color=C_NEG, fontweight="bold", rotation=-90)

# ---- RIGHT: Sigmoid comparison ----
ax3r = fig.add_subplot(gs[3, 1])

# Sigmoid
ax3r.plot(z, sigmoid_z, color=C_B, lw=2.5, label="σ(z)  (sigmoid)",
          zorder=4)

# Sigmoid gradient
ax3r.plot(z, sigmoid_grad, color=C_RULE, lw=2.5, label="σ'(z)  (gradient)",
          zorder=4)

# Shade near-zero gradient regions
ax3r.fill_between(z[z < -3], sigmoid_grad[z < -3], 0,
                  alpha=0.12, color=C_NEG)
ax3r.fill_between(z[z > 3], sigmoid_grad[z > 3], 0,
                  alpha=0.12, color=C_NEG)

# Annotations
ax3r.annotate("Gradient ≈ 0\nLearning is\npainfully slow",
    xy=(-4.5, 0.01), xytext=(-4.5, 0.4),
    fontsize=9.5, color=C_NEG, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.2),
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=4, alpha=0.9))

ax3r.annotate("Gradient ≈ 0\nSame problem\non this side",
    xy=(4.5, 0.01), xytext=(4.5, 0.4),
    fontsize=9.5, color=C_NEG, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.2),
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=4, alpha=0.9))

ax3r.annotate("Peak gradient\nonly 0.25 !",
    xy=(0, 0.25), xytext=(2.5, 0.6),
    fontsize=9.5, color=C_RULE, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_RULE, lw=1.2),
    bbox=dict(facecolor=BG, edgecolor=C_RULE, pad=4, alpha=0.9))

ax3r.set_xlim(-6, 6); ax3r.set_ylim(-0.08, 1.1)
ax3r.set_xlabel("z", fontsize=12, color=C_I)
ax3r.set_ylabel("value", fontsize=12, color=C_I)
ax3r.set_title("Sigmoid: Near-Zero Gradients Are Almost as Bad",
               fontsize=13, fontweight="bold", color=C_B, pad=10)
ax3r.legend(fontsize=9, facecolor=BG, edgecolor=C_DIM,
            labelcolor=TEXT, loc="center right")
ax3r.axhline(y=0, color=C_DIM, lw=0.8, alpha=0.4)
ax3r.axvline(x=0, color=C_DIM, lw=0.8, alpha=0.4)
style_ax(ax3r)


# ─────────────────────────────────────────────────────────
#  ROW 4: Remedies — Leaky ReLU + proper initialization
# ─────────────────────────────────────────────────────────

# ---- LEFT: Leaky ReLU ----
ax4l = fig.add_subplot(gs[4, 0])

ax4l.plot(z, relu_z, color=C_DIM, lw=2, ls="--", alpha=0.5,
          label="ReLU  (dead zone)")
ax4l.plot(z, leaky_relu_z, color=C_POS, lw=3, zorder=4,
          label="Leaky ReLU  α = 0.01")

# Shade the small-slope region
ax4l.fill_between(z[z <= 0], 0, leaky_relu_z[z <= 0],
                  alpha=0.08, color=C_POS)

ax4l.annotate("Tiny slope (α = 0.01)\ninstead of flat zero —\ngradient stays alive!",
    xy=(-4, -0.04), xytext=(-3.5, 3),
    fontsize=10, color=C_POS, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_POS, lw=1.5),
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=5, alpha=0.9))

# Leaky ReLU equation
ax4l.text(3, 5, "Leaky ReLU(z) =", fontsize=10, color=C_POS,
          fontweight="bold",
          bbox=dict(facecolor=BG, edgecolor="none", pad=3))
ax4l.text(3, 4.2, "  z          if z > 0", fontsize=10,
          fontfamily="monospace", color=C_POS)
ax4l.text(3, 3.5, "  α · z      if z ≤ 0", fontsize=10,
          fontfamily="monospace", color=C_POS)

ax4l.set_xlim(-6, 6); ax4l.set_ylim(-1, 6)
ax4l.set_xlabel("z", fontsize=12, color=C_I)
ax4l.set_ylabel("output", fontsize=12, color=C_I)
ax4l.set_title("Remedy 1: Leaky ReLU — No More Dead Zone",
               fontsize=13, fontweight="bold", color=C_POS, pad=10)
ax4l.legend(fontsize=9, facecolor=BG, edgecolor=C_DIM,
            labelcolor=TEXT, loc="upper left")
ax4l.axhline(y=0, color=C_DIM, lw=0.8, alpha=0.4)
ax4l.axvline(x=0, color=C_DIM, lw=0.8, alpha=0.4)
style_ax(ax4l)


# ---- RIGHT: Initialization ----
ax4r = fig.add_subplot(gs[4, 1])
ax4r.axis("off")
ax4r.set_xlim(0, 10); ax4r.set_ylim(-1, 9)
ax4r.set_title("Remedy 2: Proper Initialization",
               fontsize=13, fontweight="bold", color=C_NODE, pad=10)

remedies = [
    ("Scale weights by  1 / √d_in", C_NODE,
     "Keeps the initial weighted sums near zero — a mix of\n"
     "positive and negative values, so most neurons start alive."),
    ("Avoid large weight magnitudes", C_WARN,
     "Large weights push inputs far into the negative region,\n"
     "causing neurons to start dead from the very first step."),
    ("Use bias = 0 or small positive bias", C_B,
     "A small positive bias gives each neuron a slight head start\n"
     "toward the active zone, reducing initial dead neurons."),
]

for i, (title, col, desc) in enumerate(remedies):
    y = 7.5 - i * 3.0
    ax4r.add_patch(patches.FancyBboxPatch(
        (0.3, y - 1.2), 9.4, 2.5, boxstyle="round,pad=0.15",
        facecolor=col, alpha=0.06, edgecolor=col, linewidth=1.5))
    ax4r.text(5, y + 0.8, title, fontsize=11, ha="center",
              fontweight="bold", color=col)
    ax4r.text(5, y - 0.3, desc, fontsize=9.5, ha="center",
              color=MEDIUM, linespacing=1.4)


# ─────────────────────────────────────────────────────────
#  ROW 5: Key Takeaway
# ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[5, :])
ax5.axis("off")
ax5.set_xlim(0, 16); ax5.set_ylim(-3, 3)

ax5.add_patch(patches.FancyBboxPatch(
    (1.0, -2.2), 14, 4.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax5.text(8, 1.8, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("A dead neuron always outputs 0, receives 0 gradient, and can never recover.", True),
    ("This wastes model capacity and can stall training entirely.", True),
    ("Near-zero gradients (sigmoid) are almost as dangerous as exact zeros.", False),
    ("Remedies: Leaky ReLU, careful initialization, and proper weight scaling.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax5.text(8, 1.0 - i * 0.6, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_4_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "dead_neurons.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
