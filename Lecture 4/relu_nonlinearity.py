"""
Lecture 4 Visual: ReLU — Why Nonlinearity Is the Turning Point

Shows:
1. The ReLU function plotted clearly with dead-zone / active-zone annotations
2. The ReLU gradient (step function) — why dead neurons get no signal
3. Without ReLU: same 3-neuron hidden layer collapses to a straight line
4. With ReLU: the same weights produce a piecewise-linear function with bends
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

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


def style_ax(ax):
    ax.tick_params(colors=SUBTLE, labelsize=9)
    ax.axhline(y=0, color=C_DIM, lw=0.8, alpha=0.4)
    ax.axvline(x=0, color=C_DIM, lw=0.8, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color(C_DIM)


# ── Data ────────────────────────────────────────────────
z = np.linspace(-6, 6, 600)
relu_z = np.maximum(0, z)

x = np.linspace(-4, 4, 600)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 28))
fig.suptitle("ReLU: Why Nonlinearity Is the Turning Point",
             fontsize=18, fontweight="bold", y=0.995, color=C_BOUND)

gs = GridSpec(4, 2, figure=fig,
              height_ratios=[0.8, 3.5, 3.8, 1.6],
              hspace=0.22, wspace=0.28,
              top=0.96, bottom=0.02, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 0: Intro
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-0.2, 3)

ax0.text(8, 2.4,
    "Stacking linear layers without activation = still just one linear layer.",
    fontsize=13, ha="center", color=TEXT)
ax0.text(8, 1.4,
    "\"No matter how many transparent sheets of glass you stack, you see straight through.\"",
    fontsize=12, ha="center", color=MEDIUM, fontstyle="italic")
ax0.text(8, 0.3,
    "ReLU breaks this by introducing a single bend — the simplest useful nonlinearity.",
    fontsize=13, ha="center", color=C_I, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: The ReLU function
# ─────────────────────────────────────────────────────────
ax_r = fig.add_subplot(gs[1, 0])

# Identity reference
ax_r.plot(z, z, color=C_DIM, lw=1.2, ls="--", alpha=0.35,
          label="y = z  (identity)")

# ReLU curve
ax_r.plot(z, relu_z, color=C_I, lw=3, label="ReLU(z) = max(0, z)", zorder=4)

# Kink marker
ax_r.plot(0, 0, 'o', color=C_BOUND, ms=10, zorder=5,
          markeredgecolor="white", markeredgewidth=1.5)

# Shade dead / active zones
ax_r.fill_between(z[z <= 0], relu_z[z <= 0], -2.5, alpha=0.06, color=C_NEG)
ax_r.fill_between(z[z >= 0], relu_z[z >= 0], 0, alpha=0.04, color=C_POS)

# Annotations
ax_r.annotate("Dead zone\noutput = 0",
    xy=(-3, 0), xytext=(-4.5, 3),
    fontsize=10, color=C_NEG, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.5),
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=4, alpha=0.9))

ax_r.annotate("Active zone\noutput = z  (unchanged)",
    xy=(4, 4), xytext=(2.2, 5.3),
    fontsize=10, color=C_POS, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_POS, lw=1.5),
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=4, alpha=0.9))

ax_r.annotate("The kink at z = 0",
    xy=(0, 0), xytext=(2.5, -1.5),
    fontsize=10, color=C_BOUND, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_BOUND, lw=1.5),
    bbox=dict(facecolor=BG, edgecolor=C_BOUND, pad=4, alpha=0.9))

ax_r.set_xlim(-6, 6); ax_r.set_ylim(-2.5, 6.5)
ax_r.set_xlabel("z  (input to neuron)", fontsize=12, color=C_I)
ax_r.set_ylabel("ReLU(z)", fontsize=12, color=C_I)
ax_r.set_title("The ReLU Function",
               fontsize=14, fontweight="bold", color=C_I, pad=10)
ax_r.legend(fontsize=9, facecolor=BG, edgecolor=C_DIM,
            labelcolor=TEXT, loc="upper left")
style_ax(ax_r)


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: ReLU gradient
# ─────────────────────────────────────────────────────────
ax_g = fig.add_subplot(gs[1, 1])

z_neg = z[z < 0]
z_pos = z[z > 0]

ax_g.plot(z_neg, np.zeros_like(z_neg), color=C_NEG, lw=3, zorder=4)
ax_g.plot(z_pos, np.ones_like(z_pos), color=C_POS, lw=3, zorder=4)

# Discontinuity dots
ax_g.plot(0, 0, 'o', color=C_NEG, ms=9, zorder=5,
          markeredgecolor="white", markeredgewidth=1.5)
ax_g.plot(0, 1, 'o', color=C_POS, ms=9, zorder=5,
          markeredgecolor="white", markeredgewidth=1.5)

# Subtle shading
ax_g.fill_between(z_neg, 0, -0.15, alpha=0.06, color=C_NEG)
ax_g.fill_between(z_pos, 1, 1.15, alpha=0.04, color=C_POS)

ax_g.text(-3, 0.25,
    "Gradient = 0\nNo learning signal\npasses backward",
    fontsize=10, color=C_NEG, fontweight="bold", ha="center",
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=5, alpha=0.9))

ax_g.text(3, 0.75,
    "Gradient = 1\nFull signal passes\nthrough to earlier layers",
    fontsize=10, color=C_POS, fontweight="bold", ha="center",
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=5, alpha=0.9))

ax_g.set_xlim(-6, 6); ax_g.set_ylim(-0.3, 1.45)
ax_g.set_xlabel("z  (input to neuron)", fontsize=12, color=C_I)
ax_g.set_ylabel("∂ReLU / ∂z", fontsize=12, color=C_B)
ax_g.set_title("ReLU Gradient (Derivative)",
               fontsize=14, fontweight="bold", color=C_B, pad=10)
ax_g.set_yticks([0, 1])
style_ax(ax_g)


# ─────────────────────────────────────────────────────────
#  ROW 2, LEFT: Without ReLU — everything stays linear
# ─────────────────────────────────────────────────────────
ax_no = fig.add_subplot(gs[2, 0])

# Architecture: 3 hidden neurons, output = h1 - 2*h2 + h3
# h1 = 0.5x + 1,  h2 = -0.5x,  h3 = 0.5x - 1
# WITHOUT ReLU: output = (0.5x+1) + (-2)(-0.5x) + (0.5x-1) = 2x
h1_lin = 0.5 * x + 1
h2_lin = -0.5 * x               # raw hidden value
h3_lin = 0.5 * x - 1
out_lin = h1_lin + (-2) * h2_lin + h3_lin   # = 2x

# Individual weighted contributions (faint)
ax_no.plot(x, h1_lin,        color=C_POS, lw=1, ls="--", alpha=0.3,
           label="h₁ = 0.5x + 1")
ax_no.plot(x, -2 * h2_lin,  color=C_NEG, lw=1, ls="--", alpha=0.3,
           label="−2 h₂ = x")
ax_no.plot(x, h3_lin,        color=C_B,  lw=1, ls="--", alpha=0.3,
           label="h₃ = 0.5x − 1")

# Composed output
ax_no.plot(x, out_lin, color=C_NEG, lw=3, zorder=4,
           label="Sum = 2x  (straight line)")

ax_no.set_xlim(-4, 4); ax_no.set_ylim(-8, 8)
ax_no.set_xlabel("x", fontsize=12, color=C_I)
ax_no.set_ylabel("output", fontsize=12, color=C_I)
ax_no.set_title("Without ReLU — Same Weights, Still a Line",
                fontsize=13, fontweight="bold", color=C_NEG, pad=10)
ax_no.legend(fontsize=8.5, facecolor=BG, edgecolor=C_DIM,
             labelcolor=TEXT, loc="upper left")
style_ax(ax_no)

ax_no.text(0, -6.5,
    "Three linear neurons sum to one line.\n"
    "No depth advantage at all.",
    fontsize=10, ha="center", color=C_NEG, fontweight="bold",
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=5, alpha=0.9))


# ─────────────────────────────────────────────────────────
#  ROW 2, RIGHT: With ReLU — bends appear
# ─────────────────────────────────────────────────────────
ax_yes = fig.add_subplot(gs[2, 1])

# SAME weights, but with ReLU after hidden layer
h1_relu = np.maximum(0, 0.5 * x + 1)     # activates at x = -2
h2_relu = np.maximum(0, -0.5 * x)         # activates for x < 0
h3_relu = np.maximum(0, 0.5 * x - 1)     # activates at x = 2
out_relu = h1_relu + (-2) * h2_relu + h3_relu

# Individual weighted contributions (faint)
ax_yes.plot(x, h1_relu,        color=C_POS, lw=1, ls="--", alpha=0.35,
            label="ReLU(0.5x + 1)")
ax_yes.plot(x, -2 * h2_relu,  color=C_NEG, lw=1, ls="--", alpha=0.35,
            label="−2 · ReLU(−0.5x)")
ax_yes.plot(x, h3_relu,        color=C_B,  lw=1, ls="--", alpha=0.35,
            label="ReLU(0.5x − 1)")

# Combined output
ax_yes.plot(x, out_relu, color=C_POS, lw=3, zorder=4,
            label="Sum  (piecewise linear!)")

# Mark the kinks
kinks_x = [-2, 0, 2]
for kx in kinks_x:
    ky = (max(0, 0.5*kx+1) + (-2)*max(0, -0.5*kx)
          + max(0, 0.5*kx-1))
    ax_yes.plot(kx, ky, 'o', color=C_BOUND, ms=9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.5)

# Slope annotations between kinks
slope_info = [
    (-3.2, "slope = 1"),
    (-1.0, "slope = 1.5"),
    ( 1.0, "slope = 0.5"),
    ( 3.2, "slope = 1"),
]
for sx, stxt in slope_info:
    sy = (max(0, 0.5*sx+1) + (-2)*max(0, -0.5*sx)
          + max(0, 0.5*sx-1))
    ax_yes.annotate(stxt, xy=(sx, sy), xytext=(sx, sy + 1.8),
        fontsize=8.5, color=C_BOUND, ha="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_BOUND, lw=1, alpha=0.7),
        bbox=dict(facecolor=BG, edgecolor=C_BOUND, pad=3, alpha=0.85))

ax_yes.set_xlim(-4, 4); ax_yes.set_ylim(-8, 8)
ax_yes.set_xlabel("x", fontsize=12, color=C_I)
ax_yes.set_ylabel("output", fontsize=12, color=C_I)
ax_yes.set_title("With ReLU — Same Weights, Bends Appear!",
                 fontsize=13, fontweight="bold", color=C_POS, pad=10)
ax_yes.legend(fontsize=8.5, facecolor=BG, edgecolor=C_DIM,
              labelcolor=TEXT, loc="upper left")
style_ax(ax_yes)

ax_yes.text(0, -6.5,
    "Each ReLU neuron adds a possible bend.\n"
    "More neurons = more bends = richer shapes.",
    fontsize=10, ha="center", color=C_POS, fontweight="bold",
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=5, alpha=0.9))


# ─────────────────────────────────────────────────────────
#  ROW 3: Key Takeaway
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3, :])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-3, 3)

ax4.add_patch(patches.FancyBboxPatch(
    (1.0, -2.2), 14, 4.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax4.text(8, 1.8, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("ReLU(z) = max(0, z) — the simplest nonlinearity that works.", True),
    ("Without it, any depth of linear layers collapses to a single layer.", True),
    ("With it, each neuron adds a potential bend. Together they can", False),
    ("approximate arbitrarily complex piecewise-linear functions.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.0 - i * 0.6, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_4_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "relu_nonlinearity.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
