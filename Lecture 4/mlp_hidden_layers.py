"""
Lecture 4 Visual: Multi-Layer Perceptrons — Why Hidden Layers Help

Shows:
1. XOR data that a linear model cannot separate — any line fails
2. An MLP with one hidden layer that solves it cleanly
3. THE KEY INSIGHT: the hidden layer remaps data into a new space
   where a straight boundary suddenly works
4. Why going deeper builds increasingly abstract features
5. The catch: vanishing and exploding gradients
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba

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
    for spine in ax.spines.values():
        spine.set_color(C_DIM)


# ── XOR data ────────────────────────────────────────────
np.random.seed(42)
n_per = 80
X = np.vstack([
    np.random.randn(n_per, 2) * 0.13 + [0, 0],   # class A
    np.random.randn(n_per, 2) * 0.13 + [1, 1],   # class A
    np.random.randn(n_per, 2) * 0.13 + [0, 1],   # class B
    np.random.randn(n_per, 2) * 0.13 + [1, 0],   # class B
])
y = np.array([0]*n_per + [0]*n_per + [1]*n_per + [1]*n_per)
cA = y == 0   # class A: (0,0) and (1,1) corners
cB = y == 1   # class B: (0,1) and (1,0) corners

# ── Hand-crafted MLP that solves XOR ────────────────────
# Hidden layer: 2 neurons
#   h1 = ReLU(x1 + x2 - 1.5)   → fires when both inputs high
#   h2 = ReLU(-x1 - x2 + 0.5)  → fires when both inputs low
# Output: score = 5*h1 + 5*h2 - 1
#   class A when score > 0,  class B when score < 0
W1 = np.array([[1, 1], [-1, -1]])
b1 = np.array([-1.5, 0.5])
W2 = np.array([5.0, 5.0])
b2 = -1.0

# Hidden representations for all data
H = np.maximum(0, X @ W1.T + b1)

# Grid for decision boundaries
res = 250
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, res),
                      np.linspace(-0.5, 1.5, res))
grid = np.column_stack([xx.ravel(), yy.ravel()])

linear_score = (grid[:, 0] + grid[:, 1] - 1.0).reshape(xx.shape)

H_grid = np.maximum(0, grid @ W1.T + b1)
mlp_score = (H_grid @ W2 + b2).reshape(xx.shape)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 34))
fig.suptitle("Multi-Layer Perceptrons:\nWhy Hidden Layers Change Everything",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(5, 2, figure=fig,
              height_ratios=[0.7, 3.2, 3.8, 3.8, 1.5],
              hspace=0.22, wspace=0.28,
              top=0.957, bottom=0.02, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 0: Intro
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-0.2, 2.8)

ax0.text(8, 2.2,
    "A linear model learns weights and classifies.  An MLP does something deeper:",
    fontsize=13, ha="center", color=TEXT)
ax0.text(8, 1.2,
    "the hidden layer learns a new REPRESENTATION of the data,",
    fontsize=13, ha="center", color=C_I, fontweight="bold")
ax0.text(8, 0.3,
    "then a linear classifier on that representation does the final job.",
    fontsize=13, ha="center", color=C_I, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: Linear model fails on XOR
# ─────────────────────────────────────────────────────────
ax_lin = fig.add_subplot(gs[1, 0])

# Decision regions
ax_lin.contourf(xx, yy, linear_score, levels=[-100, 0, 100],
                colors=[to_rgba(C_NEG, 0.08), to_rgba(C_POS, 0.08)])
ax_lin.contour(xx, yy, linear_score, levels=[0],
               colors=[C_BOUND], linewidths=2.5, linestyles="--")

# Data
ax_lin.scatter(X[cA, 0], X[cA, 1], c=C_POS, s=25, alpha=0.8,
               edgecolors="none", label="Class A  (0,0) & (1,1)", zorder=4)
ax_lin.scatter(X[cB, 0], X[cB, 1], c=C_NEG, s=25, alpha=0.8,
               edgecolors="none", label="Class B  (0,1) & (1,0)", zorder=4)

# Cluster labels
for (cx, cy), lbl in zip([(0,0),(1,1),(0,1),(1,0)], ["A","A","B","B"]):
    col = C_POS if lbl == "A" else C_NEG
    ax_lin.text(cx, cy - 0.32, lbl, fontsize=10, ha="center",
                color=col, fontweight="bold",
                bbox=dict(facecolor=BG, edgecolor=col, pad=2, alpha=0.8))

ax_lin.set_xlim(-0.5, 1.5); ax_lin.set_ylim(-0.5, 1.5)
ax_lin.set_xlabel("$x_1$", fontsize=12, color=C_I)
ax_lin.set_ylabel("$x_2$", fontsize=12, color=C_I)
ax_lin.set_title("Linear Model — Fails on XOR",
                 fontsize=13, fontweight="bold", color=C_NEG, pad=10)
ax_lin.legend(fontsize=8, facecolor=BG, edgecolor=C_DIM,
              labelcolor=TEXT, loc="upper right")
ax_lin.set_aspect("equal")
style_ax(ax_lin)

ax_lin.text(0.5, -0.4,
    "Every straight line leaves\nmixed classes on both sides!",
    fontsize=10, ha="center", color=C_NEG, fontweight="bold",
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=4, alpha=0.9))


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: MLP solves XOR
# ─────────────────────────────────────────────────────────
ax_mlp = fig.add_subplot(gs[1, 1])

# Decision regions
ax_mlp.contourf(xx, yy, mlp_score, levels=[-100, 0, 100],
                colors=[to_rgba(C_NEG, 0.10), to_rgba(C_POS, 0.10)])
ax_mlp.contour(xx, yy, mlp_score, levels=[0],
               colors=[C_BOUND], linewidths=2.5)

# Data
ax_mlp.scatter(X[cA, 0], X[cA, 1], c=C_POS, s=25, alpha=0.8,
               edgecolors="none", label="Class A", zorder=4)
ax_mlp.scatter(X[cB, 0], X[cB, 1], c=C_NEG, s=25, alpha=0.8,
               edgecolors="none", label="Class B", zorder=4)

# Annotate the two boundary lines
ax_mlp.text(0.25, 1.35, "$x_1 + x_2 = 1.7$", fontsize=9, color=C_BOUND,
            fontweight="bold", rotation=-45)
ax_mlp.text(-0.15, 0.65, "$x_1 + x_2 = 0.3$", fontsize=9, color=C_BOUND,
            fontweight="bold", rotation=-45)

ax_mlp.set_xlim(-0.5, 1.5); ax_mlp.set_ylim(-0.5, 1.5)
ax_mlp.set_xlabel("$x_1$", fontsize=12, color=C_I)
ax_mlp.set_ylabel("$x_2$", fontsize=12, color=C_I)
ax_mlp.set_title("MLP (1 Hidden Layer) — Solves XOR!",
                 fontsize=13, fontweight="bold", color=C_POS, pad=10)
ax_mlp.legend(fontsize=8, facecolor=BG, edgecolor=C_DIM,
              labelcolor=TEXT, loc="upper right")
ax_mlp.set_aspect("equal")
style_ax(ax_mlp)

ax_mlp.text(0.5, -0.4,
    "Two hidden neurons → two boundary lines\n→ a nonlinear decision strip!",
    fontsize=10, ha="center", color=C_POS, fontweight="bold",
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=4, alpha=0.9))


# ─────────────────────────────────────────────────────────
#  ROW 2: THE KEY INSIGHT — input space vs hidden space
# ─────────────────────────────────────────────────────────

# ---- LEFT: Input space ----
ax_in = fig.add_subplot(gs[2, 0])

ax_in.scatter(X[cA, 0], X[cA, 1], c=C_POS, s=30, alpha=0.8,
              edgecolors="none", label="Class A", zorder=4)
ax_in.scatter(X[cB, 0], X[cB, 1], c=C_NEG, s=30, alpha=0.8,
              edgecolors="none", label="Class B", zorder=4)

# Draw arrows connecting to hidden space (for a few points)
for (cx, cy), lbl in zip([(0, 0), (1, 1), (0, 1), (1, 0)],
                          ["A", "A", "B", "B"]):
    col = C_POS if lbl == "A" else C_NEG
    ax_in.add_patch(plt.Circle((cx, cy), 0.22, fill=False,
                    edgecolor=col, lw=1.5, ls="--", alpha=0.5))
    ax_in.text(cx, cy + 0.3, f"({cx},{cy})", fontsize=9, ha="center",
               color=col, fontweight="bold")

ax_in.set_xlim(-0.5, 1.5); ax_in.set_ylim(-0.6, 1.6)
ax_in.set_xlabel("$x_1$", fontsize=12, color=C_I)
ax_in.set_ylabel("$x_2$", fontsize=12, color=C_I)
ax_in.set_title("Input Space — NOT Linearly Separable",
                fontsize=13, fontweight="bold", color=C_NEG, pad=10)
ax_in.legend(fontsize=9, facecolor=BG, edgecolor=C_DIM,
             labelcolor=TEXT, loc="upper right")
ax_in.set_aspect("equal")
style_ax(ax_in)

ax_in.text(0.5, -0.5,
    "Green and pink are interleaved.\nNo single line can separate them.",
    fontsize=10, ha="center", color=MEDIUM,
    bbox=dict(facecolor=BG, edgecolor=C_DIM, pad=4, alpha=0.9))

# ---- RIGHT: Hidden space ----
ax_hid = fig.add_subplot(gs[2, 1])

ax_hid.scatter(H[cA, 0], H[cA, 1], c=C_POS, s=30, alpha=0.8,
               edgecolors="none", label="Class A", zorder=4)
ax_hid.scatter(H[cB, 0], H[cB, 1], c=C_NEG, s=30, alpha=0.8,
               edgecolors="none", label="Class B", zorder=4)

# Separating line in hidden space: h1 + h2 = 0.2
hline = np.linspace(-0.05, 0.7, 100)
ax_hid.plot(hline, 0.2 - hline, color=C_BOUND, lw=2.5,
            label="$h_1 + h_2 = 0.2$  (linear boundary)", zorder=3)

# Label clusters
ax_hid.annotate("(0,0) cluster\nmoved here",
    xy=(0.02, 0.45), xytext=(0.25, 0.6),
    fontsize=9, color=C_POS, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_POS, lw=1),
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=3, alpha=0.9))

ax_hid.annotate("(1,1) cluster\nmoved here",
    xy=(0.45, 0.02), xytext=(0.6, 0.25),
    fontsize=9, color=C_POS, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_POS, lw=1),
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=3, alpha=0.9))

ax_hid.annotate("(0,1) & (1,0)\ncollapsed to origin",
    xy=(0.01, 0.01), xytext=(0.35, -0.18),
    fontsize=9, color=C_NEG, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1),
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=3, alpha=0.9))

ax_hid.set_xlim(-0.1, 0.75); ax_hid.set_ylim(-0.25, 0.75)
ax_hid.set_xlabel("$h_1$ = ReLU($x_1 + x_2 - 1.5$)", fontsize=10, color=C_I)
ax_hid.set_ylabel("$h_2$ = ReLU($-x_1 - x_2 + 0.5$)", fontsize=10, color=C_I)
ax_hid.set_title("Hidden Space — NOW Linearly Separable!",
                 fontsize=13, fontweight="bold", color=C_POS, pad=10)
ax_hid.legend(fontsize=8.5, facecolor=BG, edgecolor=C_DIM,
              labelcolor=TEXT, loc="upper right")
style_ax(ax_hid)

ax_hid.text(0.33, -0.2,
    "The hidden layer REMAPPED the data.\nA straight line now works!",
    fontsize=10, ha="center", color=C_POS, fontweight="bold",
    bbox=dict(facecolor=BG, edgecolor=C_POS, pad=4, alpha=0.9))

# ---- Transformation arrow between the two panels ----
fig.text(0.50, 0.545, "h = ReLU(W₁x + b₁)",
         fontsize=13, ha="center", va="center",
         color=C_BOUND, fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor=C_BOUND, pad=6, alpha=0.9))
fig.text(0.50, 0.525, "→  hidden layer  →",
         fontsize=11, ha="center", va="center",
         color=C_BOUND, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 3, LEFT: Why deeper helps — abstraction hierarchy
# ─────────────────────────────────────────────────────────
ax3l = fig.add_subplot(gs[3, 0])
ax3l.axis("off")
ax3l.set_xlim(-0.5, 10.5); ax3l.set_ylim(-2, 9)
ax3l.set_title("Why Deeper Helps: Abstraction Hierarchy",
               fontsize=13, fontweight="bold", color=C_B, pad=10)

# Pipeline boxes
stages = [
    (1.5, 7, "Raw Input",    "pixels, numbers",          C_DIM,  1.0),
    (1.5, 5, "Layer 1",      "detects simple patterns\n(edges, thresholds)",  C_I,    1.0),
    (1.5, 3, "Layer 2",      "combines into parts\n(shapes, motifs)",   C_B,    1.0),
    (1.5, 1, "Layer 3",      "recognizes objects\n(faces, digits)",     C_POS,  1.0),
]
for cx, cy, title, desc, col, _ in stages:
    ax3l.add_patch(patches.FancyBboxPatch(
        (cx - 1.3, cy - 0.7), 8.5, 1.5, boxstyle="round,pad=0.12",
        facecolor=col, alpha=0.07, edgecolor=col, linewidth=1.5))
    ax3l.text(cx + 0.5, cy + 0.2, title, fontsize=11,
              fontweight="bold", color=col)
    ax3l.text(cx + 4.5, cy, desc, fontsize=9.5, color=MEDIUM,
              ha="center", va="center")

# Down arrows
for y_top, y_bot in [(6.2, 5.8), (4.2, 3.8), (2.2, 1.8)]:
    ax3l.annotate("", xy=(5, y_bot), xytext=(5, y_top),
        arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=2))

# Key annotation
ax3l.text(5, -1.2,
    "Each layer builds more abstract features\nfrom the previous layer's output.",
    fontsize=10.5, ha="center", color=C_B, fontweight="bold",
    bbox=dict(facecolor=BG, edgecolor=C_B, pad=5, alpha=0.9))


# ─────────────────────────────────────────────────────────
#  ROW 3, RIGHT: The catch — vanishing / exploding gradients
# ─────────────────────────────────────────────────────────
ax3r = fig.add_subplot(gs[3, 1])

L = np.arange(1, 21)
w_vanish = 0.8 ** L
w_stable = 1.0 ** L
w_explode = 1.2 ** L

ax3r.semilogy(L, w_vanish, color=C_NEG, lw=2.5,
              label="w = 0.8  →  vanishes", marker="o", ms=4)
ax3r.semilogy(L, w_stable, color=C_POS, lw=2.5,
              label="w = 1.0  →  stable", marker="s", ms=4)
ax3r.semilogy(L, w_explode, color=C_RULE, lw=2.5,
              label="w = 1.2  →  explodes", marker="^", ms=4)

# Annotations
ax3r.annotate(f"0.8²⁰ = {0.8**20:.4f}\n(signal dies)",
    xy=(20, 0.8**20), xytext=(14, 0.001),
    fontsize=9, color=C_NEG, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1),
    bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=3, alpha=0.9))

ax3r.annotate(f"1.2²⁰ = {1.2**20:.0f}\n(signal explodes)",
    xy=(20, 1.2**20), xytext=(14, 15),
    fontsize=9, color=C_RULE, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_RULE, lw=1),
    bbox=dict(facecolor=BG, edgecolor=C_RULE, pad=3, alpha=0.9))

ax3r.set_xlim(0.5, 20.5)
ax3r.set_xlabel("Number of layers (depth)", fontsize=12, color=C_I)
ax3r.set_ylabel("Signal magnitude  ($w^L$)", fontsize=12, color=C_I)
ax3r.set_title("The Catch: Gradient Instability with Depth",
               fontsize=13, fontweight="bold", color=C_RULE, pad=10)
ax3r.legend(fontsize=9, facecolor=BG, edgecolor=C_DIM,
            labelcolor=TEXT, loc="center left")
ax3r.axhline(y=1, color=C_DIM, lw=1, ls="--", alpha=0.4)
style_ax(ax3r)

ax3r.text(10.5, 0.00003,
    "Depth is powerful but fragile.\n"
    "Even slightly off from 1.0, signals\n"
    "vanish or explode exponentially.",
    fontsize=9.5, ha="center", color=C_RULE, fontweight="bold",
    bbox=dict(facecolor=BG, edgecolor=C_RULE, pad=5, alpha=0.9))


# ─────────────────────────────────────────────────────────
#  ROW 4: Key Takeaway
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4, :])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-3.5, 3)

ax4.add_patch(patches.FancyBboxPatch(
    (0.5, -3.0), 15, 5.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax4.text(8, 2.0, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("A hidden layer is not just \"more parameters\" — it learns a new", True),
    ("representation of the data in which the problem becomes easier.", True),
    ("Deeper layers build increasingly abstract features (edges → parts → objects).", False),
    ("But depth amplifies gradient instability — signals vanish or explode.", False),
    ("Residual connections + normalization + careful init tame this.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.1 - i * 0.7, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_4_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "mlp_hidden_layers.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
