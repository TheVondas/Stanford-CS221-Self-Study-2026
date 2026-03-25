"""
Lecture 2 Visual: Gradient Descent on a Convex Surface

Shows:
1. 3D surface of a convex loss function L(w, b) — the "bowl"
2. Contour plot (top-down view) with gradient descent steps
3. Step-by-step trace: how the gradient, negative gradient, and learning rate
   combine to move parameters toward the minimum
4. The learning rate tradeoff: too small, just right, too large
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# ── Style ─────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_I = "#89b4fa"       # blue
C_J = "#a6e3a1"       # green: gradient arrows
C_OUT = "#f9e2af"     # yellow
C_RULE = "#fab387"    # orange
C_K = "#f38ba8"       # pink: path
C_B = "#cba6f7"       # purple
C_WARN = "#f5c2e7"
C_DIM = "#585b70"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Loss function: L(w, b) = (w - 3)^2 + (b - 2)^2 ──────
# Simple convex bowl centered at (3, 2) with minimum = 0
def loss(w, b):
    return (w - 3)**2 + (b - 2)**2

def grad_loss(w, b):
    return np.array([2*(w - 3), 2*(b - 2)])


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
#  FIGURE
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 40))
fig.suptitle("Gradient Descent on a Convex Loss Surface",
             fontsize=17, fontweight="bold", y=0.995, color=C_OUT)

gs = GridSpec(5, 2, figure=fig,
              height_ratios=[2.2, 2.5, 2.0, 2.2, 1.0],
              hspace=0.22, wspace=0.25,
              top=0.97, bottom=0.02, left=0.05, right=0.95)


# ───────────────────────────────────────────────────────────
#  ROW 1, LEFT: 3D surface
# ───────────────────────────────────────────────────────────
ax3d = fig.add_subplot(gs[0, 0], projection='3d')
ax3d.set_facecolor(BG)

w_range = np.linspace(-1, 7, 80)
b_range = np.linspace(-2, 6, 80)
W, B = np.meshgrid(w_range, b_range)
L = loss(W, B)

ax3d.plot_surface(W, B, L, cmap='magma', alpha=0.7,
                  edgecolor='none', antialiased=True)

# Mark the minimum
ax3d.scatter([3], [2], [0], color=C_OUT, s=100, zorder=5,
             edgecolors='white', linewidths=1.5)
ax3d.text(3, 2, -3, "minimum\n(w*,b*)", fontsize=8, color=C_OUT,
          ha="center")

# Run gradient descent for the 3D path
eta = 0.15
w_path, b_path = [0.0], [-1.0]
for _ in range(15):
    g = grad_loss(w_path[-1], b_path[-1])
    w_path.append(w_path[-1] - eta * g[0])
    b_path.append(b_path[-1] - eta * g[1])
l_path = [loss(w, b) for w, b in zip(w_path, b_path)]

ax3d.plot(w_path, b_path, l_path, color=C_K, lw=2.5, zorder=10)
ax3d.scatter(w_path, b_path, l_path, color=C_K, s=25, zorder=10)
ax3d.scatter([w_path[0]], [b_path[0]], [l_path[0]], color=C_K, s=80,
             zorder=10, edgecolors='white', linewidths=1.5)

ax3d.set_xlabel("w", fontsize=10, color=C_I, labelpad=5)
ax3d.set_ylabel("b", fontsize=10, color=C_I, labelpad=5)
ax3d.set_zlabel("Loss", fontsize=10, color=C_OUT, labelpad=5)
ax3d.set_title("The Convex \"Bowl\"", fontsize=13, fontweight="bold",
               color=C_RULE, pad=10)
ax3d.view_init(elev=30, azim=225)
ax3d.tick_params(colors=SUBTLE, labelsize=7)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor(C_DIM)
ax3d.yaxis.pane.set_edgecolor(C_DIM)
ax3d.zaxis.pane.set_edgecolor(C_DIM)


# ───────────────────────────────────────────────────────────
#  ROW 1, RIGHT: Contour plot with GD path
# ───────────────────────────────────────────────────────────
ax_cont = fig.add_subplot(gs[0, 1])

contour = ax_cont.contour(W, B, L, levels=15, colors=SUBTLE,
                           linewidths=0.6, alpha=0.5)
ax_cont.contourf(W, B, L, levels=30, cmap='magma', alpha=0.3)

# GD path
ax_cont.plot(w_path, b_path, '-o', color=C_K, lw=2, markersize=5,
             markeredgecolor='white', markeredgewidth=0.8, zorder=5)
ax_cont.plot(w_path[0], b_path[0], 'o', color=C_K, markersize=10,
             markeredgecolor='white', markeredgewidth=2, zorder=6)
ax_cont.annotate("start", xy=(w_path[0], b_path[0]),
                 xytext=(w_path[0] - 1.2, b_path[0] - 0.8),
                 fontsize=9, color=C_K, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_K, lw=1.2))

# Minimum
ax_cont.plot(3, 2, '*', color=C_OUT, markersize=18,
             markeredgecolor='white', markeredgewidth=1.5, zorder=6)
ax_cont.annotate("minimum (3, 2)", xy=(3, 2),
                 xytext=(4.5, 3.5),
                 fontsize=9, color=C_OUT, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.2))

# Gradient arrows at first few steps
for i in range(min(4, len(w_path) - 1)):
    g = grad_loss(w_path[i], b_path[i])
    g_norm = np.linalg.norm(g)
    if g_norm > 0.3:
        scale = 0.4 / g_norm  # normalize arrow length
        # Negative gradient direction (where we step)
        ax_cont.annotate("",
                         xy=(w_path[i] - g[0]*scale,
                             b_path[i] - g[1]*scale),
                         xytext=(w_path[i], b_path[i]),
                         arrowprops=dict(arrowstyle="->,head_width=0.15",
                                         color=C_J, lw=1.5))

ax_cont.set_xlabel("w  (weight)", fontsize=10, color=C_I)
ax_cont.set_ylabel("b  (bias)", fontsize=10, color=C_I)
ax_cont.set_title("Top-Down View (contour lines = equal loss)",
                  fontsize=13, fontweight="bold", color=C_RULE, pad=10)
ax_cont.tick_params(colors=SUBTLE, labelsize=8)
for spine in ax_cont.spines.values():
    spine.set_color(C_DIM)

# Legend
ax_cont.plot([], [], '-o', color=C_K, lw=2, markersize=5,
             label="GD path")
ax_cont.plot([], [], color=C_J, lw=1.5,
             label="−gradient direction")
ax_cont.legend(fontsize=8, facecolor=BG, edgecolor=C_DIM,
               labelcolor=TEXT, loc="upper left")


# ───────────────────────────────────────────────────────────
#  ROW 2: Step-by-step trace of the first few iterations
# ───────────────────────────────────────────────────────────
ax_trace = fig.add_subplot(gs[1, :])
ax_trace.axis("off")
ax_trace.set_xlim(0, 16)
ax_trace.set_ylim(-7, 5.5)

ax_trace.text(0.5, 5.0, "Step-by-Step: What Happens at Each Iteration",
              fontsize=14, fontweight="bold", color=C_RULE)

ax_trace.text(0.5, 4.1,
              "Update rule:    w ← w − η · ∂L/∂w          b ← b − η · ∂L/∂b",
              fontsize=13, fontfamily="monospace", color=C_OUT,
              fontweight="bold")
ax_trace.text(0.5, 3.4,
              f"Learning rate  η = {eta}       "
              "Loss  L(w,b) = (w−3)² + (b−2)²       "
              "Start:  w=0, b=−1",
              fontsize=10.5, color=MEDIUM)

# Table header
hy = 2.4
cols = [0.5, 2.0, 4.0, 6.5, 8.5, 11.0, 13.5]
headers = ["Step", "w, b", "Loss", "∂L/∂w, ∂L/∂b", "−η·grad",
           "New w, b", "New Loss"]
h_colors = [TEXT, C_I, C_OUT, C_J, C_K, C_I, C_OUT]
for x, h, c in zip(cols, headers, h_colors):
    ax_trace.text(x, hy, h, fontsize=9.5, fontweight="bold", color=c)

ax_trace.plot([0.3, 15.5], [hy - 0.25, hy - 0.25], "-",
              color=SUBTLE, alpha=0.4, lw=1)

# Rows
w_cur, b_cur = 0.0, -1.0
for step in range(7):
    y = hy - 0.75 - step * 0.95
    l_cur = loss(w_cur, b_cur)
    g = grad_loss(w_cur, b_cur)
    step_w = -eta * g[0]
    step_b = -eta * g[1]
    w_new = w_cur + step_w
    b_new = b_cur + step_b
    l_new = loss(w_new, b_new)

    row_data = [
        (cols[0], f"{step}", TEXT),
        (cols[1], f"({w_cur:.2f}, {b_cur:.2f})", C_I),
        (cols[2], f"{l_cur:.2f}", C_OUT),
        (cols[3], f"({g[0]:.2f}, {g[1]:.2f})", C_J),
        (cols[4], f"({step_w:.2f}, {step_b:.2f})", C_K),
        (cols[5], f"({w_new:.2f}, {b_new:.2f})", C_I),
        (cols[6], f"{l_new:.2f}", C_OUT),
    ]

    for x, val, c in row_data:
        ax_trace.text(x, y, val, fontsize=9, fontfamily="monospace",
                      color=c)

    # Light divider
    if step < 6:
        ax_trace.plot([0.3, 15.5], [y - 0.35, y - 0.35], "-",
                      color=C_DIM, alpha=0.2, lw=0.5)

    w_cur, b_cur = w_new, b_new

# Annotation: loss decreasing
ax_trace.annotate("loss keeps\ndecreasing!",
                  xy=(cols[6] + 0.3, hy - 0.75 - 4 * 0.95),
                  xytext=(cols[6] + 0.3, hy - 0.75 - 1.5 * 0.95),
                  fontsize=9, color=C_OUT, fontweight="bold",
                  ha="center",
                  arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.3))


# ───────────────────────────────────────────────────────────
#  ROW 3: What the gradient and negative gradient MEAN
# ───────────────────────────────────────────────────────────
ax_meaning = fig.add_subplot(gs[2, :])
ax_meaning.axis("off")
ax_meaning.set_xlim(0, 16)
ax_meaning.set_ylim(-4, 5)

ax_meaning.text(0.5, 4.5, "Why Move Opposite the Gradient?", fontsize=14,
                fontweight="bold", color=C_RULE)

# Left: gradient direction
gx = 1.5
ax_meaning.add_patch(patches.FancyBboxPatch(
    (gx - 0.3, 0.5), 6.5, 3.3, boxstyle="round,pad=0.15",
    facecolor=C_J, alpha=0.06, edgecolor=C_J, linewidth=1.5))

ax_meaning.text(gx + 3, 3.5, "The Gradient", fontsize=13,
                ha="center", fontweight="bold", color=C_J)

ax_meaning.text(gx + 3, 2.7, "∇L  =  (∂L/∂w,  ∂L/∂b)",
                fontsize=12, ha="center", fontfamily="monospace",
                color=C_J)

ax_meaning.text(gx + 3, 1.8,
                "Points in the direction of\nSTEEPEST INCREASE of L",
                fontsize=11, ha="center", color=TEXT)

ax_meaning.text(gx + 3, 0.85,
                '"Where the hill goes UP fastest"',
                fontsize=10, ha="center", color=C_J, fontstyle="italic")

# Right: negative gradient direction
ngx = 8.5
ax_meaning.add_patch(patches.FancyBboxPatch(
    (ngx - 0.3, 0.5), 7.0, 3.3, boxstyle="round,pad=0.15",
    facecolor=C_K, alpha=0.06, edgecolor=C_K, linewidth=1.5))

ax_meaning.text(ngx + 3.2, 3.5, "The NEGATIVE Gradient", fontsize=13,
                ha="center", fontweight="bold", color=C_K)

ax_meaning.text(ngx + 3.2, 2.7, "−∇L  =  (−∂L/∂w,  −∂L/∂b)",
                fontsize=12, ha="center", fontfamily="monospace",
                color=C_K)

ax_meaning.text(ngx + 3.2, 1.8,
                "Points in the direction of\nSTEEPEST DECREASE of L",
                fontsize=11, ha="center", color=TEXT)

ax_meaning.text(ngx + 3.2, 0.85,
                '"Where the hill goes DOWN fastest"',
                fontsize=10, ha="center", color=C_K, fontstyle="italic")

# Arrow between them
ax_meaning.annotate("", xy=(ngx - 0.5, 2.2), xytext=(gx + 6.4, 2.2),
                    arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))
ax_meaning.text((gx + 6.4 + ngx - 0.5) / 2, 2.6, "flip",
                fontsize=10, ha="center", color=TEXT, fontweight="bold")

# The conclusion
ax_meaning.text(8, -0.5,
                "We want to REDUCE loss  →  move in the direction of steepest DECREASE  →  subtract the gradient",
                fontsize=11, ha="center", color=C_OUT, fontweight="bold")

# The fog analogy
ax_meaning.add_patch(patches.FancyBboxPatch(
    (1.0, -3.5), 14, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_meaning.text(8, -1.6, "Percy's Analogy: Fog on a Hill", fontsize=12,
                ha="center", fontweight="bold", color=C_B)
ax_meaning.text(8, -2.25,
                "Imagine standing on a hill in thick fog.  You can't see the whole landscape.",
                fontsize=10.5, ha="center", color=MEDIUM)
ax_meaning.text(8, -2.8,
                "But you can FEEL the slope under your feet.  The gradient is that slope.",
                fontsize=10.5, ha="center", color=MEDIUM)
ax_meaning.text(8, -3.35,
                "To go downhill, step in the direction opposite the steepest uphill slope.",
                fontsize=10.5, ha="center", color=C_B, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  ROW 4: Learning rate comparison — too small, right, too big
# ───────────────────────────────────────────────────────────

for col_idx, (eta_val, title, title_col, note) in enumerate([
    (0.02, "Too Small  (η = 0.02)", C_I, "Safe but painfully slow"),
    (0.15, "Just Right  (η = 0.15)", C_J, "Steady convergence"),
    (0.48, "Too Large  (η = 0.48)", C_K, "Overshoots / oscillates"),
]):
    if col_idx == 0:
        ax_lr = fig.add_subplot(gs[3, 0])
    elif col_idx == 1:
        # Split the right column into two
        inner_gs = gs[3, 1].subgridspec(1, 2, wspace=0.3)
        ax_lr = fig.add_subplot(inner_gs[0, 0])
    else:
        ax_lr = fig.add_subplot(inner_gs[0, 1])

    # Contours
    ax_lr.contourf(W, B, L, levels=20, cmap='magma', alpha=0.25)
    ax_lr.contour(W, B, L, levels=12, colors=SUBTLE,
                  linewidths=0.4, alpha=0.4)

    # Run GD with this learning rate
    w_p, b_p = [0.0], [-1.0]
    for _ in range(20):
        g = grad_loss(w_p[-1], b_p[-1])
        w_n = w_p[-1] - eta_val * g[0]
        b_n = b_p[-1] - eta_val * g[1]
        # Clip for display if diverging
        if abs(w_n) > 15 or abs(b_n) > 15:
            break
        w_p.append(w_n)
        b_p.append(b_n)

    ax_lr.plot(w_p, b_p, '-o', color=title_col, lw=1.8, markersize=4,
               markeredgecolor='white', markeredgewidth=0.6)
    ax_lr.plot(w_p[0], b_p[0], 'o', color=title_col, markersize=8,
               markeredgecolor='white', markeredgewidth=1.5)
    ax_lr.plot(3, 2, '*', color=C_OUT, markersize=14,
               markeredgecolor='white', markeredgewidth=1)

    ax_lr.set_xlim(-1.5, 7.5)
    ax_lr.set_ylim(-2.5, 6.5)
    ax_lr.set_title(title, fontsize=10, fontweight="bold",
                    color=title_col, pad=8)
    ax_lr.set_xlabel(note, fontsize=8.5, color=MEDIUM)
    ax_lr.tick_params(colors=SUBTLE, labelsize=6)
    for spine in ax_lr.spines.values():
        spine.set_color(C_DIM)


# ───────────────────────────────────────────────────────────
#  ROW 5: Summary
# ───────────────────────────────────────────────────────────
ax_sum = fig.add_subplot(gs[4, :])
ax_sum.axis("off")
ax_sum.set_xlim(0, 16)
ax_sum.set_ylim(-3, 3)

ax_sum.text(8, 2.5, "Summary", fontsize=14, ha="center",
            fontweight="bold", color=C_RULE)

insight_box(ax_sum, 2.0, -2.5, [
    ("Gradient = direction of steepest INCREASE      "
     "Negative gradient = steepest DECREASE", True),
    ("Update:  w ← w − η·∂L/∂w     (subtract = move downhill)", True),
    ("Learning rate η controls step size: "
     "too small = slow,  too large = overshoot,  just right = convergence", False),
    ("For convex losses (like linear regression), the bowl has ONE minimum — "
     "gradient descent will find it", False),
], w=12, line_h=0.55)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/gradient_descent.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
