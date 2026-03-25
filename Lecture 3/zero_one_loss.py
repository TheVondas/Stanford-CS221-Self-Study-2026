"""
Lecture 3 Visual: 0–1 Loss and Why It Breaks Gradient Descent

Shows:
1. The 0–1 loss as a function of margin — a flat step function
2. The gradient of 0–1 loss — zero everywhere except an unusable spike at 0
3. A worked example: nudging parameters does nothing to the loss
4. Summary of why gradient descent gets stuck
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec

# ── Style ────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_POS = "#a6e3a1"        # green
C_NEG = "#f38ba8"         # pink
C_BOUNDARY = "#f9e2af"    # yellow
C_RULE = "#fab387"        # orange
C_I = "#89b4fa"           # blue
C_B = "#cba6f7"           # purple
C_DIM = "#585b70"
C_WARN = "#f5c2e7"        # light pink

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


def insight_box(ax, x, y, lines, w_box=5.5, line_h=0.45):
    h = len(lines) * line_h + 0.25
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w_box, h, boxstyle="round,pad=0.12",
        facecolor=C_RULE, alpha=0.10, edgecolor=C_RULE, linewidth=1.5))
    for i, (txt, bold) in enumerate(lines):
        yy = y + h - 0.28 - i * line_h
        ax.text(x + w_box / 2, yy, txt, ha="center", fontsize=9.5,
                color=C_RULE, fontweight="bold" if bold else "normal")


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 32))
fig.suptitle("0–1 Loss: Why It Breaks Gradient Descent",
             fontsize=17, fontweight="bold", y=0.995, color=C_BOUNDARY)

gs = GridSpec(4, 2, figure=fig,
              height_ratios=[2.5, 2.5, 2.8, 1.2],
              hspace=0.28, wspace=0.28,
              top=0.97, bottom=0.02, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: 0–1 Loss as a function of margin
# ─────────────────────────────────────────────────────────
ax_loss = fig.add_subplot(gs[0, 0])

margin = np.linspace(-4, 4, 1000)

# 0-1 loss: 1 if margin <= 0, else 0
loss_01 = np.where(margin <= 0, 1.0, 0.0)

# Draw the two flat segments
# Wrong side (margin < 0): loss = 1
ax_loss.plot(margin[margin < 0], loss_01[margin < 0],
             color=C_NEG, lw=3.5, solid_capstyle="butt", zorder=4)
# Correct side (margin > 0): loss = 0
ax_loss.plot(margin[margin > 0], loss_01[margin > 0],
             color=C_POS, lw=3.5, solid_capstyle="butt", zorder=4)

# The discontinuity at margin = 0
ax_loss.plot(0, 1, 'o', color=C_NEG, markersize=10,
             markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax_loss.plot(0, 0, 'o', color=C_POS, markersize=10,
             markerfacecolor=BG, markeredgecolor=C_POS,
             markeredgewidth=2, zorder=5)  # open circle

# Vertical dashed line at margin = 0
ax_loss.axvline(x=0, color=C_BOUNDARY, lw=1.5, linestyle=":",
                alpha=0.5, zorder=2)

# Shade regions
ax_loss.axvspan(-4, 0, alpha=0.06, color=C_NEG, zorder=1)
ax_loss.axvspan(0, 4, alpha=0.06, color=C_POS, zorder=1)

# Region labels
ax_loss.text(-2, 0.5, "WRONG\nmargin < 0", fontsize=11,
             ha="center", color=C_NEG, fontweight="bold", alpha=0.7)
ax_loss.text(2, 0.5, "CORRECT\nmargin > 0", fontsize=11,
             ha="center", color=C_POS, fontweight="bold", alpha=0.7)

# Annotate the flat segments
ax_loss.annotate("loss = 1\n(flat)", xy=(-2.5, 1.0),
                 xytext=(-2.5, 1.35),
                 fontsize=10, color=C_NEG, fontweight="bold",
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.2))

ax_loss.annotate("loss = 0\n(flat)", xy=(2.5, 0.0),
                 xytext=(2.5, -0.4),
                 fontsize=10, color=C_POS, fontweight="bold",
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_POS, lw=1.2))

# Annotate the discontinuity
ax_loss.annotate("discontinuous\njump",
                 xy=(0, 0.5), xytext=(1.5, 1.35),
                 fontsize=9, color=C_BOUNDARY, fontweight="bold",
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_BOUNDARY, lw=1.2))

ax_loss.set_xlabel("margin  =  y · (w · x + b)", fontsize=11, color=C_I)
ax_loss.set_ylabel("0–1 Loss", fontsize=11, color=C_BOUNDARY)
ax_loss.set_title("0–1 Loss vs Margin",
                  fontsize=14, fontweight="bold", color=C_RULE, pad=12)
ax_loss.set_xlim(-4, 4)
ax_loss.set_ylim(-0.6, 1.7)
ax_loss.set_yticks([0, 1])
ax_loss.set_yticklabels(["0", "1"])
ax_loss.tick_params(colors=SUBTLE, labelsize=9)
for spine in ax_loss.spines.values():
    spine.set_color(C_DIM)


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: The gradient of 0–1 loss
# ─────────────────────────────────────────────────────────
ax_grad = fig.add_subplot(gs[0, 1])

# Gradient is 0 everywhere (the function is flat on both sides)
margin_left = np.linspace(-4, -0.05, 200)
margin_right = np.linspace(0.05, 4, 200)

ax_grad.plot(margin_left, np.zeros_like(margin_left),
             color=C_NEG, lw=3.5, solid_capstyle="butt", zorder=4)
ax_grad.plot(margin_right, np.zeros_like(margin_right),
             color=C_POS, lw=3.5, solid_capstyle="butt", zorder=4)

# Undefined spike at 0 (the derivative doesn't exist)
ax_grad.annotate("", xy=(0, 1.3), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->,head_width=0.15",
                                 color=C_BOUNDARY, lw=2.5))
ax_grad.text(0, 1.5, "undefined\n(−∞ spike)", fontsize=9,
             ha="center", color=C_BOUNDARY, fontweight="bold")

# Vertical dashed line at margin = 0
ax_grad.axvline(x=0, color=C_BOUNDARY, lw=1.5, linestyle=":",
                alpha=0.3, zorder=2)

# Shade regions
ax_grad.axvspan(-4, 0, alpha=0.06, color=C_NEG, zorder=1)
ax_grad.axvspan(0, 4, alpha=0.06, color=C_POS, zorder=1)

# Big "ZERO" labels
ax_grad.text(-2, 0.35, "gradient = 0", fontsize=13,
             ha="center", color=C_NEG, fontweight="bold")
ax_grad.text(-2, -0.25, "no signal", fontsize=10,
             ha="center", color=C_NEG, alpha=0.7)
ax_grad.text(2, 0.35, "gradient = 0", fontsize=13,
             ha="center", color=C_POS, fontweight="bold")
ax_grad.text(2, -0.25, "no signal", fontsize=10,
             ha="center", color=C_POS, alpha=0.7)

ax_grad.set_xlabel("margin  =  y · (w · x + b)", fontsize=11, color=C_I)
ax_grad.set_ylabel("Gradient of 0–1 Loss", fontsize=11, color=C_BOUNDARY)
ax_grad.set_title("Gradient of 0–1 Loss — Zero Almost Everywhere",
                  fontsize=14, fontweight="bold", color=C_RULE, pad=12)
ax_grad.set_xlim(-4, 4)
ax_grad.set_ylim(-0.6, 1.9)
ax_grad.set_yticks([0])
ax_grad.set_yticklabels(["0"])
ax_grad.tick_params(colors=SUBTLE, labelsize=9)
for spine in ax_grad.spines.values():
    spine.set_color(C_DIM)


# ─────────────────────────────────────────────────────────
#  ROW 2: Worked example — nudging parameters changes nothing
# ─────────────────────────────────────────────────────────
ax_ex = fig.add_subplot(gs[1, :])
ax_ex.axis("off")
ax_ex.set_xlim(0, 16)
ax_ex.set_ylim(-6.5, 5)

ax_ex.text(8, 4.5, "Worked Example: Why Gradient Descent Gets Stuck",
           fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# Setup
ax_ex.text(8, 3.5,
           "Suppose we have a misclassified point with margin = −2.0    (wrong side, loss = 1)",
           fontsize=11, ha="center", color=MEDIUM)

# The three nudges
scenarios = [
    ("Nudge w slightly", "margin: −2.0 → −1.9", "loss: 1 → 1",
     "still on the wrong side", "gradient = 0", C_NEG),
    ("Nudge w more", "margin: −2.0 → −0.5", "loss: 1 → 1",
     "still on the wrong side", "gradient = 0", C_NEG),
    ("Nudge w a LOT", "margin: −2.0 → +0.1", "loss: 1 → 0",
     "finally crosses boundary!", "but gradient was 0 the whole way", C_POS),
]

for idx, (title, margin_txt, loss_txt, note, grad_note, color) in enumerate(scenarios):
    x_start = 0.5 + idx * 5.2
    box_w = 4.8

    ax_ex.add_patch(patches.FancyBboxPatch(
        (x_start, -0.5), box_w, 3.5, boxstyle="round,pad=0.15",
        facecolor=color, alpha=0.06, edgecolor=color, linewidth=1.5))

    cx = x_start + box_w / 2
    ax_ex.text(cx, 2.6, title, fontsize=11, ha="center",
               fontweight="bold", color=color)
    ax_ex.text(cx, 1.9, margin_txt, fontsize=10, ha="center",
               fontfamily="monospace", color=MEDIUM)
    ax_ex.text(cx, 1.3, loss_txt, fontsize=10, ha="center",
               fontfamily="monospace", color=color, fontweight="bold")
    ax_ex.text(cx, 0.6, note, fontsize=9.5, ha="center", color=TEXT)
    ax_ex.text(cx, -0.05, grad_note, fontsize=9.5, ha="center",
               fontweight="bold", color=color)

# Arrow between scenario 1 and 2
ax_ex.annotate("", xy=(5.5, 1.3), xytext=(5.15, 1.3),
               arrowprops=dict(arrowstyle="->", color=SUBTLE, lw=1.5))
# Arrow between scenario 2 and 3
ax_ex.annotate("", xy=(10.7, 1.3), xytext=(10.35, 1.3),
               arrowprops=dict(arrowstyle="->", color=SUBTLE, lw=1.5))

# The core problem statement
ax_ex.add_patch(patches.FancyBboxPatch(
    (1.0, -5.8), 14, 4.5, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.5))

ax_ex.text(8, -1.8, "The Core Problem",
           fontsize=13, ha="center", fontweight="bold", color=C_BOUNDARY)

ax_ex.text(8, -2.6,
           "Gradient descent asks:  \"If I nudge the parameters slightly, does the loss improve?\"",
           fontsize=11, ha="center", color=TEXT)

ax_ex.text(8, -3.4,
           "For 0–1 loss, the answer is almost always:  NO.",
           fontsize=12, ha="center", color=C_NEG, fontweight="bold")

ax_ex.text(8, -4.2,
           "Small changes to w keep you on the same flat plateau.  The loss stays at 1 or stays at 0.",
           fontsize=10.5, ha="center", color=MEDIUM)

ax_ex.text(8, -5.0,
           "The loss only changes when a point actually CROSSES the boundary — but gradients can't tell you how to get there.",
           fontsize=10.5, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 3: Plateau analogy + comparison visual
# ─────────────────────────────────────────────────────────
# Left: the plateau analogy
ax_plateau = fig.add_subplot(gs[2, 0])

# Draw a stylized plateau landscape
x_land = np.array([-4, -1.5, -1.5, -0.02, -0.02, 0.02, 0.02, 1.5, 1.5, 4])
y_land = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

ax_plateau.fill_between(x_land, -0.3, y_land, alpha=0.15, color=SUBTLE,
                        step="mid", zorder=2)
ax_plateau.plot(x_land, y_land, color=C_BOUNDARY, lw=3, zorder=3)

# Hiker on the left plateau (stuck)
ax_plateau.plot(-2.5, 1.08, "^", color=C_NEG, markersize=18,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax_plateau.text(-2.5, 1.35, "You are here\n(stuck!)", fontsize=10,
                ha="center", color=C_NEG, fontweight="bold")

# Arrows showing "which way?" — all flat
for dx in [-0.8, 0.8]:
    ax_plateau.annotate("",
                        xy=(-2.5 + dx, 1.0),
                        xytext=(-2.5, 1.0),
                        arrowprops=dict(arrowstyle="->,head_width=0.12",
                                        color=C_WARN, lw=1.5, linestyle="--"))

ax_plateau.text(-2.5, 0.7, "slope = 0\nboth directions", fontsize=9,
                ha="center", color=C_WARN, fontstyle="italic")

# Label the cliff
ax_plateau.annotate("vertical drop\n(no slope here either!)",
                    xy=(0, 0.5), xytext=(2.2, 1.35),
                    fontsize=9, color=C_BOUNDARY, fontweight="bold",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=C_BOUNDARY, lw=1.2))

# Goal on the right plateau
ax_plateau.plot(2.5, 0.08, "*", color=C_POS, markersize=18,
                markeredgecolor="white", markeredgewidth=1, zorder=5)
ax_plateau.text(2.5, -0.25, "Goal\n(loss = 0)", fontsize=9,
                ha="center", color=C_POS, fontweight="bold")

# Labels
ax_plateau.text(-3, 1.55, "loss = 1", fontsize=11,
                color=C_NEG, fontweight="bold")
ax_plateau.text(2.5, 0.55, "loss = 0", fontsize=11,
                color=C_POS, fontweight="bold")

ax_plateau.set_xlim(-4, 4)
ax_plateau.set_ylim(-0.4, 1.8)
ax_plateau.set_xlabel("margin", fontsize=10, color=C_I)
ax_plateau.set_ylabel("loss", fontsize=10, color=C_BOUNDARY)
ax_plateau.set_title("The Plateau Problem",
                     fontsize=13, fontweight="bold", color=C_RULE, pad=10)
ax_plateau.set_yticks([0, 1])
ax_plateau.tick_params(colors=SUBTLE, labelsize=9)
for spine in ax_plateau.spines.values():
    spine.set_color(C_DIM)


# Right: what gradient descent needs vs what it gets
ax_need = fig.add_subplot(gs[2, 1])
ax_need.axis("off")
ax_need.set_xlim(0, 10)
ax_need.set_ylim(-5, 8)

ax_need.text(5, 7.5, "What Gradient Descent Needs",
             fontsize=13, ha="center", fontweight="bold", color=C_RULE)

# What it needs
ax_need.add_patch(patches.FancyBboxPatch(
    (0.5, 4.0), 9, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.5))

ax_need.text(5, 6.5, "A smooth loss function that says:",
             fontsize=11, ha="center", color=C_POS, fontweight="bold")
ax_need.text(5, 5.7,
             "\"You're wrong, and here's which direction to move,\"",
             fontsize=10.5, ha="center", color=TEXT, fontstyle="italic")
ax_need.text(5, 5.0,
             "\"and here's roughly how much to move.\"",
             fontsize=10.5, ha="center", color=TEXT, fontstyle="italic")
ax_need.text(5, 4.3,
             "That requires a nonzero gradient.",
             fontsize=10, ha="center", color=C_POS)

# What it gets from 0-1 loss
ax_need.add_patch(patches.FancyBboxPatch(
    (0.5, 0.0), 9, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1.5))

ax_need.text(5, 3.0, "What 0–1 loss actually gives:",
             fontsize=11, ha="center", color=C_NEG, fontweight="bold")
ax_need.text(5, 2.2,
             "\"You're wrong.\"",
             fontsize=11, ha="center", color=C_NEG, fontstyle="italic",
             fontweight="bold")
ax_need.text(5, 1.5,
             "No direction.  No magnitude.  No signal.",
             fontsize=10.5, ha="center", color=TEXT)
ax_need.text(5, 0.7,
             "Just a flat  gradient = 0  in every direction.",
             fontsize=10.5, ha="center", color=C_NEG)
ax_need.text(5, 0.2,
             "Gradient descent sits still.",
             fontsize=10, ha="center", color=C_NEG, fontweight="bold")

# The preview of the solution
ax_need.add_patch(patches.FancyBboxPatch(
    (0.5, -4.5), 9, 3.8, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_need.text(5, -1.2, "The Fix (next visual)",
             fontsize=11, ha="center", fontweight="bold", color=C_B)
ax_need.text(5, -2.0,
             "Replace 0–1 loss with logistic loss,",
             fontsize=10.5, ha="center", color=MEDIUM)
ax_need.text(5, -2.7,
             "which is smooth and always has a gradient.",
             fontsize=10.5, ha="center", color=MEDIUM)
ax_need.text(5, -3.4,
             "Wrong predictions get a strong push.",
             fontsize=10.5, ha="center", color=C_B)
ax_need.text(5, -4.1,
             "Barely-correct predictions still get a nudge.",
             fontsize=10.5, ha="center", color=C_B)


# ─────────────────────────────────────────────────────────
#  ROW 4: Summary
# ─────────────────────────────────────────────────────────
ax_sum = fig.add_subplot(gs[3, :])
ax_sum.axis("off")
ax_sum.set_xlim(0, 16)
ax_sum.set_ylim(-3, 3)

ax_sum.text(8, 2.5, "Summary", fontsize=14, ha="center",
            fontweight="bold", color=C_RULE)

insight_box(ax_sum, 2.0, -2.5, [
    ("0–1 loss is the natural classification loss:  wrong → 1,  correct → 0", True),
    ("But it is flat on both sides — gradient = 0 almost everywhere.", True),
    ("Gradient descent needs local slope to know which direction to move. "
     "0–1 loss gives none.", False),
    ("Solution: replace it with a smooth surrogate (logistic loss) that "
     "preserves the classification goal but provides gradient signal.", False),
], w_box=12, line_h=0.55)


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_3_visuals/zero_one_loss.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
