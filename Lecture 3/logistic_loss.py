"""
Lecture 3 Visual: Logistic Loss — From Probability to Loss

Shows:
1. The logistic loss curve L = −log σ(margin) = log(1 + e^{−margin})
2. Side-by-side with σ(margin) to show the probability-to-loss pipeline
3. Interpretation: what different margins mean in terms of loss
4. Comparison with 0–1 loss to show the smooth gradient advantage
5. The gradient of logistic loss — always nonzero, always useful
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(margin):
    return np.log(1.0 + np.exp(-margin))


def logistic_loss_grad(margin):
    """Derivative of log(1 + e^{-m}) = -σ(-m) = -(1 - σ(m)) = σ(m) - 1"""
    return sigmoid(margin) - 1.0


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
fig = plt.figure(figsize=(16, 38))
fig.suptitle("Logistic Loss: Smooth, Optimizable Classification Loss",
             fontsize=17, fontweight="bold", y=0.995, color=C_BOUNDARY)

gs = GridSpec(5, 2, figure=fig,
              height_ratios=[2.8, 2.8, 2.2, 2.5, 1.2],
              hspace=0.28, wspace=0.28,
              top=0.97, bottom=0.02, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: σ(margin) — probability of correct class
# ─────────────────────────────────────────────────────────
ax_sig = fig.add_subplot(gs[0, 0])

m = np.linspace(-6, 6, 500)

ax_sig.axvspan(-6, 0, alpha=0.06, color=C_NEG, zorder=1)
ax_sig.axvspan(0, 6, alpha=0.06, color=C_POS, zorder=1)

ax_sig.plot(m, sigmoid(m), color=C_I, lw=3, zorder=4)

ax_sig.axhline(y=0.5, color=C_BOUNDARY, lw=1, linestyle=":", alpha=0.4)
ax_sig.axvline(x=0, color=C_BOUNDARY, lw=1, linestyle=":", alpha=0.4)

# Annotate a few key points
key_pts = [
    (-4, "σ ≈ 0.02", C_NEG, (-4, 0.18)),
    (-1, "σ ≈ 0.27", C_WARN, (-1, 0.42)),
    (0, "σ = 0.5", C_BOUNDARY, (1.3, 0.35)),
    (1, "σ ≈ 0.73", C_WARN, (1, 0.60)),
    (4, "σ ≈ 0.98", C_POS, (4, 0.82)),
]
for z_val, label, color, text_pos in key_pts:
    sig_val = sigmoid(z_val)
    ax_sig.plot(z_val, sig_val, 'o', color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax_sig.annotate(label, xy=(z_val, sig_val), xytext=text_pos,
                    fontsize=8.5, color=color, fontweight="bold",
                    ha="center")

ax_sig.text(-3, 0.85, "WRONG", fontsize=12, ha="center",
            color=C_NEG, fontweight="bold", alpha=0.6)
ax_sig.text(-3, 0.75, "margin < 0", fontsize=9, ha="center",
            color=C_NEG, alpha=0.5)
ax_sig.text(3, 0.85, "CORRECT", fontsize=12, ha="center",
            color=C_POS, fontweight="bold", alpha=0.6)
ax_sig.text(3, 0.75, "margin > 0", fontsize=9, ha="center",
            color=C_POS, alpha=0.5)

ax_sig.set_xlabel("margin = y · (w · x + b)", fontsize=10, color=C_I)
ax_sig.set_ylabel("σ(margin) = P(correct | x)", fontsize=10, color=C_I)
ax_sig.set_title("Step 1:  Margin → Probability of Correct Class",
                  fontsize=13, fontweight="bold", color=C_RULE, pad=10)
ax_sig.set_xlim(-6, 6)
ax_sig.set_ylim(-0.08, 1.05)
ax_sig.tick_params(colors=SUBTLE, labelsize=8)
for spine in ax_sig.spines.values():
    spine.set_color(C_DIM)


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: Logistic loss curve
# ─────────────────────────────────────────────────────────
ax_loss = fig.add_subplot(gs[0, 1])

loss_vals = logistic_loss(m)

ax_loss.axvspan(-6, 0, alpha=0.06, color=C_NEG, zorder=1)
ax_loss.axvspan(0, 6, alpha=0.06, color=C_POS, zorder=1)

ax_loss.plot(m, loss_vals, color=C_BOUNDARY, lw=3, zorder=4)

ax_loss.axvline(x=0, color=C_BOUNDARY, lw=1, linestyle=":", alpha=0.4)

# Annotate key points
loss_pts = [
    (-4, C_NEG, (-3.2, 4.8), "L ≈ 4.02\nhigh loss!"),
    (-1, C_WARN, (0.5, 3.5), "L ≈ 1.31"),
    (0, C_BOUNDARY, (1.8, 1.8), "L ≈ 0.69"),
    (1, C_WARN, (2.5, 1.3), "L ≈ 0.31"),
    (4, C_POS, (4.8, 1.0), "L ≈ 0.02\nlow loss"),
]
for z_val, color, text_pos, label in loss_pts:
    l_val = logistic_loss(z_val)
    ax_loss.plot(z_val, l_val, 'o', color=color, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax_loss.annotate(label, xy=(z_val, l_val), xytext=text_pos,
                     fontsize=8.5, color=color, fontweight="bold",
                     ha="center",
                     arrowprops=dict(arrowstyle="->", color=color, lw=1))

ax_loss.text(-3, 5.5, "WRONG", fontsize=12, ha="center",
             color=C_NEG, fontweight="bold", alpha=0.6)
ax_loss.text(3, 5.5, "CORRECT", fontsize=12, ha="center",
             color=C_POS, fontweight="bold", alpha=0.6)

ax_loss.set_xlabel("margin = y · (w · x + b)", fontsize=10, color=C_I)
ax_loss.set_ylabel("L = −log σ(margin)", fontsize=10, color=C_BOUNDARY)
ax_loss.set_title("Step 2:  Probability → Logistic Loss",
                  fontsize=13, fontweight="bold", color=C_RULE, pad=10)
ax_loss.set_xlim(-6, 6)
ax_loss.set_ylim(-0.3, 6.5)
ax_loss.tick_params(colors=SUBTLE, labelsize=8)
for spine in ax_loss.spines.values():
    spine.set_color(C_DIM)


# ─────────────────────────────────────────────────────────
#  ROW 2, LEFT: The pipeline explanation
# ─────────────────────────────────────────────────────────
ax_pipe = fig.add_subplot(gs[1, 0])
ax_pipe.axis("off")
ax_pipe.set_xlim(0, 10)
ax_pipe.set_ylim(-6.5, 6)

ax_pipe.text(5, 5.5, "The Two-Step Pipeline",
             fontsize=13, ha="center", fontweight="bold", color=C_RULE)

# Step 1
ax_pipe.add_patch(patches.FancyBboxPatch(
    (0.3, 2.8), 9.4, 2.2, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))
ax_pipe.text(5, 4.5, "Step 1:  Apply σ to the margin", fontsize=11,
             ha="center", fontweight="bold", color=C_I)
ax_pipe.text(5, 3.7,
             "P(correct | x)  =  σ(margin)  =  σ( y · (w·x + b) )",
             fontsize=10.5, ha="center", fontfamily="monospace", color=MEDIUM)
ax_pipe.text(5, 3.1,
             "This gives the probability the model assigns to the TRUE label.",
             fontsize=9.5, ha="center", color=TEXT)

# Arrow
ax_pipe.annotate("", xy=(5, 2.5), xytext=(5, 2.8),
                 arrowprops=dict(arrowstyle="->,head_width=0.15",
                                 color=SUBTLE, lw=1.5))

# Step 2
ax_pipe.add_patch(patches.FancyBboxPatch(
    (0.3, 0.0), 9.4, 2.2, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.5))
ax_pipe.text(5, 1.7, "Step 2:  Take the negative log", fontsize=11,
             ha="center", fontweight="bold", color=C_BOUNDARY)
ax_pipe.text(5, 0.9,
             "L  =  −log P(correct | x)  =  −log σ(margin)",
             fontsize=10.5, ha="center", fontfamily="monospace", color=MEDIUM)
ax_pipe.text(5, 0.3,
             "High probability → small loss.    Low probability → large loss.",
             fontsize=9.5, ha="center", color=TEXT)

# Why negative log?
ax_pipe.add_patch(patches.FancyBboxPatch(
    (0.3, -6.0), 9.4, 5.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))
ax_pipe.text(5, -1.0, "Why Negative Log?", fontsize=12,
             ha="center", fontweight="bold", color=C_B)

reasons = [
    ("Flips direction:", "high probability → low loss  (we want this)", True),
    ("Monotonic:", "loss always decreases as confidence increases", False),
    ("Smooth:", "differentiable everywhere — gradient descent works", False),
    ("Penalizes harshly:", "probability near 0 → loss explodes toward ∞", False),
    ("Products → sums:", "log turns P(a)·P(b) into log P(a) + log P(b)", False),
    ("", "so total loss over a dataset is a simple sum", False),
]

for i, (prefix, rest, is_first) in enumerate(reasons):
    y = -1.8 - i * 0.7
    if prefix:
        ax_pipe.text(1.0, y, prefix, fontsize=9.5, color=C_B,
                     fontweight="bold")
        ax_pipe.text(3.6, y, rest, fontsize=9.5, color=MEDIUM)
    else:
        ax_pipe.text(3.6, y, rest, fontsize=9.5, color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 2, RIGHT: Logistic loss vs 0-1 loss comparison
# ─────────────────────────────────────────────────────────
ax_cmp = fig.add_subplot(gs[1, 1])

m_cmp = np.linspace(-5, 5, 500)

# 0-1 loss
loss_01 = np.where(m_cmp <= 0, 1.0, 0.0)
ax_cmp.plot(m_cmp[m_cmp < 0], loss_01[m_cmp < 0],
            color=C_NEG, lw=2.5, alpha=0.6, zorder=3)
ax_cmp.plot(m_cmp[m_cmp > 0], loss_01[m_cmp > 0],
            color=C_NEG, lw=2.5, alpha=0.6, zorder=3, label="0–1 loss")
ax_cmp.plot(0, 1, 'o', color=C_NEG, markersize=7,
            markeredgecolor="white", markeredgewidth=1, alpha=0.6, zorder=4)
ax_cmp.plot(0, 0, 'o', color=C_NEG, markersize=7, markerfacecolor=BG,
            markeredgecolor=C_NEG, markeredgewidth=1.5, alpha=0.6, zorder=4)

# Logistic loss
loss_log = logistic_loss(m_cmp)
ax_cmp.plot(m_cmp, loss_log, color=C_BOUNDARY, lw=3, zorder=4,
            label="logistic loss")

ax_cmp.axvline(x=0, color=SUBTLE, lw=1, linestyle=":", alpha=0.3)

# Annotations
ax_cmp.annotate("0–1 loss: flat\n(no gradient)",
                xy=(-2.5, 1.0), xytext=(-2.5, 3.5),
                fontsize=9, color=C_NEG, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1))

ax_cmp.annotate("logistic loss: smooth curve\n(always has gradient)",
                xy=(-2.5, logistic_loss(-2.5)),
                xytext=(1.0, 5.0),
                fontsize=9, color=C_BOUNDARY, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_BOUNDARY, lw=1))

ax_cmp.annotate("logistic loss still\npushes correct points\nto be more confident",
                xy=(2.5, logistic_loss(2.5)),
                xytext=(3.5, 2.0),
                fontsize=8.5, color=C_POS, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_POS, lw=1))

ax_cmp.set_xlabel("margin", fontsize=10, color=C_I)
ax_cmp.set_ylabel("loss", fontsize=10, color=C_BOUNDARY)
ax_cmp.set_title("Logistic Loss vs 0–1 Loss",
                  fontsize=13, fontweight="bold", color=C_RULE, pad=10)
ax_cmp.set_xlim(-5, 5)
ax_cmp.set_ylim(-0.3, 6)
ax_cmp.legend(fontsize=9, facecolor=BG, edgecolor=C_DIM,
              labelcolor=TEXT, loc="upper right")
ax_cmp.tick_params(colors=SUBTLE, labelsize=8)
for spine in ax_cmp.spines.values():
    spine.set_color(C_DIM)


# ─────────────────────────────────────────────────────────
#  ROW 3: The gradient — always nonzero
# ─────────────────────────────────────────────────────────
ax_grad = fig.add_subplot(gs[2, :])

m_grad = np.linspace(-6, 6, 500)
grad_vals = logistic_loss_grad(m_grad)

ax_grad.axvspan(-6, 0, alpha=0.06, color=C_NEG, zorder=1)
ax_grad.axvspan(0, 6, alpha=0.06, color=C_POS, zorder=1)

ax_grad.plot(m_grad, grad_vals, color=C_B, lw=3, zorder=4)

ax_grad.axhline(y=0, color=SUBTLE, lw=1, linestyle=":", alpha=0.4)
ax_grad.axvline(x=0, color=C_BOUNDARY, lw=1, linestyle=":", alpha=0.4)

# Annotate regions
ax_grad.annotate("Large negative gradient\n→ strong push to increase margin",
                 xy=(-4, logistic_loss_grad(-4)),
                 xytext=(-2.5, -0.3),
                 fontsize=9.5, color=C_NEG, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.2))

ax_grad.annotate("Gradient at margin = 0\n= −0.5  (moderate push)",
                 xy=(0, -0.5),
                 xytext=(2.5, -0.7),
                 fontsize=9.5, color=C_BOUNDARY, fontweight="bold",
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_BOUNDARY, lw=1.2))

ax_grad.annotate("Small gradient\n→ gentle nudge\n(already correct)",
                 xy=(4, logistic_loss_grad(4)),
                 xytext=(4, -0.35),
                 fontsize=9.5, color=C_POS, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_POS, lw=1.2))

# Key contrast with 0-1 loss
ax_grad.text(0, 0.18,
             "Unlike 0–1 loss:  the gradient is NEVER zero  "
             "→  gradient descent always has a direction to move",
             fontsize=10, ha="center", color=C_BOUNDARY, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                       edgecolor=C_BOUNDARY, alpha=0.85))

ax_grad.set_xlabel("margin = y · (w · x + b)", fontsize=10, color=C_I)
ax_grad.set_ylabel("dL/d(margin)", fontsize=10, color=C_B)
ax_grad.set_title("Gradient of Logistic Loss — Always Nonzero, Always Useful",
                  fontsize=13, fontweight="bold", color=C_RULE, pad=10)
ax_grad.set_xlim(-6, 6)
ax_grad.set_ylim(-1.1, 0.3)
ax_grad.tick_params(colors=SUBTLE, labelsize=9)
for spine in ax_grad.spines.values():
    spine.set_color(C_DIM)


# ─────────────────────────────────────────────────────────
#  ROW 4: Interpretation table
# ─────────────────────────────────────────────────────────
ax_table = fig.add_subplot(gs[3, :])
ax_table.axis("off")
ax_table.set_xlim(0, 16)
ax_table.set_ylim(-6, 5)

ax_table.text(8, 4.5,
              "Reading the Loss: Margin → Probability → Loss → Gradient",
              fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# Table header
hy = 3.3
cols = [0.5, 2.5, 4.8, 7.3, 10.0, 12.5]
headers = ["Margin", "σ(margin)", "P(correct)", "Loss",
           "Gradient", "Meaning"]
h_colors = [C_I, C_I, C_POS, C_BOUNDARY, C_B, TEXT]
for x, h, c in zip(cols, headers, h_colors):
    ax_table.text(x, hy, h, fontsize=10, fontweight="bold", color=c)

ax_table.plot([0.3, 15.7], [hy - 0.3, hy - 0.3], "-",
              color=SUBTLE, alpha=0.4, lw=1)

rows = [
    (-4, C_NEG, "Confidently wrong — huge loss, strong correction"),
    (-2, C_NEG, "Wrong — high loss, solid gradient signal"),
    (-0.5, C_WARN, "Slightly wrong — moderate loss and gradient"),
    (0, C_BOUNDARY, "On the boundary — coin-flip confidence"),
    (0.5, C_WARN, "Slightly correct — some loss remains"),
    (2, C_POS, "Correct — small loss, gentle nudge to improve"),
    (4, C_POS, "Confidently correct — tiny loss, near-zero gradient"),
]

for idx, (margin, color, meaning) in enumerate(rows):
    y = hy - 0.85 - idx * 0.75
    sig_val = sigmoid(margin)
    loss_val = logistic_loss(margin)
    grad_val = logistic_loss_grad(margin)

    sign_m = "+" if margin > 0 else ""

    ax_table.text(cols[0], y, f"{sign_m}{margin:.1f}",
                  fontsize=10, fontfamily="monospace", color=C_I,
                  fontweight="bold")
    ax_table.text(cols[1], y, f"{sig_val:.3f}",
                  fontsize=10, fontfamily="monospace", color=C_I)
    ax_table.text(cols[2], y, f"{sig_val*100:.1f}%",
                  fontsize=10, fontfamily="monospace", color=color,
                  fontweight="bold")
    ax_table.text(cols[3], y, f"{loss_val:.2f}",
                  fontsize=10, fontfamily="monospace", color=C_BOUNDARY,
                  fontweight="bold")
    ax_table.text(cols[4], y, f"{grad_val:.3f}",
                  fontsize=10, fontfamily="monospace", color=C_B)
    ax_table.text(cols[5], y, meaning,
                  fontsize=8.5, color=color)

    if idx < len(rows) - 1:
        ax_table.plot([0.3, 15.7], [y - 0.3, y - 0.3], "-",
                      color=C_DIM, alpha=0.15, lw=0.5)


# ─────────────────────────────────────────────────────────
#  ROW 5: Summary
# ─────────────────────────────────────────────────────────
ax_sum = fig.add_subplot(gs[4, :])
ax_sum.axis("off")
ax_sum.set_xlim(0, 16)
ax_sum.set_ylim(-3, 3)

ax_sum.text(8, 2.5, "Summary", fontsize=14, ha="center",
            fontweight="bold", color=C_RULE)

insight_box(ax_sum, 2.0, -2.5, [
    ("Logistic loss  L = −log σ(margin)  penalizes low confidence "
     "on the correct class.", True),
    ("It is smooth everywhere — gradient descent always gets a useful signal.", True),
    ("Wrong predictions get large loss and strong gradient.  "
     "Correct predictions get small loss and gentle gradient.", False),
    ("This is the probabilistic surrogate that replaces 0–1 loss "
     "while preserving the classification goal.", False),
], w_box=12, line_h=0.55)


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_3_visuals/logistic_loss.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
