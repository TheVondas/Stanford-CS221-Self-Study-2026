"""
Lecture 3 Visual: The Logistic (Sigmoid) Function

Shows:
1. The S-curve σ(z) = 1 / (1 + e^{-z}) mapping any real z to (0, 1)
2. Key reference points annotated on the curve
3. Interpretation: what different regions of the curve mean in terms of confidence
4. A worked-examples panel showing concrete logit → probability mappings
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
fig = plt.figure(figsize=(16, 34))
fig.suptitle("The Logistic Function: Turning Scores into Probabilities",
             fontsize=17, fontweight="bold", y=0.995, color=C_BOUNDARY)

gs = GridSpec(4, 2, figure=fig,
              height_ratios=[3.5, 2.0, 2.8, 1.2],
              hspace=0.25, wspace=0.25,
              top=0.97, bottom=0.02, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 1: The S-curve (spanning both columns)
# ─────────────────────────────────────────────────────────
ax_curve = fig.add_subplot(gs[0, :])

z = np.linspace(-8, 8, 500)
sig = sigmoid(z)

# Shade confidence regions
# Very negative: confident −1
ax_curve.axvspan(-8, -3, alpha=0.07, color=C_NEG, zorder=1)
# Mildly negative: leaning −1
ax_curve.axvspan(-3, -0.5, alpha=0.04, color=C_NEG, zorder=1)
# Uncertain zone
ax_curve.axvspan(-0.5, 0.5, alpha=0.07, color=C_BOUNDARY, zorder=1)
# Mildly positive: leaning +1
ax_curve.axvspan(0.5, 3, alpha=0.04, color=C_POS, zorder=1)
# Very positive: confident +1
ax_curve.axvspan(3, 8, alpha=0.07, color=C_POS, zorder=1)

# The curve itself
ax_curve.plot(z, sig, color=C_I, lw=3.5, zorder=4)

# Asymptote lines
ax_curve.axhline(y=1.0, color=C_POS, lw=1.2, linestyle="--", alpha=0.5,
                 zorder=2)
ax_curve.axhline(y=0.0, color=C_NEG, lw=1.2, linestyle="--", alpha=0.5,
                 zorder=2)
ax_curve.axhline(y=0.5, color=C_BOUNDARY, lw=1, linestyle=":", alpha=0.4,
                 zorder=2)
ax_curve.axvline(x=0, color=C_BOUNDARY, lw=1, linestyle=":", alpha=0.4,
                 zorder=2)

# Asymptote labels
ax_curve.text(7.5, 1.04, "σ → 1", fontsize=10, color=C_POS,
              fontweight="bold", ha="center")
ax_curve.text(7.5, -0.06, "σ → 0", fontsize=10, color=C_NEG,
              fontweight="bold", ha="center")

# ── Key reference points ──
key_points = [
    (-6, "σ(−6) ≈ 0.002", C_NEG, (-5.2, 0.18),
     "Almost certain\nit's −1"),
    (-3, "σ(−3) ≈ 0.05", C_NEG, (-4.2, 0.35),
     "Quite confident\nit's −1"),
    (-1, "σ(−1) ≈ 0.27", C_WARN, (-2.5, 0.55),
     "Leaning −1"),
    (0, "σ(0) = 0.5", C_BOUNDARY, (1.4, 0.30),
     "Perfectly\nundecided"),
    (1, "σ(1) ≈ 0.73", C_WARN, (2.5, 0.52),
     "Leaning +1"),
    (3, "σ(3) ≈ 0.95", C_POS, (4.5, 0.70),
     "Quite confident\nit's +1"),
    (6, "σ(6) ≈ 0.998", C_POS, (5.0, 0.85),
     "Almost certain\nit's +1"),
]

for z_val, label, color, text_pos, interp in key_points:
    sig_val = sigmoid(z_val)
    # Dot on the curve
    ax_curve.plot(z_val, sig_val, 'o', color=color, markersize=9,
                  markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    # Label with arrow
    ax_curve.annotate(
        f"{label}\n{interp}",
        xy=(z_val, sig_val),
        xytext=text_pos,
        fontsize=8.5, color=color, fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.25", facecolor=BG,
                  edgecolor=color, alpha=0.85),
        zorder=6)

# Region labels at the top
ax_curve.text(-5.5, 1.12, "Confident −1", fontsize=11,
              ha="center", color=C_NEG, fontweight="bold", alpha=0.8)
ax_curve.text(-1.75, 1.12, "Leaning −1", fontsize=10,
              ha="center", color=C_NEG, alpha=0.6)
ax_curve.text(0, 1.12, "?", fontsize=12,
              ha="center", color=C_BOUNDARY, fontweight="bold", alpha=0.8)
ax_curve.text(1.75, 1.12, "Leaning +1", fontsize=10,
              ha="center", color=C_POS, alpha=0.6)
ax_curve.text(5.5, 1.12, "Confident +1", fontsize=11,
              ha="center", color=C_POS, fontweight="bold", alpha=0.8)

# Axis labels
ax_curve.set_xlabel("z  (logit)  =  w · x + b          ← any real number from −∞ to +∞ →",
                    fontsize=12, color=C_I)
ax_curve.set_ylabel("σ(z)  =  P(y = +1 | x)          probability ∈ (0, 1)",
                    fontsize=12, color=C_BOUNDARY)
ax_curve.set_title(
    "σ(z)  =  1 / (1 + e$^{-z}$)",
    fontsize=15, fontweight="bold", color=C_RULE, pad=14)
ax_curve.set_xlim(-8, 8)
ax_curve.set_ylim(-0.12, 1.22)
ax_curve.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_curve.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax_curve.tick_params(colors=SUBTLE, labelsize=9)
for spine in ax_curve.spines.values():
    spine.set_color(C_DIM)


# ─────────────────────────────────────────────────────────
#  ROW 2: What the logistic function DOES — the big idea
# ─────────────────────────────────────────────────────────
ax_idea = fig.add_subplot(gs[1, :])
ax_idea.axis("off")
ax_idea.set_xlim(0, 16)
ax_idea.set_ylim(-3, 5)

ax_idea.text(8, 4.5, "What the Logistic Function Does",
             fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# Input box
ax_idea.add_patch(patches.FancyBboxPatch(
    (0.5, 1.0), 4.5, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))
ax_idea.text(2.75, 3.5, "Input: the logit z", fontsize=12,
             ha="center", fontweight="bold", color=C_I)
ax_idea.text(2.75, 2.7, "z = w · x + b", fontsize=12,
             ha="center", fontfamily="monospace", color=MEDIUM)
ax_idea.text(2.75, 2.0, "Any real number", fontsize=10,
             ha="center", color=MEDIUM)
ax_idea.text(2.75, 1.4, "(−∞,  +∞)", fontsize=11,
             ha="center", fontfamily="monospace", color=C_I)

# Arrow
ax_idea.annotate("", xy=(6.0, 2.5), xytext=(5.2, 2.5),
                 arrowprops=dict(arrowstyle="->,head_width=0.2",
                                 color=C_BOUNDARY, lw=2.5))

# Transform box
ax_idea.add_patch(patches.FancyBboxPatch(
    (6.0, 1.0), 4.0, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.5))
ax_idea.text(8, 3.5, "The logistic function", fontsize=12,
             ha="center", fontweight="bold", color=C_BOUNDARY)
ax_idea.text(8, 2.6, "σ(z) = 1 / (1 + e⁻ᶻ)", fontsize=12,
             ha="center", fontfamily="monospace", color=C_BOUNDARY)
ax_idea.text(8, 1.7, "squashes into (0, 1)", fontsize=10,
             ha="center", color=MEDIUM)

# Arrow
ax_idea.annotate("", xy=(11.0, 2.5), xytext=(10.2, 2.5),
                 arrowprops=dict(arrowstyle="->,head_width=0.2",
                                 color=C_BOUNDARY, lw=2.5))

# Output box
ax_idea.add_patch(patches.FancyBboxPatch(
    (11.0, 1.0), 4.5, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.5))
ax_idea.text(13.25, 3.5, "Output: a probability", fontsize=12,
             ha="center", fontweight="bold", color=C_POS)
ax_idea.text(13.25, 2.7, "P(y = +1 | x)", fontsize=12,
             ha="center", fontfamily="monospace", color=MEDIUM)
ax_idea.text(13.25, 2.0, "A valid confidence", fontsize=10,
             ha="center", color=MEDIUM)
ax_idea.text(13.25, 1.4, "(0,  1)", fontsize=11,
             ha="center", fontfamily="monospace", color=C_POS)

# Why this matters
ax_idea.text(8, 0.0,
             "This is the key move: turn an unbounded score into a bounded "
             "probability, so losses become smooth and optimizable.",
             fontsize=10.5, ha="center", color=C_B, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 3: Concrete interpretation table
# ─────────────────────────────────────────────────────────
ax_table = fig.add_subplot(gs[2, :])
ax_table.axis("off")
ax_table.set_xlim(0, 16)
ax_table.set_ylim(-7, 5)

ax_table.text(8, 4.5, "Reading the Curve: Logit → Probability → Meaning",
              fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# Table header
hy = 3.3
cols = [1.0, 3.5, 6.0, 9.0]
headers = ["Logit z", "σ(z)", "P(+1)", "Interpretation"]
h_colors = [C_I, C_BOUNDARY, C_POS, TEXT]
for x, h, c in zip(cols, headers, h_colors):
    ax_table.text(x, hy, h, fontsize=11, fontweight="bold", color=c)

ax_table.plot([0.7, 15.3], [hy - 0.3, hy - 0.3], "-",
              color=SUBTLE, alpha=0.4, lw=1)

# Table rows
rows = [
    ("−6", "0.002", "0.2%", "Almost certain it's class −1",
     C_NEG, "██████████"),
    ("−3", "0.047", "4.7%", "Very likely class −1",
     C_NEG, "████████░░"),
    ("−1", "0.269", "26.9%", "Leaning toward −1, but not confident",
     C_WARN, "██████░░░░"),
    ("0", "0.500", "50.0%", "Perfectly undecided — coin flip",
     C_BOUNDARY, "█████░░░░░"),
    ("+1", "0.731", "73.1%", "Leaning toward +1, but not confident",
     C_WARN, "░░░░██████"),
    ("+3", "0.953", "95.3%", "Very likely class +1",
     C_POS, "░░████████"),
    ("+6", "0.998", "99.8%", "Almost certain it's class +1",
     C_POS, "░░██████████"),
]

for idx, (z_str, sig_str, prob_str, interp, color, bar) in enumerate(rows):
    y = hy - 0.85 - idx * 0.85

    ax_table.text(cols[0], y, z_str, fontsize=10.5, fontfamily="monospace",
                  color=C_I, fontweight="bold")
    ax_table.text(cols[1], y, sig_str, fontsize=10.5, fontfamily="monospace",
                  color=C_BOUNDARY)
    ax_table.text(cols[2], y, prob_str, fontsize=10.5, fontfamily="monospace",
                  color=color, fontweight="bold")
    ax_table.text(cols[3], y, interp, fontsize=10, color=color)

    # Confidence bar
    ax_table.text(13.5, y, bar, fontsize=9, fontfamily="monospace",
                  color=color, alpha=0.6)

    if idx < len(rows) - 1:
        ax_table.plot([0.7, 15.3], [y - 0.35, y - 0.35], "-",
                      color=C_DIM, alpha=0.15, lw=0.5)

# Key properties
ax_table.add_patch(patches.FancyBboxPatch(
    (1.0, -6.5), 14, 2.8, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_table.text(8, -4.2, "Key Properties of σ(z)",
              fontsize=12, ha="center", fontweight="bold", color=C_B)

props = [
    "σ(0) = 0.5   always.   Zero logit means total uncertainty.",
    "σ(z) + σ(−z) = 1   always.   Flipping the sign flips the probability.",
    "The curve never actually reaches 0 or 1 — it only approaches them asymptotically.",
]
for i, prop in enumerate(props):
    ax_table.text(8, -5.0 - i * 0.6, prop,
                  fontsize=10, ha="center", color=MEDIUM)


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
    ("The logistic function maps any real-valued logit z to a probability in (0, 1).", True),
    ("Sign of z → which class is more likely.    Magnitude of z → how confident.", True),
    ("This turns a raw score into something we can interpret, compare, and — "
     "crucially — optimize with gradient descent.", False),
    ("It is the bridge between the linear score (w · x + b) and a probabilistic "
     "loss function (logistic loss).", False),
], w_box=12, line_h=0.55)


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_3_visuals/logistic_function.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
