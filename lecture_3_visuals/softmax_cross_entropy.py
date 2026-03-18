"""
Lecture 3 Visual: Softmax & Cross-Entropy Loss — The Multiclass Generalization

Shows:
1. Binary vs multiclass: logistic function vs softmax side-by-side
2. Concrete softmax example with numbers
3. Binary vs multiclass: logistic loss vs cross-entropy loss side-by-side
4. Concrete cross-entropy loss calculation
5. The unified parallel: both pipelines do the same thing at different scales
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

C_CAT = "#f38ba8"
C_DOG = "#89b4fa"
C_BIRD = "#a6e3a1"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


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
fig = plt.figure(figsize=(16, 44))
fig.suptitle("Softmax & Cross-Entropy: The Multiclass Generalization",
             fontsize=17, fontweight="bold", y=0.997, color=C_BOUNDARY)

gs = GridSpec(6, 2, figure=fig,
              height_ratios=[2.8, 3.0, 2.8, 3.0, 2.5, 1.2],
              hspace=0.25, wspace=0.25,
              top=0.975, bottom=0.01, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 1, LEFT: Binary — logistic function
# ─────────────────────────────────────────────────────────
ax_bin_fn = fig.add_subplot(gs[0, 0])
ax_bin_fn.axis("off")
ax_bin_fn.set_xlim(0, 10)
ax_bin_fn.set_ylim(-2, 8)

ax_bin_fn.text(5, 7.5, "Binary: Logistic Function",
               fontsize=13, ha="center", fontweight="bold", color=C_I)

ax_bin_fn.add_patch(patches.FancyBboxPatch(
    (0.3, 3.0), 9.4, 4.0, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))

ax_bin_fn.text(5, 6.4, "1 logit  in,  1 probability  out", fontsize=11,
               ha="center", fontweight="bold", color=C_I)

ax_bin_fn.text(5, 5.3, "P(+1)  =  σ(z)  =  1 / (1 + e⁻ᶻ)",
               fontsize=12, ha="center", fontfamily="monospace",
               color=C_BOUNDARY, fontweight="bold")

ax_bin_fn.text(5, 4.3, "P(−1)  =  1 − σ(z)  =  σ(−z)",
               fontsize=12, ha="center", fontfamily="monospace",
               color=MEDIUM)

ax_bin_fn.text(5, 3.4,
               "The two probabilities always sum to 1.",
               fontsize=10, ha="center", color=SUBTLE)

# Example
ax_bin_fn.text(5, 2.0, "Example:  z = 2.0", fontsize=11,
               ha="center", fontweight="bold", color=MEDIUM)
s = sigmoid(2.0)
ax_bin_fn.text(5, 1.2, f"P(+1) = σ(2) = {s:.3f}",
               fontsize=11, ha="center", fontfamily="monospace",
               color=C_POS, fontweight="bold")
ax_bin_fn.text(5, 0.5, f"P(−1) = σ(−2) = {1-s:.3f}",
               fontsize=11, ha="center", fontfamily="monospace",
               color=C_NEG, fontweight="bold")
ax_bin_fn.text(5, -0.3, f"sum = {s:.3f} + {1-s:.3f} = 1.000",
               fontsize=9.5, ha="center", fontfamily="monospace",
               color=SUBTLE)


# ─────────────────────────────────────────────────────────
#  ROW 1, RIGHT: Multiclass — softmax
# ─────────────────────────────────────────────────────────
ax_multi_fn = fig.add_subplot(gs[0, 1])
ax_multi_fn.axis("off")
ax_multi_fn.set_xlim(0, 10)
ax_multi_fn.set_ylim(-2, 8)

ax_multi_fn.text(5, 7.5, "Multiclass: Softmax",
                 fontsize=13, ha="center", fontweight="bold", color=C_B)

ax_multi_fn.add_patch(patches.FancyBboxPatch(
    (0.3, 3.0), 9.4, 4.0, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_multi_fn.text(5, 6.4, "K logits  in,  K probabilities  out", fontsize=11,
                 ha="center", fontweight="bold", color=C_B)

ax_multi_fn.text(5, 5.2, "P(class c)  =  e^(zc) / Σⱼ e^(zⱼ)",
                 fontsize=12, ha="center", fontfamily="monospace",
                 color=C_BOUNDARY, fontweight="bold")

ax_multi_fn.text(5, 4.2,
                 "Exponentiate each logit,  then normalize",
                 fontsize=10.5, ha="center", color=MEDIUM)
ax_multi_fn.text(5, 3.5,
                 "so all K probabilities sum to 1.",
                 fontsize=10.5, ha="center", color=SUBTLE)

# Example with 3 classes
ax_multi_fn.text(5, 2.0, "Example:  z = [ 2, 1, −1 ]", fontsize=11,
                 ha="center", fontweight="bold", color=MEDIUM)
logits = np.array([2.0, 1.0, -1.0])
probs = softmax(logits)
ax_multi_fn.text(5, 1.2, f"P(cat) = e²/(e²+e¹+e⁻¹) = {probs[0]:.3f}",
                 fontsize=10, ha="center", fontfamily="monospace",
                 color=C_CAT, fontweight="bold")
ax_multi_fn.text(5, 0.5, f"P(dog) = e¹/(e²+e¹+e⁻¹) = {probs[1]:.3f}",
                 fontsize=10, ha="center", fontfamily="monospace",
                 color=C_DOG, fontweight="bold")
ax_multi_fn.text(5, -0.2, f"P(bird) = e⁻¹/(e²+e¹+e⁻¹) = {probs[2]:.3f}",
                 fontsize=10, ha="center", fontfamily="monospace",
                 color=C_BIRD, fontweight="bold")
ax_multi_fn.text(5, -1.0,
                 f"sum = {probs[0]:.3f} + {probs[1]:.3f} + {probs[2]:.3f} = 1.000",
                 fontsize=9.5, ha="center", fontfamily="monospace",
                 color=SUBTLE)


# ─────────────────────────────────────────────────────────
#  ROW 2: Softmax worked example — logits → exp → normalize
# ─────────────────────────────────────────────────────────
ax_work = fig.add_subplot(gs[1, :])
ax_work.axis("off")
ax_work.set_xlim(0, 16)
ax_work.set_ylim(-7, 5.5)

ax_work.text(8, 5.0, "Inside Softmax: Step by Step",
             fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# Step 1: Start with logits
step_y = 3.5
ax_work.add_patch(patches.FancyBboxPatch(
    (0.5, step_y - 0.6), 4.5, 1.5, boxstyle="round,pad=0.12",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.2))
ax_work.text(2.75, step_y + 0.5, "Step 1: Raw logits", fontsize=11,
             ha="center", fontweight="bold", color=C_I)
ax_work.text(1.2, step_y - 0.1, "z_cat  =  2", fontsize=10,
             fontfamily="monospace", color=C_CAT)
ax_work.text(2.5, step_y - 0.1, "z_dog  =  1", fontsize=10,
             fontfamily="monospace", color=C_DOG)
ax_work.text(3.8, step_y - 0.1, "z_bird = −1", fontsize=10,
             fontfamily="monospace", color=C_BIRD)

# Arrow
ax_work.annotate("", xy=(5.8, step_y), xytext=(5.2, step_y),
                 arrowprops=dict(arrowstyle="->,head_width=0.15",
                                 color=SUBTLE, lw=1.5))

# Step 2: Exponentiate
ax_work.add_patch(patches.FancyBboxPatch(
    (5.8, step_y - 0.6), 4.5, 1.5, boxstyle="round,pad=0.12",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.2))
ax_work.text(8.05, step_y + 0.5, "Step 2: Exponentiate", fontsize=11,
             ha="center", fontweight="bold", color=C_BOUNDARY)
exps = np.exp(logits)
ax_work.text(6.2, step_y - 0.1, f"e² = {exps[0]:.2f}", fontsize=10,
             fontfamily="monospace", color=C_CAT)
ax_work.text(7.8, step_y - 0.1, f"e¹ = {exps[1]:.2f}", fontsize=10,
             fontfamily="monospace", color=C_DOG)
ax_work.text(9.0, step_y - 0.1, f"e⁻¹ = {exps[2]:.2f}", fontsize=10,
             fontfamily="monospace", color=C_BIRD)

# Arrow
ax_work.annotate("", xy=(11.1, step_y), xytext=(10.5, step_y),
                 arrowprops=dict(arrowstyle="->,head_width=0.15",
                                 color=SUBTLE, lw=1.5))

# Step 3: Normalize
ax_work.add_patch(patches.FancyBboxPatch(
    (11.1, step_y - 0.6), 4.5, 1.5, boxstyle="round,pad=0.12",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.2))
ax_work.text(13.35, step_y + 0.5, "Step 3: Normalize (÷ sum)", fontsize=11,
             ha="center", fontweight="bold", color=C_POS)
total = exps.sum()
ax_work.text(11.5, step_y - 0.1,
             f"sum = {total:.2f}", fontsize=10,
             fontfamily="monospace", color=MEDIUM)
ax_work.text(13.5, step_y - 0.1,
             f"÷ {total:.2f} each", fontsize=10,
             fontfamily="monospace", color=MEDIUM)

# ── Result: bar chart comparison ──
ax_work.text(8, 1.8, "Result: Logits vs Probabilities", fontsize=12,
             ha="center", fontweight="bold", color=C_RULE)

# Logit bars (left group)
classes = [("cat", 2, C_CAT), ("dog", 1, C_DOG), ("bird", -1, C_BIRD)]
bar_w = 1.2
logit_base_y = -1.5
prob_base_y = -1.5

# Logit bars
ax_work.text(3, 1.1, "Logits (raw scores)", fontsize=10,
             ha="center", fontweight="bold", color=C_I)
for i, (cls, logit, color) in enumerate(classes):
    bx = 1.5 + i * 1.8
    bar_h = logit * 0.6
    if logit > 0:
        ax_work.add_patch(patches.FancyBboxPatch(
            (bx - bar_w/2, logit_base_y), bar_w, bar_h,
            boxstyle="round,pad=0.04",
            facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.2))
    else:
        ax_work.add_patch(patches.FancyBboxPatch(
            (bx - bar_w/2, logit_base_y + bar_h), bar_w, -bar_h,
            boxstyle="round,pad=0.04",
            facecolor=color, alpha=0.15, edgecolor=color,
            linewidth=1.2, linestyle="--"))
    ax_work.text(bx, logit_base_y + bar_h + 0.15 if logit > 0
                 else logit_base_y + bar_h - 0.35,
                 f"z={logit}", fontsize=9, ha="center",
                 fontfamily="monospace", color=color, fontweight="bold")
    ax_work.text(bx, -2.2, cls, fontsize=9, ha="center", color=color)

ax_work.plot([0.5, 5.5], [logit_base_y, logit_base_y], "--",
             color=SUBTLE, lw=0.8, alpha=0.4)

# Arrow between groups
ax_work.annotate("softmax", xy=(7.0, -0.5), xytext=(6.0, -0.5),
                 fontsize=10, color=C_BOUNDARY, fontweight="bold",
                 ha="center",
                 arrowprops=dict(arrowstyle="->,head_width=0.15",
                                 color=C_BOUNDARY, lw=1.5))

# Probability bars (right group)
ax_work.text(11.5, 1.1, "Probabilities (softmax output)", fontsize=10,
             ha="center", fontweight="bold", color=C_POS)
max_prob = max(probs)
for i, (cls, color) in enumerate([(c, col) for c, _, col in classes]):
    bx = 10 + i * 1.8
    bar_h = probs[i] / max_prob * 1.8  # scale for display
    ax_work.add_patch(patches.FancyBboxPatch(
        (bx - bar_w/2, prob_base_y), bar_w, bar_h,
        boxstyle="round,pad=0.04",
        facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5))
    ax_work.text(bx, prob_base_y + bar_h + 0.15,
                 f"{probs[i]:.1%}", fontsize=10, ha="center",
                 fontfamily="monospace", color=color, fontweight="bold")
    ax_work.text(bx, -2.2, cls, fontsize=9, ha="center", color=color)

ax_work.plot([8.8, 13.5], [prob_base_y, prob_base_y], "--",
             color=SUBTLE, lw=0.8, alpha=0.4)

# Key observations
ax_work.add_patch(patches.FancyBboxPatch(
    (0.5, -6.5), 15, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_work.text(8, -3.5, "What Softmax Does", fontsize=12,
             ha="center", fontweight="bold", color=C_B)

observations = [
    ("Exponentiate:", "makes everything positive (no negative probabilities)", C_B),
    ("Normalize:", "forces everything to sum to 1 (valid probability distribution)", C_B),
    ("Amplifies:", "the largest logit gets the most probability — "
     "differences are exaggerated", C_B),
    ("Competition:", "increasing one class's probability automatically decreases the others", C_B),
]
for i, (prefix, rest, color) in enumerate(observations):
    y = -4.3 - i * 0.6
    ax_work.text(2.0, y, prefix, fontsize=9.5, color=color, fontweight="bold")
    ax_work.text(4.5, y, rest, fontsize=9.5, color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 3: Loss comparison — logistic vs cross-entropy
# ─────────────────────────────────────────────────────────
ax_bin_loss = fig.add_subplot(gs[2, 0])
ax_bin_loss.axis("off")
ax_bin_loss.set_xlim(0, 10)
ax_bin_loss.set_ylim(-3, 8)

ax_bin_loss.text(5, 7.5, "Binary: Logistic Loss",
                 fontsize=13, ha="center", fontweight="bold", color=C_I)

ax_bin_loss.add_patch(patches.FancyBboxPatch(
    (0.3, 3.5), 9.4, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))

ax_bin_loss.text(5, 6.5, "The loss formula", fontsize=11,
                 ha="center", fontweight="bold", color=C_I)

ax_bin_loss.text(5, 5.5, "L  =  −log P(correct class)",
                 fontsize=12, ha="center", fontfamily="monospace",
                 color=C_BOUNDARY, fontweight="bold")

ax_bin_loss.text(5, 4.6, "=  −log σ(margin)",
                 fontsize=12, ha="center", fontfamily="monospace",
                 color=MEDIUM)

ax_bin_loss.text(5, 3.8, "Only 2 classes, so 1 probability decides both.",
                 fontsize=9.5, ha="center", color=SUBTLE)

# Example
ax_bin_loss.add_patch(patches.FancyBboxPatch(
    (0.3, 0.0), 9.4, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.2))

ax_bin_loss.text(5, 2.5, "Example:  true label = +1,  z = 2.0",
                 fontsize=10, ha="center", fontweight="bold", color=MEDIUM)
p_correct = sigmoid(2.0)
loss_bin = -np.log(p_correct)
ax_bin_loss.text(5, 1.7, f"P(correct) = σ(2) = {p_correct:.3f}",
                 fontsize=10.5, ha="center", fontfamily="monospace",
                 color=C_POS)
ax_bin_loss.text(5, 0.9, f"L = −log({p_correct:.3f}) = {loss_bin:.3f}",
                 fontsize=11, ha="center", fontfamily="monospace",
                 color=C_BOUNDARY, fontweight="bold")
ax_bin_loss.text(5, 0.2, "Low loss — model is fairly confident and correct.",
                 fontsize=9, ha="center", color=SUBTLE)


# ─────────────────────────────────────────────────────────
#  ROW 3, RIGHT: Multiclass — cross-entropy loss
# ─────────────────────────────────────────────────────────
ax_multi_loss = fig.add_subplot(gs[2, 1])
ax_multi_loss.axis("off")
ax_multi_loss.set_xlim(0, 10)
ax_multi_loss.set_ylim(-3, 8)

ax_multi_loss.text(5, 7.5, "Multiclass: Cross-Entropy Loss",
                   fontsize=13, ha="center", fontweight="bold", color=C_B)

ax_multi_loss.add_patch(patches.FancyBboxPatch(
    (0.3, 3.5), 9.4, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_multi_loss.text(5, 6.5, "The loss formula", fontsize=11,
                   ha="center", fontweight="bold", color=C_B)

ax_multi_loss.text(5, 5.5, "L  =  −log P(correct class)",
                   fontsize=12, ha="center", fontfamily="monospace",
                   color=C_BOUNDARY, fontweight="bold")

ax_multi_loss.text(5, 4.6, "=  −log softmax(z)_true",
                   fontsize=12, ha="center", fontfamily="monospace",
                   color=MEDIUM)

ax_multi_loss.text(5, 3.8, "K classes, but we only look at the TRUE class's probability.",
                   fontsize=9.5, ha="center", color=SUBTLE)

# Example
ax_multi_loss.add_patch(patches.FancyBboxPatch(
    (0.3, 0.0), 9.4, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.2))

ax_multi_loss.text(5, 2.5, "Example:  true = cat,  z = [2, 1, −1]",
                   fontsize=10, ha="center", fontweight="bold", color=MEDIUM)
p_cat = probs[0]
loss_multi = -np.log(p_cat)
ax_multi_loss.text(5, 1.7, f"P(cat) = softmax([2,1,−1])₀ = {p_cat:.3f}",
                   fontsize=10.5, ha="center", fontfamily="monospace",
                   color=C_CAT)
ax_multi_loss.text(5, 0.9, f"L = −log({p_cat:.3f}) = {loss_multi:.3f}",
                   fontsize=11, ha="center", fontfamily="monospace",
                   color=C_BOUNDARY, fontweight="bold")
ax_multi_loss.text(5, 0.2,
                   "Moderate loss — model is correct but could be more confident.",
                   fontsize=9, ha="center", color=SUBTLE)


# ─────────────────────────────────────────────────────────
#  ROW 4: The unified parallel — same idea, different scale
# ─────────────────────────────────────────────────────────
ax_parallel = fig.add_subplot(gs[3, :])
ax_parallel.axis("off")
ax_parallel.set_xlim(0, 16)
ax_parallel.set_ylim(-7.5, 6)

ax_parallel.text(8, 5.5, "The Key Insight: It's the Same Idea at Different Scales",
                 fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# Table header
hy = 4.2
cols = [1.0, 5.5, 10.0]
headers = ["Step", "Binary", "Multiclass"]
h_colors = [TEXT, C_I, C_B]
for x, h, c in zip(cols, headers, h_colors):
    ax_parallel.text(x, hy, h, fontsize=12, fontweight="bold", color=c)

ax_parallel.plot([0.7, 15.3], [hy - 0.35, hy - 0.35], "-",
                 color=SUBTLE, alpha=0.5, lw=1)

# Table rows
rows = [
    ("Compute scores",
     "1 logit:  z = w · x + b",
     "K logits:  zc = wc · x + bc"),
    ("Convert to\nprobabilities",
     "Logistic function\nσ(z) = 1/(1+e⁻ᶻ)",
     "Softmax\nP(c) = e^zc / Σⱼ e^zⱼ"),
    ("Properties of\nprobabilities",
     "2 probs, sum to 1\nσ(z) + σ(−z) = 1",
     "K probs, sum to 1\nΣc P(c) = 1"),
    ("Define loss",
     "−log P(correct)\n= −log σ(margin)",
     "−log P(correct)\n= −log softmax(z)_true"),
    ("Name of loss",
     "Logistic loss",
     "Cross-entropy loss"),
    ("Gradient\nbehavior",
     "Smooth, always nonzero",
     "Smooth, always nonzero"),
]

for i, (step, binary, multi) in enumerate(rows):
    y = hy - 1.1 - i * 1.5
    ax_parallel.text(cols[0], y, step, fontsize=10, color=TEXT,
                     fontweight="bold")
    ax_parallel.text(cols[1], y, binary, fontsize=9.5,
                     fontfamily="monospace", color=C_I)
    ax_parallel.text(cols[2], y, multi, fontsize=9.5,
                     fontfamily="monospace", color=C_B)

    if i < len(rows) - 1:
        ax_parallel.plot([0.7, 15.3], [y - 0.55, y - 0.55], "-",
                         color=C_DIM, alpha=0.15, lw=0.5)

# Highlight the punchline
ax_parallel.add_patch(patches.FancyBboxPatch(
    (1.0, -7.0), 14, 1.5, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.5))

ax_parallel.text(8, -5.9,
                 "Both losses ask exactly the same question:",
                 fontsize=11, ha="center", color=TEXT)
ax_parallel.text(8, -6.6,
                 "\"How much probability did you put on the correct answer?\"",
                 fontsize=12, ha="center", color=C_BOUNDARY,
                 fontweight="bold", fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 5: Why softmax & cross-entropy — what they buy you
# ─────────────────────────────────────────────────────────
ax_why = fig.add_subplot(gs[4, :])
ax_why.axis("off")
ax_why.set_xlim(0, 16)
ax_why.set_ylim(-5, 5)

ax_why.text(8, 4.5, "Why Not Just Pick the Largest Logit?",
            fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# Left: argmax problem
ax_why.add_patch(patches.FancyBboxPatch(
    (0.5, 0.5), 7, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1.5))

ax_why.text(4, 3.5, "Just using argmax (pick largest zc)",
            fontsize=11, ha="center", fontweight="bold", color=C_NEG)
ax_why.text(4, 2.6,
            "Same problem as 0–1 loss in binary!",
            fontsize=10.5, ha="center", color=C_NEG, fontweight="bold")
ax_why.text(4, 1.8,
            "Small changes to w rarely change which",
            fontsize=10, ha="center", color=MEDIUM)
ax_why.text(4, 1.2,
            "class has the highest score.",
            fontsize=10, ha="center", color=MEDIUM)
ax_why.text(4, 0.65,
            "Gradient = 0 almost everywhere.  Stuck.",
            fontsize=10, ha="center", color=C_NEG)

# Right: softmax + cross-entropy solution
ax_why.add_patch(patches.FancyBboxPatch(
    (8.5, 0.5), 7, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.5))

ax_why.text(12, 3.5, "Softmax + cross-entropy loss",
            fontsize=11, ha="center", fontweight="bold", color=C_POS)
ax_why.text(12, 2.6,
            "Same fix as logistic loss in binary!",
            fontsize=10.5, ha="center", color=C_POS, fontweight="bold")
ax_why.text(12, 1.8,
            "Probabilities are smooth and continuous.",
            fontsize=10, ha="center", color=MEDIUM)
ax_why.text(12, 1.2,
            "−log makes the loss differentiable.",
            fontsize=10, ha="center", color=MEDIUM)
ax_why.text(12, 0.65,
            "Gradient always has signal.  Optimization works.",
            fontsize=10, ha="center", color=C_POS)

# Binary = special case
ax_why.add_patch(patches.FancyBboxPatch(
    (1.5, -4.3), 13, 3.8, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax_why.text(8, -1.0, "The Relationship", fontsize=12,
            ha="center", fontweight="bold", color=C_B)
ax_why.text(8, -1.8,
            "Binary classification is just multiclass with K = 2.",
            fontsize=11, ha="center", color=TEXT)
ax_why.text(8, -2.6,
            "Logistic function is softmax with 2 classes.",
            fontsize=10.5, ha="center", color=MEDIUM)
ax_why.text(8, -3.3,
            "Logistic loss is cross-entropy with 2 classes.",
            fontsize=10.5, ha="center", color=MEDIUM)
ax_why.text(8, -4.0,
            "There is only one framework.  Binary was a special case all along.",
            fontsize=10.5, ha="center", color=C_B, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 6: Summary
# ─────────────────────────────────────────────────────────
ax_sum = fig.add_subplot(gs[5, :])
ax_sum.axis("off")
ax_sum.set_xlim(0, 16)
ax_sum.set_ylim(-3, 3)

ax_sum.text(8, 2.5, "Summary", fontsize=14, ha="center",
            fontweight="bold", color=C_RULE)

insight_box(ax_sum, 2.0, -2.5, [
    ("Softmax generalizes the logistic function: "
     "K logits → K probabilities that sum to 1.", True),
    ("Cross-entropy loss generalizes logistic loss: "
     "L = −log P(correct class).", True),
    ("Both give the same smooth, optimizable objective that "
     "0–1 loss / argmax cannot provide.", False),
    ("Binary classification is just the K=2 special case of this "
     "unified framework.", False),
], w_box=12, line_h=0.55)


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_3_visuals/softmax_cross_entropy.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
