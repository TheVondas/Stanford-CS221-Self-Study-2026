"""
Lecture 3 Visual: Tokenization → One-Hot Encoding → Bag of Words

Shows:
1. Raw text split into tokens, tokens mapped to indices
2. Each token index converted to a one-hot vector, assembled into a matrix
3. The one-hot matrix summed/averaged into a single bag-of-words vector
4. What information survives and what is lost
5. The "dog bites man" vs "man bites dog" demonstration
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
C_WARN = "#f5c2e7"

# Token colors
C_THE = "#89b4fa"
C_CAT = "#f38ba8"
C_IN = "#a6e3a1"
C_HAT = "#f9e2af"
C_SAT = "#cba6f7"
C_ON = "#fab387"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


def draw_token_box(ax, x, y, word, color, w=1.4, h=0.8):
    ax.add_patch(patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08",
        facecolor=color, alpha=0.20, edgecolor=color, linewidth=1.5))
    ax.text(x, y, word, fontsize=11, ha="center", va="center",
            color=color, fontweight="bold", fontfamily="monospace")


def draw_arrow_down(ax, x, y_from, y_to, color=SUBTLE):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="->,head_width=0.12",
                                color=color, lw=1.3))


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
fig = plt.figure(figsize=(16, 48))
fig.suptitle("From Text to Tensor: Tokenization → One-Hot → Bag of Words",
             fontsize=17, fontweight="bold", y=0.997, color=C_BOUNDARY)

gs = GridSpec(6, 1, figure=fig,
              height_ratios=[2.5, 3.5, 3.5, 3.0, 3.5, 1.2],
              hspace=0.18,
              top=0.975, bottom=0.01, left=0.06, right=0.94)


# ─────────────────────────────────────────────────────────
#  ROW 1: Tokenization — raw text → tokens → indices
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.axis("off")
ax1.set_xlim(0, 16)
ax1.set_ylim(-3, 6)

ax1.text(8, 5.5, "Step 1:  Tokenization",
         fontsize=15, ha="center", fontweight="bold", color=C_RULE)

# Raw text
ax1.add_patch(patches.FancyBboxPatch(
    (2, 3.5), 12, 1.3, boxstyle="round,pad=0.12",
    facecolor=SUBTLE, alpha=0.08, edgecolor=SUBTLE, linewidth=1.2))
ax1.text(8, 4.1, '"the  cat  in  the  hat"',
         fontsize=16, ha="center", fontfamily="monospace",
         color=TEXT, fontweight="bold")
ax1.text(1.0, 4.1, "Raw text:", fontsize=10, color=SUBTLE, fontweight="bold")

draw_arrow_down(ax1, 8, 3.3, 2.8)

# Tokens
ax1.text(1.0, 2.2, "Tokens:", fontsize=10, color=SUBTLE, fontweight="bold")
tokens = [("the", C_THE), ("cat", C_CAT), ("in", C_IN),
          ("the", C_THE), ("hat", C_HAT)]
for i, (word, color) in enumerate(tokens):
    draw_token_box(ax1, 3.5 + i * 2.2, 2.2, word, color)

draw_arrow_down(ax1, 8, 1.6, 1.1)

# Indices
ax1.text(1.0, 0.4, "Indices:", fontsize=10, color=SUBTLE, fontweight="bold")
indices = [(0, C_THE), (1, C_CAT), (2, C_IN), (0, C_THE), (3, C_HAT)]
for i, (idx, color) in enumerate(indices):
    draw_token_box(ax1, 3.5 + i * 2.2, 0.4, str(idx), color, w=1.0)

# Vocabulary
ax1.add_patch(patches.FancyBboxPatch(
    (1.5, -2.5), 13, 1.8, boxstyle="round,pad=0.12",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.2))
ax1.text(8, -1.1, "Vocabulary (unique tokens):", fontsize=10,
         ha="center", fontweight="bold", color=C_B)

vocab = [("the", 0, C_THE), ("cat", 1, C_CAT),
         ("in", 2, C_IN), ("hat", 3, C_HAT)]
for i, (word, idx, color) in enumerate(vocab):
    x = 3.0 + i * 3.0
    ax1.text(x, -1.9, f'"{word}" = {idx}', fontsize=11,
             ha="center", fontfamily="monospace", color=color,
             fontweight="bold")

ax1.text(14, -1.9, "vocab size = 4", fontsize=9, color=SUBTLE)


# ─────────────────────────────────────────────────────────
#  ROW 2: One-hot encoding — indices → vectors → matrix
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_xlim(0, 16)
ax2.set_ylim(-7, 6)

ax2.text(8, 5.5, "Step 2:  One-Hot Encoding",
         fontsize=15, ha="center", fontweight="bold", color=C_RULE)

ax2.text(8, 4.5,
         "Each token index becomes a vector: all zeros except a 1 at that index.",
         fontsize=10.5, ha="center", color=MEDIUM)

# One-hot vectors for each position
onehot_data = [
    ("the",  0, [1, 0, 0, 0], C_THE),
    ("cat",  1, [0, 1, 0, 0], C_CAT),
    ("in",   2, [0, 0, 1, 0], C_IN),
    ("the",  0, [1, 0, 0, 0], C_THE),
    ("hat",  3, [0, 0, 0, 1], C_HAT),
]

# Draw each one-hot vector as a row of cells
cell_w = 0.7
cell_h = 0.55
start_x = 5.5

for pos, (word, idx, vec, color) in enumerate(onehot_data):
    y = 3.2 - pos * 1.2

    # Position label
    ax2.text(2.5, y, f'pos {pos}:  "{word}"', fontsize=10,
             fontfamily="monospace", color=color, fontweight="bold")
    ax2.text(4.8, y, "=", fontsize=11, color=SUBTLE)

    # Draw the vector cells
    ax2.text(start_x - 0.3, y, "[", fontsize=14, fontfamily="monospace",
             color=C_DIM, va="center")
    for j, val in enumerate(vec):
        cx = start_x + j * (cell_w + 0.15)
        cell_color = color if val == 1 else BG
        cell_alpha = 0.35 if val == 1 else 0.0
        edge_color = color if val == 1 else C_DIM
        edge_alpha = 1.0 if val == 1 else 0.3

        ax2.add_patch(patches.FancyBboxPatch(
            (cx - cell_w/2, y - cell_h/2), cell_w, cell_h,
            boxstyle="round,pad=0.04",
            facecolor=cell_color, alpha=cell_alpha,
            edgecolor=edge_color, linewidth=1.2))
        ax2.text(cx, y, str(val), fontsize=11, ha="center", va="center",
                 fontfamily="monospace",
                 color=color if val == 1 else SUBTLE,
                 fontweight="bold" if val == 1 else "normal")

    close_x = start_x + 3 * (cell_w + 0.15) + cell_w/2
    ax2.text(close_x, y, "]", fontsize=14, fontfamily="monospace",
             color=C_DIM, va="center")

    # Column labels (only for first row)
    if pos == 0:
        col_labels = [("the", C_THE), ("cat", C_CAT),
                      ("in", C_IN), ("hat", C_HAT)]
        for j, (lbl, lc) in enumerate(col_labels):
            cx = start_x + j * (cell_w + 0.15)
            ax2.text(cx, y + 0.7, lbl, fontsize=8, ha="center",
                     color=lc, fontfamily="monospace")

# Matrix interpretation
ax2.add_patch(patches.FancyBboxPatch(
    (0.5, -6.5), 15, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax2.text(8, -4.0, "The One-Hot Matrix (5 positions x 4 vocab)", fontsize=12,
         ha="center", fontweight="bold", color=C_B)

ax2.text(8, -4.9,
         "Each ROW = one position in the sentence.    "
         "Each COLUMN = one word in the vocabulary.",
         fontsize=10, ha="center", color=MEDIUM)

ax2.text(8, -5.7,
         "The 1 in each row says: \"the word at this position is ___\".",
         fontsize=10, ha="center", color=C_B)
ax2.text(8, -6.3,
         "This is an identity marker — it doesn't encode meaning, just which word is present.",
         fontsize=9.5, ha="center", color=SUBTLE, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 3: Bag of words — sum the one-hot vectors
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.axis("off")
ax3.set_xlim(0, 16)
ax3.set_ylim(-6, 6)

ax3.text(8, 5.5, "Step 3:  Bag of Words  (Sum the One-Hot Vectors)",
         fontsize=15, ha="center", fontweight="bold", color=C_RULE)

# Show the sum visually
ax3.text(8, 4.3,
         "Add up all 5 one-hot vectors into a single vector:",
         fontsize=11, ha="center", color=MEDIUM)

# The five vectors being summed
sum_vecs = [
    ([1, 0, 0, 0], C_THE),
    ([0, 1, 0, 0], C_CAT),
    ([0, 0, 1, 0], C_IN),
    ([1, 0, 0, 0], C_THE),
    ([0, 0, 0, 1], C_HAT),
]

small_w = 0.5
small_h = 0.4

for vi, (vec, color) in enumerate(sum_vecs):
    x_base = 1.0 + vi * 2.2
    y_base = 3.2
    ax3.text(x_base + 0.75, y_base + 0.5,
             f'"{["the","cat","in","the","hat"][vi]}"',
             fontsize=8, ha="center", color=color, fontfamily="monospace")

    ax3.text(x_base - 0.15, y_base, "[", fontsize=10,
             fontfamily="monospace", color=C_DIM, va="center")
    for j, val in enumerate(vec):
        cx = x_base + j * (small_w + 0.05)
        ax3.text(cx + small_w/2, y_base, str(val), fontsize=9,
                 ha="center", va="center", fontfamily="monospace",
                 color=color if val == 1 else SUBTLE,
                 fontweight="bold" if val == 1 else "normal")
    ax3.text(x_base + 3 * (small_w + 0.05) + small_w + 0.1, y_base,
             "]", fontsize=10, fontfamily="monospace", color=C_DIM,
             va="center")

    if vi < 4:
        ax3.text(x_base + 2.05, y_base, "+", fontsize=13,
                 ha="center", color=SUBTLE, fontweight="bold")

# Equals sign and result
ax3.text(13.5, 3.2, "=", fontsize=16, ha="center",
         color=C_BOUNDARY, fontweight="bold")

# Draw arrow down to result
draw_arrow_down(ax3, 8, 2.5, 1.8, C_BOUNDARY)

# Result vector — large and prominent
result = [2, 1, 1, 1]
result_labels = ["the", "cat", "in", "hat"]
result_colors = [C_THE, C_CAT, C_IN, C_HAT]

ax3.add_patch(patches.FancyBboxPatch(
    (2.5, -1.0), 11, 2.5, boxstyle="round,pad=0.15",
    facecolor=C_BOUNDARY, alpha=0.06, edgecolor=C_BOUNDARY, linewidth=1.5))

ax3.text(8, 1.2, "Bag-of-Words Vector", fontsize=13,
         ha="center", fontweight="bold", color=C_BOUNDARY)

big_w = 1.6
big_h = 1.0
bx_start = 3.5

ax3.text(bx_start - 0.7, 0.0, "[", fontsize=22, fontfamily="monospace",
         color=C_DIM, va="center")
for j, (val, label, color) in enumerate(zip(result, result_labels, result_colors)):
    cx = bx_start + j * (big_w + 0.5)

    ax3.add_patch(patches.FancyBboxPatch(
        (cx - big_w/2, -big_h/2), big_w, big_h,
        boxstyle="round,pad=0.06",
        facecolor=color, alpha=0.30, edgecolor=color, linewidth=2))
    ax3.text(cx, 0.0, str(val), fontsize=18, ha="center", va="center",
             fontfamily="monospace", color=color, fontweight="bold")
    ax3.text(cx, -0.85, label, fontsize=10, ha="center",
             fontfamily="monospace", color=color)

ax3.text(bx_start + 3 * (big_w + 0.5) + big_w/2 + 0.3, 0.0, "]",
         fontsize=22, fontfamily="monospace", color=C_DIM, va="center")

# Interpretation
ax3.text(8, -1.8, 'Reads as:  "the" appears 2x,   "cat" 1x,   "in" 1x,   "hat" 1x',
         fontsize=11, ha="center", color=TEXT, fontweight="bold")

# What it keeps vs loses
ax3.add_patch(patches.FancyBboxPatch(
    (0.5, -5.5), 7, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.5))

ax3.text(4, -2.9, "What it KEEPS", fontsize=12,
         ha="center", fontweight="bold", color=C_POS)
ax3.text(4, -3.7, "Which words are present", fontsize=10,
         ha="center", color=MEDIUM)
ax3.text(4, -4.3, "How often each word appears", fontsize=10,
         ha="center", color=MEDIUM)
ax3.text(4, -4.9, "Fixed-size vector (always vocab-length)",
         fontsize=10, ha="center", color=MEDIUM)

ax3.add_patch(patches.FancyBboxPatch(
    (8.5, -5.5), 7, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1.5))

ax3.text(12, -2.9, "What it LOSES", fontsize=12,
         ha="center", fontweight="bold", color=C_NEG)
ax3.text(12, -3.7, "Word ORDER (which came first)", fontsize=10,
         ha="center", color=MEDIUM)
ax3.text(12, -4.3, "Phrase STRUCTURE (who did what)", fontsize=10,
         ha="center", color=MEDIUM)
ax3.text(12, -4.9, "Position information (where in the sentence)",
         fontsize=10, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 4: The "dog bites man" demonstration
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
ax4.axis("off")
ax4.set_xlim(0, 16)
ax4.set_ylim(-5.5, 6)

ax4.text(8, 5.5, 'Why Order Loss Matters:  "dog bites man"  vs  "man bites dog"',
         fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# Vocab: dog=0, bites=1, man=2
C_DOG = "#89b4fa"
C_BITES = "#fab387"
C_MAN = "#cba6f7"
demo_vocab = [("dog", 0, C_DOG), ("bites", 1, C_BITES), ("man", 2, C_MAN)]

# Sentence 1
ax4.add_patch(patches.FancyBboxPatch(
    (0.5, 1.5), 7, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))

ax4.text(4, 4.5, '"dog  bites  man"', fontsize=14,
         ha="center", fontfamily="monospace", fontweight="bold", color=C_I)

# Tokens
for i, (word, _, color) in enumerate(demo_vocab):
    draw_token_box(ax4, 2 + i * 2.0, 3.4, word, color, w=1.3, h=0.6)

# BoW
ax4.text(4, 2.5, "Bag of words:", fontsize=10,
         ha="center", fontweight="bold", color=MEDIUM)
bow_vals = [1, 1, 1]
for j, (val, (_, _, color)) in enumerate(zip(bow_vals, demo_vocab)):
    cx = 2.0 + j * 2.0
    ax4.add_patch(patches.FancyBboxPatch(
        (cx - 0.45, 1.65), 0.9, 0.7, boxstyle="round,pad=0.04",
        facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5))
    ax4.text(cx, 2.0, str(val), fontsize=14, ha="center", va="center",
             fontfamily="monospace", color=color, fontweight="bold")

# Sentence 2
ax4.add_patch(patches.FancyBboxPatch(
    (8.5, 1.5), 7, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax4.text(12, 4.5, '"man  bites  dog"', fontsize=14,
         ha="center", fontfamily="monospace", fontweight="bold", color=C_B)

# Tokens (different order)
reorder = [("man", 2, C_MAN), ("bites", 1, C_BITES), ("dog", 0, C_DOG)]
for i, (word, _, color) in enumerate(reorder):
    draw_token_box(ax4, 10 + i * 2.0, 3.4, word, color, w=1.3, h=0.6)

# BoW — same!
ax4.text(12, 2.5, "Bag of words:", fontsize=10,
         ha="center", fontweight="bold", color=MEDIUM)
for j, (val, (_, _, color)) in enumerate(zip(bow_vals, demo_vocab)):
    cx = 10.0 + j * 2.0
    ax4.add_patch(patches.FancyBboxPatch(
        (cx - 0.45, 1.65), 0.9, 0.7, boxstyle="round,pad=0.04",
        facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5))
    ax4.text(cx, 2.0, str(val), fontsize=14, ha="center", va="center",
             fontfamily="monospace", color=color, fontweight="bold")

# EQUALS sign between them
ax4.text(8, 2.0, "=", fontsize=28, ha="center", va="center",
         color=C_NEG, fontweight="bold")

# Punchline
ax4.add_patch(patches.FancyBboxPatch(
    (1.5, -1.5), 13, 2.5, boxstyle="round,pad=0.15",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1.5))

ax4.text(8, 0.5, "[ 1, 1, 1 ]  =  [ 1, 1, 1 ]", fontsize=16,
         ha="center", fontfamily="monospace", color=C_NEG, fontweight="bold")

ax4.text(8, -0.3,
         "Completely different meanings.  Identical bag-of-words vectors.",
         fontsize=11, ha="center", color=C_NEG, fontweight="bold")

ax4.text(8, -1.1,
         "Bag of words is a lossy compression — it trades away order for simplicity.",
         fontsize=10.5, ha="center", color=MEDIUM)

# Why it still matters
ax4.add_patch(patches.FancyBboxPatch(
    (1.5, -5.0), 13, 2.8, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.5))

ax4.text(8, -2.7, "Why Bag of Words Still Works for Many Tasks",
         fontsize=12, ha="center", fontweight="bold", color=C_POS)
ax4.text(8, -3.5,
         "For tasks like spam detection or sentiment analysis, WHICH words appear",
         fontsize=10, ha="center", color=MEDIUM)
ax4.text(8, -4.1,
         "often matters more than their exact order.  \"terrible awful bad\" is negative",
         fontsize=10, ha="center", color=MEDIUM)
ax4.text(8, -4.7,
         "regardless of word order.  BoW captures that.",
         fontsize=10, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 5: Full pipeline — text to classifier
# ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[4])
ax5.axis("off")
ax5.set_xlim(0, 16)
ax5.set_ylim(-6, 6)

ax5.text(8, 5.5, "The Full Pipeline: Text → Tensor → Classifier",
         fontsize=14, ha="center", fontweight="bold", color=C_RULE)

# Pipeline boxes
stages = [
    (1.5, "Raw Text", '"the cat\nin the hat"', SUBTLE, SUBTLE),
    (4.5, "Tokenize", '["the","cat",\n"in","the","hat"]', C_I, C_I),
    (7.5, "Indices", "[0, 1, 2, 0, 3]", C_B, C_B),
    (10.5, "Bag of\nWords", "[2, 1, 1, 1]", C_BOUNDARY, C_BOUNDARY),
    (13.5, "Linear\nClassifier", "z = w . x + b", C_POS, C_POS),
]

for (cx, title, content, color, edge) in stages:
    ax5.add_patch(patches.FancyBboxPatch(
        (cx - 1.2, 1.0), 2.4, 3.5, boxstyle="round,pad=0.12",
        facecolor=color, alpha=0.06, edgecolor=edge, linewidth=1.5))
    ax5.text(cx, 4.0, title, fontsize=10.5, ha="center",
             fontweight="bold", color=color)
    ax5.text(cx, 2.5, content, fontsize=9, ha="center",
             fontfamily="monospace", color=MEDIUM)

# Arrows between stages
for i in range(4):
    x_from = stages[i][0] + 1.3
    x_to = stages[i + 1][0] - 1.3
    ax5.annotate("", xy=(x_to, 2.8), xytext=(x_from, 2.8),
                 arrowprops=dict(arrowstyle="->,head_width=0.12",
                                 color=C_BOUNDARY, lw=1.5))

# What each step does
step_labels = [
    "split into\nwords",
    "map to\nnumbers",
    "sum one-hot\nvectors",
    "compute\nlogits",
]
for i, label in enumerate(step_labels):
    x_mid = (stages[i][0] + stages[i + 1][0]) / 2
    ax5.text(x_mid, 1.7, label, fontsize=8, ha="center",
             color=SUBTLE, fontstyle="italic")

# The key message
ax5.add_patch(patches.FancyBboxPatch(
    (1.0, -3.5), 14, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax5.text(8, -0.5, "Why This Pipeline Matters", fontsize=12,
         ha="center", fontweight="bold", color=C_B)

points = [
    "The linear classifier from earlier in the lecture needs a fixed-size numeric vector as input.",
    "Text is a variable-length string — it can't be fed directly into w . x + b.",
    "This pipeline solves that: any text, any length → one fixed-size vector (dimension = vocab size).",
    "That vector can now be the x in our classifier.  The entire lecture connects end to end.",
]
for i, pt in enumerate(points):
    ax5.text(8, -1.3 - i * 0.7, pt,
             fontsize=10, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 6: Summary
# ─────────────────────────────────────────────────────────
ax_sum = fig.add_subplot(gs[5])
ax_sum.axis("off")
ax_sum.set_xlim(0, 16)
ax_sum.set_ylim(-3, 3)

ax_sum.text(8, 2.5, "Summary", fontsize=14, ha="center",
            fontweight="bold", color=C_RULE)

insight_box(ax_sum, 2.0, -2.5, [
    ("Tokenization splits text into units and assigns each a numeric index.", True),
    ("One-hot encoding turns each index into a sparse vector with a single 1.", True),
    ("Bag of words sums those vectors into one fixed-size vector — "
     "keeping word presence and frequency, losing word order.", False),
    ("This gives the linear classifier a valid input vector x, "
     "completing the text classification pipeline.", False),
], w_box=12, line_h=0.55)


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_3_visuals/tokenization_to_bow.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
