"""
Lecture 5 Visual: Search and Language Models — Test-Time Compute

Shows:
1. The problem: one-shot generation vs search — why a single sample isn't enough
2. Mapping LM generation onto the search framework (state, action, cost)
3. Why -log p: minimizing sum of -log p = maximizing product of p (with visual)
4. The verifier bonus hack: folding correctness into the cost
5. Concrete example: best-of-n with an LM on a math prompt, showing multiple
   rollouts token-by-token with costs, verifier check, and winner selection
6. The modern synthesis: learning provides costs, search finds better solutions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ── Style ────────────────────────────────────────────────
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


def box(ax, xy, w, h, color, alpha_f=0.06, lw=1.5):
    ax.add_patch(patches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.15",
        facecolor=color, alpha=alpha_f, edgecolor=color, linewidth=lw))


def node(ax, cx, cy, label, color=C_NODE, r=0.28, fs=11, lw=2.0,
         alpha_f=0.12):
    c = plt.Circle((cx, cy), r, facecolor=color, alpha=alpha_f,
                    edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(c)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color, zorder=6)


# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 56))
fig.suptitle("Search and Language Models: Test-Time Compute",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(7, 1, figure=fig,
              height_ratios=[3.5, 4.0, 3.5, 3.0, 8.5, 3.5, 1.8],
              hspace=0.12,
              top=0.975, bottom=0.008, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: The Problem — One-Shot vs Search
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-2, 8)
ax0.set_title("  The Problem: Why One Answer Isn't Enough",
              fontsize=14, fontweight="bold", color=C_NEG, pad=10, loc="left")

# One-shot side
box(ax0, (0.3, 3.0), 7.0, 4.5, C_NEG, alpha_f=0.06)
ax0.text(3.8, 7.0, "One-Shot Generation", fontsize=12, ha="center",
         fontweight="bold", color=C_NEG)

ax0.text(3.8, 6.1, 'Prompt:  "What is 37 x 48?"', fontsize=10,
         ha="center", fontfamily="monospace", color=MEDIUM)

# Arrow down
ax0.annotate("", xy=(3.8, 5.0), xytext=(3.8, 5.6),
             arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=2))
ax0.text(5.2, 5.4, "sample once", fontsize=9, color=C_NEG)

ax0.add_patch(patches.FancyBboxPatch(
    (1.2, 3.5), 5.2, 1.2, boxstyle="round,pad=0.1",
    facecolor=C_NEG, alpha=0.10, edgecolor=C_NEG, linewidth=1.5))
ax0.text(3.8, 4.1, '"1776"     (might be wrong)', fontsize=10.5,
         ha="center", fontfamily="monospace", color=C_NEG, fontweight="bold")

# Search side
box(ax0, (8.5, 3.0), 7.2, 4.5, C_POS, alpha_f=0.06)
ax0.text(12.1, 7.0, "Search Over Multiple Responses", fontsize=12,
         ha="center", fontweight="bold", color=C_POS)

ax0.text(12.1, 6.1, 'Same prompt, but generate n responses', fontsize=10,
         ha="center", color=MEDIUM)

ax0.annotate("", xy=(12.1, 5.0), xytext=(12.1, 5.6),
             arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=2))
ax0.text(13.5, 5.4, "sample n times", fontsize=9, color=C_POS)

responses = [
    ('"1776"', C_NEG, "x"),
    ('"1786"', C_NEG, "x"),
    ('"1776"', C_BOUND, "check"),
]
for i, (resp, c, status) in enumerate(responses):
    y = 4.6 - i * 0.5
    ax0.text(9.8, y, resp, fontsize=9.5, fontfamily="monospace", color=c)
    ax0.text(12.0, y, "...", fontsize=9, color=C_DIM)

ax0.add_patch(patches.FancyBboxPatch(
    (9.0, 3.3), 6.2, 0.7, boxstyle="round,pad=0.08",
    facecolor=C_POS, alpha=0.10, edgecolor=C_POS, linewidth=1.5))
ax0.text(12.1, 3.65, '"1776"   pick the best!', fontsize=10.5,
         ha="center", fontfamily="monospace", color=C_POS, fontweight="bold")

# Bottom: what is test-time compute
box(ax0, (0.3, -1.5), 15.4, 4.0, C_RULE, alpha_f=0.06)
ax0.text(8, 2.0, "This Is Test-Time Compute", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)
ax0.text(8, 1.1,
    "Instead of accepting the first answer a model produces,",
    fontsize=11, ha="center", color=MEDIUM)
ax0.text(8, 0.4,
    "spend extra computation at INFERENCE TIME to search over multiple candidates.",
    fontsize=11.5, ha="center", color=C_RULE, fontweight="bold")
ax0.text(8, -0.4,
    "Training time is fixed (already happened).  Test time is where search happens.",
    fontsize=10.5, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 1: Mapping LM Generation onto Search Framework
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(0, 16); ax1.set_ylim(-2.5, 8)
ax1.set_title("  Language Generation AS a Search Problem",
              fontsize=14, fontweight="bold", color=C_I, pad=10, loc="left")

# Walk/tram mapping on left, LM mapping on right
box(ax1, (0.3, 2.5), 7.2, 5.0, C_NODE, alpha_f=0.04, lw=1.2)
ax1.text(3.9, 7.0, "Walk/Tram  (previous)", fontsize=11, ha="center",
         fontweight="bold", color=C_NODE)

box(ax1, (8.5, 2.5), 7.2, 5.0, C_I, alpha_f=0.06, lw=1.5)
ax1.text(12.1, 7.0, "Language Model", fontsize=11, ha="center",
         fontweight="bold", color=C_I)

mappings = [
    ("State", "current location  i",
     "prompt + response\nprefix so far"),
    ("Action", "walk or tram",
     "predict the\nnext token"),
    ("Cost", "1 (walk) or 2 (tram)",
     '-log p(token | prefix)\n"how surprising\nis this token?"'),
    ("End test", "location == n",
     "end-of-sequence\ntoken generated"),
]

for i, (label, walk, lm) in enumerate(mappings):
    y = 6.0 - i * 1.1
    ax1.text(0.8, y, label, fontsize=10, fontweight="bold", color=C_BOUND)
    ax1.text(3.9, y, walk, fontsize=9.5, ha="center", color=MEDIUM,
             linespacing=1.2)
    ax1.text(12.1, y, lm, fontsize=9.5, ha="center", color=C_I,
             fontweight="bold", linespacing=1.2)

# Concrete state example
box(ax1, (0.3, -2.0), 15.4, 4.0, C_B, alpha_f=0.05)
ax1.text(8, 1.5, "Example:  What a \"State\" Looks Like in LM Search",
         fontsize=12, ha="center", fontweight="bold", color=C_B)

states_ex = [
    ('State 0:', '"What is 37 x 48? "', "just the prompt"),
    ('State 1:', '"What is 37 x 48? Let"', "+ 1 token generated"),
    ('State 2:', '"What is 37 x 48? Let me"', "+ 2 tokens generated"),
    ('State 3:', '"What is 37 x 48? Let me compute"', "+ 3 tokens ..."),
]
for i, (lbl, state, note) in enumerate(states_ex):
    y = 0.65 - i * 0.6
    ax1.text(0.8, y, lbl, fontsize=9, color=C_B, fontweight="bold")
    ax1.text(3.0, y, state, fontsize=9, fontfamily="monospace", color=C_B)
    ax1.text(13.5, y, note, fontsize=8.5, color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 2: Why -log p  (the cost conversion)
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(0, 16); ax2.set_ylim(-2.5, 7.5)
ax2.set_title("  Why Negative Log Probability?",
              fontsize=14, fontweight="bold", color=C_BOUND, pad=10, loc="left")

# The key equation
box(ax2, (0.3, 4.0), 15.4, 3.0, C_BOUND, alpha_f=0.08)
ax2.text(8, 6.5, "The Conversion", fontsize=13, ha="center",
         fontweight="bold", color=C_BOUND)
ax2.text(8, 5.5,
    "We WANT:  maximize  p(token1) x p(token2) x p(token3) x ...",
    fontsize=11, ha="center", fontfamily="monospace", color=C_POS,
    fontweight="bold")
ax2.text(8, 4.6,
    "We USE:    minimize  -log p(t1)  +  -log p(t2)  +  -log p(t3)  + ...",
    fontsize=11, ha="center", fontfamily="monospace", color=C_BOUND,
    fontweight="bold")

# Why they're equivalent
box(ax2, (0.3, -2.0), 7.2, 5.5, C_I, alpha_f=0.05)
ax2.text(3.9, 3.0, "Why These Are Equivalent", fontsize=11, ha="center",
         fontweight="bold", color=C_I)
why_lines = [
    "log turns products into sums:",
    "  log(a x b) = log(a) + log(b)",
    "",
    "Negation flips max into min:",
    "  argmax P = argmin (-log P)",
    "",
    "So search (which minimizes cost)",
    "naturally finds the most probable",
    "sequence.",
]
for i, line in enumerate(why_lines):
    ax2.text(1.0, 2.2 - i * 0.5, line, fontsize=9.5,
             fontfamily="monospace", color=MEDIUM if line else TEXT)

# Intuitive example
box(ax2, (8.5, -2.0), 7.2, 5.5, C_POS, alpha_f=0.05)
ax2.text(12.1, 3.0, "Intuitive Example", fontsize=11, ha="center",
         fontweight="bold", color=C_POS)

ex_tokens = [
    ("Token", "p(token)", "-log p", "Meaning"),
    ('"the"', "0.90", "0.10", "very expected"),
    ('"cat"', "0.30", "1.20", "somewhat expected"),
    ('"xylophone"', "0.01", "4.60", "very surprising"),
]
for i, (tok, prob, cost, meaning) in enumerate(ex_tokens):
    y = 2.2 - i * 0.7
    is_header = (i == 0)
    c_tok = C_BOUND if is_header else C_POS
    fw = "bold"
    ax2.text(9.0, y, tok, fontsize=9.5, color=c_tok, fontweight=fw,
             fontfamily="monospace")
    ax2.text(10.8, y, prob, fontsize=9.5, color=c_tok, fontweight=fw,
             fontfamily="monospace", ha="center")
    ax2.text(12.2, y, cost, fontsize=9.5, color=c_tok, fontweight=fw,
             fontfamily="monospace", ha="center")
    ax2.text(13.8, y, meaning, fontsize=9, color=MEDIUM if not is_header
             else C_BOUND, fontweight=fw)

ax2.text(12.1, -1.0,
    "High probability = low cost\nLow probability = high cost",
    fontsize=10, ha="center", color=C_BOUND, fontweight="bold",
    linespacing=1.4)


# ─────────────────────────────────────────────────────────
#  ROW 3: Verifier Bonus Hack
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-2.5, 6.5)
ax3.set_title("  The Verifier Bonus Hack",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

box(ax3, (0.3, 2.0), 15.4, 4.0, C_B, alpha_f=0.06)
ax3.text(8, 5.5, "Adding a Verifier to the Cost", fontsize=13,
         ha="center", fontweight="bold", color=C_B)
ax3.text(8, 4.5,
    "Some problems have a verifier: a function that checks if a complete response is correct.",
    fontsize=10.5, ha="center", color=MEDIUM)
ax3.text(8, 3.6,
    "If verified correct, add a large negative bonus to the cost  (e.g. -100).",
    fontsize=11.5, ha="center", color=C_B, fontweight="bold")
ax3.text(8, 2.7,
    "This makes correct responses overwhelmingly attractive to the search.",
    fontsize=10.5, ha="center", color=MEDIUM)

# Example
box(ax3, (0.3, -2.0), 15.4, 3.5, C_BOUND, alpha_f=0.05)
ax3.text(8, 1.0, "Example:  \"What is 37 x 48?\"    (correct answer: 1776)",
         fontsize=11, ha="center", fontweight="bold", color=C_BOUND)

responses_v = [
    ('"1786"', "cost = 5.2", "wrong",   "no bonus",       "total = 5.2",  C_NEG),
    ('"1776"', "cost = 6.1", "correct", "bonus = -100",   "total = -93.9", C_POS),
    ('"1800"', "cost = 4.8", "wrong",   "no bonus",       "total = 4.8",  C_NEG),
]
for i, (resp, cost, verdict, bonus, total, c) in enumerate(responses_v):
    y = 0.0 - i * 0.6
    ax3.text(1.0, y, resp, fontsize=10, fontfamily="monospace", color=c,
             fontweight="bold")
    ax3.text(4.0, y, cost, fontsize=9.5, fontfamily="monospace", color=MEDIUM)
    ax3.text(6.8, y, verdict, fontsize=9.5, color=c, fontweight="bold")
    ax3.text(9.0, y, bonus, fontsize=9.5, fontfamily="monospace", color=c)
    ax3.text(12.5, y, total, fontsize=10, fontfamily="monospace", color=c,
             fontweight="bold")
    if c == C_POS:
        ax3.text(15.0, y, "WINS", fontsize=10, fontweight="bold", color=C_POS,
                 bbox=dict(facecolor=BG, edgecolor=C_POS, pad=3,
                           boxstyle="round,pad=0.15"))

ax3.text(8, -2.0,
    "This is explicitly a hack — but a useful one for folding correctness into the cost framework.",
    fontsize=10, ha="center", color=C_RULE, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 4: Concrete Example — Best-of-n with an LM
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4])
ax4.axis("off")
ax4.set_xlim(-0.5, 16.5); ax4.set_ylim(-7, 13)
ax4.set_title("  Full Pipeline: Best-of-3 with a Language Model",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

ax4.text(8, 12.2,
    'Prompt: "What is 37 x 48?"     The LM generates token-by-token.',
    fontsize=11, ha="center", color=MEDIUM)
ax4.text(8, 11.5,
    "Each token choice has a probability from the LM, which becomes a cost via -log p.",
    fontsize=10.5, ha="center", color=C_I)

# Three rollouts shown as token-by-token chains
rollouts = [
    {
        "label": "Rollout 1",
        "tokens": ["37", "x", "48", "=", "1786"],
        "probs":  [0.95, 0.92, 0.93, 0.97, 0.15],
        "costs":  [0.05, 0.08, 0.07, 0.03, 1.90],
        "correct": False,
        "total_cost": 2.13,
        "color": C_NEG,
    },
    {
        "label": "Rollout 2",
        "tokens": ["Let", "me", "compute:", "37", "x", "48", "=", "1776"],
        "probs":  [0.40, 0.85, 0.60, 0.90, 0.88, 0.91, 0.95, 0.35],
        "costs":  [0.92, 0.16, 0.51, 0.11, 0.13, 0.09, 0.05, 1.05],
        "correct": True,
        "total_cost": 3.02,
        "color": C_I,
    },
    {
        "label": "Rollout 3",
        "tokens": ["1776"],
        "probs":  [0.08],
        "costs":  [2.53],
        "correct": True,
        "total_cost": 2.53,
        "color": C_POS,
    },
]

rollout_y_start = 10.5
rollout_spacing = 3.2

for r_idx, r in enumerate(rollouts):
    base_y = rollout_y_start - r_idx * rollout_spacing
    is_winner = (r["label"] == "Rollout 3")  # lowest total after bonus

    # Rollout label
    lbl_c = C_POS if is_winner else MEDIUM
    ax4.text(0, base_y + 0.3, r["label"], fontsize=11, fontweight="bold",
             color=r["color"])

    # Token chain
    n_tok = len(r["tokens"])
    tok_spacing = min(1.6, 13.0 / max(n_tok, 1))
    tok_start_x = 1.5

    for i, (tok, prob, cost) in enumerate(zip(r["tokens"], r["probs"],
                                               r["costs"])):
        tx = tok_start_x + i * tok_spacing

        # Token box
        tok_w = max(len(tok) * 0.22, 0.8)
        box(ax4, (tx - tok_w / 2, base_y - 0.35), tok_w, 0.7,
            r["color"], alpha_f=0.12, lw=1.2)
        ax4.text(tx, base_y, tok, fontsize=9, ha="center",
                 fontfamily="monospace", color=r["color"], fontweight="bold")

        # Probability and cost below
        ax4.text(tx, base_y - 0.65, f"p={prob:.2f}", fontsize=7.5,
                 ha="center", color=MEDIUM)
        ax4.text(tx, base_y - 1.0, f"c={cost:.2f}", fontsize=7.5,
                 ha="center", color=C_BOUND, fontweight="bold")

        # Arrow to next
        if i < n_tok - 1:
            ax4.annotate("",
                xy=(tx + tok_spacing - tok_w / 2 - 0.1, base_y),
                xytext=(tx + tok_w / 2 + 0.1, base_y),
                arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=1.0,
                                alpha=0.4))

    # Results on the right
    res_x = 14.0
    # Sum of costs
    ax4.text(res_x, base_y + 0.2,
             f"sum of costs = {r['total_cost']:.2f}",
             fontsize=9.5, fontfamily="monospace", color=C_BOUND)

    if r["correct"]:
        bonus_cost = r["total_cost"] - 100
        ax4.text(res_x, base_y - 0.3,
                 f"verifier: CORRECT",
                 fontsize=9.5, color=C_POS, fontweight="bold")
        ax4.text(res_x, base_y - 0.8,
                 f"+ bonus: {bonus_cost:.2f}",
                 fontsize=9.5, fontfamily="monospace", color=C_POS,
                 fontweight="bold")
    else:
        ax4.text(res_x, base_y - 0.3,
                 f"verifier: WRONG",
                 fontsize=9.5, color=C_NEG, fontweight="bold")
        ax4.text(res_x, base_y - 0.8,
                 f"total: {r['total_cost']:.2f}",
                 fontsize=9.5, fontfamily="monospace", color=C_NEG)

# Winner selection
box(ax4, (1, -5.0), 14, 3.5, C_POS, alpha_f=0.08, lw=2)
ax4.text(8, -2.0, "Best-of-3 Selection", fontsize=13, ha="center",
         fontweight="bold", color=C_POS)

selections = [
    ("Rollout 1:", '"...1786"', "cost = 2.13", "wrong, no bonus", "final = 2.13", C_NEG),
    ("Rollout 2:", '"Let me compute...1776"', "cost = 3.02", "correct, -100 bonus", "final = -96.98", C_I),
    ("Rollout 3:", '"1776"', "cost = 2.53", "correct, -100 bonus", "final = -97.47", C_POS),
]

for i, (lbl, resp, cost, note, final, c) in enumerate(selections):
    y = -2.8 - i * 0.65
    ax4.text(1.5, y, lbl, fontsize=9.5, color=c, fontweight="bold")
    ax4.text(3.5, y, resp, fontsize=9, fontfamily="monospace", color=c)
    ax4.text(9.5, y, note, fontsize=9, color=MEDIUM)
    ax4.text(13.5, y, final, fontsize=9.5, fontfamily="monospace",
             color=c, fontweight="bold")

ax4.text(15.0, -2.8 - 2 * 0.65, "WINNER",
         fontsize=10, fontweight="bold", color=C_POS,
         bbox=dict(facecolor=BG, edgecolor=C_POS, pad=3,
                   boxstyle="round,pad=0.15"))

ax4.text(8, -5.8,
    "Rollout 3 wins: it is correct AND has the lowest total cost (most probable correct answer).",
    fontsize=10, ha="center", color=MEDIUM, fontstyle="italic")
ax4.text(8, -6.4,
    "Note: Rollout 2 is also correct but less probable — the verifier + cost combination selects the best.",
    fontsize=9.5, ha="center", color=SUBTLE)


# ─────────────────────────────────────────────────────────
#  ROW 5: The Modern Synthesis
# ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[5])
ax5.axis("off")
ax5.set_xlim(0, 16); ax5.set_ylim(-3, 7.5)
ax5.set_title("  The Modern Synthesis: Learning + Search",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# Two pillars feeding into the synthesis
box(ax5, (0.5, 3.0), 6.5, 4.0, C_B, alpha_f=0.06)
ax5.text(3.75, 6.5, "Learning", fontsize=13, ha="center",
         fontweight="bold", color=C_B)
ax5.text(3.75, 5.5,
    "Trained model provides:\n"
    "- next-token probabilities\n"
    "- which become costs via -log p\n"
    "- a learned policy for rollouts",
    fontsize=10, ha="center", color=MEDIUM, linespacing=1.4)

box(ax5, (9.0, 3.0), 6.5, 4.0, C_I, alpha_f=0.06)
ax5.text(12.25, 6.5, "Search", fontsize=13, ha="center",
         fontweight="bold", color=C_I)
ax5.text(12.25, 5.5,
    "At inference time:\n"
    "- generate multiple candidates\n"
    "- compare them using cost/verifier\n"
    "- return the best one found",
    fontsize=10, ha="center", color=MEDIUM, linespacing=1.4)

# Arrows converging
ax5.annotate("", xy=(8, 1.8), xytext=(4.5, 2.8),
             arrowprops=dict(arrowstyle="-|>", color=C_B, lw=2.5, alpha=0.6))
ax5.annotate("", xy=(8, 1.8), xytext=(11.5, 2.8),
             arrowprops=dict(arrowstyle="-|>", color=C_I, lw=2.5, alpha=0.6))

# Result
box(ax5, (3.5, -2.5), 9, 4.0, C_BOUND, alpha_f=0.08, lw=2)
ax5.text(8, 1.0, "Together", fontsize=14, ha="center",
         fontweight="bold", color=C_BOUND)
ax5.text(8, 0.0,
    "Learning estimates local preferences (\"how likely is this token?\").",
    fontsize=10.5, ha="center", color=C_B, fontweight="bold")
ax5.text(8, -0.7,
    "Search uses those estimates to find GLOBALLY better solutions.",
    fontsize=10.5, ha="center", color=C_I, fontweight="bold")
ax5.text(8, -1.5,
    "Neither alone is enough.  Combined, they outperform one-shot generation.",
    fontsize=11, ha="center", color=C_BOUND, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 6: Key Takeaway
# ─────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[6])
ax6.axis("off")
ax6.set_xlim(0, 16); ax6.set_ylim(-3, 3.5)

box(ax6, (1.0, -2.5), 14, 5.5, C_RULE, alpha_f=0.08, lw=1.5)
ax6.text(8, 2.5, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("Test-time compute = spending extra computation at inference to search for better outputs.", True),
    ("LM generation maps directly onto search: state = prefix, action = next token, cost = -log p.", False),
    ("Minimizing -log p is the same as maximizing probability — it's a unit conversion, not a trick.", True),
    ("A verifier adds a large bonus to correct answers, steering search toward correctness.", False),
    ("Best-of-n with an LM: generate multiple responses, optionally verify, return the best.", True),
    ("The modern synthesis: learning provides local costs, search finds globally better solutions.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax6.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_5_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "search_and_lms.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
