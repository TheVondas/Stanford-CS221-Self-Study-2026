"""
Lecture 5 Visual: Best-of-n Search — The Simplest Approximate Method

Shows:
1. Why we need it: exact methods fail when the state space is too large
2. How it works: policy → rollout → repeat n times → keep the best
3. Concrete example: 5 rollouts on the walk/tram problem (n=10) with
   a uniform random policy, showing varying costs and the winner
4. Properties: simple, embarrassingly parallel, consistent in the limit
5. Constraints: required n may be exponential; quality depends on policy
"""

import os
import random
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
fig = plt.figure(figsize=(16, 40))
fig.suptitle("Best-of-n: The Simplest Approximate Search",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(5, 1, figure=fig,
              height_ratios=[3.0, 3.0, 7.5, 4.5, 1.8],
              hspace=0.14,
              top=0.975, bottom=0.01, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: Why We Need Approximate Search
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-1.5, 7)
ax0.set_title("  Why Exact Search Becomes Intractable",
              fontsize=14, fontweight="bold", color=C_NEG, pad=10, loc="left")

# The problem
box(ax0, (0.3, 3.5), 15.4, 3.0, C_NEG, alpha_f=0.06)
ax0.text(8, 6.0, "The Problem", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)
ax0.text(8, 5.2,
    "Both exhaustive search and DP need to touch every reachable state.",
    fontsize=11, ha="center", color=MEDIUM)
ax0.text(8, 4.4,
    "When the state space is enormous, exact methods are impossible.",
    fontsize=11, ha="center", color=C_NEG, fontweight="bold")

examples = [
    ("State = set of visited locations", "2^n possible subsets"),
    ("State = sequence of words generated", "vocabulary^length possibilities"),
]
for i, (ex, why) in enumerate(examples):
    y = 3.8 - i * 0.55
    ax0.text(3.5, y, ex, fontsize=10, color=MEDIUM, fontfamily="monospace",
             ha="center")
    ax0.text(11.5, y, why, fontsize=10, color=C_NEG, fontfamily="monospace",
             ha="center", fontweight="bold")

# The shift
box(ax0, (0.3, -1.0), 15.4, 3.5, C_POS, alpha_f=0.06)
ax0.text(8, 2.0, "The Shift in Goal", fontsize=13, ha="center",
         fontweight="bold", color=C_POS)

shifts = [
    ("Exact search:", "find THE best solution", C_NEG),
    ("Approximate search:", "find a GOOD ENOUGH solution", C_POS),
]
for i, (lbl, txt, c) in enumerate(shifts):
    y = 1.1 - i * 0.7
    ax0.text(4.0, y, lbl, fontsize=11, color=c, fontweight="bold",
             ha="right")
    ax0.text(4.3, y, txt, fontsize=11, color=c, fontfamily="monospace")

ax0.text(8, -0.5,
    "Search only a subset of possibilities.  Hope to find something good.",
    fontsize=10.5, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 1: How Best-of-n Works — The Algorithm
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(0, 16); ax1.set_ylim(-1.5, 7)
ax1.set_title("  How Best-of-n Works",
              fontsize=14, fontweight="bold", color=C_I, pad=10, loc="left")

# Three-step pipeline
step_data = [
    (0.5,  "1. Policy", "A mapping  pi(s) -> action\nor a distribution over\nactions from state s",
     C_B, "In the toy example:\nuniform random over\navailable successors"),
    (5.5,  "2. Rollout", "Start at start state.\nRepeatedly sample an\naction from the policy.\nStop at end state.",
     C_I, "One rollout =\none complete\ncandidate solution"),
    (10.5, "3. Best of n", "Repeat n rollouts.\nCompute each total cost.\nReturn the cheapest one.",
     C_BOUND, "That's it.\nThe whole algorithm."),
]

for x, title, desc, color, note in step_data:
    box(ax1, (x, 1.0), 4.5, 5.5, color, alpha_f=0.08, lw=1.8)
    ax1.text(x + 2.25, 6.0, title, fontsize=12, ha="center",
             fontweight="bold", color=color)
    ax1.text(x + 2.25, 4.3, desc, fontsize=10, ha="center",
             color=MEDIUM, linespacing=1.4)
    ax1.text(x + 2.25, 1.8, note, fontsize=9.5, ha="center",
             color=color, fontstyle="italic", linespacing=1.3)

# Arrows between steps
for x in [5.0, 10.0]:
    ax1.annotate("", xy=(x + 0.5, 3.8), xytext=(x, 3.8),
                 arrowprops=dict(arrowstyle="-|>", color=C_DIM, lw=2.5,
                                 alpha=0.5))

# Bottom note
ax1.text(8, -0.5,
    "The lecture deliberately demystifies this: it sounds sophisticated, "
    "but the algorithm is very simple.",
    fontsize=10, ha="center", color=C_RULE, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 2: Concrete Example — 5 Rollouts on Walk/Tram n=10
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(-0.5, 16.5); ax2.set_ylim(-4.5, 11)
ax2.set_title("  Example: 5 Rollouts with Uniform Random Policy  (n = 10)",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# Pre-defined rollouts to show variety (deterministic for reproducibility)
rollouts = [
    # (path_states, actions, total_cost, color_hint)
    ([1, 2, 4, 5, 10],
     ["walk", "tram", "walk", "tram"],
     6, C_POS),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     ["walk", "walk", "walk", "walk", "walk", "walk", "walk", "walk", "walk"],
     9, C_NEG),
    ([1, 2, 4, 8, 9, 10],
     ["walk", "tram", "tram", "walk", "walk"],
     8, C_NODE),
    ([1, 2, 4, 5, 6, 7, 8, 9, 10],
     ["walk", "tram", "walk", "walk", "walk", "walk", "walk", "walk", "walk"],
     10, C_NEG),
    ([1, 2, 3, 6, 7, 8, 9, 10],
     ["walk", "walk", "tram", "walk", "walk", "walk", "walk", "walk"],
     9, C_NODE),
]

row_h = 1.8
start_y = 9.5

for r_idx, (states, actions, cost, hint) in enumerate(rollouts):
    y = start_y - r_idx * row_h
    is_best = (cost == 6)

    # Background highlight for best
    if is_best:
        ax2.add_patch(patches.FancyBboxPatch(
            (-0.2, y - 0.65), 16.5, 1.5, boxstyle="round,pad=0.08",
            facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=2))

    # Rollout label
    lbl_color = C_POS if is_best else MEDIUM
    ax2.text(-0.1, y + 0.1, f"#{r_idx + 1}", fontsize=11,
             fontweight="bold", color=lbl_color)

    # Draw the path as nodes with action arrows
    n_states = len(states)
    # Calculate spacing to fit all states
    max_x = 12.5
    spacing = min(1.2, max_x / max(n_states - 1, 1))
    base_x = 1.0

    for i, s in enumerate(states):
        nx = base_x + i * spacing
        nc = C_POS if (s == 1) else (C_BOUND if s == 10 else C_NODE)
        if is_best:
            nc = C_POS if s == 1 else (C_BOUND if s == 10 else C_POS)
        a_f = 0.20 if is_best else 0.10
        node(ax2, nx, y, s, color=nc, r=0.22, fs=9, lw=1.8 if is_best else 1.2,
             alpha_f=a_f)

        # Action arrow
        if i < n_states - 1:
            act = actions[i]
            ac = C_POS if act == "walk" else C_I
            alpha = 0.8 if is_best else 0.4
            ax2.annotate("", xy=(nx + spacing - 0.24, y),
                         xytext=(nx + 0.24, y),
                         arrowprops=dict(arrowstyle="-|>", color=ac,
                                         lw=1.5 if is_best else 1.0,
                                         alpha=alpha),
                         zorder=3)
            # Tiny action label
            ax2.text(nx + spacing / 2, y + 0.32,
                     "w" if act == "walk" else "t",
                     fontsize=7, ha="center", color=ac, alpha=alpha,
                     fontweight="bold")

    # Cost label at end
    cost_color = C_POS if is_best else (C_NEG if cost >= 9 else MEDIUM)
    ax2.text(14.2, y, f"cost = {cost}", fontsize=11,
             fontweight="bold", color=cost_color, va="center")

    if is_best:
        ax2.text(15.8, y, "BEST", fontsize=11, fontweight="bold",
                 color=C_POS, va="center",
                 bbox=dict(facecolor=BG, edgecolor=C_POS, pad=4,
                           boxstyle="round,pad=0.2"))

# Summary below rollouts
box(ax2, (0.3, -4.2), 15.9, 3.5, C_BOUND, alpha_f=0.06)
ax2.text(8, -1.2, "Result", fontsize=13, ha="center",
         fontweight="bold", color=C_BOUND)
ax2.text(8, -2.0,
    "After 5 rollouts, best-of-n returns rollout #1 with cost 6",
    fontsize=11, ha="center", color=C_POS, fontweight="bold")
ax2.text(8, -2.7,
    "With only 5 samples and a uniform random policy, we found the optimal solution.",
    fontsize=10.5, ha="center", color=MEDIUM)
ax2.text(8, -3.4,
    "With a harder problem or worse luck, we might need many more samples.",
    fontsize=10.5, ha="center", color=C_NEG, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 3: Properties and Constraints
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-4.5, 8)
ax3.set_title("  Properties and Constraints",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

# ── LEFT: Strengths ──
box(ax3, (0.3, 1.5), 7.2, 6.0, C_POS, alpha_f=0.05, lw=1.5)
ax3.text(3.9, 7.0, "Strengths", fontsize=13, ha="center",
         fontweight="bold", color=C_POS)

strengths = [
    ("Simple", "Trivial to implement.\nNo complex data structures."),
    ("Parallel", "Each rollout is independent.\nNo coordination needed.\n\"Embarrassingly parallel.\""),
    ("Scalable", "Works with any policy —\nincluding a learned model.\nStronger policy = better results."),
    ("Consistent", "As n -> inf, will eventually\nsample the optimal solution\n(if policy covers all actions)."),
]
for i, (title, desc) in enumerate(strengths):
    y = 6.0 - i * 1.35
    ax3.text(1.0, y, title, fontsize=10.5, fontweight="bold", color=C_POS)
    ax3.text(3.0, y, desc, fontsize=9, color=MEDIUM, linespacing=1.3,
             va="top")

# ── RIGHT: Constraints ──
box(ax3, (8.5, 1.5), 7.2, 6.0, C_NEG, alpha_f=0.05, lw=1.5)
ax3.text(12.1, 7.0, "Constraints", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)

constraints = [
    ("No guarantee", "Any finite n may miss\nthe optimal solution."),
    ("Exponential n", "Required n for optimality\ncan be exponentially large\nin problem size."),
    ("Policy-dependent", "Uniform random policy is\nweak — most rollouts waste\ntime on bad paths."),
    ("No pruning", "Each rollout is blind to\nwhat other rollouts found.\nNo learning between samples."),
]
for i, (title, desc) in enumerate(constraints):
    y = 6.0 - i * 1.35
    ax3.text(9.2, y, title, fontsize=10.5, fontweight="bold", color=C_NEG)
    ax3.text(11.2, y, desc, fontsize=9, color=MEDIUM, linespacing=1.3,
             va="top")

# ── Bottom: Modern relevance ──
box(ax3, (0.3, -4.0), 15.4, 5.0, C_RULE, alpha_f=0.06)
ax3.text(8, 0.5, "Why Best-of-n Matters in Modern AI", fontsize=13,
         ha="center", fontweight="bold", color=C_RULE)

modern_lines = [
    ("Weak policy  (uniform random):", "most rollouts are bad, need huge n", C_NEG),
    ("Strong policy  (language model):", "most rollouts are decent, small n suffices", C_POS),
]
for i, (lbl, txt, c) in enumerate(modern_lines):
    y = -0.5 - i * 0.8
    ax3.text(2.0, y, lbl, fontsize=10.5, color=c, fontweight="bold")
    ax3.text(9.5, y, txt, fontsize=10.5, color=c)

ax3.text(8, -2.7,
    "This is exactly test-time compute: sample multiple responses from an LM,",
    fontsize=10.5, ha="center", color=MEDIUM)
ax3.text(8, -3.4,
    "then pick the best.  Simple, parallel, and surprisingly effective.",
    fontsize=10.5, ha="center", color=C_RULE, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 4: Key Takeaway
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-3, 3.5)

box(ax4, (1.0, -2.5), 14, 5.5, C_RULE, alpha_f=0.08, lw=1.5)
ax4.text(8, 2.5, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("Best-of-n: sample n complete solutions, keep the cheapest. That's the whole algorithm.", True),
    ("It is not exact — any finite n may miss the optimum.", False),
    ("But it is trivially simple, embarrassingly parallel, and works with any policy.", True),
    ("Quality depends almost entirely on the policy: strong policy = good results with small n.", False),
    ("This is the foundation of test-time compute in language models.", True),
    ("When exact search is impossible, best-of-n is often the first thing to try.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_5_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "best_of_n.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
