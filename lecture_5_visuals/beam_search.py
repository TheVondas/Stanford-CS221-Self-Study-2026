"""
Lecture 5 Visual: Beam Search — Structured Approximate Search

Shows:
1. How beam search works step-by-step: expand → score → prune → repeat
2. Concrete walk/tram example with beam_width=2 showing kept/discarded candidates
3. The beam width spectrum: greedy (1) → beam (k) → exhaustive (inf)
4. How beam search improves on best-of-n and compares to exhaustive search
5. Why beam search can fail: discarding a path that would have won later
6. Three-way comparison table: exhaustive vs best-of-n vs beam search
"""

import os
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


def edge(ax, x1, y1, x2, y2, color, lw=1.8, alpha=0.6, shrA=12, shrB=12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                alpha=alpha, shrinkA=shrA, shrinkB=shrB),
                zorder=3)


# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 52))
fig.suptitle("Beam Search: Structured Approximate Search",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(6, 1, figure=fig,
              height_ratios=[2.8, 8.5, 3.5, 5.0, 4.5, 1.8],
              hspace=0.12,
              top=0.975, bottom=0.008, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: The Core Idea
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-1.5, 6.5)
ax0.set_title("  The Core Idea — Prune As You Go",
              fontsize=14, fontweight="bold", color=C_I, pad=10, loc="left")

box(ax0, (0.3, 2.5), 15.4, 3.5, C_I, alpha_f=0.08)
ax0.text(8, 5.5, "Beam Search", fontsize=14, ha="center",
         fontweight="bold", color=C_I)
ax0.text(8, 4.5,
    "Instead of sampling complete trajectories blindly (best-of-n)",
    fontsize=11, ha="center", color=MEDIUM)
ax0.text(8, 3.8,
    "or exploring everything (exhaustive search),",
    fontsize=11, ha="center", color=MEDIUM)
ax0.text(8, 3.0,
    "keep only the best  beam_width  partial solutions at each step and discard the rest.",
    fontsize=12, ha="center", color=C_I, fontweight="bold")

# Metaphor
box(ax0, (3.0, -1.0), 10, 3.0, C_RULE, alpha_f=0.05)
ax0.text(8, 1.5, "Metaphor: driving at night", fontsize=11,
         ha="center", fontweight="bold", color=C_RULE)
ax0.text(8, 0.7,
    "You cannot see the whole road network — only a limited beam of headlights ahead.",
    fontsize=10.5, ha="center", color=MEDIUM, fontstyle="italic")
ax0.text(8, -0.1,
    "At each moment you choose among the visible options. The rest is darkness.",
    fontsize=10.5, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 1: Step-by-Step Example  (walk/tram n=10, beam_width=2)
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(-0.5, 16.5); ax1.set_ylim(-6, 13.5)
ax1.set_title("  Step-by-Step Example:  Walk/Tram  n = 10,  beam_width = 2",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# We trace beam search step by step.
# beam_width = 2
#
# Step 0: beam = [(path=[1], cost=0)]
# Expand: walk→2 (cost 1), tram→2 (cost 2)
# Keep best 2: [1→2 cost=1], [1→2 cost=2]  — but same state!
# Actually both go to state 2, so candidates are:
#   (1-walk->2, cost=1)  and  (1-tram->2, cost=2)
# Keep both (beam_width=2)
#
# Step 1: expand both:
# From [1,2] cost=1: walk→3 (cost 2), tram→4 (cost 3)
# From [1,2] cost=2: walk→3 (cost 3), tram→4 (cost 4)
# All 4 candidates: [1,2,3] c=2, [1,2,4] c=3, [1,2,3] c=3, [1,2,4] c=4
# Keep best 2: [1,2,3] c=2  and  [1,2,4] c=3
#
# Step 2: expand both:
# From [1,2,3] c=2: walk→4 (c=3), tram→6 (c=4)
# From [1,2,4] c=3: walk→5 (c=4), tram→8 (c=5)
# All 4: [1,2,3,4] c=3, [1,2,3,6] c=4, [1,2,4,5] c=4, [1,2,4,8] c=5
# Keep best 2: [1,2,3,4] c=3  and  [1,2,3,6] c=4  (or [1,2,4,5] c=4 — tie)
# Let's keep: c=3 and c=4 (first tiebreak)
#
# Step 3: from [1,2,3,4] c=3: walk→5 (c=4), tram→8 (c=5)
#          from [1,2,3,6] c=4: walk→7 (c=5), tram→12 (OOB — only walk→7)
# Candidates: [...,5] c=4, [...,8] c=5, [...,7] c=5
# Keep best 2: [...,5] c=4  and  [...,8] c=5  (or [...,7] c=5 — tie)
#
# Step 4: from [...,5] c=4: walk→6 (c=5), tram→10 (c=6) — 10 is END!
#          from [...,8] c=5: walk→9 (c=6), tram→16 (OOB)
# Candidates: [...,6] c=5, [...,10] c=6 END, [...,9] c=6
# Keep best 2: [...,6] c=5, [...,10] c=6 END
# But [...,10] is complete! Keep it aside.
#
# Eventually best completed = cost 6.  Correct!

# I'll show a simplified version focusing on the expand-score-prune cycle

step_y_start = 12.5
step_h = 3.0

steps = [
    {
        "label": "Step 0  —  Start",
        "beam": [("[1]", 0, C_POS)],
        "expanded": None,
        "note": "Begin with the start state",
    },
    {
        "label": "Step 1  —  Expand & Prune",
        "beam": None,
        "expanded": [
            ("[1] -walk-> [1,2]", 1, C_POS, True),
            ("[1] -tram-> [1,2]", 2, C_I, True),
        ],
        "note": "Both go to state 2 but via different actions.  Keep best 2.",
    },
    {
        "label": "Step 2  —  Expand & Prune",
        "beam": None,
        "expanded": [
            ("[1,2] -walk-> [1,2,3]", 2, C_POS, True),
            ("[1,2] -tram-> [1,2,4]", 3, C_I, True),
            ("[1,2] -walk-> [1,2,3]", 3, C_DIM, False),
            ("[1,2] -tram-> [1,2,4]", 4, C_DIM, False),
        ],
        "note": "4 candidates from expanding 2 beam entries. Keep cheapest 2.",
    },
    {
        "label": "Step 3  —  Expand & Prune",
        "beam": None,
        "expanded": [
            ("[...,3] -walk-> [...,4]", 3, C_POS, True),
            ("[...,3] -tram-> [...,6]", 4, C_I, True),
            ("[...,4] -walk-> [...,5]", 4, C_DIM, False),
            ("[...,4] -tram-> [...,8]", 5, C_DIM, False),
        ],
        "note": "Pruned candidates had higher cost — discarded.",
    },
    {
        "label": "Step 4+  —  Continue to Goal",
        "beam": None,
        "expanded": [
            ("[...,4] -> ... -> [1,2,3,4,5] -tram-> [...,10]", 6, C_BOUND, True),
        ],
        "note": "Eventually reaches state 10 (END) with total cost 6.  Optimal!",
    },
]

y = step_y_start
for s_idx, step in enumerate(steps):
    label = step["label"]
    note = step["note"]

    # Step label
    ax1.text(0, y + 0.3, label, fontsize=12, fontweight="bold", color=C_I)

    if step["beam"] is not None:
        # Just show the beam entries
        for i, (path, cost, color) in enumerate(step["beam"]):
            bx = 1.5 + i * 6
            box(ax1, (bx, y - 0.8), 5.0, 0.9, color, alpha_f=0.10, lw=1.5)
            ax1.text(bx + 2.5, y - 0.35, f"{path}   cost = {cost}",
                     fontsize=10, ha="center", fontfamily="monospace",
                     color=color, fontweight="bold")

    if step["expanded"] is not None:
        for i, (path, cost, color, kept) in enumerate(step["expanded"]):
            bx = 0.5
            by = y - 0.5 - i * 0.65

            if kept:
                marker = "KEEP"
                mc = C_POS
            else:
                marker = "PRUNE"
                mc = C_NEG

            ax1.text(bx, by, path, fontsize=9, fontfamily="monospace",
                     color=color if kept else C_DIM)
            ax1.text(12.0, by, f"cost = {cost}", fontsize=9.5,
                     fontfamily="monospace", color=color if kept else C_DIM,
                     fontweight="bold")
            ax1.text(14.5, by, marker, fontsize=9, fontweight="bold",
                     color=mc,
                     bbox=dict(facecolor=BG, edgecolor=mc, pad=2,
                               alpha=0.9 if kept else 0.5,
                               boxstyle="round,pad=0.15"))

    # Note
    n_items = 1
    if step["expanded"]:
        n_items = len(step["expanded"])
    elif step["beam"]:
        n_items = 1
    note_y = y - 0.7 - max(n_items, 1) * 0.6
    ax1.text(8, note_y, note, fontsize=9.5, ha="center",
             color=MEDIUM, fontstyle="italic")

    # Separator
    sep_y = note_y - 0.4
    if s_idx < len(steps) - 1:
        ax1.plot([0.3, 16], [sep_y, sep_y], color=C_DIM, lw=0.5,
                 alpha=0.3)

    y = sep_y - 0.4

# Result box
box(ax1, (2, -5.5), 12, 2.2, C_BOUND, alpha_f=0.08)
ax1.text(8, -3.8, "Result: beam search finds cost 6 — the optimal solution!",
         fontsize=12, ha="center", fontweight="bold", color=C_BOUND)
ax1.text(8, -4.6,
    "It explored far fewer candidates than exhaustive search, but stayed focused on the best partial paths.",
    fontsize=10, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 2: Beam Width Spectrum
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(0, 16); ax2.set_ylim(-2.5, 7)
ax2.set_title("  The Beam Width Spectrum",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

# Spectrum bar
bar_y = 4.5
bar_h = 1.0
# Gradient bar from left (narrow) to right (wide)
n_seg = 50
for i in range(n_seg):
    x = 1.5 + i * (13 / n_seg)
    frac = i / n_seg
    # Interpolate color from C_NEG to C_POS
    r_neg, g_neg, b_neg = int(C_NEG[1:3], 16), int(C_NEG[3:5], 16), int(C_NEG[5:7], 16)
    r_pos, g_pos, b_pos = int(C_POS[1:3], 16), int(C_POS[3:5], 16), int(C_POS[5:7], 16)
    r = int(r_neg + (r_pos - r_neg) * frac) / 255
    g = int(g_neg + (g_pos - g_neg) * frac) / 255
    b = int(b_neg + (b_pos - b_neg) * frac) / 255
    ax2.add_patch(patches.Rectangle(
        (x, bar_y), 13 / n_seg + 0.02, bar_h,
        facecolor=(r, g, b), alpha=0.25, edgecolor="none"))

# Labels along the spectrum
spectrum = [
    (1.8, "beam = 1", "GREEDY SEARCH", C_NEG,
     "Only 1 candidate.\nPick locally cheapest\naction every time.\nVery fast, very myopic."),
    (8.0, "beam = k", "BEAM SEARCH", C_I,
     "k candidates tracked.\nBalances exploration\nand efficiency.\nThe practical sweet spot."),
    (13.5, "beam = inf", "EXHAUSTIVE SEARCH", C_POS,
     "All candidates kept.\nGuarantees optimum\nbut exponential cost.\nUsually infeasible."),
]

for x, lbl, title, color, desc in spectrum:
    # Marker on bar
    ax2.plot([x, x], [bar_y - 0.1, bar_y + bar_h + 0.1],
             color=color, lw=3, zorder=5)
    ax2.text(x, bar_y + bar_h + 0.35, lbl, fontsize=10, ha="center",
             fontweight="bold", color=color, fontfamily="monospace")
    ax2.text(x, bar_y + bar_h + 0.85, title, fontsize=10.5, ha="center",
             fontweight="bold", color=color)
    # Description below
    ax2.text(x, bar_y - 0.6, desc, fontsize=9, ha="center", color=MEDIUM,
             linespacing=1.3, va="top")

# Arrows along the bar
ax2.annotate("", xy=(14.2, bar_y + 0.5), xytext=(2.0, bar_y + 0.5),
             arrowprops=dict(arrowstyle="-|>", color=SUBTLE, lw=1.5,
                             alpha=0.4))
ax2.text(8, bar_y + 0.5, "more exploration, more cost",
         fontsize=8.5, ha="center", color=SUBTLE,
         bbox=dict(facecolor=BG, edgecolor="none", pad=2))


# ─────────────────────────────────────────────────────────
#  ROW 3: How Beam Search Improves on Best-of-n +
#          Why Beam Search Can Fail
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(-0.5, 16.5); ax3.set_ylim(-5.5, 8.5)
ax3.set_title("  How Beam Search Improves on Best-of-n  —  and How It Can Fail",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# ── LEFT: Improvement over best-of-n ──
box(ax3, (0, 1.0), 7.8, 7.0, C_POS, alpha_f=0.04, lw=1.2)
ax3.text(3.9, 7.5, "Improvement Over Best-of-n", fontsize=12,
         ha="center", fontweight="bold", color=C_POS)

improvements = [
    ("Structured, not blind",
     "Best-of-n samples complete paths\nindependently — each rollout ignores\nwhat others found.\n\nBeam search shares information:\nat each step, it compares ALL partial\npaths and keeps the most promising."),
    ("Efficient exploration",
     "Best-of-n may waste most rollouts\non bad paths if the policy is weak.\n\nBeam search prunes bad prefixes early\nbefore wasting effort completing them."),
    ("Deterministic",
     "Given the same cost function,\nbeam search returns the same result.\nNo variance from random sampling."),
]

y = 6.5
for title, desc in improvements:
    ax3.text(0.5, y, title, fontsize=10, fontweight="bold", color=C_POS)
    ax3.text(0.5, y - 0.4, desc, fontsize=8.5, color=MEDIUM,
             linespacing=1.3, va="top")
    y -= 2.2

# ── RIGHT: Why it can fail ──
box(ax3, (8.5, 1.0), 7.8, 7.0, C_NEG, alpha_f=0.04, lw=1.2)
ax3.text(12.4, 7.5, "Why Beam Search Can Fail", fontsize=12,
         ha="center", fontweight="bold", color=C_NEG)

# Failure diagram: two paths, one looks worse now but wins later
diag_y = 5.8
ax3.text(12.4, diag_y, "The Pruning Trap", fontsize=10.5, ha="center",
         fontweight="bold", color=C_NEG)

# Path A: looks good now, bad later
ax3.text(9.0, diag_y - 0.8, "Path A:", fontsize=9, color=C_POS,
         fontweight="bold")
pa_nodes = [(10.2, diag_y - 0.8), (11.5, diag_y - 0.8),
            (12.8, diag_y - 0.8), (14.1, diag_y - 0.8)]
pa_costs = ["c=2", "c=4", "c=9", "c=15"]
for i, ((nx, ny), c) in enumerate(zip(pa_nodes, pa_costs)):
    nc = C_POS if i < 2 else C_NEG
    node(ax3, nx, ny, c, color=nc, r=0.32, fs=8, lw=1.5,
         alpha_f=0.12 if i < 2 else 0.08)
    if i < len(pa_nodes) - 1:
        edge(ax3, nx + 0.34, ny, pa_nodes[i + 1][0] - 0.34, ny,
             nc, lw=1.2, alpha=0.4, shrA=2, shrB=2)

# Path B: looks bad now, wins later
ax3.text(9.0, diag_y - 2.0, "Path B:", fontsize=9, color=C_NEG,
         fontweight="bold")
pb_nodes = [(10.2, diag_y - 2.0), (11.5, diag_y - 2.0),
            (12.8, diag_y - 2.0), (14.1, diag_y - 2.0)]
pb_costs = ["c=3", "c=5", "c=6", "c=7"]
for i, ((nx, ny), c) in enumerate(zip(pb_nodes, pb_costs)):
    nc = C_NEG if i < 2 else C_POS
    node(ax3, nx, ny, c, color=nc, r=0.32, fs=8, lw=1.5,
         alpha_f=0.08 if i < 2 else 0.15)
    if i < len(pb_nodes) - 1:
        edge(ax3, nx + 0.34, ny, pb_nodes[i + 1][0] - 0.34, ny,
             nc, lw=1.2, alpha=0.4, shrA=2, shrB=2)

# Pruning annotation
ax3.annotate("", xy=(14.5, diag_y - 2.0), xytext=(14.5, diag_y - 1.3),
             arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=2,
                             alpha=0.7))
ax3.text(15.5, diag_y - 1.65, "actual\nwinner!", fontsize=8,
         ha="center", color=C_POS, fontweight="bold")

ax3.plot([9.8, 15.0], [diag_y - 1.4, diag_y - 1.4],
         color=C_DIM, lw=0.8, ls="--", alpha=0.4)
ax3.text(12.4, diag_y - 1.4, "beam prunes here",
         fontsize=8, ha="center", color=C_NEG,
         bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=2,
                   boxstyle="round,pad=0.15"))

# Explanation
ax3.text(12.4, diag_y - 3.2,
    "Path B has higher cost early on,\nso beam search discards it.",
    fontsize=9, ha="center", color=C_NEG, linespacing=1.3)
ax3.text(12.4, diag_y - 4.2,
    "But Path B would have won in the end.\nBeam search missed the optimal solution.",
    fontsize=9, ha="center", color=C_NEG, fontweight="bold",
    linespacing=1.3)

# Core assumption
box(ax3, (0.3, -5.2), 15.9, 5.7, C_BOUND, alpha_f=0.06)
ax3.text(8, 0.0, "The Core Assumption Behind Beam Search", fontsize=12,
         ha="center", fontweight="bold", color=C_BOUND)
ax3.text(8, -0.8,
    "Low cost so far  is correlated with  low cost at the end.",
    fontsize=12, ha="center", color=C_BOUND, fontweight="bold",
    fontfamily="monospace")
ax3.text(8, -1.7,
    "This hope is often reasonable — but it is NOT guaranteed.",
    fontsize=11, ha="center", color=MEDIUM)
ax3.text(8, -2.6,
    "Beam search is heuristic, not exact.  It can return a suboptimal solution.",
    fontsize=11, ha="center", color=C_NEG, fontweight="bold")

# Pruning optimization
ax3.text(8, -3.6,
    "Pruning bonus:  if a completed path has cost 6, and a partial path already costs 8,",
    fontsize=10, ha="center", color=MEDIUM)
ax3.text(8, -4.3,
    "we can safely discard that partial path — but ONLY if all future costs are nonneg.",
    fontsize=10, ha="center", color=C_RULE, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 4: Three-Way Comparison Table
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-5, 7.5)
ax4.set_title("  Comparison: Exhaustive  vs  Best-of-n  vs  Beam Search",
              fontsize=14, fontweight="bold", color=C_BOUND, pad=10, loc="left")

box(ax4, (0.3, -4.5), 15.4, 11.5, C_BOUND, alpha_f=0.03, lw=1.5)

# Headers
hx = [1.8, 5.5, 9.2, 13.0]
hy = 6.5
headers = ["", "Exhaustive", "Best-of-n", "Beam Search"]
hcolors = [TEXT, C_NEG, C_B, C_I]
for x, h, c in zip(hx, headers, hcolors):
    ax4.text(x, hy, h, fontsize=11.5, fontweight="bold", color=c, ha="center")
ax4.plot([0.5, 15.5], [hy - 0.35, hy - 0.35], color=SUBTLE, lw=1, alpha=0.5)

rows = [
    ("Type",
     "Exact", "Approximate\n(stochastic)", "Approximate\n(deterministic)"),
    ("Explores",
     "All paths", "n random\ncomplete paths", "Best k partial\npaths per step"),
    ("Time",
     "O(b^d)\nexponential", "O(n x d)\nlinear in n", "O(k x b x d)\nlinear in k"),
    ("Memory",
     "O(d)\ncall stack", "O(d)\none rollout", "O(k)\nthe beam"),
    ("Guarantee",
     "Optimal\n(always)", "Optimal only\nas n -> inf", "No guarantee\n(can miss)"),
    ("Parallelism",
     "Hard to\nparallelize", "Embarrassingly\nparallel", "Step-level\nparallelism"),
    ("Needs policy?",
     "No", "Yes", "No\n(uses cost)"),
]

for r_idx, (label, exh, bon, beam) in enumerate(rows):
    y = hy - 1.1 - r_idx * 1.35
    ax4.text(hx[0], y, label, fontsize=10, fontweight="bold", color=C_BOUND,
             ha="center", va="top")
    for j, (txt, c) in enumerate(zip([exh, bon, beam],
                                      [C_NEG, C_B, C_I])):
        ax4.text(hx[j + 1], y, txt, fontsize=9, color=c, ha="center",
                 va="top", linespacing=1.3)
    if r_idx < len(rows) - 1:
        ax4.plot([0.5, 15.5], [y - 0.75, y - 0.75], color=C_DIM, lw=0.4,
                 alpha=0.2)


# ─────────────────────────────────────────────────────────
#  ROW 5: Key Takeaway
# ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[5])
ax5.axis("off")
ax5.set_xlim(0, 16); ax5.set_ylim(-3, 3.5)

box(ax5, (1.0, -2.5), 14, 5.5, C_RULE, alpha_f=0.08, lw=1.5)
ax5.text(8, 2.5, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("Beam search keeps the best k partial solutions at each depth and prunes the rest.", True),
    ("It improves on best-of-n by sharing information across candidates and pruning early.", False),
    ("beam_width = 1 is greedy.  beam_width = inf is exhaustive.  In between is practical.", True),
    ("It can fail: a path that looks worse now might have been best later (pruning trap).", False),
    ("The core assumption: low cost so far predicts low cost at the end. Often true, not always.", True),
    ("Beam search is deterministic, structured, and widely used in practice (NLP, planning).", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax5.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_5_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "beam_search.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
