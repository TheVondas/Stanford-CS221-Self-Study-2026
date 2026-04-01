"""
Lecture 6 Visual: Relaxation Example 2 — Fewer States via Free Tram

Shows:
1. Original limited-travel problem: state = (location, tickets), 2D state space
2. Relaxed problem: tram is free, state = location only, 1D state space
3. The projection: (loc, tickets) → loc  — richer state maps to simpler state
4. Why fewer states makes the relaxed problem cheaper to solve
5. Accounting: UCS cost vs A* cost including relaxed preprocessing
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


def node(ax, cx, cy, label, color=C_NODE, r=0.28, fs=9, lw=1.5,
         alpha_f=0.12):
    c = plt.Circle((cx, cy), r, facecolor=color, alpha=alpha_f,
                    edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(c)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color, zorder=6)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 44))
fig.suptitle("Relaxation: Free Tram → Fewer States",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(5, 1, figure=fig,
              height_ratios=[4.0, 7.0, 5.5, 5.0, 1.8],
              hspace=0.12,
              top=0.975, bottom=0.01, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: The Problem Setup
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-2, 7.5)
ax0.set_title("  The Limited Travel Problem",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

box(ax0, (0.5, 3.5), 15, 3.5, C_NODE, alpha_f=0.04, lw=1.2)
ax0.text(8, 6.5, "Travel from location 1 to location 10", fontsize=13,
         ha="center", fontweight="bold", color=C_NODE)

ax0.text(3.5, 5.5, "Walk:", fontsize=12, color=C_POS, fontweight="bold")
ax0.text(6.5, 5.5, "loc → loc + 1       cost = 1       (always available)",
         fontsize=11, color=C_POS, fontfamily="monospace")

ax0.text(3.5, 4.5, "Tram:", fontsize=12, color=C_I, fontweight="bold")
ax0.text(6.5, 4.5, "loc → 2 × loc       cost = 2       (requires a ticket)",
         fontsize=11, color=C_I, fontfamily="monospace")

# Number line showing locations
line_y = 2.0
locs = range(1, 11)
lx = {loc: 1.2 + (loc - 1) * 1.4 for loc in locs}

ax0.plot([lx[1] - 0.5, lx[10] + 0.5], [line_y, line_y],
         color=C_DIM, lw=1.5, alpha=0.3)
for loc in locs:
    c = C_POS if loc == 1 else (C_BOUND if loc == 10 else C_DIM)
    r = 0.3 if loc in (1, 10) else 0.22
    node(ax0, lx[loc], line_y, str(loc), color=c, r=r, fs=10,
         lw=2.0 if loc in (1, 10) else 1.0,
         alpha_f=0.15 if loc in (1, 10) else 0.06)

ax0.text(lx[1], line_y - 0.6, "start", fontsize=8, ha="center",
         color=C_POS, fontweight="bold")
ax0.text(lx[10], line_y - 0.6, "end", fontsize=8, ha="center",
         color=C_BOUND, fontweight="bold")

# Walk arrow example
ax0.annotate("", xy=(lx[3], line_y + 0.45),
             xytext=(lx[2], line_y + 0.45),
             arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=1.5,
                             alpha=0.5))
ax0.text((lx[2] + lx[3]) / 2, line_y + 0.75, "walk", fontsize=7,
         ha="center", color=C_POS)

# Tram arrow example
ax0.annotate("", xy=(lx[6], line_y - 0.5),
             xytext=(lx[3], line_y - 0.5),
             arrowprops=dict(arrowstyle="-|>", color=C_I, lw=1.5,
                             alpha=0.5, connectionstyle="arc3,rad=-0.15"))
ax0.text((lx[3] + lx[6]) / 2, line_y - 1.3, "tram (3→6)", fontsize=7,
         ha="center", color=C_I)


# ─────────────────────────────────────────────────────────
#  ROW 1: Original vs Relaxed State Space
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(-0.5, 16.5); ax1.set_ylim(-5.5, 10.5)
ax1.set_title("  Original State Space vs Relaxed State Space",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

# ── LEFT: Original 2D state space ──
box(ax1, (-0.3, -1.0), 7.8, 11.0, C_NEG, alpha_f=0.03, lw=1.2)
ax1.text(3.75, 9.5, "Original Problem", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)
ax1.text(3.75, 8.8, "state = (location, tickets)", fontsize=11,
         ha="center", fontfamily="monospace", color=C_NEG)
ax1.text(3.75, 8.1, "Tram requires a ticket", fontsize=10,
         ha="center", color=MEDIUM)

# Draw 2D grid: x = location (1-10), y = tickets (0-3)
# Only show locations 1-7 to keep it readable
show_locs = range(1, 8)
tickets = range(0, 4)
cell_w = 0.85
cell_h = 0.9
grid_ox = 0.3
grid_oy = 0.2

# Axis labels
ax1.text(grid_ox - 0.3, grid_oy + len(tickets) * cell_h + 0.3,
         "tickets", fontsize=9, color=MEDIUM, fontweight="bold",
         rotation=90, va="center")
ax1.text(grid_ox + len(show_locs) * cell_w / 2,
         grid_oy - 0.55, "location", fontsize=9,
         ha="center", color=MEDIUM, fontweight="bold")

for t in tickets:
    ax1.text(grid_ox - 0.2,
             grid_oy + t * cell_h + cell_h / 2, str(t),
             fontsize=9, ha="center", va="center", color=SUBTLE)

for loc in show_locs:
    ax1.text(grid_ox + (loc - 1) * cell_w + cell_w / 2,
             grid_oy - 0.2, str(loc),
             fontsize=9, ha="center", va="center", color=SUBTLE)

# Draw cells
for loc in show_locs:
    for t in tickets:
        x = grid_ox + (loc - 1) * cell_w
        y = grid_oy + t * cell_h

        # Highlight start (1,3) and reachable end states
        if loc == 1 and t == 3:
            fc, fa = C_POS, 0.2
        else:
            fc, fa = C_NODE, 0.06

        rect = patches.FancyBboxPatch(
            (x + 0.02, y + 0.02), cell_w - 0.04, cell_h - 0.04,
            boxstyle="round,pad=0.03",
            facecolor=fc, alpha=fa, edgecolor=SUBTLE,
            linewidth=0.4, zorder=2)
        ax1.add_patch(rect)

        # Label start
        if loc == 1 and t == 3:
            ax1.text(x + cell_w / 2, y + cell_h / 2, "S",
                     fontsize=9, ha="center", va="center",
                     fontweight="bold", color=C_POS, zorder=5)

# Dots for continuation
ax1.text(grid_ox + len(show_locs) * cell_w + 0.2,
         grid_oy + 1.5 * cell_h, "...", fontsize=14,
         color=SUBTLE, ha="center", va="center")

# State count
ax1.text(3.75, -0.5, "~40 states  (10 locations × 4 ticket values)",
         fontsize=10.5, ha="center", color=C_NEG, fontweight="bold")

# ── RIGHT: Relaxed 1D state space ──
box(ax1, (8.5, -1.0), 8.0, 11.0, C_POS, alpha_f=0.03, lw=1.2)
ax1.text(12.5, 9.5, "Relaxed Problem", fontsize=13, ha="center",
         fontweight="bold", color=C_POS)
ax1.text(12.5, 8.8, "state = location", fontsize=11,
         ha="center", fontfamily="monospace", color=C_POS)
ax1.text(12.5, 8.1, "Tram is free — no tickets needed!", fontsize=10,
         ha="center", color=C_POS)

# Draw 1D: just location nodes
relax_y = 5.0
relax_locs = range(1, 11)
rlx = {loc: 9.0 + (loc - 1) * 0.72 for loc in relax_locs}

ax1.plot([rlx[1] - 0.4, rlx[10] + 0.4], [relax_y, relax_y],
         color=C_DIM, lw=1.5, alpha=0.3)

for loc in relax_locs:
    c = C_POS if loc == 1 else (C_BOUND if loc == 10 else C_NODE)
    node(ax1, rlx[loc], relax_y, str(loc), color=c, r=0.25, fs=9,
         lw=1.8 if loc in (1, 10) else 1.2,
         alpha_f=0.18 if loc in (1, 10) else 0.10)

ax1.text(rlx[1], relax_y - 0.55, "start", fontsize=7, ha="center",
         color=C_POS, fontweight="bold")
ax1.text(rlx[10], relax_y - 0.55, "end", fontsize=7, ha="center",
         color=C_BOUND, fontweight="bold")

# Walk and tram arrows (a few examples)
# Walk 1→2
ax1.annotate("", xy=(rlx[2], relax_y + 0.35),
             xytext=(rlx[1], relax_y + 0.35),
             arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=1.2,
                             alpha=0.5))
# Walk 2→3
ax1.annotate("", xy=(rlx[3], relax_y + 0.35),
             xytext=(rlx[2], relax_y + 0.35),
             arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=1.2,
                             alpha=0.5))
# Tram 2→4
ax1.annotate("", xy=(rlx[4], relax_y - 0.5),
             xytext=(rlx[2], relax_y - 0.5),
             arrowprops=dict(arrowstyle="-|>", color=C_I, lw=1.5,
                             alpha=0.5,
                             connectionstyle="arc3,rad=-0.15"))
ax1.text((rlx[2] + rlx[4]) / 2, relax_y - 1.15, "tram", fontsize=7,
         ha="center", color=C_I)
# Tram 3→6
ax1.annotate("", xy=(rlx[6], relax_y - 0.5),
             xytext=(rlx[3], relax_y - 0.5),
             arrowprops=dict(arrowstyle="-|>", color=C_I, lw=1.5,
                             alpha=0.5,
                             connectionstyle="arc3,rad=-0.2"))
# Tram 5→10
ax1.annotate("", xy=(rlx[10], relax_y - 0.5),
             xytext=(rlx[5], relax_y - 0.5),
             arrowprops=dict(arrowstyle="-|>", color=C_I, lw=1.5,
                             alpha=0.5,
                             connectionstyle="arc3,rad=-0.25"))
ax1.text((rlx[5] + rlx[10]) / 2, relax_y - 1.8, "tram (free!)",
         fontsize=7, ha="center", color=C_I)

# State count
ax1.text(12.5, 2.5, "Only 10 states  (just location)",
         fontsize=10.5, ha="center", color=C_POS, fontweight="bold")

# Reduction arrow
ax1.text(12.5, 1.5, "4× fewer states than the original!",
         fontsize=11, ha="center", color=C_BOUND, fontweight="bold")

# Bottom: the key insight
box(ax1, (8.8, -0.7), 7.4, 1.5, C_RULE, alpha_f=0.08, lw=1.5)
ax1.text(12.5, 0.35, "Fewer states = cheaper to solve", fontsize=11,
         ha="center", color=C_RULE, fontweight="bold")
ax1.text(12.5, -0.3, "DP or UCS runs faster on the relaxed problem",
         fontsize=10, ha="center", color=C_RULE)


# ─────────────────────────────────────────────────────────
#  ROW 2: The Projection and Heuristic Definition
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(0, 16); ax2.set_ylim(-4.5, 8.5)
ax2.set_title("  Building the Heuristic: Project State Down",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# Show the projection visually
box(ax2, (0.5, 3.5), 15, 4.5, C_B, alpha_f=0.04, lw=1.2)
ax2.text(8, 7.5, "The Projection Trick", fontsize=13, ha="center",
         fontweight="bold", color=C_B)

# Original state box
box(ax2, (1.5, 4.5), 4.5, 2.2, C_NEG, alpha_f=0.10, lw=1.5)
ax2.text(3.75, 6.2, "Original State", fontsize=11, ha="center",
         fontweight="bold", color=C_NEG)
ax2.text(3.75, 5.3, "(loc, tickets)", fontsize=12, ha="center",
         fontfamily="monospace", color=C_NEG, fontweight="bold")
ax2.text(3.75, 4.7, "e.g. (4, 3)", fontsize=10, ha="center",
         fontfamily="monospace", color=MEDIUM)

# Arrow
ax2.annotate("", xy=(8.8, 5.5), xytext=(6.3, 5.5),
             arrowprops=dict(arrowstyle="-|>", color=C_RULE, lw=3,
                             alpha=0.7))
ax2.text(7.55, 6.1, "project:", fontsize=10, ha="center", color=C_RULE,
         fontweight="bold")
ax2.text(7.55, 5.5, "drop tickets", fontsize=10, ha="center", color=C_RULE)

# Relaxed state box
box(ax2, (9.0, 4.5), 4.5, 2.2, C_POS, alpha_f=0.10, lw=1.5)
ax2.text(11.25, 6.2, "Relaxed State", fontsize=11, ha="center",
         fontweight="bold", color=C_POS)
ax2.text(11.25, 5.3, "loc", fontsize=12, ha="center",
         fontfamily="monospace", color=C_POS, fontweight="bold")
ax2.text(11.25, 4.7, "e.g. 4", fontsize=10, ha="center",
         fontfamily="monospace", color=MEDIUM)

# Heuristic definition
box(ax2, (0.5, 0.5), 15, 2.5, C_RULE, alpha_f=0.08, lw=1.5)
ax2.text(8, 2.5, "Heuristic Definition", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)
ax2.text(8, 1.6,
    "h(loc, tickets) = FutureCost_relaxed(loc)",
    fontsize=13, ha="center", fontfamily="monospace",
    color=C_RULE, fontweight="bold")
ax2.text(8, 0.8,
    "Ignore tickets, use the future cost from the easier problem",
    fontsize=11, ha="center", color=MEDIUM)

# Example heuristic values
box(ax2, (0.5, -4.0), 15, 4.0, C_NODE, alpha_f=0.03, lw=1.0)
ax2.text(8, -0.5, "Example: How to compute h for the relaxed problem (n = 10)",
         fontsize=11.5, ha="center", fontweight="bold", color=C_NODE)

ax2.text(8, -1.3,
    "Solve the relaxed problem once using DP  →  get FutureCost_relaxed for every location",
    fontsize=10.5, ha="center", color=MEDIUM)

# Show a few values
example_vals = [
    ("h(4, 3) = h(4, 2) = h(4, 1) = h(4, 0)",
     "= FutureCost_relaxed(4)", C_B),
    ("",
     "All ticket values map to the same heuristic!", C_BOUND),
]
for i, (left, right, c) in enumerate(example_vals):
    y = -2.2 - i * 0.7
    if left:
        ax2.text(1.5, y, left, fontsize=10, fontfamily="monospace",
                 color=c)
    ax2.text(8, y, right, fontsize=10, color=c, ha="center",
             fontweight="bold" if not left else "normal")


# ─────────────────────────────────────────────────────────
#  ROW 3: Accounting — UCS vs A* Cost Comparison
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-4, 8)
ax3.set_title("  Honest Accounting: UCS vs A* (n = 10, 3 tickets)",
              fontsize=14, fontweight="bold", color=C_BOUND, pad=10, loc="left")

# UCS box
box(ax3, (0.5, 3.5), 7.0, 4.0, C_I, alpha_f=0.06, lw=1.5)
ax3.text(4.0, 7.0, "UCS  (no heuristic)", fontsize=13, ha="center",
         fontweight="bold", color=C_I)
ax3.text(4.0, 6.0, "Explored:", fontsize=11, ha="center", color=MEDIUM)
ax3.text(4.0, 5.2, "23 states", fontsize=16, ha="center",
         fontfamily="monospace", color=C_I, fontweight="bold")
ax3.text(4.0, 4.2, "in the original problem", fontsize=10,
         ha="center", color=MEDIUM)

# A* box
box(ax3, (8.5, 3.5), 7.0, 4.0, C_POS, alpha_f=0.06, lw=1.5)
ax3.text(12.0, 7.0, "A*  (with relaxed heuristic)", fontsize=13,
         ha="center", fontweight="bold", color=C_POS)
ax3.text(12.0, 6.0, "Explored:", fontsize=11, ha="center", color=MEDIUM)
ax3.text(12.0, 5.2, "8 states", fontsize=16, ha="center",
         fontfamily="monospace", color=C_POS, fontweight="bold")
ax3.text(12.0, 4.2, "in the original problem", fontsize=10,
         ha="center", color=MEDIUM)

# But wait — preprocessing!
box(ax3, (0.5, -0.5), 15, 3.5, C_RULE, alpha_f=0.06, lw=1.5)
ax3.text(8, 2.5, "But wait — A* also had to solve the relaxed problem first!",
         fontsize=12, ha="center", fontweight="bold", color=C_RULE)

ax3.text(3.0, 1.5, "Relaxed preprocessing:", fontsize=11, color=MEDIUM)
ax3.text(9.5, 1.5, "10 states explored", fontsize=12,
         fontfamily="monospace", color=C_RULE, fontweight="bold")

ax3.text(3.0, 0.5, "A* total:", fontsize=11, color=MEDIUM,
         fontweight="bold")
ax3.text(9.5, 0.5, "8 + 10 = 18 states", fontsize=12,
         fontfamily="monospace", color=C_POS, fontweight="bold")

# Comparison
box(ax3, (0.5, -3.5), 15, 2.5, C_BOUND, alpha_f=0.08, lw=1.5)
ax3.text(8, -1.5, "Fair Comparison", fontsize=13, ha="center",
         fontweight="bold", color=C_BOUND)

ax3.text(4.0, -2.5, "UCS:  23", fontsize=14, ha="center",
         fontfamily="monospace", color=C_I, fontweight="bold")
ax3.text(8.0, -2.5, "vs", fontsize=12, ha="center", color=MEDIUM)
ax3.text(12.0, -2.5, "A*:  18   (8 + 10)", fontsize=14, ha="center",
         fontfamily="monospace", color=C_POS, fontweight="bold")

ax3.text(8, -3.2, "A* wins, even after honest accounting.",
         fontsize=11, ha="center", color=C_BOUND, fontweight="bold")


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
    ("Relaxation: make the tram free → drop tickets from the state.", True),
    ("This shrinks the state space from ~40 states to just 10.", False),
    ("The heuristic projects (loc, tickets) → loc and uses the relaxed future cost.", True),
    ("It underestimates because the relaxed problem has strictly more options.", False),
    ("Always account for preprocessing cost when comparing UCS vs A*.", True),
    ("This pattern generalizes: any constraint you drop can give a valid heuristic.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/Lecture 6/relaxation_free_tram.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
