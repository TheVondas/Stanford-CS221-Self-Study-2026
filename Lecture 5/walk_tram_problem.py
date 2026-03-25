"""
Lecture 5 Visual: The Walk/Tram Travel Problem

Shows:
1. Formal search problem specification (start state, successors, end test, costs)
2. Directed graph of locations 1–10 with walk edges (i→i+1, cost 1)
   and tram edges (i→2i, cost 2)
3. Example optimal path 1→2→4→5→10 traced with cost breakdown = 6
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ── Style (matching lecture series) ──────────────────────
BG      = "#1e1e2e"
TEXT    = "#cdd6f4"
SUBTLE  = "#6c7086"
MEDIUM  = "#a6adc8"
C_POS   = "#a6e3a1"     # green  — walk edges
C_NEG   = "#f38ba8"     # pink/red
C_BOUND = "#f9e2af"     # yellow — goal / highlights
C_RULE  = "#fab387"     # orange — key insights
C_I     = "#89b4fa"     # blue   — tram edges
C_B     = "#cba6f7"     # purple
C_DIM   = "#585b70"     # dim gray
C_NODE  = "#89dceb"     # teal   — nodes

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Helpers ─────────────────────────────────────────────

def draw_node(ax, cx, cy, label, color=C_NODE, radius=0.38, fontsize=13,
              lw=2.0, alpha_fill=0.12):
    """Circle node with label."""
    circle = plt.Circle((cx, cy), radius, facecolor=color, alpha=alpha_fill,
                         edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(circle)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=6)


def draw_walk_edge(ax, x1, y, x2, color=C_POS, lw=2.0, label="1",
                   highlight=False):
    """Straight horizontal arrow for walk (i → i+1, cost 1)."""
    ec = color
    alpha = 1.0 if highlight else 0.6
    line_lw = lw * 1.6 if highlight else lw
    ax.annotate("", xy=(x2 - 0.42, y), xytext=(x1 + 0.42, y),
                arrowprops=dict(arrowstyle="-|>", color=ec, lw=line_lw,
                                alpha=alpha), zorder=3)
    if label:
        mx = (x1 + x2) / 2
        ax.text(mx, y - 0.35, label, fontsize=8, ha="center", va="center",
                color=ec, fontweight="bold", alpha=alpha,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1))


def draw_tram_edge(ax, x1, y, x2, color=C_I, lw=2.0, label="2",
                   highlight=False, arc_height=None):
    """Curved arc arrow for tram (i → 2i, cost 2)."""
    dist = abs(x2 - x1)
    if arc_height is None:
        arc_height = 0.6 + dist * 0.12
    alpha = 1.0 if highlight else 0.5
    line_lw = lw * 1.6 if highlight else lw
    rad = 0.3 + dist * 0.04
    ax.annotate("", xy=(x2 - 0.35, y + 0.30), xytext=(x1 + 0.35, y + 0.30),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=line_lw,
                                alpha=alpha,
                                connectionstyle=f"arc3,rad=-{rad}"),
                zorder=3)
    mx = (x1 + x2) / 2
    label_y = y + 0.55 + dist * 0.12
    if label:
        ax.text(mx, label_y, label, fontsize=8, ha="center", va="center",
                color=color, fontweight="bold", alpha=alpha,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1))


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 26))
fig.suptitle("The Walk / Tram Travel Problem",
             fontsize=18, fontweight="bold", y=0.995, color=C_BOUND)

gs = GridSpec(4, 1, figure=fig,
              height_ratios=[2.8, 4.0, 3.5, 1.8],
              hspace=0.18,
              top=0.965, bottom=0.02, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: Formal Search Problem Specification
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-1, 7)
ax0.set_title("  Formal Search Problem Specification",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# Problem setup box
ax0.add_patch(patches.FancyBboxPatch(
    (0.3, 3.5), 7.0, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))
ax0.text(3.8, 6.0, "Problem Setup", fontsize=12, ha="center",
         fontweight="bold", color=C_B)

setup_lines = [
    "A street with locations numbered  1  through  n",
    "Walk from  i  to  i + 1    →  costs 1",
    "Take tram from  i  to  2i  →  costs 2",
    "Goal: get from  1  to  n  with minimum total cost",
]
for i, txt in enumerate(setup_lines):
    ax0.text(3.8, 5.3 - i * 0.55, txt, fontsize=10, ha="center",
             fontfamily="monospace", color=MEDIUM)

# Formal components box
ax0.add_patch(patches.FancyBboxPatch(
    (8.5, 3.5), 7.2, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_NODE, alpha=0.06, edgecolor=C_NODE, linewidth=1.5))
ax0.text(12.1, 6.0, "Formal Components", fontsize=12, ha="center",
         fontweight="bold", color=C_NODE)

components = [
    ("State:", "current location  i", C_NODE),
    ("Start state:", "1", C_POS),
    ("Successors(i):", "walk → (i+1, cost 1)   tram → (2i, cost 2)", MEDIUM),
    ("End test:", "is  i == n  ?", C_BOUND),
]
for i, (label, value, color) in enumerate(components):
    y = 5.3 - i * 0.55
    ax0.text(9.0, y, label, fontsize=10, fontweight="bold", color=color)
    ax0.text(11.0, y, value, fontsize=10, fontfamily="monospace", color=color)

# Key modeling insight
ax0.add_patch(patches.FancyBboxPatch(
    (0.3, -0.5), 15.4, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.06, edgecolor=C_RULE, linewidth=1.5))
ax0.text(8, 2.5, "Why This Formalization Matters", fontsize=12,
         ha="center", fontweight="bold", color=C_RULE)
ax0.text(8, 1.7,
    "You should not try to solve this in your head. Convert it into a general structure that an algorithm can solve.",
    fontsize=10, ha="center", color=MEDIUM, fontstyle="italic")
ax0.text(8, 0.9,
    "Once the problem is formalized correctly, the algorithm does the rest.",
    fontsize=10.5, ha="center", color=C_RULE, fontweight="bold")
ax0.text(8, 0.2,
    "This is the simplest version: the state only needs one number — where you currently are.",
    fontsize=10, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 1: Full Directed Graph — All Edges for n = 10
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(-0.5, 16); ax1.set_ylim(-2.5, 5.5)
ax1.set_title("  All Possible Actions  (n = 10)",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

n = 10
# Node positions: evenly spaced
node_x = {i: 1.0 + (i - 1) * 1.5 for i in range(1, n + 1)}
node_y = 1.5

# Draw all walk edges (i → i+1, cost 1)
for i in range(1, n):
    draw_walk_edge(ax1, node_x[i], node_y, node_x[i + 1])

# Draw all tram edges (i → 2i, cost 2) where 2i <= n
for i in range(1, n):
    if 2 * i <= n:
        draw_tram_edge(ax1, node_x[i], node_y, node_x[2 * i])

# Draw nodes (on top of edges)
for i in range(1, n + 1):
    if i == 1:
        draw_node(ax1, node_x[i], node_y, i, color=C_POS, lw=3.0,
                  alpha_fill=0.20)
    elif i == n:
        draw_node(ax1, node_x[i], node_y, i, color=C_BOUND, lw=3.0,
                  alpha_fill=0.20)
    else:
        draw_node(ax1, node_x[i], node_y, i)

# Start / Goal labels
ax1.text(node_x[1], node_y - 0.65, "START", fontsize=9, ha="center",
         color=C_POS, fontweight="bold")
ax1.text(node_x[n], node_y - 0.65, "GOAL", fontsize=9, ha="center",
         color=C_BOUND, fontweight="bold")

# Legend
ax1.add_patch(patches.FancyBboxPatch(
    (0.3, -2.3), 15.2, 1.3, boxstyle="round,pad=0.1",
    facecolor=BG, alpha=0.9, edgecolor=C_DIM, linewidth=1))
ax1.text(4.0, -1.35, "→  Walk:  i → i + 1    cost = 1",
         fontsize=11, color=C_POS, fontweight="bold")
ax1.text(11.5, -1.35, "~  Tram:  i → 2i      cost = 2",
         fontsize=11, color=C_I, fontweight="bold")
ax1.text(4.0, -1.95, "(short horizontal arrows)",
         fontsize=9, color=C_POS, alpha=0.6)
ax1.text(11.5, -1.95, "(curved arcs above)",
         fontsize=9, color=C_I, alpha=0.6)


# ─────────────────────────────────────────────────────────
#  ROW 2: Example Optimal Path Highlighted
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(-0.5, 16); ax2.set_ylim(-3.5, 5.5)
ax2.set_title("  Example Solution: Optimal Path  (cost = 6)",
              fontsize=14, fontweight="bold", color=C_BOUND, pad=10, loc="left")

path_y = 2.0

# Draw all edges dimmed first
for i in range(1, n):
    draw_walk_edge(ax2, node_x[i], path_y, node_x[i + 1],
                   color=C_DIM, label="", lw=1.0)
for i in range(1, n):
    if 2 * i <= n:
        draw_tram_edge(ax2, node_x[i], path_y, node_x[2 * i],
                       color=C_DIM, label="", lw=1.0)

# Highlight the optimal path: 1 →walk→ 2 →tram→ 4 →walk→ 5 →tram→ 10
# Walk: 1 → 2
draw_walk_edge(ax2, node_x[1], path_y, node_x[2], color=C_POS,
               highlight=True, label="walk (1)")
# Tram: 2 → 4
draw_tram_edge(ax2, node_x[2], path_y, node_x[4], color=C_I,
               highlight=True, label="tram (2)")
# Walk: 4 → 5
draw_walk_edge(ax2, node_x[4], path_y, node_x[5], color=C_POS,
               highlight=True, label="walk (1)")
# Tram: 5 → 10
draw_tram_edge(ax2, node_x[5], path_y, node_x[10], color=C_I,
               highlight=True, label="tram (2)")

# Draw nodes — highlight the path nodes
path_nodes = {1, 2, 4, 5, 10}
for i in range(1, n + 1):
    if i == 1:
        draw_node(ax2, node_x[i], path_y, i, color=C_POS, lw=3.0,
                  alpha_fill=0.25)
    elif i == n:
        draw_node(ax2, node_x[i], path_y, i, color=C_BOUND, lw=3.0,
                  alpha_fill=0.25)
    elif i in path_nodes:
        draw_node(ax2, node_x[i], path_y, i, color=C_NODE, lw=2.5,
                  alpha_fill=0.20)
    else:
        draw_node(ax2, node_x[i], path_y, i, color=C_DIM, lw=1.2,
                  alpha_fill=0.05)

ax2.text(node_x[1], path_y - 0.65, "START", fontsize=9, ha="center",
         color=C_POS, fontweight="bold")
ax2.text(node_x[n], path_y - 0.65, "GOAL", fontsize=9, ha="center",
         color=C_BOUND, fontweight="bold")

# Cost breakdown box
ax2.add_patch(patches.FancyBboxPatch(
    (0.3, -3.3), 15.2, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_BOUND, alpha=0.06, edgecolor=C_BOUND, linewidth=1.5))

ax2.text(8, -1.3, "Action Sequence and Cost Breakdown", fontsize=12,
         ha="center", fontweight="bold", color=C_BOUND)

# Step-by-step trace
steps = [
    ("1", "walk", "2", "1", C_POS),
    ("2", "tram", "4", "2", C_I),
    ("4", "walk", "5", "1", C_POS),
    ("5", "tram", "10", "2", C_I),
]
trace_x = 1.5
for i, (frm, action, to, cost, color) in enumerate(steps):
    x = trace_x + i * 3.2
    ax2.text(x, -2.0, f"{frm}", fontsize=11, ha="center",
             fontfamily="monospace", color=C_NODE, fontweight="bold")
    ax2.text(x + 0.7, -2.0, f"—{action}→", fontsize=10, ha="center",
             fontfamily="monospace", color=color, fontweight="bold")
    ax2.text(x + 1.6, -2.0, f"{to}", fontsize=11, ha="center",
             fontfamily="monospace", color=C_NODE, fontweight="bold")
    ax2.text(x + 0.7, -2.6, f"cost = {cost}", fontsize=9, ha="center",
             color=color, fontweight="bold")

# Total
ax2.text(14.5, -2.0, "=", fontsize=14, ha="center", color=C_BOUND,
         fontweight="bold")
ax2.text(14.5, -2.7, "Total = 1 + 2 + 1 + 2 = 6", fontsize=10.5,
         ha="center", color=C_BOUND, fontweight="bold",
         fontfamily="monospace")


# ─────────────────────────────────────────────────────────
#  ROW 3: Key Takeaway
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-3, 3.5)

ax3.add_patch(patches.FancyBboxPatch(
    (1.0, -2.5), 14, 5.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax3.text(8, 2.5, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("A solution is a sequence of actions, not just a destination.", True),
    ("Many solutions may exist — the objective is minimum total cost.", False),
    ("The minimum cost is unique, but the action sequence need not be.", False),
    ("Walking is cheap per step but slow.  The tram jumps far but costs more.", True),
    ("The optimal strategy mixes both: walk to set up good tram jumps.", False),
    ("A search algorithm should return both the cost AND the action sequence.", True),
]
for i, (txt, bold) in enumerate(takeaways):
    ax3.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_5_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "walk_tram_problem.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
