"""
Lecture 5 Visual: Exhaustive Search — Tree, Redundancy, and Cost Tradeoffs

Shows:
1. The futureCost recurrence (base case + recursive case)
2. Full search tree for n=4 showing all 9 state explorations — redundant visits
   highlighted (only 4 real locations, but 9 nodes in the tree)
3. Time vs Memory tradeoff: time = whole tree (exponential), memory = one branch
   on the call stack (linear in depth)
4. The cycle problem and the threshold hack
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
C_POS   = "#a6e3a1"     # green
C_NEG   = "#f38ba8"     # pink/red
C_BOUND = "#f9e2af"     # yellow
C_RULE  = "#fab387"     # orange
C_I     = "#89b4fa"     # blue
C_B     = "#cba6f7"     # purple
C_DIM   = "#585b70"     # dim gray
C_NODE  = "#89dceb"     # teal

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Helpers ─────────────────────────────────────────────

def draw_tree_node(ax, cx, cy, label, color=C_NODE, radius=0.32,
                   fontsize=12, lw=2.0, alpha_fill=0.12):
    """Circle node for tree diagram."""
    circle = plt.Circle((cx, cy), radius, facecolor=color, alpha=alpha_fill,
                         edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(circle)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=6)


def draw_edge(ax, x1, y1, x2, y2, color, label="", label_side="left",
              lw=2.0, alpha=0.7):
    """Arrow between two tree nodes with optional label."""
    ax.annotate("", xy=(x2, y2 + 0.32), xytext=(x1, y1 - 0.32),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                alpha=alpha, shrinkA=2, shrinkB=2),
                zorder=3)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset = -0.45 if label_side == "left" else 0.45
        ax.text(mx + offset, my, label, fontsize=8, ha="center",
                va="center", color=color, fontweight="bold", alpha=alpha,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1))


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 38))
fig.suptitle("Exhaustive Search: How It Works and What It Costs",
             fontsize=18, fontweight="bold", y=0.997, color=C_BOUND)

gs = GridSpec(5, 1, figure=fig,
              height_ratios=[2.5, 5.5, 5.5, 3.5, 1.8],
              hspace=0.15,
              top=0.97, bottom=0.015, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: The futureCost Recurrence
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-1, 7)
ax0.set_title("  The futureCost Recurrence — Foundation of Exhaustive Search",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# Base case box
ax0.add_patch(patches.FancyBboxPatch(
    (0.3, 3.5), 7.0, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.5))
ax0.text(3.8, 6.0, "Base Case", fontsize=12, ha="center",
         fontweight="bold", color=C_POS)
ax0.text(3.8, 5.1, "If  s  is an end state:", fontsize=10.5,
         ha="center", color=MEDIUM)
ax0.text(3.8, 4.3, "futureCost(s) = 0", fontsize=12, ha="center",
         fontfamily="monospace", color=C_POS, fontweight="bold")

# Recursive case box
ax0.add_patch(patches.FancyBboxPatch(
    (8.5, 3.5), 7.2, 3.0, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))
ax0.text(12.1, 6.0, "Recursive Case", fontsize=12, ha="center",
         fontweight="bold", color=C_I)
ax0.text(12.1, 5.1, "If  s  is not an end state:", fontsize=10.5,
         ha="center", color=MEDIUM)
ax0.text(12.1, 4.3, "futureCost(s) =", fontsize=11, ha="center",
         fontfamily="monospace", color=C_I, fontweight="bold")
ax0.text(12.1, 3.8, "min  [ c + futureCost(s') ]", fontsize=11,
         ha="center", fontfamily="monospace", color=C_I, fontweight="bold")

# Interpretation box
ax0.add_patch(patches.FancyBboxPatch(
    (0.3, -0.5), 15.4, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))
ax0.text(8, 2.5, "Interpretation", fontsize=12, ha="center",
         fontweight="bold", color=C_B)

interp = [
    "1. Choose a first action from the current state",
    "2. Pay its immediate cost  c",
    "3. Optimally solve the rest from the next state  s'",
    "4. Among all possible first actions, keep the minimum",
]
for i, txt in enumerate(interp):
    ax0.text(8, 1.8 - i * 0.6, txt, fontsize=10.5, ha="center",
             fontfamily="monospace", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 1: Full Search Tree for n = 4
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(-1, 17); ax1.set_ylim(-5, 8.5)
ax1.set_title("  Full Search Tree  (n = 4)  —  9 explorations, only 4 real locations",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# Tree structure for n=4:
#                    1                    depth 0
#                  /   \
#           w(1) /     \ t(2)
#               2       2                depth 1  (BOTH go to state 2!)
#             /   \   /   \
#        w(1)/  t(2) w(1) \t(2)
#           3     4    3    4             depth 2
#           |    END   |   END
#        w(1)       w(1)
#           4          4                  depth 3
#          END        END

# Positions
# Level 0
L0 = [(8, 7.5)]
# Level 1
L1 = [(4, 5.0), (12, 5.0)]
# Level 2
L2 = [(2, 2.5), (5.5, 2.5), (9.5, 2.5), (13, 2.5)]
# Level 3
L3 = [(2, 0.0), (9.5, 0.0)]

# --- Draw edges first (under nodes) ---

# Level 0 → Level 1
draw_edge(ax1, L0[0][0], L0[0][1], L1[0][0], L1[0][1],
          C_POS, "walk (1)", "left")
draw_edge(ax1, L0[0][0], L0[0][1], L1[1][0], L1[1][1],
          C_I, "tram (2)", "right")

# Level 1 left (state 2) → Level 2
draw_edge(ax1, L1[0][0], L1[0][1], L2[0][0], L2[0][1],
          C_POS, "walk (1)", "left")
draw_edge(ax1, L1[0][0], L1[0][1], L2[1][0], L2[1][1],
          C_I, "tram (2)", "right")

# Level 1 right (state 2) → Level 2
draw_edge(ax1, L1[1][0], L1[1][1], L2[2][0], L2[2][1],
          C_POS, "walk (1)", "left")
draw_edge(ax1, L1[1][0], L1[1][1], L2[3][0], L2[3][1],
          C_I, "tram (2)", "right")

# Level 2 state 3 (left) → Level 3
draw_edge(ax1, L2[0][0], L2[0][1], L3[0][0], L3[0][1],
          C_POS, "walk (1)", "left")

# Level 2 state 3 (right) → Level 3
draw_edge(ax1, L2[2][0], L2[2][1], L3[1][0], L3[1][1],
          C_POS, "walk (1)", "left")

# --- Draw nodes ---

# Level 0: start state
draw_tree_node(ax1, *L0[0], "1", color=C_POS, lw=3.0, alpha_fill=0.22)
ax1.text(L0[0][0] + 0.6, L0[0][1], "START", fontsize=9, color=C_POS,
         fontweight="bold", va="center")

# Level 1: both are state 2
for pos in L1:
    draw_tree_node(ax1, *pos, "2", color=C_NODE, lw=2.0, alpha_fill=0.15)

# Redundancy highlight between the two state-2 nodes
ax1.annotate("", xy=(L1[1][0] - 0.5, L1[1][1]),
             xytext=(L1[0][0] + 0.5, L1[0][1]),
             arrowprops=dict(arrowstyle="<->", color=C_NEG, lw=2.0,
                             linestyle="--", alpha=0.7))
ax1.text(8, 5.0, "same state\nexplored twice!",
         fontsize=9, ha="center", color=C_NEG, fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=3, alpha=0.9,
                   boxstyle="round,pad=0.2"))

# Level 2: states 3, 4, 3, 4
draw_tree_node(ax1, *L2[0], "3", color=C_NODE, lw=2.0, alpha_fill=0.15)
draw_tree_node(ax1, *L2[1], "4", color=C_BOUND, lw=3.0, alpha_fill=0.22)
draw_tree_node(ax1, *L2[2], "3", color=C_NODE, lw=2.0, alpha_fill=0.15)
draw_tree_node(ax1, *L2[3], "4", color=C_BOUND, lw=3.0, alpha_fill=0.22)

# END labels for state-4 at level 2
ax1.text(L2[1][0], L2[1][1] - 0.55, "END", fontsize=8, ha="center",
         color=C_BOUND, fontweight="bold")
ax1.text(L2[3][0], L2[3][1] - 0.55, "END", fontsize=8, ha="center",
         color=C_BOUND, fontweight="bold")

# Redundancy for the two state-3 nodes
ax1.annotate("", xy=(L2[2][0] - 0.5, L2[2][1]),
             xytext=(L2[0][0] + 0.5, L2[0][1]),
             arrowprops=dict(arrowstyle="<->", color=C_NEG, lw=1.5,
                             linestyle="--", alpha=0.6))
ax1.text((L2[0][0] + L2[2][0]) / 2, 2.5, "same!",
         fontsize=8, ha="center", color=C_NEG, fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=2, alpha=0.9,
                   boxstyle="round,pad=0.15"))

# Redundancy for the two state-4 at level 2
ax1.annotate("", xy=(L2[3][0] - 0.5, L2[3][1] + 0.5),
             xytext=(L2[1][0] + 0.5, L2[1][1] + 0.5),
             arrowprops=dict(arrowstyle="<->", color=C_NEG, lw=1.5,
                             linestyle="--", alpha=0.6))
ax1.text((L2[1][0] + L2[3][0]) / 2, 3.3, "same!",
         fontsize=8, ha="center", color=C_NEG, fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=2, alpha=0.9,
                   boxstyle="round,pad=0.15"))

# Level 3: state 4 (from walking out of state 3)
draw_tree_node(ax1, *L3[0], "4", color=C_BOUND, lw=3.0, alpha_fill=0.22)
draw_tree_node(ax1, *L3[1], "4", color=C_BOUND, lw=3.0, alpha_fill=0.22)
ax1.text(L3[0][0], L3[0][1] - 0.55, "END", fontsize=8, ha="center",
         color=C_BOUND, fontweight="bold")
ax1.text(L3[1][0], L3[1][1] - 0.55, "END", fontsize=8, ha="center",
         color=C_BOUND, fontweight="bold")

# Path cost annotations on the right
ax1.add_patch(patches.FancyBboxPatch(
    (14.5, -0.8), 2.2, 8.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.06, edgecolor=C_RULE, linewidth=1.2))
ax1.text(15.6, 7.2, "Path Costs", fontsize=10, ha="center",
         fontweight="bold", color=C_RULE)

paths = [
    ("1-2-3-4", "1+1+1=3", C_POS),
    ("1-2-4",   "1+2=3",   C_I),
    ("1-2-3-4", "2+1+1=4", C_POS),
    ("1-2-4",   "2+2=4",   C_I),
]
for i, (path, cost, color) in enumerate(paths):
    y = 6.2 - i * 1.8
    ax1.text(15.6, y, path, fontsize=8.5, ha="center",
             fontfamily="monospace", color=MEDIUM)
    ax1.text(15.6, y - 0.5, f"= {cost}", fontsize=9, ha="center",
             fontfamily="monospace", color=color, fontweight="bold")

ax1.text(15.6, -0.2, "Best = 3", fontsize=10.5, ha="center",
         color=C_BOUND, fontweight="bold")

# Count annotation
ax1.add_patch(patches.FancyBboxPatch(
    (0.5, -4.5), 15.5, 3.2, boxstyle="round,pad=0.12",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1.5))
ax1.text(8, -1.8, "The Redundancy Problem", fontsize=12, ha="center",
         fontweight="bold", color=C_NEG)
ax1.text(8, -2.6,
    "4 real locations  but  9 nodes explored in the tree",
    fontsize=11, ha="center", fontfamily="monospace", color=C_NEG,
    fontweight="bold")
ax1.text(8, -3.3,
    "State 2 is solved twice.  State 3 is solved twice.  State 4 appears four times.",
    fontsize=10, ha="center", color=MEDIUM)
ax1.text(8, -4.0,
    "Every time a state has multiple successors, the search branches — and these branches can revisit the same states.",
    fontsize=9.5, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 2: Time vs Memory Tradeoff
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(-0.5, 16.5); ax2.set_ylim(-4.5, 8)
ax2.set_title("  Time vs Memory: The Asymmetric Costs of Exhaustive Search",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

# --- LEFT: Time = entire tree ---
ax2.add_patch(patches.FancyBboxPatch(
    (0.0, -0.5), 7.5, 8.0, boxstyle="round,pad=0.15",
    facecolor=C_NEG, alpha=0.04, edgecolor=C_NEG, linewidth=1.5))
ax2.text(3.75, 7.0, "TIME  =  Whole Tree", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)

# Mini tree illustrating exponential branching
# Level 0
mini_x0, mini_y0 = 3.75, 5.8
draw_tree_node(ax2, mini_x0, mini_y0, "", C_NEG, radius=0.18,
               fontsize=1, alpha_fill=0.25)

# Level 1 (2 nodes)
for dx in [-1.5, 1.5]:
    draw_tree_node(ax2, mini_x0 + dx, mini_y0 - 1.5, "", C_NEG,
                   radius=0.18, fontsize=1, alpha_fill=0.25)
    ax2.plot([mini_x0, mini_x0 + dx],
             [mini_y0 - 0.2, mini_y0 - 1.3],
             color=C_NEG, lw=1.5, alpha=0.5, zorder=2)

# Level 2 (4 nodes)
l2_positions = [-2.5, -0.8, 0.8, 2.5]
for j, dx in enumerate(l2_positions):
    draw_tree_node(ax2, mini_x0 + dx, mini_y0 - 3.0, "", C_NEG,
                   radius=0.18, fontsize=1, alpha_fill=0.25)
    parent_dx = -1.5 if j < 2 else 1.5
    ax2.plot([mini_x0 + parent_dx, mini_x0 + dx],
             [mini_y0 - 1.7, mini_y0 - 2.8],
             color=C_NEG, lw=1.5, alpha=0.5, zorder=2)

# Level 3 (8 nodes, show as dots)
l3_positions = [-3.0, -2.2, -1.3, -0.5, 0.5, 1.3, 2.2, 3.0]
for j, dx in enumerate(l3_positions):
    draw_tree_node(ax2, mini_x0 + dx, mini_y0 - 4.5, "", C_NEG,
                   radius=0.14, fontsize=1, alpha_fill=0.25)
    parent_dx = l2_positions[j // 2]
    ax2.plot([mini_x0 + parent_dx, mini_x0 + dx],
             [mini_y0 - 3.2, mini_y0 - 4.35],
             color=C_NEG, lw=1.0, alpha=0.35, zorder=2)

# Highlight ALL nodes
ax2.text(3.75, 0.65,
    "Every node = one recursive call", fontsize=10,
    ha="center", color=C_NEG, fontweight="bold")
ax2.text(3.75, 0.05,
    "Branching factor b,  depth d", fontsize=10,
    ha="center", fontfamily="monospace", color=MEDIUM)

# --- RIGHT: Memory = single branch (call stack) ---
ax2.add_patch(patches.FancyBboxPatch(
    (8.5, -0.5), 7.5, 8.0, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.04, edgecolor=C_POS, linewidth=1.5))
ax2.text(12.25, 7.0, "MEMORY  =  One Branch  (Call Stack)", fontsize=13,
         ha="center", fontweight="bold", color=C_POS)

# Single highlighted path through a tree
stack_x = 12.25
for level in range(4):
    y = mini_y0 - level * 1.5
    # Draw the current node
    c = C_POS
    draw_tree_node(ax2, stack_x, y, "", c, radius=0.22, fontsize=1,
                   alpha_fill=0.30, lw=2.5)
    # Ghost siblings (dimmed)
    if level < 3:
        ghost_dx = 1.2 if level % 2 == 0 else -1.2
        draw_tree_node(ax2, stack_x + ghost_dx, y - 0.8, "", C_DIM,
                       radius=0.12, fontsize=1, alpha_fill=0.05, lw=0.8)
        ax2.plot([stack_x, stack_x + ghost_dx],
                 [y - 0.25, y - 0.68],
                 color=C_DIM, lw=0.8, alpha=0.3, zorder=2)
    # Edge to next level
    if level < 3:
        ax2.plot([stack_x, stack_x],
                 [y - 0.25, y - 1.25],
                 color=C_POS, lw=2.5, alpha=0.7, zorder=2)

# Call stack label
ax2.annotate("", xy=(14.5, mini_y0), xytext=(14.5, mini_y0 - 4.5),
             arrowprops=dict(arrowstyle="<->", color=C_POS, lw=2))
ax2.text(15.3, mini_y0 - 2.25, "depth\n= d",
         fontsize=10, ha="center", color=C_POS, fontweight="bold")

ax2.text(12.25, 0.65,
    "Only the current path is stored", fontsize=10,
    ha="center", color=C_POS, fontweight="bold")
ax2.text(12.25, 0.05,
    "Old branches are freed on return", fontsize=10,
    ha="center", color=MEDIUM)

# --- Bottom comparison table ---
ax2.add_patch(patches.FancyBboxPatch(
    (0.5, -4.2), 15.5, 3.3, boxstyle="round,pad=0.12",
    facecolor=C_BOUND, alpha=0.06, edgecolor=C_BOUND, linewidth=1.5))

ax2.text(8, -1.35, "Complexity Summary", fontsize=13, ha="center",
         fontweight="bold", color=C_BOUND)

# Header
ax2.text(3.5, -2.05, "Resource", fontsize=11, ha="center",
         fontweight="bold", color=TEXT)
ax2.text(8.5, -2.05, "Growth", fontsize=11, ha="center",
         fontweight="bold", color=TEXT)
ax2.text(13.0, -2.05, "Why", fontsize=11, ha="center",
         fontweight="bold", color=TEXT)
ax2.plot([1, 15.5], [-2.35, -2.35], color=SUBTLE, lw=0.8, alpha=0.4)

# Time row
ax2.text(3.5, -2.75, "Time", fontsize=11, ha="center",
         fontweight="bold", color=C_NEG)
ax2.text(8.5, -2.75, "O( b^d )  exponential", fontsize=11, ha="center",
         fontfamily="monospace", color=C_NEG, fontweight="bold")
ax2.text(13.0, -2.75, "every branch explored", fontsize=10,
         ha="center", color=MEDIUM)

ax2.plot([1, 15.5], [-3.15, -3.15], color=C_DIM, lw=0.5, alpha=0.2)

# Memory row
ax2.text(3.5, -3.55, "Memory", fontsize=11, ha="center",
         fontweight="bold", color=C_POS)
ax2.text(8.5, -3.55, "O( d )    linear", fontsize=11, ha="center",
         fontfamily="monospace", color=C_POS, fontweight="bold")
ax2.text(13.0, -3.55, "only current path on stack", fontsize=10,
         ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 3: Cycles and the Threshold Hack
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-3.5, 7)
ax3.set_title("  The Cycle Problem — When Exhaustive Search Never Terminates",
              fontsize=14, fontweight="bold", color=C_NEG, pad=10, loc="left")

# --- Left: cycle diagram A → B → C → A ---
ax3.add_patch(patches.FancyBboxPatch(
    (0.3, 1.5), 6.5, 5.0, boxstyle="round,pad=0.15",
    facecolor=C_NEG, alpha=0.04, edgecolor=C_NEG, linewidth=1.2))
ax3.text(3.55, 6.0, "Cycle:  A -> B -> C -> A -> ...", fontsize=11,
         ha="center", fontweight="bold", color=C_NEG)

# Triangle of nodes
import math
cx, cy = 3.55, 3.5
r = 1.3
angles = [90, 210, 330]  # A at top, B at bottom-left, C at bottom-right
labels = ["A", "B", "C"]
node_positions = []
for angle, label in zip(angles, labels):
    rad = math.radians(angle)
    nx = cx + r * math.cos(rad)
    ny = cy + r * math.sin(rad)
    node_positions.append((nx, ny))
    draw_tree_node(ax3, nx, ny, label, C_NEG, radius=0.35, fontsize=13,
                   alpha_fill=0.15, lw=2.5)

# Curved arrows between nodes
for i in range(3):
    x1, y1 = node_positions[i]
    x2, y2 = node_positions[(i + 1) % 3]
    ax3.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=2.5,
                                 connectionstyle="arc3,rad=-0.25",
                                 shrinkA=18, shrinkB=18, alpha=0.7),
                 zorder=3)

# Infinite loop indicator
ax3.text(3.55, 1.85, "recursion never terminates",
         fontsize=10, ha="center", color=C_NEG, fontweight="bold",
         fontstyle="italic")

# --- Right: the threshold hack ---
ax3.add_patch(patches.FancyBboxPatch(
    (7.8, 1.5), 7.9, 5.0, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.04, edgecolor=C_POS, linewidth=1.2))
ax3.text(11.75, 6.0, "The Threshold Hack", fontsize=12, ha="center",
         fontweight="bold", color=C_POS)

hack_lines = [
    ("1.", "Add step count to the state:", C_POS, True),
    ("",   "state = (location, steps_taken)", C_POS, False),
    ("2.", "Set a maximum step limit T", C_BOUND, True),
    ("3.", "If steps > T, return infinite cost", C_NEG, True),
    ("",   "This turns infinite horizon -> finite horizon", MEDIUM, False),
]
for i, (num, txt, color, bold) in enumerate(hack_lines):
    y = 5.1 - i * 0.7
    if num:
        ax3.text(8.3, y, num, fontsize=10.5, color=color, fontweight="bold")
    ax3.text(9.0, y, txt, fontsize=10.5, color=color,
             fontfamily="monospace" if not bold else "sans-serif",
             fontweight="bold" if bold else "normal")

# Caveat box
ax3.add_patch(patches.FancyBboxPatch(
    (0.3, -3.2), 15.4, 4.2, boxstyle="round,pad=0.15",
    facecolor=C_BOUND, alpha=0.06, edgecolor=C_BOUND, linewidth=1.5))
ax3.text(8, 0.5, "Caveats", fontsize=12, ha="center",
         fontweight="bold", color=C_BOUND)

caveats = [
    ("Choosing T:", "no universal answer — use problem structure", C_BOUND),
    ("Positive costs:", "number of states often gives a reasonable upper bound", C_POS),
    ("Negative cycles:", "optimal solution may loop forever toward -inf cost", C_NEG),
    ("", "this is a degenerate case requiring special handling", MEDIUM),
]
for i, (label, txt, color) in enumerate(caveats):
    y = -0.3 - i * 0.7
    if label:
        ax3.text(2.0, y, label, fontsize=10, color=color, fontweight="bold")
        ax3.text(5.5, y, txt, fontsize=10, color=MEDIUM)
    else:
        ax3.text(5.5, y, txt, fontsize=10, color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 4: Key Takeaway
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-3, 3.5)

ax4.add_patch(patches.FancyBboxPatch(
    (1.0, -2.5), 14, 5.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax4.text(8, 2.5, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("Exhaustive search is exact — it guarantees the true minimum-cost solution.", True),
    ("But it pays an exponential time cost: every path through the tree is explored.", False),
    ("Memory is cheap: only the current call-stack path is stored  (linear in depth).", True),
    ("Time is expensive: the full tree can have exponentially many nodes.", False),
    ("This is why exhaustive search can be time-impossible before it becomes memory-impossible.", True),
    ("Redundant state visits are the core inefficiency — and the motivation for dynamic programming.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_5_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "exhaustive_search.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
