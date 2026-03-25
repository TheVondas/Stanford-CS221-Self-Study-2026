"""
Lecture 6 Visual: Why UCS Is Still Inefficient

Shows UCS exploring a grid like a wave — expanding uniformly in all
directions from the start, including areas far from the goal.
Contrasts the large number of explored states with the small optimal path.
"""

import os
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

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


# ── Grid definition ──────────────────────────────────────
# 11x15 grid with a wall barrier in the middle.
# S on the left, E on the right.  UCS has to go around the wall,
# but it doesn't KNOW that — so it explores everywhere.

ROWS, COLS = 11, 15

grid = [
    "...............",  # 0
    "...............",  # 1
    "...............",  # 2
    ".......#.......",  # 3
    ".......#.......",  # 4
    "S......#......E",  # 5
    ".......#.......",  # 6
    ".......#.......",  # 7
    "...............",  # 8
    "...............",  # 9
    "...............",  # 10
]

START = (5, 0)
END   = (5, 14)


def is_valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and grid[r][c] != "#"


# ── Run UCS, recording exploration order ─────────────────
def run_ucs():
    frontier = []
    heapq.heappush(frontier, (0, START))
    costs = {START: 0}
    parents = {START: None}
    explored_order = {}
    order = 0

    while frontier:
        cost, state = heapq.heappop(frontier)
        if state in explored_order:
            continue
        explored_order[state] = order
        order += 1

        if state == END:
            break

        r, c = state
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc):
                new_cost = cost + 1
                if (nr, nc) not in costs or new_cost < costs[(nr, nc)]:
                    costs[(nr, nc)] = new_cost
                    parents[(nr, nc)] = state
                    heapq.heappush(frontier, (new_cost, (nr, nc)))

    # Reconstruct optimal path
    path = []
    s = END
    while s is not None:
        path.append(s)
        s = parents.get(s)
    path.reverse()

    return explored_order, path


explored_order, optimal_path = run_ucs()
path_set = set(optimal_path)
max_order = max(explored_order.values())


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 20))
fig.suptitle("Why UCS Is Still Inefficient",
             fontsize=18, fontweight="bold", y=0.995, color=C_BOUND)

gs = GridSpec(3, 1, figure=fig,
              height_ratios=[7.5, 2.5, 1.5],
              hspace=0.12,
              top=0.965, bottom=0.02, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: The grid with UCS exploration
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")

cell = 0.95
total_w = COLS * cell
total_h = ROWS * cell
ox = (16 - total_w) / 2
oy = 0.5

ax0.set_xlim(0, 16)
ax0.set_ylim(-1.0, total_h + oy + 2.5)

ax0.set_title("  UCS Explores Like a Wave — Uniformly in All Directions",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# Subtitle
ax0.text(8, total_h + oy + 1.5,
    "Each cell is colored by when UCS explores it  (early = blue,  late = purple)",
    fontsize=11, ha="center", color=MEDIUM)

# Build a colormap: blue → teal → purple for exploration order
cmap_colors = [
    (0.0,  C_I),
    (0.35, C_NODE),
    (0.65, C_B),
    (1.0,  C_NEG),
]

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

cmap_data = [(pos, hex_to_rgb(c)) for pos, c in cmap_colors]
cmap = LinearSegmentedColormap.from_list("ucs_wave",
    [(pos, rgb) for pos, rgb in cmap_data])

# Column labels
for c in range(COLS):
    ax0.text(ox + c * cell + cell / 2, total_h + oy + 0.3, str(c),
             fontsize=8, ha="center", va="center", color=SUBTLE)

# Draw grid
for r in range(ROWS):
    # Row label
    ax0.text(ox - 0.3, total_h + oy - r * cell - cell / 2, str(r),
             fontsize=8, ha="center", va="center", color=SUBTLE)

    for c in range(COLS):
        x = ox + c * cell
        y = total_h + oy - (r + 1) * cell
        ch = grid[r][c]

        if ch == "#":
            # Wall
            fc = "#45475a"
            fa = 0.7
            ec = SUBTLE
        elif (r, c) in explored_order:
            # Explored cell — color by order
            t = explored_order[(r, c)] / max_order
            rgba = cmap(t)
            fc = rgba[:3]
            fa = 0.35
            ec = SUBTLE
        else:
            # Unexplored
            fc = TEXT
            fa = 0.03
            ec = C_DIM

        rect = patches.FancyBboxPatch(
            (x + 0.03, y + 0.03), cell - 0.06, cell - 0.06,
            boxstyle="round,pad=0.04",
            facecolor=fc, alpha=fa, edgecolor=ec,
            linewidth=0.5, zorder=2)
        ax0.add_patch(rect)

        # Wall label
        if ch == "#":
            ax0.text(x + cell / 2, y + cell / 2, "#", fontsize=11,
                     ha="center", va="center", fontweight="bold",
                     color=C_NEG, alpha=0.7, zorder=5)

# Draw optimal path arrows on top
for i in range(len(optimal_path) - 1):
    r1, c1 = optimal_path[i]
    r2, c2 = optimal_path[i + 1]
    x1 = ox + c1 * cell + cell / 2
    y1 = total_h + oy - r1 * cell - cell / 2
    x2 = ox + c2 * cell + cell / 2
    y2 = total_h + oy - r2 * cell - cell / 2
    ax0.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=2.2,
                                 alpha=0.8, shrinkA=5, shrinkB=5),
                 zorder=6)

# S and E labels
sr, sc = START
sx = ox + sc * cell + cell / 2
sy = total_h + oy - sr * cell - cell / 2
ax0.text(sx, sy, "S", fontsize=15, ha="center", va="center",
         fontweight="bold", color=C_POS, zorder=7)

er, ec_ = END
ex = ox + ec_ * cell + cell / 2
ey = total_h + oy - er * cell - cell / 2
ax0.text(ex, ey, "E", fontsize=15, ha="center", va="center",
         fontweight="bold", color=C_BOUND, zorder=7)

# Stats box
num_explored = len(explored_order)
num_on_path = len(optimal_path)
num_reachable = sum(1 for r in range(ROWS) for c in range(COLS)
                    if grid[r][c] != "#")

stats_x = 8
stats_y = -0.5

ax0.text(stats_x, stats_y,
    f"Explored: {num_explored} cells     "
    f"Optimal path: {num_on_path} cells     "
    f"Wasted: {num_explored - num_on_path} cells",
    fontsize=11.5, ha="center", color=C_RULE, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1: The explanation
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(0, 16); ax1.set_ylim(-2, 5)

# Left box: what UCS knows
def box(ax, xy, w, h, color, alpha_f=0.06, lw=1.5):
    ax.add_patch(patches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.15",
        facecolor=color, alpha=alpha_f, edgecolor=color, linewidth=lw))

box(ax1, (0.3, -1.5), 7.2, 6.0, C_I, alpha_f=0.04, lw=1.2)
ax1.text(3.9, 4.0, "What UCS Knows", fontsize=13, ha="center",
         fontweight="bold", color=C_I)
ax1.text(3.9, 3.1,
    "\"How far am I from the start?\"", fontsize=11.5, ha="center",
    color=C_I, fontweight="bold", fontstyle="italic")
ax1.text(3.9, 2.2,
    "UCS processes states by increasing", fontsize=10.5, ha="center",
    color=MEDIUM)
ax1.text(3.9, 1.5,
    "past cost — distance FROM the start.", fontsize=10.5, ha="center",
    color=MEDIUM)
ax1.text(3.9, 0.5,
    "It expands outward like a ripple", fontsize=10.5, ha="center",
    color=C_I)
ax1.text(3.9, -0.2,
    "in a pond, uniformly in every direction.", fontsize=10.5, ha="center",
    color=C_I)
ax1.text(3.9, -1.1,
    "Hence the name: uniform-cost search.", fontsize=10, ha="center",
    color=SUBTLE, fontstyle="italic")

# Right box: what UCS doesn't know
box(ax1, (8.5, -1.5), 7.2, 6.0, C_NEG, alpha_f=0.04, lw=1.2)
ax1.text(12.1, 4.0, "What UCS Does NOT Know", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)
ax1.text(12.1, 3.1,
    "\"How far am I from the goal?\"", fontsize=11.5, ha="center",
    color=C_NEG, fontweight="bold", fontstyle="italic")
ax1.text(12.1, 2.2,
    "UCS has no sense of direction.", fontsize=10.5, ha="center",
    color=MEDIUM)
ax1.text(12.1, 1.5,
    "It explores states that are clearly", fontsize=10.5, ha="center",
    color=MEDIUM)
ax1.text(12.1, 0.5,
    "away from the goal, wasting work.", fontsize=10.5, ha="center",
    color=C_NEG)
ax1.text(12.1, -0.2,
    "It is exact and safe, but potentially", fontsize=10.5, ha="center",
    color=MEDIUM)
ax1.text(12.1, -1.1,
    "very wasteful. This motivates A*.", fontsize=10.5, ha="center",
    color=C_BOUND, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 2: Key Takeaway
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(0, 16); ax2.set_ylim(-1.5, 2.5)

box(ax2, (1.0, -1.2), 14, 3.2, C_RULE, alpha_f=0.08, lw=1.5)
ax2.text(8, 1.6, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)
ax2.text(8, 0.7,
    "UCS is correct and handles cycles — but it knows nothing about the goal.",
    fontsize=11, ha="center", color=C_RULE, fontweight="bold")
ax2.text(8, -0.1,
    "To search smarter, we need an estimate of remaining cost-to-go: a heuristic.",
    fontsize=11, ha="center", color=C_RULE)
ax2.text(8, -0.9,
    "That is exactly what A* adds.",
    fontsize=11, ha="center", color=C_BOUND, fontweight="bold")


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/Lecture 6/ucs_inefficiency.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
