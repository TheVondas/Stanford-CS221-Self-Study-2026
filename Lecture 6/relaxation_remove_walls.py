"""
Lecture 6 Visual: Relaxation Example — Remove Walls from a Grid

Shows:
1. Original grid problem with walls (the lecture's 5×5 grid)
2. Relaxed problem with walls removed — Manhattan distance is now exact
3. Side-by-side: heuristic values on each cell
4. Why it works: underestimates or matches, never overestimates
5. Where it's imperfect: a cell can look close but actually need a detour
"""

import os
import heapq
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
C_WALL  = "#45475a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


def box(ax, xy, w, h, color, alpha_f=0.06, lw=1.5):
    ax.add_patch(patches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.15",
        facecolor=color, alpha=alpha_f, edgecolor=color, linewidth=lw))


# ── Grid definitions ─────────────────────────────────────
original_grid = [
    "S....",   # 0
    "###.#",   # 1
    ".....",   # 2
    ".####",   # 3
    "....E",   # 4
]

relaxed_grid = [
    "S....",   # 0
    ".....",   # 1
    ".....",   # 2
    ".....",   # 3
    "....E",   # 4
]

ROWS, COLS = 5, 5
END = (4, 4)


def manhattan(r, c):
    return abs(END[0] - r) + abs(END[1] - c)


# Compute true future costs via BFS on the original grid
def true_future_costs(grid):
    """BFS backward from END to compute true shortest distance for each cell."""
    costs = {}
    queue = [(0, END)]
    costs[END] = 0
    while queue:
        cost, (r, c) = heapq.heappop(queue)
        if costs.get((r, c), float("inf")) < cost:
            continue
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] != "#":
                new_cost = cost + 1
                if new_cost < costs.get((nr, nc), float("inf")):
                    costs[(nr, nc)] = new_cost
                    heapq.heappush(queue, (new_cost, (nr, nc)))
    return costs


true_costs = true_future_costs(original_grid)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 40))
fig.suptitle("Relaxation: Remove Walls → Manhattan Distance Heuristic",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(5, 1, figure=fig,
              height_ratios=[5.5, 5.5, 5.0, 4.5, 1.8],
              hspace=0.12,
              top=0.975, bottom=0.01, left=0.04, right=0.96)

cell = 1.35  # cell size for grids


# ── Grid drawing helper ──────────────────────────────────
def draw_grid(ax, grid, ox, oy, title, title_color, show_values=None,
              value_label="", highlight_cells=None, show_walls=True):
    """Draw a 5×5 grid at position (ox, oy) in axes ax."""
    total_w = COLS * cell
    total_h = ROWS * cell

    ax.text(ox + total_w / 2, oy + total_h + 0.6, title, fontsize=13,
            ha="center", fontweight="bold", color=title_color)

    if value_label:
        ax.text(ox + total_w / 2, oy + total_h + 0.1, value_label,
                fontsize=10, ha="center", color=MEDIUM)

    # Column headers
    for c in range(COLS):
        ax.text(ox + c * cell + cell / 2, oy + total_h + 0.05, str(c),
                fontsize=8.5, ha="center", va="center", color=SUBTLE)

    for r in range(ROWS):
        # Row label
        ax.text(ox - 0.3, oy + total_h - r * cell - cell / 2, str(r),
                fontsize=8.5, ha="center", va="center", color=SUBTLE)

        for c in range(COLS):
            x = ox + c * cell
            y = oy + total_h - (r + 1) * cell
            ch = grid[r][c]

            # Cell background
            if ch == "#" and show_walls:
                fc = C_WALL
                fa = 0.6
            elif highlight_cells and (r, c) in highlight_cells:
                fc = highlight_cells[(r, c)]
                fa = 0.18
            else:
                fc = TEXT
                fa = 0.04

            rect = patches.FancyBboxPatch(
                (x + 0.03, y + 0.03), cell - 0.06, cell - 0.06,
                boxstyle="round,pad=0.04",
                facecolor=fc, alpha=fa, edgecolor=SUBTLE,
                linewidth=0.6, zorder=2)
            ax.add_patch(rect)

            # Wall label
            if ch == "#" and show_walls:
                ax.text(x + cell / 2, y + cell / 2, "#", fontsize=13,
                        ha="center", va="center", fontweight="bold",
                        color=C_NEG, alpha=0.7, zorder=5)
            elif ch == "S":
                ax.text(x + cell / 2, y + cell / 2 + 0.2, "S", fontsize=14,
                        ha="center", va="center", fontweight="bold",
                        color=C_POS, zorder=5)
            elif ch == "E":
                ax.text(x + cell / 2, y + cell / 2 + 0.2, "E", fontsize=14,
                        ha="center", va="center", fontweight="bold",
                        color=C_BOUND, zorder=5)

            # Value overlay
            if show_values and (r, c) in show_values and ch != "#":
                val = show_values[(r, c)]
                vy = y + cell / 2 - 0.15 if ch in ("S", "E") else y + cell / 2
                ax.text(x + cell / 2, vy, str(val), fontsize=12,
                        ha="center", va="center", fontfamily="monospace",
                        color=C_BOUND, fontweight="bold", zorder=5)


# ─────────────────────────────────────────────────────────
#  ROW 0: Side-by-side — Original vs Relaxed Grid
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(-0.5, 16.5); ax0.set_ylim(-1, 10)
ax0.set_title("  Original Problem vs Relaxed Problem",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# Left: original
draw_grid(ax0, original_grid, ox=0.5, oy=0.5,
          title="Original Problem", title_color=C_NEG,
          value_label="(grid with walls)")

# Arrow in the middle
ax0.annotate("", xy=(9.5, 5.2), xytext=(7.8, 5.2),
             arrowprops=dict(arrowstyle="-|>", color=C_RULE, lw=3,
                             alpha=0.7))
ax0.text(8.65, 5.9, "Relax:", fontsize=11, ha="center", color=C_RULE,
         fontweight="bold")
ax0.text(8.65, 5.3, "remove", fontsize=10, ha="center", color=C_RULE)
ax0.text(8.65, 4.7, "walls", fontsize=10, ha="center", color=C_RULE)

# Right: relaxed
draw_grid(ax0, relaxed_grid, ox=9.5, oy=0.5,
          title="Relaxed Problem", title_color=C_POS,
          value_label="(walls removed)")


# ─────────────────────────────────────────────────────────
#  ROW 1: Manhattan Distance Values on Both Grids
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(-0.5, 16.5); ax1.set_ylim(-2, 10)
ax1.set_title("  Heuristic Values: h(r,c) = |r − 4| + |c − 4|  (Manhattan Distance to E)",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

# Compute Manhattan distance for each cell
manhattan_vals = {}
for r in range(ROWS):
    for c in range(COLS):
        manhattan_vals[(r, c)] = manhattan(r, c)

# Left: Original grid with Manhattan values overlaid
draw_grid(ax1, original_grid, ox=0.5, oy=0.0,
          title="Original Grid with h(s) Values", title_color=C_I,
          show_values=manhattan_vals,
          value_label="h(s) = Manhattan distance to E")

# Right: Relaxed grid with Manhattan values (these ARE the true future costs)
draw_grid(ax1, relaxed_grid, ox=9.5, oy=0.0,
          title="Relaxed Grid with h(s) Values", title_color=C_POS,
          show_values=manhattan_vals,
          value_label="h(s) = true FutureCost in relaxed problem")

# Note between
ax1.text(8.5, 4.5, "Same\nvalues!", fontsize=12, ha="center",
         color=C_BOUND, fontweight="bold")
ax1.text(8.5, 3.5, "In the relaxed\nproblem, h(s) is\nexact.", fontsize=9.5,
         ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 2: Why It's a Good Heuristic — Comparison Table
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(0, 16); ax2.set_ylim(-4, 8)
ax2.set_title("  h(s) vs True Future Cost — Why the Heuristic Is Safe",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

box(ax2, (0.5, 5.0), 15, 2.5, C_RULE, alpha_f=0.06, lw=1.5)
ax2.text(8, 7.0, "Relaxation Guarantee", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)
ax2.text(8, 6.1,
    "Removing walls can only make paths shorter or equal — never longer.",
    fontsize=11, ha="center", color=C_RULE)
ax2.text(8, 5.4,
    "Therefore:   h(s) = FutureCost_relaxed(s)  ≤  FutureCost_original(s)   for every state s",
    fontsize=11, ha="center", fontfamily="monospace", color=C_BOUND,
    fontweight="bold")

# Table comparing h(s) vs true cost for selected cells
box(ax2, (0.5, -3.5), 15, 8.0, C_B, alpha_f=0.03, lw=1.0)
ax2.text(8, 4.0, "Comparison for Selected Cells", fontsize=12,
         ha="center", fontweight="bold", color=C_B)

# Headers
headers = ["Cell", "h(s)", "True\nFutureCost", "Gap", "Why?"]
header_x = [2.0, 4.5, 7.0, 9.2, 12.5]
for x, h in zip(header_x, headers):
    ax2.text(x, 3.2, h, fontsize=10, ha="center", fontweight="bold",
             color=MEDIUM)
ax2.plot([1.0, 15.0], [2.75, 2.75], color=SUBTLE, lw=0.8, alpha=0.4)

# Data — pick cells that illustrate the point
# (r, c): cell label, h(s), true_cost, note
selected = [
    ((0, 0), "S (0,0)", 8, 14, "Must detour around both walls"),
    ((0, 4), "(0,4)",   4, 4,  "No wall in the way — h = exact"),
    ((2, 4), "(2,4)",   2, 8,  "Looks close!  But wall below forces long detour"),
    ((2, 0), "(2,0)",   6, 6,  "Direct path exists — h = exact"),
    ((3, 0), "(3,0)",   5, 5,  "Direct path exists — h = exact"),
    ((4, 0), "(4,0)",   4, 4,  "Straight right to E — h = exact"),
    ((4, 4), "E (4,4)", 0, 0,  "Already at goal"),
]

for i, ((r, c), label, h, tc, note) in enumerate(selected):
    y = 2.1 - i * 0.72
    gap = tc - h
    gap_str = f"+{gap}" if gap > 0 else "0"
    gap_color = C_NEG if gap > 3 else (C_RULE if gap > 0 else C_POS)

    ax2.text(header_x[0], y, label, fontsize=10, ha="center",
             fontfamily="monospace", color=C_NODE, fontweight="bold")
    ax2.text(header_x[1], y, str(h), fontsize=10.5, ha="center",
             fontfamily="monospace", color=C_B, fontweight="bold")
    ax2.text(header_x[2], y, str(tc), fontsize=10.5, ha="center",
             fontfamily="monospace", color=C_BOUND, fontweight="bold")
    ax2.text(header_x[3], y, gap_str, fontsize=10.5, ha="center",
             fontfamily="monospace", color=gap_color, fontweight="bold")
    ax2.text(header_x[4], y, note, fontsize=9, ha="center", color=MEDIUM)

    ax2.plot([1.0, 15.0], [y - 0.35, y - 0.35], color=C_DIM, lw=0.3,
             alpha=0.2)

ax2.text(8, -3.1,
    "h(s) ≤ TrueCost  always.    h(s) = TrueCost  when no wall blocks the direct path.",
    fontsize=10.5, ha="center", color=C_BOUND, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 3: The Imperfect-But-Useful Insight
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-3.5, 7.5)
ax3.set_title("  Imperfect But Useful — The (2,4) Example",
              fontsize=14, fontweight="bold", color=C_NEG, pad=10, loc="left")

# Highlight (2,4) on a small grid — it looks close but needs a long detour
highlight = {(2, 4): C_NEG, (4, 4): C_BOUND}
draw_grid(ax3, original_grid, ox=0.3, oy=0.0,
          title="Cell (2,4): h = 2 but true cost = 8",
          title_color=C_NEG,
          highlight_cells=highlight)

# Annotate the problem
ax3.text(7.5, 6.8, "Manhattan distance says (2,4) is just 2 steps from E.",
         fontsize=11, color=C_NEG, fontweight="bold")
ax3.text(7.5, 6.1, "But the wall at row 3 blocks the direct path!", fontsize=11,
         color=C_NEG)
ax3.text(7.5, 5.4, "The real path must go all the way left and around.",
         fontsize=11, color=MEDIUM)

# Why this is OK
box(ax3, (7.2, 1.5), 8.5, 3.5, C_POS, alpha_f=0.06, lw=1.5)
ax3.text(11.45, 4.5, "Why This Is Still Fine", fontsize=12,
         ha="center", fontweight="bold", color=C_POS)
ax3.text(11.45, 3.6,
    "h(s) = 2  ≤  8 = FutureCost(s)   ✓", fontsize=11,
    ha="center", fontfamily="monospace", color=C_POS, fontweight="bold")
ax3.text(11.45, 2.8,
    "The heuristic underestimates — it never claims", fontsize=10.5,
    ha="center", color=MEDIUM)
ax3.text(11.45, 2.1,
    "the goal is farther than it really is.", fontsize=10.5,
    ha="center", color=MEDIUM)

# Admissibility / consistency note
box(ax3, (7.2, -3.0), 8.5, 4.0, C_B, alpha_f=0.06, lw=1.2)
ax3.text(11.45, 0.5, "This is what makes it safe:", fontsize=11,
         ha="center", fontweight="bold", color=C_B)
ax3.text(11.45, -0.3,
    "Admissible:   h(s) ≤ FutureCost(s)  always", fontsize=10.5,
    ha="center", fontfamily="monospace", color=C_B)
ax3.text(11.45, -1.1,
    "Consistent:   h(s) ≤ c(s,a) + h(s')  always", fontsize=10.5,
    ha="center", fontfamily="monospace", color=C_B)
ax3.text(11.45, -1.9,
    "Both hold because relaxation only removes", fontsize=10.5,
    ha="center", color=MEDIUM)
ax3.text(11.45, -2.6,
    "constraints — it can never inflate costs.", fontsize=10.5,
    ha="center", color=MEDIUM)


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
    ("Relaxation: remove walls → Manhattan distance becomes the exact future cost.", True),
    ("This gives a closed-form heuristic:  h(r,c) = |r−4| + |c−4|.  No search needed.", False),
    ("It underestimates when walls force detours, but never overestimates.", True),
    ("Underestimating is safe — it preserves optimality.", False),
    ("Overestimating is dangerous — it can make A* return the wrong answer.", True),
    ("A heuristic does not need to be exact to be useful.  It just needs to be consistent.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/Lecture 6/relaxation_remove_walls.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
