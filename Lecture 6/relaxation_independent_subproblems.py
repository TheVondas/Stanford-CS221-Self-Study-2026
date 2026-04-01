"""
Lecture 6 Visual: Relaxation Example 3 — Independent Subproblems (8-Puzzle)

Shows:
1. The 8-puzzle: current state → goal state
2. Original constraint: tiles cannot overlap → all moves are coupled
3. Relaxation: tiles CAN overlap → each tile becomes independent
4. Each tile's Manhattan distance computed individually on its own mini-grid
5. Heuristic = sum of individual distances
6. Why this decomposition is powerful and guaranteed safe
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


# ── Tile colors (each tile gets a unique color for tracking) ──
TILE_COLORS = {
    1: "#89b4fa",   # blue
    2: "#a6e3a1",   # green
    3: "#f38ba8",   # pink
    4: "#f9e2af",   # yellow
    5: "#fab387",   # orange
    6: "#cba6f7",   # purple
    7: "#89dceb",   # teal
    8: "#f5c2e7",   # mauve
}


def draw_puzzle(ax, grid, ox, oy, cell, title="", title_color=TEXT,
                highlight_tile=None):
    """Draw a 3×3 puzzle grid.
    grid is a 3×3 list where 0 = blank."""
    total = 3 * cell

    if title:
        ax.text(ox + total / 2, oy + total + 0.3, title, fontsize=11,
                ha="center", fontweight="bold", color=title_color)

    for r in range(3):
        for c in range(3):
            x = ox + c * cell
            y = oy + (2 - r) * cell  # row 0 at top
            val = grid[r][c]

            if val == 0:
                # Blank
                fc, fa, ec = C_DIM, 0.15, C_DIM
            elif highlight_tile is not None and val == highlight_tile:
                # Highlighted tile
                fc = TILE_COLORS[val]
                fa, ec = 0.25, TILE_COLORS[val]
            elif highlight_tile is not None and val != highlight_tile:
                # Dimmed tile (not the one we're focusing on)
                fc, fa, ec = C_DIM, 0.06, C_DIM
            else:
                # Normal tile
                fc = TILE_COLORS.get(val, C_NODE)
                fa, ec = 0.15, TILE_COLORS.get(val, C_NODE)

            rect = patches.FancyBboxPatch(
                (x + 0.04, y + 0.04), cell - 0.08, cell - 0.08,
                boxstyle="round,pad=0.06",
                facecolor=fc, alpha=fa, edgecolor=ec,
                linewidth=1.5 if (highlight_tile is None or
                                  val == highlight_tile) else 0.5,
                zorder=2)
            ax.add_patch(rect)

            if val != 0:
                tc = TILE_COLORS.get(val, C_NODE)
                if highlight_tile is not None and val != highlight_tile:
                    tc = C_DIM
                ax.text(x + cell / 2, y + cell / 2, str(val),
                        fontsize=14, ha="center", va="center",
                        fontweight="bold", color=tc,
                        alpha=1.0 if (highlight_tile is None or
                                      val == highlight_tile) else 0.3,
                        zorder=5)


# ── Puzzle configurations ────────────────────────────────
current_state = [
    [1, 2, 3],
    [8, 0, 4],
    [7, 6, 5],
]

goal_state = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
]

# Positions: tile → (row, col)
current_pos = {}
goal_pos = {}
for r in range(3):
    for c in range(3):
        v = current_state[r][c]
        if v != 0:
            current_pos[v] = (r, c)
        v2 = goal_state[r][c]
        if v2 != 0:
            goal_pos[v2] = (r, c)

# Manhattan distances
manhattan = {}
for tile in range(1, 9):
    cr, cc = current_pos[tile]
    gr, gc = goal_pos[tile]
    manhattan[tile] = abs(cr - gr) + abs(cc - gc)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 52))
fig.suptitle("Relaxation: Independent Subproblems (8-Puzzle)",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(6, 1, figure=fig,
              height_ratios=[4.0, 3.5, 8.5, 4.5, 3.5, 1.8],
              hspace=0.10,
              top=0.975, bottom=0.008, left=0.04, right=0.96)

cell = 1.2  # cell size for puzzle grids


# ─────────────────────────────────────────────────────────
#  ROW 0: The 8-Puzzle Problem
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-1, 7)
ax0.set_title("  The 8-Puzzle Problem",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# Current state
draw_puzzle(ax0, current_state, ox=2.0, oy=1.0, cell=cell,
            title="Current State", title_color=C_I)

# Arrow
ax0.annotate("", xy=(9.5, 3.0), xytext=(6.5, 3.0),
             arrowprops=dict(arrowstyle="-|>", color=C_RULE, lw=3,
                             alpha=0.7))
ax0.text(8.0, 3.7, "solve", fontsize=11, ha="center", color=C_RULE,
         fontweight="bold")

# Goal state
draw_puzzle(ax0, goal_state, ox=10.0, oy=1.0, cell=cell,
            title="Goal State", title_color=C_POS)

# Description
ax0.text(8, 0.2,
    "Slide tiles into the blank space until the goal configuration is reached.",
    fontsize=10.5, ha="center", color=MEDIUM)
ax0.text(8, -0.5,
    "Each slide costs 1.   Minimize total number of moves.",
    fontsize=10.5, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 1: The Constraint and the Relaxation
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(-0.5, 16.5); ax1.set_ylim(-2.5, 6.5)
ax1.set_title("  The Key Constraint and How We Relax It",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

# Left: original constraint
box(ax1, (-0.3, -2.0), 8.1, 8.0, C_NEG, alpha_f=0.04, lw=1.2)
ax1.text(3.75, 5.5, "Original Problem", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)
ax1.text(3.75, 4.5, "Tiles cannot overlap", fontsize=12, ha="center",
         fontfamily="monospace", color=C_NEG, fontweight="bold")
ax1.text(3.75, 3.5,
    "Moving one tile affects what", fontsize=10.5, ha="center", color=MEDIUM)
ax1.text(3.75, 2.8,
    "other tiles can do.", fontsize=10.5, ha="center", color=MEDIUM)
ax1.text(3.75, 1.8,
    "All 8 tiles are coupled", fontsize=11, ha="center",
    color=C_NEG, fontweight="bold")
ax1.text(3.75, 1.0,
    "into one tangled search.", fontsize=11, ha="center",
    color=C_NEG, fontweight="bold")
ax1.text(3.75, -0.1,
    "State: full board configuration", fontsize=10, ha="center",
    fontfamily="monospace", color=MEDIUM)
ax1.text(3.75, -0.8,
    "State space: 9!/2 = 181,440", fontsize=10, ha="center",
    fontfamily="monospace", color=C_NEG, fontweight="bold")

# Arrow
ax1.annotate("", xy=(9.3, 3.0), xytext=(8.0, 3.0),
             arrowprops=dict(arrowstyle="-|>", color=C_RULE, lw=3,
                             alpha=0.7))
ax1.text(8.65, 4.0, "Relax:", fontsize=10, ha="center", color=C_RULE,
         fontweight="bold")
ax1.text(8.65, 3.4, "allow", fontsize=9, ha="center", color=C_RULE)
ax1.text(8.65, 2.6, "overlap", fontsize=9, ha="center", color=C_RULE)

# Right: relaxed
box(ax1, (8.8, -2.0), 8.0, 8.0, C_POS, alpha_f=0.04, lw=1.2)
ax1.text(12.8, 5.5, "Relaxed Problem", fontsize=13, ha="center",
         fontweight="bold", color=C_POS)
ax1.text(12.8, 4.5, "Tiles CAN overlap", fontsize=12, ha="center",
         fontfamily="monospace", color=C_POS, fontweight="bold")
ax1.text(12.8, 3.5,
    "Each tile can move freely", fontsize=10.5, ha="center", color=MEDIUM)
ax1.text(12.8, 2.8,
    "without worrying about others.", fontsize=10.5, ha="center",
    color=MEDIUM)
ax1.text(12.8, 1.8,
    "8 independent subproblems!", fontsize=11, ha="center",
    color=C_POS, fontweight="bold")
ax1.text(12.8, 1.0,
    "Each tile solved on its own.", fontsize=11, ha="center",
    color=C_POS, fontweight="bold")
ax1.text(12.8, -0.1,
    "Each subproblem: just one tile", fontsize=10, ha="center",
    fontfamily="monospace", color=MEDIUM)
ax1.text(12.8, -0.8,
    "Closed-form solution: Manhattan dist", fontsize=10, ha="center",
    fontfamily="monospace", color=C_POS, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 2: Each Tile's Individual Manhattan Distance
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(-0.5, 16.5); ax2.set_ylim(-2.5, 12.5)
ax2.set_title("  Each Tile Solved Independently — Manhattan Distance to Goal",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# Draw 8 mini-puzzles, one per tile, showing current→goal on a 3×3 grid
# Layout: 4 tiles per row, 2 rows
mini_cell = 0.52
mini_w = 3 * mini_cell + 0.6  # width per mini-puzzle block
tiles_per_row = 4

for idx, tile in enumerate(range(1, 9)):
    row = idx // tiles_per_row
    col = idx % tiles_per_row

    bx = 0.2 + col * (mini_w + 0.55)
    by = 5.5 - row * 5.8

    tc = TILE_COLORS[tile]
    dist = manhattan[tile]

    # Tile label and distance
    ax2.text(bx + mini_w / 2, by + 3 * mini_cell + 0.9,
             f"Tile {tile}", fontsize=11, ha="center",
             fontweight="bold", color=tc)
    ax2.text(bx + mini_w / 2, by + 3 * mini_cell + 0.4,
             f"distance = {dist}", fontsize=10, ha="center",
             fontfamily="monospace", color=tc, fontweight="bold")

    # Draw mini 3×3 grid
    gox = bx + 0.3
    goy = by

    for r in range(3):
        for c in range(3):
            x = gox + c * mini_cell
            y = goy + (2 - r) * mini_cell

            rect = patches.FancyBboxPatch(
                (x + 0.02, y + 0.02), mini_cell - 0.04, mini_cell - 0.04,
                boxstyle="round,pad=0.02",
                facecolor=TEXT, alpha=0.03, edgecolor=SUBTLE,
                linewidth=0.4, zorder=2)
            ax2.add_patch(rect)

    # Mark current position
    cr, cc = current_pos[tile]
    cx = gox + cc * mini_cell + mini_cell / 2
    cy = goy + (2 - cr) * mini_cell + mini_cell / 2
    circle_cur = plt.Circle((cx, cy), mini_cell * 0.35,
                             facecolor=tc, alpha=0.25,
                             edgecolor=tc, linewidth=1.8, zorder=5)
    ax2.add_patch(circle_cur)
    ax2.text(cx, cy, str(tile), fontsize=10, ha="center", va="center",
             fontweight="bold", color=tc, zorder=6)

    # Mark goal position
    gr, gc = goal_pos[tile]
    gx = gox + gc * mini_cell + mini_cell / 2
    gy = goy + (2 - gr) * mini_cell + mini_cell / 2

    # If current != goal, draw a dashed circle at goal and an arrow
    if (cr, cc) != (gr, gc):
        circle_goal = plt.Circle((gx, gy), mini_cell * 0.3,
                                  facecolor="none",
                                  edgecolor=tc, linewidth=1.5,
                                  linestyle="--", alpha=0.5, zorder=4)
        ax2.add_patch(circle_goal)

        # Arrow from current to goal
        ax2.annotate("", xy=(gx, gy), xytext=(cx, cy),
                     arrowprops=dict(arrowstyle="-|>", color=tc, lw=1.8,
                                     alpha=0.6,
                                     shrinkA=mini_cell * 7,
                                     shrinkB=mini_cell * 6),
                     zorder=7)

    # Distance breakdown below mini-grid
    ax2.text(bx + mini_w / 2, by - 0.3,
             f"|{cr}−{gr}| + |{cc}−{gc}| = {dist}",
             fontsize=8, ha="center", fontfamily="monospace",
             color=MEDIUM)

# Sum line
ax2.plot([0.5, 15.5], [-1.5, -1.5], color=C_BOUND, lw=1.5, alpha=0.5)

dist_strs = [str(manhattan[t]) for t in range(1, 9)]
ax2.text(8, -2.0,
    f"h(s)  =  {' + '.join(dist_strs)}  =  {sum(manhattan.values())}",
    fontsize=14, ha="center", fontfamily="monospace",
    color=C_BOUND, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 3: Why This Decomposition Works
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-3.5, 7)
ax3.set_title("  Why This Heuristic Is Safe and Useful",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

# The guarantee
box(ax3, (0.5, 4.0), 15, 2.5, C_B, alpha_f=0.06, lw=1.5)
ax3.text(8, 6.0, "The Relaxation Guarantee", fontsize=13, ha="center",
         fontweight="bold", color=C_B)
ax3.text(8, 5.1,
    "Allowing overlap can only make the problem easier — never harder.",
    fontsize=11, ha="center", color=C_B)
ax3.text(8, 4.3,
    "Therefore:   h(s) = sum of individual distances  ≤  true FutureCost(s)",
    fontsize=11.5, ha="center", fontfamily="monospace",
    color=C_BOUND, fontweight="bold")

# Why it underestimates
box(ax3, (0.5, 0.5), 7.0, 3.0, C_POS, alpha_f=0.04, lw=1.2)
ax3.text(4.0, 3.0, "Why It Underestimates", fontsize=12, ha="center",
         fontweight="bold", color=C_POS)
ax3.text(4.0, 2.2,
    "In reality, tiles block each other.", fontsize=10.5, ha="center",
    color=MEDIUM)
ax3.text(4.0, 1.5,
    "You might need extra moves to", fontsize=10.5, ha="center",
    color=MEDIUM)
ax3.text(4.0, 0.8,
    "shuffle tiles out of each other's way.", fontsize=10.5, ha="center",
    color=MEDIUM)

# Why it's useful
box(ax3, (8.5, 0.5), 7.0, 3.0, C_RULE, alpha_f=0.04, lw=1.2)
ax3.text(12.0, 3.0, "Why It's Still Useful", fontsize=12, ha="center",
         fontweight="bold", color=C_RULE)
ax3.text(12.0, 2.2,
    "Tiles closer to their goals", fontsize=10.5, ha="center",
    color=MEDIUM)
ax3.text(12.0, 1.5,
    "genuinely need fewer moves.", fontsize=10.5, ha="center",
    color=MEDIUM)
ax3.text(12.0, 0.8,
    "The heuristic captures this well.", fontsize=10.5, ha="center",
    color=MEDIUM)

# The broader pattern
box(ax3, (0.5, -3.0), 15, 3.0, C_RULE, alpha_f=0.08, lw=1.5)
ax3.text(8, -0.5, "The Broader Pattern", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)
ax3.text(8, -1.3,
    "Removing interactions between objects decomposes one hard problem into many easy ones.",
    fontsize=11, ha="center", color=C_RULE)
ax3.text(8, -2.1,
    "1 problem with 181,440 states  →  8 subproblems each solvable in closed form",
    fontsize=11, ha="center", fontfamily="monospace",
    color=C_BOUND, fontweight="bold")
ax3.text(8, -2.8,
    "This idea appears throughout AI and optimization.",
    fontsize=10, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 4: The Three Relaxation Patterns — Summary
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-2.5, 5.5)
ax4.set_title("  The Three Relaxation Patterns",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

patterns = [
    ("1. Closed-form solution", "Remove walls → Manhattan distance",
     "No search needed at all", C_POS),
    ("2. Fewer states", "Free tram → drop tickets from state",
     "Still need search, but on a smaller problem", C_I),
    ("3. Independent subproblems", "Allow overlap → each tile on its own",
     "One hard problem → many trivial ones", C_RULE),
]

for i, (name, example, why, c) in enumerate(patterns):
    y = 4.0 - i * 2.0
    box(ax4, (0.5, y - 0.7), 15, 1.6, c, alpha_f=0.06, lw=1.2)
    ax4.text(1.2, y + 0.4, name, fontsize=12, color=c,
             fontweight="bold")
    ax4.text(6.5, y + 0.4, example, fontsize=10.5, color=MEDIUM)
    ax4.text(6.5, y - 0.3, why, fontsize=10, color=c, fontstyle="italic")


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
    ("Relaxation: allow tiles to overlap → each tile moves independently.", True),
    ("Each tile's optimal move count = Manhattan distance to its goal position.", False),
    ("The heuristic is the sum:  h = 1+1+3+1+1+1+1+3 = 12.", True),
    ("This underestimates because real tiles block each other, adding extra moves.", False),
    ("Underestimating = safe.  The heuristic is consistent by the relaxation theorem.", True),
    ("One tangled problem with 181,440 states → 8 closed-form computations.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax5.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/Lecture 6/relaxation_independent_subproblems.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
