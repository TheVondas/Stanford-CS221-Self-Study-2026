"""
Lecture 6 Visual: Line Search — UCS vs A*

Shows:
1. The number line with states, original costs, and heuristic values
2. UCS exploration: expands blindly in both directions
3. A* with h(s)=2-s: modified costs tilt search rightward
4. Side-by-side exploration comparison
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


# ── Helpers ──────────────────────────────────────────────

def box(ax, xy, w, h, color, alpha_f=0.06, lw=1.5):
    ax.add_patch(patches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.15",
        facecolor=color, alpha=alpha_f, edgecolor=color, linewidth=lw))


def node(ax, cx, cy, label, color=C_NODE, r=0.38, fs=14, lw=2.0,
         alpha_f=0.12):
    c = plt.Circle((cx, cy), r, facecolor=color, alpha=alpha_f,
                    edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(c)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color, zorder=6)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 38))
fig.suptitle("Line Search: UCS vs A*",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(5, 1, figure=fig,
              height_ratios=[4.5, 5.0, 5.5, 4.5, 1.8],
              hspace=0.12,
              top=0.975, bottom=0.01, left=0.04, right=0.96)

# State positions on the number line
states = [-3, -2, -1, 0, 1, 2, 3]
state_x = {s: 3.0 + (s + 3) * 1.65 for s in states}  # spread across figure


# ─────────────────────────────────────────────────────────
#  ROW 0: The Number Line — Setup
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-3.5, 7.5)
ax0.set_title("  The Line Search Problem",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# Problem spec
box(ax0, (0.5, 5.0), 15, 2.0, C_NODE, alpha_f=0.04, lw=1.2)
ax0.text(8, 6.5, "States: ..., -2, -1, 0, 1, 2, ...        "
         "Start: 0        End: 2        Each move costs: 1",
         fontsize=11, ha="center", color=C_NODE, fontweight="bold")
ax0.text(8, 5.5, "Actions: move left or move right along the number line",
         fontsize=10.5, ha="center", color=MEDIUM)

# Draw number line
line_y = 3.0
ax0.plot([state_x[-3] - 0.8, state_x[3] + 0.8], [line_y, line_y],
         color=C_DIM, lw=2, alpha=0.4, zorder=1)

# Nodes
for s in states:
    x = state_x[s]
    if s == 0:
        c = C_POS
    elif s == 2:
        c = C_BOUND
    else:
        c = C_NODE
    node(ax0, x, line_y, str(s), color=c, r=0.42, fs=15, lw=2.5,
         alpha_f=0.15)

# Labels
ax0.text(state_x[0], line_y - 0.8, "START", fontsize=9, ha="center",
         color=C_POS, fontweight="bold")
ax0.text(state_x[2], line_y - 0.8, "END", fontsize=9, ha="center",
         color=C_BOUND, fontweight="bold")

# Original cost labels on edges
for s in states[:-1]:
    x1 = state_x[s]
    x2 = state_x[s + 1]
    mx = (x1 + x2) / 2
    ax0.text(mx, line_y + 0.7, "1", fontsize=10, ha="center",
             color=C_RULE, fontweight="bold")
    # Small bidirectional arrow
    ax0.annotate("", xy=(x2 - 0.5, line_y + 0.45), xytext=(x1 + 0.5, line_y + 0.45),
                 arrowprops=dict(arrowstyle="<->", color=C_RULE, lw=1.5,
                                 alpha=0.5))

# Heuristic values
box(ax0, (0.5, -3.0), 15, 2.8, C_B, alpha_f=0.04, lw=1.0)
ax0.text(8, -0.6, "Heuristic:   h(s) = 2 − s",
         fontsize=12, ha="center", color=C_B, fontweight="bold")

h_vals = {s: 2 - s for s in states}
for s in states:
    x = state_x[s]
    ax0.text(x, -1.5, f"h({s})", fontsize=9, ha="center",
             fontfamily="monospace", color=MEDIUM)
    ax0.text(x, -2.3, f"= {h_vals[s]}", fontsize=11, ha="center",
             fontfamily="monospace", color=C_B, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1: Modified Costs Computation
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(0, 16); ax1.set_ylim(-4, 7.5)
ax1.set_title("  How A* Modifies the Costs",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# Formula
box(ax1, (0.5, 5.0), 15, 2.0, C_RULE, alpha_f=0.08, lw=1.5)
ax1.text(8, 6.5, "A* Modified Cost Formula", fontsize=12, ha="center",
         fontweight="bold", color=C_RULE)
ax1.text(8, 5.5, "c'(s, a) = c(s, a) + h(s') − h(s)",
         fontsize=13, ha="center", fontfamily="monospace",
         color=C_RULE, fontweight="bold")

# Draw the number line with modified costs
line_y = 3.0
ax1.plot([state_x[-3] - 0.8, state_x[3] + 0.8], [line_y, line_y],
         color=C_DIM, lw=2, alpha=0.4, zorder=1)

for s in states:
    x = state_x[s]
    if s == 0:
        c = C_POS
    elif s == 2:
        c = C_BOUND
    else:
        c = C_NODE
    node(ax1, x, line_y, str(s), color=c, r=0.42, fs=15, lw=2.5,
         alpha_f=0.15)

# Modified costs on edges — rightward
for s in states[:-1]:
    x1 = state_x[s]
    x2 = state_x[s + 1]
    mx = (x1 + x2) / 2
    # Right: c'= 1 + h(s+1) - h(s) = 1 + (2-(s+1)) - (2-s) = 1 - 1 = 0
    mod_right = 1 + h_vals[s + 1] - h_vals[s]
    c_right = C_POS if mod_right == 0 else C_RULE
    ax1.annotate("", xy=(x2 - 0.5, line_y + 0.55),
                 xytext=(x1 + 0.5, line_y + 0.55),
                 arrowprops=dict(arrowstyle="-|>", color=c_right, lw=1.8,
                                 alpha=0.6))
    ax1.text(mx, line_y + 0.95, str(mod_right), fontsize=11, ha="center",
             color=c_right, fontweight="bold")

# Modified costs on edges — leftward
for s in states[1:]:
    x1 = state_x[s]
    x2 = state_x[s - 1]
    mx = (x1 + x2) / 2
    # Left: c'= 1 + h(s-1) - h(s) = 1 + (2-(s-1)) - (2-s) = 1 + 1 = 2
    mod_left = 1 + h_vals[s - 1] - h_vals[s]
    ax1.annotate("", xy=(x2 + 0.5, line_y - 0.55),
                 xytext=(x1 - 0.5, line_y - 0.55),
                 arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=1.8,
                                 alpha=0.6))
    ax1.text(mx, line_y - 1.1, str(mod_left), fontsize=11, ha="center",
             color=C_NEG, fontweight="bold")

# Labels
ax1.text(state_x[3] + 1.0, line_y + 0.95, "→ right", fontsize=9,
         color=C_POS, fontweight="bold", va="center")
ax1.text(state_x[3] + 1.0, line_y - 1.1, "← left", fontsize=9,
         color=C_NEG, fontweight="bold", va="center")

# Worked example
box(ax1, (0.5, -3.5), 15, 3.5, C_B, alpha_f=0.04, lw=1.0)
ax1.text(8, -0.5, "Example from State 0:", fontsize=12, ha="center",
         fontweight="bold", color=C_B)

ax1.text(2.5, -1.4, "Right  (0 → 1):", fontsize=10.5, color=C_POS,
         fontweight="bold")
ax1.text(6.5, -1.4, "c'  =  1 + h(1) − h(0)  =  1 + 1 − 2  =  0",
         fontsize=10.5, fontfamily="monospace", color=C_POS)

ax1.text(2.5, -2.3, "Left   (0 → −1):", fontsize=10.5, color=C_NEG,
         fontweight="bold")
ax1.text(6.5, -2.3, "c'  =  1 + h(−1) − h(0)  =  1 + 3 − 2  =  2",
         fontsize=10.5, fontfamily="monospace", color=C_NEG)

ax1.text(8, -3.2,
    "Moving right costs 0 in the modified problem — A* strongly favors it.",
    fontsize=11, ha="center", color=C_BOUND, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 2: Side-by-side UCS vs A* Exploration
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(-0.5, 16.5); ax2.set_ylim(-5, 8.5)
ax2.set_title("  Exploration Comparison: UCS vs A*",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# ── LEFT: UCS ──
box(ax2, (-0.3, -4.5), 8.1, 12.5, C_I, alpha_f=0.03, lw=1.2)
ax2.text(3.75, 7.5, "UCS  (no heuristic)", fontsize=13, ha="center",
         fontweight="bold", color=C_I)

# UCS explores by past cost: 0, then -1 and 1 (cost 1), then -2 and 2 (cost 2)
# It explores in both directions
ucs_steps = [
    (0, 0, "Pop 0  (cost 0)", C_POS),
    (1, 1, "Pop −1  (cost 1)", C_I),
    (2, 1, "Pop 1   (cost 1)", C_I),
    (3, 2, "Pop −2  (cost 2)", C_I),
    (4, 2, "Pop 2   (cost 2)  ← END ✓", C_BOUND),
]

ucs_line_y = 5.5
ucs_sx = {s: 0.8 + (s + 3) * 0.95 for s in states}

# Draw UCS number line
ax2.plot([ucs_sx[-3] - 0.5, ucs_sx[3] + 0.5], [ucs_line_y, ucs_line_y],
         color=C_DIM, lw=1.5, alpha=0.3)
for s in states:
    c = C_POS if s == 0 else (C_BOUND if s == 2 else C_DIM)
    node(ax2, ucs_sx[s], ucs_line_y, str(s), color=c, r=0.3, fs=11,
         lw=1.5, alpha_f=0.08)

# UCS step-by-step
for i, (step_num, cost, desc, c) in enumerate(ucs_steps):
    y = 4.0 - i * 1.2
    ax2.text(0.2, y, f"{step_num + 1}.", fontsize=10.5, color=c,
             fontweight="bold")
    ax2.text(0.8, y, desc, fontsize=10, color=c,
             fontfamily="monospace")

# Exploration count
ax2.text(3.75, -2.5, "Explored: 5 states", fontsize=12, ha="center",
         color=C_I, fontweight="bold")
ax2.text(3.75, -3.3, "Including −1 and −2 (wrong direction!)",
         fontsize=10, ha="center", color=C_NEG, fontweight="bold")
ax2.text(3.75, -4.0, "UCS doesn't know where the goal is.",
         fontsize=10, ha="center", color=MEDIUM, fontstyle="italic")

# ── RIGHT: A* ──
box(ax2, (8.5, -4.5), 8.0, 12.5, C_POS, alpha_f=0.03, lw=1.2)
ax2.text(12.5, 7.5, "A*  with  h(s) = 2 − s", fontsize=13, ha="center",
         fontweight="bold", color=C_POS)

# A* explores by modified past cost:
# Pop 0 (mod cost 0)
# Expand: right to 1 (mod cost 0+0=0), left to -1 (mod cost 0+2=2)
# Pop 1 (mod cost 0)
# Expand: right to 2 (mod cost 0+0=0), left to 0 (already explored)
# Pop 2 (mod cost 0) — END!
# Never explores left at all!
astar_steps = [
    (0, 0, "Pop 0   (mod cost 0)", C_POS),
    (1, 0, "Pop 1   (mod cost 0)", C_POS),
    (2, 0, "Pop 2   (mod cost 0)  ← END ✓", C_BOUND),
]

astar_line_y = 5.5
astar_sx = {s: 9.3 + (s + 3) * 0.95 for s in states}

# Draw A* number line
ax2.plot([astar_sx[-3] - 0.5, astar_sx[3] + 0.5],
         [astar_line_y, astar_line_y],
         color=C_DIM, lw=1.5, alpha=0.3)
for s in states:
    c = C_POS if s == 0 else (C_BOUND if s == 2 else C_DIM)
    node(ax2, astar_sx[s], astar_line_y, str(s), color=c, r=0.3, fs=11,
         lw=1.5, alpha_f=0.08)

# A* step-by-step
for i, (step_num, cost, desc, c) in enumerate(astar_steps):
    y = 4.0 - i * 1.2
    ax2.text(8.9, y, f"{step_num + 1}.", fontsize=10.5, color=c,
             fontweight="bold")
    ax2.text(9.5, y, desc, fontsize=10, color=c,
             fontfamily="monospace")

# Exploration count
ax2.text(12.5, -2.5, "Explored: 3 states", fontsize=12, ha="center",
         color=C_POS, fontweight="bold")
ax2.text(12.5, -3.3, "Goes straight to the goal — no wasted work!",
         fontsize=10, ha="center", color=C_POS, fontweight="bold")
ax2.text(12.5, -4.0, "The heuristic guides search rightward.",
         fontsize=10, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 3: Why This Works — Intuition
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-3.5, 7)
ax3.set_title("  Why A* Explores Fewer States",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

box(ax3, (0.5, 3.0), 15, 3.5, C_B, alpha_f=0.06, lw=1.5)
ax3.text(8, 6.0, "The Heuristic Tilts the Cost Landscape", fontsize=13,
         ha="center", fontweight="bold", color=C_B)
ax3.text(8, 5.0,
    "UCS priority:   PastCost(s)                     →  orders by distance from start",
    fontsize=10.5, ha="center", fontfamily="monospace", color=C_I)
ax3.text(8, 4.0,
    "A* priority:    PastCost(s) + h(s)              →  orders by estimated total cost",
    fontsize=10.5, ha="center", fontfamily="monospace", color=C_POS)

# Show priorities for each state
box(ax3, (0.5, -3.0), 15, 5.5, C_RULE, alpha_f=0.04, lw=1.0)
ax3.text(8, 2.0, "Priority Comparison for Each State", fontsize=12,
         ha="center", fontweight="bold", color=C_RULE)

# Header
headers = ["State", "PastCost(s)", "h(s)", "UCS priority", "A* priority"]
hx = [2.5, 5.0, 7.2, 9.8, 13.0]
for x, h in zip(hx, headers):
    ax3.text(x, 1.2, h, fontsize=10, ha="center", fontweight="bold",
             color=MEDIUM)
ax3.plot([1.0, 15.0], [0.85, 0.85], color=SUBTLE, lw=0.8, alpha=0.4)

# Data rows (only states reachable in optimal direction for clarity)
table_data = [
    (0,  0, 2, 0, 2, C_POS),
    (1,  1, 1, 1, 2, C_NODE),
    (2,  2, 0, 2, 2, C_BOUND),
    (-1, 1, 3, 1, 4, C_NEG),
    (-2, 2, 4, 2, 6, C_NEG),
]

for i, (s, pc, h, ucs_p, astar_p, c) in enumerate(table_data):
    y = 0.3 - i * 0.65
    vals = [str(s), str(pc), str(h), str(ucs_p), str(astar_p)]
    for x, v in zip(hx, vals):
        ax3.text(x, y, v, fontsize=10.5, ha="center",
                 fontfamily="monospace", color=c, fontweight="bold")
    ax3.plot([1.0, 15.0], [y - 0.3, y - 0.3], color=C_DIM, lw=0.3,
             alpha=0.2)

ax3.text(8, -2.6,
    "A* gives states −1 and −2 high priority (4 and 6) — they get pushed to the back.",
    fontsize=10.5, ha="center", color=C_BOUND, fontweight="bold")


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
    ("UCS orders by PastCost(s) alone — it has no sense of direction.", True),
    ("A* orders by PastCost(s) + h(s) — it prefers states likely closer to the goal.", False),
    ("The heuristic makes moving toward the goal \"cheaper\" in the modified problem.", True),
    ("In this example, rightward moves cost 0 (modified) — A* goes straight there.", False),
    ("Same optimal solution, fewer states explored.", True),
    ("The original problem is unchanged — only the search order improves.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/Lecture 6/line_search_ucs_vs_astar.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
