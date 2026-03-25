"""
Lecture 5 Visual: Dynamic Programming — Memoization and Why It Works

Shows:
1. Why caching is valid: futureCost(s) depends ONLY on state s, not on
   how you arrived — the state encodes everything needed for the future
2. Side-by-side: exhaustive search tree (9 nodes) vs DP DAG (4 nodes)
   for the walk/tram problem with n=4
3. The memoization flow: check cache → hit? return → miss? compute, store, return
4. When DP helps (merging paths / diamond structure) vs when it doesn't
   (unique states / no repeated subproblems)
5. Cost tradeoff: exhaustive O(d) memory / O(b^d) time  vs  DP O(|S|) both
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

def node(ax, cx, cy, label, color=C_NODE, r=0.32, fs=12, lw=2.0,
         alpha_f=0.12):
    c = plt.Circle((cx, cy), r, facecolor=color, alpha=alpha_f,
                    edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(c)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color, zorder=6)


def edge(ax, x1, y1, x2, y2, color, label="", side="left",
         lw=2.0, alpha=0.7, shrA=16, shrB=16):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                alpha=alpha, shrinkA=shrA, shrinkB=shrB),
                zorder=3)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        off = -0.4 if side == "left" else 0.4
        ax.text(mx + off, my, label, fontsize=8, ha="center", va="center",
                color=color, fontweight="bold", alpha=alpha,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1))


def box(ax, xy, w, h, color, alpha_f=0.06, lw=1.5):
    ax.add_patch(patches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.15",
        facecolor=color, alpha=alpha_f, edgecolor=color, linewidth=lw))


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 46))
fig.suptitle("Dynamic Programming: Memoization and Why It Works",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(6, 1, figure=fig,
              height_ratios=[3.5, 5.5, 3.5, 4.5, 3.0, 1.8],
              hspace=0.14,
              top=0.975, bottom=0.01, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: WHY caching is valid
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-2.5, 7.5)
ax0.set_title("  Why We Can Cache: The Optimal Substructure Principle",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# Central insight box
box(ax0, (0.5, 4.0), 15, 3.0, C_RULE, alpha_f=0.08)
ax0.text(8, 6.5, "The Key Insight", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)
ax0.text(8, 5.7,
    "futureCost(s)  depends ONLY on state  s",
    fontsize=13, ha="center", fontfamily="monospace",
    color=C_RULE, fontweight="bold")
ax0.text(8, 4.9,
    "It does not matter HOW you arrived at  s  —  "
    "only that you ARE at  s.",
    fontsize=11.5, ha="center", color=MEDIUM)
ax0.text(8, 4.3,
    "This is why good state design matters: the state must encode "
    "everything needed for future decisions.",
    fontsize=10.5, ha="center", color=MEDIUM, fontstyle="italic")

# Two-path diagram
box(ax0, (0.5, -2.0), 15, 5.5, C_B, alpha_f=0.04, lw=1.0)
ax0.text(8, 3.2, "Example:  Two Different Paths Both Reach State s",
         fontsize=12, ha="center", fontweight="bold", color=C_B)

# Path A (top)
path_a_y = 2.0
ax0.text(1.0, path_a_y, "Path A:", fontsize=10, color=C_POS,
         fontweight="bold")
for i, (lbl, c) in enumerate([("x", C_DIM), ("y", C_DIM), ("s", C_B)]):
    nx = 3.2 + i * 1.8
    node(ax0, nx, path_a_y, lbl, color=c, r=0.28, fs=11)
    if i < 2:
        edge(ax0, nx + 0.3, path_a_y, nx + 1.5, path_a_y,
             C_POS, lw=1.5, alpha=0.5, shrA=8, shrB=8)
ax0.text(8.0, path_a_y, "cost so far = 5", fontsize=9.5, color=C_POS,
         fontfamily="monospace")

# Path B (bottom)
path_b_y = 0.5
ax0.text(1.0, path_b_y, "Path B:", fontsize=10, color=C_I,
         fontweight="bold")
for i, (lbl, c) in enumerate([("p", C_DIM), ("q", C_DIM), ("r", C_DIM),
                               ("s", C_B)]):
    nx = 3.2 + i * 1.4
    node(ax0, nx, path_b_y, lbl, color=c, r=0.28, fs=11)
    if i < 3:
        edge(ax0, nx + 0.3, path_b_y, nx + 1.1, path_b_y,
             C_I, lw=1.5, alpha=0.5, shrA=8, shrB=8)
ax0.text(8.5, path_b_y, "cost so far = 8", fontsize=9.5, color=C_I,
         fontfamily="monospace")

# Converge arrow to shared future
future_x = 11.0
for py, c in [(path_a_y, C_POS), (path_b_y, C_I)]:
    edge(ax0, 7.0 if py == path_a_y else 7.5, py, future_x - 0.3,
         (path_a_y + path_b_y) / 2, c, lw=1.5, alpha=0.4,
         shrA=10, shrB=5)

# Future from s box
box(ax0, (10.5, -0.2), 5.0, 2.8, C_BOUND, alpha_f=0.08)
ax0.text(13.0, 2.1, "Future from s", fontsize=11, ha="center",
         fontweight="bold", color=C_BOUND)
ax0.text(13.0, 1.3, "futureCost(s) = 4", fontsize=11, ha="center",
         fontfamily="monospace", color=C_BOUND, fontweight="bold")
ax0.text(13.0, 0.6, "same answer", fontsize=10, ha="center",
         color=C_BOUND)
ax0.text(13.0, 0.1, "regardless of path", fontsize=10, ha="center",
         color=C_BOUND)

# Bottom punchline
ax0.text(8, -1.0,
    "Both paths arrive at s.  The best future from s is always the same.",
    fontsize=11, ha="center", color=C_RULE, fontweight="bold")
ax0.text(8, -1.7,
    "So we compute it ONCE, store it, and reuse it every time we reach s again.",
    fontsize=11, ha="center", color=C_POS, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1: Side-by-side — Exhaustive Tree vs DP DAG  (n=4)
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(-0.5, 16.5); ax1.set_ylim(-5, 8.5)
ax1.set_title("  Exhaustive Search  vs  Dynamic Programming   (n = 4)",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# ──── LEFT: Exhaustive tree (9 nodes) ────
box(ax1, (-0.3, -4.5), 7.8, 12.5, C_NEG, alpha_f=0.03, lw=1.2)
ax1.text(3.5, 7.6, "Exhaustive Search", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)
ax1.text(3.5, 7.0, "9 nodes explored", fontsize=10.5, ha="center",
         color=C_NEG)

# Tree positions
eL0 = [(3.5, 5.8)]
eL1 = [(1.8, 3.8), (5.2, 3.8)]
eL2 = [(0.6, 1.8), (2.6, 1.8), (4.0, 1.8), (6.0, 1.8)]
eL3 = [(0.6, -0.2), (4.0, -0.2)]

# Edges
edge(ax1, *eL0[0], *eL1[0], C_POS, "w", "left", lw=1.5, alpha=0.5)
edge(ax1, *eL0[0], *eL1[1], C_I, "t", "right", lw=1.5, alpha=0.5)
edge(ax1, *eL1[0], *eL2[0], C_POS, "w", "left", lw=1.5, alpha=0.5)
edge(ax1, *eL1[0], *eL2[1], C_I, "t", "right", lw=1.5, alpha=0.5)
edge(ax1, *eL1[1], *eL2[2], C_POS, "w", "left", lw=1.5, alpha=0.5)
edge(ax1, *eL1[1], *eL2[3], C_I, "t", "right", lw=1.5, alpha=0.5)
edge(ax1, *eL2[0], *eL3[0], C_POS, "w", "left", lw=1.5, alpha=0.5)
edge(ax1, *eL2[2], *eL3[1], C_POS, "w", "left", lw=1.5, alpha=0.5)

# Nodes
node(ax1, *eL0[0], "1", C_POS, lw=2.5, alpha_f=0.2)
for pos in eL1:
    node(ax1, *pos, "2", C_NODE)
for i, (pos, lbl, c) in enumerate(zip(
        eL2, ["3", "4", "3", "4"],
        [C_NODE, C_BOUND, C_NODE, C_BOUND])):
    node(ax1, *pos, lbl, c, lw=2.5 if c == C_BOUND else 2.0,
         alpha_f=0.2 if c == C_BOUND else 0.12)
for pos in eL3:
    node(ax1, *pos, "4", C_BOUND, lw=2.5, alpha_f=0.2)

# Redundancy markers
for pos in eL2[2:3]:  # state 3 duplicate
    ax1.plot(pos[0] + 0.38, pos[1] + 0.25, marker="*", markersize=10,
             color=C_NEG, zorder=7)
for pos in eL1[1:2]:  # state 2 duplicate
    ax1.plot(pos[0] + 0.38, pos[1] + 0.25, marker="*", markersize=10,
             color=C_NEG, zorder=7)

ax1.text(3.5, -1.2, "* = redundant recomputation", fontsize=9,
         ha="center", color=C_NEG, fontweight="bold")

# Count states
counts = [("State 1:", "1x", MEDIUM), ("State 2:", "2x", C_NEG),
           ("State 3:", "2x", C_NEG), ("State 4:", "4x", C_NEG)]
for i, (lbl, cnt, c) in enumerate(counts):
    ax1.text(0.3, -2.2 - i * 0.55, lbl, fontsize=9, color=MEDIUM)
    ax1.text(2.2, -2.2 - i * 0.55, cnt, fontsize=9, fontweight="bold",
             color=c, fontfamily="monospace")
ax1.text(5.0, -2.8, "Total: 9", fontsize=11, color=C_NEG,
         fontweight="bold")

# ──── RIGHT: DP DAG (4 nodes, each computed once) ────
box(ax1, (8.5, -4.5), 7.8, 12.5, C_POS, alpha_f=0.03, lw=1.2)
ax1.text(12.4, 7.6, "Dynamic Programming", fontsize=13, ha="center",
         fontweight="bold", color=C_POS)
ax1.text(12.4, 7.0, "4 nodes explored  (each state ONCE)", fontsize=10.5,
         ha="center", color=C_POS)

# DAG positions — linear chain with branching
d1 = (12.4, 5.5)
d2 = (12.4, 3.3)
d3 = (10.5, 1.1)
d4 = (14.0, 1.1)

# Edges: 1→2, 2→3, 2→4, 3→4
edge(ax1, *d1, *d2, C_NODE, lw=2.0, alpha=0.6)
edge(ax1, *d2, *d3, C_POS, "walk", "left", lw=2.0, alpha=0.6)
edge(ax1, *d2, *d4, C_I, "tram", "right", lw=2.0, alpha=0.6)
edge(ax1, *d3, d4[0] - 0.1, d4[1] + 0.3, C_POS, "walk", "left",
     lw=2.0, alpha=0.6)

# Nodes
node(ax1, *d1, "1", C_POS, r=0.35, lw=3.0, alpha_f=0.22)
node(ax1, *d2, "2", C_NODE, r=0.35, lw=2.5, alpha_f=0.18)
node(ax1, *d3, "3", C_NODE, r=0.35, lw=2.5, alpha_f=0.18)
node(ax1, *d4, "4", C_BOUND, r=0.35, lw=3.0, alpha_f=0.22)

# Cache annotations beside each node
cache_entries = [
    (d4, "cache[4] = 0   (end state)"),
    (d3, "cache[3] = 1   (walk to 4)"),
    (d2, "cache[2] = 2   (min of walk+cache[3], tram+cache[4])"),
    (d1, "cache[1] = 3   (walk to 2, then use cache[2])"),
]
# Place to the right of the DAG
for (dx, dy), txt in cache_entries:
    # Put annotations below the DAG
    pass  # We'll use a table below

# Computation order
ax1.text(12.4, -0.4, "Computation (bottom-up):", fontsize=10,
         ha="center", fontweight="bold", color=C_POS)

comp_steps = [
    ("1.", "cache[4] = 0", "end state", C_BOUND),
    ("2.", "cache[3] = 1 + cache[4] = 1", "walk to 4", C_NODE),
    ("3.", "cache[2] = min(1+cache[3], 2+cache[4])", "= min(2, 2) = 2", C_NODE),
    ("4.", "cache[1] = 1 + cache[2] = 3", "walk to 2, use cache", C_POS),
]
for i, (num, formula, note, c) in enumerate(comp_steps):
    y = -1.2 - i * 0.7
    ax1.text(9.2, y, num, fontsize=9.5, color=c, fontweight="bold")
    ax1.text(9.7, y, formula, fontsize=9.5, fontfamily="monospace", color=c)
    ax1.text(14.8, y, note, fontsize=8.5, color=MEDIUM, fontstyle="italic")

ax1.text(12.4, -4.1, "Total: 4  (one per state)", fontsize=11,
         color=C_POS, fontweight="bold", ha="center")


# ─────────────────────────────────────────────────────────
#  ROW 2: Memoization Flow
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(0, 16); ax2.set_ylim(-2, 7.5)
ax2.set_title("  How Memoization Works — The Cache Lookup Flow",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

# Flow: call futureCost(s) → check cache → HIT: return → MISS: compute → store → return
flow_y = 4.5
step_w = 2.2

# Step boxes
steps = [
    (0.5,  "Call\nfutureCost(s)", C_I),
    (3.2,  "Check\ncache[s]", C_B),
    (6.5,  "HIT?\nReturn\nimmediately", C_POS),
    (9.5,  "MISS?\nCompute via\nrecurrence", C_NEG),
    (12.8, "Store result\nin cache[s]", C_BOUND),
]

for x, label, color in steps:
    box(ax2, (x, flow_y - 0.8), step_w, 2.2, color, alpha_f=0.10, lw=1.8)
    ax2.text(x + step_w / 2, flow_y + 0.25, label, fontsize=10,
             ha="center", va="center", color=color, fontweight="bold")

# Arrows between steps
arrow_pairs = [
    (0.5 + step_w, flow_y, 3.2, flow_y, C_DIM),
    (3.2 + step_w, flow_y + 0.4, 6.5, flow_y + 0.4, C_POS),
    (3.2 + step_w, flow_y - 0.4, 9.5, flow_y - 0.4, C_NEG),
    (9.5 + step_w, flow_y, 12.8, flow_y, C_DIM),
]
for x1, y1, x2, y2, c in arrow_pairs:
    ax2.annotate("", xy=(x2, y2), xytext=(x1, y2),
                 arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0,
                                 alpha=0.6))

# Labels on HIT/MISS arrows
ax2.text(5.0, flow_y + 0.85, "HIT", fontsize=9, color=C_POS,
         fontweight="bold", ha="center")
ax2.text(7.0, flow_y - 0.85, "MISS", fontsize=9, color=C_NEG,
         fontweight="bold", ha="center")

# Return arrow from stored back to caller
ax2.annotate("", xy=(15.5, flow_y + 1.7), xytext=(13.9, flow_y + 1.4),
             arrowprops=dict(arrowstyle="-|>", color=C_BOUND, lw=2.0,
                             alpha=0.6))
ax2.text(15.3, flow_y + 2.0, "Return\nresult", fontsize=9, color=C_BOUND,
         fontweight="bold", ha="center")

# Code-level summary
box(ax2, (0.5, -1.5), 15, 2.2, C_DIM, alpha_f=0.06, lw=1.0)
ax2.text(8, 0.3, "Pseudocode:", fontsize=11, ha="center",
         fontweight="bold", color=MEDIUM)
code_lines = [
    "def futureCost(s):",
    "    if s in cache:  return cache[s]       # O(1) lookup",
    "    result = min[ c + futureCost(s')  for (a,c,s') in Successors(s) ]",
    "    cache[s] = result                      # store for reuse",
    "    return result",
]
for i, line in enumerate(code_lines):
    ax2.text(2.0, -0.3 - i * 0.4, line, fontsize=9.5,
             fontfamily="monospace", color=C_POS if i == 1 else TEXT)


# ─────────────────────────────────────────────────────────
#  ROW 3: When DP Helps vs When It Doesn't
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(-0.5, 16.5); ax3.set_ylim(-4, 8.5)
ax3.set_title("  When Dynamic Programming Helps — and When It Doesn't",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

# ── LEFT: Merging paths (diamond) — DP helps ──
box(ax3, (0, -3.5), 7.5, 11.5, C_POS, alpha_f=0.03, lw=1.2)
ax3.text(3.75, 7.5, "DP Helps: Merging Paths", fontsize=12,
         ha="center", fontweight="bold", color=C_POS)
ax3.text(3.75, 6.8, "(diamond / lattice structure)", fontsize=10,
         ha="center", color=C_POS)

# Diamond shape: top → two middle → bottom
dt = (3.75, 5.5)
dm1 = (2.0, 3.5)
dm2 = (5.5, 3.5)
db = (3.75, 1.5)

edge(ax3, *dt, *dm1, C_POS, lw=2.0, alpha=0.6)
edge(ax3, *dt, *dm2, C_I, lw=2.0, alpha=0.6)
edge(ax3, *dm1, *db, C_POS, lw=2.0, alpha=0.6)
edge(ax3, *dm2, *db, C_I, lw=2.0, alpha=0.6)

node(ax3, *dt, "A", C_NODE, r=0.35, alpha_f=0.18)
node(ax3, *dm1, "B", C_NODE, r=0.35, alpha_f=0.18)
node(ax3, *dm2, "C", C_NODE, r=0.35, alpha_f=0.18)
node(ax3, *db, "D", C_BOUND, r=0.35, alpha_f=0.22)

ax3.text(3.75, 0.5, "D is reached from both B and C", fontsize=10,
         ha="center", color=MEDIUM)
ax3.text(3.75, -0.2, "Exhaustive: computes future from D twice",
         fontsize=9.5, ha="center", color=C_NEG, fontweight="bold")
ax3.text(3.75, -0.9, "DP: computes future from D once, reuses",
         fontsize=9.5, ha="center", color=C_POS, fontweight="bold")

# Scaling note
ax3.text(3.75, -2.0, "With many layers of merging:", fontsize=10,
         ha="center", color=MEDIUM, fontweight="bold")
ax3.text(3.75, -2.7, "Exhaustive = exponential paths",
         fontsize=10, ha="center", color=C_NEG, fontfamily="monospace")
ax3.text(3.75, -3.2, "DP = linear in distinct states",
         fontsize=10, ha="center", color=C_POS, fontfamily="monospace")

# ── RIGHT: Unique states (tree) — DP doesn't help ──
box(ax3, (8.5, -3.5), 7.8, 11.5, C_NEG, alpha_f=0.03, lw=1.2)
ax3.text(12.4, 7.5, "DP Doesn't Help: Unique States", fontsize=12,
         ha="center", fontweight="bold", color=C_NEG)
ax3.text(12.4, 6.8, "(tree with no merging)", fontsize=10,
         ha="center", color=C_NEG)

# Pure tree: every node is unique
ut = (12.4, 5.5)
ul1 = (10.8, 3.5)
ur1 = (14.0, 3.5)
ul2 = (10.0, 1.5)
ul3 = (11.6, 1.5)
ur2 = (13.2, 1.5)
ur3 = (14.8, 1.5)

for parent, child in [(ut, ul1), (ut, ur1), (ul1, ul2), (ul1, ul3),
                       (ur1, ur2), (ur1, ur3)]:
    edge(ax3, *parent, *child, C_DIM, lw=1.5, alpha=0.4)

labels_pos = [(ut, "A"), (ul1, "B"), (ur1, "C"), (ul2, "D"),
              (ul3, "E"), (ur2, "F"), (ur3, "G")]
for pos, lbl in labels_pos:
    node(ax3, *pos, lbl, C_DIM, r=0.3, alpha_f=0.08, lw=1.5)

ax3.text(12.4, 0.5, "Every state is unique", fontsize=10,
         ha="center", color=MEDIUM)
ax3.text(12.4, -0.2, "No cache hits — nothing to reuse",
         fontsize=9.5, ha="center", color=C_NEG, fontweight="bold")
ax3.text(12.4, -0.9, "DP adds cache overhead but no speedup",
         fontsize=9.5, ha="center", color=C_NEG)

# Example
ax3.text(12.4, -2.0, "Example:", fontsize=10, ha="center",
         color=MEDIUM, fontweight="bold")
ax3.text(12.4, -2.7, "Language generation: each prefix",
         fontsize=10, ha="center", color=MEDIUM)
ax3.text(12.4, -3.2, "is typically a unique state",
         fontsize=10, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 4: Cost Tradeoff Table
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-3.5, 6)
ax4.set_title("  Cost Tradeoff: Exhaustive Search vs Dynamic Programming",
              fontsize=14, fontweight="bold", color=C_BOUND, pad=10, loc="left")

box(ax4, (0.5, -3.0), 15, 8.5, C_BOUND, alpha_f=0.04, lw=1.5)

# Headers
hx = [2.5, 6.5, 11.0]
hy = 4.5
headers = ["", "Exhaustive Search", "Dynamic Programming"]
hcolors = [TEXT, C_NEG, C_POS]
for x, h, c in zip(hx, headers, hcolors):
    ax4.text(x, hy, h, fontsize=12, fontweight="bold", color=c, ha="center")

ax4.plot([0.8, 15.2], [hy - 0.35, hy - 0.35], color=SUBTLE, lw=1, alpha=0.5)

# Rows
rows = [
    ("Time",
     "O( b^d )  exponential\nExplores every path",
     "O( |S| x b )  polynomial*\nEach state processed once",
     3.3),
    ("Memory",
     "O( d )  linear\nOnly call stack stored",
     "O( |S| )  can be large\nFull cache of all states",
     1.5),
    ("Guarantee",
     "Exact optimum\n(if it terminates)",
     "Exact optimum\n(same answer, faster)",
     -0.1),
    ("Weakness",
     "Exponentially slow\non branching graphs",
     "Memory-bounded\nif |S| is huge",
     -1.7),
]

for label, exh, dp, y in rows:
    ax4.text(2.5, y, label, fontsize=11, fontweight="bold", color=C_BOUND,
             ha="center", va="top")
    ax4.text(6.5, y, exh, fontsize=9.5, color=C_NEG, ha="center", va="top",
             fontfamily="monospace", linespacing=1.4)
    ax4.text(11.0, y, dp, fontsize=9.5, color=C_POS, ha="center", va="top",
             fontfamily="monospace", linespacing=1.4)
    ax4.plot([0.8, 15.2], [y - 0.8, y - 0.8], color=C_DIM, lw=0.5,
             alpha=0.2)

ax4.text(8, -2.7,
    "*  |S| = number of distinct states,  b = branching factor,  d = max depth",
    fontsize=9, ha="center", color=SUBTLE, fontstyle="italic")


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
    ("Caching works because the future from state s depends only on s, not the path to s.", True),
    ("The recurrence does not change — only the implementation adds a cache lookup.", False),
    ("DP solves each distinct state exactly once, turning exponential time into polynomial.", True),
    ("The tradeoff: DP uses more memory (cache) but dramatically less time.", False),
    ("DP shines when many paths merge into the same states (overlapping subproblems).", True),
    ("If every state is unique, DP offers no benefit — the cache is never reused.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax5.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_5_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "dynamic_programming.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
