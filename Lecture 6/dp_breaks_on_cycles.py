"""
Lecture 6 Visual: Why Dynamic Programming Breaks on Cycles

Shows:
1. DP on an acyclic graph — clean backward computation order (topological)
2. The dependency view — futureCost(s) requires futureCost(s') first
3. A cyclic graph — circular dependencies, no valid computation order
4. Key takeaway — cycles destroy the ordering DP relies on, motivating UCS
"""

import os
import math
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
fig = plt.figure(figsize=(16, 42))
fig.suptitle("Why Dynamic Programming Breaks on Cycles",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(5, 1, figure=fig,
              height_ratios=[4.5, 3.0, 6.5, 4.0, 1.8],
              hspace=0.14,
              top=0.975, bottom=0.01, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: DP on an Acyclic Graph — It Works
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-3.5, 8)
ax0.set_title("  Acyclic Graph: DP Works — Clean Backward Computation Order",
              fontsize=14, fontweight="bold", color=C_POS, pad=10, loc="left")

# Show a simple DAG: 1 → 2 → 3 → 4(end)  with walk/tram branching
# Linear chain with an extra shortcut: 1→2→3→4, plus 2→4 (tram)
# This is acyclic, so topological order exists

box(ax0, (0.5, 2.0), 15, 5.5, C_POS, alpha_f=0.03, lw=1.2)
ax0.text(8, 7.0, "Acyclic State Graph  (no cycles)", fontsize=13,
         ha="center", fontweight="bold", color=C_POS)

# Nodes: 1, 2, 3, 4 in a line
positions_dag = [(3, 4.5), (6.5, 4.5), (10, 4.5), (13.5, 4.5)]
labels_dag = ["1", "2", "3", "4"]
colors_dag = [C_POS, C_NODE, C_NODE, C_BOUND]

for (x, y), lbl, c in zip(positions_dag, labels_dag, colors_dag):
    node(ax0, x, y, lbl, color=c, r=0.38, fs=14, lw=2.5,
         alpha_f=0.18 if c != C_BOUND else 0.22)

# START / END labels
ax0.text(3, 3.7, "START", fontsize=9, ha="center", color=C_POS,
         fontweight="bold")
ax0.text(13.5, 3.7, "END", fontsize=9, ha="center", color=C_BOUND,
         fontweight="bold")

# Forward edges: 1→2, 2→3, 3→4
for i in range(3):
    x1, y1 = positions_dag[i]
    x2, y2 = positions_dag[i + 1]
    edge(ax0, x1, y1, x2, y2, C_POS, "walk", "left", lw=2.0, alpha=0.6,
         shrA=20, shrB=20)

# Shortcut edge: 2→4
edge(ax0, 6.5, 4.9, 13.5, 4.9, C_I, "tram", "right", lw=2.0, alpha=0.6,
     shrA=20, shrB=20)

# Computation order arrows below — show backward order
box(ax0, (0.5, -3.0), 15, 4.5, C_B, alpha_f=0.04, lw=1.0)
ax0.text(8, 1.0, "DP Computation Order  (backward — end to start)",
         fontsize=12, ha="center", fontweight="bold", color=C_B)

# Steps
comp_steps = [
    (13.5, "Step 1", "futureCost(4) = 0", "end state", C_BOUND),
    (10.0, "Step 2", "futureCost(3) = 1 + futureCost(4) = 1", "needs 4 ✓", C_NODE),
    (6.5,  "Step 3", "futureCost(2) = min(1+fc(3), 2+fc(4)) = 2",
     "needs 3,4 ✓", C_NODE),
    (3.0,  "Step 4", "futureCost(1) = 1 + futureCost(2) = 3",
     "needs 2 ✓", C_POS),
]
for i, (x, step, formula, note, c) in enumerate(comp_steps):
    y = -0.1 - i * 0.7
    ax0.text(1.5, y, step, fontsize=10, color=c, fontweight="bold")
    ax0.text(4.5, y, formula, fontsize=9.5, fontfamily="monospace", color=c)
    ax0.text(14.5, y, note, fontsize=9, color=C_POS, fontweight="bold")

# Checkmark summary
ax0.text(8, -2.7, "Every dependency is satisfied before it is needed.",
         fontsize=11, ha="center", color=C_POS, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1: The Dependency Principle
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(0, 16); ax1.set_ylim(-2.5, 6.5)
ax1.set_title("  The Dependency Principle — Why Order Matters for DP",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# Central insight box
box(ax1, (0.5, 2.5), 15, 3.5, C_RULE, alpha_f=0.08)
ax1.text(8, 5.5, "The DP Recurrence Creates Dependencies", fontsize=13,
         ha="center", fontweight="bold", color=C_RULE)
ax1.text(8, 4.6,
    "futureCost(s) = min [ c + futureCost(s') ]",
    fontsize=13, ha="center", fontfamily="monospace",
    color=C_RULE, fontweight="bold")
ax1.text(8, 3.7,
    "To compute futureCost(s), you MUST already know futureCost(s')",
    fontsize=11.5, ha="center", color=MEDIUM)
ax1.text(8, 3.0,
    "s  depends on  s'   →   s'  must be computed BEFORE  s",
    fontsize=11.5, ha="center", fontfamily="monospace",
    color=C_BOUND, fontweight="bold")

# Two-case comparison
# Left: acyclic
box(ax1, (0.5, -2.0), 7.0, 4.0, C_POS, alpha_f=0.04, lw=1.2)
ax1.text(4.0, 1.5, "Acyclic Graph", fontsize=12, ha="center",
         fontweight="bold", color=C_POS)
ax1.text(4.0, 0.8,
    "Dependencies form a DAG", fontsize=10.5, ha="center", color=MEDIUM)
ax1.text(4.0, 0.2,
    "Topological sort gives valid order", fontsize=10.5, ha="center",
    color=C_POS, fontweight="bold")
ax1.text(4.0, -0.5,
    "Compute end states first,", fontsize=10, ha="center", color=MEDIUM)
ax1.text(4.0, -1.1,
    "work backward to start", fontsize=10, ha="center", color=MEDIUM)
ax1.text(4.0, -1.7, "✓  DP works", fontsize=12, ha="center",
         color=C_POS, fontweight="bold")

# Right: cyclic
box(ax1, (8.5, -2.0), 7.0, 4.0, C_NEG, alpha_f=0.04, lw=1.2)
ax1.text(12.0, 1.5, "Cyclic Graph", fontsize=12, ha="center",
         fontweight="bold", color=C_NEG)
ax1.text(12.0, 0.8,
    "Dependencies form a cycle", fontsize=10.5, ha="center", color=MEDIUM)
ax1.text(12.0, 0.2,
    "No topological sort exists", fontsize=10.5, ha="center",
    color=C_NEG, fontweight="bold")
ax1.text(12.0, -0.5,
    "A needs B, B needs A", fontsize=10, ha="center",
    fontfamily="monospace", color=C_NEG)
ax1.text(12.0, -1.1,
    "chicken-and-egg deadlock", fontsize=10, ha="center", color=MEDIUM)
ax1.text(12.0, -1.7, "✗  DP breaks", fontsize=12, ha="center",
         color=C_NEG, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 2: Cyclic Graph — Circular Dependencies
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(-1, 17); ax2.set_ylim(-5.5, 9)
ax2.set_title("  Cyclic Graph: Circular Dependencies Prevent DP",
              fontsize=14, fontweight="bold", color=C_NEG, pad=10, loc="left")

# ── LEFT: The diamond graph with bidirectional edges ──
box(ax2, (-0.5, -0.5), 7.5, 9.0, C_NODE, alpha_f=0.03, lw=1.2)
ax2.text(3.25, 8.0, "The Diamond Graph", fontsize=13, ha="center",
         fontweight="bold", color=C_NODE)
ax2.text(3.25, 7.3, "(bidirectional edges = cycles)", fontsize=10.5,
         ha="center", color=C_NODE)

# Diamond layout: A top-left, B top-right, C bottom-left, D bottom-right
# Actually from the notes: A↔B(1), A↔C(100), B↔C(1), B↔D(100), C↔D(1)
# Better layout: A left, B top, C bottom, D right
dA = (1.5, 4.5)
dB = (3.25, 6.2)
dC = (3.25, 2.8)
dD = (5.0, 4.5)

node(ax2, *dA, "A", C_POS, r=0.4, fs=14, lw=2.5, alpha_f=0.18)
node(ax2, *dB, "B", C_NODE, r=0.4, fs=14, lw=2.5, alpha_f=0.18)
node(ax2, *dC, "C", C_NODE, r=0.4, fs=14, lw=2.5, alpha_f=0.18)
node(ax2, *dD, "D", C_BOUND, r=0.4, fs=14, lw=2.5, alpha_f=0.22)

ax2.text(1.5, 3.7, "START", fontsize=8, ha="center", color=C_POS,
         fontweight="bold")
ax2.text(5.0, 3.7, "END", fontsize=8, ha="center", color=C_BOUND,
         fontweight="bold")

# Bidirectional edges with costs — use curved arrows
bidir_edges = [
    (dA, dB, "1",  0.25),
    (dA, dC, "100", -0.25),
    (dB, dC, "1",  0.25),
    (dB, dD, "100", 0.25),
    (dC, dD, "1",  -0.25),
]

for (x1, y1), (x2, y2), cost, rad in bidir_edges:
    # Forward
    ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=C_I, lw=1.8,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=20, shrinkB=20, alpha=0.6),
                 zorder=3)
    # Backward
    ax2.annotate("", xy=(x1, y1), xytext=(x2, y2),
                 arrowprops=dict(arrowstyle="-|>", color=C_I, lw=1.8,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=20, shrinkB=20, alpha=0.6),
                 zorder=3)
    # Cost label
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    off_x = rad * 1.8 * (y2 - y1) / max(0.01,
            math.sqrt((x2-x1)**2 + (y2-y1)**2))
    off_y = -rad * 1.8 * (x2 - x1) / max(0.01,
            math.sqrt((x2-x1)**2 + (y2-y1)**2))
    ax2.text(mx + off_x, my + off_y, cost, fontsize=9, ha="center",
             va="center", color=C_BOUND, fontweight="bold",
             bbox=dict(facecolor=BG, edgecolor="none", pad=1))

ax2.text(3.25, 0.0, "Cycles:  A→B→A,  A→C→A,\nB→C→B,  etc.",
         fontsize=10, ha="center", color=C_NEG, fontweight="bold")


# ── RIGHT: The dependency graph — circular ──
box(ax2, (8.5, -0.5), 8.0, 9.0, C_NEG, alpha_f=0.03, lw=1.2)
ax2.text(12.5, 8.0, "DP Dependency Graph", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)
ax2.text(12.5, 7.3, "(who needs whose value first?)", fontsize=10.5,
         ha="center", color=C_NEG)

# Show the dependency: futureCost(A) needs futureCost(B), futureCost(C)
#                       futureCost(B) needs futureCost(A), futureCost(C), futureCost(D)
#                       futureCost(C) needs futureCost(A), futureCost(B), futureCost(D)
# This creates circular deps: A→B, B→A, A→C, C→A, B→C, C→B

# Place dependency nodes
dep_positions = {
    "fc(A)": (10.5, 5.8),
    "fc(B)": (14.5, 5.8),
    "fc(C)": (10.5, 3.2),
    "fc(D)": (14.5, 3.2),
}

for lbl, (x, y) in dep_positions.items():
    c = C_POS if "A" in lbl else (C_BOUND if "D" in lbl else C_NODE)
    box(ax2, (x - 0.9, y - 0.35), 1.8, 0.7, c, alpha_f=0.12, lw=1.8)
    ax2.text(x, y, lbl, fontsize=11, ha="center", va="center",
             fontfamily="monospace", color=c, fontweight="bold")

# Dependency arrows — "needs" relationships
# A needs B, C  (because A can go to B and C)
# B needs A, C, D
# C needs A, B, D
dep_arrows = [
    ("fc(A)", "fc(B)", C_NEG, 0.15),   # A needs B
    ("fc(B)", "fc(A)", C_NEG, 0.15),   # B needs A  ← CYCLE!
    ("fc(A)", "fc(C)", C_NEG, -0.15),  # A needs C
    ("fc(C)", "fc(A)", C_NEG, -0.15),  # C needs A  ← CYCLE!
    ("fc(B)", "fc(C)", C_NEG, 0.15),   # B needs C
    ("fc(C)", "fc(B)", C_NEG, 0.15),   # C needs B  ← CYCLE!
    ("fc(B)", "fc(D)", C_POS, 0.0),    # B needs D (ok)
    ("fc(C)", "fc(D)", C_POS, 0.0),    # C needs D (ok)
]

for src, dst, c, rad in dep_arrows:
    x1, y1 = dep_positions[src]
    x2, y2 = dep_positions[dst]
    ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=22, shrinkB=22,
                                 alpha=0.7 if c == C_NEG else 0.4),
                 zorder=3)

# Label the problem arrows
ax2.text(12.5, 6.4, "CYCLE", fontsize=10, ha="center", color=C_NEG,
         fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=3, alpha=0.9,
                   boxstyle="round,pad=0.15"))
ax2.text(9.9, 4.5, "CYCLE", fontsize=10, ha="center", color=C_NEG,
         fontweight="bold", rotation=90,
         bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=3, alpha=0.9,
                   boxstyle="round,pad=0.15"))
ax2.text(12.5, 2.6, "CYCLE", fontsize=10, ha="center", color=C_NEG,
         fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=3, alpha=0.9,
                   boxstyle="round,pad=0.15"))

# Legend
ax2.text(12.5, 1.5, "\"needs\"", fontsize=10, ha="center", color=C_NEG,
         fontweight="bold")
ax2.text(12.5, 0.9,
    "fc(A) needs fc(B)  =  \"to compute futureCost(A),", fontsize=9,
    ha="center", color=MEDIUM)
ax2.text(12.5, 0.3,
    "I must already know futureCost(B)\"", fontsize=9,
    ha="center", color=MEDIUM)

# Bottom explanation
box(ax2, (-0.5, -5.0), 16.5, 4.0, C_NEG, alpha_f=0.06, lw=1.5)
ax2.text(8, -1.5, "The Deadlock", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)
ax2.text(8, -2.3,
    "To compute  futureCost(A)  you need  futureCost(B)",
    fontsize=11.5, ha="center", fontfamily="monospace", color=C_NEG)
ax2.text(8, -3.0,
    "To compute  futureCost(B)  you need  futureCost(A)",
    fontsize=11.5, ha="center", fontfamily="monospace", color=C_NEG)
ax2.text(8, -3.8,
    "Neither can go first.  There is no valid computation order.",
    fontsize=12, ha="center", color=C_BOUND, fontweight="bold")
ax2.text(8, -4.5,
    "DP's topological sort requires a DAG — cycles destroy it.",
    fontsize=11, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 3: What Goes Wrong Concretely
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-3.5, 7)
ax3.set_title("  What Goes Wrong Concretely — Trying DP on the Diamond Graph",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

box(ax3, (0.5, 2.5), 15, 4.0, C_B, alpha_f=0.04, lw=1.2)
ax3.text(8, 6.0, "Attempting the DP Recurrence on the Cyclic Diamond Graph",
         fontsize=12, ha="center", fontweight="bold", color=C_B)

# Show the attempted computation
attempts = [
    ("1.", "Start at end state:", "futureCost(D) = 0", C_BOUND, C_POS),
    ("2.", "Try to compute C:", "futureCost(C) = min(100+fc(A), 1+fc(B), 1+fc(D))",
     C_NODE, C_NEG),
    ("",   "", "But fc(A) and fc(B) are unknown!", "", C_NEG),
    ("3.", "Try to compute B:", "futureCost(B) = min(1+fc(A), 1+fc(C), 100+fc(D))",
     C_NODE, C_NEG),
    ("",   "", "But fc(A) and fc(C) are unknown!", "", C_NEG),
    ("4.", "Try to compute A:", "futureCost(A) = min(1+fc(B), 100+fc(C))",
     C_POS, C_NEG),
    ("",   "", "But fc(B) and fc(C) are unknown!", "", C_NEG),
]

y = 5.2
for num, prefix, formula, c1, c2 in attempts:
    if num:
        ax3.text(1.2, y, num, fontsize=10.5, color=c1, fontweight="bold")
        ax3.text(2.0, y, prefix, fontsize=10, color=MEDIUM)
        ax3.text(6.5, y, formula, fontsize=9.5, fontfamily="monospace",
                 color=c1)
    else:
        ax3.text(6.5, y, formula, fontsize=9.5, color=c2,
                 fontweight="bold", fontstyle="italic")
    y -= 0.55

# Circular arrow showing the loop
box(ax3, (0.5, -3.0), 15, 5.0, C_RULE, alpha_f=0.06, lw=1.5)
ax3.text(8, 1.5, "The Circular Trap", fontsize=12, ha="center",
         fontweight="bold", color=C_RULE)

# Show the loop visually: A → B → C → A
loop_nodes = [("A", 4, 0.2, C_POS), ("B", 8, 0.2, C_NODE),
              ("C", 12, 0.2, C_NODE)]
for lbl, x, y_pos, c in loop_nodes:
    node(ax3, x, y_pos, lbl, color=c, r=0.32, fs=12, lw=2.0, alpha_f=0.15)

# Arrows: A "needs" B, B "needs" C, C "needs" A
edge(ax3, 4.4, 0.35, 7.6, 0.35, C_NEG, "needs", "right",
     lw=2.5, alpha=0.7, shrA=12, shrB=12)
edge(ax3, 8.4, 0.35, 11.6, 0.35, C_NEG, "needs", "right",
     lw=2.5, alpha=0.7, shrA=12, shrB=12)
# C needs A — curved arrow going back
ax3.annotate("", xy=(4, -0.15), xytext=(12, -0.15),
             arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=2.5,
                             connectionstyle="arc3,rad=0.35",
                             shrinkA=16, shrinkB=16, alpha=0.7),
             zorder=3)
ax3.text(8, -1.1, "needs", fontsize=9, ha="center", color=C_NEG,
         fontweight="bold")

# Bottom text
ax3.text(8, -2.0,
    "No matter where you start, you always need something you haven't computed yet.",
    fontsize=11, ha="center", color=C_RULE, fontweight="bold")
ax3.text(8, -2.7,
    "The recurrence chases its own tail — there is no base-to-goal ordering.",
    fontsize=10.5, ha="center", color=MEDIUM, fontstyle="italic")


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
    ("DP computes future costs backward: from end states toward the start.", True),
    ("This requires a topological order — each state's successors are computed first.", False),
    ("Cycles create circular dependencies: A needs B, B needs A.", True),
    ("No valid computation order exists — the recurrence has no starting point.", False),
    ("Solution: abandon future costs.  Switch to past costs, computed forward from the start.", True),
    ("That is the core idea behind Uniform-Cost Search.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/Lecture 6/dp_breaks_on_cycles.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
