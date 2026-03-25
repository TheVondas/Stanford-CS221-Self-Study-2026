"""
Lecture 2 Visual: Why Topological Order Matters

From the lecture notes:
- The computation graph is a GRAPH, not necessarily a simple chain.
- A node may feed into MULTIPLE later nodes.
- Forward: compute values only AFTER dependencies are ready.
- Backward: compute gradients only AFTER downstream gradients are ready.
- Topological ordering guarantees this bookkeeping is correct.

This visual shows:
1. What topological order IS (a valid processing sequence)
2. A graph where a node feeds into multiple later nodes
3. What goes WRONG if you ignore the order
4. What goes RIGHT when you follow it
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Style ─────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_I = "#89b4fa"       # blue
C_J = "#a6e3a1"       # green: forward / ready
C_OUT = "#f9e2af"     # yellow: output
C_RULE = "#fab387"    # orange: rules
C_K = "#f38ba8"       # pink: gradients / backward
C_B = "#cba6f7"       # purple: intermediate
C_DIM = "#585b70"     # dimmed / not yet computed
C_WARN = "#f5c2e7"    # callouts
C_ERR = "#f38ba8"     # red: error / wrong order

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Drawing helpers ───────────────────────────────────────

def draw_node(ax, x, y, label, col, r=0.55, fs=14, sublabel=None,
              sublabel_pos="below", sublabel_col=None):
    circle = patches.Circle((x, y), r, facecolor=col, alpha=0.18,
                             edgecolor=col, linewidth=2.5)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=col)
    if sublabel:
        sc = sublabel_col or col
        if sublabel_pos == "below":
            ax.text(x, y - r - 0.25, sublabel, ha="center",
                    fontsize=9, color=sc, fontweight="bold",
                    fontfamily="monospace")
        else:
            ax.text(x, y + r + 0.25, sublabel, ha="center",
                    fontsize=9, color=sc, fontweight="bold",
                    fontfamily="monospace")


def draw_edge(ax, x1, y1, x2, y2, col, r=0.55, lw=2):
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    sx = x1 + ux * (r + 0.05)
    sy = y1 + uy * (r + 0.05)
    ex = x2 - ux * (r + 0.12)
    ey = y2 - uy * (r + 0.12)
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->,head_width=0.2",
                                color=col, lw=lw))


def draw_order_badge(ax, x, y, number, col, r=0.25):
    """Small numbered circle for processing order."""
    circle = patches.Circle((x, y), r, facecolor=col, alpha=0.8,
                             edgecolor="white", linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, str(number), ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1e1e2e")


def insight_box(ax, x, y, lines, w=5.5, line_h=0.45):
    h = len(lines) * line_h + 0.25
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=C_RULE, alpha=0.10, edgecolor=C_RULE, linewidth=1.5))
    for i, (txt, bold) in enumerate(lines):
        yy = y + h - 0.28 - i * line_h
        ax.text(x + w / 2, yy, txt, ha="center", fontsize=9.5,
                color=C_RULE, fontweight="bold" if bold else "normal")


# ═══════════════════════════════════════════════════════════
#  GRAPH for sections 2–4:  y = (x1 + x2) * x2
#  This has a FAN-OUT: x2 feeds into both sum and mul
#
#  x1 ──┐
#        ├──► sum ──┐
#  x2 ──┤           ├──► mul ──► y
#        └──────────┘
#
#  Topological order matters because x2 is used in TWO places,
#  and sum must be computed before mul.
# ═══════════════════════════════════════════════════════════

# Node positions (shared across sections)
x1_p = (1.5, 2)
x2_p = (1.5, -1)
sum_p = (5.5, 1.5)
mul_p = (9.5, 0.5)

def draw_full_graph(ax, node_colors, edge_col, labels=None,
                    sublabels=None, sublabel_positions=None,
                    order_badges=None, badge_col=None):
    """Draw the y = (x1+x2)*x2 graph with configurable styling."""
    positions = [x1_p, x2_p, sum_p, mul_p]
    default_labels = ["x₁", "x₂", "sum", "y"]
    labs = labels or default_labels

    for i, ((x, y), lbl, col) in enumerate(zip(positions, labs, node_colors)):
        sl = sublabels[i] if sublabels else None
        sp = sublabel_positions[i] if sublabel_positions else "below"
        draw_node(ax, x, y, lbl, col, sublabel=sl, sublabel_pos=sp)

    # Edges
    # x1 → sum
    draw_edge(ax, *x1_p, *sum_p, edge_col)
    # x2 → sum
    draw_edge(ax, *x2_p, *sum_p, edge_col)
    # sum → mul
    draw_edge(ax, *sum_p, *mul_p, edge_col)
    # x2 → mul  (the fan-out!)
    draw_edge(ax, *x2_p, *mul_p, edge_col)

    # Order badges
    if order_badges:
        for (x, y), num in zip(positions, order_badges):
            if num is not None:
                draw_order_badge(ax, x + 0.45, y + 0.45, num,
                                 badge_col or C_J)


# ═══════════════════════════════════════════════════════════
#  FIGURE — 5 sections
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 1, figsize=(14, 42),
                         gridspec_kw={"height_ratios": [1.6, 1.8, 2.4, 2.8, 1.2],
                                      "hspace": 0.08})
fig.suptitle("Why Topological Order Matters",
             fontsize=16, fontweight="bold", y=0.99, color=C_OUT)

for a in axes:
    a.axis("off")
    a.set_aspect("equal")


# ───────────────────────────────────────────────────────────
#  SECTION 1: What IS topological order?
# ───────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-1, 15)
ax.set_ylim(-3.5, 5)

ax.text(0, 4.5, "What IS Topological Order?", fontsize=14,
        fontweight="bold", color=C_RULE)

ax.text(0, 3.5,
        "A topological order is a sequence where every node appears AFTER all its dependencies.",
        fontsize=11.5, color=MEDIUM)
ax.text(0, 2.7,
        "It answers:  \"In what order can I process nodes so that, whenever I reach a node,",
        fontsize=11, color=MEDIUM)
ax.text(0, 2.1,
        "everything it depends on is already done?\"",
        fontsize=11, color=MEDIUM)

# Simple linear example: a → b → c
draw_node(ax, 2, -0.5, "a", C_I, r=0.5)
draw_node(ax, 5.5, -0.5, "b", C_B, r=0.5)
draw_node(ax, 9, -0.5, "c", C_OUT, r=0.5)
draw_edge(ax, 2, -0.5, 5.5, -0.5, MEDIUM, r=0.5)
draw_edge(ax, 5.5, -0.5, 9, -0.5, MEDIUM, r=0.5)

draw_order_badge(ax, 2, -0.5 + 0.7, 1, C_J)
draw_order_badge(ax, 5.5, -0.5 + 0.7, 2, C_J)
draw_order_badge(ax, 9, -0.5 + 0.7, 3, C_J)

ax.text(2, -1.4, "no deps", fontsize=8, color=SUBTLE, ha="center")
ax.text(5.5, -1.4, "needs a", fontsize=8, color=SUBTLE, ha="center")
ax.text(9, -1.4, "needs b", fontsize=8, color=SUBTLE, ha="center")

ax.text(11, -0.5, "Simple chain:\nonly one valid\nforward order",
        fontsize=10, color=MEDIUM, va="center")

ax.text(0, -2.8,
        "For a simple chain, topological order is obvious.  "
        "It gets interesting when nodes FAN OUT.",
        fontsize=11, color=C_WARN, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 2: The graph where order matters
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(-1, 15)
ax.set_ylim(-4, 5.5)

ax.text(0, 5.0, "A Graph Where Order Matters:  y = (x₁ + x₂) × x₂",
        fontsize=14, fontweight="bold", color=C_RULE)
ax.text(0, 4.2,
        "x₂  feeds into TWO later nodes.  This is called a fan-out.",
        fontsize=11, color=MEDIUM)

draw_full_graph(ax,
    node_colors=[C_I, C_I, C_B, C_OUT],
    edge_col=MEDIUM)

# Label the operations
ax.text(sum_p[0], sum_p[1] + 0.85, "x₁ + x₂", fontsize=10,
        color=C_B, ha="center", fontfamily="monospace")
ax.text(mul_p[0], mul_p[1] + 0.85, "sum × x₂", fontsize=10,
        color=C_OUT, ha="center", fontfamily="monospace")

# Highlight the fan-out
ax.annotate("x₂ feeds into\nTWO nodes!",
            xy=(x2_p[0] + 0.5, x2_p[1] + 0.3),
            xytext=(x2_p[0] + 2.5, x2_p[1] - 1.8),
            fontsize=11, color=C_WARN, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5,
                            connectionstyle="arc3,rad=0.2"))

# Show dependency list
tx = 11.5
ax.text(tx, 3.5, "Dependencies:", fontsize=11,
        fontweight="bold", color=TEXT)
deps = [
    ("x₁", "none", C_I),
    ("x₂", "none", C_I),
    ("sum", "x₁, x₂", C_B),
    ("y", "sum, x₂", C_OUT),
]
for idx, (node, dep, col) in enumerate(deps):
    y = 2.6 - idx * 0.6
    ax.text(tx, y, f"{node}:", fontsize=10, color=col, fontweight="bold")
    ax.text(tx + 1.0, y, dep, fontsize=10, color=MEDIUM,
            fontfamily="monospace")

ax.text(tx, 0.3,
        "y  needs BOTH\nsum AND x₂",
        fontsize=10, color=C_WARN, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: Wrong order vs right order (FORWARD)
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(-1, 15)
ax.set_ylim(-8, 6)

ax.text(0, 5.5, "Forward Pass: What Goes Wrong Without Topological Order?",
        fontsize=14, fontweight="bold", color=C_ERR)

# ── WRONG ORDER ──
ax.text(0, 4.3, "WRONG:  Try to compute  y  before  sum", fontsize=12,
        fontweight="bold", color=C_ERR)

wy = 2.0
# x1, x2 ready
draw_node(ax, 1.5, wy + 1, "x₁", C_J, r=0.45, fs=12,
          sublabel="val = 2", sublabel_col=C_J)
draw_node(ax, 1.5, wy - 1.5, "x₂", C_J, r=0.45, fs=12,
          sublabel="val = 3", sublabel_col=C_J)
# sum NOT computed yet
draw_node(ax, 4.5, wy, "sum", C_DIM, r=0.45, fs=12,
          sublabel="val = ???", sublabel_col=C_ERR)
# Try to compute y
draw_node(ax, 7.5, wy - 0.2, "y", C_DIM, r=0.45, fs=12,
          sublabel="val = ???", sublabel_col=C_ERR)

draw_edge(ax, 1.5, wy + 1, 4.5, wy, C_DIM, r=0.45, lw=1.5)
draw_edge(ax, 1.5, wy - 1.5, 4.5, wy, C_DIM, r=0.45, lw=1.5)
draw_edge(ax, 4.5, wy, 7.5, wy - 0.2, C_DIM, r=0.45, lw=1.5)
draw_edge(ax, 1.5, wy - 1.5, 7.5, wy - 0.2, C_DIM, r=0.45, lw=1.5)

# Big X over the attempt
ax.text(7.5, wy - 0.2 - 1.0, "y = sum × x₂ = ??? × 3", fontsize=10,
        ha="center", color=C_ERR, fontfamily="monospace", fontweight="bold")

# Error callout
ax.add_patch(patches.FancyBboxPatch(
    (9, wy - 0.8), 5.5, 1.4, boxstyle="round,pad=0.12",
    facecolor=C_ERR, alpha=0.1, edgecolor=C_ERR, linewidth=1.5))
ax.text(11.75, wy + 0.2, "sum hasn't been computed yet!", fontsize=10.5,
        ha="center", color=C_ERR, fontweight="bold")
ax.text(11.75, wy - 0.4, "y  depends on  sum,  but we", fontsize=9.5,
        ha="center", color=MEDIUM)
ax.text(11.75, wy - 0.85, "tried to compute  y  first.", fontsize=9.5,
        ha="center", color=MEDIUM)

draw_order_badge(ax, 1.5 + 0.35, wy + 1 + 0.35, 1, C_J, r=0.2)
draw_order_badge(ax, 1.5 + 0.35, wy - 1.5 + 0.35, 2, C_J, r=0.2)
draw_order_badge(ax, 7.5 + 0.35, wy - 0.2 + 0.35, 3, C_ERR, r=0.2)
draw_order_badge(ax, 4.5 + 0.35, wy + 0.35, 4, C_ERR, r=0.2)

ax.text(5.5, wy + 1.6, "order: x₁, x₂, y, sum  — WRONG",
        fontsize=10, color=C_ERR, fontweight="bold", ha="center")

# ── Divider ──
ax.plot([0, 14.5], [-1.2, -1.2], "-", color=SUBTLE, alpha=0.4, lw=1)

# ── RIGHT ORDER ──
ax.text(0, -1.8, "RIGHT:  Topological order — compute  sum  before  y",
        fontsize=12, fontweight="bold", color=C_J)

ry = -4.5
draw_node(ax, 1.5, ry + 1, "x₁", C_J, r=0.45, fs=12,
          sublabel="val = 2", sublabel_col=C_J)
draw_node(ax, 1.5, ry - 1.5, "x₂", C_J, r=0.45, fs=12,
          sublabel="val = 3", sublabel_col=C_J)
draw_node(ax, 4.5, ry, "sum", C_J, r=0.45, fs=12,
          sublabel="val = 5", sublabel_col=C_J)
draw_node(ax, 7.5, ry - 0.2, "y", C_J, r=0.45, fs=12,
          sublabel="val = 15", sublabel_col=C_OUT)

draw_edge(ax, 1.5, ry + 1, 4.5, ry, C_J, r=0.45, lw=1.5)
draw_edge(ax, 1.5, ry - 1.5, 4.5, ry, C_J, r=0.45, lw=1.5)
draw_edge(ax, 4.5, ry, 7.5, ry - 0.2, C_J, r=0.45, lw=1.5)
draw_edge(ax, 1.5, ry - 1.5, 7.5, ry - 0.2, C_J, r=0.45, lw=1.5)

draw_order_badge(ax, 1.5 + 0.35, ry + 1 + 0.35, 1, C_J, r=0.2)
draw_order_badge(ax, 1.5 + 0.35, ry - 1.5 + 0.35, 2, C_J, r=0.2)
draw_order_badge(ax, 4.5 + 0.35, ry + 0.35, 3, C_J, r=0.2)
draw_order_badge(ax, 7.5 + 0.35, ry - 0.2 + 0.35, 4, C_J, r=0.2)

ax.text(5.5, ry + 1.6, "order: x₁, x₂, sum, y  — CORRECT",
        fontsize=10, color=C_J, fontweight="bold", ha="center")

ax.text(7.5, ry - 0.2 - 1.0, "y = sum × x₂ = 5 × 3 = 15",
        fontsize=10, ha="center", color=C_OUT, fontfamily="monospace",
        fontweight="bold")

# Right side: the rule
ax.add_patch(patches.FancyBboxPatch(
    (9, ry - 0.8), 5.5, 1.4, boxstyle="round,pad=0.12",
    facecolor=C_J, alpha=0.1, edgecolor=C_J, linewidth=1.5))
ax.text(11.75, ry + 0.2, "Every dependency is ready!", fontsize=10.5,
        ha="center", color=C_J, fontweight="bold")
ax.text(11.75, ry - 0.4, "When we reach  y,  both", fontsize=9.5,
        ha="center", color=MEDIUM)
ax.text(11.75, ry - 0.85, "sum  and  x₂  are computed.", fontsize=9.5,
        ha="center", color=MEDIUM)


# ───────────────────────────────────────────────────────────
#  SECTION 4: Backward — the same issue, reversed
# ───────────────────────────────────────────────────────────
ax = axes[3]
ax.set_xlim(-1, 15)
ax.set_ylim(-8.5, 6.5)

ax.text(0, 6.0, "Backward Pass: Same Principle, Reversed Direction",
        fontsize=14, fontweight="bold", color=C_RULE)
ax.text(0, 5.2,
        "In the backward pass, the problem is the opposite: a node may receive",
        fontsize=11, color=MEDIUM)
ax.text(0, 4.5,
        "gradients from MULTIPLE downstream nodes. You must wait for ALL of them.",
        fontsize=11, color=MEDIUM)

# ── The graph with gradient flow ──
gy = 1.5

draw_node(ax, 1.5, gy + 1.5, "x₁", C_I, r=0.5, fs=13)
draw_node(ax, 1.5, gy - 2, "x₂", C_I, r=0.5, fs=13)
draw_node(ax, 5.5, gy + 0.5, "sum", C_B, r=0.5, fs=13)
draw_node(ax, 9.5, gy, "y", C_OUT, r=0.5, fs=13)

# Forward edges (dimmed)
draw_edge(ax, 1.5, gy + 1.5, 5.5, gy + 0.5, C_DIM, r=0.5, lw=1)
draw_edge(ax, 1.5, gy - 2, 5.5, gy + 0.5, C_DIM, r=0.5, lw=1)
draw_edge(ax, 5.5, gy + 0.5, 9.5, gy, C_DIM, r=0.5, lw=1)
draw_edge(ax, 1.5, gy - 2, 9.5, gy, C_DIM, r=0.5, lw=1)

# Backward arrows (prominent)
draw_edge(ax, 9.5, gy, 5.5, gy + 0.5, C_K, r=0.5, lw=2.5)
draw_edge(ax, 9.5, gy, 1.5, gy - 2, C_K, r=0.5, lw=2.5)
draw_edge(ax, 5.5, gy + 0.5, 1.5, gy + 1.5, C_K, r=0.5, lw=2.5)
draw_edge(ax, 5.5, gy + 0.5, 1.5, gy - 2, C_K, r=0.5, lw=2.5)

# Gradient values
ax.text(9.5, gy + 0.85, "grad = 1", fontsize=9, ha="center",
        color=C_K, fontfamily="monospace", fontweight="bold")
ax.text(5.5, gy + 0.5 + 0.85, "grad = x₂ = 3", fontsize=9, ha="center",
        color=C_K, fontfamily="monospace", fontweight="bold")
ax.text(1.5, gy + 1.5 + 0.85, "grad = x₂ = 3", fontsize=9, ha="center",
        color=C_K, fontfamily="monospace", fontweight="bold")

# x2 receives gradient from TWO sources
ax.text(1.5, gy - 2 - 0.85, "grad = sum + x₂", fontsize=9, ha="center",
        color=C_K, fontfamily="monospace", fontweight="bold")
ax.text(1.5, gy - 2 - 1.25, "= 5 + 3 = 8", fontsize=9, ha="center",
        color=C_OUT, fontfamily="monospace", fontweight="bold")

# Highlight the fan-in on x2
ax.annotate("x₂ receives gradient\nfrom TWO paths!",
            xy=(1.5, gy - 2 + 0.5),
            xytext=(5, gy - 3.5),
            fontsize=11, color=C_WARN, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.8,
                            connectionstyle="arc3,rad=-0.2"))

# Label the two paths
ax.text(6.5, gy - 1.8, "path 1:\ny → x₂\n(direct)", fontsize=8.5,
        color=C_K, ha="center", fontweight="bold")
ax.text(3.5, gy - 0.9, "path 2:\ny → sum → x₂\n(through sum)", fontsize=8.5,
        color=C_K, ha="center", fontweight="bold")

# Backward order
bw_y = gy - 4.5
ax.text(0, bw_y + 1.3, "Backward topological order:", fontsize=12,
        fontweight="bold", color=C_K)

bw_steps = [
    ("①", "y", "grad(y) = 1", "start at root"),
    ("②", "sum", "grad(sum) = grad(y) × x₂ = 1 × 3 = 3",
     "y's gradient is ready"),
    ("③", "x₂",
     "grad(x₂) = grad(y) × sum  +  grad(sum) × 1 = 5 + 3 = 8",
     "BOTH y and sum gradients ready"),
    ("④", "x₁", "grad(x₁) = grad(sum) × 1 = 3",
     "sum's gradient is ready"),
]

for idx, (num, node, formula, reason) in enumerate(bw_steps):
    y = bw_y + 0.3 - idx * 1.0
    ax.text(0.3, y, num, fontsize=11, color=C_K, fontweight="bold")
    ax.text(1.0, y, node, fontsize=11, color=C_OUT if node == "y" else
            C_B if node == "sum" else C_I, fontweight="bold")
    ax.text(2.2, y, formula, fontsize=9.5, fontfamily="monospace",
            color=TEXT)
    ax.text(2.2, y - 0.4, reason, fontsize=8.5, color=SUBTLE,
            fontstyle="italic")

# Key point about x2
ax.add_patch(patches.FancyBboxPatch(
    (0.3, bw_y - 3.8), 14, 1.2, boxstyle="round,pad=0.12",
    facecolor=C_WARN, alpha=0.08, edgecolor=C_WARN, linewidth=1.5))
ax.text(7.3, bw_y - 3.0,
        "x₂'s gradient is the SUM of contributions from all downstream nodes.",
        fontsize=11, ha="center", color=C_WARN, fontweight="bold")
ax.text(7.3, bw_y - 3.6,
        "If we computed grad(x₂) before grad(sum), we'd miss the contribution through sum!",
        fontsize=10.5, ha="center", color=MEDIUM)


# ───────────────────────────────────────────────────────────
#  SECTION 5: Summary
# ───────────────────────────────────────────────────────────
ax = axes[4]
ax.set_xlim(0, 14)
ax.set_ylim(-3.5, 4)

ax.text(0.5, 3.5, "Summary", fontsize=14,
        fontweight="bold", color=C_RULE)

ax.text(0.5, 2.5, "The guarantee of topological order:", fontsize=12,
        fontweight="bold", color=MEDIUM)
ax.text(0.5, 1.8,
        "Forward:  when you reach a node, all its INPUTS are already computed.",
        fontsize=11, color=C_J, fontweight="bold")
ax.text(0.5, 1.1,
        "Backward: when you reach a node, all its DOWNSTREAM gradients are already computed.",
        fontsize=11, color=C_K, fontweight="bold")

insight_box(ax, 1.0, -2.5, [
    ("For simple chains, topological order is trivial (just left-to-right / right-to-left)", False),
    ("For graphs with fan-out, a node feeds multiple later nodes — order matters", True),
    ("Fan-out in forward = fan-IN in backward: gradients must be SUMMED from all paths", True),
    ("Topological sort ensures you never read a value or gradient that hasn't been computed yet", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/topological_order.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
