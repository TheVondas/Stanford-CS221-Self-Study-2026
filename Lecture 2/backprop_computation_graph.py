"""
Lecture 2 Visual: Backpropagation on a Computation Graph

Example:  y = (x1 + x2)^2

Shows:
1. The computation graph: nodes, dependencies, operations
2. Forward pass: compute values from leaves to root
3. Backward pass: compute gradients from root to leaves
4. Chain rule: how each backward step works
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Style ─────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_I = "#89b4fa"       # blue:   inputs / forward
C_J = "#a6e3a1"       # green:  forward values
C_OUT = "#f9e2af"     # yellow: root / output
C_RULE = "#fab387"    # orange: rules
C_K = "#f38ba8"       # pink:   gradients / backward
C_B = "#cba6f7"       # purple: intermediate
C_DIM = "#585b70"     # dimmed
C_WARN = "#f5c2e7"    # callouts

CELL = 0.55
GAP = 0.06
S = CELL + GAP

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Drawing helpers ───────────────────────────────────────

def draw_node(ax, x, y, label, value, grad, node_col,
              r=0.55, val_show=True, grad_show=True):
    """Draw a computation graph node as a circle with value and gradient."""
    circle = patches.Circle((x, y), r, facecolor=node_col, alpha=0.2,
                             edgecolor=node_col, linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y + 0.05, label, ha="center", va="center",
            fontsize=12, fontweight="bold", color=node_col)
    if val_show and value is not None:
        ax.text(x, y - r - 0.25, f"val = {value}", ha="center",
                fontsize=9.5, color=C_J, fontweight="bold",
                fontfamily="monospace")
    if grad_show and grad is not None:
        ax.text(x, y + r + 0.25, f"grad = {grad}", ha="center",
                fontsize=9.5, color=C_K, fontweight="bold",
                fontfamily="monospace")


def draw_edge(ax, x1, y1, x2, y2, col, r=0.55, label=None,
              label_side="above"):
    """Draw an arrow between two nodes, accounting for radius."""
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / dist, dy / dist
    sx = x1 + ux * r
    sy = y1 + uy * r
    ex = x2 - ux * (r + 0.08)
    ey = y2 - uy * (r + 0.08)
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color=col, lw=2))
    if label:
        mx = (sx + ex) / 2
        my = (sy + ey) / 2
        offset = 0.25 if label_side == "above" else -0.25
        # Perpendicular offset
        px, py = -uy * offset, ux * offset
        ax.text(mx + px, my + py, label, ha="center", va="center",
                fontsize=9, color=col, fontweight="bold",
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG,
                          edgecolor=col, alpha=0.8, linewidth=0.8))


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
#  FIGURE — 5 sections
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 1, figsize=(14, 42),
                         gridspec_kw={"height_ratios": [1.5, 2.0, 2.0, 3.0, 1.2],
                                      "hspace": 0.10})
fig.suptitle("Backpropagation on a Computation Graph\n"
             "y  =  (x₁ + x₂)²",
             fontsize=16, fontweight="bold", y=0.99, color=C_OUT)

for a in axes:
    a.axis("off")
    a.set_aspect("equal")


# ───────────────────────────────────────────────────────────
#  SECTION 1: The Computation Graph — Structure
# ───────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-2, 16)
ax.set_ylim(-2.5, 5)

ax.text(0, 4.5, "Step 1 — Build the Computation Graph", fontsize=14,
        fontweight="bold", color=C_RULE)
ax.text(0, 3.7, "Break the expression into primitive operations. Each gets a node.",
        fontsize=11, color=MEDIUM)

# Node positions
x1_pos = (2, 1.5)
x2_pos = (2, -0.5)
sum_pos = (6, 0.5)
y_pos = (10, 0.5)

# Draw nodes (no values or grads yet)
draw_node(ax, *x1_pos, "x₁", None, None, C_I,
          val_show=False, grad_show=False)
draw_node(ax, *x2_pos, "x₂", None, None, C_I,
          val_show=False, grad_show=False)
draw_node(ax, *sum_pos, "sum", None, None, C_B,
          val_show=False, grad_show=False)
draw_node(ax, *y_pos, "y", None, None, C_OUT,
          val_show=False, grad_show=False)

# Edges
draw_edge(ax, *x1_pos, *sum_pos, MEDIUM)
draw_edge(ax, *x2_pos, *sum_pos, MEDIUM)
draw_edge(ax, *sum_pos, *y_pos, MEDIUM)

# Operation labels on edges
ax.text(sum_pos[0], sum_pos[1] + 0.85, "x₁ + x₂", ha="center",
        fontsize=10, color=C_B, fontfamily="monospace")
ax.text(y_pos[0], y_pos[1] + 0.85, "sum²", ha="center",
        fontsize=10, color=C_OUT, fontfamily="monospace")

# Node descriptions
ax.text(x1_pos[0], x1_pos[1] + 1.1, "leaf\n(input)", ha="center",
        fontsize=8, color=SUBTLE)
ax.text(x2_pos[0], x2_pos[1] - 1.1, "leaf\n(input)", ha="center",
        fontsize=8, color=SUBTLE)
ax.text(sum_pos[0], sum_pos[1] - 1.1, "internal node\n(operation)", ha="center",
        fontsize=8, color=SUBTLE)
ax.text(y_pos[0], y_pos[1] - 1.1, "root\n(final output)", ha="center",
        fontsize=8, color=SUBTLE)

# Each node stores
ax.text(12.5, 3.7, "Each node stores:", fontsize=11,
        fontweight="bold", color=TEXT)
items = [
    ("name", "what it's called", C_I),
    ("dependencies", "which nodes it needs", MEDIUM),
    ("value", "computed in forward pass", C_J),
    ("gradient", "computed in backward pass", C_K),
]
for idx, (field, desc, col) in enumerate(items):
    y = 2.9 - idx * 0.55
    ax.text(12.5, y, f"• {field}", fontsize=10, color=col,
            fontweight="bold")
    ax.text(14.0, y, desc, fontsize=9, color=MEDIUM)


# ───────────────────────────────────────────────────────────
#  SECTION 2: Forward Pass — Compute Values
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(-2, 16)
ax.set_ylim(-3, 5.5)

ax.text(0, 5.0, "Step 2 — Forward Pass  (left → right: compute values)",
        fontsize=14, fontweight="bold", color=C_J)
ax.text(0, 4.2, '"What is the output?"  Evaluate from leaves to root.',
        fontsize=11, color=MEDIUM)

# Example values
x1_val, x2_val = 2, 3
sum_val = x1_val + x2_val  # 5
y_val = sum_val ** 2       # 25

# Draw graph with values
draw_node(ax, *x1_pos, "x₁", x1_val, None, C_I, grad_show=False)
draw_node(ax, *x2_pos, "x₂", x2_val, None, C_I, grad_show=False)
draw_node(ax, *sum_pos, "sum", sum_val, None, C_B, grad_show=False)
draw_node(ax, *y_pos, "y", y_val, None, C_OUT, grad_show=False)

# Forward arrows (green)
draw_edge(ax, *x1_pos, *sum_pos, C_J, label="+")
draw_edge(ax, *x2_pos, *sum_pos, C_J, label="+", label_side="below")
draw_edge(ax, *sum_pos, *y_pos, C_J, label="( )²")

# Step-by-step trace on the right
tx = 12
ax.text(tx, 3.5, "Forward computation:", fontsize=12,
        fontweight="bold", color=C_J)

steps = [
    ("①", "x₁ = 2", "(given)"),
    ("②", "x₂ = 3", "(given)"),
    ("③", "sum = x₁ + x₂ = 2 + 3 = 5", "(add)"),
    ("④", "y = sum² = 5² = 25", "(square)"),
]
for idx, (num, expr, note) in enumerate(steps):
    y = 2.6 - idx * 0.7
    ax.text(tx, y, num, fontsize=11, color=C_J, fontweight="bold")
    ax.text(tx + 0.5, y, expr, fontsize=10.5, fontfamily="monospace",
            color=TEXT)
    ax.text(tx + 0.5, y - 0.3, note, fontsize=8.5, color=SUBTLE)

# Direction arrow
ax.annotate("", xy=(10, -2.2), xytext=(2, -2.2),
            arrowprops=dict(arrowstyle="->,head_width=0.3",
                            color=C_J, lw=2.5))
ax.text(6, -2.6, "FORWARD  →  values flow left to right",
        fontsize=10, ha="center", color=C_J, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: Backward Pass — Compute Gradients
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(-2, 16)
ax.set_ylim(-3, 5.5)

ax.text(0, 5.0, "Step 3 — Backward Pass  (right → left: compute gradients)",
        fontsize=14, fontweight="bold", color=C_K)
ax.text(0, 4.2,
        '"If I nudge each node, how much does y change?"  Propagate from root to leaves.',
        fontsize=11, color=MEDIUM)

# Gradients
# dy/dy = 1
# dy/dsum = 2*sum = 2*5 = 10
# dy/dx1 = dy/dsum * dsum/dx1 = 10 * 1 = 10
# dy/dx2 = dy/dsum * dsum/dx2 = 10 * 1 = 10

y_grad = 1
sum_grad = 10
x1_grad = 10
x2_grad = 10

# Draw graph with both values and gradients
draw_node(ax, *x1_pos, "x₁", x1_val, x1_grad, C_I)
draw_node(ax, *x2_pos, "x₂", x2_val, x2_grad, C_I)
draw_node(ax, *sum_pos, "sum", sum_val, sum_grad, C_B)
draw_node(ax, *y_pos, "y", y_val, y_grad, C_OUT)

# Backward arrows (pink, reversed direction)
draw_edge(ax, *sum_pos, *x1_pos, C_K, label="×1")
draw_edge(ax, *sum_pos, *x2_pos, C_K, label="×1", label_side="below")
draw_edge(ax, *y_pos, *sum_pos, C_K, label="×2·sum")

# Step-by-step trace on the right
tx = 12
ax.text(tx, 3.5, "Backward computation:", fontsize=12,
        fontweight="bold", color=C_K)

steps = [
    ("①", "grad(y) = 1", "dy/dy = 1 always"),
    ("②", "grad(sum) = 1 × 2·sum = 10", "chain rule: dy/dsum"),
    ("③", "grad(x₁) = 10 × 1 = 10", "chain rule: dy/dx₁"),
    ("④", "grad(x₂) = 10 × 1 = 10", "chain rule: dy/dx₂"),
]
for idx, (num, expr, note) in enumerate(steps):
    y = 2.6 - idx * 0.7
    ax.text(tx, y, num, fontsize=11, color=C_K, fontweight="bold")
    ax.text(tx + 0.5, y, expr, fontsize=10, fontfamily="monospace",
            color=TEXT)
    ax.text(tx + 0.5, y - 0.3, note, fontsize=8.5, color=SUBTLE)

# Direction arrow
ax.annotate("", xy=(2, -2.2), xytext=(10, -2.2),
            arrowprops=dict(arrowstyle="->,head_width=0.3",
                            color=C_K, lw=2.5))
ax.text(6, -2.6, "←  BACKWARD  gradients flow right to left",
        fontsize=10, ha="center", color=C_K, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 4: Chain Rule — How Each Step Works
# ───────────────────────────────────────────────────────────
ax = axes[3]
ax.set_xlim(-1, 15)
ax.set_ylim(-7.5, 7)

ax.text(0, 6.5, "Step 4 — The Chain Rule at Each Node", fontsize=14,
        fontweight="bold", color=C_WARN)
ax.text(0, 5.7,
        "Each backward step has the same pattern: "
        "downstream gradient  ×  local derivative",
        fontsize=11, color=MEDIUM)

# ── Node 1: y (root) ──
ny = 4.2
ax.add_patch(patches.FancyBboxPatch(
    (0, ny - 0.3), 14, 1.7, boxstyle="round,pad=0.15",
    facecolor=C_OUT, alpha=0.06, edgecolor=C_OUT, linewidth=1.5))

ax.text(0.3, ny + 1.0, "① Root:  y", fontsize=12,
        fontweight="bold", color=C_OUT)
ax.text(0.3, ny + 0.4,
        '"If I change y by ε, how much does y change?"   Answer: exactly ε.',
        fontsize=10.5, color=MEDIUM)
ax.text(0.3, ny - 0.1,
        "grad(y)  =  1", fontsize=13,
        fontfamily="monospace", color=C_OUT, fontweight="bold")
ax.text(5.5, ny - 0.1,
        "← This is WHY backprop starts at 1.  The root is its own reference point.",
        fontsize=9.5, color=SUBTLE, fontstyle="italic")

# ── Node 2: sum ──
ny2 = 1.5
ax.add_patch(patches.FancyBboxPatch(
    (0, ny2 - 0.8), 14, 2.7, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))

ax.text(0.3, ny2 + 1.5, "② Internal node:  sum", fontsize=12,
        fontweight="bold", color=C_B)

ax.text(0.3, ny2 + 0.9,
        "y = sum²      →      local derivative:  dy/dsum = 2 · sum = 2 × 5 = 10",
        fontsize=11, fontfamily="monospace", color=TEXT)

ax.text(0.3, ny2 + 0.15,
        "grad(sum)", fontsize=12,
        fontfamily="monospace", color=C_K, fontweight="bold")
ax.text(3.3, ny2 + 0.15,
        "=  grad(y)  ×  local derivative", fontsize=12,
        fontfamily="monospace", color=TEXT)

ax.text(3.3, ny2 - 0.45,
        "=     1     ×     2 · sum", fontsize=12,
        fontfamily="monospace", color=MEDIUM)

# Color-code the parts
ax.text(3.5, ny2 - 1.05,
        "↑ from upstream", fontsize=9, color=C_K, fontweight="bold")
ax.text(7.0, ny2 - 1.05,
        "↑ local (how y responds to sum)", fontsize=9,
        color=C_B, fontweight="bold")

ax.text(3.3, ny2 - 0.45 - 0.6,
        "=    10", fontsize=13,
        fontfamily="monospace", color=C_K, fontweight="bold")

# ── Node 3: x1 ──
ny3 = -2.2
ax.add_patch(patches.FancyBboxPatch(
    (0, ny3 - 0.8), 14, 2.7, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))

ax.text(0.3, ny3 + 1.5, "③ Leaf:  x₁", fontsize=12,
        fontweight="bold", color=C_I)

ax.text(0.3, ny3 + 0.9,
        "sum = x₁ + x₂  →  local derivative:  dsum/dx₁ = 1",
        fontsize=11, fontfamily="monospace", color=TEXT)

ax.text(0.3, ny3 + 0.15,
        "grad(x₁)", fontsize=12,
        fontfamily="monospace", color=C_K, fontweight="bold")
ax.text(3.0, ny3 + 0.15,
        "=  grad(sum)  ×  local derivative", fontsize=12,
        fontfamily="monospace", color=TEXT)

ax.text(3.0, ny3 - 0.45,
        "=     10      ×       1", fontsize=12,
        fontfamily="monospace", color=MEDIUM)

ax.text(3.3, ny3 - 1.05,
        "↑ from upstream", fontsize=9, color=C_K, fontweight="bold")
ax.text(7.3, ny3 - 1.05,
        "↑ local (sum changes 1-for-1 with x₁)", fontsize=9,
        color=C_I, fontweight="bold")

ax.text(3.0, ny3 - 0.45 - 0.6,
        "=    10", fontsize=13,
        fontfamily="monospace", color=C_K, fontweight="bold")

# ── Node 4: x2 (brief, same pattern) ──
ny4 = -5.5
ax.add_patch(patches.FancyBboxPatch(
    (0, ny4 - 0.3), 14, 1.4, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.06, edgecolor=C_I, linewidth=1.5))

ax.text(0.3, ny4 + 0.7, "④ Leaf:  x₂", fontsize=12,
        fontweight="bold", color=C_I)
ax.text(0.3, ny4 + 0.05,
        "grad(x₂)  =  grad(sum) × dsum/dx₂  =  10 × 1  =  10",
        fontsize=11, fontfamily="monospace", color=TEXT)
ax.text(0.3, ny4 - 0.5,
        "Same as x₁ because addition treats both inputs equally.",
        fontsize=10, color=SUBTLE, fontstyle="italic")

# ── The universal formula ──
ax.text(7, -7.0,
        "Every backward step:   grad(node)  =  grad(parent)  ×  local derivative",
        fontsize=12, ha="center", fontweight="bold", color=C_OUT)


# ───────────────────────────────────────────────────────────
#  SECTION 5: Summary — The Big Picture
# ───────────────────────────────────────────────────────────
ax = axes[4]
ax.set_xlim(0, 14)
ax.set_ylim(-3.5, 4)

ax.text(0.5, 3.5, "Step 5 — The Big Picture", fontsize=14,
        fontweight="bold", color=C_RULE)

# Verification
ax.text(0.5, 2.5, "Verify:  y = (x₁ + x₂)²", fontsize=12,
        fontweight="bold", color=MEDIUM)
ax.text(0.5, 1.8,
        "dy/dx₁ = 2(x₁ + x₂) = 2(2 + 3) = 10  ✓     "
        "dy/dx₂ = 2(x₁ + x₂) = 2(2 + 3) = 10  ✓",
        fontsize=11, fontfamily="monospace", color=C_J)
ax.text(0.5, 1.15,
        "Backprop gives the same answer as manual calculus — but it scales to millions of parameters.",
        fontsize=10, color=MEDIUM)

# Summary insight
insight_box(ax, 1.0, -2.8, [
    ("Forward pass:  compute values from leaves → root  (just evaluation)", True),
    ("Backward pass:  compute gradients from root → leaves  (chain rule)", True),
    ("Root gradient = 1  (the output's sensitivity to itself)", False),
    ("At each step:  upstream gradient  ×  local derivative  =  this node's gradient", False),
    ("Backprop is just the chain rule, organized as a graph traversal", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/backprop_computation_graph.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
