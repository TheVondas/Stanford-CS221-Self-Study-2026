"""
Lecture 2 Visual: The Chain Rule in Graph Form

Shows the general pattern:
- For a chain  a → b → c,  dc/da = (dc/db)(db/da)
- Each node only needs: the gradient from downstream + its own local derivative
- Extends to longer chains and branching graphs
- Concrete numeric example alongside the abstract pattern
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
C_J = "#a6e3a1"       # green: forward
C_OUT = "#f9e2af"     # yellow: output
C_RULE = "#fab387"    # orange: rules
C_K = "#f38ba8"       # pink: gradients / backward
C_B = "#cba6f7"       # purple: intermediate
C_DIM = "#585b70"
C_WARN = "#f5c2e7"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Drawing helpers ───────────────────────────────────────

def draw_node(ax, x, y, label, col, r=0.6, fs=16):
    circle = patches.Circle((x, y), r, facecolor=col, alpha=0.18,
                             edgecolor=col, linewidth=2.5)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=col)


def draw_edge(ax, x1, y1, x2, y2, col, r=0.6, lw=2.2):
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / dist, dy / dist
    sx = x1 + ux * (r + 0.05)
    sy = y1 + uy * (r + 0.05)
    ex = x2 - ux * (r + 0.12)
    ey = y2 - uy * (r + 0.12)
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->,head_width=0.25",
                                color=col, lw=lw))


def edge_label(ax, x1, y1, x2, y2, label, col, offset=0.35):
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    px, py = -dy / dist * offset, dx / dist * offset
    ax.text(mx + px, my + py, label, ha="center", va="center",
            fontsize=11, color=col, fontweight="bold",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=BG,
                      edgecolor=col, alpha=0.85, linewidth=0.8))


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
fig, axes = plt.subplots(5, 1, figsize=(14, 38),
                         gridspec_kw={"height_ratios": [1.6, 2.0, 2.4, 2.2, 1.2],
                                      "hspace": 0.08})
fig.suptitle("The Chain Rule in Graph Form",
             fontsize=16, fontweight="bold", y=0.99, color=C_OUT)

for a in axes:
    a.axis("off")
    a.set_aspect("equal")


# ───────────────────────────────────────────────────────────
#  SECTION 1: The Simplest Chain  a → b → c
# ───────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-1, 15)
ax.set_ylim(-3.5, 4.5)

ax.text(0, 4.0, "The Simplest Case:  a → b → c", fontsize=14,
        fontweight="bold", color=C_RULE)
ax.text(0, 3.2,
        "If  c  depends on  b,  and  b  depends on  a,  "
        "how does  c  respond to a nudge in  a ?",
        fontsize=11, color=MEDIUM)

# Nodes
a_pos = (2, 0.5)
b_pos = (6.5, 0.5)
c_pos = (11, 0.5)

draw_node(ax, *a_pos, "a", C_I)
draw_node(ax, *b_pos, "b", C_B)
draw_node(ax, *c_pos, "c", C_OUT)

# Forward edges
draw_edge(ax, *a_pos, *b_pos, C_J)
draw_edge(ax, *b_pos, *c_pos, C_J)

# Local derivative labels on forward edges
edge_label(ax, *a_pos, *b_pos, "db/da", C_J, offset=0.5)
edge_label(ax, *b_pos, *c_pos, "dc/db", C_J, offset=0.5)

# The chain rule formula below
ax.text(6.5, -1.5, "dc/da", fontsize=16, ha="center",
        fontweight="bold", color=C_K, fontfamily="monospace")
ax.text(6.5, -2.3, "=   dc/db   ×   db/da", fontsize=15,
        ha="center", color=TEXT, fontfamily="monospace")
ax.text(6.5, -3.0,
        '"how c responds to b"  ×  "how b responds to a"',
        fontsize=10, ha="center", color=SUBTLE, fontstyle="italic")


# ───────────────────────────────────────────────────────────
#  SECTION 2: Forward vs Backward on the Chain
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(-1, 15)
ax.set_ylim(-4.5, 5.5)

ax.text(0, 5.0, "Forward Pass vs Backward Pass on the Same Chain",
        fontsize=14, fontweight="bold", color=C_RULE)

# ── Forward row ──
fy = 2.5
ax.text(0, fy + 1.5, "Forward  →  compute VALUES", fontsize=12,
        fontweight="bold", color=C_J)

draw_node(ax, 2, fy, "a", C_I, r=0.5, fs=14)
draw_node(ax, 6.5, fy, "b", C_B, r=0.5, fs=14)
draw_node(ax, 11, fy, "c", C_OUT, r=0.5, fs=14)
draw_edge(ax, 2, fy, 6.5, fy, C_J, r=0.5)
draw_edge(ax, 6.5, fy, 11, fy, C_J, r=0.5)

ax.text(2, fy - 0.85, "val = 2", fontsize=9, color=C_J,
        ha="center", fontfamily="monospace", fontweight="bold")
ax.text(6.5, fy - 0.85, "val = 5", fontsize=9, color=C_J,
        ha="center", fontfamily="monospace", fontweight="bold")
ax.text(11, fy - 0.85, "val = 25", fontsize=9, color=C_J,
        ha="center", fontfamily="monospace", fontweight="bold")

ax.text(4.25, fy + 0.6, "b = a + 3", fontsize=9, color=C_J,
        ha="center", fontfamily="monospace")
ax.text(8.75, fy + 0.6, "c = b²", fontsize=9, color=C_J,
        ha="center", fontfamily="monospace")

# ── Backward row ──
by = -1.5
ax.text(0, by + 1.5, "Backward  ←  compute GRADIENTS", fontsize=12,
        fontweight="bold", color=C_K)

draw_node(ax, 2, by, "a", C_I, r=0.5, fs=14)
draw_node(ax, 6.5, by, "b", C_B, r=0.5, fs=14)
draw_node(ax, 11, by, "c", C_OUT, r=0.5, fs=14)

# Backward arrows (reversed)
draw_edge(ax, 11, by, 6.5, by, C_K, r=0.5)
draw_edge(ax, 6.5, by, 2, by, C_K, r=0.5)

ax.text(11, by + 0.85, "grad = 1", fontsize=9, color=C_K,
        ha="center", fontfamily="monospace", fontweight="bold")
ax.text(6.5, by + 0.85, "grad = 10", fontsize=9, color=C_K,
        ha="center", fontfamily="monospace", fontweight="bold")
ax.text(2, by + 0.85, "grad = 10", fontsize=9, color=C_K,
        ha="center", fontfamily="monospace", fontweight="bold")

# Labels on backward edges: what gets multiplied
edge_label(ax, 11, by, 6.5, by, "× 2b = ×10", C_K, offset=-0.55)
edge_label(ax, 6.5, by, 2, by, "× 1", C_K, offset=-0.55)

ax.text(8.75, by - 0.75, "dc/db = 2b = 10", fontsize=8.5,
        color=SUBTLE, ha="center", fontfamily="monospace")
ax.text(4.25, by - 0.75, "db/da = 1", fontsize=8.5,
        color=SUBTLE, ha="center", fontfamily="monospace")

# Directional context
ax.text(6.5, -3.7,
        "Forward: values propagate LEFT → RIGHT       "
        "Backward: gradients propagate RIGHT → LEFT",
        fontsize=10, ha="center", color=MEDIUM, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 3: The Pattern — Upstream × Local
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(-1, 15)
ax.set_ylim(-5.5, 6)

ax.text(0, 5.5, "The Pattern at Every Node", fontsize=14,
        fontweight="bold", color=C_RULE)
ax.text(0, 4.7,
        "Each node asks only ONE question: "
        '"I know how the output responds to ME — how do I respond to MY inputs?"',
        fontsize=10.5, color=MEDIUM)

# Generic diagram showing the pattern
# Parent → [this node] → child
parent_x, node_x, child_x = 2, 7, 12
mid_y = 2.0

draw_node(ax, parent_x, mid_y, "parent", C_OUT, r=0.7, fs=11)
draw_node(ax, node_x, mid_y, "this\nnode", C_B, r=0.8, fs=11)
draw_node(ax, child_x, mid_y, "child", C_I, r=0.7, fs=11)

# Forward edges
draw_edge(ax, child_x, mid_y, node_x, mid_y, C_J, r=0.75)
draw_edge(ax, node_x, mid_y, parent_x, mid_y, C_J, r=0.75)

# Backward flow annotations
# Upstream gradient arrives
ax.annotate("",
            xy=(node_x + 0.85, mid_y + 1.2),
            xytext=(parent_x + 0.2, mid_y + 1.2),
            arrowprops=dict(arrowstyle="->,head_width=0.25",
                            color=C_K, lw=2.5,
                            connectionstyle="arc3,rad=-0.15"))
ax.text((parent_x + node_x) / 2 + 0.3, mid_y + 1.8,
        "upstream gradient\narrives from parent",
        fontsize=10, ha="center", color=C_K, fontweight="bold")

ax.text(parent_x, mid_y - 1.1, "grad(parent)\n= known", fontsize=9,
        ha="center", color=C_K, fontfamily="monospace", fontweight="bold")

# Local derivative — within the node
ax.text(node_x, mid_y - 1.3,
        "local derivative\n∂(parent)/∂(this node)\n= computed from\nforward values",
        fontsize=8.5, ha="center", color=C_B, fontweight="bold")

# Result flows to child
ax.annotate("",
            xy=(child_x - 0.2, mid_y - 1.2),
            xytext=(node_x + 0.85, mid_y - 1.2),
            arrowprops=dict(arrowstyle="->,head_width=0.25",
                            color=C_K, lw=2.5,
                            connectionstyle="arc3,rad=-0.15"))
ax.text((node_x + child_x) / 2 + 0.3, mid_y - 1.8,
        "pass gradient\ndown to child",
        fontsize=10, ha="center", color=C_K, fontweight="bold")

ax.text(child_x, mid_y + 1.1, "grad(child) = ?", fontsize=9.5,
        ha="center", color=C_K, fontfamily="monospace", fontweight="bold")

# ── The formula ──
fy = -2.5
ax.add_patch(patches.FancyBboxPatch(
    (1.5, fy - 0.6), 11, 2.4, boxstyle="round,pad=0.2",
    facecolor=C_OUT, alpha=0.08, edgecolor=C_OUT, linewidth=2))

ax.text(7, fy + 1.3, "The universal backward formula:", fontsize=12,
        ha="center", fontweight="bold", color=C_OUT)

ax.text(7, fy + 0.4, "grad(child)  =  grad(parent)  ×  local derivative",
        fontsize=14, ha="center", fontweight="bold", color=C_OUT,
        fontfamily="monospace")

ax.text(3.8, fy - 0.35, "\"how does the", fontsize=9.5,
        ha="center", color=C_K, fontstyle="italic")
ax.text(3.8, fy - 0.75, "root respond", fontsize=9.5,
        ha="center", color=C_K, fontstyle="italic")
ax.text(3.8, fy - 1.15, "to parent?\"", fontsize=9.5,
        ha="center", color=C_K, fontstyle="italic")

ax.text(7.6, fy - 0.75, "×", fontsize=14, ha="center",
        color=TEXT, fontweight="bold")

ax.text(10.2, fy - 0.35, "\"how does this", fontsize=9.5,
        ha="center", color=C_B, fontstyle="italic")
ax.text(10.2, fy - 0.75, "node respond", fontsize=9.5,
        ha="center", color=C_B, fontstyle="italic")
ax.text(10.2, fy - 1.15, "to its child?\"", fontsize=9.5,
        ha="center", color=C_B, fontstyle="italic")


# ───────────────────────────────────────────────────────────
#  SECTION 4: Longer chain — shows accumulation
# ───────────────────────────────────────────────────────────
ax = axes[3]
ax.set_xlim(-1, 15)
ax.set_ylim(-5.5, 5.5)

ax.text(0, 5.0, "Longer Chain:  a → b → c → d", fontsize=14,
        fontweight="bold", color=C_RULE)
ax.text(0, 4.2,
        "The chain rule COMPOSES — each step multiplies one more local derivative.",
        fontsize=11, color=MEDIUM)

# Nodes
positions = [(1.5, 1.5), (5, 1.5), (8.5, 1.5), (12, 1.5)]
labels = ["a", "b", "c", "d"]
colors = [C_I, C_B, C_B, C_OUT]
r = 0.55

for (x, y), lbl, col in zip(positions, labels, colors):
    draw_node(ax, x, y, lbl, col, r=r, fs=15)

# Forward edges with local derivatives
for i in range(3):
    draw_edge(ax, *positions[i], *positions[i+1], C_J, r=r)

edge_label(ax, *positions[0], *positions[1], "db/da", C_J, offset=0.5)
edge_label(ax, *positions[1], *positions[2], "dc/db", C_J, offset=0.5)
edge_label(ax, *positions[2], *positions[3], "dd/dc", C_J, offset=0.5)

# ── Show gradient at each node ──
grads = [
    ("d", "grad(d) = 1", "start here"),
    ("c", "grad(c) = 1 × dd/dc", "= dd/dc"),
    ("b", "grad(b) = 1 × dd/dc × dc/db", "two factors"),
    ("a", "grad(a) = 1 × dd/dc × dc/db × db/da", "three factors"),
]

ty = -0.5
for idx, (node, formula, note) in enumerate(grads):
    y = ty - idx * 1.0
    col = colors[3 - idx]

    ax.text(1.0, y, f"grad({node})", fontsize=11,
            fontfamily="monospace", color=C_K, fontweight="bold")
    ax.text(3.8, y, "=", fontsize=11, color=TEXT,
            fontfamily="monospace")

    # Build the product chain with coloring
    if idx == 0:
        ax.text(4.3, y, "1", fontsize=12, fontfamily="monospace",
                color=C_OUT, fontweight="bold")
        ax.text(7.5, y, "← root always starts at 1", fontsize=9,
                color=SUBTLE, fontstyle="italic")
    elif idx == 1:
        ax.text(4.3, y, "1  ×  dd/dc", fontsize=11,
                fontfamily="monospace", color=TEXT)
        # Highlight the new factor
        ax.add_patch(patches.FancyBboxPatch(
            (6.2, y - 0.2), 1.8, 0.45, boxstyle="round,pad=0.05",
            facecolor=C_K, alpha=0.12, edgecolor=C_K, linewidth=1,
            linestyle="--"))
        ax.text(9.5, y, "← 1 new local derivative", fontsize=9,
                color=SUBTLE, fontstyle="italic")
    elif idx == 2:
        ax.text(4.3, y, "1  ×  dd/dc  ×  dc/db", fontsize=11,
                fontfamily="monospace", color=TEXT)
        ax.add_patch(patches.FancyBboxPatch(
            (8.7, y - 0.2), 1.8, 0.45, boxstyle="round,pad=0.05",
            facecolor=C_K, alpha=0.12, edgecolor=C_K, linewidth=1,
            linestyle="--"))
        ax.text(11.5, y, "← another one", fontsize=9,
                color=SUBTLE, fontstyle="italic")
    else:
        ax.text(4.3, y, "1  ×  dd/dc  ×  dc/db  ×  db/da", fontsize=11,
                fontfamily="monospace", color=TEXT)
        ax.add_patch(patches.FancyBboxPatch(
            (11.2, y - 0.2), 1.8, 0.45, boxstyle="round,pad=0.05",
            facecolor=C_K, alpha=0.12, edgecolor=C_K, linewidth=1,
            linestyle="--"))

# Key insight
ax.text(7, -5.0,
        "Each backward step multiplies ONE more local derivative onto the running product.",
        fontsize=11, ha="center", fontweight="bold", color=C_OUT)


# ───────────────────────────────────────────────────────────
#  SECTION 5: Summary
# ───────────────────────────────────────────────────────────
ax = axes[4]
ax.set_xlim(0, 14)
ax.set_ylim(-3, 4)

ax.text(0.5, 3.5, "Summary", fontsize=14,
        fontweight="bold", color=C_RULE)

ax.text(0.5, 2.5, "The chain rule in calculus:", fontsize=12,
        fontweight="bold", color=MEDIUM)
ax.text(0.5, 1.8,
        "dc/da  =  dc/db  ×  db/da", fontsize=14,
        fontfamily="monospace", color=C_OUT, fontweight="bold")
ax.text(0.5, 1.1,
        "Backprop is this same rule, applied one edge at a time, from root to leaves.",
        fontsize=11, color=MEDIUM)

insight_box(ax, 1.0, -2.5, [
    ("The chain rule says: to get the end-to-end sensitivity, multiply the local sensitivities", True),
    ("Backprop organizes this as a traversal: start at root (grad=1), walk backward", True),
    ("At each edge: multiply the running gradient by one local derivative", False),
    ("Each node only needs to know its own local rule — it never sees the full chain", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/chain_rule_graph.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
