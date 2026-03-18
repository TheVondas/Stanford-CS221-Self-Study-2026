"""
Lecture 4 Visual: Computation Graphs — Values vs Gradient Flow

Shows:
1. Connected computation graph: forward values + backward gradients flow freely
2. Chain rule breakdown: how each gradient value is computed step by step
3. Detached graph: same forward values, broken gradient chain
4. Spreadsheet analogy + key takeaway
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ── Style (matching lecture_3_visuals) ──────────────────
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

def draw_node(ax, cx, cy, name, value, color=C_NODE, w=2.4, h=1.2):
    """Rounded-rectangle node with name + value."""
    rect = patches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.12",
        facecolor=color, alpha=0.10, edgecolor=color, linewidth=2, zorder=5)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.18, name, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=color, zorder=6)
    ax.text(cx, cy - 0.22, value, ha="center", va="center",
            fontsize=10, fontfamily="monospace", color=MEDIUM, zorder=6)


def draw_edge(ax, x1, y1, x2, y2, color, label="", label_y=0.35,
              lw=2.0, style="-|>", sA=55, sB=55, ls="-"):
    """Arrow from (x1,y1) -> (x2,y2) with optional midpoint label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        shrinkA=sA, shrinkB=sB, linestyle=ls),
        zorder=3)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + label_y
        ax.text(mx, my, label, ha="center", va="center", fontsize=9,
                color=color, fontweight="bold",
                bbox=dict(facecolor=BG, edgecolor="none", pad=2), zorder=4)


def draw_x_mark(ax, cx, cy, s=0.35):
    """Red X marking blocked gradient flow."""
    ax.plot([cx-s, cx+s], [cy-s, cy+s], color=C_NEG, lw=4, zorder=7)
    ax.plot([cx-s, cx+s], [cy+s, cy-s], color=C_NEG, lw=4, zorder=7)


# ── Node positions ──────────────────────────────────────
POS = {
    'a': (2.0, 1.8), 'b': (2.0, -1.8),
    'c': (6.5, 0.0), 'd': (10.5, 0.0), 'L': (14.5, 0.0),
}


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 30))
fig.suptitle("Computation Graphs: Values vs Gradient Flow",
             fontsize=18, fontweight="bold", y=0.995, color=C_BOUND)

gs = GridSpec(5, 1, figure=fig,
              height_ratios=[0.7, 4.5, 4.0, 2.5, 1.8],
              hspace=0.15,
              top=0.97, bottom=0.02, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: Intro
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(0, 2)

ax0.text(8, 1.5,
    "Every value in a computation graph plays two roles:",
    fontsize=13, ha="center")
ax0.text(8, 0.85,
    "1.  It carries a numerical value  (forward pass)",
    fontsize=12, ha="center", color=C_I, fontweight="bold")
ax0.text(8, 0.25,
    "2.  It may sit in a dependency chain for gradients  (backward pass)",
    fontsize=12, ha="center", color=C_NEG, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 1: Connected Graph + Chain Rule
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(-0.5, 16.5); ax1.set_ylim(-7.0, 5.0)
ax1.set_title("  Connected Graph — Gradients Flow All the Way Back",
              fontsize=14, fontweight="bold", color=C_POS, pad=10, loc="left")

# ---- Nodes ----
draw_node(ax1, *POS['a'], "a", "= 2", C_I)
draw_node(ax1, *POS['b'], "b", "= 3", C_I)
draw_node(ax1, *POS['c'], "c = a × b", "= 6", C_NODE)
draw_node(ax1, *POS['d'], "d = c + 1", "= 7", C_NODE)
draw_node(ax1, *POS['L'], "Loss = d²", "= 49", C_BOUND)

# ---- Forward arrows (blue, slightly above center line) ----
draw_edge(ax1, 2.0, 2.05, 6.5, 0.25, C_I, sA=40, sB=40)       # a -> c
draw_edge(ax1, 2.0, -2.05, 6.5, -0.25, C_I, sA=40, sB=40)     # b -> c
draw_edge(ax1, 6.5, 0.2, 10.5, 0.2, C_I, sA=55, sB=55)        # c -> d
draw_edge(ax1, 10.5, 0.2, 14.5, 0.2, C_I, sA=55, sB=55)       # d -> Loss

# Forward operation labels
ax1.text(3.8, 1.9, "× b", fontsize=9, color=C_I, fontweight="bold",
         ha="center", bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax1.text(3.8, -1.9, "× a", fontsize=9, color=C_I, fontweight="bold",
         ha="center", bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax1.text(8.5, 0.55, "+ 1", fontsize=9, color=C_I, fontweight="bold",
         ha="center", bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax1.text(12.5, 0.55, "( )²", fontsize=9, color=C_I, fontweight="bold",
         ha="center", bbox=dict(facecolor=BG, edgecolor="none", pad=2))

# ---- Backward arrows (red, slightly below center line) ----
draw_edge(ax1, 14.5, -0.2, 10.5, -0.2, C_NEG,
          "∂L/∂d = 14", -0.55, sA=55, sB=55)                   # Loss -> d
draw_edge(ax1, 10.5, -0.2, 6.5, -0.2, C_NEG,
          "∂L/∂c = 14", -0.55, sA=55, sB=55)                   # d -> c
draw_edge(ax1, 6.5, -0.25, 2.0, 1.55, C_NEG,
          "∂L/∂a = 42", -0.5, sA=40, sB=40)                    # c -> a
draw_edge(ax1, 6.5, 0.25, 2.0, -1.55, C_NEG,
          "∂L/∂b = 28", 0.5, sA=40, sB=40)                     # c -> b

# Legend
ax1.text(1, 4.3, "→  Blue: forward (values)",
         fontsize=10.5, color=C_I, fontweight="bold")
ax1.text(9, 4.3, "←  Red: backward (gradients)",
         fontsize=10.5, color=C_NEG, fontweight="bold")

# Green bar
ax1.add_patch(patches.FancyBboxPatch(
    (0, -2.8), 16.5, 0.7, boxstyle="round,pad=0.1",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1))
ax1.text(8.25, -2.45,
    "All parameters (a, b) receive gradient updates and can learn",
    fontsize=11, ha="center", color=C_POS, fontweight="bold")

# ---- Chain rule explanation ----
ax1.add_patch(patches.FancyBboxPatch(
    (0.5, -6.8), 15.5, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.06, edgecolor=C_RULE, linewidth=1.5))
ax1.text(8.25, -3.7, "How Are the Gradients Computed?  (Chain Rule)",
         fontsize=12, ha="center", fontweight="bold", color=C_RULE)

chain = [
    "∂Loss/∂d  =  2d  =  2(7)  =  14",
    "∂Loss/∂c  =  (∂Loss/∂d)(∂d/∂c)  =  14 × 1  =  14",
    "∂Loss/∂a  =  (∂Loss/∂c)(∂c/∂a)  =  14 × b  =  14 × 3  =  42",
    "∂Loss/∂b  =  (∂Loss/∂c)(∂c/∂b)  =  14 × a  =  14 × 2  =  28",
]
for i, txt in enumerate(chain):
    ax1.text(8.25, -4.5 - i * 0.6, txt, fontsize=10, ha="center",
             fontfamily="monospace", color=C_NEG)

ax1.text(8.25, -6.5,
    "Each gradient flows backward by multiplying local derivatives at every step.",
    fontsize=10, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 2: Detached Graph
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(-0.5, 16.5); ax2.set_ylim(-4.5, 5.0)
ax2.set_title("  Detached Graph — Same Values, Broken Gradient Chain",
              fontsize=14, fontweight="bold", color=C_NEG, pad=10, loc="left")

# ---- Nodes (a, b, c grayed out to show they are cut off) ----
draw_node(ax2, *POS['a'], "a", "= 2", C_DIM)
draw_node(ax2, *POS['b'], "b", "= 3", C_DIM)
draw_node(ax2, *POS['c'], "c = a × b", "= 6", C_DIM)
draw_node(ax2, *POS['d'], "d = c + 1", "= 7", C_NODE)
draw_node(ax2, *POS['L'], "Loss = d²", "= 49", C_BOUND)

# DETACH marker (low zorder so arrows pass over it)
det_x = 8.5
ax2.add_patch(patches.FancyBboxPatch(
    (det_x - 1.0, -0.45), 2.0, 0.9, boxstyle="round,pad=0.1",
    facecolor=C_NEG, alpha=0.12, edgecolor=C_NEG, linewidth=2.5, zorder=2))
ax2.text(det_x, 0.0, "DETACH", ha="center", va="center",
         fontsize=10.5, fontweight="bold", color=C_NEG, zorder=10)

# ---- Forward arrows (blue — values still flow!) ----
draw_edge(ax2, 2.0, 2.05, 6.5, 0.25, C_I, sA=40, sB=40)       # a -> c
draw_edge(ax2, 2.0, -2.05, 6.5, -0.25, C_I, sA=40, sB=40)     # b -> c
draw_edge(ax2, 6.5, 0.2, 10.5, 0.2, C_I, sA=55, sB=55)        # c -> d (over detach)
draw_edge(ax2, 10.5, 0.2, 14.5, 0.2, C_I, sA=55, sB=55)       # d -> Loss

ax2.text(det_x, 0.85, "value 6 passes through",
         fontsize=9, ha="center", color=C_I, fontstyle="italic")

ax2.text(3.8, 1.9, "× b", fontsize=9, color=C_I, fontweight="bold",
         ha="center", bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax2.text(3.8, -1.9, "× a", fontsize=9, color=C_I, fontweight="bold",
         ha="center", bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax2.text(12.5, 0.55, "( )²", fontsize=9, color=C_I, fontweight="bold",
         ha="center", bbox=dict(facecolor=BG, edgecolor="none", pad=2))

# Legend
ax2.text(1, 4.3, "→  Blue: forward (values still flow)",
         fontsize=10.5, color=C_I, fontweight="bold")
ax2.text(9, 4.3, "←  Red / Gray: backward (BROKEN)",
         fontsize=10.5, color=C_NEG, fontweight="bold")

# ---- Backward: Loss -> d (works, solid red) ----
draw_edge(ax2, 14.5, -0.2, 10.5, -0.2, C_NEG,
          "∂L/∂d = 14", -0.55, sA=55, sB=55)

# ---- Backward: d -> c direction (blocked at DETACH, gray dashed) ----
draw_edge(ax2, 10.5, -0.2, 6.5, -0.2, C_DIM, "", 0, sA=55, sB=55, ls="--")
draw_x_mark(ax2, det_x, -0.35, s=0.3)
ax2.text(det_x, -1.1, "gradient blocked!",
         fontsize=10, ha="center", color=C_NEG, fontweight="bold")

# ---- Blocked: c -> a, c -> b (gray dashed) ----
draw_edge(ax2, 6.5, -0.25, 2.0, 1.55, C_DIM,
          "∂L/∂a = 0", -0.4, sA=40, sB=40, ls="--")
draw_edge(ax2, 6.5, 0.25, 2.0, -1.55, C_DIM,
          "∂L/∂b = 0", 0.4, sA=40, sB=40, ls="--")

# Red bar
ax2.add_patch(patches.FancyBboxPatch(
    (0, -3.5), 16.5, 0.7, boxstyle="round,pad=0.1",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1))
ax2.text(8.25, -3.15,
    "Parameters a and b receive NO gradient — they cannot learn from this path",
    fontsize=11, ha="center", color=C_NEG, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 3: Analogy
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-2, 5)
ax3.set_title("  Analogy: The Spreadsheet",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

# Connected = cell reference
ax3.add_patch(patches.FancyBboxPatch(
    (0.3, 0.5), 7.2, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_POS, alpha=0.06, edgecolor=C_POS, linewidth=1.5))
ax3.text(3.9, 3.5, "Connected = Cell Reference",
         fontsize=12, ha="center", fontweight="bold", color=C_POS)
ax3.text(3.9, 2.7, "Cell B has formula:  = A1 × 3",
         fontsize=10.5, ha="center", fontfamily="monospace", color=TEXT)
ax3.text(3.9, 2.0, "Change A1  →  B updates automatically.",
         fontsize=10.5, ha="center", color=MEDIUM)
ax3.text(3.9, 1.2, "The dependency is alive.",
         fontsize=11, ha="center", fontweight="bold", color=C_POS)

# Detached = copy-paste
ax3.add_patch(patches.FancyBboxPatch(
    (8.5, 0.5), 7.2, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_NEG, alpha=0.06, edgecolor=C_NEG, linewidth=1.5))
ax3.text(12.1, 3.5, "Detached = Copy-Paste Value",
         fontsize=12, ha="center", fontweight="bold", color=C_NEG)
ax3.text(12.1, 2.7, "You type A1's number into B manually.",
         fontsize=10.5, ha="center", color=TEXT)
ax3.text(12.1, 2.0, "B has the same number, no link to A1.",
         fontsize=10.5, ha="center", color=MEDIUM)
ax3.text(12.1, 1.2, "The dependency is broken.",
         fontsize=11, ha="center", fontweight="bold", color=C_NEG)

ax3.text(8, -0.3, "Same number.  Different causal relationship.",
         fontsize=12, ha="center", color=C_BOUND, fontweight="bold")
ax3.text(8, -1.2,
    "That is the meaning of detaching: same value, severed chain.",
    fontsize=11, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 4: Key Takeaway
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4])
ax4.axis("off")
ax4.set_xlim(0, 16); ax4.set_ylim(-3, 3)

ax4.add_patch(patches.FancyBboxPatch(
    (1.0, -2.2), 14, 4.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax4.text(8, 1.8, "Key Takeaway", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

takeaways = [
    ("Backpropagation only works along preserved graph connections.", True),
    ("Detaching copies the number but severs the gradient path.", True),
    ("The forward result is identical — the backward behavior", False),
    ("is completely different.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.05 - i * 0.55, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
           "/lecture_4_visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "computation_graph_flow.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
