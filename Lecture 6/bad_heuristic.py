"""
Lecture 6 Visual: Not Every Heuristic Works

Shows the lecture's counterexample where h(C) = 1000 sabotages A*.
The huge heuristic value creates a negative modified cost on edge C→D,
which breaks UCS's correctness guarantee.
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


def node(ax, cx, cy, label, color=C_NODE, r=0.42, fs=16, lw=2.5,
         alpha_f=0.14):
    c = plt.Circle((cx, cy), r, facecolor=color, alpha=alpha_f,
                    edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(c)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color, zorder=6)


def edge(ax, x1, y1, x2, y2, color, lw=2.0, alpha=0.6,
         shrA=22, shrB=22):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                alpha=alpha, shrinkA=shrA, shrinkB=shrB),
                zorder=3)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 36))
fig.suptitle("Not Every Heuristic Works",
             fontsize=18, fontweight="bold", y=0.998, color=C_BOUND)

gs = GridSpec(5, 1, figure=fig,
              height_ratios=[5.5, 4.0, 5.0, 4.0, 1.8],
              hspace=0.12,
              top=0.975, bottom=0.01, left=0.04, right=0.96)

# Graph layout — diamond: A left, B top, C bottom, D right
gA = (3.5, 4.0)
gB = (8.0, 6.5)
gC = (8.0, 1.5)
gD = (12.5, 4.0)


# ─────────────────────────────────────────────────────────
#  ROW 0: The Graph with Original Costs and Heuristic
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16); ax0.set_ylim(-2.5, 9)
ax0.set_title("  The Counterexample Graph",
              fontsize=14, fontweight="bold", color=C_NODE, pad=10, loc="left")

box(ax0, (0.5, -2.0), 15, 10.5, C_NODE, alpha_f=0.03, lw=1.2)

# Nodes
node(ax0, *gA, "A", C_POS)
node(ax0, *gB, "B", C_NODE)
node(ax0, *gC, "C", C_NEG)
node(ax0, *gD, "D", C_BOUND)

# Labels
ax0.text(gA[0], gA[1] - 0.75, "start", fontsize=9, ha="center",
         color=C_POS, fontweight="bold")
ax0.text(gD[0], gD[1] - 0.75, "end", fontsize=9, ha="center",
         color=C_BOUND, fontweight="bold")

# Edges with original costs
edge(ax0, *gA, *gB, C_I)
edge(ax0, *gA, *gC, C_I)
edge(ax0, *gB, *gD, C_I)
edge(ax0, *gC, *gD, C_I)

# Cost labels on edges
ax0.text(5.3, 5.8, "cost = 1", fontsize=11, ha="center", color=C_I,
         fontweight="bold", rotation=25,
         bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax0.text(5.3, 2.3, "cost = 2", fontsize=11, ha="center", color=C_I,
         fontweight="bold", rotation=-25,
         bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax0.text(10.7, 5.8, "cost = 5", fontsize=11, ha="center", color=C_I,
         fontweight="bold", rotation=-25,
         bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax0.text(10.7, 2.3, "cost = 1", fontsize=11, ha="center", color=C_I,
         fontweight="bold", rotation=25,
         bbox=dict(facecolor=BG, edgecolor="none", pad=2))

# Heuristic values next to nodes
h_vals = {"A": 0, "B": 0, "C": 1000, "D": 0}
h_positions = [(gA[0] - 0.9, gA[1] + 0.7),
               (gB[0] + 0.9, gB[1] + 0.4),
               (gC[0] + 0.9, gC[1] - 0.4),
               (gD[0] + 0.9, gD[1] + 0.7)]
h_colors = [C_POS, C_NODE, C_NEG, C_BOUND]

for (hx, hy), (name, val), c in zip(h_positions, h_vals.items(), h_colors):
    color = C_NEG if val == 1000 else C_B
    fs = 13 if val == 1000 else 11
    ax0.text(hx, hy, f"h({name}) = {val}", fontsize=fs, ha="center",
             color=color, fontweight="bold",
             bbox=dict(facecolor=BG, edgecolor=color, pad=3,
                       alpha=0.9, boxstyle="round,pad=0.15") if val == 1000
             else dict(facecolor=BG, edgecolor="none", pad=2))

# Optimal path note
ax0.text(8, -0.7, "True optimal:  A → C → D   (cost 2 + 1 = 3)",
         fontsize=12, ha="center", color=C_POS, fontweight="bold")
ax0.text(8, -1.5, "Suboptimal:    A → B → D   (cost 1 + 5 = 6)",
         fontsize=12, ha="center", color=MEDIUM)


# ─────────────────────────────────────────────────────────
#  ROW 1: Modified Costs — The Problem
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(0, 16); ax1.set_ylim(-3.5, 6.5)
ax1.set_title("  Modified Costs with the Bad Heuristic",
              fontsize=14, fontweight="bold", color=C_RULE, pad=10, loc="left")

box(ax1, (0.5, 3.5), 15, 2.5, C_RULE, alpha_f=0.06, lw=1.2)
ax1.text(8, 5.5, "c'(s, a) = c(s, a) + h(s') − h(s)",
         fontsize=13, ha="center", fontfamily="monospace",
         color=C_RULE, fontweight="bold")
ax1.text(8, 4.2, "Apply to each edge:",
         fontsize=11, ha="center", color=MEDIUM)

# Modified cost computations
computations = [
    ("A → B:", "1 + h(B) − h(A)", "= 1 + 0 − 0",   "= 1",    C_POS),
    ("B → D:", "5 + h(D) − h(B)", "= 5 + 0 − 0",   "= 5",    C_POS),
    ("A → C:", "2 + h(C) − h(A)", "= 2 + 1000 − 0", "= 1002", C_RULE),
    ("C → D:", "1 + h(D) − h(C)", "= 1 + 0 − 1000", "= −999", C_NEG),
]

for i, (edge_lbl, formula, calc, result, c) in enumerate(computations):
    y = 2.7 - i * 1.1
    ax1.text(1.5, y, edge_lbl, fontsize=11, color=c, fontweight="bold")
    ax1.text(4.0, y, formula, fontsize=10.5, fontfamily="monospace",
             color=MEDIUM)
    ax1.text(9.5, y, calc, fontsize=10.5, fontfamily="monospace",
             color=MEDIUM)
    ax1.text(13.5, y, result, fontsize=12, fontfamily="monospace",
             color=c, fontweight="bold")

    # Highlight the negative one
    if "−999" in result:
        box(ax1, (12.8, y - 0.35), 2.5, 0.7, C_NEG, alpha_f=0.15, lw=2.0)
        ax1.text(15.5, y, "NEGATIVE!", fontsize=10, color=C_NEG,
                 fontweight="bold", va="center")


# ─────────────────────────────────────────────────────────
#  ROW 2: Why This Breaks — The Modified Graph
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(-0.5, 16.5); ax2.set_ylim(-4, 9)
ax2.set_title("  What A* Actually Does with This Bad Heuristic",
              fontsize=14, fontweight="bold", color=C_NEG, pad=10, loc="left")

# ── LEFT: Modified cost graph ──
box(ax2, (-0.3, -0.5), 8.1, 9.0, C_NEG, alpha_f=0.03, lw=1.2)
ax2.text(3.75, 8.0, "Modified Cost Graph", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)

# Smaller diamond
mA = (1.8, 4.5)
mB = (3.75, 6.5)
mC = (3.75, 2.5)
mD = (5.7, 4.5)

node(ax2, *mA, "A", C_POS, r=0.35, fs=14)
node(ax2, *mB, "B", C_NODE, r=0.35, fs=14)
node(ax2, *mC, "C", C_NEG, r=0.35, fs=14)
node(ax2, *mD, "D", C_BOUND, r=0.35, fs=14)

edge(ax2, *mA, *mB, C_POS, shrA=18, shrB=18)
edge(ax2, *mA, *mC, C_RULE, shrA=18, shrB=18)
edge(ax2, *mB, *mD, C_POS, shrA=18, shrB=18)
edge(ax2, *mC, *mD, C_NEG, shrA=18, shrB=18)

# Modified cost labels
ax2.text(2.4, 5.9, "1", fontsize=12, ha="center", color=C_POS,
         fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax2.text(2.4, 3.1, "1002", fontsize=12, ha="center", color=C_RULE,
         fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax2.text(5.1, 5.9, "5", fontsize=12, ha="center", color=C_POS,
         fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax2.text(5.1, 3.1, "−999", fontsize=12, ha="center", color=C_NEG,
         fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor=C_NEG, pad=3, alpha=0.9,
                   boxstyle="round,pad=0.12"))

ax2.text(3.75, 0.8, "UCS on this graph is broken", fontsize=11,
         ha="center", color=C_NEG, fontweight="bold")
ax2.text(3.75, 0.1, "because of the negative edge.", fontsize=10,
         ha="center", color=MEDIUM)

# ── RIGHT: What goes wrong ──
box(ax2, (8.5, -0.5), 8.0, 9.0, C_RULE, alpha_f=0.03, lw=1.2)
ax2.text(12.5, 8.0, "What Goes Wrong", fontsize=13, ha="center",
         fontweight="bold", color=C_RULE)

steps = [
    ("1.", "A* pops A (mod cost 0)", C_POS),
    ("",   "Expands: B with mod cost 1", C_NODE),
    ("",   "Expands: C with mod cost 1002", C_RULE),
    ("2.", "A* pops B (mod cost 1)", C_NODE),
    ("",   "Expands: D with mod cost 1 + 5 = 6", C_NODE),
    ("3.", "A* pops D (mod cost 6) ← END", C_BOUND),
    ("",   "Returns path A → B → D", C_BOUND),
    ("",   "", ""),
    ("",   "A* never even reaches C → D !", C_NEG),
    ("",   "The −999 edge was never used.", C_NEG),
]

for i, (num, desc, c) in enumerate(steps):
    if not desc:
        continue
    y = 7.0 - i * 0.7
    if num:
        ax2.text(9.0, y, num, fontsize=10.5, color=c, fontweight="bold")
    ax2.text(9.6, y, desc, fontsize=10, color=c,
             fontweight="bold" if num else "normal",
             fontfamily="monospace" if num else "sans-serif")

# Result comparison
box(ax2, (8.8, -3.5), 7.4, 3.0, C_NEG, alpha_f=0.06, lw=1.5)
ax2.text(12.5, -1.0, "A* returns:  A → B → D   (cost 6)", fontsize=11,
         ha="center", color=C_NEG, fontweight="bold")
ax2.text(12.5, -1.8, "True optimum:  A → C → D   (cost 3)", fontsize=11,
         ha="center", color=C_POS, fontweight="bold")
ax2.text(12.5, -2.7, "A* gave the WRONG answer!", fontsize=12,
         ha="center", color=C_NEG, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 3: The Root Cause — Consistency Violated
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16); ax3.set_ylim(-3.5, 7)
ax3.set_title("  The Root Cause: Consistency Is Violated",
              fontsize=14, fontweight="bold", color=C_B, pad=10, loc="left")

box(ax3, (0.5, 4.0), 15, 2.5, C_B, alpha_f=0.06, lw=1.5)
ax3.text(8, 6.0, "Consistency requires:   h(s) ≤ c(s,a) + h(s')",
         fontsize=13, ha="center", fontfamily="monospace",
         color=C_B, fontweight="bold")
ax3.text(8, 5.0, "Equivalently:   c(s,a) + h(s') − h(s) ≥ 0",
         fontsize=12, ha="center", fontfamily="monospace",
         color=C_B)
ax3.text(8, 4.3, "(all modified costs must be non-negative)",
         fontsize=10.5, ha="center", color=MEDIUM, fontstyle="italic")

# Check each edge
checks = [
    ("A → B:", "h(A) ≤ c(A,B) + h(B)", "0 ≤ 1 + 0 = 1",     "✓", C_POS),
    ("B → D:", "h(B) ≤ c(B,D) + h(D)", "0 ≤ 5 + 0 = 5",     "✓", C_POS),
    ("A → C:", "h(A) ≤ c(A,C) + h(C)", "0 ≤ 2 + 1000 = 1002","✓", C_POS),
    ("C → D:", "h(C) ≤ c(C,D) + h(D)", "1000 ≤ 1 + 0 = 1",  "✗ VIOLATED", C_NEG),
]

for i, (edge_lbl, condition, calc, result, c) in enumerate(checks):
    y = 3.0 - i * 1.0
    ax3.text(1.0, y, edge_lbl, fontsize=11, color=c, fontweight="bold")
    ax3.text(3.5, y, condition, fontsize=10.5, fontfamily="monospace",
             color=MEDIUM)
    ax3.text(9.0, y, calc, fontsize=10.5, fontfamily="monospace", color=c)
    ax3.text(14.0, y, result, fontsize=11, color=c, fontweight="bold")

    if "VIOLATED" in result:
        box(ax3, (0.7, y - 0.4), 14.8, 0.85, C_NEG, alpha_f=0.08, lw=2.0)

# Bottom explanation
box(ax3, (0.5, -3.2), 15, 2.5, C_RULE, alpha_f=0.06, lw=1.5)
ax3.text(8, -1.2,
    "h(C) = 1000 wildly overestimates the true future cost from C  (which is just 1).",
    fontsize=10.5, ha="center", color=C_RULE, fontweight="bold")
ax3.text(8, -2.0,
    "This makes the modified cost on C → D negative, breaking UCS's guarantee.",
    fontsize=10.5, ha="center", color=C_RULE)
ax3.text(8, -2.8,
    "Lesson: A* is not \"use any heuristic and hope.\"  Consistency is required.",
    fontsize=11, ha="center", color=C_BOUND, fontweight="bold")


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
    ("A bad heuristic can produce negative modified costs.", True),
    ("UCS requires non-negative costs — negative edges break its \"finalize when popped\" logic.", False),
    ("In this example, h(C) = 1000 makes A* avoid C entirely and return the wrong answer.", True),
    ("The fix: require consistency —  h(s) ≤ c(s,a) + h(s')  for every edge.", False),
    ("Consistency guarantees all modified costs are non-negative, so UCS works correctly.", True),
    ("This is why heuristics must be designed carefully, not guessed.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax4.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ─────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/Lecture 6/bad_heuristic.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
