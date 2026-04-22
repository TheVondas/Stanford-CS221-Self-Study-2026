"""
Lecture 7 Visual: The Flaky Tram MDP (Example 1)

Shows:
1. Formal MDP specification (states, actions, transitions, rewards, gamma)
2. "Markov tree" view of a single state i=3 — how an action in an MDP
   produces a *distribution* over next states, via the alternation of
   decision (state) nodes, action nodes, and chance nodes.
3. Full state graph (1..10) with all walk edges and stochastic tram forks.
4. Key takeaway — the conceptual leap from search to MDPs.
"""

import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ── Style (matching lecture series) ──────────────────────
BG      = "#1e1e2e"
TEXT    = "#cdd6f4"
SUBTLE  = "#6c7086"
MEDIUM  = "#a6adc8"
C_POS   = "#a6e3a1"   # green  — walk / deterministic
C_NEG   = "#f38ba8"   # pink   — tram failure branch
C_BOUND = "#f9e2af"   # yellow — goal / highlights
C_RULE  = "#fab387"   # orange — key insights / action nodes
C_I     = "#89b4fa"   # blue   — tram / stochastic info
C_B     = "#cba6f7"   # purple — chance nodes
C_DIM   = "#585b70"
C_NODE  = "#89dceb"   # teal   — state nodes

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Helpers ─────────────────────────────────────────────

def draw_state(ax, cx, cy, label, color=C_NODE, radius=0.42,
               fontsize=14, lw=2.2, alpha_fill=0.15):
    """State (decision) node — circle."""
    circle = plt.Circle((cx, cy), radius, facecolor=color, alpha=alpha_fill,
                        edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(circle)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=6)


def draw_action(ax, cx, cy, label, color=C_RULE, w=1.25, h=0.55, fontsize=11):
    """Action node — rounded rectangle."""
    ax.add_patch(patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06", facecolor=color, alpha=0.15,
        edgecolor=color, linewidth=1.8, zorder=5))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=6)


def draw_chance(ax, cx, cy, color=C_B, size=0.22):
    """Chance node — small diamond."""
    diamond = patches.RegularPolygon(
        (cx, cy), numVertices=4, radius=size, orientation=0,
        facecolor=color, alpha=0.85, edgecolor=color, linewidth=1.5, zorder=6)
    ax.add_patch(diamond)


def draw_edge(ax, x1, y1, x2, y2, color=TEXT, lw=2.0, alpha=0.9,
              shrink_from=0.42, shrink_to=0.42, connectionstyle="arc3",
              style="-|>"):
    """Arrow from (x1,y1) to (x2,y2), shrunk by the node radii."""
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    if d == 0:
        return
    ux, uy = dx / d, dy / d
    sx, sy = x1 + ux * shrink_from, y1 + uy * shrink_from
    ex, ey = x2 - ux * shrink_to, y2 - uy * shrink_to
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                alpha=alpha,
                                connectionstyle=connectionstyle), zorder=3)


def edge_label(ax, x1, y1, x2, y2, text, color, offset_perp=0.0, fontsize=9,
               boxed=True, fontweight="bold"):
    """Place a label near the midpoint of an edge, optionally nudged perpendicular."""
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    if offset_perp:
        dx = x2 - x1
        dy = y2 - y1
        d = math.hypot(dx, dy)
        if d:
            # perpendicular unit vector
            px, py = -dy / d, dx / d
            mx += px * offset_perp
            my += py * offset_perp
    kwargs = dict(fontsize=fontsize, ha="center", va="center",
                  color=color, fontweight=fontweight)
    if boxed:
        kwargs["bbox"] = dict(facecolor=BG, edgecolor="none", pad=1.5)
    ax.text(mx, my, text, **kwargs)


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 28))
fig.suptitle("The Flaky Tram MDP  —  Example 1",
             fontsize=18, fontweight="bold", y=0.995, color=C_BOUND)

gs = GridSpec(4, 1, figure=fig,
              height_ratios=[3.0, 5.2, 3.8, 1.8],
              hspace=0.16,
              top=0.965, bottom=0.02, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0: Formal MDP Specification
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16)
ax0.set_ylim(-2, 7)
ax0.set_title("  Formal MDP Specification",
              fontsize=14, fontweight="bold", color=C_RULE,
              pad=10, loc="left")

# Problem setup box (left)
ax0.add_patch(patches.FancyBboxPatch(
    (0.3, 3.2), 7.2, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))
ax0.text(3.9, 6.35, "Problem Setup", fontsize=12, ha="center",
         fontweight="bold", color=C_B)

setup_lines = [
    "Locations numbered  1  through  n = 10",
    "Walk   i  →  i + 1      reward = -1      (deterministic)",
    "Tram   i  →  2i           reward = -2      (flaky!)",
    "Tram fails with probability  p = 0.4  →  you stay at  i",
    "Goal:  reach location 10 in minimum expected time",
]
for i, txt in enumerate(setup_lines):
    ax0.text(3.9, 5.75 - i * 0.55, txt, fontsize=10, ha="center",
             fontfamily="monospace", color=MEDIUM)

# Formal components box (right)
ax0.add_patch(patches.FancyBboxPatch(
    (8.3, 3.2), 7.4, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_NODE, alpha=0.06, edgecolor=C_NODE, linewidth=1.5))
ax0.text(12.0, 6.35, "Formal MDP Components", fontsize=12, ha="center",
         fontweight="bold", color=C_NODE)

components = [
    ("States  S:",       "{ 1, 2, ..., 10 }",                                 C_NODE),
    ("Actions  A(s):",   "{ walk, tram }   (tram only if 2s <= 10)",          C_RULE),
    ("Transition  T:",   "T(s, a, s') = P(s' | s, a)",                         C_I),
    ("Reward  R:",       "R(s, a, s') = -1 (walk)  |  -2 (tram)",              C_NEG),
    ("Start / End:",     "s_start = 1    isEnd(s) = (s == 10)",                C_POS),
    ("Discount:",        "gamma = 1   (no discounting in this example)",       C_BOUND),
]
for i, (label, value, color) in enumerate(components):
    y = 5.75 - i * 0.48
    ax0.text(8.7, y, label, fontsize=10, fontweight="bold", color=color)
    ax0.text(10.6, y, value, fontsize=10, fontfamily="monospace", color=color)

# "What's new" callout
ax0.add_patch(patches.FancyBboxPatch(
    (0.3, -1.9), 15.4, 4.6, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.06, edgecolor=C_RULE, linewidth=1.5))
ax0.text(8, 2.2, "The MDP Twist  —  What Changed From Search?",
         fontsize=12, ha="center", fontweight="bold", color=C_RULE)
ax0.text(8, 1.4,
         "In search:  action  a  from state  s   →   one guaranteed next state  s'.",
         fontsize=10.5, ha="center", color=MEDIUM, fontfamily="monospace")
ax0.text(8, 0.7,
         "In an MDP:  action  a  from state  s   →   a distribution over possible next states.",
         fontsize=10.5, ha="center", color=C_RULE, fontweight="bold",
         fontfamily="monospace")
ax0.text(8, -0.2,
         "The agent chooses the action.  Nature chooses which outcome occurs.",
         fontsize=10, ha="center", color=MEDIUM, fontstyle="italic")
ax0.text(8, -1.1,
         "We no longer optimize a single path cost — we optimize expected total reward.",
         fontsize=10, ha="center", color=C_BOUND)


# ─────────────────────────────────────────────────────────
#  ROW 1: Markov Tree for a single state (i = 3)
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(0, 16)
ax1.set_ylim(-1.5, 8.5)
ax1.set_title("  Markov Tree  —  One Step From State  i = 3",
              fontsize=14, fontweight="bold", color=C_NODE,
              pad=10, loc="left")

# Layout columns:
#   col 0 (x=1.5)  : root state
#   col 1 (x=5.5)  : action nodes
#   col 2 (x=9.5)  : chance node (tram only)
#   col 3 (x=13.5) : next states
X_STATE  = 1.6
X_ACTION = 5.6
X_CHANCE = 9.6
X_NEXT   = 13.6

Y_WALK   = 6.5    # walk branch y
Y_TRAM   = 2.5    # tram action-node y
Y_TRAM_U = 4.2    # tram success next-state
Y_TRAM_D = 0.6    # tram fail next-state

# --- Root state ---
draw_state(ax1, X_STATE, (Y_WALK + Y_TRAM) / 2, "3",
           color=C_NODE, radius=0.55, fontsize=18, lw=2.6, alpha_fill=0.2)
ax1.text(X_STATE, (Y_WALK + Y_TRAM) / 2 - 0.9, "current\nstate",
         ha="center", va="top", fontsize=9, color=MEDIUM,
         fontstyle="italic")

# --- Walk branch ---
# state -> action box
draw_edge(ax1, X_STATE, (Y_WALK + Y_TRAM) / 2, X_ACTION, Y_WALK,
          color=C_POS, lw=2.2, shrink_from=0.55, shrink_to=0.65)
draw_action(ax1, X_ACTION, Y_WALK, "walk", color=C_POS, w=1.4, h=0.7,
            fontsize=12)

# action box -> next state (deterministic, so no chance node)
draw_edge(ax1, X_ACTION, Y_WALK, X_NEXT, Y_WALK,
          color=C_POS, lw=2.2, shrink_from=0.7, shrink_to=0.42)
edge_label(ax1, X_ACTION + 0.7, Y_WALK, X_NEXT - 0.42, Y_WALK,
           "p = 1.0    r = -1", color=C_POS, fontsize=10)
ax1.text((X_ACTION + X_NEXT) / 2, Y_WALK - 0.55,
         "(deterministic — one next state)",
         ha="center", fontsize=9, color=MEDIUM, fontstyle="italic")

draw_state(ax1, X_NEXT, Y_WALK, "4", color=C_NODE, radius=0.48,
           fontsize=15, lw=2.2)

# --- Tram branch ---
# state -> tram action
draw_edge(ax1, X_STATE, (Y_WALK + Y_TRAM) / 2, X_ACTION, Y_TRAM,
          color=C_I, lw=2.2, shrink_from=0.55, shrink_to=0.65)
draw_action(ax1, X_ACTION, Y_TRAM, "tram", color=C_I, w=1.4, h=0.7,
            fontsize=12)

# tram action -> chance node
draw_edge(ax1, X_ACTION, Y_TRAM, X_CHANCE, Y_TRAM,
          color=C_I, lw=2.2, shrink_from=0.7, shrink_to=0.28)
draw_chance(ax1, X_CHANCE, Y_TRAM, color=C_B, size=0.28)
ax1.text(X_CHANCE, Y_TRAM - 0.65, "chance",
         ha="center", fontsize=9, color=C_B, fontstyle="italic",
         fontweight="bold")

# chance -> success branch (up)
draw_edge(ax1, X_CHANCE, Y_TRAM, X_NEXT, Y_TRAM_U,
          color=C_POS, lw=2.2, shrink_from=0.30, shrink_to=0.48)
edge_label(ax1, X_CHANCE, Y_TRAM, X_NEXT, Y_TRAM_U,
           "p = 0.6    r = -2", color=C_POS, offset_perp=0.25, fontsize=10)
draw_state(ax1, X_NEXT, Y_TRAM_U, "6", color=C_NODE, radius=0.48,
           fontsize=15, lw=2.2)
ax1.text(X_NEXT + 0.85, Y_TRAM_U, "tram succeeds  →  jump to  2i = 6",
         ha="left", va="center", fontsize=10, color=C_POS,
         fontweight="bold")

# chance -> fail branch (down, back to state 3)
draw_edge(ax1, X_CHANCE, Y_TRAM, X_NEXT, Y_TRAM_D,
          color=C_NEG, lw=2.2, shrink_from=0.30, shrink_to=0.48)
edge_label(ax1, X_CHANCE, Y_TRAM, X_NEXT, Y_TRAM_D,
           "p = 0.4    r = -2", color=C_NEG, offset_perp=-0.25, fontsize=10)
draw_state(ax1, X_NEXT, Y_TRAM_D, "3", color=C_NEG, radius=0.48,
           fontsize=15, lw=2.2, alpha_fill=0.12)
ax1.text(X_NEXT + 0.85, Y_TRAM_D, "tram fails  →  stay put at  i = 3",
         ha="left", va="center", fontsize=10, color=C_NEG,
         fontweight="bold")

# Legend (bottom strip)
ax1.add_patch(patches.FancyBboxPatch(
    (0.3, -1.3), 15.4, 0.95, boxstyle="round,pad=0.08",
    facecolor=BG, alpha=0.9, edgecolor=C_DIM, linewidth=1))

# legend: state
draw_state(ax1, 1.3, -0.82, "s", color=C_NODE, radius=0.25, fontsize=10, lw=1.8)
ax1.text(1.9, -0.82, "state (decision)", fontsize=10, va="center",
         color=C_NODE, fontweight="bold")
# legend: action
draw_action(ax1, 5.4, -0.82, "a", color=C_RULE, w=0.7, h=0.4, fontsize=10)
ax1.text(6.0, -0.82, "action node", fontsize=10, va="center",
         color=C_RULE, fontweight="bold")
# legend: chance
draw_chance(ax1, 9.0, -0.82, color=C_B, size=0.16)
ax1.text(9.35, -0.82, "chance (nature samples)", fontsize=10, va="center",
         color=C_B, fontweight="bold")
# legend: edge label
ax1.text(13.0, -0.82, "edges:  p = prob,  r = reward",
         fontsize=10, va="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 2: Full state graph (1..10) with stochastic tram forks
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(-0.5, 16)
ax2.set_ylim(-3.5, 5.5)
ax2.set_title("  Full State Graph  (n = 10)    —    walk edges (green) and stochastic tram forks (blue/pink)",
              fontsize=13, fontweight="bold", color=C_BOUND,
              pad=10, loc="left")

n = 10
node_x = {i: 1.0 + (i - 1) * 1.55 for i in range(1, n + 1)}
node_y = 0.8

# --- Walk edges (deterministic) ---
for i in range(1, n):
    draw_edge(ax2, node_x[i], node_y, node_x[i + 1], node_y,
              color=C_POS, lw=2.0, shrink_from=0.42, shrink_to=0.42,
              alpha=0.85)
    mx = (node_x[i] + node_x[i + 1]) / 2
    ax2.text(mx, node_y - 0.32, "-1", fontsize=8, ha="center", va="center",
             color=C_POS, fontweight="bold",
             bbox=dict(facecolor=BG, edgecolor="none", pad=1))

# --- Tram arcs with stochastic forks ---
# For each i where 2i <= n, draw:
#   - upward curved arc i -> 2i     labeled  p=0.6, r=-2
#   - small self-loop on i          labeled  p=0.4, r=-2
for i in range(1, n):
    if 2 * i > n:
        continue
    # success arc
    dist = abs(node_x[2 * i] - node_x[i])
    rad = 0.30 + dist * 0.04
    ax2.annotate("", xy=(node_x[2 * i] - 0.32, node_y + 0.28),
                 xytext=(node_x[i] + 0.32, node_y + 0.28),
                 arrowprops=dict(arrowstyle="-|>", color=C_I, lw=2.0,
                                 alpha=0.85,
                                 connectionstyle=f"arc3,rad=-{rad}"),
                 zorder=3)
    mx = (node_x[i] + node_x[2 * i]) / 2
    label_y = node_y + 0.55 + dist * 0.11
    ax2.text(mx, label_y, "p=0.6, r=-2", fontsize=8, ha="center",
             va="center", color=C_I, fontweight="bold",
             bbox=dict(facecolor=BG, edgecolor="none", pad=1))

    # small self-loop on i (failure branch)
    loop_cx = node_x[i]
    loop_cy = node_y - 0.95
    # draw a small downward loop with an arrowhead returning to the node
    ax2.annotate("", xy=(node_x[i] + 0.05, node_y - 0.42),
                 xytext=(node_x[i] - 0.05, node_y - 0.42),
                 arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=1.8,
                                 alpha=0.85,
                                 connectionstyle="arc3,rad=-2.6"),
                 zorder=3)
    ax2.text(node_x[i], node_y - 1.15, "0.4\nstay", fontsize=7,
             ha="center", va="center", color=C_NEG, fontweight="bold",
             bbox=dict(facecolor=BG, edgecolor="none", pad=1))

# --- Nodes (draw on top) ---
for i in range(1, n + 1):
    if i == 1:
        draw_state(ax2, node_x[i], node_y, i, color=C_POS, radius=0.42,
                   lw=3.0, alpha_fill=0.22)
    elif i == n:
        draw_state(ax2, node_x[i], node_y, i, color=C_BOUND, radius=0.42,
                   lw=3.0, alpha_fill=0.22)
    else:
        draw_state(ax2, node_x[i], node_y, i, color=C_NODE, radius=0.40)

ax2.text(node_x[1], node_y - 1.95, "START", fontsize=9, ha="center",
         color=C_POS, fontweight="bold")
ax2.text(node_x[n], node_y - 0.7, "GOAL", fontsize=9, ha="center",
         color=C_BOUND, fontweight="bold")

# Legend strip at bottom
ax2.add_patch(patches.FancyBboxPatch(
    (0.3, -3.3), 15.2, 0.9, boxstyle="round,pad=0.08",
    facecolor=BG, alpha=0.9, edgecolor=C_DIM, linewidth=1))
ax2.text(2.1, -2.85, "Walk:  i  →  i+1     p = 1.0    r = -1",
         fontsize=10, color=C_POS, fontweight="bold", va="center")
ax2.text(7.4, -2.85, "Tram (success):  i  →  2i     p = 0.6    r = -2",
         fontsize=10, color=C_I, fontweight="bold", va="center")
ax2.text(13.5, -2.85, "Tram (fail):  stay at  i     p = 0.4    r = -2",
         fontsize=10, color=C_NEG, fontweight="bold", va="center",
         ha="center")


# ─────────────────────────────────────────────────────────
#  ROW 3: Key Takeaways
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16)
ax3.set_ylim(-3, 3.5)

ax3.add_patch(patches.FancyBboxPatch(
    (1.0, -2.5), 14, 5.5, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax3.text(8, 2.5, "Key Takeaways",
         fontsize=13, ha="center", fontweight="bold", color=C_RULE)

takeaways = [
    ("An action in an MDP leads to a distribution over next states, not a single next state.", True),
    ("The agent controls the action.  Nature controls which stochastic outcome occurs.", True),
    ("Rewards are collected on every transition — a failed tram still costs -2.", False),
    ("Walking is deterministic (one next state). Tramming is stochastic (two possible next states).", False),
    ("Because the future is uncertain, the solution is a policy  pi(s) = a, not a fixed action sequence.", True),
    ("We now optimize expected total reward over all possible rollouts — not a single path cost.", False),
]
for i, (txt, bold) in enumerate(takeaways):
    ax3.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = "/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026/Lecture 7"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "flaky_tram_mdp.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
