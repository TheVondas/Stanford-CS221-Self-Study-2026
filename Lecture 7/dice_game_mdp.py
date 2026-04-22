"""
Lecture 7 Visual: The Dice Game MDP (Example 2)

Shows:
1. Formal MDP specification (states, actions, transitions, rewards, gamma)
2. Markov tree from state `in` — quit (deterministic, reward 10) vs stay
   (stochastic: 1/3 game-ends with reward 4, 2/3 continues with reward 4)
3. Die-roll mechanic (which faces end the game / which continue) paired
   with the value derivation showing V_stay(in) = 12 > V_quit(in) = 10
4. Key takeaways
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
C_POS   = "#a6e3a1"   # green  — continue playing / favourable outcome
C_NEG   = "#f38ba8"   # pink   — game ends from gamble
C_BOUND = "#f9e2af"   # yellow — terminal state
C_RULE  = "#fab387"   # orange — quit action / key insights
C_I     = "#89b4fa"   # blue   — stay action / stochastic info
C_B     = "#cba6f7"   # purple — chance node
C_DIM   = "#585b70"
C_NODE  = "#89dceb"   # teal   — regular state

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Helpers ─────────────────────────────────────────────

def draw_state(ax, cx, cy, label, color=C_NODE, radius=0.55,
               fontsize=14, lw=2.4, alpha_fill=0.18):
    circle = plt.Circle((cx, cy), radius, facecolor=color, alpha=alpha_fill,
                        edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(circle)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=6)


def draw_action(ax, cx, cy, label, color=C_RULE, w=1.4, h=0.65, fontsize=12):
    ax.add_patch(patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06", facecolor=color, alpha=0.15,
        edgecolor=color, linewidth=1.8, zorder=5))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=6)


def draw_chance(ax, cx, cy, color=C_B, size=0.28):
    diamond = patches.RegularPolygon(
        (cx, cy), numVertices=4, radius=size, orientation=0,
        facecolor=color, alpha=0.85, edgecolor=color, linewidth=1.5, zorder=6)
    ax.add_patch(diamond)


def draw_edge(ax, x1, y1, x2, y2, color=TEXT, lw=2.0, alpha=0.9,
              shrink_from=0.55, shrink_to=0.55, connectionstyle="arc3",
              style="-|>"):
    dx, dy = x2 - x1, y2 - y1
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


def edge_label(ax, x1, y1, x2, y2, text, color, offset_perp=0.0,
               fontsize=10, boxed=True, fontweight="bold"):
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    if offset_perp:
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy)
        if d:
            px, py = -dy / d, dx / d
            mx += px * offset_perp
            my += py * offset_perp
    kwargs = dict(fontsize=fontsize, ha="center", va="center",
                  color=color, fontweight=fontweight)
    if boxed:
        kwargs["bbox"] = dict(facecolor=BG, edgecolor="none", pad=1.5)
    ax.text(mx, my, text, **kwargs)


def draw_die_face(ax, cx, cy, value, size=0.9, color=C_POS, alpha_fill=0.15):
    """Draw a die face with standard pip arrangement."""
    ax.add_patch(patches.FancyBboxPatch(
        (cx - size / 2, cy - size / 2), size, size,
        boxstyle="round,pad=0.04",
        facecolor=color, alpha=alpha_fill,
        edgecolor=color, linewidth=2.2, zorder=5))
    # Normalised pip positions (0-1 within face)
    patterns = {
        1: [(0.5, 0.5)],
        2: [(0.28, 0.72), (0.72, 0.28)],
        3: [(0.28, 0.72), (0.5, 0.5), (0.72, 0.28)],
        4: [(0.28, 0.28), (0.28, 0.72), (0.72, 0.28), (0.72, 0.72)],
        5: [(0.28, 0.28), (0.28, 0.72), (0.5, 0.5),
            (0.72, 0.28), (0.72, 0.72)],
        6: [(0.28, 0.25), (0.28, 0.5), (0.28, 0.75),
            (0.72, 0.25), (0.72, 0.5), (0.72, 0.75)],
    }
    pip_r = size * 0.075
    for px, py in patterns[value]:
        ax_ = cx - size / 2 + px * size
        ay_ = cy - size / 2 + py * size
        ax.add_patch(plt.Circle((ax_, ay_), pip_r,
                                facecolor=color, alpha=0.95,
                                edgecolor=color, zorder=6))


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 26))
fig.suptitle("The Dice Game MDP  —  Example 2",
             fontsize=18, fontweight="bold", y=0.995, color=C_BOUND)

gs = GridSpec(4, 1, figure=fig,
              height_ratios=[3.0, 5.0, 4.6, 2.0],
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

# Problem setup (left)
ax0.add_patch(patches.FancyBboxPatch(
    (0.3, 3.2), 7.2, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_B, alpha=0.06, edgecolor=C_B, linewidth=1.5))
ax0.text(3.9, 6.35, "Problem Setup", fontsize=12, ha="center",
         fontweight="bold", color=C_B)

setup_lines = [
    "At each round you choose:  quit  or  stay",
    "Quit   →   receive  $10   and the game ends",
    "Stay   →   receive  $4,   then roll a fair six-sided die",
    "  die = 1 or 2    →   game ends",
    "  die = 3,4,5,6   →   continue to next round",
    "Goal:  maximize expected total earnings",
]
for i, txt in enumerate(setup_lines):
    ax0.text(3.9, 5.75 - i * 0.46, txt, fontsize=10, ha="center",
             fontfamily="monospace", color=MEDIUM)

# Formal components (right)
ax0.add_patch(patches.FancyBboxPatch(
    (8.3, 3.2), 7.4, 3.5, boxstyle="round,pad=0.15",
    facecolor=C_NODE, alpha=0.06, edgecolor=C_NODE, linewidth=1.5))
ax0.text(12.0, 6.35, "Formal MDP Components", fontsize=12, ha="center",
         fontweight="bold", color=C_NODE)

components = [
    ("States  S:",        "{ in, end }",                                    C_NODE),
    ("Actions  A(in):",   "{ quit, stay }       A(end) = {}",               C_RULE),
    ("Transition  T:",    "T(in, quit, end) = 1",                           C_I),
    ("",                  "T(in, stay, end) = 1/3    T(in, stay, in) = 2/3", C_I),
    ("Reward  R:",        "R(in, quit, end) = 10    R(in, stay, *) = 4",    C_POS),
    ("Start / End:",      "s_start = in     isEnd(s) = (s == end)",         C_BOUND),
    ("Discount:",         "gamma = 1",                                      C_BOUND),
]
for i, (label, value, color) in enumerate(components):
    y = 5.85 - i * 0.42
    if label:
        ax0.text(8.7, y, label, fontsize=9.5, fontweight="bold", color=color)
    ax0.text(10.8, y, value, fontsize=9.5, fontfamily="monospace", color=color)

# Callout (bottom)
ax0.add_patch(patches.FancyBboxPatch(
    (0.3, -1.9), 15.4, 4.6, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.06, edgecolor=C_RULE, linewidth=1.5))
ax0.text(8, 2.2, "The Core Decision",
         fontsize=12, ha="center", fontweight="bold", color=C_RULE)
ax0.text(8, 1.4,
         "Quit guarantees $10.   Stay gives $4 now plus a 2/3 chance to roll the dice again.",
         fontsize=10.5, ha="center", color=MEDIUM, fontfamily="monospace")
ax0.text(8, 0.5,
         "A risky action can still beat a safe one  —  if the expected total reward is higher.",
         fontsize=10.5, ha="center", color=C_RULE, fontweight="bold")
ax0.text(8, -0.4,
         "Because the only non-terminal state is `in`, the policy is just a single action choice.",
         fontsize=10, ha="center", color=MEDIUM, fontstyle="italic")
ax0.text(8, -1.2,
         "So here the question is purely:  always quit,  or  always stay?",
         fontsize=10, ha="center", color=C_BOUND)


# ─────────────────────────────────────────────────────────
#  ROW 1: Markov Tree from state `in`
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(0, 16)
ax1.set_ylim(-1.5, 8.5)
ax1.set_title("  Markov Tree  —  One Step From State  `in`",
              fontsize=14, fontweight="bold", color=C_NODE,
              pad=10, loc="left")

# Column positions
X_STATE  = 1.8
X_ACTION = 5.8
X_CHANCE = 9.8
X_NEXT   = 13.6

# Vertical positions
Y_QUIT   = 6.5     # quit branch
Y_STAY   = 2.5     # stay action node
Y_STAY_U = 4.2     # stay → end (dice 1/2)
Y_STAY_D = 0.8     # stay → in  (dice 3-6)

# --- Root state `in` ---
draw_state(ax1, X_STATE, (Y_QUIT + Y_STAY) / 2, "in",
           color=C_NODE, radius=0.65, fontsize=18, lw=2.8, alpha_fill=0.22)
ax1.text(X_STATE, (Y_QUIT + Y_STAY) / 2 - 1.0, "current\nstate",
         ha="center", va="top", fontsize=9, color=MEDIUM, fontstyle="italic")

# --- QUIT branch ---
draw_edge(ax1, X_STATE, (Y_QUIT + Y_STAY) / 2, X_ACTION, Y_QUIT,
          color=C_RULE, lw=2.2, shrink_from=0.65, shrink_to=0.72)
draw_action(ax1, X_ACTION, Y_QUIT, "quit", color=C_RULE, w=1.5, h=0.75,
            fontsize=13)

# action → end (deterministic)
draw_edge(ax1, X_ACTION, Y_QUIT, X_NEXT, Y_QUIT,
          color=C_RULE, lw=2.2, shrink_from=0.75, shrink_to=0.55)
edge_label(ax1, X_ACTION + 0.75, Y_QUIT, X_NEXT - 0.55, Y_QUIT,
           "p = 1.0    r = 10", color=C_RULE, fontsize=10)
ax1.text((X_ACTION + X_NEXT) / 2, Y_QUIT - 0.65,
         "(deterministic — guaranteed payout, game ends)",
         ha="center", fontsize=9, color=MEDIUM, fontstyle="italic")
draw_state(ax1, X_NEXT, Y_QUIT, "end", color=C_BOUND, radius=0.55,
           fontsize=13, lw=2.4, alpha_fill=0.18)

# --- STAY branch ---
draw_edge(ax1, X_STATE, (Y_QUIT + Y_STAY) / 2, X_ACTION, Y_STAY,
          color=C_I, lw=2.2, shrink_from=0.65, shrink_to=0.72)
draw_action(ax1, X_ACTION, Y_STAY, "stay", color=C_I, w=1.5, h=0.75,
            fontsize=13)

# action → chance node
draw_edge(ax1, X_ACTION, Y_STAY, X_CHANCE, Y_STAY,
          color=C_I, lw=2.2, shrink_from=0.75, shrink_to=0.32)
draw_chance(ax1, X_CHANCE, Y_STAY, color=C_B, size=0.32)
ax1.text(X_CHANCE, Y_STAY - 0.75, "chance\n(die roll)",
         ha="center", fontsize=9, color=C_B, fontstyle="italic",
         fontweight="bold")

# chance → end (die 1 or 2)
draw_edge(ax1, X_CHANCE, Y_STAY, X_NEXT, Y_STAY_U,
          color=C_NEG, lw=2.2, shrink_from=0.32, shrink_to=0.55)
edge_label(ax1, X_CHANCE, Y_STAY, X_NEXT, Y_STAY_U,
           "p = 1/3    r = 4", color=C_NEG, offset_perp=0.28, fontsize=10)
draw_state(ax1, X_NEXT, Y_STAY_U, "end", color=C_NEG, radius=0.55,
           fontsize=13, lw=2.2, alpha_fill=0.15)
ax1.text(X_NEXT + 0.95, Y_STAY_U, "die = 1 or 2   →   game ends",
         ha="left", va="center", fontsize=10, color=C_NEG, fontweight="bold")

# chance → in (die 3-6)
draw_edge(ax1, X_CHANCE, Y_STAY, X_NEXT, Y_STAY_D,
          color=C_POS, lw=2.2, shrink_from=0.32, shrink_to=0.55)
edge_label(ax1, X_CHANCE, Y_STAY, X_NEXT, Y_STAY_D,
           "p = 2/3    r = 4", color=C_POS, offset_perp=-0.28, fontsize=10)
draw_state(ax1, X_NEXT, Y_STAY_D, "in", color=C_POS, radius=0.55,
           fontsize=13, lw=2.2, alpha_fill=0.18)
ax1.text(X_NEXT + 0.95, Y_STAY_D, "die = 3, 4, 5, or 6   →   keep playing",
         ha="left", va="center", fontsize=10, color=C_POS, fontweight="bold")

# Legend strip
ax1.add_patch(patches.FancyBboxPatch(
    (0.3, -1.3), 15.4, 0.95, boxstyle="round,pad=0.08",
    facecolor=BG, alpha=0.9, edgecolor=C_DIM, linewidth=1))
draw_state(ax1, 1.3, -0.82, "s", color=C_NODE, radius=0.25, fontsize=10, lw=1.8)
ax1.text(1.9, -0.82, "state (decision)", fontsize=10, va="center",
         color=C_NODE, fontweight="bold")
draw_action(ax1, 5.4, -0.82, "a", color=C_RULE, w=0.7, h=0.4, fontsize=10)
ax1.text(6.0, -0.82, "action node", fontsize=10, va="center",
         color=C_RULE, fontweight="bold")
draw_chance(ax1, 9.0, -0.82, color=C_B, size=0.16)
ax1.text(9.35, -0.82, "chance (nature samples)", fontsize=10, va="center",
         color=C_B, fontweight="bold")
ax1.text(13.0, -0.82, "edges:  p = prob,  r = reward",
         fontsize=10, va="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 2: Die mechanic  +  Policy value comparison
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(0, 16)
ax2.set_ylim(-1, 8.5)

# Left half: die faces
ax2.text(3.8, 8.0, "The Die Roll",
         fontsize=14, fontweight="bold", color=C_B, ha="center")

# 6 dice in a row
die_y = 6.0
die_size = 0.95
die_gap = 1.25
die_x0 = 0.9
for v in range(1, 7):
    cx = die_x0 + (v - 1) * die_gap
    if v in (1, 2):
        draw_die_face(ax2, cx, die_y, v, size=die_size,
                      color=C_NEG, alpha_fill=0.18)
    else:
        draw_die_face(ax2, cx, die_y, v, size=die_size,
                      color=C_POS, alpha_fill=0.18)

# Grouping brackets + labels
# Bracket for 1,2 (ends)
ax2.plot([die_x0 - 0.35, die_x0 - 0.35,
          die_x0 + die_gap + 0.35, die_x0 + die_gap + 0.35],
         [die_y - 0.6, die_y - 0.85, die_y - 0.85, die_y - 0.6],
         color=C_NEG, lw=1.8, alpha=0.9)
ax2.text(die_x0 + die_gap / 2, die_y - 1.25,
         "p = 2/6 = 1/3",
         ha="center", fontsize=10, color=C_NEG, fontweight="bold")
ax2.text(die_x0 + die_gap / 2, die_y - 1.65,
         "→  game ends",
         ha="center", fontsize=10, color=C_NEG, fontweight="bold")

# Bracket for 3-6 (continue)
ax2.plot([die_x0 + 2 * die_gap - 0.35, die_x0 + 2 * die_gap - 0.35,
          die_x0 + 5 * die_gap + 0.35, die_x0 + 5 * die_gap + 0.35],
         [die_y - 0.6, die_y - 0.85, die_y - 0.85, die_y - 0.6],
         color=C_POS, lw=1.8, alpha=0.9)
ax2.text(die_x0 + 3.5 * die_gap, die_y - 1.25,
         "p = 4/6 = 2/3",
         ha="center", fontsize=10, color=C_POS, fontweight="bold")
ax2.text(die_x0 + 3.5 * die_gap, die_y - 1.65,
         "→  continue (back to `in`)",
         ha="center", fontsize=10, color=C_POS, fontweight="bold")

# Subtle caption
ax2.text(3.8, 3.3,
         "The die is the source of randomness.",
         ha="center", fontsize=10, color=MEDIUM, fontstyle="italic")
ax2.text(3.8, 2.75,
         "A fair roll yields  $1/6$  per face, grouped into two outcomes.",
         ha="center", fontsize=10, color=MEDIUM, fontstyle="italic")

# Right half: value comparison
X_L = 8.3
ax2.add_patch(patches.FancyBboxPatch(
    (X_L, 0.3), 7.5, 7.8, boxstyle="round,pad=0.15",
    facecolor=C_BOUND, alpha=0.06, edgecolor=C_BOUND, linewidth=1.5))
ax2.text(X_L + 3.75, 7.6, "Which Policy Is Better?",
         fontsize=14, fontweight="bold", color=C_BOUND, ha="center")

# Quit policy
ax2.text(X_L + 0.3, 6.7, "Always quit:",
         fontsize=11, fontweight="bold", color=C_RULE)
ax2.text(X_L + 0.3, 6.1,
         "V_quit(in)  =  10",
         fontsize=12, fontfamily="monospace", color=C_RULE)
ax2.text(X_L + 0.3, 5.5,
         "→  guaranteed, no randomness",
         fontsize=9.5, color=MEDIUM, fontstyle="italic")

# Divider
ax2.plot([X_L + 0.3, X_L + 7.2], [4.95, 4.95],
         color=C_DIM, lw=0.8, alpha=0.8)

# Stay policy derivation
ax2.text(X_L + 0.3, 4.45, "Always stay:    let  V = V_stay(in)",
         fontsize=11, fontweight="bold", color=C_I)

stay_derivation = [
    "V  =  (1/3)·(4 + 0)   +   (2/3)·(4 + V)",
    "V  =  4  +  (2/3)·V",
    "(1/3)·V  =  4",
    "V  =  12",
]
for i, line in enumerate(stay_derivation):
    ax2.text(X_L + 0.5, 3.85 - i * 0.55, line,
             fontsize=11, fontfamily="monospace", color=C_I)

# Divider
ax2.plot([X_L + 0.3, X_L + 7.2], [1.55, 1.55],
         color=C_DIM, lw=0.8, alpha=0.8)

# Conclusion
ax2.text(X_L + 3.75, 1.05,
         "V_stay(in) = 12  >  V_quit(in) = 10",
         ha="center", fontsize=12, fontweight="bold", color=C_POS,
         fontfamily="monospace")
ax2.text(X_L + 3.75, 0.55,
         "→  stay is optimal:  π*(in) = stay",
         ha="center", fontsize=10.5, fontweight="bold", color=C_POS)


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
    ("Quit is safe — $10 guaranteed.  Stay is risky — $4 plus a 2/3 chance to play again.", True),
    ("Expected value can reward the gamble: V_stay(in) = 12  beats  V_quit(in) = 10 by $2.", True),
    ("Because the only non-terminal state is `in`, the policy is a single action choice.", False),
    ("A rule like \"stay three times then quit\" is NOT representable — the state doesn't track the round.", False),
    ("To make round-dependent strategies, the state would need to include additional information.", False),
    ("This example shows value iteration's logic in miniature: compare action values, pick the max.", True),
]
for i, (txt, bold) in enumerate(takeaways):
    ax3.text(8, 1.6 - i * 0.65, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal")


# ── Save ────────────────────────────────────────────────
out_dir = "/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026/Lecture 7"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "dice_game_mdp.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
