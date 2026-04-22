"""
Lecture 7 Visual: Q-Values in the Flaky Tram MDP

Expanding Section 16's warm-up (Q(9, walk; V) = -1) into a full explanation
of what a Q-value is, why it's an expectation, and how it relates to V.

Rows:
1. Formula anatomy       — Q(s,a;V) = sum_{s'} T(s,a,s') · (R(s,a,s') + γ V(s'))
2. Worked example 1      — Q(9, walk; V)   (deterministic — sum collapses)
3. Worked example 2      — Q(5, tram; V)   (stochastic — full sum in action)
4. Q vs V relationships  — key intuitions
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
C_POS   = "#a6e3a1"   # green  — success / final answers
C_NEG   = "#f38ba8"   # pink   — failure / self-loop
C_BOUND = "#f9e2af"   # yellow — γ / terminal / highlights
C_RULE  = "#fab387"   # orange — R / action / insights
C_I     = "#89b4fa"   # blue   — action (secondary)
C_B     = "#cba6f7"   # purple — T / chance nodes
C_DIM   = "#585b70"
C_NODE  = "#89dceb"   # teal   — V / state nodes

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Helpers ─────────────────────────────────────────────

def draw_state(ax, cx, cy, label, color=C_NODE, radius=0.48,
               fontsize=14, lw=2.2, alpha_fill=0.18):
    circle = plt.Circle((cx, cy), radius, facecolor=color, alpha=alpha_fill,
                        edgecolor=color, linewidth=lw, zorder=5)
    ax.add_patch(circle)
    ax.text(cx, cy, str(label), ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=6)


def draw_action(ax, cx, cy, label, color=C_RULE, w=1.35, h=0.62, fontsize=12):
    ax.add_patch(patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06", facecolor=color, alpha=0.15,
        edgecolor=color, linewidth=1.8, zorder=5))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=6)


def draw_chance(ax, cx, cy, color=C_B, size=0.26):
    diamond = patches.RegularPolygon(
        (cx, cy), numVertices=4, radius=size, orientation=0,
        facecolor=color, alpha=0.85, edgecolor=color, linewidth=1.5, zorder=6)
    ax.add_patch(diamond)


def draw_edge(ax, x1, y1, x2, y2, color=TEXT, lw=2.0, alpha=0.9,
              shrink_from=0.48, shrink_to=0.48, connectionstyle="arc3",
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
               fontsize=9, boxed=True, fontweight="bold"):
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


def component_tag(ax, cx, cy, symbol, label, color, w=3.4, h=1.35,
                  sym_size=15, lab_size=9.5):
    """A small 'legend card' for one piece of the Q-formula.
    Symbol on top, label underneath, both centered."""
    ax.add_patch(patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08", facecolor=color, alpha=0.12,
        edgecolor=color, linewidth=1.6, zorder=5))
    ax.text(cx, cy + 0.26, symbol,
            fontsize=sym_size, fontweight="bold", color=color,
            ha="center", va="center", fontfamily="monospace")
    ax.text(cx, cy - 0.30, label, fontsize=lab_size, color=color,
            ha="center", va="center")


# ═════════════════════════════════════════════════════════
#  FIGURE
# ═════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 32))
fig.suptitle("Q-Values in the Flaky Tram MDP",
             fontsize=18, fontweight="bold", y=0.995, color=C_BOUND)

gs = GridSpec(4, 1, figure=fig,
              height_ratios=[4.6, 5.0, 5.8, 2.6],
              hspace=0.14,
              top=0.965, bottom=0.02, left=0.04, right=0.96)


# ─────────────────────────────────────────────────────────
#  ROW 0:  What is Q(s, a; V)?  Formula anatomy
# ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 16)
ax0.set_ylim(-1, 10)
ax0.set_title("  What is  Q(s, a; V) ?     —    Formula Anatomy",
              fontsize=14, fontweight="bold", color=C_RULE,
              pad=10, loc="left")

# The Big Formula (mathtext — proper Σ subscript, Greek, etc.)
ax0.text(8, 8.7,
         r"$Q(s,\, a \,;\, V) \;\;=\;\; \sum_{s'}\; T(s,\, a,\, s') \;\cdot\; \left(\, R(s,\, a,\, s') \;+\; \gamma \, V(s') \,\right)$",
         fontsize=20, ha="center", va="center", color=TEXT)

# Caption
ax0.text(8, 7.55,
         '"If I take action  a  from state  s,  and then continue using values  V,  what is my expected return?"',
         fontsize=11, ha="center", fontstyle="italic", color=MEDIUM)

# Component legend cards — stacked (symbol on top, label below), 4 in a row
ax0.text(8, 6.6, "Each Piece of the Formula",
         fontsize=12, ha="center", fontweight="bold", color=C_BOUND)

card_y = 5.3
component_tag(ax0, 2.25,  card_y, "T(s,a,s')", "probability of landing in s'",     C_B)
component_tag(ax0, 6.25,  card_y, "R(s,a,s')", "reward on that transition",        C_RULE)
component_tag(ax0, 9.75,  card_y, "γ",          "discount on future value",         C_BOUND)
component_tag(ax0, 13.75, card_y, "V(s')",      "continuation value of successor",  C_NODE)

# Sum explanation
ax0.text(8, 3.75,
         "The  Σ  sums over every possible next state  s',  weighting each by its transition probability  T(s, a, s').",
         fontsize=10.5, ha="center", color=MEDIUM)

# Big idea summary
ax0.add_patch(patches.FancyBboxPatch(
    (0.3, -0.8), 15.4, 3.2, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.06, edgecolor=C_RULE, linewidth=1.5))
ax0.text(8, 2.05, "In Plain English",
         fontsize=12, ha="center", fontweight="bold", color=C_RULE)
ax0.text(8, 1.35,
         "Commit to action  a.   Nature sends you to some  s'  with probability  T.   You collect  R  now, and  γ·V(s')  from then on.",
         fontsize=10.5, ha="center", color=MEDIUM)
ax0.text(8, 0.65,
         "Average over all possible  s'  weighted by their probability.   That average is  Q.",
         fontsize=10.5, ha="center", color=C_RULE, fontweight="bold")
ax0.text(8, -0.1,
         "Q is action-specific  (Q(s, a)).     V is state-specific  (V(s)).     The  V  inside  Q(s, a ; V)  is your current continuation estimate — an input.",
         fontsize=10, ha="center", color=MEDIUM, fontstyle="italic")


# ─────────────────────────────────────────────────────────
#  ROW 1:  Warm-up —  Q(9, walk ; V) = -1   (deterministic)
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_xlim(0, 16)
ax1.set_ylim(-1, 9.5)
ax1.set_title("  Example 1  —   Q(9, walk ; V)     (deterministic case)",
              fontsize=14, fontweight="bold", color=C_NODE,
              pad=10, loc="left")

# --- Mini Markov tree (top half) ---
X_S, X_A, X_NEXT = 2.2, 5.8, 9.4
Y_TREE = 7.4

draw_state(ax1, X_S, Y_TREE, "9", color=C_NODE, radius=0.58,
           fontsize=18, lw=2.6, alpha_fill=0.22)
ax1.text(X_S, Y_TREE - 0.95, "state  s = 9",
         ha="center", fontsize=9, color=MEDIUM, fontstyle="italic")

draw_edge(ax1, X_S, Y_TREE, X_A, Y_TREE, color=C_POS, lw=2.2,
          shrink_from=0.58, shrink_to=0.68)
draw_action(ax1, X_A, Y_TREE, "walk", color=C_POS, w=1.45, h=0.72, fontsize=13)

draw_edge(ax1, X_A, Y_TREE, X_NEXT, Y_TREE, color=C_POS, lw=2.2,
          shrink_from=0.72, shrink_to=0.58)
edge_label(ax1, X_A + 0.72, Y_TREE, X_NEXT - 0.58, Y_TREE,
           "p = 1.0    r = -1", color=C_POS, fontsize=10)
draw_state(ax1, X_NEXT, Y_TREE, "10", color=C_BOUND, radius=0.58,
           fontsize=16, lw=2.6, alpha_fill=0.22)
ax1.text(X_NEXT, Y_TREE - 0.95, "terminal   V(10) = 0",
         ha="center", fontsize=9, color=C_BOUND, fontstyle="italic")
ax1.text(X_NEXT + 1.1, Y_TREE, "goal",
         ha="left", va="center", fontsize=10, color=C_BOUND,
         fontweight="bold")

# Inputs panel on the right
ax1.add_patch(patches.FancyBboxPatch(
    (11.5, 5.6), 4.2, 3.3, boxstyle="round,pad=0.1",
    facecolor=C_DIM, alpha=0.15, edgecolor=C_DIM, linewidth=1.2))
ax1.text(13.6, 8.55, "Given", fontsize=11, ha="center",
         fontweight="bold", color=C_BOUND)
input_lines = [
    ("T(9, walk, 10)", "= 1.0",   C_B),
    ("R(9, walk, 10)", "= -1",    C_RULE),
    ("γ",              "= 1",     C_BOUND),
    ("V(10)",          "= 0",     C_NODE),
]
for i, (lab, val, col) in enumerate(input_lines):
    y = 8.0 - i * 0.55
    ax1.text(11.85, y, lab, fontsize=10.5, color=col,
             fontfamily="monospace", fontweight="bold")
    ax1.text(14.2, y, val, fontsize=10.5, color=col,
             fontfamily="monospace", fontweight="bold")

# --- Computation block (bottom half) ---
ax1.add_patch(patches.FancyBboxPatch(
    (0.3, 0.1), 15.4, 5.0, boxstyle="round,pad=0.15",
    facecolor=C_NODE, alpha=0.05, edgecolor=C_NODE, linewidth=1.3))
ax1.text(8, 4.7, "Step-By-Step Computation",
         fontsize=12, ha="center", fontweight="bold", color=C_NODE)

lines = [
    ("Q(9, walk ; V)   =   Σ  T(9, walk, s')  ·  ( R(9, walk, s')  +  γ · V(s') )",
     "general formula", MEDIUM),
    ("                     walk is deterministic  →  only  s' = 10",
     "sum collapses to ONE term", C_RULE),
    ("                 =   T(9, walk, 10)  ·  ( R(9, walk, 10)  +  γ · V(10) )",
     "identify the single successor", MEDIUM),
    ("                 =   1.0  ·  ( -1  +  1 · 0 )",
     "plug in the inputs", C_BOUND),
    ("                 =   1.0  ·  ( -1 )",
     "simplify inside the brackets", MEDIUM),
    ("Q(9, walk ; V)   =   -1",
     "→  takes ~1 minute in expectation to reach the goal", C_POS),
]
for i, (code, note, col) in enumerate(lines):
    y = 4.0 - i * 0.6
    is_final = (i == len(lines) - 1)
    ax1.text(0.7, y, code,
             fontsize=11 if not is_final else 12.5,
             fontfamily="monospace",
             color=col,
             fontweight="bold" if is_final or i == 0 else "normal")
    ax1.text(10.8, y, note, fontsize=9.5, color=col,
             fontstyle="italic", va="center")


# ─────────────────────────────────────────────────────────
#  ROW 2:  Q(5, tram ; V)    (stochastic — full sum)
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.axis("off")
ax2.set_xlim(0, 16)
ax2.set_ylim(-1, 11)
ax2.set_title("  Example 2  —   Q(5, tram ; V)     (stochastic case — the sum actually has two terms)",
              fontsize=14, fontweight="bold", color=C_I,
              pad=10, loc="left")

# --- Mini Markov tree (top half) ---
X_S, X_A, X_C, X_NEXT = 1.9, 5.2, 8.5, 11.6
Y_ROOT = 8.4
Y_SUCCESS = 9.7
Y_FAIL = 7.1

draw_state(ax2, X_S, Y_ROOT, "5", color=C_NODE, radius=0.62,
           fontsize=18, lw=2.8, alpha_fill=0.22)
ax2.text(X_S, Y_ROOT - 1.0, "state  s = 5",
         ha="center", fontsize=9, color=MEDIUM, fontstyle="italic")

# state -> tram action
draw_edge(ax2, X_S, Y_ROOT, X_A, Y_ROOT, color=C_I, lw=2.2,
          shrink_from=0.62, shrink_to=0.68)
draw_action(ax2, X_A, Y_ROOT, "tram", color=C_I, w=1.45, h=0.72, fontsize=13)

# action -> chance
draw_edge(ax2, X_A, Y_ROOT, X_C, Y_ROOT, color=C_I, lw=2.2,
          shrink_from=0.72, shrink_to=0.30)
draw_chance(ax2, X_C, Y_ROOT, color=C_B, size=0.30)
ax2.text(X_C, Y_ROOT - 0.7, "chance", ha="center",
         fontsize=9, color=C_B, fontweight="bold", fontstyle="italic")

# chance -> success (s' = 10)
draw_edge(ax2, X_C, Y_ROOT, X_NEXT, Y_SUCCESS, color=C_POS, lw=2.2,
          shrink_from=0.32, shrink_to=0.58)
edge_label(ax2, X_C, Y_ROOT, X_NEXT, Y_SUCCESS,
           "p = 0.6    r = -2", color=C_POS, offset_perp=0.25, fontsize=10)
draw_state(ax2, X_NEXT, Y_SUCCESS, "10", color=C_BOUND, radius=0.55,
           fontsize=15, lw=2.4, alpha_fill=0.22)
ax2.text(X_NEXT + 1.1, Y_SUCCESS,
         "tram succeeds    V(10) = 0",
         ha="left", va="center", fontsize=10, color=C_POS, fontweight="bold")

# chance -> failure (s' = 5)
draw_edge(ax2, X_C, Y_ROOT, X_NEXT, Y_FAIL, color=C_NEG, lw=2.2,
          shrink_from=0.32, shrink_to=0.58)
edge_label(ax2, X_C, Y_ROOT, X_NEXT, Y_FAIL,
           "p = 0.4    r = -2", color=C_NEG, offset_perp=-0.25, fontsize=10)
draw_state(ax2, X_NEXT, Y_FAIL, "5", color=C_NEG, radius=0.55,
           fontsize=15, lw=2.2, alpha_fill=0.15)
ax2.text(X_NEXT + 1.1, Y_FAIL,
         "tram fails  (stay)   V(5) = -10/3",
         ha="left", va="center", fontsize=10, color=C_NEG, fontweight="bold")

# V values are from exact policy evaluation under "tram if possible"
# See section 21 of the notes

# --- Computation block (bottom half) ---
ax2.add_patch(patches.FancyBboxPatch(
    (0.3, 0.1), 15.4, 5.7, boxstyle="round,pad=0.15",
    facecolor=C_I, alpha=0.05, edgecolor=C_I, linewidth=1.3))
ax2.text(8, 5.4, "Step-By-Step Computation",
         fontsize=12, ha="center", fontweight="bold", color=C_I)

lines = [
    ("Q(5, tram ; V)   =   Σ  T(5, tram, s')  ·  ( R(5, tram, s')  +  γ · V(s') )",
     "general formula", MEDIUM),
    ("                     tram is stochastic  →  TWO possible successors",
     "sum has two terms", C_I),
    ("                 =   T(5, tram, 10) · ( R + γ·V(10) )   +   T(5, tram, 5) · ( R + γ·V(5) )",
     "one term per possible s'", MEDIUM),
    ("                 =   0.6 · ( -2 + 1·0 )   +   0.4 · ( -2 + 1·(-10/3) )",
     "plug in the inputs", C_BOUND),
    ("                 =   0.6 · ( -2 )         +   0.4 · ( -16/3 )",
     "simplify inside the brackets", MEDIUM),
    ("                 =   -1.2                 +   -2.133...",
     "one sub-term per successor", MEDIUM),
    ("Q(5, tram ; V)   =   -10/3   ≈   -3.333",
     "→  expected cost ~3.33 minutes from state 5", C_POS),
]
for i, (code, note, col) in enumerate(lines):
    y = 4.75 - i * 0.63
    is_final = (i == len(lines) - 1)
    ax2.text(0.5, y, code,
             fontsize=10.5 if not is_final else 12.5,
             fontfamily="monospace",
             color=col,
             fontweight="bold" if is_final or i == 0 else "normal")
    ax2.text(11.4, y, note, fontsize=9.5, color=col,
             fontstyle="italic", va="center")

# Bellman-consistency observation
ax2.add_patch(patches.FancyBboxPatch(
    (0.3, -0.85), 15.4, 0.85, boxstyle="round,pad=0.08",
    facecolor=C_POS, alpha=0.10, edgecolor=C_POS, linewidth=1.3))
ax2.text(8, -0.42,
         "Notice:   Q(5, tram ; V)  =  V(5)  =  -10/3   —   this is the Bellman consistency for the policy  π(5) = tram.",
         ha="center", fontsize=10.5, color=C_POS, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  ROW 3:  Q vs V — key intuitions
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.axis("off")
ax3.set_xlim(0, 16)
ax3.set_ylim(-3, 3.5)

ax3.add_patch(patches.FancyBboxPatch(
    (0.5, -2.8), 15.0, 6.1, boxstyle="round,pad=0.12",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax3.text(8, 2.9, "Q vs V  —  Key Intuitions",
         fontsize=13, ha="center", fontweight="bold", color=C_RULE)

intuitions = [
    ("Q is action-specific.   V is state-specific.   Q(s, a) asks:  \"how good is this ACTION from here?\"   V(s) asks:  \"how good is this STATE?\"",    True),
    ("Q(s, a ; V)  is a one-step lookahead:  commit to  a  now,  then trust  V  for everything after.",                                                  True),
    ("The  V  inside  Q(s, a ; V)  is an INPUT — your current estimate of state values.  As  V  improves,  Q  updates with it.",                          False),
    ("For a policy π:       V_π(s)  =  Q(s, π(s) ; V_π)       — the state's value equals the Q-value of the policy's chosen action.",                    True),
    ("For the optimal π*:   V*(s)  =  max_a  Q(s, a ; V*)    — the state's optimal value is the best Q across all available actions.",                   True),
    ("Both policy evaluation and value iteration work by repeatedly computing Q-values and folding them back into V.  Q is the update engine.",          False),
]
for i, (txt, bold) in enumerate(intuitions):
    ax3.text(8, 2.15 - i * 0.78, txt, fontsize=10.5, ha="center",
             color=C_RULE, fontweight="bold" if bold else "normal",
             fontfamily="monospace" if "=" in txt and bold else "sans-serif")


# ── Save ────────────────────────────────────────────────
out_dir = "/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026/Lecture 7"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "q_value_flaky_tram.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
