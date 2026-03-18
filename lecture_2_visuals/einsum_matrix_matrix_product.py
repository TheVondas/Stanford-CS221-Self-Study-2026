"""
Lecture 2 Visual: einsum("i k, j k -> i j", A, B) — Matrix-Matrix Product

Shows WHY the notation works:
- A has axes i (rows) and k
- B has axes j (rows) and k
- k appears in BOTH inputs → elements PAIRED along k
- k is NOT in output → SUMMED OVER (contracted)
- i and j both in output → BOTH KEPT
- Each output[i,j] = dot product of row i of A with row j of B
- Note: this is A @ B.T — the k axis tells you WHICH product you get
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Style ─────────────────────────────────────────────────
BG = "#1e1e2e"
TEXT = "#cdd6f4"
SUBTLE = "#6c7086"
MEDIUM = "#a6adc8"

C_I = "#89b4fa"       # blue:   i axis — KEPT
C_J = "#a6e3a1"       # green:  j axis — KEPT
C_OUT = "#f9e2af"     # yellow: output / result
C_RULE = "#fab387"    # orange: rule highlights
C_K = "#f38ba8"       # pink:   k axis — CONTRACTED
C_DIM = "#585b70"     # dimmed
C_WARN = "#f5c2e7"    # light pink: callouts

CELL = 0.55
GAP = 0.06
S = CELL + GAP

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
})


# ── Drawing helpers ───────────────────────────────────────

def dc(ax, x, y, val, col, alpha=0.7, fs=13):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), CELL, CELL, boxstyle="round,pad=0.02",
        facecolor=col, alpha=alpha, edgecolor="white", linewidth=1))
    ax.text(x + CELL / 2, y + CELL / 2, str(val), ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#1e1e2e")


def dm(ax, data, x0, y0, col=None, cc=None):
    R, C_ = data.shape
    for r in range(R):
        for c in range(C_):
            color = col if cc is None else cc[r][c]
            dc(ax, x0 + c * S, y0 - r * S, data[r, c], color)


def dvh(ax, vals, x0, y0, col):
    for i, v in enumerate(vals):
        dc(ax, x0 + i * S, y0, v, col)


def arr_h(ax, x0, w, y, col, label, above=True):
    ax.annotate("", xy=(x0 + w + 0.08, y), xytext=(x0 - 0.08, y),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    dy = 0.18 if above else -0.22
    va = "bottom" if above else "top"
    ax.text(x0 + w / 2, y + dy, label, ha="center", fontsize=10,
            color=col, fontweight="bold", va=va)


def arr_v(ax, x, y_top, h, col, label):
    ax.annotate("", xy=(x, y_top - h - 0.08), xytext=(x, y_top + 0.08),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax.text(x - 0.22, y_top - h / 2, label, ha="center", va="center",
            fontsize=10, color=col, fontweight="bold", rotation=90)


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
#  FIGURE — 4 sections
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(14, 36),
                         gridspec_kw={"height_ratios": [1, 3.2, 2.4, 2.0],
                                      "hspace": 0.10})
fig.suptitle('einsum("i k, j k  →  i j") — Matrix-Matrix Product',
             fontsize=16, fontweight="bold", y=0.98, color=C_OUT)

for a in axes:
    a.axis("off")


# ───────────────────────────────────────────────────────────
#  SECTION 1: Read the Notation
# ───────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 14)
ax.set_ylim(-2, 4.5)

ax.text(0.5, 4.0, "Step 1 — Read the Notation", fontsize=14,
        fontweight="bold", color=C_RULE)

ny = 2.8
ax.text(0.3, ny, 'einsum("', fontsize=14, fontfamily="monospace",
        color=MEDIUM)
ax.text(2.65, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(3.05, ny, ' ', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(3.2, ny, 'k', fontsize=18, fontfamily="monospace",
        color=C_K, fontweight="bold")
ax.text(3.55, ny, ',', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(3.9, ny, ' j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(4.45, ny, ' ', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(4.6, ny, 'k', fontsize=18, fontfamily="monospace",
        color=C_K, fontweight="bold")
ax.text(5.0, ny, '  ->  ', fontsize=14, fontfamily="monospace",
        color=TEXT)
ax.text(7.0, ny, 'i', fontsize=18, fontfamily="monospace",
        color=C_I, fontweight="bold")
ax.text(7.4, ny, ' ', fontsize=14, fontfamily="monospace", color=TEXT)
ax.text(7.55, ny, 'j', fontsize=18, fontfamily="monospace",
        color=C_J, fontweight="bold")
ax.text(7.95, ny, '", A, B)', fontsize=14, fontfamily="monospace",
        color=MEDIUM)

# Bracket connecting the two k's
ax.annotate("", xy=(4.8, ny + 0.55), xytext=(3.35, ny + 0.55),
            arrowprops=dict(arrowstyle="<->", color=C_K, lw=1.8,
                            connectionstyle="arc3,rad=-0.35"))
ax.text(4.07, ny + 1.15, "same  k  = PAIRED (×)", fontsize=9, color=C_K,
        ha="center", fontweight="bold", fontstyle="italic")

# Highlight output
ax.add_patch(patches.FancyBboxPatch(
    (6.85, ny - 0.15), 1.05, 0.55, boxstyle="round,pad=0.05",
    facecolor=C_OUT, alpha=0.15, edgecolor=C_OUT, linewidth=1.5,
    linestyle="--"))

# Annotations
ax.annotate("A: rows  i ,  cols  k",
            xy=(2.9, ny - 0.2), xytext=(0.5, ny - 1.3),
            fontsize=10, color=C_I, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=C_I, lw=1.3,
                            connectionstyle="arc3,rad=0.15"))

ax.annotate("B: rows  j ,  cols  k",
            xy=(4.3, ny - 0.2), xytext=(3.5, ny - 1.3),
            fontsize=10, color=C_J, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.3,
                            connectionstyle="arc3,rad=0.15"))

ax.text(8.5, ny + 0.2, "i, j  KEPT  →  matrix output",
        fontsize=10, color=C_OUT, fontweight="bold")
ax.text(8.5, ny - 0.4, "k  GONE  →  summed (contracted)",
        fontsize=10, color=C_K, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 2: What Happens — the grid of dot products
# ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(-8, 7.5)

ax.text(0.5, 7.0, "Step 2 — What Happens", fontsize=14,
        fontweight="bold", color=C_RULE)

A = np.array([[2, 1],
              [3, 4]])
B = np.array([[1, 3],
              [2, 0]])
# Result = A @ B.T because both share k on cols
# C[i,j] = Σ_k A[i,k] * B[j,k]
C = A @ B.T  # [[5, 4], [15, 6]]

# ── Matrix A ──
ax_x, ay = 0.5, 4.8
A_cc = [[C_I, C_I], [C_I, C_I]]
dm(ax, A, ax_x, ay, cc=A_cc)
ax.text(ax_x + 0.5 * S, ay + CELL + 0.55, "A  (2×2)", fontsize=12,
        fontweight="bold", ha="center")
arr_h(ax, ax_x, 2 * S - GAP, ay + CELL + 0.15, C_K, "k")
arr_v(ax, ax_x - 0.45, ay + CELL, 2 * S - GAP, C_I, "i")

for idx in range(2):
    ax.text(ax_x + idx * S + CELL / 2, ay + CELL + 0.9,
            f"k={idx}", fontsize=8, color=C_K, ha="center",
            fontfamily="monospace")
    ax.text(ax_x - 0.85, ay - idx * S + CELL / 2,
            f"i={idx}", fontsize=8, color=C_I, ha="center",
            fontfamily="monospace", va="center")

# ── Matrix B ──
bx, by = 4.0, 4.8
B_cc = [[C_J, C_J], [C_J, C_J]]
dm(ax, B, bx, by, cc=B_cc)
ax.text(bx + 0.5 * S, by + CELL + 0.55, "B  (2×2)", fontsize=12,
        fontweight="bold", ha="center")
arr_h(ax, bx, 2 * S - GAP, by + CELL + 0.15, C_K, "k")
arr_v(ax, bx - 0.45, by + CELL, 2 * S - GAP, C_J, "j")

for idx in range(2):
    ax.text(bx + idx * S + CELL / 2, by + CELL + 0.9,
            f"k={idx}", fontsize=8, color=C_K, ha="center",
            fontfamily="monospace")
    ax.text(bx - 0.85, by - idx * S + CELL / 2,
            f"j={idx}", fontsize=8, color=C_J, ha="center",
            fontfamily="monospace", va="center")

# ── "same k!" connector ──
ax.annotate("same k!",
            xy=(bx + 0.1, ay + CELL + 0.15),
            xytext=(ax_x + 2 * S + 0.1, ay + CELL + 0.15),
            fontsize=9, color=C_K, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="<->", color=C_K, lw=1.3,
                            connectionstyle="arc3,rad=-0.15"))

# ── Arrow to result ──
eq_x = bx + 2 * S + 0.8
ax.text(eq_x, ay - 0.5 * S + CELL / 2, "=", fontsize=18,
        ha="center", va="center", color=TEXT)

# ── Result matrix ──
cx = eq_x + 0.8
cy = ay
dm(ax, C, cx, cy, C_OUT)
ax.text(cx + 0.5 * S, cy + CELL + 0.55, "C  (2×2)", fontsize=12,
        fontweight="bold", ha="center", color=C_OUT)
arr_h(ax, cx, 2 * S - GAP, cy + CELL + 0.15, C_J, "j  (kept)")
arr_v(ax, cx - 0.45, cy + CELL, 2 * S - GAP, C_I, "i  (kept)")

# ── Highlight one element: C[0,1] ──
hi_r, hi_c = 0, 1
hx = cx + hi_c * S
hy = cy - hi_r * S
ax.add_patch(patches.FancyBboxPatch(
    (hx - 0.04, hy - 0.04), CELL + 0.08, CELL + 0.08,
    boxstyle="round,pad=0.02",
    facecolor="none", edgecolor=C_WARN, linewidth=2.5))

ax.annotate(
    "C[i=0, j=1]\n= row 0 of A  ·  row 1 of B\n= [2,1] · [2,0]\n= 2×2 + 1×0 = 4",
    xy=(hx + CELL + 0.1, hy + CELL / 2),
    xytext=(hx + 1.5, hy + 1.8),
    fontsize=9.5, color=C_WARN, fontweight="bold", ha="left",
    arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5,
                    connectionstyle="arc3,rad=-0.2"))

# ── Visual: row i of A dot-products with row j of B ──
ax.text(0.5, 2.2, "Each output element = dot product of a row from A with a row from B:",
        fontsize=12, fontweight="bold", color=MEDIUM)

# Show all four computations
traces = [
    ("C[0,0]:", "A row 0 · B row 0", "[2,1] · [1,3]", "2×1 + 1×3", "= 5"),
    ("C[0,1]:", "A row 0 · B row 1", "[2,1] · [2,0]", "2×2 + 1×0", "= 4"),
    ("C[1,0]:", "A row 1 · B row 0", "[3,4] · [1,3]", "3×1 + 4×3", "= 15"),
    ("C[1,1]:", "A row 1 · B row 1", "[3,4] · [2,0]", "3×2 + 4×0", "= 6"),
]

for idx, (label, desc, vecs, calc, res) in enumerate(traces):
    y = 1.4 - idx * 0.6
    ax.text(0.5, y, label, fontsize=10, fontfamily="monospace",
            color=C_OUT, fontweight="bold")
    ax.text(2.0, y, desc, fontsize=10, fontfamily="monospace", color=MEDIUM)
    ax.text(5.7, y, vecs, fontsize=9.5, fontfamily="monospace", color=MEDIUM)
    ax.text(8.5, y, calc, fontsize=9.5, fontfamily="monospace", color=MEDIUM)
    ax.text(11.5, y, res, fontsize=11, fontfamily="monospace",
            color=C_OUT, fontweight="bold")

# ── The formula ──
ax.text(0.5, -1.3,
        "C[i, j]  =  Σ_k  A[i, k] × B[j, k]",
        fontsize=13, fontfamily="monospace", color=C_OUT, fontweight="bold")

# ── The pseudocode ──
ax.text(0.5, -2.3, "The algorithm:", fontsize=12,
        fontweight="bold", color=TEXT)

py = -2.9
ax.text(1.2, py,
        "for each  i :", fontsize=10.5, fontfamily="monospace", color=C_I)
ax.text(1.2, py - 0.5,
        "    for each  j :", fontsize=10.5, fontfamily="monospace", color=C_J)
ax.text(1.2, py - 1.0,
        "        output[i, j] = 0", fontsize=10.5, fontfamily="monospace",
        color=SUBTLE)
ax.text(1.2, py - 1.5,
        "        for each  k :", fontsize=10.5, fontfamily="monospace",
        color=C_K)
ax.text(1.2, py - 2.0,
        "            output[i, j]  +=  A[i, k] × B[j, k]",
        fontsize=11, fontfamily="monospace", color=TEXT, fontweight="bold")

# Annotate the ops
ax.annotate("i, j  survive\n→ 2D output",
            xy=(3.5, py + 0.1), xytext=(7.5, py + 0.3),
            fontsize=9.5, color=C_OUT, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.3,
                            connectionstyle="arc3,rad=-0.1"))

ax.annotate("k  innermost loop\n→ contracted away",
            xy=(4.5, py - 1.5 + 0.1), xytext=(7.5, py - 1.3),
            fontsize=9.5, color=C_K, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=C_K, lw=1.3,
                            connectionstyle="arc3,rad=-0.1"))

# Shape note
ax.text(0.5, py - 3.0,
        "Shape:  A(2×2) , B(2×2)  →  C(2×2)     k=2 matched and contracted",
        fontsize=10, fontfamily="monospace", color=MEDIUM)


# ───────────────────────────────────────────────────────────
#  SECTION 3: WHY the k axis determines which product
# ───────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 14)
ax.set_ylim(-4.5, 6)

ax.text(0.5, 5.5, "Step 3 — The k Axis Determines WHICH Matrix Product",
        fontsize=14, fontweight="bold", color=C_WARN)

ax.text(0.5, 4.7,
        "The reading rule is always the same.  But WHERE you put  k  changes the result.",
        fontsize=11, color=MEDIUM)
ax.text(0.5, 4.1,
        "k  is the shared label — it tells einsum which axes to pair and contract.",
        fontsize=11, color=MEDIUM)

# Three variants
variants = [
    ("i k, k j -> i j",  "A @ B",    "standard",
     "k = cols of A, rows of B",
     "Pair A's columns with B's rows"),
    ("i k, j k -> i j",  "A @ B.T",  "this lecture",
     "k = cols of A, cols of B",
     "Pair A's columns with B's columns"),
    ("k i, k j -> i j",  "A.T @ B",  "variant",
     "k = rows of A, rows of B",
     "Pair A's rows with B's rows"),
]

for idx, (pattern, equiv, note, k_meaning, explanation) in enumerate(variants):
    bx = 0.5
    by = 3.0 - idx * 2.3
    bw = 13.0
    bh = 1.8

    # Highlight the lecture's variant
    border_col = C_WARN if idx == 1 else SUBTLE
    fill_alpha = 0.08 if idx == 1 else 0.03
    ax.add_patch(patches.FancyBboxPatch(
        (bx, by), bw, bh, boxstyle="round,pad=0.12",
        facecolor=border_col, alpha=fill_alpha,
        edgecolor=border_col, linewidth=2 if idx == 1 else 1))

    ax.text(bx + 0.3, by + bh - 0.35, pattern, fontsize=14,
            fontweight="bold", color=C_OUT, fontfamily="monospace")
    ax.text(bx + 4.5, by + bh - 0.35, f"=  {equiv}", fontsize=12,
            color=TEXT, fontweight="bold", fontfamily="monospace")
    if idx == 1:
        ax.text(bx + 8.5, by + bh - 0.35, f"← {note}", fontsize=10,
                color=C_WARN, fontweight="bold")
    else:
        ax.text(bx + 8.5, by + bh - 0.35, f"({note})", fontsize=10,
                color=SUBTLE)

    ax.text(bx + 0.3, by + 0.55, k_meaning, fontsize=10,
            color=C_K, fontweight="bold")
    ax.text(bx + 0.3, by + 0.15, explanation, fontsize=9.5,
            color=MEDIUM)

# Key insight
ax.text(7.0, -3.3,
        "Same rule, same mechanics — the label placement is the ONLY thing that changes.",
        fontsize=10.5, ha="center", color=C_OUT, fontweight="bold")


# ───────────────────────────────────────────────────────────
#  SECTION 4: The General Rule + Building Blocks
# ───────────────────────────────────────────────────────────
ax = axes[3]
ax.set_xlim(0, 14)
ax.set_ylim(-4, 5.5)

ax.text(0.5, 5.0, "Step 4 — The General Rule", fontsize=14,
        fontweight="bold", color=C_RULE)

# Three-column layout
c1x = 0.5
ax.add_patch(patches.FancyBboxPatch(
    (c1x, 2.2), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_K, alpha=0.08, edgecolor=C_K, linewidth=1.5))
ax.text(c1x + 1.9, 4.2, "INPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_K)
ax.text(c1x + 1.9, 3.55, "i k , j k", fontsize=18, ha="center",
        fontweight="bold", color=C_K, fontfamily="monospace")
ax.text(c1x + 1.9, 2.8, "k  shared → paired (×)",
        fontsize=9.5, ha="center", color=C_K, fontweight="bold")
ax.text(c1x + 1.9, 2.4, "i  in A only,  j  in B only",
        fontsize=9.5, ha="center", color=MEDIUM)

ax.annotate("", xy=(4.8, 3.4), xytext=(4.4, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c2x = 5.0
ax.add_patch(patches.FancyBboxPatch(
    (c2x, 2.2), 3.8, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_OUT, alpha=0.08, edgecolor=C_OUT, linewidth=1.5))
ax.text(c2x + 1.9, 4.2, "OUTPUT labels", fontsize=11,
        ha="center", fontweight="bold", color=C_OUT)
ax.text(c2x + 1.9, 3.55, "i  j", fontsize=22, ha="center",
        fontweight="bold", color=C_OUT, fontfamily="monospace")
ax.text(c2x + 1.9, 2.8, "i, j  KEPT  →  matrix output",
        fontsize=9.5, ha="center", color=C_OUT, fontweight="bold")
ax.text(c2x + 1.9, 2.4, "k  GONE  →  contracted",
        fontsize=9.5, ha="center", color=C_K, fontweight="bold")

ax.annotate("", xy=(9.3, 3.4), xytext=(8.9, 3.4),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

c3x = 9.5
ax.add_patch(patches.FancyBboxPatch(
    (c3x, 2.2), 4.0, 2.3, boxstyle="round,pad=0.15",
    facecolor=C_RULE, alpha=0.08, edgecolor=C_RULE, linewidth=1.5))
ax.text(c3x + 2.0, 4.2, "RESULT", fontsize=11,
        ha="center", fontweight="bold", color=C_RULE)
ax.text(c3x + 2.0, 3.55, "Σ_k A[i,k]×B[j,k]", fontsize=12, ha="center",
        fontweight="bold", color=C_RULE, fontfamily="monospace")
ax.text(c3x + 2.0, 2.8, "dot product per (row, row) pair",
        fontsize=9.5, ha="center", color=C_RULE)
ax.text(c3x + 2.0, 2.4, "output is a matrix (i × j)",
        fontsize=9.5, ha="center", color=MEDIUM)

# ── How it builds on previous patterns ──
ax.text(7.0, 1.5, "Matrix-Matrix Product Builds on Everything Before",
        fontsize=11.5, ha="center", fontweight="bold", color=C_OUT)

steps = [
    ("Shared  k", "→  PAIR entries from A and B",
     "(same as dot product's shared  i )", C_K),
    ("k  absent from output", "→  SUM products over  k",
     "(same as any vanishing axis)", C_K),
    ("i  present in output", "→  KEEP one slot per A-row",
     "(same as row sums keeping  i )", C_I),
    ("j  present in output", "→  KEEP one slot per B-row",
     "(same as outer product's  j )", C_J),
]

for idx, (label, action, callback, col) in enumerate(steps):
    y = 0.7 - idx * 0.6
    ax.text(1.0, y, label, fontsize=10.5, color=col, fontweight="bold")
    ax.text(4.0, y, action, fontsize=10, color=MEDIUM)
    ax.text(9.0, y, callback, fontsize=9, color=SUBTLE, fontstyle="italic")

# Bottom insight
insight_box(ax, 1.0, -3.3, [
    ("Matrix multiplication = grid of dot products, indexed by the two surviving axes", True),
    ("The shared label  k  is the \"language\" of the dot product — it gets consumed", True),
    ("WHERE you place  k  determines which product:  A@B  vs  A@B.T  vs  A.T@B", False),
    ("All matrix products are the same einsum rule — only label positions differ", False),
], w=12, line_h=0.5)


# ── Save ──────────────────────────────────────────────────
out = ("/Users/dominiclarner/Documents/GitHub/Stanford-CS221-Self-Study-2026"
       "/lecture_2_visuals/einsum_matrix_matrix_product.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
