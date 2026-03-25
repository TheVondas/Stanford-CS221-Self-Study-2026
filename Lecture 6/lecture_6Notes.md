# CS221 Lecture 6 — Search II (UCS and A*)

## Learning Objectives

By the end of this lecture, you should be able to:

- explain why Lecture 5 methods need to be extended once search problems contain cycles
- distinguish clearly between **past cost** and **future cost**
- state the core idea of **Uniform-Cost Search (UCS)** and why it works with cycles
- explain why UCS requires **non-negative edge costs**
- trace UCS step by step on a graph using a **priority queue**, **frontier**, **explored set**, and **backpointers**
- understand the theorem that when UCS removes a state from the frontier, its priority equals its true minimum **past cost**
- explain why UCS can still be wasteful even though it is exact
- define **A\*** as “UCS on modified costs”
- understand what a **heuristic** is and why A\* can still be exact despite using one
- define **consistency** and explain why it is the key condition for A\* graph search
- understand the **telescoping sum** argument showing that A\* preserves optimality
- explain how to build heuristics from **relaxations**
- recognize three useful relaxation patterns:
  - closed-form relaxed solutions
  - fewer states
  - independent subproblems
- explain why taking the **max** of two consistent heuristics is still consistent

---

## Concept Inventory

### 1. Why search again?

Lecture 5 introduced search as the missing ingredient beyond pure learning. Learning gives a predictor that maps input to output, often reflexively. But many important problems are not reflexive:

- planning a route
- solving a puzzle
- doing multi-step reasoning
- proving something
- writing a correct sequence of operations

These require searching over **sequences of actions**, not just making a one-shot prediction.

---

### 2. Search problem formalism

A search problem is defined by:

- a **start state**
- a **successor function** giving possible actions, successor states, and action costs
- an **end test**
- an objective: minimize total path cost

If a solution consists of steps with costs $c_1, c_2, \dots, c_k$, then

$$
\text{Cost}(\text{solution}) = \sum_{i=1}^k c_i
$$

The goal is to return the **sequence of actions** with minimum total cost.

---

### 3. Recap from Lecture 5

Previously, we saw:

**Exact methods**
- exhaustive search
- dynamic programming

**Approximate methods**
- best-of-$n$
- beam search

The key exact object from last lecture was the **future cost**:

$$
\text{FutureCost}(s) = \text{minimum cost from } s \text{ to an end state}
$$

Dynamic programming computes this recursively:

$$
\text{FutureCost}(s) = \min_{(a,c,s') \in \text{Succ}(s)} \left[c + \text{FutureCost}(s')\right]
$$

This works well when the state graph has **no cycles**.

---

### 4. The new problem: cycles

Dynamic programming relies on a clean ordering: you compute values for “later” states before “earlier” ones. That fails when there are cycles.

If state $A$ depends on $B$ and $B$ depends on $A$, there is no topological ordering.

So the problem becomes:

> If there are cycles, in what order should we process states?

That is the main motivation for UCS.

---

### 5. Future cost vs past cost

This lecture introduces a second crucial quantity.

### Future cost

$$
\text{FutureCost}(s) = \min_{\pi: s \to \text{end}} \text{Cost}(\pi)
$$

This looks **forward** from state $s$ to the goal.

### Past cost

$$
\text{PastCost}(s) = \min_{\pi: \text{start} \to s} \text{Cost}(\pi)
$$

This looks **backward** from $s$ to the start.

A very useful mental picture:

- **past cost** = best way to get **to** a state
- **future cost** = best way to get **from** that state to the goal

And:

$$
\text{PastCost}(s) + \text{FutureCost}(s)
$$

is the cost of the best complete solution that passes through $s$.

---

### 6. The shift from DP to UCS

Dynamic programming:
- computes **future costs**
- works backward from the end
- assumes **no cycles**

Uniform-Cost Search:
- computes **past costs**
- grows outward from the start
- can handle **cycles**
- assumes **non-negative costs**

This is the key trade:

> To allow cycles, we stop computing future costs directly and instead compute past costs in increasing order.

---

### 7. Uniform-Cost Search (UCS)

UCS is also known as [[Dijkstra’s algorithm]].

Its central rule is:

> Always process the frontier state with the smallest current priority.

The lecture later proves that this priority is the true minimum **past cost** once the state is removed from the frontier.

So UCS processes states in **non-decreasing past cost order**.

---

### 8. Three buckets of states in UCS

UCS partitions states into three groups:

#### Explored
States whose minimum-cost path has already been finalized.

#### Frontier
States we have discovered, but for which we are still maintaining the best known route so far.

#### Unexplored
States we have not yet seen.

The algorithm repeatedly moves the lowest-priority frontier state into explored.

---

### 9. Why a priority queue?

A priority queue supports the essential operation:

- remove the state with smallest priority

That is exactly what UCS needs.

Each frontier state has a priority equal to the best cost found so far to reach it.

Later, the theorem shows that when a state is popped, that priority is actually its true minimum past cost.

---

### 10. Backpointers

To recover the actual solution path, not just the final cost, UCS stores a **backpointer** for each reached state.

A backpointer records:

- previous state
- action taken
- cost of that action

Once UCS reaches an end state, it reconstructs the path by walking backward through backpointers.

This matters because in search we usually want the **action sequence**, not just the number.

---

## Slide-by-Slide Walkthrough

## Slide 1 — Recap: why search?

The lecture begins by reconnecting search to the broader AI story:

- learning is powerful, but not enough for multi-step reasoning
- search is needed when solving a problem requires planning and structured exploration
- last lecture formalized search problems and introduced exact and approximate methods

This is important context: Lecture 6 is not replacing Lecture 5, but extending it to harder graphs.

---

## Slide 2 — Today’s agenda

The lecture narrows focus to **exact** search algorithms that can handle **cycles**:

- Uniform-Cost Search
- A\*

A\* will eventually be shown to be “UCS in disguise,” which is a very important unifying idea.

---

## Slide 3 — Two mathematical quantities: future cost and past cost

The lecture asks you to internalize two idealized quantities:

- future cost: best remaining cost from a state to the goal
- past cost: best accumulated cost from the start to a state

This is not just terminology. The entire lecture depends on separating these two roles.

Lecture 5 mostly reasoned in terms of **future cost**.

Lecture 6 will pivot to **past cost**.

---

## Slide 4 — Why dynamic programming breaks on cycles

Dynamic programming uses the recurrence for future cost, which requires successor values first.

That means it naturally computes values from the goal backward.

But with cycles, that dependency graph is circular, so the recurrence no longer gives a clean computation order.

Intuition:
- acyclic graph: there is a clean “later to earlier” order
- cyclic graph: no such order exists

So we need a new ordering principle.

---

## Slide 5 — Diamond graph motivation

The lecture’s motivating cyclic graph is:

- $A \leftrightarrow B$ with cost $1$
- $A \leftrightarrow C$ with cost $100$
- $B \leftrightarrow C$ with cost $1$
- $B \leftrightarrow D$ with cost $100$
- $C \leftrightarrow D$ with cost $1$

Start: $A$  
End: $D$

At a glance, the best route is not the obvious direct-looking one.

Possible routes:
- $A \to B \to D$: cost $1 + 100 = 101$
- $A \to C \to D$: cost $100 + 1 = 101$
- $A \to B \to C \to D$: cost $1 + 1 + 1 = 3$

This is the point:
- local appearances can be misleading
- cycles create ambiguity
- we need a systematic algorithm

---

## Slide 6 — UCS high-level idea

UCS says:

1. start at the start state with cost $0$
2. repeatedly remove the frontier state with minimum priority
3. expand its successors
4. update frontier costs if a cheaper route is found
5. stop when an end state is popped

This does not use a topological order.

Instead, it uses **cost order**.

That is the decisive innovation.

---

## Slide 7 — UCS walk-through on the diamond graph

Initial frontier:

- $A: 0$

Pop $A$:
- explore successors
- add $B: 1$
- add $C: 100$

Frontier now:
- $B: 1$
- $C: 100$

Pop $B$:
- $A$ is already explored, ignore
- reaching $C$ via $B$ gives cost $1 + 1 = 2$, which improves $C$ from $100$ to $2$
- reaching $D$ via $B$ gives cost $1 + 100 = 101$

Frontier now:
- $C: 2$
- $D: 101$

Pop $C$:
- $A$ and $B$ already explored
- reaching $D$ via $C$ gives cost $2 + 1 = 3$, improving $D$ from $101$ to $3$

Frontier now:
- $D: 3$

Pop $D$:
- $D$ is an end state, stop

Recovered path via backpointers:

$$
A \to B \to C \to D
$$

with total cost

$$
3
$$

This example shows two crucial things:
- UCS can revise a state’s frontier priority when it finds a cheaper route
- UCS does not commit too early; it only finalizes a state when it is popped

---

## Slide 8 — What exactly is being updated?

This is one of the most important implementation details.

When UCS considers successor $s'$ from current state $s$ with current past cost $g(s)$, it proposes a new candidate cost:

$$
g(s) + c(s,a)
$$

If this is smaller than the state’s current frontier priority, UCS updates:

- the state’s priority
- the backpointer

So frontier means:

> “I know at least one way to get here, but I may still discover a better one.”

Explored means:

> “I now know the best possible way to get here.”

---

## Slide 9 — Grid example

The lecture then moves to a grid navigation problem:

- states are grid cells
- actions are up, down, left, right
- each legal move costs $1$
- walls are blocked cells

Start at $S$, end at $E$.

The specific grid’s shortest path cost is:

$$
14
$$

This example helps ground UCS in a more spatial setting.

The important lesson is not just the answer 14, but the behavior:
- UCS expands outward from the start
- it does not yet know where the goal is “in a smart way”
- it simply explores by increasing path cost

---

## Slide 10 — Why UCS is called “uniform-cost”

On the large map visualization, UCS expands like a wave.

That is why it is called **uniform-cost** search:
- it spreads out more or less uniformly in all directions
- it prioritizes closeness to the **start**, not closeness to the **goal**

This is exact and safe, but potentially wasteful.

A large amount of computation may go into areas that have no realistic chance of helping reach the goal efficiently.

This sets up the need for A\*.

---

## Slide 11 — UCS correctness theorem

The key theorem is:

> When UCS moves a state $s$ from the frontier to the explored set, its priority equals $\text{PastCost}(s)$.

This is the core correctness property.

It means:
- once a state is popped, its best route is finalized
- UCS never later discovers an even cheaper route to an explored state

That is why explored states can be treated as “done.”

---

## Slide 12 — Base case of the proof

For the start state:

$$
\text{priority}(\text{start}) = \text{PastCost}(\text{start}) = 0
$$

This is immediate.

No action is needed to be at the start.

---

## Slide 13 — Inductive step intuition

Assume every already explored state has correct priority.

Now consider the next frontier state $s$ being popped.

UCS is claiming:
- the path giving priority$(s)$ is not just a discovered path
- it is the best possible path

To prove that, compare:
- the **blue path**: the path UCS currently believes is best
- any **red path**: an arbitrary alternative route to $s$

The proof shows every alternative red path must cost at least as much as the blue one.

---

## Slide 14 — Core inequality chain in the proof

The lecture’s proof idea is:

1. Any alternative red path must cross from explored into frontier somewhere, say from $t$ to $u$.
2. Because $t$ is explored, its priority already equals its true past cost.
3. When $t$ was explored, UCS would have used it to update $u$.
4. Since $s$ was chosen as the minimum-priority frontier state, priority$(s)$ cannot exceed priority$(u)$.
5. Since all costs are non-negative, detours after crossing into frontier cannot magically reduce total cost.

So the red path cannot beat the blue path.

This is where the **non-negative cost** assumption matters.

---

## Slide 15 — Why non-negative costs are necessary

If negative edges were allowed, you could:

- go far away
- accumulate a huge negative cost
- come back cheaper than expected

That would break UCS’s “finalize when popped” logic.

So UCS needs:

$$
c(s,a) \ge 0
$$

for all actions.

This is not a cosmetic assumption. It is structurally necessary.

---

## Slide 16 — UCS summary

At this stage, the lecture summarizes UCS:

- computes past costs
- processes states in non-decreasing past cost order
- uses a priority queue
- is exact
- handles cycles
- requires non-negative costs

This is a major upgrade over dynamic programming for cyclic graphs.

---

## Slide 17 — But UCS is still inefficient

Even though UCS is correct, it may explore many irrelevant states.

On the map example, UCS expands into regions that obviously do not point toward the goal.

Why?

Because UCS knows only:
- “how far from the start am I?”

It does **not** know:
- “how promising is this state for eventually reaching the end?”

That missing ingredient is exactly what A\* adds.

---

## Slide 18 — The dream ordering: past cost + future cost

Ideally, you would like to explore states by:

$$
\text{PastCost}(s) + \text{FutureCost}(s)
$$

This would be amazing because it accounts for:
- cost already spent
- cost still needed

In standard A\* notation, these are often written as:

- $g(s) = \text{PastCost}(s)$
- $h^*(s) = \text{FutureCost}(s)$
- $f(s) = g(s) + h^*(s)$

The problem is that computing true future cost is basically solving the search problem.

So we approximate it.

---

## Slide 19 — Heuristic function

A **heuristic** is a function

$$
h(s)
$$

that approximates future cost.

The lecture uses the word in a very technical sense:

> a numeric estimate of remaining cost-to-go from state $s$ to the goal

A good heuristic should usually be:
- cheap to compute
- informative
- safe enough to preserve exactness when used in A\*

---

## Slide 20 — A\* as UCS on modified costs

This is the lecture’s central conceptual move.

A\* is defined as:

> run UCS, but on modified edge costs

For a transition from $s$ to successor $s'$ via action $a$:

$$
c'(s,a) = c(s,a) + h(s') - h(s)
$$

This looks strange at first, but it is incredibly important.

Interpretation:
- if an action moves you closer to the goal according to the heuristic, then $h(s') - h(s)$ is negative, so the modified cost becomes smaller
- if an action moves you away from the goal, the modified cost becomes larger

So the heuristic tilts the search toward states that appear closer to the goal.

---

## Slide 21 — Line-search example

The lecture gives a line of states:

$$
\dots, -2, -1, 0, 1, 2, \dots
$$

Start: $0$  
End: $2$

Each move left or right costs $1$.

Without a heuristic, UCS would expand both directions.

The lecture defines

$$
h(s) = 2 - s
$$

So:

- $h(2)=0$
- $h(1)=1$
- $h(0)=2$
- $h(-1)=3$
- $h(-2)=4$

This heuristic strongly favors moving right.

---

## Slide 22 — Modified costs in the line example

From state $0$:

### Going right to $1$

Original cost:
$$
1
$$

Modified cost:
$$
1 + h(1) - h(0) = 1 + 1 - 2 = 0
$$

### Going left to $-1$

Modified cost:
$$
1 + h(-1) - h(0) = 1 + 3 - 2 = 2
$$

So A\* makes the right move “cheaper” in the modified problem.

This does not mean the original problem changed. It means the **search order** changed.

That distinction matters.

---

## Slide 23 — Why the implementation is elegant

The lecture code makes this beautifully simple:

1. define a wrapper around the original search problem
2. keep states and actions the same
3. replace original costs with modified costs
4. call UCS

So A\* is not implemented as a completely separate engine.

It is literally:

> UCS applied to a transformed search problem

That is why Percy says A\* is “UCS in disguise.”

---

## Slide 24 — Recovering original path costs

Because UCS runs on modified costs, the resulting action sequence is correct, but the numerical step costs stored in that modified solution are not the original costs.

So after solving, the algorithm reconstructs original step costs using:

$$
c(s,a) = c'(s,a) - h(s') + h(s)
$$

This is bookkeeping, not new search logic.

Important:
- modified costs guide exploration
- original costs are what define the real objective

---

## Slide 25 — Not every heuristic works

The lecture gives a counterexample where a bad heuristic assigns a huge value to one state and effectively sabotages the search.

It can even produce **negative modified costs**, which breaks UCS.

This is a critical lesson:

> A\* is not “use any heuristic and hope.”

There is a precise condition required.

---

## Slide 26 — Consistency

A heuristic is **consistent** if:

$$
h(\text{end}) = 0
$$

and for every transition from $s$ to $s'$:

$$
h(s) \le c(s,a) + h(s')
$$

Equivalently,

$$
c(s,a) + h(s') - h(s) \ge 0
$$

That second form is usually the easiest one to remember in the context of this lecture, because it says:

> all modified edge costs must be non-negative

Since UCS needs non-negative costs, consistency is exactly the condition that makes A\* safe as UCS-on-modified-costs.

This is the cleanest conceptual payoff of the lecture.

---

## Slide 27 — Why A\* is correct: telescoping sum

Suppose a path is:

$$
s_0 \to s_1 \to \dots \to s_k
$$

with $s_0 = \text{start}$ and $s_k = \text{end}$.

The modified path cost is:

$$
\sum_{i=0}^{k-1} \left[c(s_i,a_i) + h(s_{i+1}) - h(s_i)\right]
$$

Expanding gives:

$$
\sum_{i=0}^{k-1} c(s_i,a_i) + \sum_{i=0}^{k-1} \left[h(s_{i+1}) - h(s_i)\right]
$$

The heuristic terms telescope:

$$
h(s_1)-h(s_0)+h(s_2)-h(s_1)+\dots+h(s_k)-h(s_{k-1})
= h(s_k)-h(s_0)
$$

Since $h(\text{end}) = 0$:

$$
\text{ModifiedCost}(\text{path}) = \text{OriginalCost}(\text{path}) - h(\text{start})
$$

And $h(\text{start})$ is just a constant across all complete start-to-end paths.

Therefore:

- minimizing modified cost
- minimizing original cost

are exactly the same optimization problem.

That is the core correctness argument.

---

## Slide 28 — Why A\* can be faster

UCS explores in order of low past cost.

A\* explores in order of low:

$$
\text{PastCost}(s) + h(s)
$$

So A\* is trying to prefer states that are:
- cheap so far
- likely cheap to finish from

The lecture emphasizes three cases:

### Case 1: $h(s)=0$
Then

$$
A^* = UCS
$$

No benefit.

### Case 2: $h(s)=\text{FutureCost}(s)$
Then A\* is perfect and explores only states on an optimal path.

Amazing, but usually impossible because that is the exact answer we are trying to approximate.

### Case 3: realistic heuristics
Heuristics lie somewhere in between:
- informative enough to reduce search
- cheap enough to compute
- consistent enough to preserve optimality

This is the actual design game.

---

## Slide 29 — Admissibility vs consistency

The lecture briefly mentions **admissibility**:

$$
h(s) \le \text{FutureCost}(s)
$$

So admissibility means the heuristic never overestimates true remaining cost.

However, for this lecture and graph search, **consistency** is the more important condition.

Key relationship:

- consistency implies admissibility
- admissibility alone is weaker
- consistency is the condition used here because A\* is being framed as UCS on modified non-negative costs

So for your notes, keep the hierarchy straight:

$$
\text{consistency} \implies \text{admissibility}
$$

but not necessarily the other way around.

---

## Slide 30 — The practical problem: where do heuristics come from?

At this point, the lecture shifts from theory to construction.

We know A\* needs a good heuristic.

But how do we get one?

The answer is the most important practical idea of the lecture:

> Build heuristics by solving a relaxed version of the original problem.

---

## Slide 31 — Relaxation recipe

The lecture gives a general recipe:

1. define a relaxed problem by removing some constraints
2. compute the future cost in that relaxed problem
3. use that future cost as the heuristic for the original problem

Formally:

$$
h(s) = \text{FutureCost}_{\text{relaxed}}(s)
$$

This is brilliant because it gives you a heuristic with structure and meaning, not an arbitrary guess.

---

## Slide 32 — Relaxation example 1: remove walls in a grid

Original problem:
- grid world with walls

Relaxed problem:
- same grid, but walls are removed

Now the shortest path from any state $(r,c)$ to the goal is just Manhattan distance.

If goal is $(r_g, c_g)$, then:

$$
h(r,c) = |r-r_g| + |c-c_g|
$$

This is a closed-form solution to the relaxed problem.

That makes it very fast to compute.

Intuition:
- it ignores obstacles
- so it underestimates or matches the true cost
- it is informative because being geometrically closer to the goal is usually helpful

This is one of the classic A\* heuristics.

---

## Slide 33 — Why the heuristic is imperfect but still useful

The lecture points out that Manhattan distance is not exact in the original blocked grid.

A state can look very close to the goal geometrically but still require a long detour because of walls.

That is fine.

A heuristic does **not** need to be exact to be useful.

It only needs to preserve the right correctness conditions and reduce wasted exploration.

---

## Slide 34 — Relaxation example 2: fewer states via free tram

Recall the limited travel problem from the previous lecture:

- you want to go from location $1$ to location $n$
- actions are walk and tram
- tram use is limited by tickets

Original state:
- location
- number of tickets

That makes the state space larger.

Relaxed problem:
- tram is free again
- state only needs location

Now the relaxed problem has fewer states, so it is cheaper to solve.

The heuristic is:

$$
h(\text{loc}, \text{tickets}) = \text{FutureCost}_{\text{relaxed}}(\text{loc})
$$

In other words, project the richer original state down to the simpler relaxed state.

This is a very general trick.

---

## Slide 35 — Accounting honestly for heuristic computation cost

A subtle but important point from the lecture:

If you solve the relaxed problem in order to get your heuristic, that preprocessing itself costs computation.

So if you compare UCS and A\*, you should count:

- search work in the original problem
- plus the cost of solving the relaxed problem

In the lecture’s example:
- UCS explored 23 states
- A\* explored 8 in the original problem
- but the relaxed problem cost another 10
- so the total should be compared as 18 vs 23, not 8 vs 23

This is very good algorithmic hygiene.

---

## Slide 36 — What if the relaxed problem has cycles too?

Dynamic programming cannot compute future costs on cyclic relaxed problems either.

So what then?

The lecture introduces a neat trick:

### Reverse the relaxed problem

If there is an edge:

$$
A \to B
$$

then in the reversed problem you create:

$$
B \to A
$$

Why does this help?

Because:

- **past costs** in the reversed problem
- equal **future costs** in the original relaxed problem

So by running UCS on the reversed relaxed problem, you can compute the future-cost heuristic you need.

This is a very elegant reuse of machinery.

---

## Slide 37 — Relaxation example 3: independent subproblems (8-puzzle)

Original problem:
- tiles cannot overlap

Relaxed problem:
- tiles are allowed to overlap

Once that constraint is removed, each tile can be treated independently.

So instead of one big tangled search problem, you get several independent subproblems.

In the 8-puzzle example, each tile’s distance to its target can be computed separately, and the heuristic is the sum of those distances.

This is powerful because it shows a broader pattern:

> Removing interactions between objects can decompose one hard problem into many easy ones.

This idea appears all over AI and optimization.

---

## Slide 38 — The unifying principle behind relaxation

The lecture now generalizes.

All these relaxations can be understood as reducing costs.

Sometimes an impossible move was effectively assigned infinite cost. Relaxation changes that to a finite cost.

Formally, a relaxation keeps:

- the same states
- the same actions
- the same successor structure

but changes costs so that:

$$
c_{\text{relaxed}}(s,a) \le c(s,a)
$$

This is the formal definition.

---

## Slide 39 — Why relaxation implies consistency

If

$$
h(s) = \text{FutureCost}_{\text{relaxed}}(s)
$$

then by the triangle inequality in the relaxed problem:

$$
h(s) \le c_{\text{relaxed}}(s,a) + h(s')
$$

Since relaxed costs are no larger than original costs:

$$
c_{\text{relaxed}}(s,a) \le c(s,a)
$$

therefore:

$$
h(s) \le c(s,a) + h(s')
$$

which is exactly consistency.

So:

> future cost of a relaxed problem is automatically a consistent heuristic for the original problem.

This is the key theorem that makes relaxation so useful.

You no longer have to guess whether a heuristic is safe. If it comes from a proper relaxation, you get consistency “for free.”

---

## Slide 40 — Important caution: not every relaxation is useful

A relaxed problem is not automatically easier.

If you reduce costs randomly, you might get a mathematically valid relaxation that gives a terrible heuristic or is still hard to solve.

The lecture emphasizes that good relaxations usually reduce complexity in one of three structured ways:

- closed-form solution
- fewer states
- independent subproblems

That is the practical design lens to keep.

---

## Slide 41 — Combining heuristics

Suppose you have two consistent heuristics:

- $h_1(s)$
- $h_2(s)$

Which one should you use?

The lecture’s answer:

$$
h(s) = \max(h_1(s), h_2(s))
$$

This new heuristic is still consistent.

Why is this useful?

Because a larger consistent heuristic is usually more informative and prunes more search, while still preserving correctness.

So you do not have to choose between good heuristic ideas. You can combine them by taking their max.

This is another standard A\* technique worth remembering.

---

## Slide 42 — Final summary of the lecture

This lecture gives you the full conceptual progression:

1. dynamic programming handles acyclic search using future costs
2. cycles break that approach
3. UCS handles cycles by computing past costs in increasing order
4. A\* improves UCS by biasing search toward states that look closer to the goal
5. the correct way to do that is via consistent heuristics
6. the principled way to build such heuristics is via relaxations

That is the structure you should be able to reproduce from memory.

---

## Core Equations to Memorize

### Search objective

$$
\text{Cost}(\text{solution}) = \sum_i c_i
$$

### Future cost recurrence

$$
\text{FutureCost}(s) = \min_{(a,c,s') \in \text{Succ}(s)} \left[c + \text{FutureCost}(s')\right]
$$

### Past cost

$$
\text{PastCost}(s) = \min_{\pi:\text{start} \to s} \text{Cost}(\pi)
$$

### A\* modified cost

$$
c'(s,a) = c(s,a) + h(s') - h(s)
$$

### Consistency

$$
h(s) \le c(s,a) + h(s')
$$

equivalently,

$$
c(s,a) + h(s') - h(s) \ge 0
$$

### Admissibility

$$
h(s) \le \text{FutureCost}(s)
$$

### Relaxation

$$
c_{\text{relaxed}}(s,a) \le c(s,a)
$$

### Heuristic from relaxation

$$
h(s) = \text{FutureCost}_{\text{relaxed}}(s)
$$

### Max of heuristics

$$
h(s)=\max(h_1(s), h_2(s))
$$

---

## Common Confusions and How to Fix Them

### Confusion 1: “Is UCS computing future cost or past cost?”

UCS computes **past cost**.

It starts at the start state and grows outward, finalizing the cheapest known route **to** each state.

Dynamic programming in the earlier lecture was computing **future cost**.

Do not mix them up.

---

### Confusion 2: “Why can’t we just use any heuristic?”

Because A\* is implemented as UCS on modified costs.

If the heuristic makes modified costs negative, UCS’s correctness guarantee breaks.

That is why consistency matters.

---

### Confusion 3: “Does A\* change the actual optimization problem?”

No.

A\* changes the **search order**, not the real objective.

The telescoping argument shows all complete paths have their costs shifted by the same constant $-h(\text{start})$.

So optimal paths remain optimal.

---

### Confusion 4: “If a heuristic is not exact, why is it still useful?”

Because exactness is not the point.

A heuristic only needs to be informative enough to guide exploration and structured enough to preserve correctness.

Manhattan distance in a blocked grid is not exact, but it still helps a lot.

---

### Confusion 5: “Why is consistency stronger than admissibility?”

Admissibility only says:

- never overestimate future cost

Consistency says more:

- every single edge must satisfy the triangle-inequality-style constraint

That extra structure is what makes graph-search A\* work cleanly as UCS on non-negative modified costs.

---

## Summary

Lecture 6 is really about one big transition:

- Lecture 5: exact search in acyclic settings, mainly through **future cost**
- Lecture 6: exact search in cyclic settings, mainly through **past cost**

The two headline algorithms are:

### Uniform-Cost Search
- exact
- handles cycles
- computes states in increasing past cost order
- requires non-negative edge costs

### A\*
- also exact, under consistent heuristics
- is just UCS on modified costs
- uses a heuristic estimate of future cost to reduce wasted exploration

The most important conceptual tool for building good heuristics is **relaxation**:
- remove constraints
- solve the easier problem
- use that easier problem’s future cost as the heuristic

That gives you a principled route from:
- domain knowledge
- to heuristic
- to faster exact search

---

## Real-World Applications

### Route planning and maps
UCS corresponds to shortest-path algorithms like Dijkstra’s. A\* is widely used in navigation systems because it can use distance-to-go information to focus search toward the destination.

### Robotics and motion planning
Robots often need to search over positions, orientations, and action sequences. Relaxed models of the world can supply heuristics that speed up exact planning.

### Puzzle solving
8-puzzle, 15-puzzle, Rubik-like search spaces, and many combinatorial planning problems rely on A\* with carefully engineered heuristics from relaxations.

### Games and planning
Even when exact search becomes too expensive, the ideas of:
- state abstraction
- cost-to-go estimates
- decomposition
- relaxed constraints

remain foundational.

### AI systems more broadly
This lecture shows a recurring AI pattern:
- solve a simpler version of the problem
- use that solution to guide the harder one

That pattern reappears in search, optimization, probabilistic inference, and even modern test-time reasoning systems.

---

## Final Takeaways

If you only remember five things, remember these:

1. **Cycles break dynamic programming’s clean computation order.**
2. **UCS fixes this by processing states in increasing past cost.**
3. **A\* is UCS on modified costs, guided by a heuristic.**
4. **Consistency is what makes A\* exact in graph search.**
5. **The best heuristics often come from relaxations.**
