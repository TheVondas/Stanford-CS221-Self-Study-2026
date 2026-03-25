# CS221 Lecture 5 — Search I

## Learning Objectives

By the end of this lecture, you should be able to:

- explain why search is still a core AI idea even in the deep learning era
- define a search problem formally using states, successors, costs, and end conditions
- distinguish between modeling a problem and solving a problem
- derive the future-cost recurrence for exact search
- explain how exhaustive search works and why it can be exponentially slow
- explain how dynamic programming improves exhaustive search by caching subproblem results
- identify when dynamic programming helps and when it does not
- explain why cycles create difficulties for naive recursive search
- describe approximate search methods including best-of-$n$ and beam search
- connect search to modern test-time compute for language models

---

## Concept Inventory

### Search
A method for reasoning in deterministic worlds by explicitly considering sequences of actions rather than producing an answer reflexively.

### Search problem
A formal specification of a problem using:

- a start state
- a successor function
- an end test
- action costs

### State
A representation containing all information needed to evaluate future actions, future costs, and successor states.

### Step / successor
A possible move from one state to another, consisting of:

- an action label
- an immediate cost
- a resulting next state

### Solution
A sequence of actions/steps from the start state to an end state.

### Total cost
The sum of all action costs in a solution:

$$
\text{Cost of solution} = \sum_{t=1}^{T} c_t
$$

### Future cost
The minimum possible remaining cost from a state $s$ to some end state.

### Exhaustive search
An exact method that tries all possible solutions.

### Dynamic programming
Exhaustive search plus caching. Solve each state once, then reuse that answer whenever the state appears again.

### Policy
A function mapping a state to an action or successor. It may be deterministic or stochastic.

### Rollout
Executing a policy repeatedly from the start state until termination, thereby producing one complete candidate solution.

### Best-of-$n$
Run $n$ rollouts, then keep the best one.

### Beam search
Keep only the best `beam_width` partial solutions at each depth and discard the rest.

### Test-time compute
Using extra computation at inference time to search for better outputs rather than accepting the first output a model produces.

---

## Slide-by-Slide Walkthrough

### 1. From learning to search
Last week’s material was about machine learning:

- training data consists of input-output pairs
- a learning algorithm turns training data into a predictor
- the predictor maps inputs to outputs
- for regression, the output is numeric
- for classification, the output is one of $K$ labels/classes

This fits the “perception-action-learning” part of intelligence well. A learned predictor is often a fast feed-forward mapping from percepts to actions.

But many intelligent tasks are not reflexive:

- solving a hard math problem
- solving a difficult programming problem
- planning a route
- reasoning through multiple possibilities before acting

That is where search comes in. In this lecture, search is presented as one form of reasoning for deterministic environments.

Core idea: learning gives you a direct mapping when one is feasible; search gives you structured deliberation when one-step reflexes are not enough.

### 2. Why search still matters in 2025
A natural question is: search was central to early symbolic AI in the 1950s, so why revisit it now?

The lecture’s answer is based on Rich Sutton’s essay, *The Bitter Lesson*:

- general methods that leverage computation tend to dominate handcrafted approaches in the long run
- two especially scalable families are learning and search

The point is not “search alone is enough.” That was part of the old mistake. The modern view is:

- search alone is limited
- learning alone is also limited in some settings
- combining learned models with search can be very powerful

This is especially relevant for test-time compute in language models, where you may want to spend extra computation at inference time to search over multiple candidate outputs.

So the strategic claim of the lecture is:

> Modern AI strength often comes from coupling learned cost/probability models with search procedures.

### 3. Formalizing a search problem
The lecture insists on formalization before solution.

You should not first try to solve the word problem in your head. Instead, convert it into a general structure that an algorithm can solve.

A search problem contains three main ingredients:

- `start_state()`: where you begin
- `successors(state)`: what you can do from a state, including actions, costs, and next states
- `is_end(state)`: whether the current state is terminal

This is the modeling step.

Why does this matter?

Because AI methods should work on classes of problems, not just on one hand-solved puzzle. Once the problem is formalized correctly, the algorithm does the rest.

### 4. Example: the walk/tram travel problem
The toy domain is a street with locations numbered $1$ through $n$.

Rules:

- walking from location $i$ to $i+1$ costs $1$
- taking a tram from $i$ to $2i$ costs $2$
- goal: get from $1$ to $n$ with minimum total cost

For this problem:

- state = current location
- start state = $1$
- successors from state $i$ are:
  - walk to $i+1$ with cost $1$, if still within bounds
  - tram to $2i$ with cost $2$, if still within bounds
- end test = whether current location equals $n$

This is the simplest possible version because the state only needs one number: where you currently are.

### 5. Objective: a solution is a minimum-cost action sequence
A solution is not just a destination. It is a sequence of actions.

For example, for $n=10$, one candidate solution is:

- walk
- tram
- walk
- tram

Its cost is:

$$
1 + 2 + 1 + 2 = 6
$$

Important distinction:

- many solutions may exist
- the objective is to find one with minimum total cost
- the minimum-cost value is unique, but the actual minimizing action sequence need not be unique

So a search algorithm should return both:

- the cost
- the action sequence that achieves it

### 6. Adding constraints changes the state
Now the lecture modifies the domain: the tram requires tickets, and you have only a limited number.

This immediately changes the modeling.

If the state were only the location, then the algorithm would not know how many tickets remain. That means it would be unable to judge whether taking the tram is valid.

So the new state must be composite:

$$
\text{state} = (\text{location}, \text{tickets remaining})
$$

Now:

- walking changes location but leaves tickets unchanged
- tram changes location and decreases tickets by $1$
- tram is only a valid successor if tickets remaining $> 0$

This is a major lecture theme:

> The state must contain every piece of information needed to evaluate future actions and successors.

### 7. Why correct modeling matters
The lecture emphasizes a subtle but very important point: if you define the problem incorrectly, the algorithm will exploit your mistake.

Example:

If you forget to check whether tickets are positive before allowing the tram action, the search algorithm may keep taking trams with negative ticket counts.

This is not the algorithm being “wrong.” It is the model being wrong.

Percy explicitly frames this as a kind of reward hacking or system gaming:

- you intended a constrained problem
- your formal definition failed to include the constraint
- the algorithm solved the formal problem you actually wrote, not the one you intended

This is a deep AI lesson that extends far beyond toy search problems.

### 8. Another constraint: cannot take the tram twice in a row
Suppose the rules change again:

- tram rides cannot occur on two consecutive moves

What must change?

The state must now also remember whether the previous action was a tram.

So the state might become:

$$
\text{state} = (\text{location}, \text{tickets remaining}, \text{last action was tram?})
$$

Why is this necessary?

Because legality of a current action depends on previous history.

This illustrates the broader principle:

- state is not just “where you are”
- state is whatever summary of history is needed so that the future can be evaluated correctly

### 9. Why not store the entire history?
A tempting idea is: just store everything that has happened so far. Then you are guaranteed to have enough information.

That is true, but inefficient.

The lecture warns that some algorithms, especially dynamic programming, scale with the number of distinct states. If the state includes full history, then the number of states can explode.

So there is a design tension:

- too little information in the state means invalid modeling
- too much information in the state means computational blow-up

Good modeling means storing exactly the information required for future decisions, and no more.

### 10. Modeling versus solving
At this point, the lecture separates two tasks:

#### Modeling
Translating a natural-language problem into:

- states
- successors
- costs
- end conditions

#### Solving
Running an algorithm on the formal representation to find a good solution.

This separation matters because once the problem becomes complex enough, you often cannot solve it intuitively in your head. But if the formalization is correct, a general algorithm still can.

This is one of the cleanest abstractions in the course so far.

---

## Exact Search

### 11. Goal of exact search
Given a search problem, we want the true minimum-cost solution.

The most direct method is exhaustive search:

- enumerate all possible action sequences
- compute their total costs
- return the best one

This is guaranteed to work in principle, but it may be very expensive.

### 12. The key abstraction: future cost
The lecture introduces the concept of future cost.

For a state $s$, define:

$$
\text{futureCost}(s)
$$

as the minimum cost of getting from $s$ to an end state.

This is the most important idea in the lecture because it creates the recurrence used by both exhaustive search and dynamic programming.

#### Base case
If $s$ is already an end state, then no more actions are needed:

$$
\text{futureCost}(s) = 0
$$

#### Recursive case
If $s$ is not an end state, then any valid solution must begin with some successor step $(a, c, s')$.

So:

$$
\text{futureCost}(s) = \min_{(a,c,s') \in \text{Successors}(s)} \left[c + \text{futureCost}(s')\right]
$$

Interpretation:

- choose a first action
- pay its immediate cost $c$
- then optimally solve the rest from the next state $s'$
- among all possible first actions, keep the minimum

This is the recurrence relation for exact search.

### 13. Why this recurrence is so important
This recurrence is doing several things at once.

It gives us:

- a mathematical definition of optimality
- a recursive decomposition into subproblems
- a bridge to later topics like MDPs, reinforcement learning, and game-playing algorithms

The lecture explicitly chooses this formulation because it generalizes well.

So even if it initially seems like more machinery than necessary, it is foundational for the rest of the course.

### 14. Exhaustive search as recursive expansion
The implementation idea is:

- start from the start state
- recursively compute the best future solution from each successor state
- build complete candidate solutions
- choose the cheapest one

In the lecture code, the helper function returns not just a numeric future cost, but a full solution object containing:

- the steps
- the total cost

That is useful because we do not only want to know the optimum value. We also want the actual action sequence.

### 15. Base case in exhaustive search
If the current state is already an end state, the best future solution is the empty solution:

- no additional steps
- zero additional cost

This mirrors the recurrence base case exactly.

### 16. Example behavior of exhaustive search
For the travel problem with $n=4$, exhaustive search returns a cost-$3$ solution and explores $9$ states total, even though there are only $4$ actual locations.

That reveals the inefficiency:

- the algorithm revisits some states multiple times through different paths

For the larger examples used in the lecture source:

- for $n=10$, exhaustive search still finds the optimal cost $6$
- for $n=17$, it still works, but the number of explored states grows rapidly

The lecture’s big point is not the exact counts. The point is the growth pattern.

### 17. Complexity of exhaustive search
Exhaustive search can take exponential time in the number of states or in the depth of the solution space.

Why?

Because every time a state has multiple successors, the search branches. If branching continues over many levels, the total number of explored paths grows exponentially.

Memory is better behaved than time here:

- recursive exhaustive search mainly stores the current path on the call stack
- so memory is roughly linear in the longest solution length

This is why exhaustive search can be time-impossible before it becomes memory-impossible.

### 18. Cycles break naive exhaustive search
Suppose the graph of states contains a cycle:

$$
A \to B \to C \to A
$$

Now the recursive formulation may never terminate, because the search keeps revisiting states forever.

This creates two issues:

- computational non-termination
- conceptual ill-definedness if cycles can keep improving the cost

The lecture temporarily avoids this by suggesting a practical hack:

- include the number of steps taken so far in the state
- impose a maximum step threshold
- if the threshold is exceeded, treat the resulting path as having effectively infinite cost

This turns an infinite-horizon problem into a finite-horizon one.

### 19. Caveat: choosing the threshold
A student asks the natural question: what if the real solution lies beyond the threshold?

The lecture’s answer is basically:

- there is no universal answer
- choose the threshold based on the problem structure
- if costs are all positive, the number of states often gives a reasonable upper bound

But if negative cycles are possible, the situation becomes trickier.

### 20. Negative costs and degenerate cases
If the problem allows negative-cost cycles, then the “optimal” solution may be to loop forever and drive the total cost toward $-\infty$.

That is a degenerate case. In such settings, optimization behaves very differently and special handling is required.

This matters because some pruning arguments later depend on costs being nonnegative.

---

## Dynamic Programming

### 21. Core idea
Dynamic programming is introduced as:

> exhaustive search + caching

Also called memoization.

The reason it works is that exhaustive search often solves the same subproblem repeatedly.

If the future solution from state $s$ has already been computed once, there is no need to compute it again.

Store it in a cache and reuse it.

### 22. What gets cached?
For each state $s$, cache the best future solution from that state.

So conceptually:

$$
\text{cache}[s] = \text{optimal solution from } s
$$

Then when recursion reaches $s$ again:

- check the cache first
- if present, return the stored answer immediately
- if absent, compute it, store it, and return it

This is only a tiny code change relative to exhaustive search, but it can produce huge speedups.

### 23. Dynamic programming recurrence
The recurrence itself does not change.

We still use:

$$
\text{futureCost}(s) = \min_{(a,c,s') \in \text{Successors}(s)} \left[c + \text{futureCost}(s')\right]
$$

What changes is only whether repeated states are recomputed.

That is an elegant and important lesson:

- the optimization principle stays the same
- the implementation strategy becomes more efficient

### 24. Why dynamic programming can be exponentially faster
If many different paths reach the same state, exhaustive search re-solves the same future problem over and over.

Dynamic programming solves each distinct state once.

So in favorable problems, the runtime drops from exponential in the number of paths to roughly linear in the number of distinct states (more precisely, number of states plus the work of processing their successors).

The lecture’s examples show this starkly:

- exhaustive search explores more states than actually exist because it repeats work
- dynamic programming explores each state once in the travel example

### 25. Example intuition: when DP helps
Dynamic programming helps when the search graph has lots of merging.

Think of a diamond or lattice structure:

- many different paths split apart
- then reconverge into the same states

In that situation:

- exhaustive search repeats the downstream computation from the merged state many times
- dynamic programming computes that downstream answer once and reuses it everywhere

This is the sweet spot for DP.

### 26. When DP does not help much
If every action always leads to a completely new state, there is no repeated subproblem structure.

Then:

- cache hits are rare or nonexistent
- dynamic programming provides little benefit

The lecture notes that some sequence-generation settings, such as raw language generation, can look like this because each generated prefix may be unique.

### 27. Memory tradeoff
The lecture stresses a very practical computational lesson:

- time can often be traded away by running longer
- memory is harder to expand in the moment

Dynamic programming requires storing solutions for states, so its memory cost is proportional to the number of cached states.

Therefore use dynamic programming when:

- the number of states fits in memory
- the problem has enough repeated substructure to justify caching

If the state space is enormous, DP may simply be infeasible even if it would be theoretically elegant.

### 28. Bellman origin of the term
The lecture briefly explains the terminology:

- “dynamic” = sequential decisions over time
- “programming” = optimization, not writing code

This comes from Richard Bellman in the 1950s.

This historical note matters because many students initially misread “dynamic programming” as something about mutable programming languages or writing programs dynamically. It is not that.

---

## Approximate Search

### 29. Why exact search becomes intractable
So far, both exhaustive search and dynamic programming aim for the exact minimum-cost solution.

But exact methods fail when the effective state space becomes too large.

Examples mentioned in the lecture:

- state contains a set of visited locations
- state contains the sequence of words generated so far

In such problems, the number of states may be enormous or exponential.

So now the goal changes:

- do not insist on exact optimality
- search only a subset of possibilities
- hope to find a good enough solution

This is heuristic search.

### 30. Best-of-$n$: the simplest approximate method
Best-of-$n$ works like this:

- sample one full solution by repeatedly choosing actions
- do this $n$ times
- return the best sampled solution

That is it.

The lecture almost deliberately demystifies the term: it sounds sophisticated, but the basic algorithm is very simple.

### 31. Policy in best-of-$n$
To sample actions, we need a policy.

A policy is a mapping:

$$
\pi(s) \to a
$$

or more generally a distribution over actions.

In the lecture’s travel example, the policy is uniform random over the available successors.

So if from a state you can walk or take the tram, the policy samples one of those uniformly at random.

### 32. Rollout
A rollout means:

- start at the start state
- repeatedly apply the policy
- continue until reaching an end state or hitting a max-step limit

The result is one complete candidate solution.

Best-of-$n$ is therefore just:

- rollout
- rollout
- rollout
- ...
- keep the cheapest final result

### 33. Guarantee of best-of-$n$
As $n \to \infty$, if the policy gives positive probability to all relevant actions, then best-of-$n$ will eventually sample an optimal solution.

So in the infinite limit it is consistent.

But the lecture immediately adds the real caveat:

- the required $n$ may be exponentially large

So this is not an efficient exact method. It is a practical approximation method.

### 34. Why best-of-$n$ is still useful
Despite its simplicity, best-of-$n$ has strong practical advantages:

- extremely simple to implement
- embarrassingly parallel
- works surprisingly well if the policy is informative

“Embarrassingly parallel” means each rollout can be computed independently with no coordination. This is a major engineering advantage.

In modern AI, if the policy is a strong language model, then best-of-$n$ becomes much more effective than the uniform-random toy example might suggest.

### 35. Beam search: keep the best partial solutions
Beam search is a more structured approximate method.

Instead of sampling complete trajectories independently, it does the following at each depth:

- keep a set of partial solutions, called the beam
- expand every candidate in the beam by one step
- score all resulting partial solutions by cumulative cost so far
- keep only the best `beam_width` of them
- discard the rest

Then repeat.

The metaphor in the lecture is a car driving at night:

- you do not see the whole road network
- you see only a limited beam ahead

### 36. Beam width
The beam width is the number of partial solutions retained after each expansion round.

Special cases:

- `beam_width = 1` gives greedy search
- as beam width grows, the search becomes less myopic
- in the limit of infinite beam width, beam search becomes exhaustive search

But that limit is usually useless in practice because memory and time blow up.

### 37. How beam search proceeds
At step $t$:

- take each partial candidate currently in the beam
- if it already ended, keep it as-is
- otherwise expand it using all possible successors
- collect all newly extended candidates
- sort them by accumulated cost
- keep only the cheapest `beam_width`

At the end, among completed candidates, return the cheapest one.

This means beam search aggressively prunes the search tree.

### 38. Why beam search can fail
Beam search can discard a partial path that looks slightly worse now but would have become best later.

So it is heuristic, not exact.

It relies on the hope that low current cost is correlated with low final cost.

That hope is often reasonable, but not guaranteed.

### 39. Deterministic versus stochastic search
The lecture contrasts approximate methods:

#### Best-of-$n$
- stochastic
- depends on sampled rollouts
- easy to parallelize
- can benefit strongly from a good policy prior

#### Beam search
- deterministic given the cost function and tie-breaking
- prunes according to cumulative cost
- less naturally parallel than independent best-of-$n$ rollouts
- does not need an explicit policy prior in the same way

There is also a stochastic relative of beam search called particle filtering.

### 40. Early pruning using a known completed solution
A student raises an important optimization idea:

- if a completed path already has cost $6$
- and another partial path already costs $8$
- can we safely throw away the partial one?

Answer:

- yes, if all future costs are nonnegative
- no, not in general if negative costs are possible, because the partial path could later improve enough to win

This is a classic pruning condition and a good example of how cost structure affects algorithm design.

---

## Search and Language Models

### 41. Test-time compute framing
The lecture ends by connecting classical search to modern language-model inference.

Given:

- a prompt
- a language model that predicts next-token probabilities
- optionally, a verifier that checks whether a full response is correct

Goal:

- produce a response that is both likely under the model and good according to the verifier

Rather than sampling one answer and stopping, we can spend extra computation at inference time to search over multiple candidates.

This is test-time compute.

### 42. Casting language generation as a search problem
The lecture defines the search components as follows.

#### State
The prompt plus the response prefix generated so far.

So if the prompt is a math expression prefix, each state is a partially completed response string.

#### Action
Predict the next token.

#### Cost
Negative log probability of that token:

$$
\text{cost} = -\log p(\text{next token} \mid \text{current prefix})
$$

Why negative log probability?

Because minimizing sums of negative log probabilities is equivalent to maximizing the product of probabilities:

$$
\arg\min \sum_t -\log p_t = \arg\max \prod_t p_t
$$

This is the standard probabilistic sequence-modeling conversion.

### 43. Verifier bonus hack
If a verifier exists and says a full response is correct, the lecture adds a large negative bonus, e.g. $-100$.

This means:

- verified correct complete responses become much more attractive
- the search objective combines model likelihood with correctness preference

This is explicitly described as a hack, but it is a useful one for folding extra preferences into the cost framework.

### 44. Best-of-$n$ with language models
In the lecture’s toy example:

- a small language model generates token-level successors
- a policy samples among top-$k$ next-token options using their probabilities
- rollouts produce multiple completed responses
- best-of-$n$ chooses the best final response

This shows the full pipeline:

- learning provides probabilities/costs
- search organizes computation over candidate outputs

That is the lecture’s modern AI takeaway.

---

## Important Themes and Takeaways

### 45. Search is a representation game before it is an algorithm game
Most beginner mistakes in search come from bad state definitions, not from bad search code.

Ask first:

- what information determines legal actions?
- what information determines future costs?
- what information determines successor states?

Whatever is needed for those must be in the state.

### 46. Exactness versus tractability
There is a three-way tension:

- exactness
- time
- memory

Exhaustive search is exact but often too slow.

Dynamic programming is still exact and often much faster, but can consume too much memory.

Approximate methods reduce cost by giving up optimality guarantees.

### 47. Repeated subproblems are the reason DP works
Dynamic programming is not magic. It helps only when multiple paths reuse the same downstream computation.

So whenever you see merging paths or repeated states, you should start thinking about memoization.

### 48. Search and learning are complementary
Learning alone gives a local scoring mechanism.

Search turns that local scoring mechanism into global decision-making.

That is why language-model probabilities plus search can outperform naive one-shot generation.

---

## Summary

This lecture introduces search as a formal framework for reasoning in deterministic settings.

The most important modeling concept is the state:

- it must contain enough information to evaluate future actions, costs, and successors
- but not so much information that the state space becomes unnecessarily huge

A search problem is defined by:

- a start state
- a successor function
- an end test
- action costs

The objective is to find a sequence of actions with minimum total cost.

The key mathematical idea is the future-cost recurrence:

$$
\text{futureCost}(s) = \min_{(a,c,s') \in \text{Successors}(s)} \left[c + \text{futureCost}(s')\right]
$$

This leads to exhaustive search, which is exact but can take exponential time.

Dynamic programming improves exhaustive search by caching the optimal future solution for each state. It can yield massive speedups when many paths reach the same state, but it requires enough memory to store cached states.

When exact search is infeasible, the lecture introduces approximate methods:

- best-of-$n$: sample $n$ complete rollouts and keep the best
- beam search: keep only the best few partial solutions at each depth

Both methods sacrifice guarantees for tractability.

Finally, the lecture shows how search connects directly to modern AI systems. In language models, next-token probabilities can define costs, and test-time compute can be viewed as search over candidate responses. This is the core modern synthesis:

- learning estimates local preferences or probabilities
- search uses those estimates to find globally better solutions

---

## Real-World Applications

### Route planning
A navigation system can model locations as states and roads as successors with travel-time costs. Search then finds a good route from origin to destination.

### Robotics and planning
A robot often cannot act reflexively. It must reason about sequences of actions, future consequences, and constraints such as battery limits, collisions, or task order.

### Puzzle solving and games
Rubik’s Cube, board games, and deterministic puzzles all naturally fit search formulations where the agent must evaluate many possible future action sequences.

### Program synthesis and theorem proving
When generating a proof or code solution, each partial derivation can be treated as a state, and search explores promising continuations.

### Language-model inference
Modern language models can use search at test time:

- best-of-$n$ sampling
- beam search
- verifier-guided decoding

This is exactly the lecture’s argument that search has become newly relevant in the era of large learned models.

### Reinforcement learning and MDPs
The lecture also foreshadows later material. The future-cost recurrence here is a close cousin of value recurrences in MDPs and reinforcement learning. So Search I is laying conceptual groundwork for later parts of the course.
