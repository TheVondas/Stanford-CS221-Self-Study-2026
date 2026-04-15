# CS221 Lecture 10 — Games I
[[Stanford CS221 Autumn 2025]]

## Learning Objectives

By the end of this lecture, you should be able to:

1. Explain how a sequential game differs from an MDP.

2. Model a two-player zero-sum game as a game tree.

3. Define states, players, successors, terminal states, and utilities for a game.

4. Compute the value of a game under fixed policies using game evaluation.

5. Explain and apply expectimax when the opponent policy is known.

6. Explain and apply minimax when the opponent policy is unknown and assumed adversarial.

7. Interpret minimax both as optimal play against the strongest opponent and as a worst-case guarantee.

8. Understand how different policy assumptions produce different “optimal” policies.

9. Explain expectiminimax for games that contain both adversaries and randomness.

10. Explain how alpha-beta pruning speeds up minimax without changing the answer.

11. Explain how evaluation functions and depth-limited search approximate minimax in large games.

12. Connect game-tree methods to earlier ideas from search, MDPs, and RL.

---

## Concept Inventory

### 1. What is a game in this lecture?

The lecture focuses on **two-player zero-sum sequential games**.

- **Two-player**: there is an agent and an opponent.

- **Zero-sum**: the agent’s utility is the negative of the opponent’s utility.

If the agent gets utility $u$, the opponent gets utility $-u$.

So the total utility is always

$$  
u + (-u) = 0  
$$

This means one player’s gain is exactly the other player’s loss.

---

### 2. Why games are different from MDPs

In an MDP:

- the agent tries to maximise utility,

- the environment is random,

- the transition dynamics may be known or learned.

In a game:

- the agent still tries to maximize utility,

- but part of the environment is now an **opponent**,

- and the opponent is not random by default,

- the opponent may act strategically to hurt you.

That is the crucial shift.

MDPs ask:

> What should I do in an uncertain world?

Games ask:

> What should I do when another intelligent entity is actively responding to me?

---

### 3. Game tree

A **game tree** is the core mental model.

- The root is the start state.

- Edges are actions.

- Internal nodes are decision points.

- Leaf nodes are terminal outcomes with utilities.

Each root-to-leaf path is one complete game outcome.

A game is therefore a kind of state-based model, just like search problems and MDPs, but with **multiple decision-makers**.

---

### 4. Formal game definition

A game is defined by:

- **Start state**: where play begins.

- **IsEnd$(s)$**: whether the game is over.

- **Player$(s)$**: whose turn it is in state $s$.

- **Successors$(s)$**: mapping from available actions to successor states.

- **Utility$(s)$**: the terminal payoff to the agent.

A key point: **the state must encode whose turn it is**.

That is why the player function is really a property of state.

---

### 5. Policies in games

A policy still maps states to actions.

Deterministic policy for player $p$:

$$  
\pi_p(s) = a  
$$

Stochastic policy for player $p$:

$$  
\pi_p(a \mid s)  
$$

This is the probability that player $p$ chooses action $a$ in state $s$.

The lecture allows stochastic policies in general.

---

### 6. Sparse reward in games

The lecture assumes utility is received only at the end of the game.

So most intermediate states do not tell you directly whether you are “winning” in a formal sense.

This creates a **sparse reward** problem:

- a state may look promising,

- but all that matters is the final terminal result.

This is one reason large games are hard.

---

### 7. The two recurring examples

#### Game 1: the three-bin game

The agent chooses one of $A$, $B$, or $C$.

Then the opponent chooses a number from that bin.

Utilities:

- Bin $A$: $-50$ or $50$

- Bin $B$: $1$ or $3$

- Bin $C$: $-5$ or $15$

This example is perfect because different assumptions about the opponent lead to different “best” moves.

#### Halving game

State contains a number $n$ and whose turn it is.

On each turn, a player can either:

- decrement: $n \to n-1$

- half: $n \to \lfloor n/2 \rfloor$

The player who leaves the game at $n=0$ wins.

This example makes minimax more concrete.

---

## Slide-by-Slide Walkthrough

## 1. Position of games in the course

The lecture opens by placing games after:

- search,

- MDPs,

- reinforcement learning.

Search, MDPs, RL, and games are all **state-based models**.

Games complete this little tour by introducing an intelligent adversary.

This is an important course-level insight:

- **Search**: deterministic world.

- **MDPs / RL**: stochastic world.

- **Games**: strategic world.

---

## 2. The opening bin example: why opponent modeling matters

You are asked to choose between bins $A$, $B$, and $C$.

At first glance, this seems like a simple decision problem.

But it is actually ill-defined until you specify how the opponent behaves.

### If the opponent is random

You might prefer $C$, because its average is attractive:

$$  
\frac{-5 + 15}{2} = 5  
$$

### If the opponent is adversarial

You should prefer $B$, because its worst-case payoff is best:

- Worst-case $A = -50$

- Worst-case $B = 1$

- Worst-case $C = -5$

So the best choice is $B$.

This is the core lesson of the opening example:

> In games, the correct strategy depends on your assumptions about the opponent.

Do not blur together:

- “best on average”

- “best in the worst case”

They are not the same.

---

## 3. Formalising a game

The lecture then turns the intuitive picture into a formal object.

For the three-bin game:

- Start state is the root.

- If you are at the root, it is the agent’s turn.

- If you are at node $A$, $B$, or $C$, it is the opponent’s turn.

- If you are at $A1$, $A2$, $B1$, $B2$, $C1$, or $C2$, the game is over and you read off utility.

This formalization matters because every later recurrence depends on exactly these ingredients.

A strong exam instinct here is:

> Before solving a game, always ask: what is the state, who moves, what are the successors, and what is the terminal payoff?

---

## 4. Special properties of games in this lecture

Two features are emphasised.

### Utility is at the end

The game is evaluated only at terminal states.

That means there is no discounted sum of intermediate rewards as in the usual MDP setup.

Because there is only one terminal utility value, discounting is not relevant here.

### Different players control different nodes

This is the major structural difference from earlier material.

In search:

- the agent controls the decisions.

In MDPs:

- the agent chooses actions,

- chance determines transitions.

In games:

- sometimes the agent controls the node,

- sometimes the opponent controls the node,

- later, sometimes a chance process controls the node.

So the recurrence must adapt to the type of node.

---

## 5. Policies in game trees

The lecture reviews deterministic and stochastic policies.

This is not just notation. It matters because:

- game evaluation uses fixed policies for all players,

- expectimax assumes a fixed stochastic opponent policy,

- minimax removes that assumption and instead assumes the opponent chooses the minimising action.

So policy assumptions drive the recurrence.

---

## 6. The halving game: state includes the turn

The halving game reinforces an important structural idea:

the state is not just the number $n$.

It must include:

- the current number,

- whose turn it is.

So conceptually the state is something like

$$  
s = (n, \text{player})  
$$

That is a subtle but important modeling point.

If you forget the player, you are no longer representing the game correctly.

---

## 7. Game evaluation: value under fixed policies

Now the lecture asks:

> If I give you a game and a policy for each player, can you compute the value of the game?

Yes.

This is **game evaluation**.

The value is the expected utility of the agent over all possible rollouts generated by the fixed policies.

Let

- $\pi_{\text{agent}}$ be the agent policy,

- $\pi_{\text{opp}}$ be the opponent policy.

Then the game-evaluation recurrence is:

$$  
V_{\text{eval}}(s)=  
\begin{cases}  
\mathrm{Utility}(s) & \text{if } \mathrm{IsEnd}(s) \  
\sum_a \pi_{\text{agent}}(a \mid s), V_{\text{eval}}(\mathrm{Succ}(s,a)) & \text{if } \mathrm{Player}(s)=\text{agent} \  
\sum_a \pi_{\text{opp}}(a \mid s), V_{\text{eval}}(\mathrm{Succ}(s,a)) & \text{if } \mathrm{Player}(s)=\text{opp}  
\end{cases}  
$$

### Intuition

At any non-terminal state:

- identify whose turn it is,

- use that player’s policy to weight the possible successor values,

- sum them.

This is just expectation.

---

## 8. Simulation versus exact evaluation

The lecture shows two ways to compute game value.

### Monte Carlo simulation

Roll out the game multiple times under the fixed policies and average the utilities.

This gives an estimate of the value.

For example, if:

- agent always chooses $A$,

- opponent chooses between $1$ and $2$ uniformly,

then the true value is

$$  
0.5(-50) + 0.5(50) = 0  
$$

But if you only simulate a small number of rollouts, you might get a noisy estimate such as $15$.

That is just sampling error.

### Exact recurrence

Instead of sampling, compute the expectation exactly using the recurrence above.

This gives the true value, but it may require exploring an exponentially large tree.

---

## 9. Why game evaluation is analogous to policy evaluation in MDPs

This is one of the cleanest conceptual bridges in the lecture.

Game evaluation is analogous to **policy evaluation** in MDPs.

Why?

Because in both cases:

- the policy is fixed,

- you are not optimizing actions,

- you are just computing the value induced by fixed behavior.

The only difference is that in the game setting, “the environment” is represented as another player with a policy.

So conceptually:

- MDP policy evaluation: evaluate one fixed agent policy in a stochastic environment.

- Game evaluation: evaluate fixed policies for all players in a game tree.

---

## 10. Expectimax: optimal play against a known opponent policy

Now the lecture changes the question.

Instead of evaluating a fixed agent policy, we want the **best** agent policy, while still assuming the opponent policy is fixed and known.

This is **expectimax**.

The lecture sometimes phrases it as “expected max,” but the standard name is **expectimax**.

The recurrence is:

$$  
V_{\text{exp}}(s)=  
\begin{cases}  
\mathrm{Utility}(s) & \text{if } \mathrm{IsEnd}(s) \  
\max_a V_{\text{exp}}(\mathrm{Succ}(s,a)) & \text{if } \mathrm{Player}(s)=\text{agent} \  
\sum_a \pi_{\text{opp}}(a \mid s), V_{\text{exp}}(\mathrm{Succ}(s,a)) & \text{if } \mathrm{Player}(s)=\text{opp}  
\end{cases}  
$$

### Intuition

- At agent nodes: choose the action with highest expected value.

- At opponent nodes: average according to the known opponent policy.

---

## 11. Expectimax on the three-bin game

Assume the opponent is random and chooses each number with probability $0.5$.

Then the expected values of bins are:

### Bin $A$

$$  
0.5(-50)+0.5(50)=0  
$$

### Bin $B$

$$  
0.5(1)+0.5(3)=2  
$$

### Bin $C$

$$  
0.5(-5)+0.5(15)=5  
$$

So expectimax chooses $C$.

That is why the students who chose $C$ were implicitly assuming a random rather than adversarial opponent.

---

## 12. Why you cannot naively Monte Carlo expectimax

The lecture makes an important point:

game evaluation can be estimated by sampling, because expectation is linear.

But expectimax contains a **max** operation.

That breaks the naive Monte Carlo idea.

The issue is that

$$  
\max_a \mathbb{E}[X_a]  
$$

is not the same computation as simply sampling a trajectory and averaging outcomes.

To choose the maximizing action, you need comparative information about all the branches, not just one sampled rollout.

So expectimax is not handled by naive rollout averaging in the same easy way as game evaluation.

---

## 13. Expectimax as the game analogue of value iteration

The lecture relates expectimax to **value iteration** in MDPs.

That is a very good exam connection.

- **Game evaluation** $\leftrightarrow$ policy evaluation

- **Expectimax** $\leftrightarrow$ value iteration

Why?

Because expectimax performs optimization at the agent nodes, exactly as value iteration performs optimization over actions.

Up to this point, the lecture says nothing fundamentally new has happened relative to MDPs; it is still max-plus-expectation.

The true game-specific ingredient arrives with minimax.

---

## 14. Minimax: optimal play against an unknown adversary

Now the opponent policy is no longer assumed known.

So what should the agent do?

The minimax principle says:

> Assume the opponent is trying to hurt you as much as possible.

So:

- the agent maximizes utility,

- the opponent minimizes the agent’s utility.

The recurrence is:

$$  
V_{\text{minmax}}(s)=  
\begin{cases}  
\mathrm{Utility}(s) & \text{if } \mathrm{IsEnd}(s) \  
\max_a V_{\text{minmax}}(\mathrm{Succ}(s,a)) & \text{if } \mathrm{Player}(s)=\text{agent} \  
\min_a V_{\text{minmax}}(\mathrm{Succ}(s,a)) & \text{if } \mathrm{Player}(s)=\text{opp}  
\end{cases}  
$$

This is simpler than expectimax because there is no averaging.

---

## 15. Minimax on the three-bin game

Take the minimum utility in each bin:

- $A \to \min(-50, 50) = -50$

- $B \to \min(1, 3) = 1$

- $C \to \min(-5, 15) = -5$

Then the agent takes the maximum of those worst cases:

$$  
\max(-50, 1, -5)=1  
$$

So minimax chooses $B$.

This is the safe option.

### Core interpretation

Minimax does not choose the action with the highest average payoff.

It chooses the action with the best guaranteed payoff under the worst opponent response.

That is the entire philosophy of minimax.

---

## 16. Returning both value and action

The lecture’s implementation returns not just the minimax value but also the associated action.

That is conceptually important.

A recurrence can tell you:

- the value of each state,

- and the policy implied by taking $\arg\max$ or $\arg\min$.

So from minimax you get:

- $\pi_{\max}$ for the agent,

- $\pi_{\min}$ for the opponent.

These are the corresponding optimal policies under the minimax assumptions.

---

## 17. Minimax policy in the halving game

The lecture then applies minimax to the halving game and uses the resulting agent policy against a random opponent policy.

The minimax agent wins consistently.

This demonstrates a powerful point:

- a minimax policy is designed to survive the strongest opponent,

- so against a weaker random opponent it often performs extremely well.

The lecture also interprets values in the halving game:

- value $+1$: the agent can force a win.

- value $-1$: the opponent can force a win if the opponent plays optimally.

This is a very important distinction.

If the value is $-1$, that does **not** mean the agent will always lose against every opponent.

It means the agent loses against optimal opposition.

Against weaker play, the agent may still win.

---

## 18. Perfect play and solved games

The lecture introduces some standard terminology.

### Perfect play

Both players play optimally.

### Solved game

The outcome under perfect play is known.

The lecture’s examples are:

- strongly solved: the value is known for every state,

- weakly solved: the value is known from the initial position,

- unsolved: optimal value/policy is not fully known.

The central conceptual point is not memorizing the examples. It is understanding that:

> “A game being solved” means we know the minimax truth of the game, not merely that computers are very good at it.

So “superhuman play” and “solved” are not the same thing.

---

## 19. Why minimax is fundamentally new relative to MDPs

The lecture explicitly says minimax has no direct MDP analogue.

That is because MDPs use:

- max over agent actions,

- expectation over stochastic transitions.

Games introduce:

- min over opponent actions.

That “min” is the genuinely new operation.

This is the mathematical fingerprint of adversarial reasoning.

---

## 20. Face-off between policies: optimal relative to what?

This is one of the deepest conceptual parts of the lecture.

It compares different policies generated under different assumptions.

The lecture defines:

- $\pi_{\max}$: agent policy from minimax

- $\pi_{\min}$: opponent policy from minimax

- $\pi_7$: some arbitrary fixed opponent policy

- $\pi_{\text{exp}(7)}$: agent policy from expectimax against $\pi_7$

The warning is:

> Never say “optimal” without specifying optimal with respect to what.

That is exactly right.

A policy can be optimal against one opponent model and poor against another.

---

## 21. The three key policy relationships

Let

$$  
V(\pi_{\text{agent}}, \pi_{\text{opp}})  
$$

denote the value of the game when those two fixed policies play each other.

Then the lecture gives three core facts.

### Property 1: $\pi_{\max}$ is best against $\pi_{\min}$

Since $\pi_{\max}$ is the maximizing policy against $\pi_{\min}$,

$$  
V(\pi_{\text{exp}(7)}, \pi_{\min}) \le V(\pi_{\max}, \pi_{\min})  
$$

In words: if the opponent is playing the minimax-minimizing policy, the minimax agent policy is at least as good as any alternative agent policy, including the expectimax one.

---

### Property 2: $\pi_{\min}$ is worst for $\pi_{\max}$

Since $\pi_{\min}$ is the minimizing policy against $\pi_{\max}$,

$$  
V(\pi_{\max}, \pi_{\min}) \le V(\pi_{\max}, \pi_7)  
$$

In words: once you fix the minimax agent, the minimax opponent is the most harmful opponent. Any weaker opponent can only help the agent.

This gives the lower-bound interpretation of minimax.

If the minimax value is already good, you are safe against any opponent.

---

### Property 3: $\pi_{\text{exp}(7)}$ is best against $\pi_7$

Since expectimax is tailored to the known policy $\pi_7$,

$$  
V(\pi_{\max}, \pi_7) \le V(\pi_{\text{exp}(7)}, \pi_7)  
$$

In words: if you know the opponent’s policy, you can often exploit it better than minimax can.

Minimax is robust, but not necessarily exploitative.

---

## 22. Numerical relationship in the three-bin game

For the lecture’s three-bin example:

- $V(\pi_{\text{exp}(7)}, \pi_{\min}) = -5$

- $V(\pi_{\max}, \pi_{\min}) = 1$

- $V(\pi_{\max}, \pi_7) = 2$

- $V(\pi_{\text{exp}(7)}, \pi_7) = 5$

So:

$$  
-5 \le 1 \le 2 \le 5  
$$

This ordering is extremely instructive.

It shows:

- expectimax can be terrible if you assume the wrong opponent,

- minimax guarantees safety,

- but minimax may leave value on the table against a predictable weak opponent.

This is one of the most important takeaways of the lecture.

---

## 23. The two meanings of minimax

The lecture gives two ways to interpret minimax.

### Interpretation 1: optimal against the strongest possible opponent

This is the direct game-theoretic interpretation.

### Interpretation 2: a lower bound against any unknown opponent

This is often the more practically useful interpretation.

If you do not trust your opponent model, minimax says:

> Here is what I can guarantee no matter what the opponent does.

This is the robust-control mindset.

---

## 24. Expectiminimax: games with both opponents and randomness

The lecture then adds a chance node.

You choose a bin, then a coin flip may shift you left before the opponent picks a number.

Now the tree has three types of nodes:

- max nodes for the agent,

- min nodes for the opponent,

- expectation nodes for chance.

This gives **expectiminimax**.

The recurrence is:

$$  
V_{\text{exp-min-max}}(s)=  
\begin{cases}  
\mathrm{Utility}(s) & \text{if } \mathrm{IsEnd}(s) \  
\max_a V_{\text{exp-min-max}}(\mathrm{Succ}(s,a)) & \text{if } \mathrm{Player}(s)=\text{agent} \  
\min_a V_{\text{exp-min-max}}(\mathrm{Succ}(s,a)) & \text{if } \mathrm{Player}(s)=\text{opp} \  
\sum_a \pi_{\text{chance}}(a \mid s), V_{\text{exp-min-max}}(\mathrm{Succ}(s,a)) & \text{if } \mathrm{Player}(s)=\text{chance}  
\end{cases}  
$$

### Big idea

The recurrence is modular.

You do whatever operation corresponds to the node type:

- agent $\to \max$

- opponent $\to \min$

- chance $\to \mathbb{E}$

That is the general game-tree recipe.

---

## 25. What the game-tree framework can and cannot represent

The lecture then makes an important modeling boundary explicit.

### It can handle

- more than two players,

- multiple opponents,

- weird turn mechanics,

- extra turns,

- skipping turns,

- player identity encoded in state.

As long as the next state and the next player are fully determined by the current state and action structure, you can write a recurrence.

### It does not naturally handle

#### Imperfect-information games

Example: poker.

Why not? Because different players know different things, but a standard game tree assumes all relevant information is represented in a single fully specified state.

#### Non-zero-sum games

Example: prisoner’s dilemma.

Now utilities are not strict negatives of each other.

#### Simultaneous-move games

Example: rock-paper-scissors.

There is no sequential “your turn, then my turn” structure.

This is a valuable modeling warning: not every strategic interaction is a minimax tree.

---

## 26. Minimax is exponential: can we speed it up exactly?

The lecture now shifts from modeling to computational efficiency.

Plain minimax explores the entire tree.

If the branching factor is $b$ and the depth is $d$, the number of nodes can be exponential in $d$.

So the question becomes:

> Can we avoid searching branches that cannot affect the final decision?

That motivates alpha-beta pruning.

---

## 27. Alpha-beta pruning: core idea

Alpha-beta pruning is an instance of **branch and bound**.

The intuition is simple:

if you already know one branch is at least as good as another branch could ever possibly become, you do not need to finish exploring the worse branch.

The lecture’s warm-up interval example captures this nicely:

- Option $A \in [3,5]$

- Option $B \in [5,100]$

You should always choose $B$.

You do not need exact values.

You only need enough information to prove dominance.

---

## 28. A simple pruning example

Suppose the root is a max node.

You fully evaluate the left child and get value $3$.

Then you move to the right child, which is a min node, and one of its descendants already gives value $2$.

So the right child can be at most $2$.

But the root already has an option worth $3$.

Therefore the right subtree can never beat the current best option.

So you prune the rest of the right subtree.

That is the central alpha-beta insight:

> You do not need the exact value of a subtree if you already know it cannot influence the ancestor’s choice.

---

## 29. Meaning of alpha and beta

The lecture presents alpha and beta as bounds.

### Alpha

A lower bound on the value of a max node.

It is how good the max side can already guarantee from explored children.

### Beta

An upper bound on the value of a min node.

It is how small the min side can already force from explored children.

So while exploring:

- max nodes accumulate stronger lower bounds,

- min nodes accumulate stronger upper bounds.

If the current bounds become incompatible with ancestor bounds, pruning is possible.

A compact version of the standard pruning rule is:

- prune below a max node if its current $\alpha$ is at least some relevant ancestor $\beta$,

- prune below a min node if its current $\beta$ is at most some relevant ancestor $\alpha$.

The lecture explains this geometrically as interval non-overlap along the ancestor path.

That is a very nice way to think about it.

---

## 30. Why move ordering matters

Alpha-beta is exact, but its speed depends heavily on the order in which you explore children.

If you visit strong moves first:

- max nodes get large $\alpha$ values early,

- min nodes get small $\beta$ values early,

- pruning happens sooner.

If you visit poor moves first, pruning is weaker.

So action ordering does not change correctness, but it changes runtime a lot.

This is analogous to a recurring theme in the course:

a good heuristic can make an exact algorithm dramatically faster.

---

## 31. How to order moves in practice

The lecture’s rule of thumb:

- at max nodes, try children in decreasing estimated value,

- at min nodes, try children in increasing estimated value.

In other words, visit the most promising child first for whichever side is moving.

This requires some rough estimate of state quality.

That naturally leads into evaluation functions.

---

## 32. Evaluation functions: approximate state quality

Exact minimax to terminal states is often impossible in large games.

So we define an **evaluation function** $Eval(s)$.

This is a quick estimate of how good a state is for the agent.

For chess, a simple evaluation function might combine features such as:

- material,

- mobility,

- king safety,

- center control,

- other positional structure.

A typical design pattern is:

$$  
Eval(s)=\sum_i w_i f_i(s)  
$$

where:

- $f_i(s)$ are features of the position,

- $w_i$ are weights expressing importance.

This is not the true game-theoretic value.

It is an informed guess.

---

## 33. Why a greedy evaluation function alone is not enough

You could choose the move that immediately leads to the state with highest $Eval(s)$.

But that would be too shallow.

It misses tactical consequences.

So instead the lecture combines evaluation functions with search.

This produces **depth-limited minimax**.

---

## 34. Depth-limited minimax

The idea is:

- search only to some depth $d$,

- if you hit a true terminal state, return its true utility,

- if your depth budget runs out, return the evaluation function instead.

Conceptually:

$$  
V(s,d)=  
\begin{cases}  
\mathrm{Utility}(s) & \text{if } \mathrm{IsEnd}(s) \  
Eval(s) & \text{if depth limit reached} \  
\max_a V(\mathrm{Succ}(s,a), d') & \text{if } \mathrm{Player}(s)=\text{agent} \  
\min_a V(\mathrm{Succ}(s,a), d') & \text{if } \mathrm{Player}(s)=\text{opp}  
\end{cases}  
$$

In the lecture’s version, the depth is decremented after the opponent move so that one unit of depth corresponds roughly to one full agent-opponent cycle.

That is a detail worth remembering because different texts count depth differently.

---

## 35. Tradeoff in depth-limited search

There is a direct tradeoff.

### Deeper search

- more expensive,

- less dependent on evaluation quality.

### Shallower search

- cheaper,

- much more dependent on evaluation quality.

So evaluation functions and search depth compensate for each other.

A poor evaluation function can sometimes be rescued by deeper search.

A very shallow search requires a strong evaluation function.

---

## 36. Connection to reinforcement learning

The lecture closes by pointing out that evaluation functions should remind you of value functions from RL.

That is exactly right.

An evaluation function is trying to estimate:

- how good a state is,

- before you see the full future.

That is very close in spirit to learned values or Q-values.

This sets up the next lecture:

> instead of hand-designing the evaluation function, can we learn it?

That leads into TD learning.

---

## Summary

This lecture introduces sequential adversarial reasoning.

The major ideas are:

1. A game is modelled as a tree of states, actions, players, successors, terminal checks, and utilities.

2. In this lecture’s setting, games are two-player and zero-sum.

3. Game evaluation computes expected utility when all player policies are fixed.

4. Expectimax chooses the optimal agent action against a known stochastic opponent policy.

5. Minimax chooses the optimal agent action against an unknown adversary by assuming the opponent minimises your utility.

6. Minimax is both:

    - optimal against the strongest opponent,

    - and a worst-case guarantee against any unknown opponent.

7. Different assumptions about the opponent produce different “optimal” policies.

8. Expectiminimax handles mixed trees containing agent, opponent, and chance nodes.

9. Alpha-beta pruning speeds up minimax exactly by eliminating branches that cannot affect the final decision.

10. Evaluation functions plus depth-limited search give an approximate but practical way to handle large games.

11. Evaluation functions are conceptually close to learned value functions in RL, which motivates the next lecture.

### The single most important conceptual contrast

- **Expectimax**: “What should I do if I know how the opponent behaves?”

- **Minimax**: “What should I do if I want guarantees against the worst possible opponent?”

Do not confuse those.

---

## Real-World Applications

### Chess, checkers, Go, Connect Four

These are the obvious game-tree examples.

- minimax formalizes adversarial reasoning,

- alpha-beta pruning makes exact search more efficient,

- evaluation functions help when the tree is too large.

### Cybersecurity

Defender versus attacker can often be modeled adversarially.

Minimax-style reasoning captures robust strategies under worst-case attacks.

### Negotiation and competitive markets

When another party responds strategically to your moves, treating the world as a passive environment is wrong.

Game-theoretic reasoning is more appropriate.

### Robotics and autonomous driving with adversarial agents

If another driver or agent may behave strategically or unpredictably, robust planning becomes important.

### RL for games

The evaluation function / value-function connection is critical.

Modern game-playing systems often combine:

- search,

- learned evaluation/value functions,

- sometimes self-play.

That is exactly the bridge this lecture is building toward.

---

## Final Takeaways to Remember

If you only remember five things, make them these:

1. **State must include whose turn it is.**

2. **Game evaluation = fixed policies; expectimax = optimise against known policy; minimax = optimise against worst-case opponent.**

3. **Minimax is a guarantee, not an exploitation strategy.**

4. **Alpha-beta pruning changes runtime, not the answer.**

5. **Evaluation functions are approximate values, and next lecture is about learning them.**
