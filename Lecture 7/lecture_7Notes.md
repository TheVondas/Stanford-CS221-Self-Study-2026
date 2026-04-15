# CS221 Lecture 7 — Markov Decision Processes (MDPs)

## Learning Objectives

By the end of this lecture, you should be able to:

- explain why search problems are too limited for many real-world decision problems;
- define a Markov Decision Process and identify its core components;
- distinguish between states, actions, transitions, rewards, rollouts, utility, value, policy, and Q-values;
- explain why the solution to an MDP is a policy rather than a fixed action sequence;
- compute the utility of a rollout using discounted rewards;
- explain the difference between Monte Carlo policy evaluation and exact policy evaluation by recurrence;
- derive and interpret the policy evaluation update;
- derive and interpret the value iteration update;
- explain why value iteration differs from policy evaluation by essentially replacing “follow the policy’s action” with “take the max over actions”;
- interpret the optimal policy for the flaky tram example and the dice game.

---

## Concept Inventory

### Search problem
A search problem has:

- a start state;
- a successor function;
- an end test;
- deterministic actions.

If you take an action from a state, you know exactly which next state you will reach.

### Markov Decision Process (MDP)
An MDP generalises a search problem by allowing actions to have stochastic outcomes.

Instead of:

- “from state $s$, action $a$ leads to exactly one next state,”

we now have:

- “from state $s$, action $a$ leads to a distribution over possible next states.”

### Markov property
The state contains all relevant information needed for future decision-making.

Formally, once you know the current state, the past and future are independent.

Intuition:
- the state is a sufficient summary of history;
- if your state is badly designed, the MDP model is bad.

### Policy
A policy is a function from states to actions:

$$
\pi(s) = a
$$

It tells you what to do in every state.

### Rollout
A rollout is what happens when you run a policy inside an MDP.

It is one realized trajectory of:
- states,
- actions,
- rewards,
- random outcomes.

### Utility
The utility of one rollout is the discounted sum of rewards:

$$
U = \sum_{t=0}^{T-1} \gamma^t r_t
$$

where:
- $r_t$ is the reward at time $t$;
- $\gamma \in [0,1]$ is the discount factor.

### Value of a policy
The value of a policy is the expected utility of following that policy from a state:

$$
V_\pi(s) = \mathbb{E}[U \mid S_0 = s, \pi]
$$

This is not one rollout. It is the average over all possible rollouts induced by the policy.

### Q-value
The Q-value for an action measures the value of taking a particular action now, then continuing according to some continuation values.

In this lecture’s notation:

$$
Q(s,a;V) = \sum_{s'} T(s,a,s') \big(R(s,a,s') + \gamma V(s')\big)
$$

Interpretation:
- take action $a$ in state $s$;
- nature randomly chooses the next state $s'$;
- collect immediate reward $R(s,a,s')$;
- then continue with value $V(s')$.

### Policy evaluation
Compute $V_\pi$, the value of a given fixed policy.

### Value iteration
Compute $V^*$, the value of the optimal policy, by taking the maximum over actions at each update.

---

## Big Picture: Why MDPs?

Last lecture, search gave us a model for reasoning:
- choose actions;
- move through states;
- optimize total cost.

But search assumes the world is deterministic.

That is often unrealistic.

Real life contains uncertainty:
- a tram may break down;
- a road may be congested;
- a dice roll may end a game;
- a robot action may succeed or fail;
- a medical treatment may help one patient and not another.

So we need a framework for optimization under uncertainty.

That framework is the MDP.

---

## Walkthrough

## 1. From Search to MDPs

### Search recap
A search problem is defined by:
- start state;
- successors $(\text{action}, \text{cost}, \text{next state})$;
- end test.

The critical assumption is:

> each action deterministically leads to one next state.

That means a solution can simply be a sequence of actions.

### Why that fails under uncertainty
Suppose you are deciding how to travel:
- walk,
- bike,
- drive.

The outcome is uncertain because of:
- traffic;
- parking delays;
- weather;
- random failures.

Now an action does not imply one guaranteed next state.
It implies a probability distribution over next states.

That is exactly the jump from search to MDPs.

---

## 2. What “Markov Decision Process” Means

### Markov
“Markov” comes from Markov chains.

The key idea is:

$$
P(\text{future} \mid \text{present}, \text{past}) = P(\text{future} \mid \text{present})
$$

Once you know the current state, the past does not add extra predictive power.

### Decision
Unlike a pure Markov chain, an MDP includes an agent choosing actions.

### Process
The problem unfolds sequentially over time.

So an MDP is a sequential decision problem under uncertainty.

---

## 3. Formal Ingredients of an MDP

In lecture form, the main ingredients are:

- start state;
- successors with probabilities and rewards;
- end test;
- discount factor $\gamma$.

In mathematical notation, you should think of an MDP as containing:

- a set of states $\mathcal{S}$;
- a set of actions $\mathcal{A}(s)$ available in each state;
- a transition function $T(s,a,s')$;
- a reward function $R(s,a,s')$;
- terminal states or an end condition;
- a discount factor $\gamma$.

### Transition function

$$
T(s,a,s') = P(S_{t+1}=s' \mid S_t=s, A_t=a)
$$

For every state-action pair, the outgoing probabilities must sum to 1:

$$
\sum_{s'} T(s,a,s') = 1
$$

### Reward function

$$
R(s,a,s')
$$

This is the immediate reward obtained when:
- you are in state $s$,
- take action $a$,
- end up in state $s'$.

### Important note from the lecture
Sometimes textbooks define reward as depending only on:
- $s$, or
- $(s,a)$.

Percy used $R(s,a,s')$ because it is the most flexible and natural for specifying examples.
These formulations are usually inter-convertible with small modeling tricks.

---

## 4. Example 1 — The Flaky Tram MDP

## Problem setup
We have locations $1,2,\dots,n$ with $n=10$.

Available actions from state $i$:
- walk to $i+1$ with cost 1 minute;
- tram to $2i$ with cost 2 minutes.

New twist:
- the tram fails with probability $p=0.4$.

Goal:
- get from location 1 to location 10 in the least time in expectation.

That phrase “in expectation” is essential.

In a deterministic search problem, there is one path cost.
In an MDP, there are many possible trajectories.
So we optimize average total reward/cost over possible outcomes.

## Start state

$$
s_{\text{start}} = 1
$$

## Successors

### Walk from state $i$
If $i+1 \le 10$:
- probability $1$;
- reward $-1$;
- next state $i+1$.

So walking is deterministic.

### Tram from state $i$
If $2i \le 10$:
- with probability $0.6$, reward $-2$, next state $2i$;
- with probability $0.4$, reward $-2$, next state $i$.

That second transition means the tram failed and you stayed put.

### End condition

$$
\text{isEnd}(s) \iff s=10
$$

## Why the reward is negative
The lecture switches from costs to rewards.

- cost 1 becomes reward $-1$;
- cost 2 becomes reward $-2$.

This is purely convention.

Minimizing cost is the same as maximizing negative cost.

---

## 5. Search Problems vs MDPs

| Aspect | Search | MDP |
|---|---:|---:|
| Start state | Yes | Yes |
| End test | Yes | Yes |
| Actions | Yes | Yes |
| Successor function | Yes | Yes |
| Immediate feedback | Cost | Reward |
| Outcome of action | One next state | Distribution over next states |
| Solution type | Action sequence | Policy |

### Superficial difference
Search uses costs, MDPs use rewards.

### Deep difference
In search, each action has one next state.
In MDPs, each action has a distribution over next states.

That is the real conceptual leap.

---

## 6. Example 2 — The Dice Game MDP

## Rules
At each round, you choose:
- quit;
- stay.

If you quit:
- you get $10;
- the game ends.

If you stay:
- you get $4;
- then a fair six-sided die is rolled;
- if the die is 1 or 2, the game ends;
- otherwise, the game continues.

## State space
This lecture uses only two states:
- `in`;
- `end`.

## Transitions
From `in`:

### Quit
- probability $1$;
- reward $10$;
- next state `end`.

### Stay
- probability $1/3$, reward $4$, next state `end`;
- probability $2/3$, reward $4$, next state `in`.

This is an extremely compact MDP with only one meaningful decision state.

---

## 7. Why the Solution Must Be a Policy

In search, a solution can be an action sequence because the world is deterministic.

Example:
- from state 1, do this;
- then from state 2, do that;
- then from state 5, do something else.

You know in advance that you will reach those states.

In an MDP, you do not know exactly which state you will reach after an action.

So a fixed action sequence is not enough.

Instead, you need a rule for every state you might end up in.

That rule is a policy:

$$
\pi: \mathcal{S} \to \mathcal{A}
$$

### Examples from the lecture

#### Always walk

$$
\pi(s)=\text{walk}
$$

#### Tram if possible
Take tram if it stays within bounds; otherwise walk.

This depends on the state.

---

## 8. Rollouts: What Happens When You Execute a Policy

To evaluate a policy, we simulate it.

That simulation is called a rollout.

A rollout alternates between:
1. the policy choosing an action;
2. the MDP sampling an outcome according to transition probabilities.

### Important distinction
- the policy controls the action;
- nature controls which random successor happens.

In the flaky tram problem:
- the agent can choose `tram`;
- the agent cannot choose whether the tram succeeds.

That is determined by chance.

---

## 9. Utility and Discounting

## Utility of one rollout
If a rollout produces rewards $r_0,r_1,\dots,r_{T-1}$, then its utility is:

$$
U = \sum_{t=0}^{T-1} \gamma^t r_t
$$

where $\gamma$ is the discount factor.

## Interpretation of discounting
Discounting tells you how much future rewards matter relative to present rewards.

### Case 1: $\gamma = 1$
No discounting.

Future rewards matter just as much as present rewards.

### Case 2: $\gamma = 0$
Full discounting.

Only the immediate reward matters.
All future rewards are ignored.

### Case 3: $0 < \gamma < 1$
Future rewards still matter, but less than present rewards.

For example, with $\gamma=0.5$:
- reward at time 0 gets full weight;
- reward at time 1 gets weight $0.5$;
- reward at time 2 gets weight $0.25$;
- and so on.

## Intuition
A smaller $\gamma$ pushes the agent toward short-term rewards.
A larger $\gamma$ makes the agent care more about the long run.

## Important lecture intuition
Percy described discounting as a way of saying:
- “a reward now may be worth more than the same reward later.”

This is exactly the right intuition.

You can think of it as:
- time preference;
- impatience;
- length penalty;
- preference for faster payoff.

## But be careful
If the real task truly requires long-term planning, then too much discounting can distort the objective.
So you usually choose $\gamma=1$ or something very close to 1 when the future genuinely matters.

---

## 10. The Same Policy Can Produce Different Rollouts

This is one of the most important conceptual points in the lecture.

The same policy, run multiple times in the same MDP, can produce different trajectories because of randomness.

So a policy does not map to one utility.
It maps to a distribution over utilities.

That means:
- one rollout is not enough in general;
- we need the expected utility.

---

## 11. Value of a Policy

The value of policy $\pi$ starting from state $s$ is:

$$
V_\pi(s) = \mathbb{E}[U \mid S_0 = s, \pi]
$$

This is the expected discounted sum of rewards if we start at $s$ and follow policy $\pi$ forever or until termination.

### Important distinction
- utility = one realized rollout’s score;
- value = average score over all possible rollouts.

This distinction is foundational.

A lot of confusion in RL comes from mixing up:
- a sample trajectory;
- the expectation over trajectories.

---

## 12. Monte Carlo Policy Evaluation

The first evaluation strategy shown in the lecture is the most intuitive one:

1. roll out the policy many times;
2. compute utility for each rollout;
3. average the utilities.

This is Monte Carlo policy evaluation.

## Why it works
By the law of large numbers, averaging many rollouts converges to the expected value.

## Why it is noisy
Because it uses sampling.

Different runs give different estimates.

The lecture mentions the rough error rate:

$$
\text{error} \approx \frac{1}{\sqrt{N}}
$$

where $N$ is the number of rollouts.

So to improve precision a lot, you need many more samples.

That is why we would like a more exact recurrence-based method.

---

## 13. Dice Game: Why “Stay” Beats “Quit” in Expectation

### Quit policy
If you always quit, value is clearly:

$$
V_{\text{quit}}(\text{in}) = 10
$$

### Stay policy
If you always stay, let the value be $V$.

Then:

$$
V = \frac{1}{3}(4) + \frac{2}{3}(4 + V)
$$

A cleaner way to write the same thing is:

$$
V = 4 + \frac{2}{3}V
$$

So:

$$
V - \frac{2}{3}V = 4
$$

$$
\frac{1}{3}V = 4
$$

$$
V = 12
$$

Therefore:
- always quit gives value 10;
- always stay gives value 12.

So the better policy is to stay.

## Important subtlety
In this lecture’s MDP, the only non-terminal state is `in`.
So a policy can only choose one action in that state.

That means a hybrid rule like “stay three times, then quit” is not representable unless the state includes additional information such as round number.

So in this specific formulation, it really is a straight comparison between:
- always quit;
- always stay.

---

## 14. Exact Policy Evaluation: Why Recurrences Help

Monte Carlo evaluation is conceptually simple but statistically noisy.

Can we compute the value more exactly?

Yes.

The key idea is exactly the same dynamic-programming intuition from search:

> value now = immediate effect + value of what happens next

The only extra ingredient is averaging over random next states.

---

## 15. Q-Values

The lecture introduces a very useful intermediate quantity:

$$
Q(s,a;V) = \sum_{s'} T(s,a,s') \big(R(s,a,s') + \gamma V(s')\big)
$$

Interpret it carefully.

### Step-by-step meaning
1. You are currently in state $s$.
2. You commit to action $a$.
3. Nature may send you to several possible next states $s'$.
4. Each successor is weighted by its probability.
5. For each successor, you collect:
   - immediate reward $R(s,a,s')$;
   - discounted continuation value $\gamma V(s')$.
6. Sum over all possible next states.

So a Q-value is a one-step lookahead quantity.

It says:
- “if I take this action now, and then continue according to values $V$, what is my expected return?”

---

## 16. Warm-Up: Computing a Q-Value in the Flaky Tram Problem

Suppose we are at state 9.

Under the “tram if possible” policy, state 9 cannot take the tram within bounds, so the action is walk.

Walking from 9:
- goes to 10 with probability 1;
- reward is $-1$.

If terminal state 10 has value 0, then:

$$
Q(9,\text{walk};V) = 1 \cdot (-1 + \gamma V(10))
$$

With $\gamma=1$ and $V(10)=0$:

$$
Q(9,\text{walk};V) = -1
$$

That is why state 9 quickly gets updated to value $-1$.

---

## 17. Bootstrapping

Bootstrapping is a key idea in RL and dynamic programming.

We start with some rough values, then repeatedly use them to compute improved values.

### Initial values in the lecture code
- terminal states get value 0;
- non-terminal states get a large negative placeholder, here $-100$.

This is not the true value.
It is just an initialization.

### Interpretation by iteration depth
At iteration 0:
- values represent “terminate immediately.”

At iteration 1:
- values represent “follow the policy for one step, then terminate.”

At iteration 2:
- “follow the policy for two steps, then terminate.”

And so on.

So each iteration propagates value information one step farther backward through the state graph.

This is exactly why values near the terminal state become correct first.

---

## 18. Policy Evaluation Recurrence

For a fixed policy $\pi$:

$$
V_\pi(s) = \sum_{s'} T(s,\pi(s),s') \big(R(s,\pi(s),s') + \gamma V_\pi(s')\big)
$$

This is the Bellman equation for policy evaluation.

### Iterative version
The lecture implements it iteratively:

$$
V_\pi^{(t)}(s) = \sum_{s'} T(s,\pi(s),s') \big(R(s,\pi(s),s') + \gamma V_\pi^{(t-1)}(s')\big)
$$

Read this as:
- new value at state $s$;
- equals expected immediate reward plus discounted old value of successor states;
- where the action is fixed by the policy.

### Critical point
Policy evaluation does not optimize over actions.
It simply asks:

> if I commit to policy $\pi$, how good is it?

---

## 19. Policy Evaluation Algorithm in Words

1. Initialize values.
2. For each state:
   - if terminal, set value to 0;
   - otherwise get the policy action $\pi(s)$;
   - get successors for that action;
   - compute the Q-value from current values.
3. Replace old values with new values.
4. Repeat until values stop changing much.

This is an iterative dynamic programming algorithm.

---

## 20. Convergence and the Distance Test

How do we know when to stop iterating?

We compare the old values and new values using the maximum absolute difference:

$$
\|V - V'\|_\infty = \max_s |V(s)-V'(s)|
$$

If this distance is below some tolerance, such as $10^{-5}$, we stop.

## Why the maximum?
Because we need all states to have stabilized.

Using the maximum ensures that no state is still changing significantly.

## Lecture intuition about the convergence plot
The lecture emphasizes two phases:

### Phase 1: reachability / propagation
The algorithm first propagates non-placeholder values backward through the state space.

### Phase 2: refinement
Once all relevant states have non-trivial values, the updates shrink rapidly.

That is why the convergence plot often looks like:
- flat/high at first;
- then sharply decreasing.

---

## 21. Exact Policy Evaluation for the Flaky Tram Policy

Consider the policy:
- take tram if possible;
- otherwise walk.

With $\gamma=1$, the exact values satisfy the following equations.

### States 6 through 9
These can only sensibly walk to the end.

$$
V(9) = -1
$$

$$
V(8) = -1 + V(9) = -2
$$

Actually, from 8 you walk directly to 9 then 10, so more carefully:

$$
V(8) = -1 + V(9) = -2
$$

$$
V(7) = -1 + V(8) = -3
$$

$$
V(6) = -1 + V(7) = -4
$$

### State 5
At state 5, the policy chooses tram:

$$
V(5) = 0.6(-2 + V(10)) + 0.4(-2 + V(5))
$$

Since $V(10)=0$:

$$
V(5) = 0.6(-2) + 0.4(-2 + V(5))
$$

A cleaner simplification is:

$$
V(5) = -2 + 0.4V(5)
$$

So:

$$
0.6V(5) = -2
$$

$$
V(5) = -\frac{10}{3} \approx -3.333
$$

### State 4

$$
V(4) = -2 + 0.6V(8) + 0.4V(4)
$$

Substitute $V(8)=-2$:

$$
V(4) = -2 + 0.6(-2) + 0.4V(4)
$$

$$
0.6V(4) = -3.2
$$

$$
V(4) \approx -5.333
$$

### State 3

$$
V(3) = -2 + 0.6V(6) + 0.4V(3)
$$

Using $V(6)=-4$:

$$
0.6V(3) = -4.4
$$

$$
V(3) \approx -7.333
$$

### State 2

$$
V(2) = -2 + 0.6V(4) + 0.4V(2)
$$

Using $V(4)\approx -5.333$:

$$
0.6V(2) = -5.2
$$

$$
V(2) \approx -8.667
$$

### State 1

$$
V(1) = -2 + 0.6V(2) + 0.4V(1)
$$

Using $V(2)\approx -8.667$:

$$
0.6V(1) = -7.2
$$

$$
V(1) = -12
$$

So the exact value of this policy from the start is:

$$
V_\pi(1) = -12
$$

Interpreting reward as negative cost, this means the expected travel time is 12 minutes.

This matches the lecture’s Monte Carlo estimate being “roughly in the ballpark” and policy evaluation sharpening it to the exact value.

---

## 22. From Policy Evaluation to Value Iteration

Now we ask a harder question:

> not “how good is this policy?” but “what is the best possible policy?”

At first glance, that sounds much harder.

But the lecture’s key insight is:
- policy evaluation already does almost all the hard work;
- the only missing step is choosing the best action.

This is why Percy says it is almost a tiny code change.

---

## 23. Value Iteration Recurrence

### Policy evaluation recurrence

$$
V_\pi(s) = \sum_{s'} T(s,\pi(s),s') \big(R(s,\pi(s),s') + \gamma V_\pi(s')\big)
$$

### Value iteration recurrence

$$
V^*(s) = \max_a \sum_{s'} T(s,a,s') \big(R(s,a,s') + \gamma V^*(s')\big)
$$

### Iterative form

$$
V^{(t)}(s) = \max_a \sum_{s'} T(s,a,s') \big(R(s,a,s') + \gamma V^{(t-1)}(s')\big)
$$

This is the Bellman optimality update.

### The only conceptual change
Instead of:
- taking the action prescribed by a fixed policy,

we now:
- evaluate every possible action,
- choose the maximum.

That is why value iteration is so closely related to policy evaluation.

---

## 24. Value Iteration Algorithm in Words

1. Initialize values.
2. For each state:
   - if terminal, set value to 0;
   - for each available action, compute its Q-value;
   - take the action with highest Q-value;
   - store both:
     - the best value,
     - the best action.
3. Repeat until the values stabilize.

At the end, you recover:
- $V^*(s)$, the optimal value function;
- $\pi^*(s)$, the optimal policy.

---

## 25. Why the Optimal Policy Can Be Taken as Deterministic

A student asked whether the optimal policy is always deterministic.

For standard MDPs of this kind, yes: you can always take an optimal deterministic policy.

Why?

Because the Bellman optimality equation uses:

$$
\max_a
$$

If one action achieves the maximum, just pick it.

If multiple actions tie for the maximum, any one of them is fine, or any mixture of them is also optimal.

So randomness is not needed to achieve optimality in ordinary MDPs.

### Important contrast
This will stop being true for game settings with adversaries, where mixed strategies can matter.

---

## 26. Solving the Flaky Tram Problem by Value Iteration

The lecture’s value iteration solution finds the optimal policy:

- walk until state 5;
- then take the tram.

Let us verify the logic.

### Terminal-side values

$$
V^*(9) = -1
$$

$$
V^*(8) = -2
$$

$$
V^*(7) = -3
$$

$$
V^*(6) = -4
$$

### State 5
Compare:

#### Walk

$$
Q(5,\text{walk}) = -1 + V^*(6) = -5
$$

#### Tram

$$
Q(5,\text{tram}) = -2 + 0.6V^*(10) + 0.4V^*(5)
$$

So if tram is chosen optimally:

$$
V^*(5) = -2 + 0.4V^*(5)
$$

$$
V^*(5) = -\frac{10}{3} \approx -3.333
$$

Since $-3.333 > -5$, tram is better.

### State 4
Compare:

#### Walk

$$
Q(4,\text{walk}) = -1 + V^*(5) \approx -4.333
$$

#### Tram

$$
Q(4,\text{tram}) = -2 + 0.6V^*(8) + 0.4V^*(4)
$$

This solves to about $-5.333$.

So walk is better.

### State 3

#### Walk

$$
Q(3,\text{walk}) = -1 + V^*(4) \approx -5.333
$$

#### Tram
Tram solves to about $-7.333$.

So walk is better.

### State 2

#### Walk

$$
Q(2,\text{walk}) = -1 + V^*(3) \approx -6.333
$$

#### Tram
Tram solves to about $-7.667$.

So walk is better.

### State 1

#### Walk

$$
Q(1,\text{walk}) = -1 + V^*(2) \approx -7.333
$$

#### Tram
Tram solves to about $-9.667$.

So walk is better.

Therefore the optimal policy is:

$$
\pi^*(1)=\text{walk},\ \pi^*(2)=\text{walk},\ \pi^*(3)=\text{walk},\ \pi^*(4)=\text{walk},\ \pi^*(5)=\text{tram}
$$

and then from 6 onward, walking is effectively forced/best.

So the optimal value from the start is:

$$
V^*(1) \approx -7.333
$$

Interpreted as cost, the expected travel time is about 7.333 minutes.

That is much better than:
- always walk: cost 9;
- tram whenever possible: cost 12.

### Intuition
The tram is risky.
If you use it too early, failures waste time repeatedly.
But at state 5, a successful tram takes you directly to 10, which is so valuable that the gamble becomes worthwhile.

This is a perfect example of what MDPs are good at:
- balancing immediate cost,
- future upside,
- and uncertainty.

---

## 27. Policy Evaluation vs Value Iteration

| Question | Policy Evaluation | Value Iteration |
|---|---|---|
| What does it compute? | Value of a fixed policy $V_\pi$ | Optimal value function $V^*$ |
| Action choice | Use policy action $\pi(s)$ | Max over all actions |
| Output | Values | Values and optimal policy |
| Recurrence | Bellman expectation equation | Bellman optimality equation |
| Optimization involved? | No | Yes |

### Core mental model
Policy evaluation answers:
- “If I commit to this behavior, how good is it?”

Value iteration answers:
- “What behavior should I commit to?”

---

## 28. Common Confusions to Avoid

### Confusion 1: utility vs value
Utility is for one rollout.
Value is expectation over rollouts.

### Confusion 2: action choice vs random outcome
The agent chooses the action.
Nature chooses which probabilistic successor occurs.

### Confusion 3: reward vs cost
They are just sign flips in this lecture.
Reward $=-$ cost.

### Confusion 4: policy vs path
In a stochastic world, you do not pre-commit to one path.
You commit to a mapping from states to actions.

### Confusion 5: Q-value vs value
- value asks: “how good is this state?”
- Q-value asks: “how good is this action from this state, assuming continuation values $V$?”

### Confusion 6: why terminal values are zero
Because once the process ends, there are no future rewards left to collect.

### Confusion 7: why value iteration and policy evaluation look so similar
Because both are Bellman-style dynamic programming recurrences.
The only difference is whether the action is fixed or optimized.

---

## 29. Connections to Previous Lecture on Search

There is a strong structural analogy with dynamic programming for search.

### Search DP idea
Future cost from state $s$ is:
- immediate step cost;
- plus future cost of successor state.

### MDP idea
Expected future value from state $s$ is:
- expected immediate reward;
- plus discounted expected value of successor states.

So MDPs are not a completely new way of thinking.
They are dynamic programming under uncertainty.

---

## 30. Why This Lecture Matters for Reinforcement Learning

This lecture is the foundation for reinforcement learning.

In RL, the agent typically does not know:
- transition probabilities exactly;
- rewards exactly;
- sometimes even the full state space.

But the objects remain the same:
- policies;
- rollouts;
- utilities;
- value functions;
- Q-values;
- Bellman recurrences.

So RL is not replacing MDPs.
It is solving MDP-like problems when the model is unknown and must be learned from interaction.

---

## 31. Real-World Applications

### Transportation and routing
Choose among walking, cycling, driving, trains, or buses under uncertain traffic and delay conditions.

### Robotics
A robot chooses actions, but motors can slip and sensors can be noisy.

### Finance
Sequential portfolio or execution decisions under uncertain price movements.

### Operations research
Inventory control, queuing, scheduling, and maintenance decisions under uncertain demand/failure.

### Medicine
Treatment decisions under uncertain patient outcomes.

### Online platforms
Ad placement, recommendation, or intervention decisions when user responses are stochastic.

---

## 32. Summary

### Core definitions
- MDPs generalize search by allowing stochastic transitions.
- The Markov property means the current state is a sufficient summary of the past.
- The solution to an MDP is a policy, not a fixed action sequence.

### Core quantities
- rollout = one sampled trajectory;
- utility = discounted sum of rewards for one rollout;
- value = expected utility of a policy;
- Q-value = value of taking one action and then continuing with values $V$.

### Core algorithms
- Monte Carlo policy evaluation estimates values by repeated simulation.
- Policy evaluation computes the value of a fixed policy exactly via a recurrence.
- Value iteration computes the optimal value function by replacing the fixed action with a max over actions.

### Core intuitions
- uncertainty means one action can lead to many possible futures;
- expected value replaces deterministic path cost;
- bootstrapping propagates value information backward through the state space;
- the Bellman update is the engine behind both policy evaluation and value iteration.

### Key takeaway from the flaky tram example
The optimal strategy is not:
- “always walk,” and not
- “always take the tram when possible.”

It is:
- walk first;
- take the tram only when the potential upside is large enough relative to the risk.

That is the essence of optimization under uncertainty.

---

## 33. Quick Self-Test Questions

1. Why is a fixed action sequence not a valid general solution for an MDP?
2. What is the precise difference between a rollout’s utility and a policy’s value?
3. What role does the transition function $T(s,a,s')$ play in policy evaluation?
4. Why is $Q(s,a;V)$ an expectation rather than a single number from one successor?
5. In the flaky tram example, why is state 5 the point where tram becomes worthwhile?
6. What is the only conceptual difference between policy evaluation and value iteration?
7. Why can ordinary MDPs be solved with deterministic optimal policies?
8. How does the discount factor change the tradeoff between short-term and long-term reward?

---

## 34. One-Sentence Memory Anchor

An MDP is dynamic programming for sequential decisions when actions lead not to one guaranteed future, but to a probability distribution over futures.
