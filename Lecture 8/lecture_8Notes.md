[[Stanford CS221 Autumn 2025]]
## Learning Objectives

By the end of this lecture, you should be able to:

- explain how reinforcement learning (RL) extends the MDP framework

- state precisely what information is known in an MDP versus unknown in RL

- distinguish between a policy and an RL agent

- explain why exploration is necessary when the environment is unknown

- describe model-based RL and how it estimates an MDP from experience

- explain how model-free Monte Carlo estimates $Q$-values from complete rollouts

- explain why Monte Carlo is limited when episodes are long

- understand bootstrapping and why SARSA can update before the episode ends

- distinguish on-policy learning from off-policy learning

- explain the key difference between SARSA and Q-learning

- connect $V$, $Q$, policy evaluation, value iteration, SARSA, and Q-learning into one coherent picture

---

## Concept Inventory

### 1. MDP recap

A Markov Decision Process is defined by:

- a start state

- successors, which specify:

    - action

    - probability

    - reward

    - next state

- an end-state test

- a discount factor $\gamma$

In this lecture, the running example is the flaky tram MDP:

- states are locations $1$ through $10$

- from state $i$:

    - you can walk to $i+1$

    - or take the tram to $2i$

- the tram is flaky:

    - with probability $0.6$, it succeeds

    - with probability $0.4$, it fails and you stay put

Important interpretation:

- reward is just negative cost

- so minimising cost is equivalent to maximising reward

---

### 2. State nodes and chance nodes

The MDP graph has two kinds of nodes:

- state nodes: where the agent chooses an action

- chance nodes: where nature samples the outcome


This is the core difference from deterministic search:

- in search, an action leads to one next state

- in an MDP, an action can lead to multiple possible next states with probabilities

---

### 3. Policy

A policy $\pi$ maps each state to an action:

$$  
\pi(s) = a  
$$

In deterministic search, a solution can be a sequence of actions.

In an MDP, that is not enough, because randomness means you may not arrive where you expected. Therefore, a solution must specify what to do in every state you might encounter.

---

### 4. Rollout

A rollout is one sampled trajectory through the MDP:

- start in the initial state

- choose actions according to a policy

- let the environment sample transitions

- collect rewards along the way

A rollout is one concrete experience, not the full expected behavior.

---

### 5. Utility of a rollout

The utility of a rollout is the discounted sum of rewards:

$$  
U = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots  
$$

If $\gamma = 1$, utility is just the ordinary sum of rewards.

If $\gamma < 1$, future rewards matter less than immediate rewards.

Interpretation of the discount factor:

- it downweights future outcomes

- it encodes preference for earlier reward

- it helps with infinite-horizon problems by making sums finite

- it reflects uncertainty about the far future

---

### 6. Value of a policy

The value of a policy is the expected utility when following that policy from a state:

$$  
V_\pi(s) = \mathbb{E}[U \mid \text{start at } s, \text{ follow } \pi]  
$$

This is not one rollout. It is the average over all possible rollouts induced by the policy and the environment’s randomness.

---

### 7. Q-value

The $Q$-value asks a slightly more specific question:

$$  
Q_\pi(s,a) = \text{value of taking action } a \text{ in state } s \text{, then following } \pi  
$$

So:

- $V_\pi(s)$ = value of the state under policy $\pi$

- $Q_\pi(s,a)$ = value of a specific action from that state under policy $\pi$

They are related by:

$$  
V_\pi(s) = Q_\pi(s,\pi(s))  
$$

---

### 8. Policy evaluation recurrence

For a fixed policy $\pi$:

$$  
V_\pi(s) = Q_\pi(s,\pi(s))  
$$

and

$$  
Q_\pi(s,a) = \sum_{s'} T(s,a,s') \Big(R(s,a,s') + \gamma V_\pi(s')\Big)  
$$

Meaning:

- consider every possible next state $s'$

- weight by the transition probability

- add immediate reward plus discounted future value

This is the recurrence behind policy evaluation.

---

### 9. Optimal value and value iteration

For the optimal policy:

$$  
V^*_{s} = \max_a Q^*_{s,a}  
$$

and

$$  
Q^*_{(s,a)} = \sum_{s'} T(s,a,s') \Big(R(s,a,s') + \gamma V^*_{(s')}\Big)  
$$

Then the optimal policy is:

$$  
\pi^*_{(s)} = \arg\max_a Q^*_{(s,a)}  
$$

This is the recurrence behind value iteration.

Core idea:

- policy evaluation computes the value of one fixed policy

- value iteration computes the value of the best possible policy

---

### 10. Reinforcement learning

Reinforcement learning is the setting where:

- the agent interacts with an environment

- the agent chooses actions

- the environment returns rewards and observations

- the agent must improve from experience

In this lecture, the environment is assumed to be an MDP, and the observation is the full state.

So the simplification is:

- fully observed world

- unknown transition dynamics and rewards at the start

That means RL here is basically:

> MDP optimisation when the MDP is not given in advance.

---

### 11. Agent versus policy

This distinction matters.

A policy is static:

- input: state

- output: action

- does not change

An RL agent is dynamic:

- it chooses actions

- it receives feedback

- it updates itself over time

The lecture frames every RL algorithm as having two methods:

- `get_action(state)`

- `incorporate_feedback(state, action, reward, next_state, is_end)`

That is the correct mental model.

A policy is just behavior.  
An RL agent is behavior plus learning.

---

### 12. Exploration versus exploitation

This is the central tension in RL.

- Exploration: try actions to gather information

- Exploitation: use current knowledge to choose what seems best

If you only exploit:

- you may get stuck doing something suboptimal forever

If you only explore:

- you never capitalize on what you have learned

Good RL balances both.

---

### 13. Epsilon-greedy

The lecture uses $\epsilon$-greedy exploration.

With probability $\epsilon$:

- choose a random exploratory action

With probability $1-\epsilon$:

- choose the action that currently looks best

So:

- larger $\epsilon$ means more exploration
- smaller $\epsilon$ means more exploitation

This is a simple but extremely important RL strategy.

---

### 14. On-policy versus off-policy

This distinction becomes crucial in SARSA versus Q-learning.

On-policy:

- you learn the value of the policy you are actually following

Off-policy:

- you behave using one policy, but learn the value of another policy

SARSA is on-policy.  
Q-learning is off-policy.

---

### 15. Bootstrapping

Bootstrapping means:

- instead of waiting for the full return all the way to the end,

- estimate the future using your current value function


So instead of:

$$  
U = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots  
$$

you approximate:

$$  
U \approx r_0 + \gamma \hat{Q}(\text{next state, next action})  
$$

This lets you update immediately after one step.

It is the key move behind temporal-difference methods like SARSA and Q-learning.

---

## Slide-by-Slide Walkthrough

## MDP review

The lecture starts by re-grounding everything in the MDP formalism.

You are reminded that an MDP requires:

- start state

- successors

- end test

- discount factor

This is not filler. It matters because RL will inherit all of this structure. The only new twist is that the agent does not initially know the MDP.

Key takeaway:  
RL is not replacing MDPs. RL is solving MDP-like problems without being handed the model.

---

## Graphical view of an MDP

The flaky tram example is shown again as a graph with:

- state nodes where the agent decides

- chance nodes where the environment samples

This is important because it makes the decomposition of decision and uncertainty visually explicit.

The agent chooses whether to walk or tram.  
Nature decides whether the tram succeeds.

That separation is exactly why $Q$-value recurrences are sums over possible next states.

---

## Policy, rollout, utility, value

The lecture reviews:

- policy $\pi$

- rollout

- utility

- value of a policy

This is the bridge between planning and learning.

You need to be absolutely clear on this:

- a rollout is one sample path

- value is an expectation over many possible rollout paths

A common mistake is to confuse “what happened once” with “what is expected in general.” RL learns from individual rollouts, but its goal is to approximate expected value.

---

## Recurrences behind policy evaluation and value iteration

Percy then re-expresses the ideas mathematically.

For a fixed policy:

$$  
V_\pi(s) = Q_\pi(s,\pi(s))  
$$

$$  
Q_\pi(s,a) = \sum_{s'} T(s,a,s') \Big(R(s,a,s') + \gamma V_\pi(s')\Big)  
$$

For the optimal policy:

$$  
V^*_{(s)} = \max_a Q^*_{(s,a)}  
$$

$$  
Q^*_{(s,a)} = \sum_{s'} T(s,a,s') \Big(R(s,a,s') + \gamma V^*_{(s')}\Big)  
$$

$$  
\pi^*_{(s)} = \arg\max_a Q^*_{(s,a)}  
$$

This section is conceptually central.

The entire rest of the lecture is asking:

> If we do not know $T$ and $R$, how can we still estimate the right $Q$-values and therefore the right policy?

That is the real transition from MDP planning to RL.

---

## RL setup: agent and environment

Now the lecture introduces the generic RL loop:

1. agent takes action

2. environment returns reward and observation

3. repeat

This is intentionally minimal.

Why?  
Because almost every RL algorithm can be viewed as repeated interaction of this form.

The lecture also notes that in real life, observations are often partial, leading to POMDPs. But this lecture assumes full observability, where the observation is simply the next state.

Important simplification:

- we are not solving partial observability here

- we are learning in a fully observed MDP-like environment

---

## Static agent and simulation

Before learning, Percy defines the simplest possible “agent”:

- store a fixed policy

- use it in `get_action`

- ignore all feedback in `incorporate_feedback`

This is pedagogically smart, because it isolates what is new about RL.

Simulation works like this:

- start from the environment’s start state

- ask the agent for an action

- sample the environment transition

- send the resulting feedback to the agent

- continue until terminal state

- compute the rollout utility

- average over many trials

The key point is:

- even a non-learning policy can be wrapped in the RL interface

- what makes it RL is not the loop itself

- what makes it RL is updating from feedback

---

## Model-based RL

This is the first real learning algorithm.

### Big idea

If RL is hard because the MDP is unknown, then estimate the MDP from experience.

So model-based RL proceeds in three stages:

1. explore and collect transitions

2. build an estimated MDP

3. run value iteration on the estimated MDP

This is conceptually the most natural algorithm if you already understand MDPs.

### What is being learned?

The agent estimates:

- start state

- rewards $R(s,a,s')$

- transition counts for $(s,a,s')$

- end states

From transition counts, it estimates probabilities by normalisation:

$$  
\hat{T}(s,a,s') = \frac{\text{count}(s,a,s')}{\sum_{s''}\text{count}(s,a,s'')}  
$$

### Why random exploration?

Because if you never try an action, you cannot estimate what it does.

This is a brutally important RL lesson:  
untried actions remain unknown actions.

### Strength of model-based RL

It learns a full internal model of the world.

That means it does not just learn what action is good.  
It learns what happens if you take actions.

This is more general and often more interpretable.

### Weakness

It can be expensive or difficult to learn the full model accurately, especially when the state space is large or complex.

### Lecture takeaway

Model-based RL is:

- principled

- close to the original MDP formulation

- often sample-efficient in small structured settings

- but potentially hard in very large environments

---

## Why look for a more direct method?

The lecture then asks a smart question:

> Do we really need to estimate the whole MDP if our final goal is just a good policy?

This is where model-free RL begins.

Instead of estimating:

- transitions $T$

- rewards $R$

and then deriving $Q$,

maybe we can estimate $Q$ directly.

That is the conceptual leap.

---

## Model-free Monte Carlo

### Big idea

Estimate action values directly from complete rollouts.

Recall:

$$  
Q_\pi(s,a) = \text{expected utility of taking } a \text{ in } s \text{ and then following } \pi  
$$

So if you experience many complete trajectories, you can average the realized returns for each $(s,a)$ pair.

### Utility-from-each-step idea

If a rollout is:

- step 0: reward $r_0$

- step 1: reward $r_1$

- step 2: reward $r_2$

then the utility from each step is:

$$  
u_0 = r_0 + \gamma r_1 + \gamma^2 r_2  
$$

$$  
u_1 = r_1 + \gamma r_2  
$$

$$  
u_2 = r_2  
$$

with recurrence:

$$  
u_t = r_t + \gamma u_{t+1}  
$$

This lets the algorithm assign a return to every visited state-action pair in the episode.

### What the algorithm stores

For each $(s,a)$:

- total sum of observed returns

- count of visits

Then:

$$  
\hat{Q}(s,a) = \frac{\text{sum of returns}}{\text{count of visits}}  
$$

### Important nuance

This estimates $Q_\pi$, not directly $Q^*$.

That was explicitly caught in the lecture discussion, and it is an important correction.

Monte Carlo here is evaluating the current policy induced by the current exploration/exploitation mix.

So this method is on-policy in spirit.

### Strength

- conceptually simple

- no need to estimate the full model

- directly tied to actual experience

### Weakness

It must wait until the episode ends before updating.

If the episode is very long, learning is slow.

If the task is continuing and has no natural terminal state, plain Monte Carlo is awkward.

---

## SARSA

SARSA solves the “must wait until the end” problem.

The name comes from the update tuple:

- State

- Action

- Reward

- next State

- next Action

Hence SARSA.

### Big idea

Use one-step bootstrapping.

Instead of waiting for the full return:

$$  
Q_\pi(s,a) \leftarrow \text{full rollout return}  
$$

SARSA uses:

$$  
u = r + \gamma Q_\pi(s',a')  
$$

where $a'$ is the next action actually chosen by the current policy.

Then update toward that target:

$$  
Q_\pi(s,a) \leftarrow Q_\pi(s,a) + \alpha \Big(u - Q_\pi(s,a)\Big)  
$$

Substituting for $u$ gives:

$$  
Q_\pi(s,a) \leftarrow Q_\pi(s,a) + \alpha \Big(r + \gamma Q_\pi(s',a') - Q_\pi(s,a)\Big)  
$$

This is a temporal-difference update.

### Why this works conceptually

You do not know the full future yet.  
But you do know:

- the immediate reward $r$

- your current estimate of future value after the next step

So you combine reality now with your best guess about the future.

That is bootstrapping.

### Why SARSA is on-policy

Because the update uses the next action $a'$ that the agent actually takes under its current behavior policy, including exploration.

So SARSA learns the value of the behavior policy itself.

This means SARSA naturally accounts for the fact that you are still exploring.

### Intuition

SARSA asks:

> “Given how I actually behave, including occasional random exploration, how good is this action?”

That makes it more conservative in some settings.

---

## Q-learning

Now the final step:

> Can we estimate the optimal action-values directly?

Yes. That is Q-learning.

### Big idea

Keep the same one-step bootstrapping structure, but replace the next action actually taken with the best possible next action according to the current estimate.

So instead of:

$$  
u = r + \gamma Q(s',a')  
$$

use:

$$  
u = r + \gamma \max_{a'} Q(s',a')  
$$

Then update:

$$  
Q(s,a) \leftarrow Q(s,a) + \alpha \Big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big)  
$$

### Why this is off-policy

Behaviour may still be $\epsilon$-greedy and exploratory.

But the target assumes that from the next state onward, the agent behaves optimally.

So:

- behaviour policy = exploratory

- target policy = greedy/optimal estimate

That separation makes Q-learning off-policy.

### Intuition

Q-learning asks:

> “Even if I am currently exploring, what would the return be if I behaved optimally from the next state onward?”

That is why it can learn toward $Q^*$ even while behaving non-greedily.

### Relationship to SARSA

SARSA target:

$$  
r + \gamma Q(s',a')  
$$

Q-learning target:

$$  
r + \gamma \max_{a'} Q(s',a')  
$$

That one change is the entire conceptual difference.

Do not treat that as cosmetic.  
It changes what is being learned.

---

## End-of-lecture synthesis

The lecture ends with a clean hierarchy:

### Model-based value iteration

- learn the MDP

- run value iteration

- act using derived policy

### Model-free Monte Carlo

- learn $Q$ from complete rollouts

- no model needed

- but must wait until episode ends

### SARSA

- learn $Q_\pi$ online

- bootstrap

- on-policy

### Q-learning

- learn toward $Q^*$

- bootstrap

- off-policy

That is the core conceptual staircase of the lecture.

---

## Putting the Algorithms Side by Side

## 1. Model-Based Value Iteration

What it learns:

- estimated $T$

- estimated $R$

How it gets a policy:

- run value iteration on estimated model

Pros:

- learns a world model

- highly interpretable

- connects directly to MDP theory

Cons:

- must estimate the whole model

- can be hard in large spaces

---

## 2. Model-Free Monte Carlo

What it learns:

- empirical estimates of $Q_\pi(s,a)$ from complete episodes

Update style:

- only after terminal state

Pros:

- simple

- no need to estimate model

Cons:

- must wait until episode ends

- can be slow for long horizons

---

## 3. SARSA

What it learns:

- $Q_\pi(s,a)$ for the current behaviour policy

Update target:  
$$  
r + \gamma Q(s',a')  
$$

Pros:

- online updates

- naturally accounts for exploratory behaviour

Cons:

- learns value of current policy, not directly optimal policy

---

## 4. Q-learning

What it learns:

- approximation to $Q^*(s,a)$

Update target:  
$$  
r + \gamma \max_{a'} Q(s',a')  
$$

Pros:

- directly targets optimal action-values

- foundational RL algorithm

Cons:

- can be unstable in complex settings

- still depends on good exploration

---

## Common Pitfalls and Clarifications

### 1. “RL means no MDP”

Wrong.

In this lecture, RL is solving an environment assumed to behave like an MDP, except the agent does not know the model in advance.

---

### 2. “Monte Carlo directly gives $Q^*$”

Not in this lecture’s setup.

It estimates the value of the policy being followed, so it is better thought of as estimating $Q_\pi$.

---

### 3. “SARSA and Q-learning are basically identical”

Superficially yes, conceptually no.

The update target is different, and that changes the learning objective:

- SARSA: current policy

- Q-learning: optimal policy

---

### 4. “A policy and an agent are the same”

No.

A policy is a mapping.  
An agent is a learning system that may change its mapping over time.

---

### 5. “Exploration is just inefficiency”

No.

Exploration is the price you pay to reduce uncertainty.  
Without it, you may confidently lock into the wrong behavior forever.

---

### 6. “Reward only happens at the end”

Not necessarily.

Rewards can occur on every transition.  
Sparse-reward problems are a special case, not the default.

---

## Intuition Check

A good way to mentally separate the algorithms is this:

- Model-based RL says:

    - “Let me first learn how the world works.”

- Monte Carlo says:

    - “Let me just remember how good actions turned out after full episodes.”

- SARSA says:

    - “Let me update using what I actually did next.”

- Q-learning says:

    - “Let me update using what the best next action would be.”


If you can say those four sentences and explain them, you understand the lecture’s backbone.

---

## Summary

Reinforcement learning is the problem of learning good behavior from interaction rather than from a fully specified MDP model.

This lecture starts from the MDP foundation:

- states

- actions

- rewards

- transition uncertainty

- policy

- value

- $Q$-values

- value iteration

Then it asks the central question:

> What if the MDP is unknown?

The answer is RL.

The lecture frames an RL algorithm as an agent with two jobs:

- choose actions

- incorporate feedback

From there, it builds a progression of algorithms:

- Model-based value iteration learns an internal model of the world, then plans inside that learned model.

- Model-free Monte Carlo skips the model and estimates action values directly from complete rollouts.

- SARSA improves on Monte Carlo by updating during the rollout via bootstrapping, but it learns the value of the current behavior policy.

- Q-learning changes the target so that it learns toward the optimal action-values, making it off-policy.

The deepest conceptual distinctions from this lecture are:

- known MDP versus unknown environment

- policy versus agent

- model-based versus model-free

- Monte Carlo versus bootstrapping

- on-policy versus off-policy

- SARSA versus Q-learning

If Lecture 7 was about planning under uncertainty when the model is known, Lecture 8 is about learning to plan when the model must be discovered through experience.

---

## Real-World Applications

### Robotics

A robot often does not know the exact probabilities of success for each action in advance. It must try actions, observe rewards, and improve over time.

### Game playing

An agent can learn which moves lead to long-term success without being given explicit transition probabilities for every board state.

### Recommendation systems

A system can explore different suggestions, observe feedback such as clicks or watch time, and adapt its policy over time.

### Operations and control

In inventory, routing, or scheduling problems, an agent can learn better decisions from repeated interaction with a noisy environment.

### Self-driving systems

A vehicle may need to learn which behaviours are safer or more effective under uncertainty, especially when exact environment dynamics are too complex to model perfectly.

### Personal decision-making analogy

The lecture’s “metaphor for life” is actually useful:  
you often do not know the transition probabilities of your choices in advance, so you act, get feedback, and adjust your behaviour. That is the informal human analogue of reinforcement learning.

---