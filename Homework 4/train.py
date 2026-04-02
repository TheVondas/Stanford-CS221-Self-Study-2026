import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from util_rl import DiscreteGymMDP, ContinuousGymMDP, simulate

# Import implementations from the notebook-equivalent code
import math, random
from typing import List, Optional, Tuple, Any, Iterable, Callable
import util_rl as util


def value_iteration(transitions, rewards, discount, epsilon=0.001, valid_actions=None, state_ids=None, action_ids=None):
    transitions = np.asarray(transitions, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    num_states, num_actions, _ = transitions.shape

    action_mask = np.ones((num_states, num_actions), dtype=bool) if valid_actions is None else np.asarray(valid_actions, dtype=bool)
    state_ids = np.arange(num_states) if state_ids is None else np.array(list(state_ids), dtype=object)
    action_ids = np.arange(num_actions) if action_ids is None else np.array(list(action_ids), dtype=object)
    tie_breaker = (np.arange(num_actions, dtype=np.float64) * 1e-12)[np.newaxis, :]

    def compute_q(v):
        return np.sum(transitions * (rewards + discount * v[np.newaxis, np.newaxis, :]), axis=2)

    def compute_policy(q):
        masked_q = np.where(action_mask, q + tie_breaker, -np.inf)
        best_actions = np.argmax(masked_q, axis=1)
        best_values = q[np.arange(num_states), best_actions]
        best_values[~action_mask.any(axis=1)] = 0.0
        return best_actions, best_values

    print('Running value iteration...')
    terminal_mask = ~action_mask.any(axis=1)
    v = np.zeros(num_states, dtype=np.float64)

    while True:
        q = compute_q(v)
        best_actions, new_v = compute_policy(q)
        new_v[terminal_mask] = 0.0
        if np.max(np.abs(new_v - v)) < epsilon:
            v = new_v
            break
        v = new_v

    policy = np.full(num_states, None, dtype=object)
    for i in range(num_states):
        if action_mask[i].any():
            policy[i] = action_ids[best_actions[i]]
    return policy


class ModelBasedMonteCarlo(util.RLAlgorithm):
    def __init__(self, actions, discount, num_states, state_to_index,
                 index_to_state=None, calc_val_iter_every=10000, exploration_prob=0.2):
        self.actions = list(actions)
        self.discount = discount
        self.num_states = int(num_states)
        self.state_to_index = state_to_index
        self.index_to_state = index_to_state or (lambda idx: idx)
        self.calc_val_iter_every = int(calc_val_iter_every)
        self.exploration_prob = exploration_prob
        self.num_iters = 0

        self.num_actions = len(self.actions)
        self.actions_array = np.array(self.actions, dtype=object)
        self.state_ids = np.array([self.index_to_state(i) for i in range(self.num_states)], dtype=object)

        self.transition_counts = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float64)
        self.reward_sums = np.zeros_like(self.transition_counts)
        self.valid_actions = np.zeros((self.num_states, self.num_actions), dtype=bool)

        self.pi_actions = np.full(self.num_states, None, dtype=object)
        self.pi_indices = np.full(self.num_states, -1, dtype=int)

    def _sync_policy_indices(self):
        self.pi_indices[:] = -1
        valid_mask = self.pi_actions != None
        for idx in np.where(valid_mask)[0]:
            self.pi_indices[idx] = int(np.argmax(self.actions_array == self.pi_actions[idx]))

    def get_action(self, state, explore=True):
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4:
            exploration_prob = 1.0
        elif self.num_iters > 1e6:
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)
        state_idx = int(self.state_to_index(state))
        policy_idx = self.pi_indices[state_idx]

        if explore and random.random() < exploration_prob:
            return random.choice(self.actions)
        if policy_idx == -1:
            return random.choice(self.actions)
        return self.actions[policy_idx]

    def incorporate_feedback(self, state, action, reward, next_state, terminal):
        state_idx = int(self.state_to_index(state))
        matches = np.where(self.actions_array == action)[0]
        action_idx = int(matches[0])
        next_idx = int(self.state_to_index(next_state))

        self.transition_counts[state_idx, action_idx, next_idx] += 1.0
        self.reward_sums[state_idx, action_idx, next_idx] += reward
        self.valid_actions[state_idx, action_idx] = True

        if self.num_iters > 0 and self.num_iters % self.calc_val_iter_every == 0:
            count_sums = self.transition_counts.sum(axis=2, keepdims=True)
            safe_sums = np.where(count_sums > 0, count_sums, 1.0)
            transitions = self.transition_counts / safe_sums

            safe_counts = np.where(self.transition_counts > 0, self.transition_counts, 1.0)
            rewards = self.reward_sums / safe_counts

            self.pi_actions = value_iteration(
                transitions, rewards, self.discount,
                valid_actions=self.valid_actions,
                state_ids=self.state_ids,
                action_ids=self.actions_array,
            )
            self._sync_policy_indices()


def train_value_iteration(num_training_trials=1000, num_test_trials=20, num_runs=3, bins=10):
    all_train_rewards = []

    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Trial {run + 1}/{num_runs}")
        print(f"{'='*60}")

        mdp = DiscreteGymMDP("MountainCar-v0", feature_bins=bins, discount=0.99)
        rl = ModelBasedMonteCarlo(
            actions=mdp.actions,
            discount=mdp.discount,
            num_states=mdp.num_states,
            state_to_index=mdp.state_to_index,
            index_to_state=mdp.index_to_state,
            calc_val_iter_every=10000,
            exploration_prob=0.2,
        )

        train_rewards = simulate(mdp, rl, num_trials=num_training_trials, train=True, verbose=True)
        all_train_rewards.append(train_rewards)

        print(f"\nTesting trial {run + 1}...")
        test_rewards = simulate(mdp, rl, num_trials=num_test_trials, train=False, verbose=False)
        print(f"Test avg reward: {np.mean(test_rewards):.2f}")

    # Save weights from last trial
    with open("vi_weights.pkl", "wb") as f:
        pickle.dump({
            "pi_actions": rl.pi_actions,
            "pi_indices": rl.pi_indices,
            "transition_counts": rl.transition_counts,
            "reward_sums": rl.reward_sums,
            "valid_actions": rl.valid_actions,
        }, f)
    print("\nWeights saved to vi_weights.pkl")

    return all_train_rewards


def plot_rewards(all_train_rewards, window=50):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, rewards in enumerate(all_train_rewards):
        # Compute rolling average
        rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(rewards)), rolling, label=f"Trial {i + 1}", alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-episode rolling avg)")
    ax.set_title("Model-Based Value Iteration: Training Reward per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vi_training_curves.png", dpi=150)
    print("Plot saved to vi_training_curves.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train Mountain Car RL agents")
    parser.add_argument("--agent", type=str, required=True,
                        choices=["value-iteration"],
                        help="Agent type to train")
    parser.add_argument("--num-trials", type=int, default=1000,
                        help="Number of training episodes per run")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of independent training runs")
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of bins for discretization")
    args = parser.parse_args()

    if args.agent == "value-iteration":
        all_rewards = train_value_iteration(
            num_training_trials=args.num_trials,
            num_runs=args.num_runs,
            bins=args.bins,
        )
        plot_rewards(all_rewards)


if __name__ == "__main__":
    main()
