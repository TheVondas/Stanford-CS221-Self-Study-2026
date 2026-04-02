#!/usr/bin/env python3
"""Grader for CS221 Homework 4."""
import sys
import math
import random
import numpy as np
import gymnasium as gym
from custom_mountain_car import CustomMountainCarEnv
import util_rl as util
from util_rl import ContinuousGymMDP, simulate

# Register custom env
gym.register(
    id="CustomMountainCar-v1",
    entry_point="custom_mountain_car:CustomMountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)


def fourier_feature_extractor(state, max_coeff=5, scale=None):
    if scale is None:
        scale = np.ones_like(state)
    coeffs = np.arange(max_coeff + 1)
    curr = coeffs * scale[0] * state[0]
    for i in range(1, len(state)):
        new = coeffs * scale[i] * state[i]
        curr = (curr.reshape(-1, 1) + new.reshape(1, -1)).flatten()
    return np.cos(np.pi * curr)


class FunctionApproxQLearning(util.RLAlgorithm):
    def __init__(self, feature_dim, feature_extractor, actions, discount, exploration_prob=0.2):
        self.feature_dim = feature_dim
        self.feature_extractor = feature_extractor
        self.actions = actions
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.w = np.random.standard_normal(size=(feature_dim, len(actions)))
        self.num_iters = 0

    def get_q(self, state, action):
        features = self.feature_extractor(state)
        action_idx = self.actions.index(action)
        return float(features @ self.w[:, action_idx])

    def get_action(self, state, explore=True):
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4:
            exploration_prob = 1.0
        elif self.num_iters > 1e5:
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)
        if explore and random.random() < exploration_prob:
            return random.choice(self.actions)
        q_values = [self.get_q(state, a) for a in self.actions]
        return self.actions[int(np.argmax(q_values))]

    def get_step_size(self):
        return 0.005 * (0.99)**(self.num_iters / 500)

    def incorporate_feedback(self, state, action, reward, next_state, terminal):
        eta = self.get_step_size()
        features = self.feature_extractor(state)
        action_idx = self.actions.index(action)
        q_old = float(features @ self.w[:, action_idx])
        if terminal:
            v_next = 0.0
        else:
            q_next = [self.get_q(next_state, a) for a in self.actions]
            v_next = max(q_next)
        target = reward + self.discount * v_next
        self.w[:, action_idx] += eta * (target - q_old) * features


class ConstrainedQLearning(FunctionApproxQLearning):
    def __init__(self, feature_dim, feature_extractor, actions, discount,
                 force, gravity, max_speed=None, exploration_prob=0.2):
        super().__init__(feature_dim, feature_extractor, actions, discount, exploration_prob)
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed

    def get_action(self, state, explore=True):
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4:
            exploration_prob = 1.0
        elif self.num_iters > 1e5:
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)

        position, velocity = state[0], state[1]
        valid_actions = []
        for a in self.actions:
            next_velocity = velocity + (a - 1) * self.force - math.cos(3 * position) * self.gravity
            if self.max_speed is None or abs(next_velocity) < self.max_speed:
                valid_actions.append(a)

        if len(valid_actions) == 0:
            return None

        if explore and random.random() < exploration_prob:
            return random.choice(valid_actions)

        q_values = [(self.get_q(state, a), a) for a in valid_actions]
        best_action = max(q_values, key=lambda x: x[0])[1]
        return best_action


def test_5c_helper():
    """Compare constrained Q-learning with two different max_speed values."""
    print("=" * 60)
    print("5c-helper: Comparing ConstrainedQLearning with two MDPs")
    print("=" * 60)

    mdp1 = ContinuousGymMDP("CustomMountainCar-v1", discount=0.999, time_limit=1000)
    mdp2 = ContinuousGymMDP("CustomMountainCar-v1", discount=0.999, time_limit=1000)

    print("\nTraining RL1 (max_speed=10000, effectively unconstrained)...")
    rl1 = ConstrainedQLearning(
        36,
        lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
        mdp1.actions,
        mdp1.discount,
        mdp1.env.unwrapped.force,
        mdp1.env.unwrapped.gravity,
        10000,
        exploration_prob=0.2,
    )
    simulate(mdp1, rl1, num_trials=1000, train=True, verbose=True)

    print("\nTraining RL2 (max_speed=0.065, constrained)...")
    rl2 = ConstrainedQLearning(
        36,
        lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
        mdp2.actions,
        mdp2.discount,
        mdp2.env.unwrapped.force,
        mdp2.env.unwrapped.gravity,
        0.065,
        exploration_prob=0.2,
    )
    simulate(mdp2, rl2, num_trials=1000, train=True, verbose=True)

    def count_actions(mdp, rl, label):
        accelerate_left, no_accelerate, accelerate_right = 0, 0, 0
        for n in range(100):
            traj = util.sample_rl_trajectory(mdp, rl, train=False)
            accelerate_left += traj.count(0)
            no_accelerate += traj.count(1)
            accelerate_right += traj.count(2)
        print(f"\nRL with MDP -> max_speed:{rl.max_speed} ({label})")
        print(f"  * total accelerate left actions: {accelerate_left}, "
              f"total no acceleration actions: {no_accelerate}, "
              f"total accelerate right actions: {accelerate_right}")

    print("\n" + "=" * 60)
    print("Comparing learned policies (100 test trajectories each)")
    print("=" * 60)
    count_actions(mdp1, rl1, "unconstrained")
    count_actions(mdp2, rl2, "constrained to 0.065")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grader.py <test-name>")
        print("Available tests: 5c-helper")
        sys.exit(1)

    test_name = sys.argv[1]
    if test_name == "5c-helper":
        test_5c_helper()
    else:
        print(f"Unknown test: {test_name}")
        sys.exit(1)
