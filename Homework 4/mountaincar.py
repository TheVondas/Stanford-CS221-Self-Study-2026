import argparse
import pickle
import numpy as np
import gymnasium as gym
from util_rl import ContinuousGymMDP, DiscreteGymMDP, RandomAgent, FixedRLAlgorithm, simulate


def fourier_feature_extractor(state, max_coeff=5, scale=None):
    if scale is None:
        scale = np.ones_like(state)
    coeffs = np.arange(max_coeff + 1)
    curr = coeffs * scale[0] * state[0]
    for i in range(1, len(state)):
        new = coeffs * scale[i] * state[i]
        curr = (curr.reshape(-1, 1) + new.reshape(1, -1)).flatten()
    return np.cos(np.pi * curr)


class FunctionApproxFixedAgent(FixedRLAlgorithm):
    """Agent that uses pre-trained function approximation weights."""
    def __init__(self, w, feature_extractor, actions, exploration_prob=0.0):
        self.w = w
        self.feature_extractor = feature_extractor
        self.actions = actions
        self.exploration_prob = exploration_prob

    def get_action(self, state, explore=True):
        import random as _random
        if explore and _random.random() < self.exploration_prob:
            return _random.choice(self.actions)
        features = self.feature_extractor(state)
        q_values = features @ self.w
        return self.actions[int(np.argmax(q_values))]

    def incorporate_feedback(self, state, action, reward, next_state, terminal):
        pass


def main():
    parser = argparse.ArgumentParser(description="Mountain Car RL")
    parser.add_argument("--agent", type=str, default="naive",
                        choices=["naive", "value-iteration", "tabular", "function-approximation"],
                        help="Agent type to use")
    parser.add_argument("--mdp", type=str, default="discrete", choices=["continuous", "discrete"],
                        help="MDP type: continuous or discrete")
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Number of trials to simulate")
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of bins for discretization")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment visually")
    args = parser.parse_args()

    # Create MDP
    if args.mdp == "continuous":
        mdp = ContinuousGymMDP("MountainCar-v0", discount=0.99)
    else:
        mdp = DiscreteGymMDP("MountainCar-v0", feature_bins=args.bins, discount=0.99)

    # Enable rendering if requested
    if args.render:
        mdp.env.close()
        mdp.env = gym.make("MountainCar-v0", render_mode="human")

    # Create agent
    if args.agent == "naive":
        agent = RandomAgent(mdp.actions)
    elif args.agent == "value-iteration":
        with open("vi_weights.pkl", "rb") as f:
            weights = pickle.load(f)
        pi_actions = weights["pi_actions"]
        pi = {}
        for i in range(len(pi_actions)):
            state = mdp.index_to_state(i)
            if pi_actions[i] is not None:
                pi[state] = pi_actions[i]
        agent = FixedRLAlgorithm(pi, mdp.actions, exploration_prob=0.0)
    elif args.agent == "tabular":
        with open("tabular_weights.pkl", "rb") as f:
            weights = pickle.load(f)
        q = weights["q"]
        pi = {}
        for i in range(q.shape[0]):
            state = mdp.index_to_state(i)
            pi[state] = mdp.actions[int(np.argmax(q[i]))]
        agent = FixedRLAlgorithm(pi, mdp.actions, exploration_prob=0.0)
    elif args.agent == "function-approximation":
        with open("fa_weights.pkl", "rb") as f:
            weights = pickle.load(f)
        agent = FunctionApproxFixedAgent(
            w=weights["w"],
            feature_extractor=lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
            actions=mdp.actions,
            exploration_prob=0.0,
        )

    print(f"Running Mountain Car with {args.agent} agent ({args.mdp} MDP)")
    print(f"Number of trials: {args.num_trials}")
    print("-" * 50)

    rewards = simulate(mdp, agent, num_trials=args.num_trials, train=False, verbose=True, demo=args.render)

    print("-" * 50)
    print(f"Average reward: {sum(rewards) / len(rewards):.2f}")
    print(f"Max reward: {max(rewards):.2f}")
    print(f"Min reward: {min(rewards):.2f}")


if __name__ == "__main__":
    main()
