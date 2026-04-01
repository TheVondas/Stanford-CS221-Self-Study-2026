import argparse
import gymnasium as gym
from util_rl import ContinuousGymMDP, DiscreteGymMDP, RandomAgent, simulate


def main():
    parser = argparse.ArgumentParser(description="Mountain Car RL")
    parser.add_argument("--agent", type=str, default="naive", choices=["naive"],
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
