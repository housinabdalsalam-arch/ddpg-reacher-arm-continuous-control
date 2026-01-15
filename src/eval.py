import argparse
import numpy as np

from unityagents import UnityEnvironment
from ddpg_agent import DDPGAgent, DDPGConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, type=str, help="Path to Unity Reacher executable")
    p.add_argument("--actor", type=str, default="checkpoint_actor.pth")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_t", type=int, default=1000)
    args = p.parse_args()

    env = UnityEnvironment(file_name=args.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    agent = DDPGAgent(state_size, action_size, seed=0, cfg=DDPGConfig())
    agent.load_actor(args.actor)

    means = []
    for ep in range(1, args.episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents, dtype=np.float32)

        for _ in range(args.max_t):
            actions = agent.act(states, add_noise=False)
            env_info = env.step(actions)[brain_name]
            states = env_info.vector_observations
            scores += np.array(env_info.rewards, dtype=np.float32)
            if np.any(env_info.local_done):
                break

        m = float(np.mean(scores))
        means.append(m)
        print(f"Episode {ep:02d}: mean score = {m:.2f}")

    print("Average over episodes:", float(np.mean(means)))
    env.close()


if __name__ == "__main__":
    main()
