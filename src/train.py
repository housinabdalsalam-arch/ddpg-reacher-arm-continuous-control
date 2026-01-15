import argparse
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from ddpg_agent import DDPGAgent, DDPGConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, type=str)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max_t", type=int, default=1000)
    p.add_argument("--solve_score", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--worker_id", type=int, default=0)
    args = p.parse_args()

    env = UnityEnvironment(file_name=args.env, seed=args.seed, worker_id=args.worker_id)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    print(f"[env] agents={num_agents}, state_size={state_size}, action_size={action_size}")

    agent = DDPGAgent(state_size, action_size, seed=args.seed, cfg=DDPGConfig())

    scores = []
    window = deque(maxlen=100)
    solved_episode = None

    for i_episode in range(1, args.episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        ep_scores = np.zeros(num_agents, dtype=np.float32)

        for _ in range(args.max_t):
            actions = agent.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = np.array(env_info.rewards, dtype=np.float32)
            dones = np.array(env_info.local_done, dtype=np.bool_)
            agent.step(states, actions, rewards, next_states, dones)
            ep_scores += rewards
            states = next_states
            if np.any(dones):
                break

        score = float(np.mean(ep_scores))
        scores.append(score)
        window.append(score)

        if i_episode % 10 == 0:
            print(f"Episode {i_episode}\tScore: {score:.2f}\tAverage(100): {np.mean(window):.2f}")

        if solved_episode is None and len(window) == 100 and np.mean(window) >= args.solve_score:
            solved_episode = i_episode
            print(f"\n[SOLVED] in {i_episode} episodes! Average(100)={np.mean(window):.2f}\n")
            agent.save("checkpoint_actor.pth", "checkpoint_critic.pth")
            break

    np.save("scores.npy", np.array(scores, dtype=np.float32))

    plt.figure()
    plt.plot(np.arange(1, len(scores) + 1), scores, label="Score")
    plt.axhline(args.solve_score, linestyle="--", color="red", label=f"Solved ({args.solve_score})")
    plt.xlabel("Episode")
    plt.ylabel("Score (mean over agents)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("scores.png", dpi=150)
    plt.show()

    env.close()

    if solved_episode is None:
        print("Did not solve within episode limit. Check scores.png and consider more episodes.")
    else:
        print(f"Solved at episode: {solved_episode}")


if __name__ == "__main__":
    main()
