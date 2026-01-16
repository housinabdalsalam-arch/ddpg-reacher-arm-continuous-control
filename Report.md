# Project 2 — Continuous Control (Reacher)

This repo contains my solution for Udacity’s Deep Reinforcement Learning Nanodegree **Project 2: Continuous Control**.

## Environment

The agent controls a double-jointed robotic arm and must keep its hand on the target:

- **Reward:** +0.1 for each step the hand stays in the goal region  
- **State space:** 33 continuous values (position, rotation, velocity, angular velocity)  
- **Action space:** 4 continuous actions (joint torques), each in **[-1, 1]**  
- **Solved when (20 agents):** average score **>= +30** over **100** consecutive episodes  
  (score per episode = average of the 20 agent scores)

## Approach — Deep Deterministic Policy Gradient (DDPG)

I implemented a classic **DDPG** agent in PyTorch (actor–critic) with the common stability components:

- **Actor network** outputs deterministic continuous actions: `mu(s)`
- **Critic network** estimates action-value: `Q(s, a)`
- **Replay buffer** for off-policy learning with decorrelated mini-batches
- **Target networks** for stable bootstrapped targets
- **OU noise** added to actions during training for exploration

### Update rule (DDPG)

For each transition `(state, action, reward, next_state, done)` sampled from the replay buffer:

**Critic target**
- If the episode ended (`done = True`):
  - `target = reward`
- Otherwise, bootstrap from the target networks:
  - `target = reward + gamma * Q_target(next_state, mu_target(next_state))`

The critic minimizes the mean-squared TD error:
- `loss_critic = MSE(Q_local(state, action), target)`

**Actor update**
The actor is trained to maximize the critic’s value of its actions:
- `loss_actor = - mean( Q_local(state, mu_local(state)) )`

After each learning step, both target networks are softly updated:
- `theta_target = tau * theta_local + (1 - tau) * theta_target`

## Hyperparameters

From `DDPGConfig` in `src/ddpg_agent.py`:

- **BUFFER_SIZE:** 1e6  
- **BATCH_SIZE:** 256  
- **GAMMA:** 0.99  
- **TAU:** 1e-3  
- **LR_ACTOR:** 1e-4  
- **LR_CRITIC:** 1e-3  
- **WEIGHT_DECAY:** 0.0  
- **LEARN_EVERY:** 20 steps  
- **LEARN_UPDATES:** 10 updates per learn step  
- **OU noise:** theta = 0.15, sigma = 0.20  
- **Critic grad clip:** 1.0  

## Neural networks

Fully-connected MLPs (classic Udacity-style):

**Actor**
- Input: 33  
- Hidden: 400 (ReLU)  
- Hidden: 300 (ReLU)  
- Output: 4 (Tanh)

Architecture: **33 → 400 → 300 → 4**

**Critic**
- State path: 33 → 400 (ReLU)  
- Concatenate action (4)  
- 404 → 300 (ReLU) → 1

Architecture: **(33 → 400) + 4 → 300 → 1**

## Results

The environment is solved when the average score over 100 consecutive episodes is **>= +30**.

My agent solved the environment in **XXX episodes**, reaching an average score of **YY.YY** over the last 100 episodes.

![Training curve](scores.png)

Evaluation using the saved actor checkpoint (`checkpoint_actor.pth`) achieved a mean score of **ZZ.ZZ** over 10 episodes.

## Future Work

Since this is a basic DDPG solution, there are a few well-known upgrades that could be tried next:

- **SAC (Soft Actor-Critic):** SAC is usually more stable for continuous control tasks because it uses a stochastic policy and an entropy term that encourages exploration.

- **Prioritized Experience Replay:** Right now the replay buffer samples uniformly. Prioritized replay focuses learning on more informative transitions (large TD error), which can speed up learning.

- **Hyperparameter tuning:** Update frequency (`LEARN_EVERY`, `LEARN_UPDATES`), batch size, and network sizes can be tuned for faster or more stable convergence.
