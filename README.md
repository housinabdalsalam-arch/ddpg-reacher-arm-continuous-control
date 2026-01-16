# double-jointed arm Continuous Control (Reacher) — DDPG 

This repository contains a **classic DDPG (Actor–Critic)** solution for Udacity DRLND Project 2 (Continuous Control / Reacher).

## Demo

<p align="center">
  <img src="media/reacher.gif" alt="Reacher Demo" width="700">
</p>


## Environment
- **State space**: 33 continuous values (positions, rotations, velocities, angular velocities).
- **Action space**: 4 continuous actions (torques), each in **[-1, 1]**.
- **Reward**: +0.1 for every step the hand is in the target location.
- **Solved criterion**:
  - Version 1 (1 agent): average score >= **+30** over 100 consecutive episodes
  - Version 2 (20 agents): average of the 20 agent scores per episode, then average over 100 episodes >= **+30**

## Files
- `model.py` — Actor/Critic networks
- `ddpg_agent.py` - DDPG agent (replay buffer, target networks)
- `train.py` - training script (saves weights + plot)
- `eval.py` - evaluate a saved actor
- `checkpoint_actor.pth`, `checkpoint_critic.pth` — trained weights
- `scores.png` - training curve
- `Report.md` - project report

## Setup

### 1) Download the Unity environment

#### Version 2: 1 Agent
Download and unzip the environment into the **repository root**.

- **Windows (64-bit):** [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
- **Linux:** [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- **Mac:** [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)



#### Version 2: 20 Agents
Download and unzip the environment into the **repository root**.

- **Windows (64-bit):** [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
- **Linux:** [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- **Mac:** [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)




## Training
From the repository root:
```bash
python train.py --env ./Reacher_Windows_x86_64/Reacher.exe
```
Training stops once solved and writes:
- `checkpoint_actor.pth`
- `checkpoint_critic.pth`
- `scores.png`

## Evaluation (10 episodes)
```bash
python eval.py --env ./Reacher_Windows_x86_64/Reacher.exe --actor checkpoint_actor.pth --episodes 10
```

## Results

- Solved in **226episodes**
- Average score (last 100 episodes): **30.03**

![DDPG Training Scores](scores.png)

## Notes
- You can adjust hyperparameters in `DDPGConfig` inside `ddpg_agent.py`.
