import random
from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_

from model import Actor, Critic


@dataclass
class DDPGConfig:
    buffer_size: int = int(1e6)
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 1e-3
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    weight_decay: float = 0.0
    learn_every: int = 20
    learn_updates: int = 10
    ou_theta: float = 0.15
    ou_sigma: float = 0.20
    critic_grad_clip: float = 1.0


class OUNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.randn(len(self.state)).astype(np.float32)
        self.state = self.state + dx
        return self.state


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.device = device
        self.exp = namedtuple("Experience", ["s", "a", "r", "ns", "d"])

    def add(self, s, a, r, ns, d):
        self.memory.append(self.exp(s, a, float(r), ns, bool(d)))

    def sample(self):
        exps = self.rng.sample(self.memory, k=self.batch_size)
        s = torch.from_numpy(np.vstack([e.s for e in exps]).astype(np.float32)).to(self.device)
        a = torch.from_numpy(np.vstack([e.a for e in exps]).astype(np.float32)).to(self.device)
        r = torch.from_numpy(np.vstack([e.r for e in exps]).astype(np.float32)).to(self.device)
        ns = torch.from_numpy(np.vstack([e.ns for e in exps]).astype(np.float32)).to(self.device)
        d = torch.from_numpy(np.vstack([e.d for e in exps]).astype(np.uint8)).float().to(self.device)
        return s, a, r, ns, d

    def __len__(self):
        return len(self.memory)


class DDPGAgent:
    def __init__(self, state_size, action_size, seed=0, cfg=None, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.cfg = cfg or DDPGConfig()
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.actor_local = Actor(state_size, action_size, seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, seed).to(self.device)
        self.critic_local = Critic(state_size, action_size, seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, seed).to(self.device)

        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=self.cfg.lr_actor)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=self.cfg.lr_critic, weight_decay=self.cfg.weight_decay)

        self._hard_update(self.actor_target, self.actor_local)
        self._hard_update(self.critic_target, self.critic_local)

        self.noise = OUNoise(action_size, seed, theta=self.cfg.ou_theta, sigma=self.cfg.ou_sigma)
        self.replay = ReplayBuffer(self.cfg.buffer_size, self.cfg.batch_size, seed, self.device)
        self.t = 0

    def reset(self):
        self.noise.reset()

    def act(self, states, add_noise=True):
        s = torch.from_numpy(states.astype(np.float32)).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(s).cpu().numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1.0, 1.0)

    def step(self, states, actions, rewards, next_states, dones):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.replay.add(s, a, r, ns, d)

        self.t = (self.t + 1) % self.cfg.learn_every
        if self.t == 0 and len(self.replay) >= self.cfg.batch_size:
            for _ in range(self.cfg.learn_updates):
                self.learn(self.replay.sample())

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.cfg.gamma * q_next * (1 - dones)

        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_local.parameters(), self.cfg.critic_grad_clip)
        self.critic_opt.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.critic_local, self.critic_target, self.cfg.tau)
        self.soft_update(self.actor_local, self.actor_target, self.cfg.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for tp, lp in zip(target_model.parameters(), local_model.parameters()):
            tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)

    @staticmethod
    def _hard_update(target_model, local_model):
        for tp, lp in zip(target_model.parameters(), local_model.parameters()):
            tp.data.copy_(lp.data)

    def save(self, actor_path="checkpoint_actor.pth", critic_path="checkpoint_critic.pth"):
        torch.save(self.actor_local.state_dict(), actor_path)
        torch.save(self.critic_local.state_dict(), critic_path)

    def load_actor(self, actor_path):
        self.actor_local.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor_local.eval()
