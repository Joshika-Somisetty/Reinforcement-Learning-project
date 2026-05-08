"""
DDPG (Deep Deterministic Policy Gradient) agent for irrigation scheduling.

Off-policy deterministic baseline for comparison with SAC.
Uses flat MLP (no BiLSTM) matching the paper's comparison setup.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import copy


class ReplayBuffer:
    """Simple flat replay buffer for DDPG."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.full(self.size, self.mu, dtype=np.float32)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()


class DeterministicActor(nn.Module):
    """Deterministic policy: obs → action (scaled via tanh)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256),
                 action_scale: float = 30.0, action_bias: float = 30.0):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, obs):
        return self.net(obs) * self.action_scale + self.action_bias


class QCritic(nn.Module):
    """Q-network: (obs, action) → Q-value."""

    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim + action_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class DDPGAgent:
    """
    DDPG agent with Ornstein-Uhlenbeck exploration noise.

    Compatible with evaluate_policy() via select_action(obs).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        noise_sigma: float = 0.2,
        noise_decay: float = 0.9995,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Networks
        self.actor = DeterministicActor(obs_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = QCritic(obs_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(action_dim, sigma=noise_sigma)
        self.noise_scale = 1.0
        self.noise_decay = noise_decay
        self._update_count = 0

    @torch.no_grad()
    def select_action(self, obs, deterministic: bool = False):
        if isinstance(obs, np.ndarray):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        else:
            obs_t = obs.unsqueeze(0).to(self.device)

        action = self.actor(obs_t).cpu().numpy().flatten()

        if not deterministic:
            noise = self.noise.sample() * self.noise_scale * 15.0
            action = action + noise
            self.noise_scale *= self.noise_decay

        return np.clip(action, 0.0, 60.0).astype(np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def reset_noise(self):
        self.noise.reset()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = [
            x.to(self.device) for x in self.buffer.sample(self.batch_size)
        ]

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target = rewards + self.gamma * (1 - dones) * self.critic_target(
                next_states, next_actions
            )

        q_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q_current, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # Soft target update
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self._update_count += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
