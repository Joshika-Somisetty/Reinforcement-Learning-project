"""
PPO (Proximal Policy Optimization) agent for irrigation scheduling.

On-policy baseline algorithm for comparison with off-policy SAC.
Uses a flat MLP (no BiLSTM) matching the paper's comparison setup.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class RolloutBuffer:
    """Stores on-policy rollout data for PPO updates."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def push(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95):
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = 0.0
        for t in reversed(range(n)):
            delta = (
                self.rewards[t]
                + gamma * next_value * (1.0 - self.dones[t])
                - self.values[t]
            )
            gae = delta + gamma * lam * (1.0 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        return returns, advantages

    def get_tensors(self, device):
        return (
            torch.FloatTensor(np.array(self.states)).to(device),
            torch.FloatTensor(np.array(self.actions)).to(device),
            torch.FloatTensor(np.array(self.log_probs)).to(device),
        )

    def __len__(self):
        return len(self.rewards)


class PPOActorCritic(nn.Module):
    """Shared-backbone actor-critic for continuous action space."""

    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256),
                 action_scale: float = 30.0, action_bias: float = 30.0):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, obs):
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(-5, 2)
        value = self.value_head(features)
        return mean, log_std, value

    def get_action_and_value(self, obs, action=None):
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        if action is None:
            z = dist.rsample()
        else:
            # Reverse the tanh squashing to recover z from action
            y = (action - self.action_bias) / self.action_scale
            y = y.clamp(-0.999, 0.999)
            z = torch.atanh(y)

        y = torch.tanh(z)
        action_out = y * self.action_scale + self.action_bias

        # Log prob with tanh correction
        log_prob = dist.log_prob(z) - torch.log(
            self.action_scale * (1 - y.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return action_out, log_prob, value.squeeze(-1), entropy


class PPOAgent:
    """
    PPO agent with clipped surrogate objective.

    Uses flat MLP (no sequence encoder) for fair comparison with
    paper baselines. Compatible with evaluate_policy() via select_action().
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        mini_batch_size: int = 64,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.device = torch.device(device)

        self.network = PPOActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def select_action(self, obs, deterministic: bool = False):
        """Compatible interface for evaluate_policy()."""
        if isinstance(obs, np.ndarray):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        else:
            obs_t = obs.unsqueeze(0).to(self.device)

        action, log_prob, value, _ = self.network.get_action_and_value(obs_t)

        if deterministic:
            mean, _, _ = self.network(obs_t)
            action = torch.tanh(mean) * self.network.action_scale + self.network.action_bias

        return action.cpu().numpy().flatten()

    @torch.no_grad()
    def get_action_and_value(self, obs):
        """For training: returns action, log_prob, value."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, value, _ = self.network.get_action_and_value(obs_t)
        return (
            action.cpu().numpy().flatten(),
            log_prob.cpu().item(),
            value.cpu().item(),
        )

    def store(self, state, action, log_prob, reward, done, value):
        self.buffer.push(state, action, log_prob, reward, done, value)

    def update(self):
        """Run PPO update over collected rollout buffer."""
        if len(self.buffer) == 0:
            return {}

        returns, advantages = self.buffer.compute_returns(
            self.gamma, self.gae_lambda
        )

        states, actions, old_log_probs = self.buffer.get_tensors(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Normalise advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        n = len(self.buffer)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_lp = old_log_probs[idx]
                mb_returns = returns_t[idx]
                mb_adv = advantages_t[idx]

                _, new_log_prob, new_value, entropy = (
                    self.network.get_action_and_value(mb_states, mb_actions)
                )

                # Clipped surrogate
                ratio = (new_log_prob - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_value, mb_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
