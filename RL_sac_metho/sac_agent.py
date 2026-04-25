"""
TSA-SAC style agent for irrigation scheduling.

This implementation moves the project closer to the 2025 paper by adding:
  - a 2-layer BiLSTM sequence encoder
  - temporal attention over the encoded sequence
  - feature attention before the actor / critic decision heads
  - automatic entropy tuning and twin critics from SAC
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


# ──────────────────────────────────────────────────────────────────────────────
# Sequence Replay Buffer
# Stores (seq, action, reward, next_seq, done) where seq is (seq_len, obs_dim)
# ──────────────────────────────────────────────────────────────────────────────
class SequenceReplayBuffer:
    """
    Replay buffer that stores fixed-length observation windows.
    Each transition carries the full sequence context so the BiLSTM
    can be trained fully off-policy without hidden-state bookkeeping.
    """
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, seq, action, reward, next_seq, done):
        """seq / next_seq : np.ndarray of shape (seq_len, obs_dim)"""
        self.buffer.append((
            np.array(seq,      dtype=np.float32),
            np.array(action,   dtype=np.float32),
            float(reward),
            np.array(next_seq, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        seqs, actions, rewards, next_seqs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(seqs)),                        # (B, T, obs_dim)
            torch.FloatTensor(np.array(actions)),                     # (B, action_dim)
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),        # (B, 1)
            torch.FloatTensor(np.array(next_seqs)),                   # (B, T, obs_dim)
            torch.FloatTensor(np.array(dones)).unsqueeze(1),          # (B, 1)
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal-spatial encoder
# ──────────────────────────────────────────────────────────────────────────────
class TemporalSpatialEncoder(nn.Module):
    """
    BiLSTM encoder with temporal attention followed by feature attention.

    seq -> BiLSTM hidden sequence -> temporal attention weighted context
        -> feature attention -> attended context
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.feature_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, obs_dim) -> attended context: (B, hidden_dim)"""
        hidden_seq, _ = self.bilstm(seq)                     # (B, T, 2H)
        attn_scores = self.temporal_attn(hidden_seq)        # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (hidden_seq * attn_weights).sum(dim=1)    # (B, 2H)
        context = self.context_proj(context)                # (B, H)
        feature_weights = torch.softmax(self.feature_attn(context), dim=-1)
        return context * feature_weights                    # (B, H)


# ──────────────────────────────────────────────────────────────────────────────
# MLP head
# ──────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


LOG_STD_MIN, LOG_STD_MAX = -5, 2


# ──────────────────────────────────────────────────────────────────────────────
# Actor: BiLSTM → MLP → Squashed Gaussian
# ──────────────────────────────────────────────────────────────────────────────
class BiLSTMGaussianPolicy(nn.Module):
    """
    Stochastic actor that processes an observation sequence through BiLSTM
    before mapping to a squashed Gaussian action distribution.

    Flow:
      seq (B,T,obs) → BiLSTM → context (B,H) → MLP → (mean, log_std)
                                                              ↓
                                                    tanh squash → [0, 50 mm]
    """
    def __init__(self, obs_dim: int, action_dim: int,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 mlp_hidden=(256, 256),
                 action_scale: float = 30.0, action_bias: float = 30.0):
        super().__init__()
        self.encoder       = TemporalSpatialEncoder(obs_dim, lstm_hidden, lstm_layers)
        self.mlp           = MLP(lstm_hidden, mlp_hidden[-1], mlp_hidden[:-1])
        self.mean_layer    = nn.Linear(mlp_hidden[-1], action_dim)
        self.log_std_layer = nn.Linear(mlp_hidden[-1], action_dim)

        self.action_scale = torch.FloatTensor([action_scale])
        self.action_bias  = torch.FloatTensor([action_bias])

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias  = self.action_bias.to(device)
        return super().to(device)

    def _encode(self, seq):
        ctx = self.encoder(seq)           # (B, lstm_hidden)
        return self.mlp(ctx)              # (B, mlp_hidden[-1])

    def forward(self, seq):
        h       = self._encode(seq)
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, seq):
        mean, log_std = self.forward(seq)
        std  = log_std.exp()
        eps  = torch.randn_like(mean)
        z    = mean + std * eps            # reparameterisation trick
        y    = torch.tanh(z)
        action = y * self.action_scale + self.action_bias

        # Log-prob with tanh change-of-variables correction
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(z)
            - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        ).sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


# ──────────────────────────────────────────────────────────────────────────────
# Critic: BiLSTM → MLP → Q-value  (Twin networks)
# ──────────────────────────────────────────────────────────────────────────────
class BiLSTMQNetwork(nn.Module):
    """
    Twin Q-networks, each with their own independent BiLSTM encoder.
    Separate encoders prevent gradient interference between Q1 and Q2.
    """
    def __init__(self, obs_dim: int, action_dim: int,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 mlp_hidden=(256, 256)):
        super().__init__()
        self.enc1 = TemporalSpatialEncoder(obs_dim, lstm_hidden, lstm_layers)
        self.q1   = MLP(lstm_hidden + action_dim, 1, mlp_hidden)
        self.enc2 = TemporalSpatialEncoder(obs_dim, lstm_hidden, lstm_layers)
        self.q2   = MLP(lstm_hidden + action_dim, 1, mlp_hidden)

    def forward(self, seq: torch.Tensor, action: torch.Tensor):
        """
        seq    : (B, T, obs_dim)
        action : (B, action_dim)
        """
        c1  = self.enc1(seq)
        c2  = self.enc2(seq)
        sa1 = torch.cat([c1, action], dim=-1)
        sa2 = torch.cat([c2, action], dim=-1)
        return self.q1(sa1), self.q2(sa2)


# ──────────────────────────────────────────────────────────────────────────────
# SAC Agent  (BiLSTM version — drop-in compatible with train.py)
# ──────────────────────────────────────────────────────────────────────────────
class SACAgent:
    """
    SAC with BiLSTM temporal encoder.

    Key difference from flat-MLP SAC:
      - select_action() takes a (seq_len, obs_dim) array instead of (obs_dim,)
      - remember()      stores   (seq, action, reward, next_seq, done)
      - The rolling window is managed externally in train.py via build_seq()
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        seq_len: int = 7,           # days of history fed to BiLSTM
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        mlp_hidden: tuple = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.seq_len    = seq_len
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.device     = torch.device(device)
        self.auto_alpha = auto_alpha

        # ── Networks ──────────────────────────────────────────────────────────
        self.actor = BiLSTMGaussianPolicy(
            obs_dim, action_dim, lstm_hidden, lstm_layers, mlp_hidden
        ).to(self.device)

        self.critic = BiLSTMQNetwork(
            obs_dim, action_dim, lstm_hidden, lstm_layers, mlp_hidden
        ).to(self.device)

        self.critic_target = BiLSTMQNetwork(
            obs_dim, action_dim, lstm_hidden, lstm_layers, mlp_hidden
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optimisers ────────────────────────────────────────────────────────
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # ── Entropy temperature ───────────────────────────────────────────────
        self.target_entropy = -float(action_dim)
        if auto_alpha:
            self.log_alpha   = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha       = self.log_alpha.exp().item()
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        self.buffer        = SequenceReplayBuffer(buffer_size)
        self._update_count = 0

    # ── Rolling window helper ─────────────────────────────────────────────────
    def build_seq(self, window: deque) -> np.ndarray:
        """
        Convert a rolling deque of observations to a zero-padded
        (seq_len, obs_dim) array.  Called from train.py each step.
        """
        seq = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        entries = list(window)
        n = min(len(entries), self.seq_len)
        seq[self.seq_len - n:] = entries[-n:]
        return seq

    # ── Action selection ──────────────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, seq: np.ndarray, deterministic: bool = False):
        """seq : (seq_len, obs_dim)"""
        s = torch.FloatTensor(seq).unsqueeze(0).to(self.device)  # (1, T, obs)
        if deterministic:
            _, _, action = self.actor.sample(s)
        else:
            action, _, _ = self.actor.sample(s)
        return action.cpu().numpy().flatten()

    def remember(self, seq, action, reward, next_seq, done):
        self.buffer.push(seq, action, reward, next_seq, done)

    # ── Network update ────────────────────────────────────────────────────────
    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        seqs, actions, rewards, next_seqs, dones = [
            x.to(self.device) for x in self.buffer.sample(self.batch_size)
        ]

        # Critic
        with torch.no_grad():
            next_a, log_pi, _ = self.actor.sample(next_seqs)
            q1_t, q2_t        = self.critic_target(next_seqs, next_a)
            q_target = rewards + self.gamma * (1 - dones) * (
                torch.min(q1_t, q2_t) - self.alpha * log_pi
            )

        q1, q2      = self.critic(seqs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # Actor
        new_a, log_pi, _ = self.actor.sample(seqs)
        q1_new, q2_new   = self.critic(seqs, new_a)
        actor_loss = (self.alpha * log_pi - torch.min(q1_new, q2_new)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # Alpha
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft target update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self._update_count += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "alpha_loss":  alpha_loss.item(),
        }

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "actor":     self.actor.state_dict(),
            "critic":    self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu() if self.auto_alpha else None,
            "seq_len":   self.seq_len,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic"])
        if self.auto_alpha and ckpt.get("log_alpha") is not None:
            self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.actor_optim.param_groups[0]["lr"])
