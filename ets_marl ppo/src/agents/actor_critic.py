"""
actor_critic.py
===============
PPO networks for two-phase ETS decisions:
  - AuctionPolicy:    obs_phase1(12) → 3 actions [bid_price, qty, delta_green]
  - SecondaryPolicy:  obs_phase2(15) → 2 actions [sec_price_mult, sec_qty]
  - ValueNetwork:     obs_phase2(15) → scalar V(s)

All policies output Gaussian distributions (mean + learnable log_std).
Actions are squashed through tanh and rescaled to physical bounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class AuctionPolicy(nn.Module):
    """Phase 1: obs(12) → Gaussian over 3 auction actions."""

    def __init__(self, obs_dim, action_dim, hidden_size, action_low, action_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self._init_weights()

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs, deterministic=False):
        """Returns (action_physical, raw_pretanh, log_prob)."""
        dist = self.forward(obs)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.rsample()

        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias

        log_prob = dist.log_prob(raw) - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, raw, log_prob

    def evaluate(self, obs, raw_actions):
        """Given stored raw actions, recompute log_prob and entropy."""
        dist = self.forward(obs)
        log_prob = dist.log_prob(raw_actions)
        squashed = torch.tanh(raw_actions)
        log_prob = log_prob - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)


class SecondaryPolicy(nn.Module):
    """Phase 2: obs_enriched(15) → Gaussian over 2 secondary actions."""

    def __init__(self, obs_dim, action_dim, hidden_size, action_low, action_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self._init_weights()

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.rsample()

        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias

        log_prob = dist.log_prob(raw) - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, raw, log_prob

    def evaluate(self, obs, raw_actions):
        dist = self.forward(obs)
        log_prob = dist.log_prob(raw_actions)
        squashed = torch.tanh(raw_actions)
        log_prob = log_prob - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)


class ValueNetwork(nn.Module):
    """V(s) for PPO. Takes enriched obs(15) = full state."""

    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self._init_weights()

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.zeros_(self.fc3.bias)
