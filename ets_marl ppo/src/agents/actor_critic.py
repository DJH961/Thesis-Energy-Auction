"""
actor_critic.py
===============
PPO networks for two-phase ETS decisions:
  - AuctionPolicy:    obs_phase1 → 6 actions [bid_price, qty, invest_frac, tech_logits×3]
  - SecondaryPolicy:  obs_phase2 → 2 actions [sec_price_mult, sec_qty]
  - ValueNetwork:     obs_phase2 → scalar V(s)

All policies output Gaussian distributions (mean + learnable log_std).
Actions are clipped to [-1, 1] and rescaled to physical bounds.
Clipped Gaussian replaces tanh squashing to avoid gradient saturation at boundaries.

AuctionPolicy uses conditioned action heads: qty_mean is computed from
[hidden, price_mean.detach()] to encode the price→quantity dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class AuctionPolicy(nn.Module):
    """
    Phase 1: obs → Gaussian over 6 auction actions.

    Conditioned heads architecture:
      price_head(hidden)                      → price_mean  (1D)
      qty_head(cat[hidden, price_mean.detach()]) → qty_mean (1D)
      rest_head(hidden)                       → [invest_frac, logit0, logit1, logit2] (4D)
    mean = cat[price_mean, qty_mean, rest_mean]  (6D)

    This encodes the structural dependency: bid quantity should adapt to bid price.
    The .detach() prevents gradients flowing back through price when training qty.
    """

    def __init__(self, obs_dim, action_dim, hidden_size, action_low, action_high,
                 log_std_min=-2.0, log_std_max=1.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Conditioned heads: price first, then qty conditioned on price
        self.price_head = nn.Linear(hidden_size, 1)
        self.qty_head = nn.Linear(hidden_size + 1, 1)   # +1 for price_mean signal
        # Remaining actions (invest_frac, 3 tech logits) stay independent
        self.rest_head = nn.Linear(hidden_size, action_dim - 2)

        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self._init_weights()

    def _compute_mean(self, hidden):
        """Compute full action mean with price→qty conditioning."""
        price_mean = self.price_head(hidden)                                    # (B, 1)
        qty_mean = self.qty_head(torch.cat([hidden, price_mean.detach()], dim=-1))  # (B, 1)
        rest_mean = self.rest_head(hidden)                                      # (B, 4)
        return torch.cat([price_mean, qty_mean, rest_mean], dim=-1)             # (B, 6)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self._compute_mean(x)
        log_std_clamped = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std_clamped.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs, deterministic=False):
        """Returns (action_physical, raw_pretanh, log_prob)."""
        dist = self.forward(obs)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.rsample()

        # Clipped Gaussian: clamp to [-1, 1] instead of tanh.
        # Gradient = 1.0 inside bounds (vs tanh gradient → 0 at boundaries).
        squashed = torch.clamp(raw, -1.0, 1.0)
        action = squashed * self.action_scale + self.action_bias

        # No tanh log-prob correction needed: PPO importance ratio is correct
        # because both old and new log_probs use the same uncorrected formula.
        log_prob = dist.log_prob(raw).sum(dim=-1, keepdim=True)

        return action, raw, log_prob

    def evaluate(self, obs, raw_actions):
        """Given stored raw actions, recompute log_prob and entropy."""
        dist = self.forward(obs)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.zeros_(layer.bias)
        for head in [self.price_head, self.qty_head, self.rest_head]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)


class SecondaryPolicy(nn.Module):
    """Phase 2: obs_enriched(21) → Gaussian over 2 secondary actions."""

    def __init__(self, obs_dim, action_dim, hidden_size, action_low, action_high,
                 log_std_min=-2.0, log_std_max=1.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self._init_weights()

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std_clamped = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std_clamped.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.rsample()

        # Clipped Gaussian: clamp instead of tanh (see AuctionPolicy.act).
        squashed = torch.clamp(raw, -1.0, 1.0)
        action = squashed * self.action_scale + self.action_bias

        log_prob = dist.log_prob(raw).sum(dim=-1, keepdim=True)

        return action, raw, log_prob

    def evaluate(self, obs, raw_actions):
        dist = self.forward(obs)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)


class ValueNetwork(nn.Module):
    """V(s) for PPO. Takes enriched obs(21) = full state."""

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
