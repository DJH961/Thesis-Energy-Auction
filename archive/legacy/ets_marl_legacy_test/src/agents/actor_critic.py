"""
actor_critic.py
===============
Actor and Critic neural networks for DDPG.

Architecture: Feed-Forward Neural Network (FFNN) with two hidden layers.

Reference:
    Di Persio, Garbelli, Giordano (2024) — Fig. 7 & Fig. 8
    ckrk/bidding_learning — actor_critic.py (adapted)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor network: maps observation → deterministic action.

    Output is passed through tanh and then re-scaled to the action bounds.

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    hidden_size : int
    action_low : torch.Tensor
    action_high : torch.Tensor
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

        # Store action bounds for re-scaling
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer(
            "action_scale", (action_high - action_low) / 2.0
        )
        self.register_buffer(
            "action_bias", (action_high + action_low) / 2.0
        )

        self._init_weights()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))                      # [-1, 1]
        action = x * self.action_scale + self.action_bias  # re-scale to [low, high]
        return action

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)


class Critic(nn.Module):
    """
    Critic network: maps (observation, action) → Q-value scalar.

    Implements the Bellman equation approximation.

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    hidden_size : int
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        # Process observation
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        # Merge with action
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self._init_weights()

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)
