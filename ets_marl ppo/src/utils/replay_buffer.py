"""
replay_buffer.py
================
Experience replay buffer for DDPG.
Stores (state, action, reward, next_state, done) tuples.
"""

import numpy as np
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-size circular replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    obs_dim : int
    action_dim : int
    seed : int, optional
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = None):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),          dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size

    @property
    def ready(self) -> bool:
        """True when buffer has enough samples to start training."""
        return self.size >= 64   # minimum batch size
