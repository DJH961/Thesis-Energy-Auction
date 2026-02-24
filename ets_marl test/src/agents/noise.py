"""
noise.py
========
Ornstein-Uhlenbeck exploration noise for DDPG.

Reference:
    Di Persio, Garbelli, Giordano (2024)
    Lillicrap et al. (2015) — Continuous control with deep RL
"""

import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    dX_t = theta * (mu - X_t) dt + sigma * dW_t

    Parameters
    ----------
    action_dim : int
    theta : float
        Mean reversion rate (default 0.15).
    mu : float
        Long-run mean (default 0.0).
    sigma : float
        Noise scale (default 0.20).
    seed : int, optional
    """

    def __init__(
        self,
        action_dim: int,
        theta: float = 0.15,
        mu: float = 0.0,
        sigma: float = 0.20,
        seed: int = None,
    ):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """Return one noise sample (Euler-Maruyama discretisation)."""
        dx = (
            self.theta * (self.mu - self.state)
            + self.sigma * self.rng.standard_normal(self.action_dim)
        )
        self.state = self.state + dx
        return self.state.copy()
