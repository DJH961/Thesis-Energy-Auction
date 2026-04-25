"""
noise.py
========
Ornstein-Uhlenbeck exploration noise for DDPG.

Supports per-dimension sigma (proportional to action range) and
epsilon decay to reduce exploration over training.

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
    sigma : float or np.ndarray
        Noise scale. If float, used for all dims. If array, per-dimension.
    seed : int, optional
    sigma_scale : np.ndarray, optional
        Per-dimension scale factors (e.g. action_range / max_range).
        Multiplied with sigma to get effective noise per dimension.
    """

    def __init__(
        self,
        action_dim: int,
        theta: float = 0.15,
        mu: float = 0.0,
        sigma: float = 0.20,
        seed: int = None,
        sigma_scale: np.ndarray = None,
    ):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.rng = np.random.default_rng(seed)
        self.state = np.ones(action_dim) * mu

        # Per-dimension sigma: sigma_base * scale_per_dim
        if sigma_scale is not None:
            self.sigma = sigma * np.asarray(sigma_scale, dtype=np.float64)
        else:
            self.sigma = np.full(action_dim, sigma, dtype=np.float64)

        # Epsilon for noise decay (starts at 1.0, caller can decay it)
        self.epsilon = 1.0

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """Return one noise sample (Euler-Maruyama discretisation), scaled by epsilon."""
        dx = (
            self.theta * (self.mu - self.state)
            + self.sigma * self.rng.standard_normal(self.action_dim)
        )
        self.state = self.state + dx
        return (self.state * self.epsilon).copy()

    def set_epsilon(self, epsilon: float):
        """Set the exploration decay factor (1.0 = full noise, 0.0 = no noise)."""
        self.epsilon = max(0.0, min(1.0, epsilon))
