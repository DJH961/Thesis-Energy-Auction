"""
bot_data_buffer.py
==================
Ring buffer for collecting episode-level transition data used to retrain
LSTM bot policies.  Stores complete episode sequences (all 12 years) so the
LSTM can be trained on temporally ordered data.

Each entry represents one full episode and contains, for every year-step,
the observations and actions of ALL participants (learning + bots).  The
LSTM bot retraining selects target actions based on a configurable strategy
(``imitate_best`` by default).
"""

import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple


class BotDataBuffer:
    """Fixed-capacity ring buffer storing complete episode trajectories.

    Parameters
    ----------
    capacity : int
        Maximum number of episodes to retain.  Oldest episodes are evicted
        first when the buffer is full.
    n_total : int
        Total number of participants (learning + bots).
    n_years : int
        Number of year-steps per episode.
    obs_dim_phase1 : int
        Observation dimension for Phase 1.
    obs_dim_phase2 : int
        Observation dimension for Phase 2.
    """

    def __init__(
        self,
        capacity: int,
        n_total: int,
        n_years: int,
        obs_dim_phase1: int,
        obs_dim_phase2: int,
    ):
        self.capacity = capacity
        self.n_total = n_total
        self.n_years = n_years
        self.obs_dim_phase1 = obs_dim_phase1
        self.obs_dim_phase2 = obs_dim_phase2

        # Each element: dict with keys
        #   obs1   : (n_years, n_total, obs_dim_phase1)
        #   obs2   : (n_years, n_total, obs_dim_phase2)
        #   auc    : (n_years, n_total, 6)
        #   sec    : (n_years, n_total, 2)
        #   rewards: (n_total,)  — total episode reward per agent
        self._buffer: deque = deque(maxlen=capacity)

        # Temporary accumulator for the current episode being collected
        self._current_obs1: List[np.ndarray] = []
        self._current_obs2: List[np.ndarray] = []
        self._current_auc: List[np.ndarray] = []
        self._current_sec: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Collection API (called year-by-year during an episode)
    # ------------------------------------------------------------------
    def begin_episode(self):
        """Start accumulating a new episode."""
        self._current_obs1.clear()
        self._current_obs2.clear()
        self._current_auc.clear()
        self._current_sec.clear()

    def push_year(
        self,
        obs_phase1_all: np.ndarray,   # (n_total, obs_dim_phase1)
        obs_phase2_all: np.ndarray,   # (n_total, obs_dim_phase2)
        auction_actions_all: np.ndarray,  # (n_total, 6)
        secondary_actions_all: np.ndarray,  # (n_total, 2)
    ):
        """Record one year-step of data for all participants."""
        self._current_obs1.append(obs_phase1_all.copy())
        self._current_obs2.append(obs_phase2_all.copy())
        self._current_auc.append(auction_actions_all.copy())
        self._current_sec.append(secondary_actions_all.copy())

    def end_episode(self, rewards: np.ndarray):
        """Finalise the current episode and push it to the ring buffer.

        Parameters
        ----------
        rewards : (n_total,)
            Total episode reward for each participant.
        """
        if len(self._current_obs1) == 0:
            return  # nothing collected
        entry = {
            "obs1": np.stack(self._current_obs1),   # (T, N, D1)
            "obs2": np.stack(self._current_obs2),   # (T, N, D2)
            "auc": np.stack(self._current_auc),      # (T, N, 6)
            "sec": np.stack(self._current_sec),      # (T, N, 2)
            "rewards": rewards.copy(),               # (N,)
        }
        self._buffer.append(entry)
        self._current_obs1.clear()
        self._current_obs2.clear()
        self._current_auc.clear()
        self._current_sec.clear()

    # ------------------------------------------------------------------
    # Sampling API (called at retraining time)
    # ------------------------------------------------------------------
    def sample_episodes(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Dict]:
        """Sample *n* complete episodes (with replacement if n > len)."""
        if len(self._buffer) == 0:
            return []
        rng = rng or np.random.default_rng()
        indices = rng.integers(0, len(self._buffer), size=n)
        return [self._buffer[i] for i in indices]

    def get_top_k_targets(
        self,
        episode: Dict,
        n_bots: int,
        top_k_fraction: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select target actions from the best-performing agents.

        For each bot archetype, pick the best-performing agent with the
        same archetype parity (even=financial, odd=green).

        Returns
        -------
        auc_targets : (n_years, n_bots, 6)
        sec_targets : (n_years, n_bots, 2)
        """
        rewards = episode["rewards"]           # (n_total,)
        n_total = len(rewards)
        n_learning = n_total - n_bots

        auc_targets = np.zeros_like(episode["auc"][:, n_learning:, :])  # (T, n_bots, 6)
        sec_targets = np.zeros_like(episode["sec"][:, n_learning:, :])  # (T, n_bots, 2)

        # Separate even-indexed (financial) and odd-indexed (green) learning agents
        even_agents = [i for i in range(n_learning) if i % 2 == 0]
        odd_agents = [i for i in range(n_learning) if i % 2 == 1]

        # Rank by reward
        even_ranked = sorted(even_agents, key=lambda i: rewards[i], reverse=True)
        odd_ranked = sorted(odd_agents, key=lambda i: rewards[i], reverse=True)

        # Top-K from each group
        k_even = max(1, int(len(even_ranked) * top_k_fraction))
        k_odd = max(1, int(len(odd_ranked) * top_k_fraction))
        best_even = even_ranked[:k_even]
        best_odd = odd_ranked[:k_odd]

        rng = np.random.default_rng()
        for b in range(n_bots):
            bot_global_id = n_learning + b
            # Match archetype parity
            if bot_global_id % 2 == 0:
                source = rng.choice(best_even)
            else:
                source = rng.choice(best_odd)
            auc_targets[:, b, :] = episode["auc"][:, source, :]
            sec_targets[:, b, :] = episode["sec"][:, source, :]

        return auc_targets, sec_targets

    def __len__(self):
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()
