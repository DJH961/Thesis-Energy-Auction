"""
q_learning_agent.py
===================
Tabular Q-learning baseline agent for the ETS MARL environment.

Uses state discretization (5 features × 3 bins = 243 states) and
predefined action profiles (6 Phase-1 × 4 Phase-2) to reduce the
continuous environment to a tractable tabular Q-learning problem.

This serves as a **baseline** for comparing against the PPO/HAPPO agents.
"""

import numpy as np
from typing import Tuple, Optional


# =====================================================================
# State Discretization
# =====================================================================

class StateDiscretizer:
    """
    Discretize raw continuous observations into a single integer state index.

    Uses 5 key features from the observation vector, each binned into 3 levels
    (low / medium / high), yielding 3^5 = 243 unique states.

    Features (from Phase-1 obs):
        0. time_progress     : obs[0]  — year / 12                    (early / mid / late)
        1. carbon_price_norm : obs[2]  — price_ma3 / price_max        (low / medium / high)
        2. green_fraction    : sum of mix[2:5] from company           (dirty / mixed / green)
        3. compliance_gap    : obs[13] — auction_gap / 5              (surplus / balanced / deficit)
        4. carry_forward     : obs[20] — carry_forward / 5            (none / some / heavy)

    Note: Phase-1 observation dims [23] and [24] were appended for
    auction_volume_ratio and msr_reserve_norm. Existing indices used by
    the discretizer (gap at 13 and carry-forward at 20) remain unchanged.

    Each feature is mapped to {0, 1, 2} via configurable bin edges.
    """

    # Feature extraction indices (from Phase-1 obs vector)
    _OBS_TIME = 0         # year / 12
    _OBS_PRICE_MA3 = 2    # price_ma3 / price_max
    _OBS_GAP = 13         # auction_gap / 5  (banked allowances)
    _OBS_CF = 20          # carry_forward / 5

    # Bin edges: value <= edge[0] → 0, edge[0] < value <= edge[1] → 1, else → 2
    _BINS = {
        "time":       [0.33, 0.67],    # early / mid / late episode
        "price":      [0.15, 0.40],    # low / medium / high carbon price
        "green":      [0.35, 0.65],    # dirty / mixed / green mix
        "gap":        [0.10, 0.60],    # deficit / balanced / surplus  (gap/5 scale)
        "cf":         [0.01, 0.20],    # none / some / heavy carry-forward
    }

    N_FEATURES = 5
    N_BINS = 3
    N_STATES = N_BINS ** N_FEATURES  # 243

    def discretize(self, obs: np.ndarray, company) -> int:
        """
        Convert raw observation + company state to a discrete state index.

        Parameters
        ----------
        obs : np.ndarray
            Phase-1 observation vector (variable length depending on opponent modeling).
        company : Company
            The company object for this agent.

        Returns
        -------
        state : int
            Integer in [0, 242].
        """
        features = self._extract_features(obs, company)
        return self._features_to_index(features)

    def _extract_features(self, obs: np.ndarray, company) -> np.ndarray:
        """Extract 5 features from obs + company."""
        time_progress = float(obs[self._OBS_TIME])
        price_norm = float(obs[self._OBS_PRICE_MA3])
        green_frac = company.green_frac
        gap = float(obs[self._OBS_GAP])
        cf = float(obs[self._OBS_CF]) if len(obs) > self._OBS_CF else 0.0
        return np.array([time_progress, price_norm, green_frac, gap, cf])

    def _bin_value(self, value: float, edges: list) -> int:
        """Map a continuous value to bin {0, 1, 2}."""
        if value <= edges[0]:
            return 0
        elif value <= edges[1]:
            return 1
        else:
            return 2

    def _features_to_index(self, features: np.ndarray) -> int:
        """Convert 5 binned features to a single integer (mixed-radix)."""
        bin_names = ["time", "price", "green", "gap", "cf"]
        index = 0
        for i, name in enumerate(bin_names):
            b = self._bin_value(features[i], self._BINS[name])
            index = index * self.N_BINS + b
        return index

    def index_to_bins(self, index: int) -> list:
        """Inverse: convert state index back to list of bin values (for analysis)."""
        bins = []
        for _ in range(self.N_FEATURES):
            bins.append(index % self.N_BINS)
            index //= self.N_BINS
        return bins[::-1]


# =====================================================================
# Action Profile Mapping
# =====================================================================

class ActionProfileMapper:
    """
    Maps discrete profile indices to continuous action vectors.

    Phase-1 profiles (6 strategies):
        0. Conservative:    Low bid, cover need, minimal invest, solar
        1. Moderate:        MA3-anchored bid, full coverage, moderate invest, solar
        2. Aggressive buy:  High bid, over-cover, high invest, onshore
        3. Green push:      High bid, full cover, max invest, onshore
        4. Financial:       Low bid, under-cover, no invest, solar
        5. Panic buy:       Very high bid, max coverage, moderate invest, solar

    Phase-2 profiles (4 strategies):
        0. Hold:            Do nothing on secondary market
        1. Sell surplus:    Sell at slight premium
        2. Buy shortfall:  Buy at moderate premium
        3. Aggressive buy:  Buy at high premium, large quantity
    """

    N_AUCTION_PROFILES = 6
    N_SECONDARY_PROFILES = 4

    def get_auction_action(self, profile_idx: int, company, price_ma3: float,
                           config: dict) -> np.ndarray:
        """
        Convert Phase-1 profile index to a 6D continuous action vector.

        Parameters
        ----------
        profile_idx : int
            Index in [0, 5].
        company : Company
            The company object.
        price_ma3 : float
            3-year moving average clearing price.
        config : dict
            Full config dictionary.

        Returns
        -------
        action : np.ndarray, shape (6,)
            [bid_price, qty_mult, invest_frac, logit_onshore, logit_offshore, logit_solar]
        """
        aq = config["auction"]
        inv = config["investment"]
        price_min = aq["price_min"]
        price_max = aq["price_max"]
        qty_low = aq.get("qty_mult_low", 0.3)
        qty_high = aq.get("qty_mult_high", 2.0)
        max_invest = inv["max_invest_frac"]
        penalty_rate = config.get("penalty", {}).get("rate", 100.0)

        # Base anchor price
        anchor = max(price_min, min(price_ma3, penalty_rate))

        profiles = {
            0: {  # Conservative
                "bid": np.clip(anchor * 0.90, price_min, price_max),
                "qty": 1.0,
                "invest": 0.01,
                "logits": [-1.0, -1.0, 1.0],  # solar
            },
            1: {  # Moderate
                "bid": np.clip(anchor * 1.10, price_min, price_max),
                "qty": 1.0,
                "invest": 0.03,
                "logits": [-1.0, -1.0, 1.0],  # solar
            },
            2: {  # Aggressive buy
                "bid": np.clip(anchor * 1.30, price_min, price_max),
                "qty": 1.5,
                "invest": 0.06,
                "logits": [1.0, -1.0, -1.0],  # onshore
            },
            3: {  # Green push
                "bid": np.clip(anchor * 1.20, price_min, price_max),
                "qty": 1.0,
                "invest": max_invest,
                "logits": [1.0, -1.0, -1.0],  # onshore
            },
            4: {  # Financial (cost-minimize)
                "bid": np.clip(anchor * 0.80, price_min, price_max),
                "qty": 0.7,
                "invest": 0.0,
                "logits": [-1.0, -1.0, 1.0],  # solar (cheapest if any)
            },
            5: {  # Panic buy
                "bid": np.clip(penalty_rate * 1.20, price_min, price_max),
                "qty": qty_high,
                "invest": 0.03,
                "logits": [-1.0, -1.0, 1.0],  # solar (fastest)
            },
        }

        p = profiles[profile_idx]
        qty = float(np.clip(p["qty"], qty_low, qty_high))
        invest = float(np.clip(p["invest"], 0.0, max_invest))

        return np.array([
            p["bid"], qty, invest,
            p["logits"][0], p["logits"][1], p["logits"][2],
        ], dtype=np.float32)

    def get_secondary_action(self, profile_idx: int, company,
                             clearing_price: float, config: dict) -> np.ndarray:
        """
        Convert Phase-2 profile index to a 2D continuous action vector.

        Parameters
        ----------
        profile_idx : int
            Index in [0, 3].
        company : Company
            The company object.
        clearing_price : float
            Current auction clearing price.
        config : dict
            Full config dictionary.

        Returns
        -------
        action : np.ndarray, shape (2,)
            [sec_price_multiplier, sec_qty]
        """
        aq = config["auction"]
        trading = config.get("trading", {})
        qty_max = aq["quantity_max"]
        sec_low = trading.get("sec_mult_low", 0.8)
        sec_high = trading.get("sec_mult_high", 1.3)

        # Estimate surplus/deficit
        need = company.compute_estimate_need()

        profiles = {
            0: {  # Hold
                "price_mult": 1.0,
                "qty": 0.0,
            },
            1: {  # Sell surplus
                "price_mult": 1.1,
                "qty": -min(1.0, qty_max),  # sell 1 Mt
            },
            2: {  # Buy shortfall
                "price_mult": 1.2,
                "qty": min(1.0, qty_max),   # buy 1 Mt
            },
            3: {  # Aggressive buy
                "price_mult": sec_high,
                "qty": min(2.0, qty_max),   # buy 2 Mt
            },
        }

        p = profiles[profile_idx]
        price_mult = float(np.clip(p["price_mult"], sec_low, sec_high))
        qty = float(np.clip(p["qty"], -qty_max, qty_max))

        return np.array([price_mult, qty], dtype=np.float32)

    @staticmethod
    def auction_profile_names() -> list:
        return ["Conservative", "Moderate", "Aggressive", "GreenPush",
                "Financial", "PanicBuy"]

    @staticmethod
    def secondary_profile_names() -> list:
        return ["Hold", "SellSurplus", "BuyShortfall", "AggressiveBuy"]


# =====================================================================
# Q-Learning Agent
# =====================================================================

class QLearningAgent:
    """
    Tabular Q-learning agent for the ETS environment.

    Maintains a Q-table of shape (n_states, n_auction_profiles, n_secondary_profiles)
    and uses ε-greedy exploration over discrete action profiles.

    Parameters
    ----------
    agent_id : int
        Agent index (0-based).
    alpha : float
        Learning rate for Q-update.
    gamma : float
        Discount factor.
    seed : int
        Random seed.
    """

    def __init__(self, agent_id: int, alpha: float = 0.1, gamma: float = 0.95,
                 seed: int = 42):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.discretizer = StateDiscretizer()
        self.mapper = ActionProfileMapper()

        n_states = StateDiscretizer.N_STATES
        n_a1 = ActionProfileMapper.N_AUCTION_PROFILES
        n_a2 = ActionProfileMapper.N_SECONDARY_PROFILES

        # Q-table: Q(state, auction_profile, secondary_profile)
        self.q_table = np.zeros((n_states, n_a1, n_a2), dtype=np.float64)

        # Tracking for analysis
        self.action_history = []  # list of (state, a1_idx, a2_idx)

    def select_auction_action(self, obs: np.ndarray, company,
                              price_ma3: float, config: dict,
                              epsilon: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        Select a Phase-1 action using ε-greedy over auction profiles.

        Returns
        -------
        action_vec : np.ndarray, shape (6,)
            Continuous action for the environment.
        profile_idx : int
            Which profile was selected.
        """
        state = self.discretizer.discretize(obs, company)
        n_profiles = ActionProfileMapper.N_AUCTION_PROFILES

        if self.rng.random() < epsilon:
            profile_idx = int(self.rng.integers(0, n_profiles))
        else:
            # Greedy: max Q over auction profiles, marginalized over secondary
            q_auction = self.q_table[state].max(axis=1)  # shape (n_a1,)
            profile_idx = int(np.argmax(q_auction))

        action_vec = self.mapper.get_auction_action(
            profile_idx, company, price_ma3, config)
        return action_vec, profile_idx

    def select_secondary_action(self, obs: np.ndarray, company,
                                clearing_price: float, config: dict,
                                a1_idx: int,
                                epsilon: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        Select a Phase-2 action using ε-greedy over secondary profiles.

        Uses the previously selected auction profile (a1_idx) to condition
        the Q-value lookup: Q(state, a1_idx, :).

        Returns
        -------
        action_vec : np.ndarray, shape (2,)
            Continuous action for the environment.
        profile_idx : int
            Which secondary profile was selected.
        """
        state = self.discretizer.discretize(obs, company)
        n_profiles = ActionProfileMapper.N_SECONDARY_PROFILES

        if self.rng.random() < epsilon:
            profile_idx = int(self.rng.integers(0, n_profiles))
        else:
            q_sec = self.q_table[state, a1_idx, :]  # shape (n_a2,)
            profile_idx = int(np.argmax(q_sec))

        action_vec = self.mapper.get_secondary_action(
            profile_idx, company, clearing_price, config)
        return action_vec, profile_idx

    def update(self, state: int, a1_idx: int, a2_idx: int,
               reward: float, next_state: int, done: bool):
        """
        Standard Q-learning update rule.

        Q(s, a1, a2) ← Q(s, a1, a2) + α [r + γ max_{a1',a2'} Q(s', a1', a2') - Q(s, a1, a2)]
        """
        current_q = self.q_table[state, a1_idx, a2_idx]

        if done:
            target = reward
        else:
            next_max = self.q_table[next_state].max()
            target = reward + self.gamma * next_max

        self.q_table[state, a1_idx, a2_idx] += self.alpha * (target - current_q)

    def get_top_q_values(self, top_k: int = 5) -> list:
        """Return top-k (state, a1, a2, q_value) tuples by Q-value."""
        flat_indices = np.argsort(self.q_table.ravel())[-top_k:][::-1]
        results = []
        for flat_idx in flat_indices:
            s, remainder = divmod(flat_idx, self.q_table.shape[1] * self.q_table.shape[2])
            a1, a2 = divmod(remainder, self.q_table.shape[2])
            results.append((int(s), int(a1), int(a2), self.q_table[s, a1, a2]))
        return results

    def get_greedy_policy(self) -> np.ndarray:
        """
        Return the greedy action pair for each state.

        Returns
        -------
        policy : np.ndarray, shape (243, 2)
            policy[s] = [best_a1, best_a2]
        """
        n_states = StateDiscretizer.N_STATES
        policy = np.zeros((n_states, 2), dtype=int)
        for s in range(n_states):
            flat = np.argmax(self.q_table[s])
            a1, a2 = divmod(flat, self.q_table.shape[2])
            policy[s] = [a1, a2]
        return policy
