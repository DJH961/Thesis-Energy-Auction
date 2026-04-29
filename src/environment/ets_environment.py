"""
ets_environment.py
==================
Two-phase EU ETS environment for multi-agent RL with technology-specific
energy mix, real-data-grounded investment costs, and construction queues.

Each year is split into two decision phases:
  Phase 1 (Auction + Investment):
    - Agents observe market state (20+2*(N-1) dim with opponent modeling)
    - Decide: [bid_price, quantity, invest_frac, tech_choice_logit0..2]
    - Auction clears, investments are planned
    - Returns enriched observation (phase1_dim+4) with auction results

  Phase 2 (Secondary Market):
    - Agents observe auction results (phase1_dim+4 dim)
    - Decide: [secondary_price, secondary_quantity]
    - Secondary market clears, compliance is checked
    - Returns reward and next year's Phase 1 observation

Action space (Phase 1): 6D continuous
  [bid_price, quantity, invest_frac, tech_logit_onshore, tech_logit_offshore, tech_logit_solar]
  tech_choice is derived by argmax of the 3 logits (discrete from continuous)

Action space (Phase 2): 2D continuous
  [price_abs (EUR/t), quantity]

"""

import copy
import warnings

import numpy as np
import gymnasium as gym
from typing import List, Optional

from src.auction.market_clearing_ets import market_clearing_ets, build_bids, settle_auction
from src.environment.cap_schedule import CapSchedule
from src.environment.company import Company
from src.environment.market_calibration import compute_market_params
from src.utils.price_anchor import compute_fundamental_anchor
from src.environment.phantom_bidder import PhantomBidder
from src.agents import heuristic_policy


def _compute_marginal_ef(active_companies: list, config: dict) -> float:
    """
    Compute marginal emission factor for carbon cost pass-through.

    The marginal EF is the EF of the most carbon-intensive technology with
    sufficient system-wide capacity share (>5% hard threshold, soft blend
    between 3-8%). When coal has material capacity, it is the marginal setter.

    Reference: Fabra & Reguant (2014) AER; Sijm et al. (2006) Energy Policy.
    """
    tech_efs = config.get("technologies", {}).get("emission_factors", [0.82, 0.49, 0.011, 0.012, 0.048])
    n_techs = len(tech_efs)

    # Compute system-wide capacity (TWh output) per technology
    total_output = sum(c.output_twh for c in active_companies)
    if total_output < 1e-9:
        return float(np.mean(tech_efs))

    tech_shares = np.zeros(n_techs)
    for c in active_companies:
        for t in range(min(n_techs, len(c.mix))):
            tech_shares[t] += float(c.mix[t]) * float(c.output_twh)
    tech_shares /= total_output

    system_ef = float(np.mean([c.weighted_emission_factor for c in active_companies]))

    # Sort technologies by EF descending (most carbon-intensive first)
    tech_efs_arr = np.array(tech_efs[:n_techs], dtype=float)
    sorted_indices = np.argsort(tech_efs_arr)[::-1]

    share_low = 0.03   # below this: technology not material
    share_high = 0.08  # above this: technology fully marginal

    for t in sorted_indices:
        share = float(tech_shares[t])
        ef_t = float(tech_efs_arr[t])
        if share < share_low:
            continue
        if share >= share_high:
            # Technology is fully marginal
            return ef_t
        # Soft blend between system_ef and this technology's EF
        blend = (share - share_low) / (share_high - share_low)
        return float((1.0 - blend) * system_ef + blend * ef_t)

    # No technology above threshold: fall back to system average
    return system_ef


class ETSEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__()

        # Keep environment-local mutable config to avoid caller side effects.
        self.config = copy.deepcopy(config)
        cfg = self.config

        self.n_agents = cfg["companies"]["n_agents"]           # PPO learning agents (external interface)
        self.n_bots = cfg["companies"].get("n_bot_agents", 0)  # heuristic bot agents
        self.n_total = self.n_agents + self.n_bots                # total market participants
        self.n_years = cfg["simulation"]["n_years"]

        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Keep a pristine calibration config (pre-extension arrays) so re-calibration
        # does not accidentally count appended bot entries.
        self._calibration_config = copy.deepcopy(cfg)

        params = compute_market_params(self._calibration_config, n_active_bots=self.n_bots)
        self._apply_market_params_to_config(cfg, params)
        self._market_params = params

        print(
            f"Market calibration: {params['n_active_participants']} participants, "
            f"emissions={params['total_emissions']:.1f} Mt, cap={params['cap_year_0']:.1f} Mt"
        )

        self.cap_schedule = CapSchedule(cfg)

        bot_cfg = cfg.get("bots", {})
        self._enhanced_noise_cfg = bot_cfg.get("enhanced_noise", {})
        self._enhanced_noise_enabled = bool(self._enhanced_noise_cfg.get("enabled", False))
        self._fade_cfg = bot_cfg.get("fade_schedule", {})
        self._fade_enabled = bool(self._fade_cfg.get("enabled", False))
        self._fade_schedule = self._fade_cfg.get("schedule", [[0, self.n_bots]])
        self._n_active_bots = self.n_bots
        self._bot_budget_stressed = np.zeros(self.n_bots, dtype=bool)

        # Extend config arrays to include bot entries (so Company can index by agent_id)
        if self.n_bots > 0:
            bot_mixes = cfg["companies"].get("bot_initial_mix", [])[: self.n_bots]
            bot_rw = cfg["companies"].get("bot_reward_weights", [[0.5, 0.5]] * self.n_bots)
            bot_rw = bot_rw[: self.n_bots]

            cfg["companies"]["initial_mix"] = cfg["companies"]["initial_mix"] + bot_mixes
            cfg["companies"]["reward_weights"] = cfg["companies"]["reward_weights"] + bot_rw

            bot_budgets = cfg["budget"].get("bot_annual_budgets", [1200.0] * self.n_bots)
            bot_budgets = bot_budgets[: self.n_bots]
            cfg["budget"]["annual_budgets"] = cfg["budget"].get("annual_budgets", []) + bot_budgets

            bot_capex_tp = cfg["budget"].get("bot_capex_throughputs", [130.0] * self.n_bots)
            bot_capex_tp = bot_capex_tp[: self.n_bots]
            cfg["budget"]["capex_throughputs"] = cfg["budget"].get("capex_throughputs", []) + bot_capex_tp

            bot_debt_hr = cfg["budget"].get("bot_debt_headrooms", [0.0] * self.n_bots)
            bot_debt_hr = bot_debt_hr[: self.n_bots]
            cfg["budget"]["debt_headrooms"] = cfg["budget"].get("debt_headrooms", []) + bot_debt_hr

        initial_mixes = cfg["companies"]["initial_mix"]
        self.companies: List[Company] = [
            Company(
                agent_id=i, config=cfg,
                initial_mix=initial_mixes[i], rng=self.rng,
            )
            for i in range(self.n_total)
        ]

        # Episode state — all arrays are n_total sized (learning + bots)
        self.current_year = 0
        self.current_episode = 0          # updated by training loop via set_episode()
        self._price_initial = float(config.get("price", {}).get("initial_expected", 70.0))
        self.last_clearing_price = self._price_initial
        self.expected_price = self._price_initial
        self._price_history: List[float] = []
        self.last_secondary_price = self._price_initial
        self.last_secondary_volume = 0.0
        self._last_auction_volume = self.cap_schedule.get_cap(0)
        self._last_gaps = np.zeros(self.n_total)
        self.holdings = np.zeros(self.n_total)
        self.episode_done = False

        # shaping weight — decays from 1.0 to 0.0 over training (set by train.py)
        self.shaping_weight = 1.0

        # Private per-episode urgency scalars — initialised to 1.0; sampled each reset()
        self._urgency_scalars = np.ones(self.n_agents, dtype=float)

        # Secondary market profit tracking (EMA per agent)
        self._secondary_profit_ema = np.zeros(self.n_total)
        self._ema_alpha = 0.1

        # Per-episode EMA of secondary clearing — used by the bid-head
        # coverage-gap reward as a smooth proxy for expected remediation
        # cost. None until the first secondary clearing in the episode
        # (year 0 falls back to the fundamental anchor).
        sec_proxy_cfg = config.get("reward", {}).get("sec_proxy", {})
        self._sec_proxy_enabled    = bool(sec_proxy_cfg.get("enabled", True))
        self._sec_proxy_ema_alpha = float(sec_proxy_cfg.get("ema_alpha", 0.30))
        self._sec_proxy_cap_mult  = float(sec_proxy_cfg.get("cap_mult", 1.5))
        self._sec_price_ema: Optional[float] = None

        # Per-agent last secondary buy price (feeds the WTP exploration anchor)
        self._last_secondary_buy_price = np.zeros(self.n_total)
        self._cumulative_alloc = np.zeros(self.n_total)
        self._cumulative_emissions = np.zeros(self.n_total)

        # Auction results (stored between phase 1 and phase 2)
        self._phase1_allocations = None
        self._phase1_payments = None
        self._phase1_clearing_price = 0.0
        self._phase1_invest_costs = None
        self._phase1_obs = None
        self._phase1_log = None
        self._phase1_bid_prices = None
        self._phase1_bid_quantities = None  # actual Mt quantities after multiplier expansion

        # Per-agent fossil fraction history within the episode (last 3 years)
        self._fossil_frac_history: List[List[float]] = [[] for _ in range(self.n_total)]

        # Stochastic emission shocks — computed per year in step_auction()
        # Stores the shocked realized emissions and the shock values for obs/logging
        self._current_emissions = np.zeros(self.n_total)   # shocked
        self._current_emission_shocks = np.zeros(self.n_total)  # ε_it values

        # CF noise per agent per tech — computed per year in step_auction()
        self._current_cf_noise = np.zeros((self.n_total, 5))
        self._p6_cancellations = np.zeros(self.n_total, dtype=int)

        # Unsold allowance rollover: volume offered at auction but not allocated
        # carries forward to the next year's auction supply.
        self._unsold_rollover = 0.0

        # Default carry-forward tracking
        # _defaulted_volume_pending: allowance volume returned by defaults to add next year
        self._defaulted_volume_pending = 0.0
        # One-shot diagnostic flag: emit a warning the first time defaulted volume
        # is dropped due to auction.carry_forward_defaults=false.
        self._warned_defaults_dropped = False
        # Opponent snapshot buffers for 7D lagged opponent obs
        self._opponent_snapshots      = np.zeros((self.n_total, 7), dtype=float)
        self._opponent_snapshots_prev = np.zeros((self.n_total, 7), dtype=float)
        self._sec_bought              = np.zeros(self.n_total, dtype=float)
        self._sec_sold                = np.zeros(self.n_total, dtype=float)
        self._last_compliance_gaps    = np.zeros(self.n_total, dtype=float)
        # Per-agent collateral locked in Phase 1 (step_auction).
        # Stored so Phase 2 can charge the opportunity cost without re-deriving bids.
        self._collateral_locked = np.zeros(self.n_total)
        # Per-agent collateral load from PREVIOUS year: collateral_locked / annual_budget.
        # Exposed in Phase 1 obs so agents learn to avoid over-committing and defaulting.
        self._last_collateral_load = np.zeros(self.n_total)
        # Per-agent bid affordability from PREVIOUS year
        self._bid_affordability = np.zeros(self.n_total)
        # Bid price change limit config (read from YAML each step, not set externally).
        self._bid_change_limit: float = 0.0  # cached value for obs dim
        # Per-agent clip feedback signals (set during step_auction / step_secondary)
        self._last_bid_price_clip = np.zeros(self.n_total)        # actual - requested (EUR/t), signed
        self._last_budget_price_clip = np.zeros(self.n_total)     # budget clip delta (EUR/t), signed
        self._last_bid_qty_clip_ratio = np.ones(self.n_total)   # actual / requested qty [0,1]
        self._last_invest_clip_ratio = np.ones(self.n_total)    # actual / requested invest_frac [0,1]
        self._last_sec_qty_clip_ratio = np.ones(self.n_total)   # actual / requested sec qty (signed)
        # PCL ceiling for obs dim [38]: upper bid bound from bid_change_limit
        self._pcl_ceiling: float = float(self.config["auction"]["price_max"])
        # Cumulative per-agent count of collateral warnings
        self._collateral_warning_count: np.ndarray = np.zeros(self.n_total, dtype=int)
        self._collateral_clip_events: dict[int, int] = {}

        # Dynamic reserve tracking
        self._last_effective_reserve = config["ets"].get("reserve_price", 0.0)
        # Price history anchor: "auction" appends only successful auction clearing prices;
        # "secondary" appends secondary clearing prices (can include reserve-price fallbacks
        # when auctions fail, which distorts the MA3 and causes erratic bid spirals).
        # Default is "auction" to keep the MA3 stable and informative.
        self._reserve_anchor = config["ets"].get("price_history_anchor", "auction")
        self._consecutive_years_without_valid_auction_clear = 0

        # Secondary liquidity pool EMA anchor state (only used when pool enabled)
        self._liquidity_ref_ema = self._price_initial

        # Episode-level inflation path (shared by all participants)
        self._inflation_rates: List[float] = []
        self._inflation_factors: List[float] = [1.0]
        self._build_episode_inflation_path()

        # Per-bot persistent stochastic valuation parameters (sampled at episode start)
        self._bot_valuation_noise = np.zeros(self.n_bots)  # EUR/t, added to market_anchor
        self._bot_urgency_mult = np.ones(self.n_bots)  # multiplier on urgency

        # Opponent modeling (5D public info per opponent)
        opp_enabled = config.get("opponent_modeling", {}).get("enabled", False)
        self._opponent_modeling = opp_enabled

        # MAC fuel-switching tracking
        self._mac_reductions = np.zeros(self.n_total)
        self._mac_costs = np.zeros(self.n_total)

        # Banking timing signal: per-agent weighted-average cost basis of holdings.
        # Initialized each episode; updated after auction and secondary purchases.
        self._bank_cost_basis = np.zeros(self.n_total)

        # Terminal liquidation components from the latest reward computation.
        # These are logged into year_log so notebook diagnostics can mirror
        # the exact reward logic without re-implementing formulas.
        self._last_terminal_bank_values = np.zeros(self.n_total)
        self._last_terminal_queue_values = np.zeros(self.n_total)
        self._last_terminal_liquidation_values = np.zeros(self.n_total)
        self._last_reward_base_values = np.zeros(self.n_total)
        self._last_reward_shaping_values = np.zeros(self.n_total)
        # Phase-2 invest-stream reward (ESG + terminal-queue contribution),
        # exposed so train.py can decompose the total reward into a "main"
        # stream (compliance, secondary financials, terminal bank) and an
        # "invest" stream that drives the split-head invest critic.
        self._last_invest_reward_phase2 = np.zeros(self.n_total)

        # Price normalization constant
        self._price_norm = config["auction"]["price_max"]

        # Log environment config summary
        auction_cfg = self.config.get("auction", {})
        print(f"[ETSEnvironment] {self.n_agents} learning + {self.n_bots} bot = {self.n_total} total agents"
              f" | cancel_under_subscribed={auction_cfg.get('cancel_under_subscribed', False)}")
        # Print initial bank-seed context once per environment instance (avoid reset spam).
        self._printed_initial_bank_seed_context = False
        self._year1_tnac_warning_emitted = False

        # Sanity check: in static mode, price_min must be >= reserve_price.
        # In dynamic mode, the effective reserve is computed each year, so
        # price_min can be below the absolute floor (agents learn to bid above).
        _price_min = config["auction"]["price_min"]
        _reserve = config["ets"].get("reserve_price", 0.0)
        _reserve_mode = config["ets"].get("reserve_price_mode", "static")
        if _reserve_mode == "static":
            assert _price_min >= _reserve, (
                f"Config error: auction.price_min ({_price_min}) < ets.reserve_price ({_reserve}). "
                "Agents can produce valid-looking bids that the auction silently rejects, "
                "leaving most of the cap unallocated. Set auction.price_min = reserve_price."
            )

        # Logging
        self.episode_log: List[dict] = []

        # Phantom bidder (financial intermediary demand)
        self._phantom_bidder = PhantomBidder(config, self.rng)

        # Reward channel diagnostics
        self._last_reward_channels: dict = {}
        self._last_auction_reward_channels: dict = {}

        # Per-agent per-year diagnostics (validation diagnostics)
        self._last_per_agent_diag: dict = {}

        # Emission factor diagnostics
        self._last_system_ef = 0.0
        self._last_marginal_ef = 0.0
        self._current_marginal_ef = 0.0

    # ------------------------------------------------------------------
    # Training loop interface
    # ------------------------------------------------------------------

    def set_episode(self, episode: int):
        """Called by training loop to communicate current episode for shaping decay."""
        self.current_episode = episode
        reward_cfg = self.config.get("reward", {})
        decay_ep = reward_cfg.get("shaping_decay_episode", 3000)
        if decay_ep <= 0:
            # Auto: 12% of n_episodes, clamped to [300, 8000]
            n_ep = self.config.get("simulation", {}).get("n_episodes", 10000)
            decay_ep = max(300, min(8000, int(0.12 * n_ep)))
            # Write back so subsequent calls don't re-resolve
            reward_cfg["shaping_decay_episode"] = decay_ep
        floor = reward_cfg.get("shaping_weight_floor", 0.0)
        self.shaping_weight = max(floor, 1.0 - episode / max(decay_ep, 1))

    def set_bid_change_limit(self, limit: float):
        """Deprecated: bid_change_limit is now read from config each step."""
        pass

    @staticmethod
    def _apply_market_params_to_config(config: dict, params: dict):
        """Write derived calibration values into ETS config for downstream readers."""
        config.setdefault("ets", {})["cap_year_0"] = float(params["cap_year_0"])
        config.setdefault("ets", {}).setdefault("msr", {})["tnac_upper"] = float(params["tnac_upper"])
        config.setdefault("ets", {}).setdefault("msr", {})["tnac_mid"] = float(params["tnac_mid"])
        config.setdefault("ets", {}).setdefault("msr", {})["tnac_lower"] = float(params["tnac_lower"])
        config.setdefault("ets", {}).setdefault("msr", {})["release_amount"] = float(params["release_amount"])
        config.setdefault("ets", {}).setdefault("msr", {})["emergency_release_amount"] = float(
            params["emergency_release_amount"]
        )

    def _resolve_fade_active_bots(self, episode: int) -> int:
        """Resolve active bot count for the given episode from fade schedule."""
        if not self._fade_enabled:
            return self.n_bots

        n_active = self.n_bots
        schedule = sorted(self._fade_schedule, key=lambda x: int(x[0]))
        for threshold, count in schedule:
            if int(episode) >= int(threshold):
                n_active = int(count)
            else:
                break
        return max(0, min(self.n_bots, n_active))

    def _is_agent_active(self, idx: int) -> bool:
        """Return whether an internal participant index is active this episode."""
        if idx < self.n_agents:
            return True
        bot_idx = idx - self.n_agents
        return bot_idx < self._n_active_bots

    def _active_mask(self) -> np.ndarray:
        """Boolean mask of currently active participants across learning+bot arrays."""
        mask = np.ones(self.n_total, dtype=bool)
        if self.n_bots > 0 and self._n_active_bots < self.n_bots:
            start = self.n_agents + self._n_active_bots
            mask[start:] = False
        return mask

    # ------------------------------------------------------------------
    # Dynamic reserve price
    # ------------------------------------------------------------------

    def _compute_dynamic_reserve(self) -> float:
        """
        Compute effective reserve price for this year's auction.

        In 'dynamic' mode: max(absolute_floor, discount × MA3_price).
        Falls back to reserve_initial when no MA3 history is available.
        In 'static' mode: returns the configured reserve_price as-is.
        """
        ets_cfg = self.config["ets"]
        mode = ets_cfg.get("reserve_price_mode", "static")
        abs_floor = ets_cfg.get("reserve_price", 0.0)

        if mode != "dynamic":
            return abs_floor

        discount = ets_cfg.get("reserve_discount", 0.80)
        initial = ets_cfg.get("reserve_initial", 50.0)

        ma3 = self._compute_price_ma3()
        if not self._price_history:
            base_reserve = max(abs_floor, initial)
        else:
            base_reserve = max(abs_floor, discount * ma3)

        # If auctions have failed for consecutive years, decay the effective reserve
        # exponentially toward the absolute floor: decay = 0.8^n_consecutive_failures.
        if self._consecutive_years_without_valid_auction_clear >= 2:
            decay = 0.8 ** self._consecutive_years_without_valid_auction_clear
            return max(abs_floor, abs_floor + (base_reserve - abs_floor) * decay)

        return base_reserve

    def _build_episode_inflation_path(self):
        """Build one inflation path per episode, shared by all agents."""
        pen_cfg = self.config.get("penalty", {})
        base_rate = float(pen_cfg.get("inflation_rate", 0.0))
        rand_std = float(max(0.0, pen_cfg.get("inflation_random_std", 0.0)))
        rand_window = float(max(0.0, pen_cfg.get("inflation_random_window", 0.0)))

        if rand_std > 0.0:
            rates = self.rng.normal(base_rate, rand_std, size=self.n_years)
            rates = np.maximum(rates, -0.99)
            self._inflation_rates = [float(r) for r in rates]
        elif rand_window > 0.0:
            low = max(-0.99, base_rate - rand_window)
            high = base_rate + rand_window
            rates = self.rng.uniform(low, high, size=self.n_years)
            self._inflation_rates = [float(r) for r in rates]
        else:
            self._inflation_rates = [base_rate for _ in range(self.n_years)]

        self._inflation_factors = [1.0]
        for r in self._inflation_rates:
            self._inflation_factors.append(self._inflation_factors[-1] * (1.0 + r))

    def _inflation_factor(self, current_year: int) -> float:
        y = max(0, min(int(current_year), len(self._inflation_factors) - 1))
        return float(self._inflation_factors[y])

    def _inflation_rate(self, current_year: int) -> float:
        if not self._inflation_rates:
            return float(self.config.get("penalty", {}).get("inflation_rate", 0.0))
        y = max(0, min(int(current_year), len(self._inflation_rates) - 1))
        return float(self._inflation_rates[y])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        self.current_year = 0
        self.episode_done = False
        self.last_clearing_price = self._price_initial
        self.expected_price = self._price_initial
        self.last_secondary_price = self._price_initial
        self.last_secondary_volume = 0.0
        self._last_auction_volume = self.cap_schedule.get_cap(0)
        self._price_history = []
        self._last_gaps = np.zeros(self.n_total)
        self.holdings = np.zeros(self.n_total)
        self.episode_log = []
        self._fossil_frac_history = [[] for _ in range(self.n_total)]
        self._current_emissions = np.zeros(self.n_total)
        self._current_emission_shocks = np.zeros(self.n_total)
        self._current_cf_noise = np.zeros((self.n_total, 5))
        self._p6_cancellations = np.zeros(self.n_total, dtype=int)
        self._last_terminal_bank_values = np.zeros(self.n_total)
        self._last_terminal_queue_values = np.zeros(self.n_total)
        self._last_terminal_liquidation_values = np.zeros(self.n_total)
        self._last_reward_base_values = np.zeros(self.n_total)
        self._last_reward_shaping_values = np.zeros(self.n_total)
        self._last_invest_reward_phase2 = np.zeros(self.n_total)
        self._secondary_profit_ema = np.zeros(self.n_total)
        # Reset per-episode secondary-price EMA.
        self._sec_price_ema = None
        self._last_secondary_buy_price = np.zeros(self.n_total)
        self._cumulative_alloc = np.zeros(self.n_total)
        self._cumulative_emissions = np.zeros(self.n_total)
        self._last_cover_ratio = 1.0  # neutral default before first auction
        self._consecutive_years_without_valid_auction_clear = 0
        self._liquidity_ref_ema = self._price_initial

        self.cap_schedule.reset()
        self._unsold_rollover = 0.0
        self._defaulted_volume_pending = 0.0
        self._sec_bought           = np.zeros(self.n_total, dtype=float)
        self._sec_sold             = np.zeros(self.n_total, dtype=float)
        self._last_compliance_gaps = np.zeros(self.n_total, dtype=float)
        for i, c in enumerate(self.companies):
            self._opponent_snapshots[i] = [
                c.compute_emissions() / 10.0,
                c.green_frac,
                c.fossil_frac,
                float(sum(item["frac_delta"] for item in c._construction_queue)),
                1.0 / max(self.n_total, 1),  # tnac_share_norm: equal-share neutral prior
                0.0,  # net_secondary_norm: neutral prior
                0.0,  # lagged_compliance_gap_norm: neutral prior
            ]
        self._opponent_snapshots_prev = self._opponent_snapshots.copy()
        self._collateral_locked = np.zeros(self.n_total)
        self._last_collateral_load = np.zeros(self.n_total)
        self._bid_affordability = np.zeros(self.n_total)
        self._collateral_warning_count = np.zeros(self.n_total, dtype=int)
        self._collateral_clip_events = {i: 0 for i in range(self.n_total)}
        self._last_bid_price_clip = np.zeros(self.n_total)
        self._last_budget_price_clip = np.zeros(self.n_total)
        self._last_bid_qty_clip_ratio = np.ones(self.n_total)
        self._last_invest_clip_ratio = np.ones(self.n_total)
        self._last_sec_qty_clip_ratio = np.ones(self.n_total)
        self._pcl_ceiling = float(self.config["auction"]["price_max"])
        self._current_marginal_ef = 0.0
        self._build_episode_inflation_path()

        # Seed bank cost basis for the banking timing signal.
        # Pre-banked allowances are treated as acquired below the year-0 anchor,
        # reflecting that holdings were built when prices were historically lower.
        banking_cfg = self.config.get("reward", {}).get("banking_signal", {})
        _init_factor = float(banking_cfg.get("initial_bank_cost_factor", 0.80))
        _anchor_0 = compute_fundamental_anchor(0, self.config)
        self._bank_cost_basis = np.full(self.n_total, _anchor_0 * _init_factor)

        if self._fade_enabled:
            n_active_bots = self._resolve_fade_active_bots(self.current_episode)
            if n_active_bots != self._n_active_bots:
                params = compute_market_params(self._calibration_config, n_active_bots=n_active_bots)
                self._apply_market_params_to_config(self.config, params)
                if hasattr(self.cap_schedule, "update_calibration"):
                    self.cap_schedule.update_calibration(
                        cap_year_0=params["cap_year_0"],
                        tnac_upper=params["tnac_upper"],
                        tnac_mid=params["tnac_mid"],
                        tnac_lower=params["tnac_lower"],
                        release_amount=params["release_amount"],
                        emergency_release_amount=params["emergency_release_amount"],
                    )
                else:
                    self.cap_schedule.cap_year_0 = float(params["cap_year_0"])
                    self.cap_schedule.tnac_upper = float(params["tnac_upper"])
                    self.cap_schedule.tnac_mid = float(params["tnac_mid"])
                    self.cap_schedule.tnac_lower = float(params["tnac_lower"])
                    self.cap_schedule.release_amount = float(params["release_amount"])
                    self.cap_schedule.emergency_release_amount = float(params["emergency_release_amount"])
                self._market_params = params
                print(
                    f"Market recalibration: {params['n_active_participants']} participants, "
                    f"emissions={params['total_emissions']:.1f} Mt, cap={params['cap_year_0']:.1f} Mt"
                )
            self._n_active_bots = n_active_bots
        else:
            self._n_active_bots = self.n_bots

        # Sample per-bot persistent noise at episode start
        if self.n_bots > 0:
            bot_cfg = self.config.get("bots", {})
            if self._enhanced_noise_enabled:
                enh = self._enhanced_noise_cfg
                noise_std = enh.get("valuation_noise_std", bot_cfg.get("valuation_noise_std", 5.0))
                mult_low = enh.get("urgency_mult_low", bot_cfg.get("urgency_mult_low", 0.8))
                mult_high = enh.get("urgency_mult_high", bot_cfg.get("urgency_mult_high", 1.2))
            else:
                noise_std = bot_cfg.get("valuation_noise_std", 5.0)
                mult_low = bot_cfg.get("urgency_mult_low", 0.8)
                mult_high = bot_cfg.get("urgency_mult_high", 1.2)
            for b in range(self.n_bots):
                self._bot_valuation_noise[b] = self.rng.normal(0, noise_std)
                self._bot_urgency_mult[b] = self.rng.uniform(mult_low, mult_high)

            if self._enhanced_noise_enabled:
                stress_prob = float(self._enhanced_noise_cfg.get("budget_stress_prob", 0.0))
                self._bot_budget_stressed = self.rng.random(self.n_bots) < stress_prob
            else:
                self._bot_budget_stressed[:] = False

        # Episode-level warning counters — reset each episode
        self._warnings = {
            "low_alloc": 0, "price_floor": 0, "price_ceil": 0,
            "auct_fail": 0, "low_demand": 0, "no_invest": 0,
            "debt_spiral": 0, "bid_cluster": 0, "over_bank": 0,
            "one_side_sec": 0, "no_trade": 0,
            "cornering": 0, "rsv_reject": 0,
        }
        # Per-agent consecutive-shortfall counter for chronic_short detection
        self._consecutive_shortfall = np.zeros(self.n_total, dtype=int)

        # Reward channel diagnostics
        self._last_reward_channels = {}
        self._last_auction_reward_channels = {}
        self._last_per_agent_diag = {}

        # Private per-episode urgency scalars (not shared across agents).
        # Sampled from LogNormal so E[scalar]=1.0; creates private heterogeneity
        # in effective penalty sensitivity — agents face slightly different urgency.
        urgency_cfg = self.config.get("urgency_scalars", {})
        if urgency_cfg.get("enabled", False):
            sigma_u = float(urgency_cfg.get("lognormal_sigma", 0.30))
            self._urgency_scalars = self.rng.lognormal(
                mean=0.0, sigma=sigma_u, size=self.n_agents
            ).astype(float)
        else:
            self._urgency_scalars = np.ones(self.n_agents, dtype=float)

        initial_mixes = self.config["companies"]["initial_mix"]
        for i, company in enumerate(self.companies):
            company.reset(initial_mix=initial_mixes[i])
            company.rng = self.rng
            company.set_inflation_path(self._inflation_rates)

        # Warm-start — seed construction queue, holdings, price history
        ws_cfg = self.config.get("warm_start", {})
        if ws_cfg.get("enabled", False):
            if ws_cfg.get("burnin_enabled", False):
                self._run_burnin(ws_cfg)
            else:
                self._apply_warm_start(ws_cfg)
        else:
            # Seed initial bank near a configured fraction of annual need, but keep aggregate TNAC
            # inside the MSR band to avoid immediate year-0 intervention.
            initial_bank_fraction = float(self.config.get("ets", {}).get("initial_bank_fraction", 0.30))
            annual_needs = np.array([
                (max(company.compute_estimate_need(), 0.1) if self._is_agent_active(i) else 0.0)
                for i, company in enumerate(self.companies)
            ], dtype=float)
            total_need = float(annual_needs.sum())
            desired_tnac = initial_bank_fraction * total_need
            tnac_lower = float(self.cap_schedule.tnac_lower)
            tnac_upper = float(self.cap_schedule.tnac_upper)
            if total_need > 1e-9 and tnac_upper > tnac_lower:
                target_tnac = float(np.clip(desired_tnac, tnac_lower * 1.05, tnac_upper * 0.95))
                seed_multiple = target_tnac / total_need
            else:
                seed_multiple = initial_bank_fraction
            for i, annual_need in enumerate(annual_needs):
                self.holdings[i] = float(seed_multiple * annual_need) if self._is_agent_active(i) else 0.0

        # One-time startup context: what initial banking is and its unit.
        if not self._printed_initial_bank_seed_context:
            _bank_items = []
            for i in range(self.n_total):
                if not self._is_agent_active(i):
                    continue
                _tag = f"A{i+1}" if i < self.n_agents else f"B{i - self.n_agents + 1}"
                _bank_items.append(f"{_tag}={self.holdings[i]:.2f}")
            _bank_total = float(np.sum(self.holdings))
            print("[ETSEnvironment] Initial bank seed example "
                  "(episode-start allowance holdings, unit: MtCO2 allowances): "
                  + ", ".join(_bank_items))
            print("[ETSEnvironment] Context: this is each active agent's starting bank "
                  f"before year-1 actions/compliance; total TNAC seed={_bank_total:.2f} MtCO2.")
            self._printed_initial_bank_seed_context = True

        # Fully retired bots: no banks, no carry-forward debt.
        for i in range(self.n_total):
            if not self._is_agent_active(i):
                self.holdings[i] = 0.0
                self.companies[i]._carry_forward = 0.0

        # Validate post-warmup TNAC position against the MSR band.
        total_tnac = float(self.holdings.sum())
        if total_tnac > self.cap_schedule.tnac_upper:
            import warnings
            warnings.warn(
                f"[ETSEnvironment] Post-warmup TNAC ({total_tnac:.2f} Mt) exceeds "
                f"tnac_upper ({self.cap_schedule.tnac_upper:.1f} Mt). "
                "MSR will actively drain from year 0. Consider lowering bank_seed_max.",
                stacklevel=2,
            )
        elif total_tnac < self.cap_schedule.tnac_lower:
            import warnings
            warnings.warn(
                f"[ETSEnvironment] Post-warmup TNAC ({total_tnac:.2f} Mt) below "
                f"tnac_lower ({self.cap_schedule.tnac_lower:.1f} Mt). "
                "MSR will release from year 0. Consider raising bank_seed_min.",
                stacklevel=2,
            )

        # Scarcity check: warn if cap trajectory doesn't tighten enough over the episode
        cap_year_0 = self.cap_schedule.get_cap(0)
        cap_year_n = self.cap_schedule.get_cap(self.n_years - 1)
        if cap_year_0 - cap_year_n < 0.30 * cap_year_0:
            import warnings
            warnings.warn(
                f"[ETSEnvironment] Weak scarcity: cap drops only "
                f"{(cap_year_0 - cap_year_n) / cap_year_0 * 100:.1f}% over {self.n_years} years "
                f"(year-0={cap_year_0:.2f} Mt, year-{self.n_years-1}={cap_year_n:.2f} Mt). "
                f"Consider increasing n_years or LRF for meaningful price signals.",
                stacklevel=2,
            )

        obs_phase1 = self._get_obs_phase1()   # shape (n_agents, obs_dim)
        return obs_phase1, {}

    def _seed_construction_queues(self, ws_cfg: dict):
        """Seed in-flight renewable projects for each company."""
        mu_onshore = ws_cfg.get("queue_mu_onshore", 1.5)
        mu_offshore = ws_cfg.get("queue_mu_offshore", 0.5)
        mu_solar = ws_cfg.get("queue_mu_solar", 2.0)

        green_specs = [
            (2, mu_onshore),
            (3, mu_offshore),
            (4, mu_solar),
        ]

        jitter_cfg = self.config.get("construction_jitter", {})
        poisson_lambdas = jitter_cfg.get("poisson_lambdas", [1.0, 1.0, 2.0, 3.0, 1.5])

        for company in self.companies:
            total_seeded_frac = 0.0
            max_seedable = company.fossil_frac * 0.5

            for tech_idx, mu in green_specs:
                lam = float(poisson_lambdas[tech_idx])
                n_projects = int(self.rng.poisson(mu))
                for _ in range(n_projects):
                    frac_delta = float(self.rng.uniform(0.005, 0.025))
                    if total_seeded_frac + frac_delta > max_seedable:
                        frac_delta = max(0.0, max_seedable - total_seeded_frac)
                    if frac_delta < 1e-4:
                        continue
                    completion_year = int(self.rng.integers(0, max(1, int(lam) + 2)))
                    company._construction_queue.append({
                        "tech_idx": tech_idx,
                        "frac_delta": frac_delta,
                        "completion_year": completion_year,
                        "success": True,
                        "capex_spent": company.compute_investment_cost(tech_idx, frac_delta),
                    })
                    total_seeded_frac += frac_delta

    def _run_burnin(self, ws_cfg: dict):
        """
        Run a hidden heuristic pre-period to jointly initialize holdings,
        MSR reserve, price history, and construction queues.
        """
        n_burnin = max(1, int(ws_cfg.get("n_burnin_years", 4)))
        price_mean = float(ws_cfg.get("burnin_price_seed_mean", 70.0))
        price_std = float(ws_cfg.get("burnin_price_seed_std", 10.0))
        n_seed_prices = max(0, int(ws_cfg.get("n_burnin_prices", 2)))
        bank_min = float(ws_cfg.get("bank_seed_min", 0.2))
        bank_max = float(ws_cfg.get("bank_seed_max", 0.4))

        price_min = float(self.config["auction"]["price_min"])
        price_max = float(self.config["auction"]["price_max"])
        reserve_price = float(self.config["ets"].get("reserve_price", 0.0))
        qty_mult_low = float(self.config["auction"].get("qty_mult_low", 0.3))
        qty_mult_high = float(self.config["auction"].get("qty_mult_high", 2.0))
        lot_size = float(self.config["auction"].get("lot_size", 0.0))
        max_agent_share = float(self.config["auction"].get("max_agent_share", 1.0))
        cancel_under_subscribed = bool(
            self.config["auction"].get("cancel_under_subscribed", False)
        )

        pen_cfg = self.config.get("penalty", {})
        base_penalty_rate = float(pen_cfg.get("rate", 0.0))
        inflation_rate = float(pen_cfg.get("inflation_rate", 0.02))

        bot_cfg = self.config.get("bots", {})
        urgency_denoms = bot_cfg.get("urgency_denominators", [1.5] * self.n_bots)

        # (A) Seed initial bank at strategic reserve range.
        for i, company in enumerate(self.companies):
            if not self._is_agent_active(i):
                self.holdings[i] = 0.0
                continue
            annual_need = max(company.compute_estimate_need(), 0.1)
            bank_frac = float(self.rng.uniform(bank_min, bank_max))
            self.holdings[i] = bank_frac * annual_need

        # (B) Seed in-flight construction queues.
        self._seed_construction_queues(ws_cfg)

        # (C) Seed price history with synthetic starting prices.
        self._price_history = []
        for _ in range(n_seed_prices):
            seed_price = float(np.clip(
                self.rng.normal(price_mean, price_std),
                price_min,
                price_max,
            ))
            self._price_history.append(seed_price)

        # (D) Hidden burn-in years.
        for burnin_year in range(-n_burnin, 0):
            for company in self.companies:
                # apply_matured_investments removed from top (double-apply fix);
                # kept only at bottom of burn-in loop (burnin_year + 1).
                company.reset_budget()
                company.reset_capex_budget()

            cap_t = self.cap_schedule.get_cap(burnin_year)
            tnac = float(self.holdings.sum())
            last_price = self._price_history[-1] if self._price_history else price_mean
            burnin_price_ma3 = (
                float(sum(self._price_history[-3:]) / len(self._price_history[-3:]))
                if self._price_history else price_mean
            )
            auction_volume = self.cap_schedule.get_auction_volume(
                year=burnin_year,
                tnac=tnac,
                clearing_price=last_price,
                price_max=price_max,
                penalty_rate=base_penalty_rate,
                inflation_rate=inflation_rate,
                inflation_factor=self._inflation_factor(burnin_year),
                force_msr=True,
                price_ma3=burnin_price_ma3,
            )
            self._last_auction_volume = float(auction_volume)

            bid_actions = np.zeros((self.n_total, 2), dtype=np.float32)
            for i, company in enumerate(self.companies):
                if not self._is_agent_active(i):
                    bid_actions[i, 0] = reserve_price
                    bid_actions[i, 1] = 0.0
                    continue
                infl_factor = (1.0 + inflation_rate) ** burnin_year if inflation_rate > -0.99 else 1.0

                valuation_noise = 0.0
                urgency_multiplier = 1.0
                urgency_denom = 1.5
                if i >= self.n_agents and self.n_bots > 0:
                    b = i - self.n_agents
                    if b < self.n_bots:
                        valuation_noise = float(self._bot_valuation_noise[b])
                        urgency_multiplier = float(self._bot_urgency_mult[b])
                        urgency_denom = urgency_denoms[b] if b < len(urgency_denoms) else 1.5

                action = heuristic_policy.auction_action(
                    company,
                    price_ma3=float(last_price),
                    current_year=burnin_year,
                    n_years=self.n_years,
                    config=self.config,
                    bank=float(self.holdings[i]),
                    reserve_price=reserve_price,
                    inflation_factor=infl_factor,
                    auction_volume=float(auction_volume),
                    cap_t=float(cap_t),
                    valuation_noise=valuation_noise,
                    urgency_multiplier=urgency_multiplier,
                    urgency_denom=urgency_denom,
                    collateral_load_last=float(self._last_collateral_load[i]),
                )

                bid_price = float(np.clip(action[0], price_min, price_max))
                qty_mult = float(np.clip(action[1], qty_mult_low, qty_mult_high))
                annual_need = max(company.compute_estimate_need(), 0.1)
                bid_qty = qty_mult * annual_need
                if lot_size > 0:
                    bid_qty = max(lot_size, round(bid_qty / lot_size) * lot_size)

                bid_actions[i, 0] = bid_price
                bid_actions[i, 1] = max(0.0, float(bid_qty))

            bids = build_bids(bid_actions)
            clearing_price, allocations, _, _ = market_clearing_ets(
                bids=bids,
                q_cap=float(auction_volume),
                reserve_price=reserve_price,
                max_agent_share=max_agent_share,
                rng=self.rng,
                cancel_under_subscribed=cancel_under_subscribed,
                n_agents=self.n_total,
                pricing_rule=self.config["auction"].get("pricing_rule", "uniform"),
            )

            unsold = max(0.0, float(auction_volume) - float(allocations.sum()))
            self._unsold_rollover = unsold
            if self.config["ets"].get("unsold_to_msr", False):
                self.cap_schedule.absorb_unsold(unsold)
            else:
                self.cap_schedule.rollover_unsold(unsold)

            self.holdings += allocations
            for i, company in enumerate(self.companies):
                if not self._is_agent_active(i):
                    self.holdings[i] = 0.0
                    company._carry_forward = 0.0
                    continue
                emissions = company.compute_emissions()
                self.holdings[i] = max(0.0, float(self.holdings[i]) - float(emissions))

            if clearing_price > 0:
                clipped_price = float(np.clip(clearing_price, price_min, price_max))
                # Appending burnin prices seeds the MA3 warm-start so that year 0 of
                # the real episode always has a meaningful price history reference.
                self._price_history.append(clipped_price)
                self.last_clearing_price = clipped_price
                # Seed _prev_ma3 with the MA3 that now includes this clearing so
                # that year 0 of the real episode has a valid MA3 history and the
                # smoothed price guard is active from the start.
                ma3_now = float(
                    sum(self._price_history[-3:]) / len(self._price_history[-3:])
                )
                self.cap_schedule._prev_ma3 = ma3_now

            for i, company in enumerate(self.companies):
                if not self._is_agent_active(i):
                    continue
                if self.rng.random() < 0.3:
                    invest_frac = float(self.rng.uniform(0.01, 0.03))
                    tech_choice = int(self.rng.integers(0, 3))
                    company.plan_investment(
                        tech_choice=tech_choice,
                        invest_frac=invest_frac,
                        current_year=burnin_year,
                    )

            for company in self.companies:
                company.apply_matured_investments(current_year=burnin_year + 1)

        # (E) Keep only MA3-relevant history.
        if len(self._price_history) > 3:
            self._price_history = self._price_history[-3:]

        if self._price_history:
            self.last_clearing_price = float(self._price_history[-1])

        rho = self.config["price"].get("ar1_persistence", 0.85)
        price_floor = compute_fundamental_anchor(0, self.config)
        vol_std = self.config["price"].get("volatility_std", 0.15)
        base_price = self.last_clearing_price if self._price_history else price_mean
        shock = self.rng.normal(0, vol_std) * base_price
        self.expected_price = max(
            rho * base_price + (1.0 - rho) * price_floor + shock,
            price_floor,
        )
        self.last_secondary_price = self.last_clearing_price

        # Clear any unsold rollover that accumulated during burn-in so it does
        # not inflate the year-0 auction volume of the real episode.
        self.cap_schedule._unsold_rollover_pending = 0.0

        # Seed MSR reserve to the configured fraction of cap_year_0.
        # Burn-in never triggers MSR withholding (TNAC stays below tnac_upper),
        # so without this seed the reserve is 0 for the first 2 real years.
        msr_seed_frac = float(ws_cfg.get("msr_initial_reserve_frac", 0.0))
        if msr_seed_frac > 0.0:
            self.cap_schedule._msr_reserve = max(
                self.cap_schedule._msr_reserve,
                msr_seed_frac * self.cap_schedule.cap_year_0,
            )

        # Reflect real 2026 EU ETS starting state: MSR reserve must not exceed
        # tnac_lower. Clamp after seeding so we stay within the design band.
        self.cap_schedule._msr_reserve = min(
            self.cap_schedule._msr_reserve, self.cap_schedule.tnac_lower
        )

        self._calibrate_post_init_bank(ws_cfg)

    def _calibrate_post_init_bank(self, ws_cfg: dict):
        """Calibrate post-initialization holdings to the configured TNAC band."""
        active_mask = self._active_mask()
        tnac_lower = float(self.cap_schedule.tnac_lower)
        tnac_upper = float(self.cap_schedule.tnac_upper)
        tnac_target = float(ws_cfg.get("burnin_tnac_target", 0.5 * (tnac_lower + tnac_upper)))
        tnac_target = float(np.clip(tnac_target, tnac_lower, tnac_upper))

        # Keep all agents strictly positive at year 0 while preserving heterogeneity.
        min_bank_frac = float(ws_cfg.get("bank_min_floor_frac", 0.02))
        min_banks = np.array([
            (max(1e-6, min_bank_frac * max(company.compute_estimate_need(), 0.1))
             if active_mask[i] else 0.0)
            for i, company in enumerate(self.companies)
        ])

        holdings = self.holdings.astype(float).copy()
        holdings[~active_mask] = 0.0
        holdings = np.maximum(holdings, min_banks)
        min_total = float(min_banks.sum())
        if min_total >= tnac_target:
            self.holdings = min_banks
            self.holdings[~active_mask] = 0.0
            return

        weights = np.array([
            (max(company.compute_estimate_need(), 0.1) if active_mask[i] else 0.0)
            for i, company in enumerate(self.companies)
        ], dtype=float)
        wsum = float(weights.sum())
        if wsum <= 0.0:
            self.holdings = np.zeros(self.n_total, dtype=float)
            return
        else:
            weights /= wsum

        excess = holdings - min_banks
        excess_total = float(excess.sum())
        target_excess = tnac_target - min_total

        if excess_total <= 1e-9:
            calibrated = min_banks + target_excess * weights
        else:
            calibrated = min_banks + excess * (target_excess / excess_total)

        self.holdings = np.maximum(calibrated, min_banks)
        self.holdings[~active_mask] = 0.0

    def _apply_warm_start(self, ws_cfg: dict):
        """
        Fallback warm-start seeding (used when burn-in is disabled).

        (A) Seed construction queues.
        (B) Seed initial bank: sample from Uniform(bank_min, bank_max) × annual_need.
        (C) Seed synthetic price history for MA3 bootstrap.
        """
        bank_min = ws_cfg.get("bank_seed_min", 0.2)
        bank_max = ws_cfg.get("bank_seed_max", 0.4)
        n_burnin = ws_cfg.get("n_burnin_prices", 2)

        self._seed_construction_queues(ws_cfg)

        # (B) Seed initial bank
        for i, company in enumerate(self.companies):
            if self._is_agent_active(i):
                annual_need = company.compute_estimate_need()  # Mt
                seed_multiple = float(self.rng.uniform(bank_min, bank_max))
                self.holdings[i] = annual_need * seed_multiple
            else:
                self.holdings[i] = 0.0

        # (C) Seed price history with synthetic burn-in prices
        burnin_mean = float(ws_cfg.get("burnin_price_seed_mean",
                                       self.config["price"].get("initial_expected", 80.0)))
        burnin_std = float(ws_cfg.get("burnin_price_seed_std",
                                      self.config["price"].get("burnin_std", 10.0)))
        for _ in range(n_burnin):
            synthetic_price = float(np.clip(
                self.rng.normal(burnin_mean, burnin_std),
                self.config["auction"]["price_min"],
                self.config["auction"]["price_max"],
            ))
            self._price_history.append(synthetic_price)
            self.last_clearing_price = synthetic_price
            rho = self.config["price"].get("ar1_persistence", 0.85)
            price_floor = compute_fundamental_anchor(0, self.config)
            vol_std = self.config["price"].get("volatility_std", 0.15)
            shock = self.rng.normal(0, vol_std) * synthetic_price
            self.expected_price = max(
                rho * synthetic_price + (1.0 - rho) * price_floor + shock,
                price_floor,
            )
        self.last_secondary_price = self.last_clearing_price
        self._calibrate_post_init_bank(ws_cfg)

    # ------------------------------------------------------------------
    # Phase 1: Auction + Green Investment
    # ------------------------------------------------------------------

    def _generate_bot_auction_actions(self, auction_volume: float = None,
                                      cap_t: float = None) -> np.ndarray:
        """Generate Phase-1 actions for all bot agents using heuristic_policy."""
        if self.n_bots == 0:
            return np.zeros((0, 6), dtype=np.float32)
        price_ma3 = self._compute_price_ma3()
        reserve = self._compute_dynamic_reserve()
        price_min = float(self.config["auction"]["price_min"])
        infl_factor = self._inflation_factor(self.current_year)
        bot_cfg = self.config.get("bots", {})
        urgency_denoms = bot_cfg.get("urgency_denominators", [1.5] * self.n_bots)
        budget_stress_qty_mult = float(self._enhanced_noise_cfg.get("budget_stress_qty_mult", 0.65))
        qty_low = float(self.config["auction"].get("qty_mult_low", 0.3))
        qty_high = float(self.config["auction"].get("qty_mult_high", 2.0))
        actions = np.zeros((self.n_bots, 6), dtype=np.float32)
        for b in range(self.n_bots):
            if b >= self._n_active_bots:
                actions[b] = np.array([price_min, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                continue

            idx = self.n_agents + b  # bots indexed after learning agents
            urgency_denom = urgency_denoms[b] if b < len(urgency_denoms) else 1.5
            action = heuristic_policy.auction_action(
                self.companies[idx], price_ma3, self.current_year,
                self.n_years, self.config,
                bank=float(self.holdings[idx]),
                reserve_price=reserve,
                inflation_factor=infl_factor,
                auction_volume=auction_volume,
                cap_t=cap_t,
                valuation_noise=float(self._bot_valuation_noise[b]),
                urgency_multiplier=float(self._bot_urgency_mult[b]),
                urgency_denom=urgency_denom,
                collateral_load_last=float(self._last_collateral_load[idx]),
            )
            if self._enhanced_noise_enabled and bool(self._bot_budget_stressed[b]):
                action[1] = float(np.clip(action[1] * budget_stress_qty_mult, qty_low, qty_high))
            actions[b] = action
        return actions

    def _generate_bot_secondary_actions(self, clearing_price: float) -> np.ndarray:
        """Generate Phase-2 actions for all bot agents using heuristic_policy."""
        if self.n_bots == 0:
            return np.zeros((0, 2), dtype=np.float32)
        bot_cfg = self.config.get("bots", {})
        urgency_denoms = bot_cfg.get("urgency_denominators", [1.5] * self.n_bots)
        actions = np.zeros((self.n_bots, 2), dtype=np.float32)
        for b in range(self.n_bots):
            if b >= self._n_active_bots:
                actions[b] = np.array([0.0, 0.0], dtype=np.float32)
                continue

            idx = self.n_agents + b  # bots indexed after learning agents
            urgency_denom = urgency_denoms[b] if b < len(urgency_denoms) else 1.5
            actions[b] = heuristic_policy.secondary_action(
                self.companies[idx],
                bank=float(self.holdings[idx]),
                allocation=float(self._phase1_allocations[idx]),
                clearing_price=clearing_price,
                config=self.config,
                current_year=self.current_year,
                n_years=self.n_years,
                valuation_noise=float(self._bot_valuation_noise[b]),
                urgency_multiplier=float(self._bot_urgency_mult[b]),
                urgency_denom=urgency_denom,
            )
        return actions

    def _compute_marginal_ef(self) -> float:
        """
        Compute marginal emission factor for carbon cost pass-through.

        Uses self.companies (active companies only) and self.config.
        Stores result in self._current_marginal_ef before returning.
        """
        active_companies = [c for c in self.companies if self._is_agent_active(c.agent_id)]
        marginal_ef = _compute_marginal_ef(active_companies, self.config)
        self._current_marginal_ef = marginal_ef
        return marginal_ef

    def step_auction(self, auction_actions: np.ndarray):
        """
        Phase 1: Execute auction and green investments.

        Parameters
        ----------
        auction_actions : np.ndarray, shape (n_learning, 6)
            Actions for learning agents only.
            [bid_price, quantity, invest_frac, tech_logit0, tech_logit1, tech_logit2]
            Bot actions are generated internally via heuristic_policy.

        Returns
        -------
        obs_phase2 : np.ndarray, shape (n_learning, obs_dim_phase2)
        year_info : dict
        """
        assert not self.episode_done, "Episode done. Call reset()."

        year = self.current_year
        log = {"year": year}

        # Capture holdings at start of year for diagnostics
        bank_start = self.holdings.copy()
        log["bank_start"] = bank_start.tolist()

        # 1. Cancellation check — before matured investments
        cancellations = np.zeros(self.n_total, dtype=int)
        cancel_recoveries = np.zeros(self.n_total)
        jitter_cfg = self.config.get("construction_jitter", {})
        if jitter_cfg.get("enabled", False):
            for i, company in enumerate(self.companies):
                pre_count = len(company._construction_queue)
                recovered = company.cancel_queued_projects(self.rng)
                post_count = len(company._construction_queue)
                cancellations[i] = pre_count - post_count
                cancel_recoveries[i] = recovered
        self._p6_cancellations = cancellations

        # 2. Apply matured investments + reset annual budget
        for company in self.companies:
            company.apply_matured_investments(year)

        # ---- Revenue-based dynamic budget ----
        budget_mode = self.config.get("budget", {}).get("mode", "fixed")
        if budget_mode == "revenue_based":
            # Smoothed price: MA3 from price history, padded with initial_expected
            init_p = self._price_initial
            hist = list(self._price_history)
            while len(hist) < 3:
                hist.insert(0, init_p)
            smoothed_price = float(np.mean(hist[-3:]))
            carbon_price_for_budget = smoothed_price
            # System-wide average emission factor (also exposed in the observation space)
            active_companies = [c for c in self.companies if self._is_agent_active(c.agent_id)]
            ef_values = [c.weighted_emission_factor for c in active_companies]
            system_ef = float(np.mean(ef_values)) if ef_values else 0.0
            # Marginal EF — EF of most carbon-intensive technology with significant system share.
            # Carbon cost pass-through in electricity markets prices off the marginal setter (typically
            # coal when it has material system presence). Using system_ef understates coal revenue.
            # Reference: Fabra & Reguant (2014), Sijm et al. (2006).
            marginal_ef = self._compute_marginal_ef()
            # Update budgets for all companies (agents + bots)
            for c in active_companies:
                rev_c = c.compute_revenue(marginal_ef, carbon_price_for_budget,
                                          c.inflation_factor(self.current_year))
                c.update_capex_revenue_factor(rev_c)
                c.set_annual_budget(
                    c.compute_dynamic_budget(carbon_price_for_budget, marginal_ef, self.current_year)
                )
                c.settle_treasury_year_end()   # 1st: capture unspent before reset
                c.apply_loan_repayment()       # 2nd: repay from fresh dynamic budget
                c.reset_budget()               # 3rd
                c.reset_capex_budget()         # 4th
            # Store both for logging/obs space
            self._last_system_ef = system_ef
            self._last_marginal_ef = marginal_ef
        else:
            active_companies = [c for c in self.companies if self._is_agent_active(c.agent_id)]
            for company in active_companies:
                company.settle_treasury_year_end()   # 1st: capture unspent before reset
                company.apply_loan_repayment()       # 2nd
                company.reset_budget()               # 3rd
                company.reset_capex_budget()         # 4th

        # 3. Compute TNAC and auction volume
        cap_t = self.cap_schedule.get_cap(year)
        tnac = float(self.holdings.sum())
        price_max = float(self.config["auction"]["price_max"])
        # Pass base penalty rate (not inflation-adjusted) and inflation rate to MSR
        base_penalty_rate = float(self.config["penalty"]["rate"])
        inflation_rate = float(self._inflation_rate(year))
        price_ma3 = self._compute_price_ma3()
        base_auction_volume = self.cap_schedule.get_auction_volume(
            year, tnac, self.last_clearing_price, price_max,
            base_penalty_rate, inflation_rate,
            inflation_factor=self._inflation_factor(year),
            price_ma3=price_ma3,
        )
        auction_volume = base_auction_volume
        log["cap"] = cap_t
        log["tnac"] = tnac

        # Keep unsold and defaulted rollovers as independent accounting streams.
        unsold_rolled_in = float(getattr(self.cap_schedule, "_last_unsold_rollover_in", 0.0))
        msr_withheld = float(getattr(self.cap_schedule, "_last_msr_withheld", 0.0))
        msr_released = float(getattr(self.cap_schedule, "_last_msr_released", 0.0))

        # Add defaulted volume from previous year to this year's supply
        defaulted_rolled_in = 0.0
        if self._defaulted_volume_pending > 0.0:
            if self.config["auction"].get("carry_forward_defaults", True):
                auction_volume += self._defaulted_volume_pending
                defaulted_rolled_in = self._defaulted_volume_pending
            else:
                # Defaulted volume is silently dropped from supply when toggle is
                # off; warn once per env instance so this is not invisible.
                if not self._warned_defaults_dropped:
                    print(
                        f"[warn] auction.carry_forward_defaults=false: "
                        f"{self._defaulted_volume_pending:.3f} Mt of defaulted volume "
                        f"is being dropped from supply at year {year}."
                    )
                    self._warned_defaults_dropped = True
            self._defaulted_volume_pending = 0.0
        log["defaulted_volume_rolled_in"] = round(defaulted_rolled_in, 4)

        # Keep a non-fatal diagnostic: equal values can occur legitimately.
        if (
            unsold_rolled_in > 0.0
            and defaulted_rolled_in > 0.0
            and np.isclose(unsold_rolled_in, defaulted_rolled_in, rtol=0.0, atol=1e-9)
        ):
            log["rollover_channels_equal"] = True

        log["auction_volume"] = auction_volume
        self._last_auction_volume = float(auction_volume)
        self._unsold_rollover = unsold_rolled_in
        log["unsold_rollover_in"] = round(unsold_rolled_in, 4)

        # MSR tracking: reserve level and cumulative cancellations
        msr_reserve_before = self.cap_schedule.msr_reserve()
        log["msr_reserve"] = msr_reserve_before
        log["msr_total_cancelled"] = self.cap_schedule._total_cancelled

        # Track MSR withholding/release applied inside cap_schedule.get_auction_volume.
        log["msr_withhold_this_year"] = round(msr_withheld, 4)
        log["msr_release_this_year"] = round(msr_released, 4)

        # Combine learning agent actions with bot actions after auction supply
        # is known, so bot urgency can react to supply restrictions.
        bot_auc = self._generate_bot_auction_actions(
            auction_volume=auction_volume,
            cap_t=cap_t,
        )
        auction_actions = np.concatenate([auction_actions, bot_auc], axis=0)

        # 4. Generate correlated emission shocks
        # ε_it = ρ × η_t + √(1-ρ²) × ξ_it,  η_t ~ N(0,1),  ξ_it ~ N(0,1)
        unc_cfg = self.config.get("uncertainty", {})
        if unc_cfg.get("enabled", False):
            sigma = unc_cfg.get("sigma_demand", 0.07)
            rho = unc_cfg.get("corr_rho", 0.40)
            eta_common = float(self.rng.normal(0, 1))  # system-wide shock
            idio = self.rng.normal(0, 1, self.n_total)  # idiosyncratic shocks
            epsilons = rho * eta_common + np.sqrt(max(0.0, 1.0 - rho ** 2)) * idio
            epsilons *= sigma
        else:
            epsilons = np.zeros(self.n_total)
        self._current_emission_shocks = epsilons

        # 5. Generate capacity factor noise per tech per agent
        cf_sigma = np.array(jitter_cfg.get("cf_sigma", [0.0, 0.0, 0.08, 0.08, 0.05]))
        cf_noise = np.zeros((self.n_total, 5))
        if jitter_cfg.get("enabled", False):
            for i in range(self.n_total):
                for t in range(5):
                    if cf_sigma[t] > 0:
                        cf_noise[i, t] = float(self.rng.normal(0, cf_sigma[t]))
        self._current_cf_noise = cf_noise

        # 6. Compute realized emissions (capacity-factor noise + demand shock applied)
        realized_emissions = np.zeros(self.n_total)
        for i, company in enumerate(self.companies):
            if not self._is_agent_active(i):
                realized_emissions[i] = 0.0
                continue
            e_cf = company.compute_emissions_with_cf_noise(cf_noise[i])
            e_shocked = e_cf * (1.0 + epsilons[i])
            realized_emissions[i] = max(0.0, e_shocked)
        self._current_emissions = realized_emissions

        # Store aggregate CF shock per agent for logging (mean across green techs)
        cf_shock_agg = np.array([
            float(np.mean(cf_noise[i, [2, 3, 4]]))
            for i in range(self.n_total)
        ])

        # 7. Auction
        bid_actions = auction_actions[:, :2].copy()

        price_min = self.config["auction"]["price_min"]
        price_max = self.config["auction"]["price_max"]

        # Direct bid price: agent action[0] is the bid price in [price_min, price_max]
        bid_actions[:, 0] = np.clip(bid_actions[:, 0], price_min, price_max)

        # Year-over-year bid price change limit (BCL) for learning agents.
        # Active in year 0 too (anchored on max(price_ma3_early, anchor)),
        # so policies can't emit price_max bids that pollute the MA3/AR(1)
        # forecast for the rest of the episode.
        _bcl_cfg = self.config["auction"].get("bid_change_limit", {})
        _bcl_enabled = bool(_bcl_cfg.get("enabled", False))
        _bcl_value = float(_bcl_cfg.get("value", 50.0))
        self._bid_change_limit = _bcl_value if _bcl_enabled else 0.0
        # Compute price_ma3 now so both the clipping reference and the pcl_ceiling
        # obs dimension are consistent (price_ma3 is also re-used below).
        price_ma3_early = self._compute_price_ma3()
        if _bcl_enabled and _bcl_value > 0.0:
            ref = max(price_ma3_early, compute_fundamental_anchor(year, self.config, cap_t_actual=cap_t))
            lo = float(np.clip(ref - _bcl_value, price_min, price_max))
            hi = float(np.clip(ref + _bcl_value, price_min, price_max))
            self._pcl_ceiling = hi
            for _i in range(self.n_agents):
                orig = float(bid_actions[_i, 0])
                clipped = float(np.clip(orig, lo, hi))
                bid_actions[_i, 0] = clipped
                self._last_bid_price_clip[_i] = clipped - orig  # negative if clipped down
        else:
            self._pcl_ceiling = float(price_max)
            self._last_bid_price_clip[:self.n_agents] = 0.0

        # Quantity reparameterization: action[1] is a coverage MULTIPLIER on estimated need.
        # actual_qty = multiplier × compute_estimate_need()
        # This keeps the strategic decision centred on compliance coverage ratio rather
        # than an absolute volume, avoiding the zero-quantity collapse.
        qty_mult_low = self.config["auction"].get("qty_mult_low", 0.3)
        qty_mult_high = self.config["auction"].get("qty_mult_high", 2.0)
        lot_size = self.config["auction"].get("lot_size", 0.0)
        bid_qty_multipliers = np.zeros(self.n_total)
        estimate_needs = np.zeros(self.n_total)
        bid_coverages = np.zeros(self.n_total)
        requested_qtys = np.zeros(self.n_total)  # before leverage/collateral/budget gates
        for i, company in enumerate(self.companies):
            if not self._is_agent_active(i):
                bid_actions[i, 1] = 0.0
                bid_qty_multipliers[i] = 0.0
                estimate_needs[i] = 0.0
                bid_coverages[i] = 0.0
                continue

            multiplier = float(np.clip(auction_actions[i, 1], qty_mult_low, qty_mult_high))
            # Base need is intentionally unbuffered (expected emissions + debt);
            # agents learn safety buffers through the bid multiplier itself.
            base_need = max(company.compute_estimate_need(), 0.1)
            bid_actions[i, 1] = multiplier * base_need
            bid_qty_multipliers[i] = multiplier
            estimate_needs[i] = base_need
            bid_coverages[i] = bid_actions[i, 1] / max(base_need, 1e-6)
            # EU lot-size discretization: round to nearest multiple of lot_size
            if lot_size > 0:
                bid_actions[i, 1] = max(lot_size, round(bid_actions[i, 1] / lot_size) * lot_size)
            requested_qtys[i] = bid_actions[i, 1]  # capture before any gates

        # Leverage gate — clip bid_quantity by leverage_multiplier × available_cash / bid_price.
        # Prevents agents from submitting notional bids far exceeding their cash.
        aq_cfg = self.config["auction"]
        lev_mult = float(aq_cfg.get("leverage_multiplier", 3.0))
        if lev_mult > 0.0:
            for i, company in enumerate(self.companies):
                if not self._is_agent_active(i):
                    continue
                cash = max(0.0, float(company.annual_budget - company.budget_spent_this_year))
                bid_p = float(bid_actions[i, 0])
                if bid_p > 1e-6:
                    max_notional_qty = lev_mult * cash / bid_p
                    if bid_actions[i, 1] > max_notional_qty:
                        bid_actions[i, 1] = max_notional_qty

        # Compute effective reserve price (dynamic or static)
        effective_reserve = self._compute_dynamic_reserve()
        self._last_effective_reserve = effective_reserve
        price_ma3 = self._compute_price_ma3()
        expected_clearing = max(effective_reserve, price_ma3)

        # This clip exists as a training-stability safety net for learning agents during
        # exploration, not as an economic mechanism. The heuristic policy is self-consistent
        # and should not trigger it. If clip events fire for bot-only runs, this indicates a
        # heuristic/env mismatch.
        coll_cfg = self.config.get("auction", {}).get("collateral", {})
        if coll_cfg.get("enabled", True):
            coll_frac = float(coll_cfg.get("collateral_fraction",
                                            coll_cfg.get("opportunity_cost_rate", 0.05)
                                            * coll_cfg.get("hold_fraction", 0.02)))
            max_coll_share = float(coll_cfg.get("max_collateral_budget_share", 0.50))

            if coll_frac > 0.0:
                for i, company in enumerate(self.companies):
                    if not self._is_agent_active(i):
                        continue
                    bid_p = float(bid_actions[i, 0])
                    bid_q = float(bid_actions[i, 1])
                    if bid_q < 1e-6 or bid_p < 1e-6:
                        continue
                    budget_remaining = max(
                        0.0,
                        float(company.annual_budget - company.budget_spent_this_year),
                    )
                    above_clearing = max(0.0, bid_p - expected_clearing)
                    collateral = coll_frac * above_clearing * bid_q
                    max_collateral = max_coll_share * budget_remaining
                    if collateral > max_collateral and max_collateral > 0 and budget_remaining > 1.0:
                        scale = max_collateral / max(collateral, 1e-9)
                        bid_actions[i, 1] *= scale
                        self._collateral_clip_events[i] = self._collateral_clip_events.get(i, 0) + 1

        self._phase1_bid_prices = bid_actions[:, 0].copy()
        self._phase1_bid_quantities = bid_actions[:, 1].copy()  # Mt after multiplier expansion

        # Budget-based gate: agents who cannot cover 10% of bid notional get qty
        # scaled (NOT zeroed) so the bid fits within available cash. Hard zero
        # caused a gradient discontinuity and, combined with the bid_qty_clip_ratio
        # observation feedback, drove policies to under-bid systematically.
        # v8.4.2 bug fix: replace hard zero with a soft scale.
        for i in range(self.n_total):
            if not self._is_agent_active(i):
                continue
            cash = max(0.0, float(self.companies[i].annual_budget
                                  - self.companies[i].budget_spent_this_year)
                       + self.companies[i].get_treasury_available())
            bid_p = float(bid_actions[i, 0])
            bid_q = float(bid_actions[i, 1])
            if bid_p > 1e-6 and bid_q > 1e-6:
                required_cash = bid_p * bid_q * 0.10
                if cash < required_cash:
                    # Scale qty so that bid_p × bid_q × 0.10 == cash
                    max_affordable_qty = cash / (bid_p * 0.10)
                    bid_actions[i, 1] = max(0.0, max_affordable_qty)

        # Record bid qty clip ratios (actual / requested after all gates)
        for i in range(self.n_agents):
            rq = requested_qtys[i]
            aq = float(bid_actions[i, 1])
            self._last_bid_qty_clip_ratio[i] = float(np.clip(aq / max(rq, 1e-6), 0.0, 1.0)) if rq > 1e-6 else 1.0

        # Budget price clip: soft clip at 1.5x max affordable price.
        budget_price_clip = self.config["auction"].get("budget_price_clip", True)
        self._last_budget_price_clip[:] = 0.0
        if budget_price_clip:
            for i, company in enumerate(self.companies):
                if not self._is_agent_active(i):
                    continue
                op_remaining = max(0.0, float(company.annual_budget - company.budget_spent_this_year)
                                   - float(self._collateral_locked[i]))
                cash = op_remaining + company.get_treasury_available()
                cash = max(cash, 1.0)
                bid_q = max(float(bid_actions[i, 1]), 1e-6)
                max_affordable_price = cash / bid_q
                if bid_actions[i, 0] > 1.5 * max_affordable_price:
                    _orig_bid = float(bid_actions[i, 0])
                    _clipped_price = float(np.clip(
                        max_affordable_price,
                        float(self.config["auction"]["price_min"]),
                        float(self.config["auction"]["price_max"]),
                    ))
                    bid_actions[i, 0] = _clipped_price
                    self._last_budget_price_clip[i] = _clipped_price - _orig_bid  # negative if clipped down

        # Pre-bid collateral locking — fraction of margin above reserve.
        # Reduces effective cash available when checking ability to settle payment.
        coll_frac_e4 = float(coll_cfg.get("collateral_fraction",
                                           coll_cfg.get("opportunity_cost_rate", 0.05)
                                           * coll_cfg.get("hold_fraction", 0.02)))
        collateral_locked = np.zeros(self.n_total)
        for i in range(self.n_total):
            if not self._is_agent_active(i):
                continue
            bid_p = float(bid_actions[i, 0])
            bid_q = float(bid_actions[i, 1])
            if bid_p > 1e-6 and bid_q > 1e-6:
                collateral_locked[i] = (
                    coll_frac_e4 * max(0.0, bid_p - effective_reserve) * bid_q
                )

        # Store for Phase 2: collateral opportunity cost uses locked amount directly.
        self._collateral_locked = collateral_locked.copy()

        # Update per-agent collateral load for next year's Phase 1 obs.
        # Tracks collateral_locked / annual_budget so agents can see if they
        # over-committed and learn to avoid bids that risk default/suspension.
        for i in range(self.n_total):
            budget_i = float(self.companies[i].annual_budget)
            self._last_collateral_load[i] = float(np.clip(
                collateral_locked[i] / max(budget_i, 1e-6), 0.0, 1.0,
            ))

        bids = build_bids(bid_actions)

        # Phantom bidder: inject financial intermediary demand before clearing.
        # The phantom uses agent_id = n_total (one beyond all real participants).
        # market_clearing_ets is called with n_agents = n_total + 1 so it
        # pre-allocates the extra slot; the slot is stripped immediately after.
        effective_penalty_rate_for_phantom = (
            float(self.config["penalty"]["rate"]) * self._inflation_factor(self.current_year)
        )
        if self._phantom_bidder.enabled:
            ph_price, ph_qty = self._phantom_bidder.sample_bid(
                price_ma3=price_ma3,
                reserve_price=effective_reserve,
                penalty_rate=effective_penalty_rate_for_phantom,
                auction_supply=auction_volume,
            )
            phantom_row = np.array([[self.n_total, ph_qty, ph_price]], dtype=float)
            bids_for_clearing = np.vstack([bids, phantom_row])
            n_agents_clearing = self.n_total + 1
        else:
            bids_for_clearing = bids
            n_agents_clearing = self.n_total

        clearing_price, all_allocs, all_pays, auction_stats = market_clearing_ets(
            bids=bids_for_clearing,
            q_cap=auction_volume,
            reserve_price=effective_reserve,
            max_agent_share=self.config["auction"].get("max_agent_share", 1.0),
            rng=self.rng,
            cancel_under_subscribed=self.config["auction"].get(
                "cancel_under_subscribed", False),
            n_agents=n_agents_clearing,
            pricing_rule=self.config["auction"].get("pricing_rule", "uniform"),
        )
        # Strip phantom slot — its allocation represents supply consumed by
        # financial traders; it is NOT credited to any holdings array.
        allocations = all_allocs[:self.n_total]
        payments = all_pays[:self.n_total]

        # Log phantom state for year-level diagnostics
        log["phantom_bid_price"] = self._phantom_bidder.last_price
        log["phantom_bid_qty"] = self._phantom_bidder.last_qty
        log["phantom_active"] = self._phantom_bidder.last_active

        # Post-clearing settlement — canonical waterfall: operating → treasury → loan → default.
        loan_cfg = self.config.get("budget", {}).get("emergency_loan", {})
        max_loan_frac = float(loan_cfg.get("max_loan_fraction", 0.15)) if loan_cfg.get("enabled", False) else 0.0
        operating_cash = np.array([
            max(0.0, float(c.annual_budget - c.budget_spent_this_year)
                - float(self._collateral_locked[i]))
            for i, c in enumerate(self.companies)
        ])
        treasury_cash = np.array([c.get_treasury_available() for c in self.companies])
        agent_cash = operating_cash + treasury_cash
        max_loan_budgets = np.array([
            max_loan_frac * max(float(c.annual_budget), 1.0) for c in self.companies
        ])
        (allocations, payments,
         defaults_mask, defaulted_volume,
         _, loan_amounts) = settle_auction(
            allocations=allocations,
            payments=payments,
            agent_cash=agent_cash,
            collateral_locked=np.zeros(self.n_total),  # collateral already netted in operating_cash
            suspension_length=0,
            max_loan_budgets=max_loan_budgets,
        )
        # Post-settlement waterfall (Cases A / B / C); Case D = default (alloc==0, skip)
        for i in range(self.n_total):
            if allocations[i] < 1e-9:
                continue
            payment = float(payments[i])
            op = float(operating_cash[i])
            treas = float(treasury_cash[i])
            if payment <= op:                  # Case A: operating covers it
                self.companies[i].record_spending(payment)
            elif payment <= op + treas:        # Case B: dip into treasury
                self.companies[i].record_spending(op)
                self.companies[i].draw_treasury(payment - op)
            else:                              # Case C: treasury exhausted, take loan
                self.companies[i].record_spending(op)
                self.companies[i].draw_treasury(treas)
                self.companies[i].apply_emergency_loan(loan_amounts[i])
        # Carry forward defaulted volume to next year's q_cap
        if defaulted_volume > 0.0:
            self._defaulted_volume_pending += defaulted_volume
        # Augment auction_stats with default info
        auction_stats["defaults"] = int(defaults_mask.sum())
        auction_stats["defaulted_volume"] = float(defaulted_volume)
        auction_stats["defaults_agents"] = sorted(int(i) for i, d in enumerate(defaults_mask) if d)

        # Unsold allowances (non-default residual): either absorbed into MSR
        # or rolled over to next year's auction. Defaulted volume is tracked
        # separately in _defaulted_volume_pending and must not be double counted.
        unsold = max(0.0, auction_volume - float(allocations.sum()) - float(defaulted_volume))
        self._unsold_rollover = unsold
        log["unsold_rollover_out"] = round(unsold, 4)
        if self.config["ets"].get("unsold_to_msr", False):
            self.cap_schedule.absorb_unsold(unsold)
        else:
            self.cap_schedule.rollover_unsold(unsold)
        self.last_clearing_price = clearing_price
        self._phase1_clearing_price = clearing_price
        log["clearing_price"] = clearing_price
        log["effective_reserve"] = effective_reserve
        log["auction_stats"] = auction_stats
        self._last_cover_ratio = float(auction_stats.get("cover_ratio", 1.0))

        auction_succeeded = not bool(auction_stats.get("auction_failed", False))
        if auction_succeeded:
            self._consecutive_years_without_valid_auction_clear = 0
            if self._reserve_anchor == "auction":
                self._price_history.append(clearing_price)
        else:
            self._consecutive_years_without_valid_auction_clear += 1

        # 8. MAC fuel-switching (based on auction clearing price)
        mac_reductions = np.zeros(self.n_total)
        mac_costs = np.zeros(self.n_total)
        for i, company in enumerate(self.companies):
            if not self._is_agent_active(i):
                continue
            reduction, cost = company.apply_mac_switching(clearing_price, current_year=year)
            mac_reductions[i] = reduction
            mac_costs[i] = cost
        self._current_emissions = np.maximum(self._current_emissions - mac_reductions, 0.0)
        self._mac_reductions = mac_reductions
        self._mac_costs = mac_costs

        # 9. Green investments
        invest_costs = np.zeros(self.n_total)
        invest_fracs = np.zeros(self.n_total)  # actual investment fractions used
        invest_tech_choices = np.zeros(self.n_total, dtype=int)
        budget_cfg = self.config.get("budget", {})
        hard_cap_frac_cfg = float(budget_cfg.get("hard_cap_fraction", 1.15))

        def _estimate_decommission_cost(company: Company, frac_delta: float) -> float:
            frac_to_retire = min(max(frac_delta, 0.0), company.fossil_frac)
            decom_cost = 0.0
            for fossil_idx in [0, 1]:
                if frac_to_retire <= 0.0:
                    break
                retire_this = min(frac_to_retire, float(company.mix[fossil_idx]))
                if retire_this > 1e-6:
                    decom_cost += company.compute_decommission_cost(fossil_idx, retire_this, year)
                    frac_to_retire -= retire_this
            return float(decom_cost)

        def _find_recovered_invest_frac(company: Company, tech_idx: int,
                                        lo_frac: float, hi_frac: float,
                                        max_total_cost: float, max_capex_cost: float) -> float:
            """Bisection solve for max feasible invest_frac under boosted limits."""
            best = lo_frac
            for _ in range(24):
                mid = 0.5 * (lo_frac + hi_frac)
                capex_mid = company.compute_investment_cost(tech_idx, mid, year)
                total_mid = capex_mid + _estimate_decommission_cost(company, mid)
                feasible = (total_mid <= max_total_cost + 1e-9) and (capex_mid <= max_capex_cost + 1e-9)
                if feasible:
                    best = mid
                    lo_frac = mid
                else:
                    hi_frac = mid
            return best

        for i, company in enumerate(self.companies):
            if not self._is_agent_active(i):
                invest_fracs[i] = 0.0
                invest_costs[i] = 0.0
                company.prev_invest_frac = 0.0
                continue

            # Direct physical-space action: action[2] is invest_frac in
            # [0, max_invest_frac] (already scaled by policy/action bounds).
            invest_frac = float(auction_actions[i, 2])
            invest_frac = float(np.clip(invest_frac, 0.0, company.max_invest_frac))
            requested_invest_frac = invest_frac
            tech_logits = np.asarray(auction_actions[i, 3:6], dtype=np.float64)
            # Softmax tech split: distribute invest_frac across the three
            # buildable green technologies according to softmax(logits / T),
            # so each year's investment can diversify across techs (mirrors
            # how a real utility would allocate capex across a portfolio
            # rather than picking one tech per year).
            tech_temp = float(self.config.get("investment", {})
                              .get("tech_softmax_temperature", 1.0))
            tech_temp = max(tech_temp, 1e-6)
            scaled_logits = tech_logits / tech_temp
            scaled_logits -= np.max(scaled_logits)  # numeric stability
            tech_weights = np.exp(scaled_logits)
            tech_weights /= max(tech_weights.sum(), 1e-9)
            # argmax kept for legacy diagnostic logging
            tech_choice = int(np.argmax(tech_weights))
            invest_tech_choices[i] = tech_choice

            tech_indices = [2, 3, 4]  # onshore, offshore, solar in mix index space

            def _capex_for(frac_total: float) -> float:
                """Sum of per-tech capex at frac_total (split by tech_weights)."""
                if frac_total <= 0.0:
                    return 0.0
                cost = 0.0
                for ti, w in zip(tech_indices, tech_weights):
                    sub = float(frac_total) * float(w)
                    if sub > 1e-9:
                        cost += company.compute_investment_cost(ti, sub, year)
                return cost

            budget_ceiling = company.annual_budget * hard_cap_frac_cfg
            budget_remaining = max(0.0, budget_ceiling - company.budget_spent_this_year)
            capex_cost = _capex_for(invest_frac)
            total_proj_cost = capex_cost + _estimate_decommission_cost(company, invest_frac)
            budget_clipped = False
            capex_clipped = False

            # Investment hard gate — block if would exceed hard cap (single
            # source of truth: hard_cap_fraction).
            if budget_cfg.get("investment_hard_gate", True):
                hard_cap_abs = hard_cap_frac_cfg * max(company.annual_budget, 1.0)
                if company.budget_spent_this_year + total_proj_cost > hard_cap_abs:
                    available = max(0.0, hard_cap_abs - company.budget_spent_this_year)
                    if total_proj_cost > 1e-6:
                        scale = available / total_proj_cost
                        invest_frac *= scale
                        capex_cost = _capex_for(invest_frac)
                        total_proj_cost = capex_cost + _estimate_decommission_cost(company, invest_frac)
                        budget_clipped = True

            if total_proj_cost > budget_remaining and total_proj_cost > 1e-6:
                invest_frac *= budget_remaining / total_proj_cost
                capex_cost = _capex_for(invest_frac)
                budget_clipped = True

            capex_remaining = max(0.0, company.effective_capex_throughput - company.capex_spent_this_year)
            if capex_cost > capex_remaining and capex_cost > 1e-6:
                invest_frac *= capex_remaining / capex_cost
                capex_clipped = True

            # Optional green-finance boost can recover clipped investment.
            # The boost solves a single-tech bisection on the dominant tech as
            # a cheap proxy; the fine-grained per-tech split happens after.
            if (budget_clipped or capex_clipped) and company._gf_enabled and requested_invest_frac > invest_frac:
                max_total_with_loan = budget_remaining + company.green_loan_headroom
                max_capex_with_boost = capex_remaining + company.green_capex_headroom
                # Use the dominant tech for the bisection; result then drives
                # the per-tech split below proportionally.
                dominant_tech_idx = tech_indices[tech_choice]
                recovered_frac = _find_recovered_invest_frac(
                    company=company,
                    tech_idx=dominant_tech_idx,
                    lo_frac=invest_frac,
                    hi_frac=requested_invest_frac,
                    max_total_cost=max_total_with_loan,
                    max_capex_cost=max_capex_with_boost,
                )
                if recovered_frac > invest_frac + 1e-9:
                    recovered_total = _capex_for(recovered_frac)
                    recovered_total += _estimate_decommission_cost(company, recovered_frac)
                    extra_invest_cost = max(0.0, recovered_total - budget_remaining)
                    company.record_green_loan(extra_invest_cost)
                    invest_frac = recovered_frac

            invest_frac = float(np.clip(invest_frac, 0.0, company.max_invest_frac))
            invest_frac = min(invest_frac, company.fossil_frac)
            invest_fracs[i] = invest_frac
            self._last_invest_clip_ratio[i] = float(np.clip(
                invest_frac / max(requested_invest_frac, 1e-6), 0.0, 1.0,
            )) if requested_invest_frac > 1e-6 else 1.0

            # Issue one plan_investment per tech with non-trivial weight.
            total_cost_i = 0.0
            for tw_idx, w in enumerate(tech_weights):
                sub_frac = invest_frac * float(w)
                if sub_frac < 1e-9:
                    continue
                # plan_investment expects buildable index 0..2 (= tech_indices - 2)
                total_cost_i += company.plan_investment(tw_idx, sub_frac, year)
            invest_costs[i] = total_cost_i
            company.prev_invest_frac = invest_frac
            # Apply any cancellation recovery as a credit to invest_costs
            invest_costs[i] -= cancel_recoveries[i]

        # Store for phase 2
        self._phase1_allocations = allocations
        self._phase1_payments = payments
        self._phase1_invest_costs = invest_costs
        self._phase1_mac_costs = mac_costs
        self._phase1_log = log

        # Update per-agent bank cost basis with newly acquired auction allowances.
        # Weighted-average: (old_basis × old_bank + clearing_price × alloc) / new_bank.
        if clearing_price > 0.0:
            for _i in range(self.n_total):
                _alloc_i = float(allocations[_i])
                if _alloc_i < 1e-9:
                    continue
                _old_bank = max(float(self.holdings[_i]), 0.0)
                _new_bank = _old_bank + _alloc_i
                self._bank_cost_basis[_i] = (
                    (self._bank_cost_basis[_i] * _old_bank + clearing_price * _alloc_i)
                    / _new_bank
                )

        # Compute bid affordability for next year's observation
        bid_prices = self._phase1_bid_prices
        bid_qtys = self._phase1_bid_quantities
        coll_frac_c2 = float(coll_cfg.get("collateral_fraction",
                                           coll_cfg.get("opportunity_cost_rate", 0.05)
                                           * coll_cfg.get("hold_fraction", 0.02)))
        for i in range(self.n_total):
            c = self.companies[i]
            budget_remaining = max(c.annual_budget - c.budget_spent_this_year, 1e-6)
            bid_total = bid_prices[i] * bid_qtys[i] * (1.0 + coll_frac_c2)
            self._bid_affordability[i] = float(np.clip(bid_total / budget_remaining, 0.0, 1.0))

        # 10. Build phase 2 observations (learning agents only)
        obs_phase1 = self._get_obs_phase1()   # shape (n_agents, obs_dim)

        obs_phase2 = np.stack([
            self.companies[i].get_observation_phase2(
                obs_phase1=obs_phase1[i],
                allocation=allocations[i],
                clearing_price=clearing_price,
                emissions=realized_emissions[i],
                banked=self.holdings[i],
                emission_shock=float(epsilons[i]),
                payment=float(payments[i]),           # for auction_savings dim
                collateral_locked_norm=float(np.clip(
                    self._collateral_locked[i] / max(self.companies[i].annual_budget, 1e-6),
                    0.0, 1.0,
                )),
                current_holdings=float(self.holdings[i] + allocations[i]),
                current_year=year,
                last_sec_qty_clip_ratio=float(self._last_sec_qty_clip_ratio[i]),
            )
            for i in range(self.n_agents)
        ]) if self.n_agents > 0 else np.zeros((0,), dtype=np.float32)

        # Log shock values for year-level diagnostics
        log["emission_shocks"] = epsilons.tolist()
        log["cf_shocks"] = cf_shock_agg.tolist()
        log["cancellations"] = cancellations.tolist()
        log["mac_reductions"] = mac_reductions.tolist()
        log["mac_costs"] = mac_costs.tolist()
        log["bid_quantities"] = self._phase1_bid_quantities.tolist()  # Mt per agent
        log["invest_fracs"] = invest_fracs.tolist()                   # actual executed invest_frac per agent
        log["bid_qty_multipliers"] = bid_qty_multipliers.tolist()     # raw multiplier action
        log["estimate_needs"] = estimate_needs.tolist()               # Mt before multiplier
        log["bid_coverages"] = bid_coverages.tolist()                 # bid_qty / est_need
        log["invest_tech_choices"] = invest_tech_choices.tolist()     # 0=onshore,1=offshore,2=solar
        log["bid_to_reserve_ratio"] = (
            self._phase1_bid_prices / max(self._last_effective_reserve, 1e-6)
        ).tolist()

        # ── Auction-phase warning counters ────────────────────────────────────
        _reserve = self._last_effective_reserve
        _price_max = self.config["auction"]["price_max"]
        if clearing_price <= _reserve + 1.0:
            self._warnings["price_floor"] += 1
        if clearing_price >= 0.9 * _price_max:
            self._warnings["price_ceil"] += 1
        if auction_stats.get("auction_failed", False):
            self._warnings["auct_fail"] += 1
        if auction_stats.get("total_demand", 0.0) < 0.7 * auction_volume:
            self._warnings["low_demand"] += 1
        if float(allocations.sum()) < 0.3 * auction_volume:
            self._warnings["low_alloc"] += 1
        if np.std(self._phase1_bid_prices) < 5.0:
            self._warnings["bid_cluster"] += 1
        if np.all(invest_fracs < 0.001):
            self._warnings["no_invest"] += 1
        if tnac > 2.0 * float(self._current_emissions.sum()):
            self._warnings["over_bank"] += 1
        # Cornering warning: agent's allocation share > 2× its emissions share
        total_alloc = float(allocations.sum())
        total_emiss = float(self._current_emissions.sum())
        if total_alloc > 0.3 * auction_volume and total_emiss > 1e-9:
            for _ci in range(self.n_total):
                alloc_share = float(allocations[_ci]) / total_alloc
                need_share = float(self._current_emissions[_ci]) / total_emiss
                if alloc_share > 2.0 * need_share and alloc_share > 0.30:
                    self._warnings["cornering"] += 1
                    break  # count once per year-step
        # Reserve-price rejection: fire when rejected volume > 25% of total bid volume
        _below_mask = self._phase1_bid_prices < _reserve - 1e-6
        _rejected_vol = float(self._phase1_bid_quantities[_below_mask].sum())
        _total_bid_vol = float(self._phase1_bid_quantities.sum())
        if _total_bid_vol > 1e-9 and _rejected_vol > 0.25 * _total_bid_vol:
            self._warnings["rsv_reject"] += 1

        return obs_phase2, log

    # ------------------------------------------------------------------
    # Split Rewards: Auction-phase intermediate reward
    # ------------------------------------------------------------------

    def compute_auction_rewards(self):
        """
        Compute per-agent intermediate reward for the auction phase.

        Returns
        -------
        r_auction : np.ndarray, shape (n_agents,)
            Total auction-phase reward (= bid + invest stream).
        r_auction_bid : np.ndarray, shape (n_agents,)
            Bid sub-head reward (compliance + coverage-gap penalty only).
        r_auction_invest : np.ndarray, shape (n_agents,)
            Investment sub-head reward (capital cost only).

        ``r_auction = r_auction_bid + r_auction_invest`` so the ``r_auction``
        return value is identical to the pre-v8.5 single-stream value, while
        the split components let the train loop route the bid and investment
        sub-heads to their own advantage streams.
        """
        r_auction = np.zeros(self.n_agents)
        r_auction_bid = np.zeros(self.n_agents)
        r_auction_invest = np.zeros(self.n_agents)
        _cap_t_now = float(self.cap_schedule.get_cap(self.current_year))
        anchor_t  = compute_fundamental_anchor(self.current_year, self.config, cap_t_actual=_cap_t_now)

        for i in range(self.n_agents):
            company = self.companies[i]
            infl    = company.inflation_factor(self.current_year)

            budget_real      = max(company.annual_budget / infl, 1.0)
            anchor_real      = anchor_t / infl
            need             = max(company.compute_estimate_need(), 1e-6)
            compliance_denom = max(anchor_real * need, 1.0)

            auction_cost      = float(self._phase1_payments[i]) / infl
            invest_cost       = float(self._phase1_invest_costs[i]) / infl
            # Real-terms OPEX delta vs initial-mix baseline. Deflate first,
            # then subtract baseline_opex (already real, since infl(0)=1) so
            # the delta is zero when the mix is unchanged regardless of year.
            opex_delta        = (company.compute_operational_cost(self.current_year) / infl
                                 - company.baseline_opex)
            mac_cost_i        = float(self._phase1_mac_costs[i]) / infl
            collateral_cost_i = (float(self._collateral_locked[i]) *
                                 float(self.config["auction"]["collateral"]
                                       .get("collateral_rate", 0.05))) / infl

            compliance_norm = (auction_cost + mac_cost_i + collateral_cost_i) / compliance_denom
            capital_norm    = (invest_cost + opex_delta) / budget_real

            # baseline_cost — fair-price reference. Subtracting `need ×
            # clearing_price` lets the bid head see a deviation-from-fair
            # signal: buying exactly `need` at the clearing price → ≈0,
            # over-buying → small positive cost, under-buying → small
            # "saving" but `gap_penalty` (priced at the remediation rate,
            # which is several times the clearing price) dominates.
            clearing_price_nom = float(self.last_clearing_price)
            baseline_cost = (need * clearing_price_nom) / max(infl, 1e-9)
            compliance_norm_excess = (
                (auction_cost + mac_cost_i + collateral_cost_i - baseline_cost)
                / compliance_denom
            )

            coverage_gap = max(0.0, need - float(self._phase1_allocations[i]))
            # Coverage-gap rate = expected cost of remediation per missing Mt.
            # Floor: max(eff_pen, sec_ema, anchor). Cap: cap_mult × eff_pen.
            # When `sec_proxy.enabled=false`, fall back to bare eff_pen / infl
            # (legacy v8.5.0 behaviour) — the rolling+capped proxy is the
            # only place these knobs are read.
            eff_pen_rate_nom = company.effective_penalty_rate(self.current_year)
            if self._sec_proxy_enabled:
                sec_ema_nom = self._sec_price_ema  # None until first sec clear
                sec_proxy_nom = (
                    float(sec_ema_nom) if sec_ema_nom is not None else float(anchor_t)
                )
                remediation_floor_nom = max(eff_pen_rate_nom, sec_proxy_nom, anchor_t)
                remediation_cap_nom   = self._sec_proxy_cap_mult * eff_pen_rate_nom
                effective_remediation_nom = min(remediation_floor_nom, remediation_cap_nom)
            else:
                sec_proxy_nom = float(anchor_t)
                effective_remediation_nom = eff_pen_rate_nom
            expected_remediation_rate_real = effective_remediation_nom / max(infl, 1e-9)
            gap_penalty = (coverage_gap * expected_remediation_rate_real) / compliance_denom

            # Bid sub-head: fair-price-adjusted compliance + coverage gap.
            r_auction_bid[i] = -(compliance_norm_excess) - gap_penalty
            # Investment sub-head: capital costs only.
            r_auction_invest[i] = -capital_norm
            r_auction[i] = r_auction_bid[i] + r_auction_invest[i]

            self._last_auction_reward_channels[i] = {
                "auction_cost":         float(auction_cost),
                "baseline_cost":        float(baseline_cost),
                "collateral_cost":      float(collateral_cost_i),
                "investment_cost":      float(invest_cost),
                "opex_delta":           float(opex_delta),
                "mac_cost":             float(mac_cost_i),
                "compliance_norm":      float(compliance_norm),         # legacy diag (full auction_cost)
                "compliance_norm_excess": float(compliance_norm_excess), # actual reward signal
                "capital_norm":         float(capital_norm),
                "coverage_gap_penalty": float(gap_penalty),
                "expected_remediation_rate_real": float(expected_remediation_rate_real),
                "sec_price_ema":        float(sec_proxy_nom),
                "r_bid":                float(r_auction_bid[i]),
                "r_invest":             float(r_auction_invest[i]),
            }
        return r_auction, r_auction_bid, r_auction_invest

    # ------------------------------------------------------------------
    # Phase 2: Secondary Market + Compliance + Rewards
    # ------------------------------------------------------------------

    def step_secondary(self, secondary_actions: np.ndarray):
        """
        Phase 2: Execute secondary market, compliance, and rewards.

        Parameters
        ----------
        secondary_actions : np.ndarray, shape (n_learning, 2)
            Actions for learning agents only.
            [price_abs (EUR/t), quantity] per agent.
            Bot actions are generated internally via heuristic_policy.

        Returns
        -------
        obs_next (n_learning,), rewards (n_learning,), terminated, truncated, info
        """
        # Combine learning agent actions with bot actions
        bot_sec = self._generate_bot_secondary_actions(self._phase1_clearing_price)
        secondary_actions = np.concatenate([secondary_actions, bot_sec], axis=0)

        allocations = self._phase1_allocations
        payments = self._phase1_payments
        invest_costs = self._phase1_invest_costs
        mac_costs = self._phase1_mac_costs
        clearing_price = self._phase1_clearing_price
        log = self._phase1_log

        # 5. Secondary market (absolute prices)
        trading_cfg = self.config.get("trading", {})
        sec_price_min = trading_cfg.get("sec_price_min", 30.0)
        sec_price_max_mult = trading_cfg.get("sec_price_max_mult", 2.0)

        realized_emissions = self._current_emissions
        old_carry_forward = np.array([c._carry_forward for c in self.companies], dtype=float)
        active_mask = self._active_mask()

        # Per-agent secondary price cap = 2.0 × effective_penalty_rate
        secondary_prices = np.zeros(self.n_total)
        for i in range(self.n_total):
            if not active_mask[i]:
                secondary_prices[i] = float(sec_price_min)
                continue
            agent_sec_max = sec_price_max_mult * self.companies[i].effective_penalty_rate(self.current_year)
            secondary_prices[i] = float(np.clip(
                secondary_actions[i, 0], sec_price_min, agent_sec_max))

        raw_secondary_qtys = np.clip(
            secondary_actions[:, 1],
            -self.config["auction"]["quantity_max"],
            self.config["auction"]["quantity_max"],
        )
        raw_secondary_qtys[~active_mask] = 0.0

        # No sell gating — agents sell whatever they choose, up to what they hold
        # (no short selling enforced in _settle_double_auction via max_sell = alloc + bank)
        secondary_qtys = raw_secondary_qtys.copy()

        # Pre-trade resources for compliance tracking
        pretrade_holdings = self.holdings + allocations

        trade_costs, trade_qtys, secondary_clearing, secondary_volume, liquidity_pool_info = \
            self._settle_double_auction(
                allocations=allocations.copy(),
                secondary_prices=secondary_prices,
                secondary_qtys=secondary_qtys,
                clearing_price=clearing_price,
            )

        self.last_secondary_price = secondary_clearing
        self.last_secondary_volume = secondary_volume

        # Update per-episode EMA of secondary clearing (feeds the bid-head
        # remediation-rate proxy). Only update on real volume so no-trade
        # years (where secondary_clearing == auction clearing_price) don't
        # contaminate the EMA with an unrealised auction price.
        if self._sec_proxy_enabled and secondary_volume > 0.0 and secondary_clearing > 0.0:
            if self._sec_price_ema is None:
                self._sec_price_ema = float(secondary_clearing)
            else:
                a = self._sec_proxy_ema_alpha
                self._sec_price_ema = (
                    a * float(secondary_clearing) + (1.0 - a) * float(self._sec_price_ema)
                )

        # Record secondary qty clip ratio for learning agents' obs dim
        for _i in range(self.n_agents):
            requested_sq = float(secondary_qtys[_i])
            actual_sq = float(trade_qtys[_i])
            if abs(requested_sq) > 1e-6:
                self._last_sec_qty_clip_ratio[_i] = float(np.clip(
                    actual_sq / requested_sq, -1.0, 1.0,
                ))
            else:
                self._last_sec_qty_clip_ratio[_i] = 1.0

        # Update bank cost basis for secondary purchases (buyers only).
        if secondary_clearing > 0.0:
            for _i in range(self.n_total):
                _sec_qty = float(trade_qtys[_i])
                if _sec_qty < 1e-9:
                    continue
                _sec_price = float(trade_costs[_i]) / _sec_qty
                _old_bank = max(float(self.holdings[_i]) + float(allocations[_i]), 0.0)
                _new_bank = _old_bank + _sec_qty
                self._bank_cost_basis[_i] = (
                    (self._bank_cost_basis[_i] * _old_bank + _sec_price * _sec_qty)
                    / _new_bank
                )

        # Update secondary profit EMA
        for i in range(self.n_total):
            if not active_mask[i]:
                self._secondary_profit_ema[i] = 0.0
                continue
            if trade_qtys[i] < -1e-6:
                profit_per_mt = -trade_costs[i] / abs(trade_qtys[i])
                margin = profit_per_mt - clearing_price
                self._secondary_profit_ema[i] = (
                    self._ema_alpha * margin +
                    (1 - self._ema_alpha) * self._secondary_profit_ema[i]
                )
            elif trade_qtys[i] > 1e-6:
                cost_per_mt = trade_costs[i] / trade_qtys[i]
                margin = clearing_price - cost_per_mt
                self._secondary_profit_ema[i] = (
                    self._ema_alpha * margin +
                    (1 - self._ema_alpha) * self._secondary_profit_ema[i]
                )
                # Record what this agent actually paid on secondary (feeds WTP anchor)
                self._last_secondary_buy_price[i] = cost_per_mt

        # Holdings after secondary market
        holdings = self.holdings + allocations + trade_qtys

        # Collateral cost: rate × locked collateral from Phase 1.
        # self._collateral_locked is computed in step_auction() as:
        #   collateral_fraction × max(0, bid_price − reserve) × bid_qty
        # Charging rate × locked_amount is equivalent to the financing cost of
        # tying up margin capital for the settlement period.
        collateral_costs = np.zeros(self.n_total)
        collateral_cfg = self.config.get("auction", {}).get("collateral", {})
        if collateral_cfg.get("enabled", False):
            rate = float(collateral_cfg.get("collateral_rate", 0.05))
            collateral_costs = rate * self._collateral_locked
            collateral_costs[~active_mask] = 0.0

        # 6. Compliance (against realized emissions + carry-forward obligations).
        # `realized_emissions` and `old_carry_forward` were captured earlier in
        # this method, before settlement updates the carry-forward state.
        penalties = np.zeros(self.n_total)
        for i, company in enumerate(self.companies):
            if not active_mask[i]:
                penalties[i] = 0.0
                company._carry_forward = 0.0
                continue
            penalties[i] = company.settle_compliance_realized(
                allowances_held=holdings[i],
                realized_emissions=realized_emissions[i],
                current_year=self.current_year,
            )

        # Banking: surplus after surrendering for emissions + old carry-forward
        for i in range(self.n_total):
            if not active_mask[i]:
                self.holdings[i] = 0.0
                continue
            total_obligation = realized_emissions[i] + old_carry_forward[i]
            self.holdings[i] = max(0.0, holdings[i] - total_obligation)
        self._last_gaps = self.holdings.copy()

        # Update fossil fraction history (for lock-in penalty)
        for i, company in enumerate(self.companies):
            if not active_mask[i]:
                self._fossil_frac_history[i] = []
                continue
            self._fossil_frac_history[i].append(company.fossil_frac)
            if len(self._fossil_frac_history[i]) > 3:
                self._fossil_frac_history[i].pop(0)

        # Update cumulative allocation and emission trackers for obs.
        for i in range(self.n_total):
            self._cumulative_alloc[i] += float(allocations[i])
            self._cumulative_emissions[i] += float(realized_emissions[i])

        # 7. Rewards (raw — normalisation happens in PPOAgent)
        rewards = self._compute_rewards(
            payments,
            trade_costs,
            penalties,
            invest_costs,
            realized_emissions,
            clearing_price,
            mac_costs,
            collateral_costs=collateral_costs,
            precompliance_holdings=pretrade_holdings,
            old_carry_forward=old_carry_forward,
            active_mask=active_mask,
            trade_qtys=trade_qtys,
            allocations=allocations,
        )

        # Compute per-agent shortfall for diagnostics.
        # Total obligation = realized emissions + carry-forward from prior years.
        # Uses local `holdings` (pre-compliance: prev_bank + alloc + secondary).
        shortfalls = np.array([
            (max(0.0, realized_emissions[i] + old_carry_forward[i] - holdings[i]) if active_mask[i] else 0.0)
            for i in range(self.n_total)
        ])

        # ── Secondary-phase warning counters ─────────────────────────────────
        if secondary_volume < 0.01:
            self._warnings["no_trade"] += 1
        sec_qty_raw = secondary_actions[:, 1]
        if np.all(sec_qty_raw > 0) or np.all(sec_qty_raw < 0):
            self._warnings["one_side_sec"] += 1
        for _i in range(self.n_total):
            if not active_mask[_i]:
                self._consecutive_shortfall[_i] = 0
                continue
            if shortfalls[_i] > 1e-6:
                self._consecutive_shortfall[_i] += 1
                if self._consecutive_shortfall[_i] >= 3 and self.current_year >= 3:
                    self._warnings["debt_spiral"] += 1
                    self._consecutive_shortfall[_i] = 0
            else:
                self._consecutive_shortfall[_i] = 0

        # 8. Log
        sec_action_sides = np.where(
            secondary_actions[:, 1] > 1e-6, 1,
            np.where(secondary_actions[:, 1] < -1e-6, -1, 0)
        )
        log.update({
            "allocations": allocations.tolist(),
            "payments": payments.tolist(),
            "emissions": realized_emissions.tolist(),
            "trade_costs": trade_costs.tolist(),
            "trade_qtys": trade_qtys.tolist(),
            "collateral_costs": collateral_costs.tolist(),
            "secondary_clearing": secondary_clearing,
            "secondary_volume": secondary_volume,
            "raw_secondary_qtys": raw_secondary_qtys.tolist(),
            "gated_secondary_qtys": secondary_qtys.tolist(),
            "penalties": penalties.tolist(),
            "invest_costs": invest_costs.tolist(),
            "rewards": rewards.tolist(),
            "rewards_base": self._last_reward_base_values.tolist(),
            "rewards_shaping": self._last_reward_shaping_values.tolist(),
            "inflation_rate": self._inflation_rate(self.current_year),
            "inflation_factor": self._inflation_factor(self.current_year),
            "green_fracs": [c.green_frac for c in self.companies],
            "tech_mixes": [c.mix.tolist() for c in self.companies],
            "holdings": self.holdings.tolist(),
            "shortfalls": shortfalls.tolist(),
            "bid_prices": self._phase1_bid_prices.tolist() if self._phase1_bid_prices is not None else [],
            "delta_greens": [c.green_frac - c.prev_green_frac for c in self.companies],
            "queue_sizes": [len(c._construction_queue) for c in self.companies],
            "terminal_bank_values": self._last_terminal_bank_values.tolist(),
            "terminal_queue_values": self._last_terminal_queue_values.tolist(),
            "terminal_liquidation_values": self._last_terminal_liquidation_values.tolist(),
            "mac_reductions": self._mac_reductions.tolist(),
            "mac_costs": self._mac_costs.tolist(),
            "sec_price_mults": secondary_prices.tolist(),  # Phase 2 absolute prices (clipped)
            "sec_qty_actions": secondary_actions[:, 1].tolist(),  # Phase 2 action[1] (raw)
            "sec_action_sides": sec_action_sides.tolist(),         # -1=sell, 0=hold, 1=buy intent
            "liquidity_pool": liquidity_pool_info,
            "friction_costs": [
                float(payments[i]) + float(collateral_costs[i]) + float(self._mac_costs[i])
                for i in range(self.n_total)
            ],
            "opex_savings": [
                -(self.companies[i].compute_operational_cost(self.current_year)
                  - self.companies[i].baseline_opex)
                for i in range(self.n_total)
            ],
            "compliance_costs": [
                float(payments[i]) + float(trade_costs[i])
                for i in range(self.n_total)
            ],
            "investment_costs": invest_costs.tolist(),
            "old_carry_forward": old_carry_forward.tolist(),  # carry-forward debt at year start (pre-compliance)
        })

        # ── Per-agent per-year diagnostics ───────────────────────────────────
        # Diagnostic fields for validation (Run 1/2 from validation sequence).
        price_ma3_now = self._compute_price_ma3()
        per_agent_diag = {}
        for i, company in enumerate(self.companies):
            if not active_mask[i]:
                continue
            annual_need_i = max(company.compute_estimate_need(), 1e-6)
            inf_i = company.inflation_factor(self.current_year)
            revenue_i = company.compute_revenue(self._last_marginal_ef, price_ma3_now, inf_i)
            compliance_cost_i = float(payments[i]) + float(trade_costs[i])
            coverage_post = float(self.holdings[i]) / annual_need_i
            per_agent_diag[i] = {
                "wtp": float(getattr(company, "_last_wtp", float("nan"))),
                "wtp_economic": float(getattr(company, "_last_wtp_economic", float("nan"))),
                "wtp_budget": float(getattr(company, "_last_wtp_budget", float("nan"))),
                "wtp_binding": str(getattr(company, "_last_wtp_binding", "")),
                "bid_price": float(getattr(company, "_last_bid_price_heuristic", float("nan"))),
                "bid_qty": float(self._phase1_bid_quantities[i]) if self._phase1_bid_quantities is not None else float("nan"),
                "qty_target": float(getattr(company, "_last_qty_target", float("nan"))),
                "expected_clearing_ma3": float(price_ma3_now),
                "actual_clearing": float(clearing_price),
                "actual_pay": float(payments[i]),
                "coverage_ratio_post_compliance": float(coverage_post),
                "marginal_ef": float(self._last_marginal_ef),
                "system_ef": float(self._last_system_ef),
                "revenue": float(revenue_i),
                "compliance_cost_share_of_budget": float(compliance_cost_i) / max(float(company.annual_budget), 1e-6),
                "invest_frac_pre_compliance_clip": float(getattr(company, "_last_invest_frac_pre_compliance_clip", float("nan"))),
                "invest_frac_post_compliance_clip": float(getattr(company, "_last_invest_frac_post_compliance_clip", float("nan"))),
                "available_budget": float(company.annual_budget - company.budget_spent_this_year),
            }
        self._last_per_agent_diag = per_agent_diag
        log["per_agent_diag"] = per_agent_diag
        log["marginal_ef_used"] = float(self._last_marginal_ef)
        # Store reward channels in year log for diagnostics (esg_vs_penalty_ratio, etc.)
        log["reward_channels"] = {i: dict(ch) for i, ch in self._last_reward_channels.items()}

        self.episode_log.append(log)

        for company in self.companies:
            company.prev_green_frac = company.green_frac

        # 9. Advance year + AR(1) price
        if self._reserve_anchor == "secondary":
            self._price_history.append(secondary_clearing)
        rho = self.config["price"].get("ar1_persistence", 0.85)
        next_year = min(self.current_year + 1, self.n_years - 1)
        _cap_next = float(self.cap_schedule.get_cap(next_year))
        price_floor = compute_fundamental_anchor(next_year, self.config, cap_t_actual=_cap_next)
        vol_std = self.config["price"].get("volatility_std", 0.15)

        # Only update the AR(1) expected-price forecast from a meaningful (successful)
        # auction clearing price.  If the auction failed, clearing_price equals the
        # reserve price floor, which would snap expected_price to price_floor and give
        # agents a misleading signal; in that case we keep the previous forecast.
        auction_succeeded_this_step = not bool(
            log.get("auction_stats", {}).get("auction_failed", False)
        )
        if auction_succeeded_this_step and clearing_price > 0:
            shock = self.rng.normal(0, vol_std) * clearing_price
            self.expected_price = max(
                rho * clearing_price + (1.0 - rho) * price_floor + shock,
                price_floor,
            )

        # Year 1 TNAC diagnostic: run once per episode after first full year settles.
        # Keep these values in logs for offline checks; warning emission is opt-in
        # to avoid noisy test output under stochastic/random-policy runs.
        if self.current_year == 0:
            diag_cfg = self.config.get("diagnostics", {})
            year1_tnac = float(np.sum(self.holdings))
            year1_low = float(diag_cfg.get("year1_tnac_expected_low", 1.0))
            year1_high = float(diag_cfg.get("year1_tnac_expected_high", 8.0))
            warn_year1_tnac = bool(diag_cfg.get("warn_year1_tnac_out_of_range", False))
            in_band = (year1_low <= year1_tnac <= year1_high)
            log["year1_tnac"] = year1_tnac
            log["year1_tnac_expected_low"] = year1_low
            log["year1_tnac_expected_high"] = year1_high
            log["year1_tnac_in_range"] = bool(in_band)
            if warn_year1_tnac and (not in_band) and (not self._year1_tnac_warning_emitted):
                warnings.warn(
                    f"[ETSEnvironment] Year 1 TNAC diagnostic out of expected range "
                    f"[{year1_low:.1f}, {year1_high:.1f}] Mt: observed {year1_tnac:.2f} Mt.",
                    stacklevel=2,
                )
                self._year1_tnac_warning_emitted = True

        self.current_year += 1
        terminated = self.current_year >= self.n_years
        self.episode_done = terminated
        if terminated:
            years_completed = max(self.current_year, 1)
            log["collateral_clip_events_episode"] = dict(self._collateral_clip_events)
            log["collateral_clip_rate_episode"] = {
                i: float(self._collateral_clip_events.get(i, 0)) / float(years_completed)
                for i in range(self.n_total)
            }

        # ── Per-year sec tracking + compliance gaps (must populate BEFORE snapshot write)
        for _si in range(self.n_total):
            if trade_qtys[_si] > 1e-6:
                self._sec_bought[_si] += float(trade_qtys[_si])
            elif trade_qtys[_si] < -1e-6:
                self._sec_sold[_si] += float(abs(trade_qtys[_si]))
            # compliance gap = emissions - surrendered; surrendered = pre - post compliance holdings
            self._last_compliance_gaps[_si] = float(
                realized_emissions[_si] - max(0.0, holdings[_si] - self.holdings[_si])
            )
        # ── Opponent snapshot two-buffer update (_prev MUST be assigned before current is overwritten)
        self._opponent_snapshots_prev = self._opponent_snapshots.copy()   # save year t-1
        opp_cfg = self.config.get("opponent_obs", {})
        queue_sigma = float(opp_cfg.get("queue_noise_sigma", 0.15))
        # Total holdings across all participants (publicly inferable from
        # aggregate TNAC reports). Used to convert per-firm holdings into a
        # market-share signal that respects EU ETS confidentiality rules.
        _total_holdings = float(self.holdings.sum())
        for _si, _sc in enumerate(self.companies):
            need_i = max(_sc.compute_estimate_need(), 1e-6)
            queue_raw = float(sum(item["frac_delta"] for item in _sc._construction_queue))
            queue_noisy = float(np.clip(queue_raw + self.rng.normal(0, queue_sigma), 0.0, 1.0))
            tnac_share = float(np.clip(self.holdings[_si] / max(_total_holdings, 1e-6), 0.0, 1.0))
            net_sec = float(np.clip(
                (self._sec_bought[_si] - self._sec_sold[_si]) / need_i, -1.0, 1.0
            ))
            lag_gap = float(np.clip(self._last_compliance_gaps[_si] / need_i, -1.0, 1.0))
            self._opponent_snapshots[_si] = [
                _sc.compute_emissions() / 10.0,
                _sc.green_frac, _sc.fossil_frac,
                queue_noisy, tnac_share, net_sec, lag_gap,
            ]
        # Reset per-year secondary counters for next year
        self._sec_bought[:] = 0.0
        self._sec_sold[:]   = 0.0

        log.update({
            "treasury_reserves":          [c._treasury_reserve for c in self.companies],
            "treasury_drawn":             [c._treasury_drawn_this_year for c in self.companies],
            "loan_outstanding":           [c._loan_outstanding for c in self.companies],
            "effective_capex_throughput": [c.effective_capex_throughput for c in self.companies],
            "anchor_t":                   self._last_reward_channels.get(0, {}).get("anchor_t"),
            "opponent_snapshots":         self._opponent_snapshots.tolist(),
        })

        obs_next = self._get_obs_phase1()   # shape (n_agents, obs_dim)
        # Compute per-agent diagnostic scores and expose via info dict
        try:
            diag_scores = self.compute_diagnostic_score()
        except Exception:
            diag_scores = []
        return obs_next, rewards[:self.n_agents], terminated, False, {
            "year_log": log,
            "diagnostic_scores": diag_scores,
        }

    # ------------------------------------------------------------------
    # Legacy step (calls both phases — for testing)
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray):
        """Single-call step for backward compat. actions shape (n_learning, 8)."""
        obs2, _ = self.step_auction(actions[:, :6])
        return self.step_secondary(actions[:, 6:])

    # ------------------------------------------------------------------
    # Secondary market — double auction
    # ------------------------------------------------------------------

    def _settle_double_auction(self, allocations, secondary_prices,
                                secondary_qtys, clearing_price):
        """
        Double auction:
          - Spread tolerance: trades clear if buyer_price + tol >= seller_price
          - Short positions: agents can sell from banked holdings (not only allocation)
          - Returns (trade_costs, trade_qtys, secondary_clearing_price, total_volume)
        """
        cfg = self.config["trading"]
        trade_costs = np.zeros(self.n_total)
        trade_qtys = np.zeros(self.n_total)

        liquidity_pool_info = {
            "enabled": False,
            "reference_price": float(clearing_price),
            "buy_price": float(clearing_price),
            "sell_price": float(clearing_price),
            "buy_volume": 0.0,
            "sell_volume": 0.0,
        }

        if not cfg["enabled"]:
            return trade_costs, trade_qtys, clearing_price, 0.0, liquidity_pool_info

        tx_cost = cfg["transaction_cost"]
        # spread tolerance as fraction of clearing price
        spread_tol = cfg.get("spread_tolerance", 0.0) * max(clearing_price, 1.0)

        buyers = []
        sellers = []

        for i in range(self.n_total):
            qty = float(secondary_qtys[i])
            price = float(secondary_prices[i])
            if qty > 1e-6:
                buyers.append([i, price, qty])
            elif qty < -1e-6:
                # Enforce no-short-selling: subtract expected compliance need
                max_sell = max(0.0, float(allocations[i]) + float(max(0.0, self.holdings[i]))
                              - float(self._current_emissions[i])
                              - float(self.companies[i]._carry_forward))
                sell_qty = min(abs(qty), max_sell)
                if sell_qty > 1e-6:
                    sellers.append([i, price, sell_qty])

        total_value = 0.0
        total_qty = 0.0

        # 1) Internal matching between agents
        if buyers and sellers:
            buyers.sort(key=lambda x: -x[1])
            sellers.sort(key=lambda x: x[1])

            executed_trades = []
            b_idx, s_idx = 0, 0
            while b_idx < len(buyers) and s_idx < len(sellers):
                buyer_id, buyer_price, buy_qty_rem = buyers[b_idx]
                seller_id, seller_price, sell_qty_rem = sellers[s_idx]

                # trade if buyer_price + spread_tol >= seller_price
                if buyer_price + spread_tol < seller_price:
                    break

                trade_price = (buyer_price + seller_price) / 2.0
                trade_qty = min(buy_qty_rem, sell_qty_rem)
                executed_trades.append((buyer_id, seller_id, trade_price, trade_qty))

                buyers[b_idx][2] -= trade_qty
                sellers[s_idx][2] -= trade_qty
                if buyers[b_idx][2] < 1e-6:
                    b_idx += 1
                if sellers[s_idx][2] < 1e-6:
                    s_idx += 1

            for buyer_id, seller_id, trade_price, trade_qty in executed_trades:
                cost_buyer = trade_qty * (trade_price + tx_cost)
                revenue_seller = trade_qty * (trade_price - tx_cost)

                trade_costs[buyer_id] += cost_buyer
                trade_costs[seller_id] -= revenue_seller

                trade_qtys[buyer_id] += trade_qty
                trade_qtys[seller_id] -= trade_qty

                total_value += trade_price * trade_qty
                total_qty += trade_qty

        # 2) External liquidity pool for unmatched flow
        sec_cfg = self.config.get("secondary", {})
        pool_cfg = sec_cfg.get("liquidity_pool", {})
        pool_enabled = bool(pool_cfg.get("enabled", False))
        if pool_enabled:
            ema_alpha = float(pool_cfg.get("ema_alpha", 0.30))
            penalty_weight = float(pool_cfg.get("penalty_anchor_weight", 0.30))
            penalty_weight = float(np.clip(penalty_weight, 0.0, 1.0))
            spread = float(pool_cfg.get("spread", 0.05))
            spread = max(0.0, spread)

            self._liquidity_ref_ema = (
                ema_alpha * float(clearing_price) +
                (1.0 - ema_alpha) * self._liquidity_ref_ema
            )
            # Liquidity pool floor: prevent pool from pricing at the auction
            # reserve price floor — ensures secondary market remains active even
            # when the primary auction collapses to reserve price.
            # floor_fraction_of_penalty = 0.25 means pool never prices below
            # 25% of the non-compliance penalty (≈34€ at current calibration).
            _eff_penalty_for_pool = (
                float(self.config["penalty"]["rate"])
                * self._inflation_factor(self.current_year)
            )
            _floor_frac = float(pool_cfg.get("floor_fraction_of_penalty", 0.25))
            _pool_price_floor = _floor_frac * _eff_penalty_for_pool
            self._liquidity_ref_ema = max(self._liquidity_ref_ema, _pool_price_floor)
            # Keep pool pricing anchored to market reference by default.
            # Optional override allows explicit anchor experiments.
            penalty_anchor_price = float(pool_cfg.get("penalty_anchor_price", clearing_price))
            ref_price = (
                (1.0 - penalty_weight) * self._liquidity_ref_ema
                + penalty_weight * penalty_anchor_price
            )
            pool_buy_price = ref_price * (1.0 - spread)   # pool buys from agents
            pool_sell_price = ref_price * (1.0 + spread)  # pool sells to agents

            pool_buy_volume = 0.0
            pool_sell_volume = 0.0

            for buyer_id, buyer_price, buy_qty_rem in buyers:
                rem = float(buy_qty_rem)
                if rem > 1e-6 and buyer_price >= pool_sell_price:
                    cost_buyer = rem * (pool_sell_price + tx_cost)
                    trade_costs[buyer_id] += cost_buyer
                    trade_qtys[buyer_id] += rem
                    total_value += pool_sell_price * rem
                    total_qty += rem
                    pool_sell_volume += rem

            for seller_id, seller_price, sell_qty_rem in sellers:
                rem = float(sell_qty_rem)
                if rem > 1e-6 and seller_price <= pool_buy_price:
                    revenue_seller = rem * (pool_buy_price - tx_cost)
                    trade_costs[seller_id] -= revenue_seller
                    trade_qtys[seller_id] -= rem
                    total_value += pool_buy_price * rem
                    total_qty += rem
                    pool_buy_volume += rem

            liquidity_pool_info = {
                "enabled": True,
                "reference_price": float(ref_price),
                "buy_price": float(pool_buy_price),
                "sell_price": float(pool_sell_price),
                "buy_volume": float(pool_buy_volume),
                "sell_volume": float(pool_sell_volume),
            }

        sec_clearing = total_value / total_qty if total_qty > 0 else clearing_price
        return trade_costs, trade_qtys, sec_clearing, total_qty, liquidity_pool_info

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------

    def _compute_rewards(self, payments, trade_costs, penalties,
                         invest_costs, emissions, clearing_price,
                         mac_costs=None, collateral_costs=None,
                         precompliance_holdings=None,
                         old_carry_forward=None, active_mask=None,
                         trade_qtys=None, allocations=None):
        """
        Reward function: inflation-deflated cost buckets, budget_real anchor for
        penalty/ESG, scarcity-amplified prospective penalty, ESG speed bonus,
        no time-decay on ESG, fragility-capped esg_anchor_ratio.
        """
        rewards      = np.zeros(self.n_total)
        base_rewards = np.zeros(self.n_total)
        terminal_bank_values  = np.zeros(self.n_total)
        terminal_queue_values = np.zeros(self.n_total)
        # Reset Phase-2 invest-stream contribution before this year's accumulation.
        self._last_invest_reward_phase2 = np.zeros(self.n_total)

        reward_cfg  = self.config.get("reward", {})
        esg_cfg     = self.config.get("esg", {})
        esg_enabled = esg_cfg.get("enabled", False)
        esg_scale   = float(esg_cfg.get("scale", 2.0))
        _speed_early = float(esg_cfg.get("speed_coef", 0.5))
        _speed_late  = float(esg_cfg.get("speed_coef_late", _speed_early))
        _year_frac   = self.current_year / max(self.n_years - 1, 1)
        esg_speed_coef = _speed_early + _year_frac * (_speed_late - _speed_early)

        if mac_costs is None:
            mac_costs = np.zeros(self.n_total)
        if collateral_costs is None:
            collateral_costs = np.zeros(self.n_total)

        budget_norm_mode = reward_cfg.get("budget_norm_anchor", "dynamic")
        budget_0_fixed   = float(reward_cfg.get("budget_norm_budget_0", 1000.0))

        # Scarcity factor (pre-loop)
        cap_0      = float(self.cap_schedule.get_cap(0))
        cap_t      = float(self.cap_schedule.get_cap(self.current_year))
        scarcity_t = max(0.0, 1.0 - cap_t / max(cap_0, 1e-9))

        # Cache anchor once — avoid redundant calls per agent
        anchor_t        = compute_fundamental_anchor(self.current_year, self.config, cap_t_actual=cap_t)
        # Next-year anchor used by the forward-looking remediation term in penalty_norm.
        next_year       = min(self.current_year + 1, self.n_years - 1)
        cap_t_next      = float(self.cap_schedule.get_cap(next_year))
        anchor_next_nom = compute_fundamental_anchor(next_year, self.config, cap_t_actual=cap_t_next)

        # Banking timing signal config (read once outside agent loop)
        banking_cfg     = reward_cfg.get("banking_signal", {})
        banking_enabled = bool(banking_cfg.get("enabled", True))
        _w_banking      = float(banking_cfg.get("w_banking", 0.3))
        _w_imputed      = float(banking_cfg.get("w_imputed", 1.0))
        _imputed_cap_factor = float(banking_cfg.get("imputed_cap_factor", 2.0))

        for i, company in enumerate(self.companies):
            if active_mask is not None and not bool(active_mask[i]):
                continue

            # Inflation deflator
            infl = company.inflation_factor(self.current_year)

            auction_cost       = float(payments[i])
            secondary_cost     = float(trade_costs[i])
            penalty_cost       = float(penalties[i])
            investment_cost    = float(invest_costs[i])
            # Real-terms OPEX delta: deflate current-year nominal OPEX before
            # subtracting baseline_opex (year-0 snapshot, already real).
            # Avoids an inflation-driven cost term when the mix is unchanged.
            opex_delta_real    = (company.compute_operational_cost(self.current_year) / infl
                                  - company.baseline_opex)
            mac_cost_i         = float(mac_costs[i])
            collateral_cost_i  = float(collateral_costs[i])
            loan_interest_cost = company.compute_green_loan_cost()

            company.record_spending(
                auction_cost + secondary_cost + investment_cost
                + mac_cost_i + collateral_cost_i + loan_interest_cost
                + penalty_cost
            )
            company.record_capex_spending(investment_cost)
            budget_penalty = company.compute_budget_penalty()
            capex_penalty  = company.compute_capex_penalty()

            # Three real cost buckets (all in real terms)
            compliance_cost_real = (auction_cost + secondary_cost + mac_cost_i) / infl
            capital_cost_real    = (investment_cost / infl) + opex_delta_real
            soft_penalty_real    = (budget_penalty + capex_penalty + loan_interest_cost) / infl

            anchor_real = anchor_t / infl
            need        = max(company.compute_estimate_need(), 1e-6)

            if budget_norm_mode == "fixed":
                budget_real = budget_0_fixed / infl
            else:
                budget_real = company.annual_budget / infl

            compliance_denom = max(anchor_real * need, 1.0)
            soft_denom       = max(budget_real, 1.0)

            # Banking timing signal —————————————————————————————————————
            # bank_drawdown: portion of the compliance obligation met by pre-
            # existing banked allowances rather than fresh market purchases.
            #   imputed_bank_norm: marks that drawdown to the current clearing
            #     price so the compliance norm is the same whether the agent
            #     bought or drew — killing the zero-bid free-compliance shortcut.
            #   banking_signal: timing P&L = drawdown × (price − cost_basis),
            #     rewarding cheap-bank/expensive-market and penalising the reverse.
            bank_drawdown_i      = 0.0
            imputed_bank_norm_i  = 0.0
            banking_signal_i     = 0.0
            if (banking_enabled
                    and precompliance_holdings is not None
                    and allocations is not None
                    and trade_qtys is not None):
                _bank_start    = max(float(precompliance_holdings[i]) - float(allocations[i]), 0.0)
                _net_sec_buy   = max(0.0, float(trade_qtys[i]))
                _total_new     = float(allocations[i]) + _net_sec_buy
                _total_oblig   = (float(emissions[i])
                                  + (float(old_carry_forward[i]) if old_carry_forward is not None else 0.0))
                bank_drawdown_i = max(0.0, min(_bank_start, _total_oblig - _total_new))
                # Imputed compliance cost: drawdown marked to clearing price, real terms
                _imputed_real   = bank_drawdown_i * clearing_price / infl
                _imputed_cap    = _imputed_cap_factor * compliance_denom
                imputed_bank_norm_i = min(_imputed_real, _imputed_cap) / compliance_denom
                # Timing P&L: profit from having banked at cost_basis vs current price
                _profit_real    = bank_drawdown_i * (clearing_price - self._bank_cost_basis[i]) / infl
                banking_signal_i = _w_banking * _profit_real / compliance_denom
            # ——————————————————————————————————————————————————————————

            # capital_norm is normalized by budget_real (soft_denom), not by
            # compliance_denom: investment is a budget commitment, not an
            # allowance commitment.
            compliance_norm_cash = compliance_cost_real / compliance_denom
            compliance_norm = compliance_norm_cash + _w_imputed * imputed_bank_norm_i
            capital_norm    = capital_cost_real    / soft_denom
            soft_norm       = soft_penalty_real    / soft_denom

            loan_sting_coef     = float(self.config.get("budget", {}).get(
                "emergency_loan", {}).get("origination_sting_coef", 0.07))
            loan_draw_this_year = float(getattr(company, '_loan_drawn_this_step', 0.0))
            loan_sting = 0.0
            if loan_draw_this_year > 0:
                loan_sting = (loan_draw_this_year / max(budget_real, 1.0)) * loan_sting_coef

            cost_norm = compliance_norm + capital_norm + soft_norm + loan_sting

            # cost_norm is centered at the expected compliance cost so that zero = perfect
            # efficiency, positive = under-spent, negative = over-spent. By construction,
            # compliance_denom = anchor_real × annual_need, so a fully-compliant agent
            # buying exactly at the anchor pays compliance_norm = 1.0. Subtracting 1.0
            # makes both financial and ESG reward scales comparable: a financial agent
            # at normal operation scores R = 0 (same break-even as a mid-journey ESG agent).
            expected_compliance_norm = 1.0
            cost_norm_centered       = cost_norm - expected_compliance_norm

            urgency_scalar = float(self._urgency_scalars[i]) if i < self.n_agents else 1.0

            # Penalty math: two genuinely different economic costs are
            # tracked separately rather than summed at the same rate:
            #   (1) penalty_realized  — money paid THIS year for shortfall.
            #       Reconstructed in REAL terms from penalty_cost (which was
            #       computed at the inflated effective rate) so the reward
            #       stays inflation-invariant.
            #   (2) remediation_cost  — carry-forward debt must be repaid
            #       NEXT year by buying replacement allowances at the
            #       expected market price (next-year anchor) under scarcity.
            #       Honors the "scarcity makes catch-up harder" intent
            #       without re-charging the penalty rate.
            eff_pen_rate       = max(company.effective_penalty_rate(self.current_year), 1e-9)
            shortfall_realized = float(penalty_cost) / eff_pen_rate

            # Divide by (infl × compliance_denom) = anchor_t × need so the
            # penalty is on the same scale as compliance_norm_cash and is
            # genuinely more expensive than simply buying at the market price.
            penalty_realized = (shortfall_realized * company.penalty_rate * urgency_scalar
                                / max(infl * compliance_denom, 1.0))

            anchor_next_real = anchor_next_nom / infl
            scarcity_amp     = 1.0 + scarcity_t
            cf_debt          = float(company._carry_forward)
            # anchor_next_real and compliance_denom are both in real M€ — ratio is
            # inflation-invariant (cost of remediating carry-forward at next anchor).
            remediation_cost = (cf_debt * anchor_next_real * scarcity_amp * urgency_scalar
                                / max(compliance_denom, 1.0))

            penalty_norm = penalty_realized + remediation_cost
            # Aliased name kept in the diagnostic log.
            penalty_prospective = remediation_cost
            shortfall = shortfall_realized

            # ESG: no time decay, budget_real anchor
            esg_signal       = 0.0
            esg_anchor_ratio = 0.0
            gate_activation  = 1.0
            compliance_gate  = 1.0

            if esg_enabled and company.initial_ef > 1e-6:
                ef_ratio    = max(0.0,
                    (company.initial_ef - company.weighted_emission_factor) / company.initial_ef
                )
                green_delta = max(0.0, company.green_frac - company.prev_green_frac)
                speed_bonus = esg_speed_coef * green_delta

                # Centered ESG: subtract a linear `year/n_years` baseline so a
                # do-nothing agent receives zero-mean signal and an
                # ahead-of-trajectory agent receives a positive one. This
                # eliminates the positive-floor that previously biased HAPPO
                # ordering against agents with `w_green > 0`.
                ef_baseline = float(self.current_year) / max(float(self.n_years - 1), 1.0)
                ef_baseline = float(np.clip(ef_baseline, 0.0, 1.0))
                ef_centered = ef_ratio - ef_baseline
                esg_raw = esg_scale * (ef_centered + speed_bonus)
                esg_anchor_ratio = 1.0  # retained as a logged channel only

                annual_need_i = max(company.compute_estimate_need(), 1e-6)
                if precompliance_holdings is not None:
                    coverage_frac = min(1.0, float(precompliance_holdings[i]) / annual_need_i)
                else:
                    coverage_frac = 1.0
                gate_blend_threshold = float(reward_cfg.get("compliance_gate_blend_threshold", 0.90))
                gate_blend_width     = float(reward_cfg.get("compliance_gate_blend_width", 0.30))
                gate_blend      = max(0.0, min(1.0, (gate_blend_threshold - coverage_frac) / gate_blend_width))
                compliance_gate = coverage_frac ** (1.0 + gate_blend)
                gate_activation = float(1.0 + gate_blend)
                # Gate only attenuates positive ESG; do not flip the sign of a
                # negative (behind-trajectory) signal under low coverage.
                if esg_raw >= 0.0:
                    esg_signal = esg_raw * compliance_gate
                else:
                    esg_signal = esg_raw

            if allocations is not None:
                coverage_frac_auction = min(
                    float(allocations[i]) / max(company.compute_estimate_need(), 1e-6), 1.0
                )
            else:
                coverage_frac_auction = 1.0
            # v8.4.2 bug fix: apply the coverage gate ONLY when the financial reward
            # is a saving (cost_norm_centered < 0 → -cost_norm_centered > 0). Gating
            # the cost branch too made "skip the auction" (reward=0) dominate
            # "win at clearing ≥ anchor" (reward<0), pushing policies toward
            # under-bidding. Costs are now applied at full weight regardless of
            # coverage; only over-savings get coverage-discounted.
            if cost_norm_centered < 0.0:
                financial_reward = coverage_frac_auction * company.w_cost * (-cost_norm_centered)
            else:
                financial_reward = company.w_cost * (-cost_norm_centered)

            base_reward = float(
                financial_reward
                + company.w_green * esg_signal
                - penalty_norm
                + banking_signal_i
            )
            base_rewards[i] = base_reward
            rewards[i]      = base_reward

            # Track invest-stream Phase-2 contribution (ESG only at this stage;
            # the terminal_queue value is added in the year-T branch below).
            self._last_invest_reward_phase2[i] = float(company.w_green * esg_signal)

            opp_cost_shaping = 0.0
            opp_cost_cfg = reward_cfg.get("opportunity_cost_shaping", {})
            if (opp_cost_cfg.get("enabled", False)
                    and self.shaping_weight > 0
                    and trade_qtys is not None
                    and float(trade_qtys[i]) > 1e-6):
                sec_buy_qty         = float(trade_qtys[i])
                sec_buy_cost_per_mt = float(trade_costs[i]) / sec_buy_qty
                opp_cost_per_mt     = max(0.0, sec_buy_cost_per_mt - clearing_price)
                opp_cost_scale      = float(opp_cost_cfg.get("scale", 1.0))
                opp_cost_shaping    = -(opp_cost_scale * opp_cost_per_mt * sec_buy_qty
                                        * self.shaping_weight / max(budget_real, 1.0))
                rewards[i] += opp_cost_shaping

            coverage_gap_shaping = 0.0
            cov_gap_cfg = reward_cfg.get("coverage_gap_shaping", {})
            if (cov_gap_cfg.get("enabled", False)
                    and self.shaping_weight > 0
                    and allocations is not None):
                alloc_i = float(allocations[i])
                need_i  = max(float(emissions[i]) + (float(old_carry_forward[i])
                              if old_carry_forward is not None else 0.0), 0.1)
                gap = max(0.0, need_i - alloc_i)
                if gap > 0.01:
                    pen_rate = float(self.config["penalty"]["rate"])
                    cov_scale = float(cov_gap_cfg.get("scale", 0.5))
                    coverage_gap_shaping = -(cov_scale * gap * pen_rate
                                             * self.shaping_weight / max(budget_real, 1.0))
                    rewards[i] += coverage_gap_shaping

            self._last_reward_channels[i] = {
                "compliance_norm":           float(compliance_norm),
                "compliance_norm_cash":      float(compliance_norm_cash),
                "imputed_bank_norm":         float(imputed_bank_norm_i),
                "bank_drawdown":             float(bank_drawdown_i),
                "bank_cost_basis":           float(self._bank_cost_basis[i]),
                "banking_signal":            float(banking_signal_i),
                "capital_norm":              float(capital_norm),
                "soft_norm":                 float(soft_norm),
                "cost_norm":                 float(cost_norm),
                "expected_compliance_norm":  float(expected_compliance_norm),
                "cost_norm_centered":        float(cost_norm_centered),
                "coverage_frac_auction":     float(coverage_frac_auction),
                "financial_reward":          float(financial_reward),
                "revenue_norm":         0.0,   # revenue is no longer in the reward; key kept for log compatibility
                "penalty_norm":         float(penalty_norm),
                "penalty_prospective":  float(penalty_prospective),  # alias of remediation_cost
                "penalty_realized":     float(penalty_realized),
                "remediation_cost":     float(remediation_cost),
                "scarcity_amp":         float(scarcity_amp),
                "esg_signal":           float(esg_signal),
                "esg_anchor_ratio":     float(esg_anchor_ratio),     # always 1.0 (logged channel only)
                "base_reward":          float(base_reward),
                "opp_cost_shaping":     float(opp_cost_shaping),
                "coverage_gap_shaping": float(coverage_gap_shaping),
                "gate_activation":      float(gate_activation),
                "compliance_gate":      float(compliance_gate),
                "esg_vs_penalty_ratio": float(company.w_green * esg_signal) / max(float(penalty_norm), 1e-9),
                "anchor_t":             float(anchor_t),
                "anchor_real":          float(anchor_real),
                "anchor_next_real":     float(anchor_next_nom / infl),
                "budget_real":          float(budget_real),
                "infl":                 float(infl),
                "shortfall":            float(shortfall),
                "carry_forward_debt":   float(company._carry_forward),
            }

        is_final_year  = self.current_year >= self.n_years - 1
        terminal_bank  = bool(reward_cfg.get("terminal_bank_value", False))
        terminal_queue = bool(reward_cfg.get("terminal_queue_value", True))

        if is_final_year:
            gamma_discount        = float(self.config["ppo"].get("gamma", 0.99))
            terminal_payoff_years = float(reward_cfg.get("terminal_payoff_years", 5.0))
            pen_cfg     = self.config["penalty"]
            eff_penalty = pen_cfg["rate"] * self._inflation_factor(self.current_year)
            terminal_price = max(clearing_price, self.last_secondary_price, eff_penalty * 0.8)
            # Asset operating life used to convert capacity additions into a
            # discounted-cash-flow terminal value. Lifetime spans well beyond
            # the simulated horizon, so late-episode investments still pay back.
            asset_lifetime = float(reward_cfg.get("terminal_asset_lifetime_years", 20.0))
            invest_rate = float(self.config.get("investment", {}).get("discount_rate", 0.05))

            for i, company in enumerate(self.companies):
                if active_mask is not None and not bool(active_mask[i]):
                    continue

                infl_t        = company.inflation_factor(self.current_year)
                budget_real_t = max(company.annual_budget / infl_t, 1.0)

                if terminal_bank:
                    # Discounted hold value: holdings × terminal_price discounted
                    # by the investment hurdle rate over the post-episode
                    # operating horizon. No linear-below-need kicker, so the
                    # incentive scales smoothly with bank size and never
                    # dominates the late-year invest-vs-hold trade-off.
                    capped_holdings = max(0.0, float(self.holdings[i]))
                    discount = (1.0 + invest_rate) ** (-terminal_payoff_years)
                    bank_value = capped_holdings * terminal_price * discount / budget_real_t
                    rewards[i]      += bank_value
                    base_rewards[i] += bank_value
                    terminal_bank_values[i] = bank_value

                if terminal_queue:
                    # NPV terminal value of pipeline projects: each queued
                    # project contributes the present value of its annual
                    # carbon savings over `asset_lifetime`, discounted from
                    # the project's completion year. This values late-episode
                    # investments at their economic worth (rather than zeroing
                    # them) and so removes the structural incentive to stop
                    # investing after the first few years.
                    queue_value = 0.0
                    for item in company._construction_queue:
                        tech_idx = int(item.get("tech_idx", -1))
                        if tech_idx < 0 or tech_idx >= len(company.emission_factors):
                            continue
                        years_to_completion = max(
                            0, int(item.get("completion_year", self.current_year)) - self.current_year,
                        )
                        delta_ef = company.weighted_emission_factor - company.emission_factors[tech_idx]
                        if delta_ef <= 0:
                            continue
                        frac_delta = float(item.get("frac_delta", 0.0))
                        if frac_delta <= 0.0:
                            continue
                        annual_saving_mt = (delta_ef * frac_delta * company.output_mwh) / 1e6
                        annual_value = annual_saving_mt * terminal_price
                        # Annuity factor over the operating lifetime
                        if invest_rate > 1e-9:
                            annuity = (1.0 - (1.0 + invest_rate) ** (-asset_lifetime)) / invest_rate
                        else:
                            annuity = float(asset_lifetime)
                        # Discount from project completion back to year 0
                        discount = (1.0 + invest_rate) ** (-years_to_completion)
                        queue_value += annual_value * annuity * discount / budget_real_t
                    rewards[i]      += queue_value
                    base_rewards[i] += queue_value
                    terminal_queue_values[i] = queue_value
                    self._last_invest_reward_phase2[i] += float(queue_value)

                if company._carry_forward > 0:
                    debt_penalty = (company._carry_forward * terminal_price * 1.5) / budget_real_t
                    rewards[i]      -= debt_penalty
                    base_rewards[i] -= debt_penalty

                if bool(reward_cfg.get("treasury_terminal_value", True)):
                    if company._treasury_enabled and company._treasury_reserve > 0:
                        t_value = company._treasury_reserve * company._treasury_terminal_rate / budget_real_t
                        rewards[i]      += t_value
                        terminal_bank_values[i] += t_value

        self._last_terminal_bank_values        = terminal_bank_values
        self._last_terminal_queue_values       = terminal_queue_values
        self._last_terminal_liquidation_values = terminal_bank_values + terminal_queue_values
        self._last_reward_base_values          = base_rewards
        self._last_reward_shaping_values       = rewards - base_rewards

        return rewards

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def compute_diagnostic_score(self, agent_id: int = None) -> dict:
        """
        Compute interpretable diagnostic scores for agents.

        Returns a dict with three normalized components (each in [0, 1]):
          S_financial: cost efficiency (lower total cost = higher score)
          S_green:     emission factor progress relative to initial
          S_penalty:   compliance score (1 - normalized penalty incurred)
          S_composite: weighted combination based on agent's w_cost/w_green

        Parameters
        ----------
        agent_id : int, optional
            If provided, return scores for that specific agent only.
        """
        results = []
        budget_ref = max(1.0, float(
            sum(c.annual_budget for c in self.companies[:self.n_agents]) / max(self.n_agents, 1)
        ))
        pen_cfg = self.config["penalty"]
        eff_penalty = pen_cfg["rate"] * self._inflation_factor(self.current_year)

        for i, company in enumerate(self.companies[:self.n_agents]):
            if not self._is_agent_active(i):
                results.append({
                    "agent_id": i, "S_financial": 0.0, "S_green": 0.0,
                    "S_penalty": 0.0, "S_composite": 0.0,
                })
                continue

            s_financial = min(1.0, max(0.0, 1.0 - company.budget_spent_this_year / max(budget_ref, 1.0)))

            if company.initial_ef > 1e-6:
                ef_progress = max(0.0, company.initial_ef - company.weighted_emission_factor)
                s_green = ef_progress / company.initial_ef
            else:
                s_green = 1.0

            s_penalty = 1.0  # conservative default; overridden by year_log data

            s_composite = (company.w_cost * s_financial
                           + company.w_green * s_green
                           + 0.3 * s_penalty)

            results.append({
                "agent_id": i,
                "S_financial": round(float(s_financial), 4),
                "S_green": round(float(s_green), 4),
                "S_penalty": round(float(s_penalty), 4),
                "S_composite": round(float(s_composite), 4),
            })

        if agent_id is not None:
            return results[agent_id] if agent_id < len(results) else {}
        return results

    def _compute_price_ma3(self) -> float:
        """3-year moving average of clearing price.
        When auctions have failed for ≥2 consecutive years, blend toward
        expected_price to prevent stale MA3 from misleading agents.
        """
        if not self._price_history:
            # Fallback to AR(1) expected price (initialized at ~80€) instead
            # of last_clearing_price which may be the reserve price after a
            # failed auction.
            return self.expected_price
        window = self._price_history[-3:]
        ma3 = float(np.mean(window))
        n = self._consecutive_years_without_valid_auction_clear
        if n >= 2:
            blend = min(1.0, (n - 1) * 0.5)
            return (1 - blend) * ma3 + blend * self.expected_price
        return ma3

    def _get_obs_phase1(self) -> np.ndarray:
        """Phase 1 observations for learning agents only.
        Base: 36D. With opponent modeling: 36 + 7*(N_total-1) dims (lagged by 1 year).
        Opponent modeling includes ALL market participants (learning + bots).
        """
        cap_t = self.cap_schedule.get_cap(self.current_year)
        price_ma3 = self._compute_price_ma3()

        # TNAC proxy: total banked allowances / cap (market-level scarcity signal)
        tnac = float(self.holdings.sum())
        tnac_proxy = tnac / max(cap_t, 1e-6)

        # THIS YEAR'S auction volume: use MSR preview (read-only, no state changes).
        # Agents see the actual supply before bidding — critical for MSR flood/drought.
        price_max = float(self.config["auction"]["price_max"])
        base_penalty_rate = float(self.config["penalty"]["rate"])
        inflation_rate = float(self._inflation_rate(self.current_year))
        this_year_auction_volume = self.cap_schedule.preview_auction_volume(
            self.current_year,
            clearing_price=self.last_clearing_price,
            price_max=price_max,
            penalty_rate=base_penalty_rate,
            inflation_rate=inflation_rate,
            inflation_factor=self._inflation_factor(self.current_year),
            price_ma3=price_ma3,
        )
        # Include both rollover channels that are added in step_auction.
        unsold_pending = float(getattr(self.cap_schedule, "_unsold_rollover_pending", 0.0))
        defaulted_pending = float(self._defaulted_volume_pending)
        this_year_auction_volume += unsold_pending + defaulted_pending
        max_rollover_mult = float(getattr(self.cap_schedule, "max_rollover_multiplier", 1.5))
        this_year_auction_volume = min(this_year_auction_volume, cap_t * max_rollover_mult)

        msr_reserve = self.cap_schedule.msr_reserve()
        opp_mode = self.config.get("opponent_obs", {}).get("mode", "lagged")

        # Cap scarcity lookahead: forward cap ratios for investment/banking planning
        cap_ahead_3y_ratio = float(np.clip(
            self.cap_schedule.get_cap(min(self.current_year + 3, self.n_years - 1))
            / max(cap_t, 1e-6), 0.0, 1.0
        ))
        cap_ahead_6y_ratio = float(np.clip(
            self.cap_schedule.get_cap(min(self.current_year + 6, self.n_years - 1))
            / max(cap_t, 1e-6), 0.0, 1.0
        ))

        obs_list = []
        for i in range(self.n_agents):  # only learning agents get observations
            c = self.companies[i]
            if self._opponent_modeling and self.n_total > 1:
                opponent_obs_list = []
                for j in range(self.n_total):
                    if j == i:
                        continue
                    if opp_mode == "full_info":
                        pub = self.companies[j].get_public_info()
                        opponent_obs_list.extend([
                            pub["emissions"], pub["carry_forward"], pub["green_frac"],
                            pub["fossil_frac"], pub["queue_total"], pub["is_active"],
                        ])
                    else:  # "lagged" (default) — read year t-1 snapshot
                        opponent_obs_list.extend(self._opponent_snapshots_prev[j].tolist())
                opponent_obs = np.array(opponent_obs_list, dtype=np.float32) if opponent_obs_list else None
            else:
                opponent_obs = None

            obs_i = c.get_observation_phase1(
                year=self.current_year,
                cap_t=cap_t,
                last_clearing_price=self.last_clearing_price,
                expected_price=self.expected_price,
                auction_gap=self._last_gaps[i],
                last_secondary_price=self.last_secondary_price,
                secondary_profit_signal=self._secondary_profit_ema[i],
                price_ma3=price_ma3,
                opponent_obs=opponent_obs,
                last_secondary_volume=self.last_secondary_volume,
                tnac_proxy=tnac_proxy,
                effective_reserve=self._last_effective_reserve,
                last_auction_volume=this_year_auction_volume,
                msr_reserve=msr_reserve,
                bank=float(self.holdings[i]),
                tnac_upper=float(self.cap_schedule.tnac_upper),
                withhold_rate=float(self.cap_schedule.withhold_rate),
                budget_spent=float(c.budget_spent_this_year),
                annual_budget=float(c.annual_budget),
                collateral_load_last=float(self._last_collateral_load[i]),
                bid_affordability_last=float(self._bid_affordability[i]),
                n_years=self.n_years,
                last_cover_ratio=self._last_cover_ratio,
                own_last_secondary_buy_price=float(self._last_secondary_buy_price[i]),
                cumulative_coverage_ratio=float(
                    self._cumulative_alloc[i] / max(self._cumulative_emissions[i], 0.1)
                ) if self._cumulative_emissions[i] > 0.01 else 1.0,
                cap_ahead_3y_ratio=cap_ahead_3y_ratio,
                cap_ahead_6y_ratio=cap_ahead_6y_ratio,
                pcl_ceiling=self._pcl_ceiling,
                last_bid_price_clip=float(self._last_bid_price_clip[i]),
                last_budget_price_clip=float(self._last_budget_price_clip[i]),
                last_bid_qty_clip_ratio=float(self._last_bid_qty_clip_ratio[i]),
                last_invest_clip_ratio=float(self._last_invest_clip_ratio[i]),
            )
            obs_list.append(obs_i)

        if not obs_list:
            return np.zeros((0,), dtype=np.float32)
        return np.stack(obs_list)
