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

Roadmap improvements (P1-P4): price MA, entropy, reward normalisation, green shaping.
Opponent modeling: Phase 1 obs augmented with last-episode (bid/200, green_frac) for N-1 agents.
Agent cycling: handled in train.py.

Roadmap improvements (P5-P8):
  P5: Stochastic annual emission shocks (correlated across agents); shock in Phase 2 obs.
  P6: Construction delay jitter + cancellation risk + capacity factor noise.
  P7: Warm-start — queue seeding + bank seeding + price history seeding at reset.
    P8: Wider secondary spread tolerance; selling from bank; cost-of-capital signal;
      secondary price/volume in Phase 1 observation.
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
        self.last_clearing_price = config["price"]["initial_expected"]
        self.expected_price = config["price"]["initial_expected"]
        self._price_history: List[float] = []
        self.last_secondary_price = config["price"]["initial_expected"]
        self.last_secondary_volume = 0.0  # P8: track volume for phase1 obs
        self._last_auction_volume = self.cap_schedule.get_cap(0)
        self._last_gaps = np.zeros(self.n_total)
        self.holdings = np.zeros(self.n_total)
        self.episode_done = False

        # P4: shaping weight — decays from 1.0 to 0.0 over training (set by train.py)
        self.shaping_weight = 1.0

        # Secondary market profit tracking (EMA per agent)
        self._secondary_profit_ema = np.zeros(self.n_total)
        self._ema_alpha = 0.1

        # Auction results (stored between phase 1 and phase 2)
        self._phase1_allocations = None
        self._phase1_payments = None
        self._phase1_clearing_price = 0.0
        self._phase1_invest_costs = None
        self._phase1_obs = None
        self._phase1_log = None
        self._phase1_bid_prices = None
        self._phase1_bid_quantities = None  # actual Mt quantities after multiplier expansion

        # P4: Per-agent fossil fraction history within the episode (last 3 years)
        self._fossil_frac_history: List[List[float]] = [[] for _ in range(self.n_total)]

        # P5: Stochastic emission shocks — computed per year in step_auction()
        # Stores the shocked realized emissions and the shock values for obs/logging
        self._current_emissions = np.zeros(self.n_total)   # shocked
        self._current_emission_shocks = np.zeros(self.n_total)  # ε_it values

        # P6: CF noise per agent per tech — computed per year in step_auction()
        self._current_cf_noise = np.zeros((self.n_total, 5))
        self._p6_cancellations = np.zeros(self.n_total, dtype=int)

        # Unsold allowance rollover: volume offered at auction but not allocated
        # carries forward to the next year's auction supply.
        self._unsold_rollover = 0.0

        # E4: Suspension and default carry-forward tracking
        # _suspension_remaining[i]: number of auction rounds agent i is still suspended
        # _defaulted_volume_pending: allowance volume returned by defaults to add next year
        self._suspension_remaining = np.zeros(self.n_total, dtype=int)
        self._defaulted_volume_pending = 0.0
        # E2/E4: Per-agent collateral locked in Phase 1 (step_auction).
        # Stored so Phase 2 can charge the opportunity cost without re-deriving bids.
        self._collateral_locked = np.zeros(self.n_total)
        # Per-agent collateral load from PREVIOUS year: collateral_locked / annual_budget.
        # Exposed in Phase 1 obs so agents learn to avoid over-committing and defaulting.
        self._last_collateral_load = np.zeros(self.n_total)
        # C2: Per-agent bid affordability from PREVIOUS year
        self._bid_affordability = np.zeros(self.n_total)
        # E2: Collateral warning counter — cumulative per-agent count of collateral warnings
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
        self._liquidity_ref_ema = float(config["price"]["initial_expected"])

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

        # Terminal liquidation components from the latest reward computation.
        # These are logged into year_log so notebook diagnostics can mirror
        # the exact reward logic without re-implementing formulas.
        self._last_terminal_bank_values = np.zeros(self.n_total)
        self._last_terminal_queue_values = np.zeros(self.n_total)
        self._last_terminal_liquidation_values = np.zeros(self.n_total)
        self._last_reward_base_values = np.zeros(self.n_total)
        self._last_reward_shaping_values = np.zeros(self.n_total)

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

        # Reward channel diagnostics
        self._last_reward_channels: dict = {}
        self._last_auction_reward_channels: dict = {}

        # Per-agent per-year diagnostics (M1, validation diagnostics)
        self._last_per_agent_diag: dict = {}

        # Emission factor diagnostics (M1)
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
        self.last_clearing_price = self.config["price"]["initial_expected"]
        self.expected_price = self.config["price"]["initial_expected"]
        self.last_secondary_price = self.config["price"]["initial_expected"]
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
        self._secondary_profit_ema = np.zeros(self.n_total)
        self._consecutive_years_without_valid_auction_clear = 0
        self._liquidity_ref_ema = float(self.config["price"]["initial_expected"])

        self.cap_schedule.reset()
        self._unsold_rollover = 0.0
        self._suspension_remaining = np.zeros(self.n_total, dtype=int)
        self._defaulted_volume_pending = 0.0
        self._collateral_locked = np.zeros(self.n_total)
        self._last_collateral_load = np.zeros(self.n_total)
        self._bid_affordability = np.zeros(self.n_total)
        self._collateral_warning_count = np.zeros(self.n_total, dtype=int)
        self._collateral_clip_events = {i: 0 for i in range(self.n_total)}
        self._current_marginal_ef = 0.0
        self._build_episode_inflation_path()

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

        initial_mixes = self.config["companies"]["initial_mix"]
        for i, company in enumerate(self.companies):
            company.reset(initial_mix=initial_mixes[i])
            company.rng = self.rng
            company.set_inflation_path(self._inflation_rates)

        # P7: Warm-start — seed construction queue, holdings, price history
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
                    suspension_remaining=int(self._suspension_remaining[i]),
                    suspension_length=int(self.config["auction"].get("suspension_length", 2)),
                    collateral_load_last=float(self._last_collateral_load[i]),
                )

                bid_price = float(np.clip(action[0], price_min, price_max))
                qty_mult = float(np.clip(action[1], qty_mult_low, qty_mult_high))
                annual_need = max(company.compute_estimate_need() + company._carry_forward, 0.1)
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
                self._price_history.append(clipped_price)
                self.last_clearing_price = clipped_price
                # Seed _prev_ma3 with the MA3 that now includes this clearing so
                # that year 0 of the real episode has a valid MA3 history and the
                # A4 smoothed guard is active from the start.
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
        price_floor = self.config["price"].get(
            "ar1_floor",
            self.config["price"].get("price_floor", 50.0),
        )
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
        P7 fallback warm-start seeding (when burn-in is disabled).

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
            price_floor = self.config["price"].get("ar1_floor",
                            self.config["price"].get("price_floor", 50.0))
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
                suspension_remaining=int(self._suspension_remaining[idx]),
                suspension_length=int(self.config["auction"].get("suspension_length", 2)),
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

        # 1. P6: Cancellation check — before matured investments
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

        # ---- Revenue-based dynamic budget (A3) ----
        budget_mode = self.config.get("budget", {}).get("mode", "fixed")
        if budget_mode == "revenue_based":
            # Smoothed price: MA3 from price history, padded with initial_expected
            init_p = self.config["price"]["initial_expected"]
            hist = list(self._price_history)
            while len(hist) < 3:
                hist.insert(0, init_p)
            smoothed_price = float(np.mean(hist[-3:]))
            carbon_price_for_budget = smoothed_price
            # System-wide average emission factor
            active_companies = [c for c in self.companies if self._is_agent_active(c.agent_id)]
            # System-wide average EF (kept for observation space)
            system_ef = float(np.mean([c.weighted_emission_factor for c in active_companies]))
            # M1: Marginal EF — EF of most carbon-intensive technology with significant system share.
            # Carbon cost pass-through in electricity markets prices off the marginal setter (typically
            # coal when it has material system presence). Using system_ef understates coal revenue.
            # Reference: Fabra & Reguant (2014), Sijm et al. (2006).
            marginal_ef = self._compute_marginal_ef()
            # Update budgets for all companies (agents + bots)
            for c in active_companies:
                c.set_annual_budget(
                    c.compute_dynamic_budget(carbon_price_for_budget, marginal_ef, self.current_year)
                )
                c.apply_loan_repayment()  # B4: repay from fresh dynamic budget
                c.reset_budget()
                c.reset_capex_budget()
            # Store both for logging/obs space
            self._last_system_ef = system_ef
            self._last_marginal_ef = marginal_ef
        else:
            active_companies = [c for c in self.companies if self._is_agent_active(c.agent_id)]
            for company in active_companies:
                company.apply_loan_repayment()
                company.reset_budget()
                company.reset_capex_budget()

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
            price_ma3=price_ma3,
        )
        auction_volume = base_auction_volume
        log["cap"] = cap_t
        log["tnac"] = tnac

        # Keep unsold and defaulted rollovers as independent accounting streams.
        unsold_rolled_in = float(getattr(self.cap_schedule, "_last_unsold_rollover_in", 0.0))
        msr_withheld = float(getattr(self.cap_schedule, "_last_msr_withheld", 0.0))
        msr_released = float(getattr(self.cap_schedule, "_last_msr_released", 0.0))

        # E4: Add defaulted volume from previous year to this year's supply
        defaulted_rolled_in = 0.0
        if self._defaulted_volume_pending > 0.0:
            if self.config["auction"].get("carry_forward_defaults", True):
                auction_volume += self._defaulted_volume_pending
                defaulted_rolled_in = self._defaulted_volume_pending
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

        # 4. P5: Generate correlated emission shocks
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

        # 5. P6: Generate capacity factor noise per tech per agent
        cf_sigma = np.array(jitter_cfg.get("cf_sigma", [0.0, 0.0, 0.08, 0.08, 0.05]))
        cf_noise = np.zeros((self.n_total, 5))
        if jitter_cfg.get("enabled", False):
            for i in range(self.n_total):
                for t in range(5):
                    if cf_sigma[t] > 0:
                        cf_noise[i, t] = float(self.rng.normal(0, cf_sigma[t]))
        self._current_cf_noise = cf_noise

        # 6. Compute realized emissions (CF noise → P6, demand shock → P5)
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
        # Quantity reparameterization: action[1] is a coverage MULTIPLIER on estimated need.
        # actual_qty = multiplier × (compute_estimate_need + carry_forward)
        # This keeps the strategic decision centred on compliance coverage ratio rather
        # than an absolute volume, avoiding the zero-quantity collapse.
        qty_mult_low = self.config["auction"].get("qty_mult_low", 0.3)
        qty_mult_high = self.config["auction"].get("qty_mult_high", 2.0)
        lot_size = self.config["auction"].get("lot_size", 0.0)
        bid_qty_multipliers = np.zeros(self.n_total)
        estimate_needs = np.zeros(self.n_total)
        bid_coverages = np.zeros(self.n_total)
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
            base_need = max(company.compute_estimate_need() + company._carry_forward, 0.1)
            bid_actions[i, 1] = multiplier * base_need
            bid_qty_multipliers[i] = multiplier
            estimate_needs[i] = base_need
            bid_coverages[i] = bid_actions[i, 1] / max(base_need, 1e-6)
            # EU lot-size discretization: round to nearest multiple of lot_size
            if lot_size > 0:
                bid_actions[i, 1] = max(lot_size, round(bid_actions[i, 1] / lot_size) * lot_size)

        # E4: Leverage gate — clip bid_quantity by leverage_multiplier × available_cash / bid_price.
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

        # E4: Suspension enforcement — suspended agents bid zero (filtered by market_clearing).
        # Decrement suspension counter so agents are released after suspension_length rounds.
        for i in range(self.n_total):
            if self._suspension_remaining[i] > 0:
                bid_actions[i, 1] = 0.0  # zero quantity → filtered by valid_mask in clearing
                self._suspension_remaining[i] -= 1

        # E4: Pre-bid collateral locking — fraction of margin above reserve.
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
        clearing_price, allocations, payments, auction_stats = market_clearing_ets(
            bids=bids,
            q_cap=auction_volume,
            reserve_price=effective_reserve,
            max_agent_share=self.config["auction"].get("max_agent_share", 1.0),
            rng=self.rng,
            cancel_under_subscribed=self.config["auction"].get(
                "cancel_under_subscribed", False),
            n_agents=self.n_total,
            pricing_rule=self.config["auction"].get("pricing_rule", "uniform"),
        )

        # E4: Post-clearing settlement — check each winner can pay; handle defaults.
        auction_susp_len = self.config.get("auction", {}).get("suspension_length", None)
        budget_susp_len = self.config.get("budget", {}).get("suspension_length", None)
        suspension_length = int(
            auction_susp_len if auction_susp_len is not None
            else (budget_susp_len if budget_susp_len is not None else 1)
        )
        agent_cash = np.array([
            max(0.0, float(c.annual_budget - c.budget_spent_this_year))
            for c in self.companies
        ])
        # B3: Compute max emergency loan budgets per agent
        loan_cfg = self.config.get("budget", {}).get("emergency_loan", {})
        if loan_cfg.get("enabled", False):
            max_loan_frac = float(loan_cfg.get("max_loan_fraction", 0.15))
            max_loan_budgets = np.array([
                max_loan_frac * float(c.annual_budget) for c in self.companies
            ])
        else:
            max_loan_budgets = None
        (allocations, payments,
         defaults_mask, defaulted_volume,
         suspension_steps, loan_amounts) = settle_auction(
            allocations=allocations,
            payments=payments,
            agent_cash=agent_cash,
            collateral_locked=collateral_locked,
            suspension_length=suspension_length,
            max_loan_budgets=max_loan_budgets,
        )
        # B4: Apply emergency loans to companies
        for i in range(self.n_total):
            if loan_amounts[i] > 0:
                self.companies[i].apply_emergency_loan(loan_amounts[i])
        # Apply defaults: exhaust remaining annual budget (signals insolvency).
        # Collateral is forfeited implicitly: settle_auction already zeroed the allocation
        # so no allowances are received, but the locked collateral amount is not returned.
        for i in range(self.n_total):
            if defaults_mask[i]:
                excess = max(0.0, self.companies[i].annual_budget
                             - self.companies[i].budget_spent_this_year)
                self.companies[i].record_spending(excess)
                self._suspension_remaining[i] = suspension_steps[i]
        # Carry forward defaulted volume to next year's q_cap
        if defaulted_volume > 0.0:
            self._defaulted_volume_pending += defaulted_volume
        # Augment auction_stats with E4 default/suspension info
        auction_stats["defaults"] = int(defaults_mask.sum())
        auction_stats["defaulted_volume"] = float(defaulted_volume)
        auction_stats["suspended_agents"] = int((self._suspension_remaining > 0).sum())
        auction_stats["defaults_agents"] = sorted(int(i) for i, d in enumerate(defaults_mask) if d)
        auction_stats["suspension_remaining_list"] = self._suspension_remaining.tolist()

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
        hard_cap_mult = float(budget_cfg.get("hard_cap_multiplier", 1.20))

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
            tech_logits = auction_actions[i, 3:6]
            tech_choice = int(np.argmax(tech_logits))
            invest_tech_choices[i] = tech_choice

            tech_idx = tech_choice + 2  # 0/1/2 -> onshore/offshore/solar in company mix

            budget_ceiling = company.annual_budget * hard_cap_mult
            budget_remaining = max(0.0, budget_ceiling - company.budget_spent_this_year)
            capex_cost = company.compute_investment_cost(tech_idx, invest_frac, year)
            total_proj_cost = capex_cost + _estimate_decommission_cost(company, invest_frac)
            budget_clipped = False
            capex_clipped = False

            # I2: Investment hard gate — block if would exceed hard cap
            if budget_cfg.get("investment_hard_gate", True):
                hard_cap_frac = float(budget_cfg.get("hard_cap_fraction", 1.15))
                hard_cap_abs = hard_cap_frac * max(company.annual_budget, 1.0)
                if company.budget_spent_this_year + total_proj_cost > hard_cap_abs:
                    available = max(0.0, hard_cap_abs - company.budget_spent_this_year)
                    if total_proj_cost > 1e-6:
                        scale = available / total_proj_cost
                        invest_frac *= scale
                        capex_cost = company.compute_investment_cost(tech_idx, invest_frac, year)
                        total_proj_cost = capex_cost + _estimate_decommission_cost(company, invest_frac)
                        budget_clipped = True

            if total_proj_cost > budget_remaining and total_proj_cost > 1e-6:
                invest_frac *= budget_remaining / total_proj_cost
                capex_cost = company.compute_investment_cost(tech_idx, invest_frac, year)
                budget_clipped = True

            capex_remaining = max(0.0, company.capex_throughput - company.capex_spent_this_year)
            if capex_cost > capex_remaining and capex_cost > 1e-6:
                invest_frac *= capex_remaining / capex_cost
                capex_clipped = True

            # Optional green-finance boost can recover clipped investment.
            if (budget_clipped or capex_clipped) and company._gf_enabled and requested_invest_frac > invest_frac:
                max_total_with_loan = budget_remaining + company.green_loan_headroom
                max_capex_with_boost = capex_remaining + company.green_capex_headroom
                recovered_frac = _find_recovered_invest_frac(
                    company=company,
                    tech_idx=tech_idx,
                    lo_frac=invest_frac,
                    hi_frac=requested_invest_frac,
                    max_total_cost=max_total_with_loan,
                    max_capex_cost=max_capex_with_boost,
                )
                if recovered_frac > invest_frac + 1e-9:
                    recovered_total = company.compute_investment_cost(tech_idx, recovered_frac, year)
                    recovered_total += _estimate_decommission_cost(company, recovered_frac)
                    extra_invest_cost = max(0.0, recovered_total - budget_remaining)
                    company.record_green_loan(extra_invest_cost)
                    invest_frac = recovered_frac

            invest_frac = float(np.clip(invest_frac, 0.0, company.max_invest_frac))
            invest_frac = min(invest_frac, company.fossil_frac)
            invest_fracs[i] = invest_frac

            invest_costs[i] = company.plan_investment(tech_choice, invest_frac, year)
            company.prev_invest_frac = invest_frac
            # Apply any cancellation recovery as a credit to invest_costs
            invest_costs[i] -= cancel_recoveries[i]

        # Store for phase 2
        self._phase1_allocations = allocations
        self._phase1_payments = payments
        self._phase1_invest_costs = invest_costs
        self._phase1_mac_costs = mac_costs
        self._phase1_log = log

        # C2: Compute bid affordability for next year's observation
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
                emission_shock=float(epsilons[i]),   # P5: shock in obs
                payment=float(payments[i]),           # for auction_savings dim
                collateral_locked_norm=float(np.clip(
                    self._collateral_locked[i] / max(self.companies[i].annual_budget, 1e-6),
                    0.0, 1.0,
                )),
                current_holdings=float(self.holdings[i] + allocations[i]),
                current_year=year,
            )
            for i in range(self.n_agents)
        ]) if self.n_agents > 0 else np.zeros((0,), dtype=np.float32)

        # Log P5/P6 values for year-level diagnostics
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
    # Split Rewards: Auction-phase intermediate reward (v7.6)
    # ------------------------------------------------------------------

    def compute_auction_rewards(self) -> np.ndarray:
        """
        Compute per-agent intermediate reward for the auction phase.

        Pure cost signal: negative sum of all phase-1 costs normalized by
        a fixed scale (REWARD_SCALE = 1000 M€). No shaping terms — the
        penalty in phase 2 provides the natural gradient for winning
        sufficient allowances.

        Must be called after step_auction() and before step_secondary().

        Returns
        -------
        r_auction : np.ndarray, shape (n_agents,)
            Auction-phase reward per learning agent (negative = cost).
        """
        REWARD_SCALE = 1000.0
        r_auction = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            company = self.companies[i]

            auction_cost = float(self._phase1_payments[i])
            investment_cost = float(self._phase1_invest_costs[i])
            opex_delta = company.compute_operational_cost(self.current_year) - company.baseline_opex
            mac_cost_i = float(self._phase1_mac_costs[i])
            collateral_cost_i = float(self._collateral_locked[i]) * float(
                self.config["auction"]["collateral"].get("collateral_rate", 0.05))

            total_cost = (auction_cost + collateral_cost_i + investment_cost
                          + opex_delta + mac_cost_i)

            r_auction[i] = -(total_cost / REWARD_SCALE)

            self._last_auction_reward_channels[i] = {
                "auction_cost": float(auction_cost / REWARD_SCALE),
                "collateral_cost": float(collateral_cost_i / REWARD_SCALE),
                "investment_cost": float(investment_cost / REWARD_SCALE),
                "opex_delta": float(opex_delta / REWARD_SCALE),
                "mac_cost": float(mac_cost_i / REWARD_SCALE),
            }
        return r_auction

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
        self.last_secondary_volume = secondary_volume  # P8: track for obs

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

        # Holdings after secondary market
        holdings = self.holdings + allocations + trade_qtys

        # Collateral cost: rate × locked collateral from Phase 1 (E2/E4).
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

        # Use P5-shocked realized emissions for compliance
        #realized_emissions = self._current_emissions

        # 6. Compliance (against realized emissions + carry-forward obligations)
        # Capture old carry-forward before it gets updated
        #old_carry_forward = np.array([c._carry_forward for c in self.companies])
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

        # P4: Update fossil fraction history (for lock-in penalty)
        for i, company in enumerate(self.companies):
            if not active_mask[i]:
                self._fossil_frac_history[i] = []
                continue
            self._fossil_frac_history[i].append(company.fossil_frac)
            if len(self._fossil_frac_history[i]) > 3:
                self._fossil_frac_history[i].pop(0)

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
        })

        # ── Per-agent per-year diagnostics ───────────────────────────────────
        # Diagnostic fields for validation (Run 1/2 from validation sequence).
        price_ma3_now = self._compute_price_ma3()
        per_agent_diag = {}
        for i, company in enumerate(self.companies):
            if not active_mask[i]:
                continue
            annual_need_i = max(company.compute_estimate_need() + company._carry_forward, 1e-6)
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

        self.episode_log.append(log)

        # 9. Advance year + AR(1) price
        if self._reserve_anchor == "secondary":
            self._price_history.append(secondary_clearing)
        rho = self.config["price"].get("ar1_persistence", 0.85)
        price_floor = self.config["price"].get("ar1_floor",
                        self.config["price"].get("price_floor", 50.0))
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

        obs_next = self._get_obs_phase1()   # shape (n_agents, obs_dim)
        # F2: Compute per-agent diagnostic scores and expose via info dict
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
    # Secondary market — double auction (P8 improvements)
    # ------------------------------------------------------------------

    def _settle_double_auction(self, allocations, secondary_prices,
                                secondary_qtys, clearing_price):
        """
        Double auction with P8 improvements:
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
        # P8: spread tolerance as fraction of clearing price
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

                # P8: trade if buyer_price + spread_tol >= seller_price
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
    # Reward function (P3 + P4 + P8 improvements)
    # ------------------------------------------------------------------

    def _compute_rewards(self, payments, trade_costs, penalties,
                         invest_costs, emissions, clearing_price,
                         mac_costs=None, collateral_costs=None,
                         precompliance_holdings=None,
                         old_carry_forward=None, active_mask=None,
                         trade_qtys=None):
        """
        Simplified reward (v8.0) with v7.12 rollback updates.

        R_i = w_cost * (-cost_norm) + w_green * esg_signal - penalty_norm
              + terminal_bank + terminal_queue (final year)
              - terminal_debt (final year)

        Key points:
          - Fixed REWARD_SCALE (1000 M EUR) for shared critic stability.
          - Penalties stay separated from costs and always apply at full strength.
          - No shaping rewards; only economic cost and ESG signal drive learning.
          - Soft budget/capex penalties remain in cost_norm until hard-gate
            coverage is fully audited.
        """
        REWARD_SCALE = 1000.0
        rewards = np.zeros(self.n_total)
        base_rewards = np.zeros(self.n_total)
        terminal_bank_values = np.zeros(self.n_total)
        terminal_queue_values = np.zeros(self.n_total)
        reward_cfg = self.config.get("reward", {})
        esg_cfg = self.config.get("esg", {})
        esg_enabled = esg_cfg.get("enabled", False)
        esg_scale = float(esg_cfg.get("scale", 2.0))

        if mac_costs is None:
            mac_costs = np.zeros(self.n_total)
        if collateral_costs is None:
            collateral_costs = np.zeros(self.n_total)

        remaining_years = max(1, self.n_years - self.current_year)

        for i, company in enumerate(self.companies):
            if active_mask is not None and not bool(active_mask[i]):
                continue

            auction_cost = float(payments[i])
            secondary_cost = float(trade_costs[i])
            penalty_cost = float(penalties[i])
            investment_cost = float(invest_costs[i])
            opex_delta = company.compute_operational_cost(self.current_year) - company.baseline_opex
            mac_cost_i = float(mac_costs[i])
            collateral_cost_i = float(collateral_costs[i])
            loan_interest_cost = company.compute_green_loan_cost()

            # Keep operational spending separate from regulatory penalty spending.
            company.record_spending(
                auction_cost
                + secondary_cost
                + investment_cost
                + mac_cost_i
                + collateral_cost_i
                + loan_interest_cost
            )
            company.record_capex_spending(investment_cost)
            budget_penalty = company.compute_budget_penalty()
            capex_penalty = company.compute_capex_penalty()

            total_cost = (
                auction_cost
                + secondary_cost
                + investment_cost
                + opex_delta
                + budget_penalty
                + capex_penalty
                + mac_cost_i
                + collateral_cost_i
                + loan_interest_cost
            )
            cost_norm = total_cost / REWARD_SCALE
            penalty_norm = penalty_cost / REWARD_SCALE

            esg_signal = 0.0
            if esg_enabled and company.initial_ef > 1e-6:
                ef_ratio = (company.initial_ef - company.weighted_emission_factor) / company.initial_ef
                time_ratio = remaining_years / self.n_years
                esg_signal = esg_scale * ef_ratio * time_ratio

            base_reward = float(
                company.w_cost * (-cost_norm)
                + company.w_green * esg_signal
                - penalty_norm
            )
            base_rewards[i] = base_reward
            rewards[i] = base_reward

            self._last_reward_channels[i] = {
                "cost_norm": float(cost_norm),
                "penalty_norm": float(penalty_norm),
                "esg_signal": float(esg_signal),
                "base_reward": float(base_reward),
            }

        is_final_year = self.current_year >= self.n_years - 1
        terminal_bank = bool(reward_cfg.get("terminal_bank_value", False))
        terminal_queue = bool(reward_cfg.get("terminal_queue_value", True))

        if is_final_year:
            gamma_discount = float(self.config["ppo"].get("gamma", 0.99))
            terminal_payoff_years = float(reward_cfg.get("terminal_payoff_years", 5.0))
            pen_cfg = self.config["penalty"]
            eff_penalty = pen_cfg["rate"] * self._inflation_factor(self.current_year)
            terminal_price = max(clearing_price, self.last_secondary_price, eff_penalty * 0.8)

            for i, company in enumerate(self.companies):
                if active_mask is not None and not bool(active_mask[i]):
                    continue

                if terminal_bank:
                    annual_need = max(company.compute_estimate_need(), 0.1)
                    capped_holdings = min(self.holdings[i], 2.0 * annual_need)
                    ratio = capped_holdings / annual_need
                    bank_value = np.log1p(ratio) * annual_need * terminal_price / REWARD_SCALE
                    rewards[i] += bank_value
                    base_rewards[i] += bank_value
                    terminal_bank_values[i] = bank_value

                # Completion discount prevents end-of-episode queue gaming.
                if terminal_queue:
                    queue_value = 0.0
                    for item in company._construction_queue:
                        tech_idx = int(item.get("tech_idx", -1))
                        if tech_idx < 0 or tech_idx >= len(company.emission_factors):
                            continue

                        years_to_completion = max(
                            0,
                            int(item.get("completion_year", self.current_year)) - self.current_year,
                        )
                        tech_delay = max(float(company.deploy_delays[tech_idx]), 1.0)
                        completion_fraction = float(
                            np.clip(1.0 - (years_to_completion / tech_delay), 0.0, 1.0)
                        )
                        if completion_fraction <= 0.0:
                            continue

                        effective_remaining = max(0.0, terminal_payoff_years - float(years_to_completion))
                        if effective_remaining <= 0.0:
                            continue

                        delta_ef = company.weighted_emission_factor - company.emission_factors[tech_idx]
                        if delta_ef <= 0:
                            continue

                        frac_delta = float(item.get("frac_delta", 0.0))
                        if frac_delta <= 0.0:
                            continue

                        annual_saving_mt = (delta_ef * frac_delta * company.output_mwh) / 1e6
                        discount = gamma_discount ** years_to_completion
                        queue_value += (
                            annual_saving_mt
                            * effective_remaining
                            * discount
                            * completion_fraction
                            * terminal_price
                            / REWARD_SCALE
                        )

                    rewards[i] += queue_value
                    base_rewards[i] += queue_value
                    terminal_queue_values[i] = queue_value

                if company._carry_forward > 0:
                    debt_penalty = (company._carry_forward * terminal_price * 1.5) / REWARD_SCALE
                    rewards[i] -= debt_penalty
                    base_rewards[i] -= debt_penalty

        self._last_terminal_bank_values = terminal_bank_values
        self._last_terminal_queue_values = terminal_queue_values
        self._last_terminal_liquidation_values = terminal_bank_values + terminal_queue_values
        self._last_reward_base_values = base_rewards
        self._last_reward_shaping_values = np.zeros(self.n_total)

        return rewards

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def compute_diagnostic_score(self, agent_id: int = None) -> dict:
        """
        F2: Compute interpretable diagnostic scores for agents.

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
        """P1: 3-year moving average of clearing price.
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
        Base: 33D. With opponent modeling: 33 + 5*(N_total-1) dims.
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
            price_ma3=price_ma3,
        )
        # Include both rollover channels that are added in step_auction.
        unsold_pending = float(getattr(self.cap_schedule, "_unsold_rollover_pending", 0.0))
        defaulted_pending = float(self._defaulted_volume_pending)
        this_year_auction_volume += unsold_pending + defaulted_pending
        max_rollover_mult = float(getattr(self.cap_schedule, "max_rollover_multiplier", 1.5))
        this_year_auction_volume = min(this_year_auction_volume, cap_t * max_rollover_mult)

        msr_reserve = self.cap_schedule.msr_reserve()
        suspension_length = max(1, int(self.config["auction"].get("suspension_length", 2)))

        # Pre-compute 5D public info for ALL participants (learning + bots)
        if self._opponent_modeling and self.n_total > 1:
            public_infos = []
            for i, company in enumerate(self.companies):
                if self._is_agent_active(i):
                    public_infos.append(company.get_public_info())
                else:
                    public_infos.append({
                        "emissions": 0.0,
                        "carry_forward": 0.0,
                        "green_frac": 0.0,
                        "fossil_frac": 0.0,
                        "queue_total": 0.0,
                        "is_active": 0.0,
                    })

        obs_list = []
        for i in range(self.n_agents):  # only learning agents get observations
            c = self.companies[i]
            if self._opponent_modeling and self.n_total > 1:
                opp_parts = []
                for j in range(self.n_total):  # all participants as opponents
                    if j != i:
                        pi = public_infos[j]
                        opp_parts.extend([
                            pi["emissions"],
                            pi["carry_forward"],
                            pi["green_frac"],
                            pi["fossil_frac"],
                            pi["queue_total"],
                            pi.get("is_active", 1.0),
                        ])
                opponent_obs = np.array(opp_parts, dtype=np.float32)
            else:
                opponent_obs = None

            susp_norm = float(self._suspension_remaining[i]) / suspension_length

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
                suspension_remaining_norm=susp_norm,
                collateral_load_last=float(self._last_collateral_load[i]),
                bid_affordability_last=float(self._bid_affordability[i]),
                n_years=self.n_years,
            )
            obs_list.append(obs_i)

        if not obs_list:
            return np.zeros((0,), dtype=np.float32)
        return np.stack(obs_list)
