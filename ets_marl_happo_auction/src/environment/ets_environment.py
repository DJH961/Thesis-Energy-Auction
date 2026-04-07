"""
ets_environment.py
==================
Two-phase EU ETS environment for multi-agent RL with technology-specific
energy mix, real-data-grounded investment costs, and construction queues.

Each year is split into two decision phases:
  Phase 1 (Auction + Investment):
    - Agents observe market state (20+2*(N-1) dim with opponent modeling)
    - Decide: [p1, q1, p2, q2, p3, q3, invest_frac, tech_choice_logit0..2]
    - 3-tranche bid ladder: each (p, q) pair is an independent price/coverage
      bid submitted to the uniform-price auction. This mirrors the demand
      curves used in real EEX/ICE call auctions.
    - Auction clears, investments are planned
    - Returns enriched observation (phase1_dim+4) with auction results

  Phase 2 (Secondary Market — Uniform-Price Call Auction):
    - Agents observe auction results (phase1_dim+4 dim)
    - Decide: [secondary_price, secondary_quantity]
    - All bids/offers are collected and cleared at a single uniform price
      (call-auction / clearinghouse mechanism), maximising social surplus.
    - Returns reward and next year's Phase 1 observation

Action space (Phase 1): 10D continuous
  [p1, q1, p2, q2, p3, q3, invest_frac, tech_logit_onshore, tech_logit_offshore, tech_logit_solar]
  Each (p_k, q_k) tranche: p_k is bid price, q_k is coverage multiplier on need.
  tech_choice is derived by argmax of the 3 logits (discrete from continuous)

Action space (Phase 2): 2D continuous
  [price_abs (EUR/t), quantity]  — positive qty = buy, negative = sell
  Cleared via uniform-price call auction (aggregate supply/demand intersection).

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
        self._phase1_tranche_prices_raw = None   # pre-B1-sort, action-slot order
        self._phase1_tranche_quantities_raw = None

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

    def _compute_tranche_fill_ratios(self, allocations: np.ndarray,
                                     clearing_price: float) -> list:
        """
        D1: Compute per-tranche fill ratios analytically.

        For a uniform-price auction: bids above clearing_price are fully
        filled (up to per-agent holding limits). Bids below clearing_price
        receive zero fill. At exactly clearing_price, filling is partial.

        We reconstruct per-tranche fills from stored tranche prices and
        quantities alongside the actual per-agent total allocation.

        Returns a list of length n_agents, each element is a list of 3
        fill ratios [fill_t1, fill_t2, fill_t3] in [0, 1].
        """
        N_TRANCHES = 3
        results = []
        for i in range(self.n_agents):
            prices_i = self._phase1_tranche_prices[i]   # list of 3 prices (ascending, after B1)
            qtys_i = self._phase1_tranche_quantities[i]  # list of 3 qtys (Mt)
            total_alloc = float(allocations[i])

            fill_ratios = [0.0] * N_TRANCHES
            remaining_alloc = total_alloc

            # Distribute allocation from highest-price tranche down (price-priority filling)
            # Since tranches are sorted ascending (B1), fill in reverse order
            for t in sorted(range(N_TRANCHES), key=lambda k: -prices_i[k]):
                if remaining_alloc <= 1e-9:
                    break
                qty_t = float(qtys_i[t])
                if qty_t < 1e-9:
                    fill_ratios[t] = 0.0
                    continue
                if prices_i[t] < clearing_price - 1e-9:
                    # Below clearing → no fill
                    fill_ratios[t] = 0.0
                else:
                    # At or above clearing → fill from remaining allocation
                    fill = min(qty_t, remaining_alloc)
                    fill_ratios[t] = fill / qty_t
                    remaining_alloc -= fill
                    remaining_alloc = max(0.0, remaining_alloc)

            results.append(fill_ratios)
        return results

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

        # Reward channel diagnostics
        self._last_reward_channels = {}
        self._last_auction_reward_channels = {}
        self._last_per_agent_diag = {}

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
        """Generate Phase-1 actions for all bot agents using heuristic_policy.

        Bot heuristic produces 6D [mid_price, total_qty_mult, invest_frac, t0, t1, t2].
                We expand to 10D 3-tranche format using configurable bot tranche settings:
                    - Quantity split from bots.tranche_qty_split (normalized)
                    - Price spread from bots.tranche_price_* config
                    - T3 spread widens with urgency in late-episode years
                    - Budget-stressed bots drop T1 first, then scale T2/T3 if needed

        Tranches are already sorted ascending by price (B1 invariant satisfied).
                Tranche prices are clipped to [price_min, price_max].
        """
        if self.n_bots == 0:
            return np.zeros((0, 10), dtype=np.float32)
        price_ma3 = self._compute_price_ma3()
        reserve = self._compute_dynamic_reserve()
        price_min = float(self.config["auction"]["price_min"])
        price_max = float(self.config["auction"]["price_max"])
        infl_factor = self._inflation_factor(self.current_year)
        bot_cfg = self.config.get("bots", {})
        urgency_denoms = bot_cfg.get("urgency_denominators", [1.5] * self.n_bots)
        budget_stress_qty_mult = float(self._enhanced_noise_cfg.get("budget_stress_qty_mult", 0.65))
        qty_low = float(self.config["auction"].get("qty_mult_low", 0.3))
        qty_high = float(self.config["auction"].get("qty_mult_high", 2.0))
        actions = np.zeros((self.n_bots, 10), dtype=np.float32)
        for b in range(self.n_bots):
            if b >= self._n_active_bots:
                actions[b] = np.array([price_min, 0.0, price_min, 0.0, price_min, 0.0,
                                       0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                continue

            idx = self.n_agents + b
            urgency_denom = urgency_denoms[b] if b < len(urgency_denoms) else 1.5
            action6 = heuristic_policy.auction_action(
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
                action6[1] = float(np.clip(action6[1] * budget_stress_qty_mult, qty_low, qty_high))

            annual_need = max(
                self.companies[idx].compute_estimate_need() + self.companies[idx]._carry_forward,
                0.1,
            )
            budget_remaining = max(
                0.0,
                float(self.companies[idx].annual_budget - self.companies[idx].budget_spent_this_year),
            )

            # C1: Expand 6D -> 10D using configurable tranche ladder.
            p_mid = float(action6[0])
            q_total = float(action6[1])
            tranche_prices, tranche_qty_mults = heuristic_policy.build_tranche_ladder(
                mid_price=p_mid,
                total_qty_mult=q_total,
                config=self.config,
                price_min=price_min,
                price_max=price_max,
                current_year=self.current_year,
                n_years=self.n_years,
                bank=float(self.holdings[idx]),
                annual_need=annual_need,
                urgency_multiplier=float(self._bot_urgency_mult[b]),
                urgency_denom=urgency_denom,
                reserve_price=reserve,
                available_budget=budget_remaining,
            )
            actions[b] = np.array([tranche_prices[0], tranche_qty_mults[0],
                                   tranche_prices[1], tranche_qty_mults[1],
                                   tranche_prices[2], tranche_qty_mults[2],
                                   action6[2], action6[3], action6[4], action6[5]],
                                  dtype=np.float32)
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

    def step_auction(self, auction_actions: np.ndarray):
        """
        Phase 1: Execute auction and green investments.

        Parameters
        ----------
        auction_actions : np.ndarray, shape (n_learning, 10)
            Actions for learning agents only.
            [p1, q1, p2, q2, p3, q3, invest_frac, tech_logit0, tech_logit1, tech_logit2]
            3-tranche bid ladder: each (p_k, q_k) pair is an independent
            price/coverage bid. Bot actions are generated internally via
            heuristic_policy (still 6D, expanded using configurable tranche
            split/spread and budget-aware dropping).

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
            init_p = self.config["price"]["initial_expected"]
            hist = list(self._price_history)
            while len(hist) < 3:
                hist.insert(0, init_p)
            smoothed_price = float(np.mean(hist[-3:]))
            active_companies = [c for c in self.companies if self._is_agent_active(c.agent_id)]
            # System-wide average EF (kept for observation space)
            system_ef = float(np.mean([c.weighted_emission_factor for c in active_companies]))
            # M1: Marginal EF — EF of most carbon-intensive technology with significant system share.
            # Carbon cost pass-through in electricity markets prices off the marginal setter (typically
            # coal when it has material system presence). Using system_ef understates coal revenue.
            # Reference: Fabra & Reguant (2014), Sijm et al. (2006).
            marginal_ef = _compute_marginal_ef(active_companies, self.config)
            for c in active_companies:
                c.set_annual_budget(
                    c.compute_dynamic_budget(smoothed_price, marginal_ef, self.current_year)
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

        # 7. Auction — 3-tranche bid ladder
        # Each agent submits 3 (price, qty_multiplier) pairs.
        # These are expanded into separate bid rows for the clearing engine.
        price_min = self.config["auction"]["price_min"]
        price_max = self.config["auction"]["price_max"]
        qty_mult_low = self.config["auction"].get("qty_mult_low", 0.3)
        qty_mult_high = self.config["auction"].get("qty_mult_high", 2.0)
        lot_size = self.config["auction"].get("lot_size", 0.0)

        N_TRANCHES = 3
        # Collect per-tranche bids: list of [agent_id, quantity, price]
        all_bid_rows = []
        bid_qty_multipliers = np.zeros(self.n_total)
        estimate_needs = np.zeros(self.n_total)
        bid_coverages = np.zeros(self.n_total)
        # Store aggregate (weighted-average) price and total qty per agent for logging
        agent_total_qty = np.zeros(self.n_total)
        agent_wavg_price = np.zeros(self.n_total)
        # Store individual tranche prices and quantities per agent [agent][tranche]
        # _prices/_quantities: B1-sorted (ascending) — used for clearing and diagnostics
        # _prices_raw/_quantities_raw: pre-sort order (action slot 0,1,2 → tranche 0,1,2)
        # Raw slots preserve policy gradient identity so backprop flows to the correct
        # action component; B1 sort is for clearing + diagnostics only.
        tranche_prices = [[0.0] * N_TRANCHES for _ in range(self.n_total)]
        tranche_quantities = [[0.0] * N_TRANCHES for _ in range(self.n_total)]
        tranche_prices_raw = [[0.0] * N_TRANCHES for _ in range(self.n_total)]
        tranche_quantities_raw = [[0.0] * N_TRANCHES for _ in range(self.n_total)]

        # E1: Aggregate bid volume cap — total bid ≤ 3× annual_need
        aggregate_bid_cap_mult = float(self.config["auction"].get("aggregate_bid_cap_mult", 3.0))

        for i, company in enumerate(self.companies):
            if not self._is_agent_active(i):
                bid_qty_multipliers[i] = 0.0
                estimate_needs[i] = 0.0
                bid_coverages[i] = 0.0
                continue

            # Base need is intentionally unbuffered (expected emissions + debt);
            # agents learn safety buffers through the bid multiplier itself.
            base_need = max(company.compute_estimate_need() + company._carry_forward, 0.1)
            estimate_needs[i] = base_need

            # Extract raw tranche (price, qty) pairs from action vector
            raw_tranches = []
            for t in range(N_TRANCHES):
                p_raw = float(auction_actions[i, 2 * t])
                q_raw = float(auction_actions[i, 2 * t + 1])
                p_clipped = float(np.clip(p_raw, price_min, price_max))
                q_mult = float(np.clip(q_raw, 0.0, qty_mult_high))
                q_abs = q_mult * base_need
                if lot_size > 0:
                    q_abs = max(lot_size, round(q_abs / lot_size) * lot_size)
                q_abs = max(0.0, q_abs)
                raw_tranches.append((p_clipped, q_abs))

            # B1: Sort tranches ascending by price (T1 = cheapest, T3 = most expensive).
            # B1 sort is for clearing + diagnostics only; raw slots preserve
            # policy gradient identity (action[2t] → tranche t before sort).
            # Save raw (pre-sort) order first.
            for t, (p_raw, q_raw) in enumerate(raw_tranches):
                tranche_prices_raw[i][t] = p_raw
                tranche_quantities_raw[i][t] = q_raw
            raw_tranches.sort(key=lambda pq: pq[0])

            # E1: Cap total bid quantity at aggregate_bid_cap_mult × base_need
            total_unsorted = sum(q for _, q in raw_tranches)
            agg_cap = aggregate_bid_cap_mult * base_need
            if total_unsorted > agg_cap + 1e-9 and total_unsorted > 1e-9:
                scale = agg_cap / total_unsorted
                raw_tranches = [(p, q * scale) for p, q in raw_tranches]

            total_qty_i = 0.0
            price_qty_sum = 0.0
            for t, (p_clipped, q_abs) in enumerate(raw_tranches):
                # Store per-tranche data (after sorting)
                tranche_prices[i][t] = p_clipped
                tranche_quantities[i][t] = q_abs
                if q_abs > 1e-6:
                    all_bid_rows.append([float(i), q_abs, p_clipped])
                    total_qty_i += q_abs
                    price_qty_sum += p_clipped * q_abs

            agent_total_qty[i] = total_qty_i
            if total_qty_i > 1e-6:
                agent_wavg_price[i] = price_qty_sum / total_qty_i
                bid_coverages[i] = total_qty_i / max(base_need, 1e-6)
                bid_qty_multipliers[i] = total_qty_i / base_need
            else:
                agent_wavg_price[i] = price_min

        # E4: Leverage gate — clip total bid notional by leverage_multiplier × cash.
        # For multi-tranche bids, scale all tranches proportionally if total notional
        # (sum of price × qty across tranches) exceeds the leverage limit.
        aq_cfg = self.config["auction"]
        lev_mult = float(aq_cfg.get("leverage_multiplier", 3.0))
        if lev_mult > 0.0:
            for i, company in enumerate(self.companies):
                if not self._is_agent_active(i):
                    continue
                if agent_total_qty[i] < 1e-9 or agent_wavg_price[i] < 1e-9:
                    continue
                cash = max(0.0, float(company.annual_budget - company.budget_spent_this_year))
                total_notional = agent_wavg_price[i] * agent_total_qty[i]
                max_notional = lev_mult * cash
                if total_notional > max_notional and total_notional > 1e-9:
                    scale = max_notional / total_notional
                    for row in all_bid_rows:
                        if int(row[0]) == i:
                            row[1] *= scale
                    agent_total_qty[i] *= scale

        # E2: Solvency sanity check (softened — fixed heuristic is self-consistent).
        # The heuristic's WTP formula already ensures bid value stays within available budget.
        # This block now only logs a warning when collateral would exceed budget threshold.
        # It no longer reshapes bids to avoid double-penalising well-intentioned bids.
        coll_cfg = self.config.get("auction", {}).get("collateral", {})
        if coll_cfg.get("enabled", True):
            coll_frac = float(coll_cfg.get("collateral_fraction",
                                            coll_cfg.get("collateral_rate", 0.05)
                                            * coll_cfg.get("hold_fraction", 0.02)))
            max_coll_share = float(coll_cfg.get("max_collateral_budget_share", 0.50))

            if coll_frac > 0.0:
                for i, company in enumerate(self.companies):
                    if not self._is_agent_active(i):
                        continue
                    total_q = agent_total_qty[i]
                    avg_p = agent_wavg_price[i]
                    if total_q < 1e-6 or avg_p < 1e-6:
                        continue
                    budget_remaining = max(
                        0.0,
                        float(company.annual_budget - company.budget_spent_this_year),
                    )
                    collateral = coll_frac * avg_p * total_q
                    max_collateral = max_coll_share * budget_remaining
                    if collateral > max_collateral and max_collateral > 0 and budget_remaining > 1.0:
                        # Log solvency warning; do not reshape bid (heuristic is self-consistent)
                        import warnings
                        warnings.warn(
                            f"[E2] Agent {i} collateral {collateral:.1f} > "
                            f"max {max_collateral:.1f} (budget_remaining={budget_remaining:.1f}); "
                            f"skipping bid rescale (heuristic WTP is self-consistent).",
                            stacklevel=2,
                        )

        # Store aggregate per-agent bid info for logging (backward-compatible)
        self._phase1_bid_prices = agent_wavg_price.copy()
        self._phase1_bid_quantities = agent_total_qty.copy()
        # Store individual tranche data for logging
        # _phase1_tranche_prices / _quantities: B1-sorted (ascending) — used for clearing
        # _phase1_tranche_prices_raw / _quantities_raw: pre-sort action-slot order
        self._phase1_tranche_prices = tranche_prices
        self._phase1_tranche_quantities = tranche_quantities
        self._phase1_tranche_prices_raw = tranche_prices_raw
        self._phase1_tranche_quantities_raw = tranche_quantities_raw

        # E4: Suspension enforcement — suspended agents cannot bid this round.
        # Decrement suspension counter; bids for suspended agents already filtered since
        # their rows were never added to all_bid_rows (qty=0 rows are dropped).
        suspended_agents_set = set()
        for i in range(self.n_total):
            if self._suspension_remaining[i] > 0:
                suspended_agents_set.add(i)
                self._suspension_remaining[i] -= 1
        # Remove any bid rows from suspended agents
        if suspended_agents_set:
            all_bid_rows = [row for row in all_bid_rows if int(row[0]) not in suspended_agents_set]
            for i in suspended_agents_set:
                agent_total_qty[i] = 0.0

        # Compute effective reserve price (dynamic or static)
        effective_reserve = self._compute_dynamic_reserve()
        self._last_effective_reserve = effective_reserve

        # E4: Pre-bid collateral locking — fraction of margin above reserve per agent.
        coll_frac_e4 = float(coll_cfg.get("collateral_fraction",
                                           coll_cfg.get("collateral_rate", 0.05)
                                           * coll_cfg.get("hold_fraction", 0.02)))
        collateral_locked = np.zeros(self.n_total)
        for i in range(self.n_total):
            if not self._is_agent_active(i) or i in suspended_agents_set:
                continue
            total_q = agent_total_qty[i]
            avg_p = agent_wavg_price[i]
            if avg_p > 1e-6 and total_q > 1e-6:
                collateral_locked[i] = (
                    coll_frac_e4 * max(0.0, avg_p - effective_reserve) * total_q
                )

        # Store for Phase 2: collateral opportunity cost uses locked amount directly.
        self._collateral_locked = collateral_locked.copy()

        # Update per-agent collateral load for next year's Phase 1 obs.
        for i in range(self.n_total):
            budget_i = float(self.companies[i].annual_budget)
            self._last_collateral_load[i] = float(np.clip(
                collateral_locked[i] / max(budget_i, 1e-6), 0.0, 1.0,
            ))

        # Build bids array from collected tranche rows
        if all_bid_rows:
            bids = np.array(all_bid_rows, dtype=float)
        else:
            bids = np.zeros((0, 3), dtype=float)
        clearing_price, allocations, payments, auction_stats = market_clearing_ets(
            bids=bids,
            q_cap=auction_volume,
            reserve_price=effective_reserve,
            max_agent_share=self.config["auction"].get("max_agent_share", 1.0),
            rng=self.rng,
            cancel_under_subscribed=self.config["auction"].get(
                "cancel_under_subscribed", False),
            n_agents=self.n_total,
        )

        # E4: Post-clearing settlement — check each winner can pay; handle defaults.
        suspension_length = int(self.config.get("budget", {}).get(
            "suspension_length",
            self.config["auction"].get("suspension_length", 1),
        ))
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
        # Per-agent lists used by the training console for split RL/bot event board.
        auction_stats["defaults_agents"]          = sorted(int(i) for i, d in enumerate(defaults_mask) if d)
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

            # Continuous linear mapping: [-1, 1] → [0, max_invest_frac]
            # Eliminates the dead zone where negative actions all map to 0.
            invest_frac = float((auction_actions[i, 6] + 1.0) / 2.0) * company.max_invest_frac
            invest_frac = float(np.clip(invest_frac, 0.0, company.max_invest_frac))
            requested_invest_frac = invest_frac
            tech_logits = auction_actions[i, 7:10]
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
                                           coll_cfg.get("collateral_rate", 0.05)
                                           * coll_cfg.get("hold_fraction", 0.02)))
        for i in range(self.n_total):
            c = self.companies[i]
            budget_remaining = max(c.annual_budget - c.budget_spent_this_year, 1e-6)
            bid_total = bid_prices[i] * bid_qtys[i] * (1.0 + coll_frac_c2)
            self._bid_affordability[i] = float(np.clip(bid_total / budget_remaining, 0.0, 1.0))

        # D1: Compute per-tranche fill ratios for each learning agent
        # This uses an analytic reconstruction: given clearing_price and each
        # tranche's (price, qty) pair, compute expected fill without re-running clearing.
        tranche_fills = self._compute_tranche_fill_ratios(
            allocations, clearing_price)

        # 10. Build phase 2 observations (learning agents only)
        obs_phase1 = self._get_obs_phase1()   # shape (n_agents, obs_dim)
        pn = float(self.config["auction"]["price_max"])

        obs_phase2 = np.stack([
            self.companies[i].get_observation_phase2(
                obs_phase1=obs_phase1[i],
                allocation=allocations[i],
                clearing_price=clearing_price,
                emissions=realized_emissions[i],
                banked=self.holdings[i],
                emission_shock=float(epsilons[i]),
                payment=float(payments[i]),
                tranche_fill_ratios=tranche_fills[i],
                tranche_price_vs_clearing=[
                    (self._phase1_tranche_prices[i][t] - clearing_price) / max(pn, 1.0)
                    for t in range(3)
                ],
                collateral_locked_norm=float(np.clip(
                    self._collateral_locked[i] / max(self.companies[i].annual_budget, 1e-6),
                    0.0, 1.0,
                )),
                current_holdings=float(self.holdings[i] + allocations[i]),
                current_year=year,
            )
            for i in range(self.n_agents)
        ])

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
    # Split Rewards: Auction-phase intermediate reward (v8.3)
    # ------------------------------------------------------------------

    def compute_auction_rewards(self) -> np.ndarray:
        """
        Compute per-agent intermediate reward for the auction phase.

        For the auction version, this collapses all tranche costs into a single
        auction-phase reward (practical shortcut). All tranche costs are already
        summed in _phase1_payments by the clearing mechanism.

        This captures costs attributable to auction-phase decisions:
        auction payment, collateral, investment, OPEX delta, MAC cost,
        loan interest, and a prospective capex-throughput penalty estimate.
        Budget penalty is intentionally excluded here because final annual
        spending is only known after secondary market settlement.
        Normalized by annual_budget for consistency with collateral_load_last obs[29].

        Must be called after step_auction() and before step_secondary().

        Returns
        -------
        r_auction : np.ndarray, shape (n_agents,)
            Auction-phase reward per learning agent (negative = cost).
        """
        r_auction = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            company = self.companies[i]
            budget_divisor = max(company.annual_budget, 1.0)

            auction_cost = float(self._phase1_payments[i])
            investment_cost = float(self._phase1_invest_costs[i])
            opex_delta = company.compute_operational_cost(self.current_year) - company.baseline_opex
            mac_cost_i = float(self._phase1_mac_costs[i])
            collateral_cost_i = float(self._collateral_locked[i]) * float(
                self.config["auction"]["collateral"].get("collateral_rate", 0.05))
            loan_interest_cost = float(company.compute_green_loan_cost())
            # loan_interest_cost is recorded in _compute_rewards() via record_spending(); not double-counted here

            projected_capex_spend = float(company.capex_spent_this_year + investment_cost)
            capex_overshoot = max(0.0, projected_capex_spend - float(company.capex_throughput))
            if capex_overshoot > 1e-6:
                capex_ratio = capex_overshoot / max(float(company.capex_throughput), 1e-6)
                capex_penalty = float(company.capex_overspend_coef) * (capex_ratio ** 2) * float(company.capex_throughput)
            else:
                capex_penalty = 0.0

            total_cost = (auction_cost + collateral_cost_i + investment_cost
                          + opex_delta + mac_cost_i + loan_interest_cost
                          + capex_penalty)
            baseline_cost = company.compute_estimate_need() * self._phase1_clearing_price / budget_divisor
            r_auction[i] = -(total_cost / budget_divisor) + baseline_cost

            self._last_auction_reward_channels[i] = {
                "auction_cost": float(auction_cost / budget_divisor),
                "collateral_cost": float(collateral_cost_i / budget_divisor),
                "investment_cost": float(investment_cost / budget_divisor),
                "opex_delta": float(opex_delta / budget_divisor),
                "mac_cost": float(mac_cost_i / budget_divisor),
                "loan_interest": float(loan_interest_cost / budget_divisor),
                "capex_penalty": float(capex_penalty / budget_divisor),
                "baseline_cost": float(baseline_cost),
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

        # Collateral opportunity cost: rate × locked collateral from Phase 1 (E2/E4).
        # self._collateral_locked is computed in step_auction() as:
        #   collateral_fraction × max(0, avg_bid_price − reserve) × bid_qty
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
            "bid_quantities": self._phase1_bid_quantities.tolist() if self._phase1_bid_quantities is not None else [],
            "tranche_prices_sorted": self._phase1_tranche_prices,
            "tranche_quantities_sorted": self._phase1_tranche_quantities,
            "tranche_prices_raw": self._phase1_tranche_prices_raw,
            "tranche_quantities_raw": self._phase1_tranche_quantities_raw,
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
            revenue_i = company.compute_revenue(price_ma3_now, self._last_marginal_ef, inf_i)
            compliance_cost_i = float(payments[i]) + float(trade_costs[i])
            coverage_post = float(self.holdings[i]) / annual_need_i
            per_agent_diag[i] = {
                "wtp": float(getattr(company, "_last_wtp", float("nan"))),
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
            }
        self._last_per_agent_diag = per_agent_diag
        log["per_agent_diag"] = per_agent_diag

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
        """Single-call step for backward compat. actions shape (n_learning, 12)."""
        obs2, _ = self.step_auction(actions[:, :10])
        return self.step_secondary(actions[:, 10:])

    # ------------------------------------------------------------------
    # Secondary market — Uniform-Price Call Auction (Clearinghouse)
    # ------------------------------------------------------------------

    def _settle_double_auction(self, allocations, secondary_prices,
                                secondary_qtys, clearing_price):
        """
        Uniform-Price Call Auction (v8.0):
        All agent bids/offers are aggregated into supply and demand curves.
        The intersection determines a single uniform clearing price at which
        all overlapping volume clears. This is how real EEX/ICE daily call
        auctions (spot fixing) work — it maximises social surplus and finds
        the exact market equilibrium a CDA would discover over time.

        Returns (trade_costs, trade_qtys, secondary_clearing_price, total_volume,
                 liquidity_pool_info)
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

        # ── Collect buy and sell orders ───────────────────────────────
        buy_orders = []   # (agent_id, price, qty)
        sell_orders = []  # (agent_id, price, qty)

        for i in range(self.n_total):
            qty = float(secondary_qtys[i])
            price = float(secondary_prices[i])
            if qty > 1e-6:
                buy_orders.append((i, price, qty))
            elif qty < -1e-6:
                # Enforce no-short-selling: subtract expected compliance need
                max_sell = max(0.0, float(allocations[i]) + float(max(0.0, self.holdings[i]))
                              - float(self._current_emissions[i])
                              - float(self.companies[i]._carry_forward))
                sell_qty = min(abs(qty), max_sell)
                if sell_qty > 1e-6:
                    sell_orders.append((i, price, sell_qty))

        if not buy_orders or not sell_orders:
            return trade_costs, trade_qtys, clearing_price, 0.0, liquidity_pool_info

        # ── Build aggregate demand curve (sorted descending by price) ──
        # and aggregate supply curve (sorted ascending by price)
        demand = sorted(buy_orders, key=lambda x: -x[1])   # highest WTP first
        supply = sorted(sell_orders, key=lambda x: x[1])     # lowest ask first

        # ── Find intersection: uniform clearing price ─────────────────
        # Walk both curves simultaneously. The clearing price is the price
        # at which cumulative demand >= cumulative supply cross.
        # We step through all price levels and find where demand = supply.

        # Gather all unique price levels from both sides
        all_prices = sorted(set([d[1] for d in demand] + [s[1] for s in supply]))

        best_volume = 0.0
        best_price = clearing_price  # fallback

        for p in all_prices:
            # Demand at price p: all buy orders with price >= p
            d_vol = sum(qty for _, bp, qty in demand if bp >= p)
            # Supply at price p: all sell orders with price <= p
            s_vol = sum(qty for _, sp, qty in supply if sp <= p)
            # Cleared volume is the minimum
            cleared = min(d_vol, s_vol)
            if cleared > best_volume:
                best_volume = cleared
                best_price = p

        if best_volume < 1e-9:
            return trade_costs, trade_qtys, clearing_price, 0.0, liquidity_pool_info

        uniform_price = best_price

        # ── Allocate cleared volume to agents at uniform price ────────
        # Buyers: fill orders with price >= uniform_price (pro-rata if excess demand)
        eligible_buys = [(i, price, qty) for i, price, qty in demand if price >= uniform_price]
        eligible_sells = [(i, price, qty) for i, price, qty in supply if price <= uniform_price]

        total_buy_qty = sum(qty for _, _, qty in eligible_buys)
        total_sell_qty = sum(qty for _, _, qty in eligible_sells)

        # The cleared volume is limited by the smaller side
        cleared_volume = min(total_buy_qty, total_sell_qty, best_volume)

        # Pro-rata allocation on the excess side
        if total_buy_qty > cleared_volume and total_buy_qty > 1e-9:
            buy_scale = cleared_volume / total_buy_qty
        else:
            buy_scale = 1.0

        if total_sell_qty > cleared_volume and total_sell_qty > 1e-9:
            sell_scale = cleared_volume / total_sell_qty
        else:
            sell_scale = 1.0

        # Execute at uniform clearing price
        for agent_id, _, qty in eligible_buys:
            filled = qty * buy_scale
            cost = filled * (uniform_price + tx_cost)
            trade_costs[agent_id] += cost
            trade_qtys[agent_id] += filled

        for agent_id, _, qty in eligible_sells:
            filled = qty * sell_scale
            revenue = filled * (uniform_price - tx_cost)
            trade_costs[agent_id] -= revenue
            trade_qtys[agent_id] -= filled

        return trade_costs, trade_qtys, uniform_price, cleared_volume, liquidity_pool_info

    # ------------------------------------------------------------------
    # Reward function (P3 + P4 + P8 improvements)
    # ------------------------------------------------------------------

    def _compute_rewards(self, payments, trade_costs, penalties,
                         invest_costs, emissions, clearing_price,
                         mac_costs=None, collateral_costs=None,
                         precompliance_holdings=None,
                         old_carry_forward=None, active_mask=None):
        """
        Reward (HAPPO-compliant, v8.3):
            R_i = w_cost * (-cost_norm_ex_penalty) + w_green * (esg_scale_i * esg_raw)
                  + green_bonus - penalty_norm - opportunity_cost

        v8.3 changes:
          - OPEX delta: only the change from baseline OPEX enters the cost signal.
          - Per-agent normalization: /annual_budget instead of /1000 for cost, penalty,
            opportunity cost, terminal bank value, and terminal debt penalty.
          - Per-agent ESG scale: compensates for the divisor change to preserve the
            50/50 ESG-to-cost balance for agents with w_green=0.5.

        Core signals:
          cost_norm_ex_penalty: total_cost_ex_penalty / annual_budget
          penalty_norm:         penalty_cost / annual_budget
          green_bonus:          diminishing-returns bonus for green investment progress
          esg_raw:              saved-carbon-years formula before weighting

        Penalty is separated from cost and applied at full strength regardless of w_cost.
        Terminal bonuses: log-scaled bank value / annual_budget + ESG terminal queue.
        """
        rewards = np.zeros(self.n_total)
        base_rewards = np.zeros(self.n_total)
        shaping_rewards = np.zeros(self.n_total)
        terminal_bank_values = np.zeros(self.n_total)
        terminal_queue_values = np.zeros(self.n_total)
        reward_cfg = self.config.get("reward", {})
        esg_cfg = self.config.get("esg", {})
        esg_enabled = esg_cfg.get("enabled", False)
        base_esg_scale = float(esg_cfg.get("scale", 2.0))

        beta_shaping = reward_cfg.get("shaping_beta", 10.0)

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
            # v8.3: OPEX delta — only the change from baseline enters the cost signal.
            # Positive delta = costs rose, negative delta = OPEX savings from greening.
            opex_delta = company.compute_operational_cost(self.current_year) - company.baseline_opex
            mac_cost_i = float(mac_costs[i])
            # NOTE: hold_fraction=0.02 (~7 days). Real EU ETS settles T+2 (~0.0055)
            # but one annual step represents ~52 real auctions; 0.02 is the balance.
            collateral_cost_i = float(collateral_costs[i])
            loan_interest_cost = company.compute_green_loan_cost()

            # Record spending: operational costs only (no non-compliance penalty).
            # Non-compliance penalty is a regulatory fine, not operational spending —
            # including it in budget tracking caused a death-spiral: penalty → budget
            # overshoot → budget_penalty explosion → carry-forward amplification.
            # Penalty already penalises the agent directly via penalty_norm in the reward.
            company.record_spending(auction_cost + secondary_cost
                                    + investment_cost + mac_cost_i
                                    + collateral_cost_i
                                    + loan_interest_cost)
            company.record_capex_spending(investment_cost)
            budget_penalty = company.compute_budget_penalty()
            capex_penalty = company.compute_capex_penalty()

            # v8.4: Per-agent financial-scale normalization using EMA budget
            budget_divisor = max(company.annual_budget, 1.0)

            # Separate penalty from other costs
            # Penalty applies at full strength to ALL agents regardless of w_cost
            total_cost_ex_penalty = (auction_cost + secondary_cost + investment_cost
                                     + opex_delta + budget_penalty + capex_penalty
                                     + mac_cost_i + collateral_cost_i + loan_interest_cost)

            cost_norm_ex_penalty = total_cost_ex_penalty / budget_divisor
            penalty_norm = penalty_cost / budget_divisor

            # Baseline-relative normalization: subtract expected cost at market price
            baseline_cost = company.compute_estimate_need() * clearing_price / budget_divisor
            cost_norm_ex_penalty -= baseline_cost

            # Green investment bonus with diminishing returns
            # Scaled by (0.2 + w_green) so financial agents still get some signal
            green_delta = max(0.0, company.green_frac - company.prev_green_frac)
            fossil_scale = max(company.fossil_frac, 0.05)
            green_bonus = beta_shaping * green_delta * fossil_scale * self.shaping_weight * (0.2 + company.w_green)

            # v8.4: ESG scale is just base_esg_scale (no per-agent budget compensation)
            esg_scale_i = base_esg_scale

            # ESG signal: saved-carbon-years formula
            esg_signal = 0.0
            if esg_enabled and company.initial_ef > 1e-6:
                ef_ratio = (company.initial_ef - company.weighted_emission_factor) / company.initial_ef
                time_ratio = remaining_years / self.n_years
                esg_raw = ef_ratio * time_ratio
                esg_signal = esg_scale_i * esg_raw

            # F1: Efficiency bonus as a shaping reward (decays with shaping_weight).
            efficiency_bonus = 0.0
            if company.initial_ef > 0.01:
                ef_improvement = max(0.0, company.initial_ef - company.weighted_emission_factor)
                ef_improvement_ratio = ef_improvement / company.initial_ef
                time_weight = remaining_years / self.n_years
                price_weight = clearing_price / 100.0
                efficiency_bonus = 1.5 * ef_improvement_ratio * time_weight * price_weight * self.shaping_weight

            base_reward = float(
                company.w_cost * (-cost_norm_ex_penalty)
                + company.w_green * esg_signal
                - penalty_norm  # penalty at full strength for all agents
            )
            shaping_reward = float(green_bonus + efficiency_bonus)

            base_rewards[i] = base_reward
            shaping_rewards[i] = shaping_reward
            rewards[i] = base_reward + shaping_reward

            self._last_reward_channels[i] = {
                "cost_norm": float(cost_norm_ex_penalty),
                "penalty_norm": float(penalty_norm),
                "green_bonus": float(green_bonus),
                "esg_signal": float(esg_signal),
                "efficiency_bonus": float(efficiency_bonus),
                "budget_penalty": float(budget_penalty / budget_divisor),
                "capex_penalty": float(capex_penalty / budget_divisor),
                "loan_interest": float(loan_interest_cost / budget_divisor),
                "baseline_cost": float(baseline_cost),
                "base_reward": float(base_reward),
                "shaping_reward": float(shaping_reward),
            }

        # Terminal value bonuses (final year only)
        is_final_year = self.current_year >= self.n_years - 1
        terminal_bank = reward_cfg.get("terminal_bank_value", False)
        terminal_queue = reward_cfg.get("terminal_queue_value", False)

        if is_final_year and (terminal_bank or terminal_queue):
            gamma_discount = self.config["ppo"].get("gamma", 0.99)
            terminal_payoff_years = reward_cfg.get("terminal_payoff_years", 5)

            # Shared terminal price: max(clearing, last_secondary, 80% of inflation-adjusted penalty)
            pen_cfg = self.config["penalty"]
            eff_penalty = pen_cfg["rate"] * self._inflation_factor(self.current_year)
            terminal_price = max(clearing_price, self.last_secondary_price, eff_penalty * 0.8)

            for i, company in enumerate(self.companies):
                if active_mask is not None and not bool(active_mask[i]):
                    continue
                budget_divisor = max(company.annual_budget, 1.0)
                # Terminal bank value with 2× annual_need cap:
                if terminal_bank:
                    annual_need = max(company.compute_estimate_need(), 0.1)
                    capped_holdings = min(self.holdings[i], 2.0 * annual_need)
                    ratio = capped_holdings / annual_need
                    bank_value = np.log1p(ratio) * annual_need * terminal_price / budget_divisor
                    rewards[i] += bank_value
                    terminal_bank_values[i] = bank_value

                # Terminal queue value: ESG from queue items with γ^years_late discount
                if terminal_queue:
                    queue_value = 0.0
                    for item in company._construction_queue:
                        years_late = max(0, item["completion_year"] - self.current_year)
                        effective_remaining = max(0, terminal_payoff_years - years_late)
                        if effective_remaining <= 0 or company.initial_ef < 1e-6:
                            continue
                        delta_ef = company.weighted_emission_factor - company.emission_factors[item["tech_idx"]]
                        if delta_ef <= 0:
                            continue
                        annual_saving = (delta_ef * item["frac_delta"]
                                         * company.output_mwh / 1e6)
                        discount = gamma_discount ** years_late
                        normalizer = company.initial_ef * company.output_twh * self.n_years
                        queue_value += (annual_saving * effective_remaining * discount
                                        * terminal_price / max(normalizer, 1e-6))
                    queue_term = queue_value
                    rewards[i] += queue_term
                    base_rewards[i] += queue_term
                    terminal_queue_values[i] = queue_term

        # Terminal debt liquidation: applied in final year regardless of terminal_bank/queue settings
        if is_final_year:
            pen_cfg = self.config["penalty"]
            eff_penalty = pen_cfg["rate"] * self._inflation_factor(self.current_year)
            terminal_price = max(clearing_price, self.last_secondary_price, eff_penalty * 0.8)

            for i, company in enumerate(self.companies):
                if active_mask is not None and not bool(active_mask[i]):
                    continue
                budget_divisor = max(company.annual_budget, 1.0)
                # Aggressively penalize outstanding carry_forward debt
                if company._carry_forward > 0:
                    debt_penalty = (company._carry_forward * terminal_price * 1.5) / budget_divisor
                    rewards[i] -= debt_penalty
                    base_rewards[i] -= debt_penalty

        self._last_terminal_bank_values = terminal_bank_values
        self._last_terminal_queue_values = terminal_queue_values
        self._last_terminal_liquidation_values = terminal_bank_values + terminal_queue_values
        self._last_reward_base_values = base_rewards
        self._last_reward_shaping_values = shaping_rewards

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

        These scores are logged to CSV for analysis and can be used to
        evaluate policy quality without reward normalization artifacts.

        Parameters
        ----------
        agent_id : int, optional
            If provided, return scores for that specific agent only.
            If None, return list of dicts for all learning agents.

        Returns
        -------
        dict or list[dict]
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

            # S_financial: 1 - (total_non_penalty_cost / annual_budget)
            # Higher score = lower non-penalty spending relative to budget
            total_cost = company.budget_spent_this_year
            s_financial = max(0.0, 1.0 - total_cost / max(budget_ref, 1.0))

            # S_green: emission factor progress vs initial
            if company.initial_ef > 1e-6:
                ef_progress = max(0.0, company.initial_ef - company.weighted_emission_factor)
                s_green = ef_progress / company.initial_ef
            else:
                s_green = 1.0  # already at zero emissions

            # S_penalty: 1 - (penalty_incurred / (annual_need * eff_penalty))
            annual_need = max(company.compute_estimate_need(), 1e-6)
            max_penalty = annual_need * eff_penalty
            # penalty_cost_this_year is not stored per-step; use proxy from last reward
            base_vals = self._last_reward_base_values
            shaping_vals = self._last_reward_shaping_values
            # Back out penalty from reward: penalty_norm = penalty / 1000
            # Approximate: we don't have per-agent penalty stored here, so use 0 as safe default
            # (train.py logs penalties separately via year_log)
            s_penalty = 1.0  # conservative default; overridden by logged year_log data

            # S_composite: weighted blend
            s_composite = (company.w_cost * s_financial
                           + company.w_green * s_green
                           + 0.3 * s_penalty)  # penalty always weighted (compliance baseline)

            score = {
                "agent_id": i,
                "S_financial": round(float(s_financial), 4),
                "S_green": round(float(s_green), 4),
                "S_penalty": round(float(s_penalty), 4),
                "S_composite": round(float(s_composite), 4),
            }
            results.append(score)

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
        Base: 26D. With opponent modeling: 26 + 5*(N_total-1) dims.
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

        return np.stack(obs_list)
