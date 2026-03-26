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
  [price_multiplier, quantity]

Roadmap improvements (P1-P4): price MA, entropy, reward normalisation, green shaping.
Opponent modeling: Phase 1 obs augmented with last-episode (bid/200, green_frac) for N-1 agents.
Agent cycling: handled in train.py.

Roadmap improvements (P5-P8):
  P5: Stochastic annual emission shocks (correlated across agents); shock in Phase 2 obs.
  P6: Construction delay jitter + cancellation risk + capacity factor noise.
  P7: Warm-start — queue seeding + bank seeding + price history seeding at reset.
  P8: Wider secondary spread tolerance; selling from bank; banking holding cost;
      secondary price/volume in Phase 1 observation.
"""

import numpy as np
import gymnasium as gym
from typing import List, Optional

from src.auction.market_clearing_ets import market_clearing_ets, build_bids
from src.environment.cap_schedule import CapSchedule
from src.environment.company import Company
from src.agents import heuristic_policy


class ETSEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__()

        self.config = config
        self.n_agents = config["companies"]["n_agents"]           # PPO learning agents (external interface)
        self.n_bots = config["companies"].get("n_bot_agents", 0)  # heuristic bot agents
        self.n_total = self.n_agents + self.n_bots                # total market participants
        self.n_years = config["simulation"]["n_years"]

        self._seed = seed
        self.rng = np.random.default_rng(seed)

        self.cap_schedule = CapSchedule(config)

        # Extend config arrays to include bot entries (so Company can index by agent_id)
        if self.n_bots > 0:
            bot_mixes = config["companies"].get("bot_initial_mix", [])
            bot_rw = config["companies"].get("bot_reward_weights",
                                              [[0.5, 0.5]] * self.n_bots)
            config["companies"]["initial_mix"] = (
                config["companies"]["initial_mix"] + bot_mixes)
            config["companies"]["reward_weights"] = (
                config["companies"]["reward_weights"] + bot_rw)
            bot_budgets = config["budget"].get("bot_annual_budgets",
                                                [1200.0] * self.n_bots)
            config["budget"]["annual_budgets"] = (
                config["budget"]["annual_budgets"] + bot_budgets)

        initial_mixes = config["companies"]["initial_mix"]
        self.companies: List[Company] = [
            Company(
                agent_id=i, config=config,
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

        # Dynamic reserve tracking
        self._last_effective_reserve = config["ets"].get("reserve_initial",
                                                          config["ets"].get("reserve_price", 0.0))
        self._reserve_anchor = config["ets"].get("reserve_anchor", "secondary")
        if self._reserve_anchor not in {"secondary", "auction"}:
            self._reserve_anchor = "secondary"
        self._consecutive_years_without_valid_auction_clear = 0

        # Secondary liquidity pool EMA anchor state (only used when pool enabled)
        self._liquidity_ref_ema = float(config["price"]["initial_expected"])

        # Episode-level inflation path (shared by all participants)
        self._inflation_rates: List[float] = []
        self._inflation_factors: List[float] = [1.0]
        self._build_episode_inflation_path()

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

        # Price normalization constant
        self._price_norm = config["auction"]["price_max"]

        # Log environment config summary
        auction_cfg = self.config.get("auction", {})
        print(f"[ETSEnvironment] {self.n_agents} learning + {self.n_bots} bot = {self.n_total} total agents"
              f" | cancel_under_subscribed={auction_cfg.get('cancel_under_subscribed', False)}")

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

    # ------------------------------------------------------------------
    # Training loop interface
    # ------------------------------------------------------------------

    def set_episode(self, episode: int):
        """Called by training loop to communicate current episode for shaping decay."""
        self.current_episode = episode
        reward_cfg = self.config.get("reward", {})
        decay_ep = reward_cfg.get("shaping_decay_episode", 3000)
        floor = reward_cfg.get("shaping_weight_floor", 0.0)
        self.shaping_weight = max(floor, 1.0 - episode / decay_ep)

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
        # by 20% per year toward the absolute floor.
        if self._consecutive_years_without_valid_auction_clear >= 2:
            return max(abs_floor, abs_floor + (base_reserve - abs_floor) * 0.8)

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
        self._consecutive_years_without_valid_auction_clear = 0
        self._liquidity_ref_ema = float(self.config["price"]["initial_expected"])

        self.cap_schedule.reset()
        self._unsold_rollover = 0.0
        self._build_episode_inflation_path()

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
            self._apply_warm_start(ws_cfg)

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

    def _apply_warm_start(self, ws_cfg: dict):
        """
        P7: Warm-start seeding.

        (A) Seed construction queues: each agent gets a random number of in-flight
            renewable projects sampled from Poisson distributions per technology.
        (B) Seed initial bank: sample from Uniform(bank_min, bank_max) × annual_need.
        (C) Seed price history: sample 2 synthetic prices from N(60, 15) to initialise
            the MA3 price signal.
        """
        mu_onshore = ws_cfg.get("queue_mu_onshore", 1.5)
        mu_offshore = ws_cfg.get("queue_mu_offshore", 0.5)
        mu_solar = ws_cfg.get("queue_mu_solar", 2.0)
        bank_min = ws_cfg.get("bank_seed_min", 0.5)
        bank_max = ws_cfg.get("bank_seed_max", 2.0)
        n_burnin = ws_cfg.get("n_burnin_prices", 2)

        # Tech → (mu, completion_year_range_max, deploy_index_in_BUILDABLE)
        green_specs = [
            (2, mu_onshore),   # onshore_wind: tech_idx=2
            (3, mu_offshore),  # offshore_wind: tech_idx=3
            (4, mu_solar),     # solar:         tech_idx=4
        ]

        jitter_cfg = self.config.get("construction_jitter", {})
        poisson_lambdas = jitter_cfg.get("poisson_lambdas", [1.0, 1.0, 2.0, 3.0, 1.5])

        for i, company in enumerate(self.companies):
            # (A) Seed construction queue
            total_seeded_frac = 0.0
            max_seedable = company.fossil_frac * 0.5  # don't seed more than half fossil

            for tech_idx, mu in green_specs:
                lam = float(poisson_lambdas[tech_idx])
                n_projects = int(self.rng.poisson(mu))
                for _ in range(n_projects):
                    frac_delta = float(self.rng.uniform(0.005, 0.025))
                    if total_seeded_frac + frac_delta > max_seedable:
                        frac_delta = max(0, max_seedable - total_seeded_frac)
                    if frac_delta < 1e-4:
                        continue
                    # Completion year uniformly distributed within [0, λ+1]
                    completion_year = int(self.rng.integers(0, max(1, int(lam) + 2)))
                    company._construction_queue.append({
                        "tech_idx": tech_idx,
                        "frac_delta": frac_delta,
                        "completion_year": completion_year,
                        "success": True,
                        "capex_spent": company.compute_investment_cost(tech_idx, frac_delta),
                    })
                    total_seeded_frac += frac_delta

            # (B) Seed initial bank
            annual_need = company.compute_emissions()  # Mt
            seed_multiple = float(self.rng.uniform(bank_min, bank_max))
            self.holdings[i] = annual_need * seed_multiple

        # (C) Seed price history with synthetic burn-in prices
        for _ in range(n_burnin):
            synthetic_price = float(np.clip(
                self.rng.normal(60.0, 15.0),
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

    # ------------------------------------------------------------------
    # Phase 1: Auction + Green Investment
    # ------------------------------------------------------------------

    def _generate_bot_auction_actions(self) -> np.ndarray:
        """Generate Phase-1 actions for all bot agents using heuristic_policy."""
        if self.n_bots == 0:
            return np.zeros((0, 6), dtype=np.float32)
        price_ma3 = self._compute_price_ma3()
        reserve = self._compute_dynamic_reserve()
        infl_factor = self._inflation_factor(self.current_year)
        actions = np.zeros((self.n_bots, 6), dtype=np.float32)
        for b in range(self.n_bots):
            idx = self.n_agents + b  # bots indexed after learning agents
            actions[b] = heuristic_policy.auction_action(
                self.companies[idx], price_ma3, self.current_year,
            self.n_years, self.config, reserve_price=reserve,
            inflation_factor=infl_factor)
        return actions

    def _generate_bot_secondary_actions(self, clearing_price: float) -> np.ndarray:
        """Generate Phase-2 actions for all bot agents using heuristic_policy."""
        if self.n_bots == 0:
            return np.zeros((0, 2), dtype=np.float32)
        actions = np.zeros((self.n_bots, 2), dtype=np.float32)
        for b in range(self.n_bots):
            idx = self.n_agents + b  # bots indexed after learning agents
            actions[b] = heuristic_policy.secondary_action(
                self.companies[idx],
                bank=float(self.holdings[idx]),
                allocation=float(self._phase1_allocations[idx]),
                clearing_price=clearing_price,
                config=self.config,
                current_year=self.current_year,
                n_years=self.n_years)
        return actions

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

        # Combine learning agent actions with bot actions
        bot_auc = self._generate_bot_auction_actions()
        auction_actions = np.concatenate([auction_actions, bot_auc], axis=0)

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
            company.reset_budget()

        # 3. Compute TNAC and auction volume
        cap_t = self.cap_schedule.get_cap(year)
        tnac = float(self.holdings.sum())
        price_max = float(self.config["auction"]["price_max"])
        base_auction_volume = self.cap_schedule.get_auction_volume(
            year, tnac, self.last_clearing_price, price_max
        )
        auction_volume = base_auction_volume
        log["cap"] = cap_t
        log["tnac"] = tnac
        log["auction_volume"] = auction_volume
        log["unsold_rollover_in"] = round(self._unsold_rollover, 4)
        log["msr_reserve"] = self.cap_schedule.msr_reserve()

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
        qty_mult_high = self.config["auction"].get("qty_mult_high", 1.3)
        lot_size = self.config["auction"].get("lot_size", 0.0)
        bid_qty_multipliers = np.zeros(self.n_total)
        estimate_needs = np.zeros(self.n_total)
        bid_coverages = np.zeros(self.n_total)
        for i, company in enumerate(self.companies):
            multiplier = float(np.clip(auction_actions[i, 1], qty_mult_low, qty_mult_high))
            base_need = max(company.compute_estimate_need() + company._carry_forward, 0.1)
            bid_actions[i, 1] = multiplier * base_need
            bid_qty_multipliers[i] = multiplier
            estimate_needs[i] = base_need
            bid_coverages[i] = bid_actions[i, 1] / max(base_need, 1e-6)
            # EU lot-size discretization: round to nearest multiple of lot_size
            if lot_size > 0:
                bid_actions[i, 1] = max(lot_size, round(bid_actions[i, 1] / lot_size) * lot_size)

        self._phase1_bid_prices = bid_actions[:, 0].copy()
        self._phase1_bid_quantities = bid_actions[:, 1].copy()  # Mt after multiplier expansion

        # Compute effective reserve price (dynamic or static)
        effective_reserve = self._compute_dynamic_reserve()
        self._last_effective_reserve = effective_reserve

        bids = build_bids(bid_actions)
        clearing_price, allocations, payments, auction_stats = market_clearing_ets(
            bids=bids,
            q_cap=auction_volume,
            reserve_price=effective_reserve,
            max_agent_share=self.config["auction"].get("max_agent_share", 1.0),
            rng=self.rng,
            cancel_under_subscribed=self.config["auction"].get(
                "cancel_under_subscribed", False),
        )
        # Unsold allowances: either absorbed into MSR or rolled over to next year's auction
        unsold = max(0.0, auction_volume - float(allocations.sum()))
        self._unsold_rollover = unsold
        log["unsold_rollover_out"] = round(unsold, 4)
        if self.config["ets"].get("unsold_to_msr", True):
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
            reduction, cost = company.apply_mac_switching(clearing_price, current_year=year)
            mac_reductions[i] = reduction
            mac_costs[i] = cost
        self._current_emissions = np.maximum(self._current_emissions - mac_reductions, 0.0)
        self._mac_reductions = mac_reductions
        self._mac_costs = mac_costs

        # 9. Green investments
        invest_costs = np.zeros(self.n_total)
        invest_fracs = np.zeros(self.n_total)  # raw action values for logging
        invest_tech_choices = np.zeros(self.n_total, dtype=int)
        for i, company in enumerate(self.companies):
            invest_frac = float(auction_actions[i, 2])
            invest_fracs[i] = invest_frac
            tech_logits = auction_actions[i, 3:6]
            tech_choice = int(np.argmax(tech_logits))
            invest_tech_choices[i] = tech_choice
            invest_costs[i] = company.plan_investment(tech_choice, invest_frac, year)
            # Apply any cancellation recovery as a credit to invest_costs
            invest_costs[i] -= cancel_recoveries[i]

        # Store for phase 2
        self._phase1_allocations = allocations
        self._phase1_payments = payments
        self._phase1_invest_costs = invest_costs
        self._phase1_mac_costs = mac_costs
        self._phase1_log = log

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
        log["invest_fracs"] = invest_fracs.tolist()                   # raw action[2] per agent
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
    # Phase 2: Secondary Market + Compliance + Rewards
    # ------------------------------------------------------------------

    def step_secondary(self, secondary_actions: np.ndarray):
        """
        Phase 2: Execute secondary market, compliance, and rewards.

        Parameters
        ----------
        secondary_actions : np.ndarray, shape (n_learning, 2)
            Actions for learning agents only.
            [price_multiplier, quantity] per agent.
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

        # 5. Secondary market
        trading_cfg = self.config.get("trading", {})
        sec_mult_low = trading_cfg.get("sec_mult_low", 0.8)
        sec_mult_high = trading_cfg.get("sec_mult_high", 1.3)

        realized_emissions = self._current_emissions
        old_carry_forward = np.array([c._carry_forward for c in self.companies], dtype=float)

        secondary_prices = clearing_price * np.clip(
            secondary_actions[:, 0], sec_mult_low, sec_mult_high)

        raw_secondary_qtys = np.clip(
            secondary_actions[:, 1],
            -self.config["auction"]["quantity_max"],
            self.config["auction"]["quantity_max"],
        )

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

        # Use P5-shocked realized emissions for compliance
        #realized_emissions = self._current_emissions

        # 6. Compliance (against realized emissions + carry-forward obligations)
        # Capture old carry-forward before it gets updated
        #old_carry_forward = np.array([c._carry_forward for c in self.companies])
        penalties = np.zeros(self.n_total)
        for i, company in enumerate(self.companies):
            penalties[i] = company.settle_compliance_realized(
                allowances_held=holdings[i],
                realized_emissions=realized_emissions[i],
                current_year=self.current_year,
            )

        # Banking: surplus after surrendering for emissions + old carry-forward
        for i in range(self.n_total):
            total_obligation = realized_emissions[i] + old_carry_forward[i]
            self.holdings[i] = max(0.0, holdings[i] - total_obligation)
        self._last_gaps = self.holdings.copy()

        # P4: Update fossil fraction history (for lock-in penalty)
        for i, company in enumerate(self.companies):
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
            precompliance_holdings=pretrade_holdings,
            old_carry_forward=old_carry_forward,
        )

        # Compute per-agent shortfall for diagnostics.
        # Total obligation = realized emissions + carry-forward from prior years.
        # Uses local `holdings` (pre-compliance: prev_bank + alloc + secondary).
        shortfalls = np.array([
            max(0.0, realized_emissions[i] + old_carry_forward[i] - holdings[i])
            for i in range(self.n_total)
        ])

        # ── Secondary-phase warning counters ─────────────────────────────────
        if secondary_volume < 0.01:
            self._warnings["no_trade"] += 1
        sec_qty_raw = secondary_actions[:, 1]
        if np.all(sec_qty_raw > 0) or np.all(sec_qty_raw < 0):
            self._warnings["one_side_sec"] += 1
        for _i in range(self.n_total):
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
            "secondary_clearing": secondary_clearing,
            "secondary_volume": secondary_volume,
            "raw_secondary_qtys": raw_secondary_qtys.tolist(),
            "gated_secondary_qtys": secondary_qtys.tolist(),
            "penalties": penalties.tolist(),
            "invest_costs": invest_costs.tolist(),
            "rewards": rewards.tolist(),
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
            "sec_price_mults": secondary_actions[:, 0].tolist(),  # Phase 2 action[0] (raw)
            "sec_qty_actions": secondary_actions[:, 1].tolist(),  # Phase 2 action[1] (raw)
            "sec_action_sides": sec_action_sides.tolist(),         # -1=sell, 0=hold, 1=buy intent
            "liquidity_pool": liquidity_pool_info,
        })
        self.episode_log.append(log)

        # 9. Advance year + AR(1) price
        if self._reserve_anchor == "secondary":
            self._price_history.append(secondary_clearing)
        rho = self.config["price"].get("ar1_persistence", 0.85)
        price_floor = self.config["price"].get("ar1_floor",
                        self.config["price"].get("price_floor", 50.0))
        vol_std = self.config["price"].get("volatility_std", 0.15)

        if clearing_price > 0:
            shock = self.rng.normal(0, vol_std) * clearing_price
            self.expected_price = max(
                rho * clearing_price + (1.0 - rho) * price_floor + shock,
                price_floor,
            )

        self.current_year += 1
        terminated = self.current_year >= self.n_years
        self.episode_done = terminated

        obs_next = self._get_obs_phase1()   # shape (n_agents, obs_dim)
        return obs_next, rewards[:self.n_agents], terminated, False, {"year_log": log}

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
                # P8: allow selling from holdings (bank) as well as auction allocation
                max_sell = float(allocations[i]) + float(max(0.0, self.holdings[i]))
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
                         mac_costs=None, precompliance_holdings=None,
                         old_carry_forward=None):
        """
        Reward (HAPPO-compliant):
          R_i = -cost_norm - emissions_intensity - penalty_norm + green_bonus + queue_bonus

        Five core economic signals:
          cost_norm:           total costs (auction + secondary + invest + ops + MAC - revenue) / 1000
          emissions_intensity: penalisable emission factor / 0.82
          penalty_norm:        linear non-compliance penalty (penalty / 100)
          green_bonus:         diminishing-returns bonus for green investment progress
          queue_bonus:         reward for active construction queue items (decays with shaping_weight)

        Note: per-agent running normalisation applied in PPOAgent.normalize_reward()
        AFTER this function returns raw rewards.
        """
        rewards = np.zeros(self.n_total)
        terminal_bank_values = np.zeros(self.n_total)
        terminal_queue_values = np.zeros(self.n_total)
        non_compliance_mult = self.config["penalty"].get("non_compliance_multiplier", 1.0)
        reward_cfg = self.config.get("reward", {})
        trading_cfg = self.config.get("trading", {})
        elec_cfg = self.config.get("electricity", {})

        green_floor_fossil = reward_cfg.get("green_floor_fossil", [0.0] * self.n_total)
        beta_shaping = reward_cfg.get("shaping_beta", 10.0)
        gamma_shaping = reward_cfg.get("shaping_gamma", 1.0)
        price_anchor_delta = reward_cfg.get("price_anchor_delta", 0.5)

        # Electricity revenue parameters (base price inflation-indexed)
        elec_enabled = elec_cfg.get("enabled", False)
        base_elec_price = elec_cfg.get("base_price", 50.0)
        carbon_passthrough = elec_cfg.get("carbon_passthrough", 0.80)
        inflation_factor = self._inflation_factor(self.current_year)

        if elec_enabled:
            system_avg_ef = float(np.mean([c.weighted_emission_factor for c in self.companies]))
            elec_price = base_elec_price * inflation_factor + carbon_passthrough * clearing_price * system_avg_ef

        if mac_costs is None:
            mac_costs = np.zeros(self.n_total)

        for i, company in enumerate(self.companies):
            auction_cost = float(payments[i])
            secondary_cost = float(trade_costs[i])
            penalty_cost = float(penalties[i])
            investment_cost = float(invest_costs[i])
            operational_cost = company.compute_operational_cost(self.current_year)
            mac_cost_i = float(mac_costs[i])

            company.record_spending(auction_cost + max(0.0, secondary_cost)
                                    + investment_cost + mac_cost_i)
            budget_penalty = company.compute_budget_penalty()

            total_cost = (auction_cost + secondary_cost + investment_cost
                         + operational_cost + budget_penalty + mac_cost_i)

            # Electricity revenue
            revenue = 0.0
            if elec_enabled:
                revenue = company.output_mwh * elec_price / 1e6  # M€

            cost_norm = (total_cost - revenue) / 1000.0

            # Emissions intensity (capped at initial fossil floor)
            fossil_floor_i = green_floor_fossil[i] if i < len(green_floor_fossil) else 0.0
            initial_ef_at_floor = fossil_floor_i * max(company.emission_factors[~company.is_green])
            penalisable_ef = max(0.0, company.weighted_emission_factor - initial_ef_at_floor)
            emissions_intensity = penalisable_ef / 0.82

            # Linear non-compliance penalty — denominator 100 so €100M penalty = 1.0 signal
            penalty_norm = (penalty_cost / 100.0) * non_compliance_mult

            # Green investment bonus with diminishing returns
            green_delta = max(0.0, company.green_frac - company.prev_green_frac)
            fossil_scale = max(company.fossil_frac, 0.05)
            green_bonus = beta_shaping * green_delta * fossil_scale * self.shaping_weight

            # Queue bonus: reward for having active construction projects
            n_active_queue = len(company._construction_queue)
            queue_bonus = gamma_shaping * n_active_queue * 0.1 * self.shaping_weight

            # Price-anchor bonus: encourage bidding near expected price (decays with shaping_weight).
            # Uses a Gaussian-shaped bonus: max at expected_price, falls off with distance.
            # Normalised so the bonus ∈ [0, price_anchor_delta] when shaping_weight=1.
            price_anchor_bonus = 0.0
            if price_anchor_delta > 0.0 and self.shaping_weight > 0.0:
                bid_price_i = float(self._phase1_bid_prices[i])
                ref_price = max(self.expected_price, 10.0)  # AR(1) expected price
                price_dev = (bid_price_i - ref_price) / ref_price  # fractional deviation
                # Gaussian kernel: exp(-dev²/2σ²) with σ=0.5 (±50% gets ~60% of max bonus)
                price_anchor_bonus = (price_anchor_delta
                                      * np.exp(-0.5 * (price_dev / 0.5) ** 2)
                                      * self.shaping_weight)

            rewards[i] = float(-cost_norm - emissions_intensity - penalty_norm
                               + green_bonus + queue_bonus + price_anchor_bonus)

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
                # Terminal bank value: banked allowances × terminal_price
                if terminal_bank:
                    bank_value = self.holdings[i] * terminal_price / 100.0
                    rewards[i] += bank_value
                    terminal_bank_values[i] = bank_value

                # Terminal queue value: NPV of future carbon savings from in-construction projects
                if terminal_queue:
                    queue_value = 0.0
                    for item in company._construction_queue:
                        delta_ef = company.weighted_emission_factor - company.emission_factors[item["tech_idx"]]
                        annual_saving = (max(0.0, delta_ef)
                                         * item["frac_delta"]
                                         * company.output_mwh / 1e6
                                         * terminal_price)
                        discount = gamma_discount ** max(0, item["completion_year"] - self.current_year)
                        queue_value += annual_saving * terminal_payoff_years * discount
                    queue_term = queue_value / 1000.0
                    rewards[i] += queue_term
                    terminal_queue_values[i] = queue_term

        self._last_terminal_bank_values = terminal_bank_values
        self._last_terminal_queue_values = terminal_queue_values
        self._last_terminal_liquidation_values = terminal_bank_values + terminal_queue_values

        return rewards

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _compute_price_ma3(self) -> float:
        """P1: 3-year moving average of clearing price."""
        if not self._price_history:
            if self._reserve_anchor == "auction":
                return self.last_clearing_price
            return self.last_secondary_price
        window = self._price_history[-3:]
        return float(np.mean(window))

    def _get_obs_phase1(self) -> np.ndarray:
        """Phase 1 observations for learning agents only.
        Base: 23D. With opponent modeling: 23 + 5*(N_total-1) dims.
        Opponent modeling includes ALL market participants (learning + bots).
        """
        cap_t = self.cap_schedule.get_cap(self.current_year)
        price_ma3 = self._compute_price_ma3()

        # TNAC proxy: total banked allowances / cap (market-level scarcity signal)
        tnac = float(self.holdings.sum())
        tnac_proxy = tnac / max(cap_t, 1e-6)

        # Pre-compute 5D public info for ALL participants (learning + bots)
        if self._opponent_modeling and self.n_total > 1:
            public_infos = [c.get_public_info() for c in self.companies]

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
                        ])
                opponent_obs = np.array(opp_parts, dtype=np.float32)
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
            )
            obs_list.append(obs_i)

        return np.stack(obs_list)
