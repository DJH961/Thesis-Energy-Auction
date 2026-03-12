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


class ETSEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__()

        self.config = config
        self.n_agents = config["companies"]["n_agents"]
        self.n_years = config["simulation"]["n_years"]

        self._seed = seed
        self.rng = np.random.default_rng(seed)

        self.cap_schedule = CapSchedule(config)

        initial_mixes = config["companies"]["initial_mix"]
        self.companies: List[Company] = [
            Company(
                agent_id=i, config=config,
                initial_mix=initial_mixes[i], rng=self.rng,
            )
            for i in range(self.n_agents)
        ]

        # Episode state
        self.current_year = 0
        self.current_episode = 0          # updated by training loop via set_episode()
        self.last_clearing_price = config["price"]["initial_expected"]
        self.expected_price = config["price"]["initial_expected"]
        self._price_history: List[float] = []
        self.last_secondary_price = config["price"]["initial_expected"]
        self.last_secondary_volume = 0.0  # P8: track volume for phase1 obs
        self._last_gaps = np.zeros(self.n_agents)
        self.holdings = np.zeros(self.n_agents)
        self.episode_done = False

        # P4: shaping weight — decays from 1.0 to 0.0 over training (set by train.py)
        self.shaping_weight = 1.0

        # Secondary market profit tracking (EMA per agent)
        self._secondary_profit_ema = np.zeros(self.n_agents)
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
        self._fossil_frac_history: List[List[float]] = [[] for _ in range(self.n_agents)]

        # P5: Stochastic emission shocks — computed per year in step_auction()
        # Stores the shocked realized emissions and the shock values for obs/logging
        self._current_emissions = np.zeros(self.n_agents)   # shocked
        self._current_emission_shocks = np.zeros(self.n_agents)  # ε_it values

        # P6: CF noise per agent per tech — computed per year in step_auction()
        self._current_cf_noise = np.zeros((self.n_agents, 5))
        self._p6_cancellations = np.zeros(self.n_agents, dtype=int)

        # P8: per-agent holding costs from last _compute_rewards call (M€)
        self._last_holding_costs = np.zeros(self.n_agents)

        # Unsold allowance rollover: volume offered at auction but not allocated
        # carries forward to the next year's auction supply.
        self._unsold_rollover = 0.0

        # Opponent modeling (Option C)
        opp_enabled = config.get("opponent_modeling", {}).get("enabled", False)
        self._opponent_modeling = opp_enabled
        self._last_episode_bids = np.zeros(self.n_agents)
        self._last_episode_greens = np.array([
            c.green_frac for c in self.companies
        ])
        self._current_episode_bid_sum = np.zeros(self.n_agents)
        self._current_episode_bid_count = np.zeros(self.n_agents, dtype=int)

        # MAC fuel-switching tracking
        self._mac_reductions = np.zeros(self.n_agents)
        self._mac_costs = np.zeros(self.n_agents)

        # Price normalization constant
        self._price_norm = config["auction"]["price_max"]

        # Sanity check: agents must not be able to produce bids below the reserve
        _price_min = config["auction"]["price_min"]
        _reserve = config["ets"].get("reserve_price", 0.0)
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
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        # Opponent modeling: finalise last episode's data before resetting companies
        if self._opponent_modeling and self._current_episode_bid_count.sum() > 0:
            mask = self._current_episode_bid_count > 0
            self._last_episode_bids = np.where(
                mask,
                self._current_episode_bid_sum / np.maximum(self._current_episode_bid_count, 1),
                self._last_episode_bids,
            )
            self._last_episode_greens = np.array([c.green_frac for c in self.companies])
        self._current_episode_bid_sum = np.zeros(self.n_agents)
        self._current_episode_bid_count = np.zeros(self.n_agents, dtype=int)

        self.current_year = 0
        self.episode_done = False
        self.last_clearing_price = self.config["price"]["initial_expected"]
        self.expected_price = self.config["price"]["initial_expected"]
        self.last_secondary_price = self.config["price"]["initial_expected"]
        self._current_reference_price = self.config["price"]["initial_expected"]
        self.last_secondary_volume = 0.0
        self._price_history = []
        self._last_gaps = np.zeros(self.n_agents)
        self.holdings = np.zeros(self.n_agents)
        self.episode_log = []
        self._fossil_frac_history = [[] for _ in range(self.n_agents)]
        self._current_emissions = np.zeros(self.n_agents)
        self._current_emission_shocks = np.zeros(self.n_agents)
        self._current_cf_noise = np.zeros((self.n_agents, 5))
        self._p6_cancellations = np.zeros(self.n_agents, dtype=int)

        self.cap_schedule.reset()
        self._unsold_rollover = 0.0

        # Episode-level warning counters — reset each episode
        self._warnings = {
            "under_alloc": 0, "price_floor": 0, "price_ceiling": 0,
            "auction_failed": 0, "cover_below_one": 0, "zero_invest": 0,
            "chronic_short": 0, "bid_cluster": 0, "bank_hoard": 0,
            "sec_one_sided": 0, "sec_zero_vol": 0,
        }
        # Per-agent consecutive-shortfall counter for chronic_short detection
        self._consecutive_shortfall = np.zeros(self.n_agents, dtype=int)

        initial_mixes = self.config["companies"]["initial_mix"]
        for i, company in enumerate(self.companies):
            company.reset(initial_mix=initial_mixes[i])
            company.rng = self.rng

        # P7: Warm-start — seed construction queue, holdings, price history
        ws_cfg = self.config.get("warm_start", {})
        if ws_cfg.get("enabled", False):
            self._apply_warm_start(ws_cfg)

        obs_phase1 = self._get_obs_phase1()
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
            price_floor = self.config["price"].get("price_floor", 50.0)
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

    def step_auction(self, auction_actions: np.ndarray):
        """
        Phase 1: Execute auction and green investments.

        Parameters
        ----------
        auction_actions : np.ndarray, shape (N_agents, 6)
            [bid_price, quantity, invest_frac, tech_logit0, tech_logit1, tech_logit2]

        Returns
        -------
        obs_phase2 : np.ndarray, shape (N_agents, obs_dim_phase2)
        year_info : dict
        """
        assert not self.episode_done, "Episode done. Call reset()."

        year = self.current_year
        log = {"year": year}

        # Capture holdings at start of year for diagnostics
        bank_start = self.holdings.copy()
        log["bank_start"] = bank_start.tolist()

        # 1. P6: Cancellation check — before matured investments
        cancellations = np.zeros(self.n_agents, dtype=int)
        cancel_recoveries = np.zeros(self.n_agents)
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
        base_auction_volume = self.cap_schedule.get_auction_volume(year, tnac)
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
            idio = self.rng.normal(0, 1, self.n_agents)  # idiosyncratic shocks
            epsilons = rho * eta_common + np.sqrt(max(0.0, 1.0 - rho ** 2)) * idio
            epsilons *= sigma
        else:
            epsilons = np.zeros(self.n_agents)
        self._current_emission_shocks = epsilons

        # 5. P6: Generate capacity factor noise per tech per agent
        cf_sigma = np.array(jitter_cfg.get("cf_sigma", [0.0, 0.0, 0.08, 0.08, 0.05]))
        cf_noise = np.zeros((self.n_agents, 5))
        if jitter_cfg.get("enabled", False):
            for i in range(self.n_agents):
                for t in range(5):
                    if cf_sigma[t] > 0:
                        cf_noise[i, t] = float(self.rng.normal(0, cf_sigma[t]))
        self._current_cf_noise = cf_noise

        # 6. Compute realized emissions (CF noise → P6, demand shock → P5)
        realized_emissions = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            e_cf = company.compute_emissions_with_cf_noise(cf_noise[i])
            e_shocked = e_cf * (1.0 + epsilons[i])
            realized_emissions[i] = max(0.0, e_shocked)
        self._current_emissions = realized_emissions

        # Store aggregate CF shock per agent for logging (mean across green techs)
        cf_shock_agg = np.array([
            float(np.mean(cf_noise[i, [2, 3, 4]]))
            for i in range(self.n_agents)
        ])

        # 7. Auction
        bid_actions = auction_actions[:, :2].copy()

        markup_low = self.config["auction"].get("price_markup_low", 0.90)
        markup_high = self.config["auction"].get("price_markup_high", 1.25)
        price_min = self.config["auction"]["price_min"]
        price_max = self.config["auction"]["price_max"]

        reference_price = max(
            price_min,
            0.5 * float(self.last_clearing_price) + 0.5 * float(self.expected_price)
        )

        self._current_reference_price = reference_price

        bid_markup = np.clip(bid_actions[:, 0], markup_low, markup_high)
        bid_actions[:, 0] = np.clip(
            bid_markup * reference_price,
            price_min,
            price_max,
        )
        # Quantity reparameterization: action[1] is a coverage MULTIPLIER on estimated need.
        # actual_qty = multiplier × (compute_estimate_need + carry_forward)
        # This keeps the strategic decision centred on compliance coverage ratio rather
        # than an absolute volume, avoiding the zero-quantity collapse.
        qty_mult_low = self.config["auction"].get("qty_mult_low", 0.3)
        qty_mult_high = self.config["auction"].get("qty_mult_high", 1.3)
        lot_size = self.config["auction"].get("lot_size", 0.0)
        for i, company in enumerate(self.companies):
            multiplier = float(np.clip(auction_actions[i, 1], qty_mult_low, qty_mult_high))
            base_need = max(company.compute_estimate_need() + company._carry_forward, 0.1)
            bid_actions[i, 1] = multiplier * base_need
            # EU lot-size discretization: round to nearest multiple of lot_size
            if lot_size > 0:
                bid_actions[i, 1] = max(lot_size, round(bid_actions[i, 1] / lot_size) * lot_size)

        self._phase1_bid_prices = bid_actions[:, 0].copy()
        self._phase1_bid_quantities = bid_actions[:, 1].copy()  # Mt after multiplier expansion
        if self._opponent_modeling:
            self._current_episode_bid_sum += self._phase1_bid_prices
            self._current_episode_bid_count += 1

        bids = build_bids(bid_actions)
        clearing_price, allocations, payments, auction_stats = market_clearing_ets(
            bids=bids,
            q_cap=auction_volume,
            reserve_price=self.config["ets"].get("reserve_price", 0.0),
            max_agent_share=self.config["auction"].get("max_agent_share", 1.0),
            rng=self.rng,
            cancel_under_subscribed=self.config["auction"].get(
                "cancel_under_subscribed", False),
        )
        # Unsold: kept as logged metric only — fed into MSR instead of next year's volume
        unsold = max(0.0, auction_volume - float(allocations.sum()))
        self._unsold_rollover = unsold  # metric only
        log["unsold_rollover_out"] = round(unsold, 4)
        if self.config["ets"].get("unsold_to_msr", True):
            self.cap_schedule.absorb_unsold(unsold)
        self.last_clearing_price = clearing_price
        self._phase1_clearing_price = clearing_price
        log["clearing_price"] = clearing_price
        log["auction_stats"] = auction_stats

        # 8. MAC fuel-switching (based on auction clearing price)
        mac_reductions = np.zeros(self.n_agents)
        mac_costs = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            reduction, cost = company.apply_mac_switching(clearing_price)
            mac_reductions[i] = reduction
            mac_costs[i] = cost
        self._current_emissions = np.maximum(self._current_emissions - mac_reductions, 0.0)
        self._mac_reductions = mac_reductions
        self._mac_costs = mac_costs

        # 9. Green investments
        invest_costs = np.zeros(self.n_agents)
        invest_fracs = np.zeros(self.n_agents)  # raw action values for logging
        for i, company in enumerate(self.companies):
            invest_frac = float(auction_actions[i, 2])
            invest_fracs[i] = invest_frac
            tech_logits = auction_actions[i, 3:6]
            tech_choice = int(np.argmax(tech_logits))
            invest_costs[i] = company.plan_investment(tech_choice, invest_frac, year)
            # Apply any cancellation recovery as a credit to invest_costs
            invest_costs[i] -= cancel_recoveries[i]

        # Store for phase 2
        self._phase1_allocations = allocations
        self._phase1_payments = payments
        self._phase1_invest_costs = invest_costs
        self._phase1_mac_costs = mac_costs
        self._phase1_log = log

        # 10. Build phase 2 observations
        obs_phase1 = self._get_obs_phase1()

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

        # ── Auction-phase warning counters ────────────────────────────────────
        _reserve = self.config["ets"].get("reserve_price", 0.0)
        _price_max = self.config["auction"]["price_max"]
        if clearing_price <= _reserve + 1.0:
            self._warnings["price_floor"] += 1
        if clearing_price >= 0.9 * _price_max:
            self._warnings["price_ceiling"] += 1
        if auction_stats.get("auction_failed", False):
            self._warnings["auction_failed"] += 1
        if auction_stats.get("total_demand", 0.0) < auction_volume:
            self._warnings["cover_below_one"] += 1
        if float(allocations.sum()) < 0.5 * auction_volume:
            self._warnings["under_alloc"] += 1
        if np.std(self._phase1_bid_prices) < 5.0:
            self._warnings["bid_cluster"] += 1
        if np.all(invest_fracs < 0.001):
            self._warnings["zero_invest"] += 1
        if tnac > 2.0 * float(self._current_emissions.sum()):
            self._warnings["bank_hoard"] += 1

        return obs_phase2, log

    # ------------------------------------------------------------------
    # Phase 2: Secondary Market + Compliance + Rewards
    # ------------------------------------------------------------------

    def step_secondary(self, secondary_actions: np.ndarray):
        """
        Phase 2: Execute secondary market, compliance, and rewards.

        Parameters
        ----------
        secondary_actions : np.ndarray, shape (N_agents, 2)
            [price_multiplier, quantity] per agent.

        Returns
        -------
        obs_next, rewards, terminated, truncated, info
        """
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
        sell_coverage_floor = trading_cfg.get("sell_coverage_floor", 1.05)

        # Use realized emissions and existing carry-forward to gate selling.
        realized_emissions = self._current_emissions
        old_carry_forward = np.array([c._carry_forward for c in self.companies], dtype=float)

        secondary_prices = clearing_price * np.clip(
            secondary_actions[:, 0], sec_mult_low, sec_mult_high)

        raw_secondary_qtys = np.clip(
            secondary_actions[:, 1],
            -self.config["auction"]["quantity_max"],
            self.config["auction"]["quantity_max"],
        )

        # Pre-trade resources and obligations
        pretrade_holdings = self.holdings + allocations
        obligations = np.maximum(realized_emissions + old_carry_forward, 1e-6)

        # Max safe sell leaves at least sell_coverage_floor × obligation after trade
        max_safe_sell = np.maximum(
            0.0,
            pretrade_holdings - sell_coverage_floor * obligations
        )

        secondary_qtys = raw_secondary_qtys.copy()
        sell_mask = secondary_qtys < 0.0
        requested_sell = np.zeros_like(secondary_qtys)
        requested_sell[sell_mask] = np.abs(secondary_qtys[sell_mask])

        allowed_sell = np.zeros_like(secondary_qtys)
        allowed_sell[sell_mask] = np.minimum(
            requested_sell[sell_mask],
            max_safe_sell[sell_mask]
        )

        secondary_qtys[sell_mask] = -allowed_sell[sell_mask]

        blocked_sell_qty = np.zeros_like(secondary_qtys)
        blocked_sell_qty[sell_mask] = requested_sell[sell_mask] - allowed_sell[sell_mask]

        trade_costs, trade_qtys, secondary_clearing, secondary_volume = \
            self._settle_double_auction(
                allocations=allocations.copy(),
                secondary_prices=secondary_prices,
                secondary_qtys=secondary_qtys,
                clearing_price=clearing_price,
            )

        self.last_secondary_price = secondary_clearing
        self.last_secondary_volume = secondary_volume  # P8: track for obs

        # Update secondary profit EMA
        for i in range(self.n_agents):
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
        penalties = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            penalties[i] = company.settle_compliance_realized(
                allowances_held=holdings[i],
                realized_emissions=realized_emissions[i],
            )

        # Banking: surplus after surrendering for emissions + old carry-forward
        for i in range(self.n_agents):
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
            precompliance_holdings=holdings,
            old_carry_forward=old_carry_forward,
            blocked_sell_qty=blocked_sell_qty,
        )

        # Compute per-agent shortfall for diagnostics.
        # Total obligation = realized emissions + carry-forward from prior years.
        # Uses local `holdings` (pre-compliance: prev_bank + alloc + secondary).
        shortfalls = np.array([
            max(0.0, realized_emissions[i] + old_carry_forward[i] - holdings[i])
            for i in range(self.n_agents)
        ])

        # ── Secondary-phase warning counters ─────────────────────────────────
        if secondary_volume < 0.01:
            self._warnings["sec_zero_vol"] += 1
        sec_qty_raw = secondary_actions[:, 1]
        if np.all(sec_qty_raw > 0) or np.all(sec_qty_raw < 0):
            self._warnings["sec_one_sided"] += 1
        for _i in range(self.n_agents):
            if shortfalls[_i] > 1e-6:
                self._consecutive_shortfall[_i] += 1
                if self._consecutive_shortfall[_i] >= 3:
                    self._warnings["chronic_short"] += 1
                    self._consecutive_shortfall[_i] = 0
            else:
                self._consecutive_shortfall[_i] = 0

        # 8. Log
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
            "max_safe_sell": max_safe_sell.tolist(),
            "blocked_sell_qty": blocked_sell_qty.tolist(),
            "penalties": penalties.tolist(),
            "invest_costs": invest_costs.tolist(),
            "rewards": rewards.tolist(),
            "green_fracs": [c.green_frac for c in self.companies],
            "tech_mixes": [c.mix.tolist() for c in self.companies],
            "holdings": self.holdings.tolist(),
            "shortfalls": shortfalls.tolist(),
            "bid_prices": self._phase1_bid_prices.tolist() if self._phase1_bid_prices is not None else [],
            "delta_greens": [c.green_frac - c.prev_green_frac for c in self.companies],
            "queue_sizes": [len(c._construction_queue) for c in self.companies],
            "holding_costs": self._last_holding_costs.tolist(),  # P8
            "mac_reductions": self._mac_reductions.tolist(),
            "mac_costs": self._mac_costs.tolist(),
            "sec_price_mults": secondary_actions[:, 0].tolist(),  # Phase 2 action[0] (raw)
            "sec_qty_actions": secondary_actions[:, 1].tolist(),  # Phase 2 action[1] (raw)
        })
        self.episode_log.append(log)

        # 9. Advance year + AR(1) price
        self._price_history.append(clearing_price)
        rho = self.config["price"].get("ar1_persistence", 0.85)
        price_floor = self.config["price"].get("price_floor", 50.0)
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

        obs_next = self._get_obs_phase1()
        return obs_next, rewards, terminated, False, {"year_log": log}

    # ------------------------------------------------------------------
    # Legacy step (calls both phases — for testing)
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray):
        """Single-call step for backward compat. actions shape (N, 8)."""
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
        trade_costs = np.zeros(self.n_agents)
        trade_qtys = np.zeros(self.n_agents)

        if not cfg["enabled"]:
            return trade_costs, trade_qtys, clearing_price, 0.0

        tx_cost = cfg["transaction_cost"]
        # P8: spread tolerance as fraction of clearing price
        spread_tol = cfg.get("spread_tolerance", 0.0) * max(clearing_price, 1.0)

        buyers = []
        sellers = []

        for i in range(self.n_agents):
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

        if not buyers or not sellers:
            return trade_costs, trade_qtys, clearing_price, 0.0

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

        if not executed_trades:
            return trade_costs, trade_qtys, clearing_price, 0.0

        total_value = 0.0
        total_qty = 0.0

        for buyer_id, seller_id, trade_price, trade_qty in executed_trades:
            cost_buyer = trade_qty * (trade_price + tx_cost)
            revenue_seller = trade_qty * (trade_price - tx_cost)

            trade_costs[buyer_id] += cost_buyer
            trade_costs[seller_id] -= revenue_seller

            trade_qtys[buyer_id] += trade_qty
            trade_qtys[seller_id] -= trade_qty

            total_value += trade_price * trade_qty
            total_qty += trade_qty

        sec_clearing = total_value / total_qty if total_qty > 0 else clearing_price
        return trade_costs, trade_qtys, sec_clearing, total_qty

    # ------------------------------------------------------------------
    # Reward function (P3 + P4 + P8 improvements)
    # ------------------------------------------------------------------

    def _compute_rewards(self, payments, trade_costs, penalties,
                         invest_costs, emissions, clearing_price,
                         mac_costs=None, precompliance_holdings=None,
                            old_carry_forward=None, blocked_sell_qty=None):
        """
        R_i = Revenue_i - (alpha × TotalCost_i + beta × EmissionsIntensity_i + Penalty_i)
              + Shaping_i

        Revenue: electricity revenue (margin-based, from electricity config).
        Costs: auction + secondary + investment + operational + MAC + budget + holding.
        P3: emissions intensity capped at initial fossil floor.
        P4: green shaping bonuses (progress + queue + lock-in).
        P8: banking holding cost on excess allowances.

        Note: per-agent running normalisation applied in PPOAgent.normalize_reward()
        AFTER this function returns raw rewards.
        """
        rewards = np.zeros(self.n_agents)
        non_compliance_mult = self.config["penalty"].get("non_compliance_multiplier", 1.0)
        reward_cfg = self.config.get("reward", {})
        trading_cfg = self.config.get("trading", {})
        elec_cfg = self.config.get("electricity", {})

        green_floor_fossil = reward_cfg.get("green_floor_fossil", [0.0] * self.n_agents)

        beta_shaping = reward_cfg.get("shaping_beta", 10.0)
        gamma_queue = reward_cfg.get("shaping_gamma", 1.0)
        lockin_penalty = reward_cfg.get("shaping_lockin_penalty", 2.0)
        lockin_start = reward_cfg.get("shaping_lockin_start_episode", 200)
        trading_bonus_weight = reward_cfg.get("shaping_trading_weight", 0.5)
        bank_weight = reward_cfg.get("shaping_bank_weight", 0.1)
        coverage_weight = reward_cfg.get("shaping_coverage_weight", 2.0)
        low_bid_shortfall_weight = reward_cfg.get("shaping_low_bid_shortfall_weight", 1.5)
        near_floor_threshold = reward_cfg.get("low_bid_near_floor_threshold", 1.05)
        unsafe_sell_penalty_weight = self.config.get("trading", {}).get("unsafe_sell_penalty_weight", 2.0)
        # P8: banking holding cost parameters
        holding_cost_rate = trading_cfg.get("banking_holding_cost", 0.0)
        excess_bank_ratio = trading_cfg.get("excess_bank_ratio", 1.5)

        # Electricity revenue parameters
        elec_enabled = elec_cfg.get("enabled", False)
        base_elec_price = elec_cfg.get("base_price", 50.0)
        carbon_passthrough = elec_cfg.get("carbon_passthrough", 0.80)

        # System average emission factor (for electricity price)
        if elec_enabled:
            system_avg_ef = float(np.mean([c.weighted_emission_factor for c in self.companies]))
            elec_price = base_elec_price + carbon_passthrough * clearing_price * system_avg_ef

        if mac_costs is None:
            mac_costs = np.zeros(self.n_agents)

        for i, company in enumerate(self.companies):
            auction_cost = float(payments[i])
            raw_secondary = float(trade_costs[i])
            secondary_cost = raw_secondary
            penalty_cost = float(penalties[i])
            investment_cost = float(invest_costs[i])
            operational_cost = company.compute_operational_cost()
            mac_cost_i = float(mac_costs[i])

            # NOTE: removed old clamping `max(secondary_cost, -auction_cost)` which
            # capped trading revenue at auction spend, killing secondary market incentive.

            company.record_spending(auction_cost + max(0.0, secondary_cost)
                                    + investment_cost + mac_cost_i)
            budget_penalty = company.compute_budget_penalty()

            # P8: Banking holding cost on excess bank above threshold
            annual_need = float(emissions[i])
            excess_bank = max(0.0, self.holdings[i] - excess_bank_ratio * max(annual_need, 1e-6))
            holding_cost = excess_bank * holding_cost_rate
            self._last_holding_costs[i] = holding_cost

            total_cost = (auction_cost + secondary_cost + investment_cost
                         + operational_cost + budget_penalty + holding_cost
                         + mac_cost_i)

            # Electricity revenue (same market price for all, green generators profit more)
            revenue = 0.0
            if elec_enabled:
                revenue = company.output_mwh * elec_price / 1e6  # M€

            net_cost = total_cost - revenue
            cost_norm = net_cost / 1000.0

            fossil_floor_i = green_floor_fossil[i] if i < len(green_floor_fossil) else 0.0
            initial_ef_at_floor = fossil_floor_i * max(company.emission_factors[~company.is_green])
            penalisable_ef = max(0.0, company.weighted_emission_factor - initial_ef_at_floor)
            emissions_intensity = penalisable_ef / 0.82

            penalty_norm = penalty_cost * non_compliance_mult / 1000.0

            # P4: Green investment shaping rewards
            green_delta = max(0.0, company.green_frac - company.prev_green_frac)
            shaping = beta_shaping * green_delta * self.shaping_weight

            n_queue_active = sum(
                1 for item in company._construction_queue
                if item.get("frac_delta", 0) > 0
            )
            shaping += gamma_queue * n_queue_active * 0.1 * self.shaping_weight

            if self.current_episode >= lockin_start and self.shaping_weight > 0:
                hist = self._fossil_frac_history[i]
                if len(hist) >= 3:
                    unchanged = all(abs(f - hist[-1]) < 1e-4 for f in hist[-2:])
                    if unchanged:
                        shaping -= lockin_penalty * self.shaping_weight

            # Reward net revenue from secondary market (sellers who trade profitably)
            trading_profit = max(0.0, -raw_secondary)
            shaping += trading_bonus_weight * (trading_profit / 1000.0) * self.shaping_weight

            # Small positive signal for holding a healthy allowance bank.
            # holdings (Mt) * clearing_price (EUR/t) = M-EUR directly;
            # the old code had an extra /1e6 making this signal ~0.
            bank_value = self.holdings[i] * clearing_price   # M-EUR
            shaping += bank_weight * (bank_value / 1000.0) * self.shaping_weight

            ############ Auction coverage shaping: peaked function rewards ~100% coverage,
                        # Coverage shaping based on final pre-compliance coverage, not only primary allocation.
            obligation_i = max(
                float(emissions[i]) + (float(old_carry_forward[i]) if old_carry_forward is not None else 0.0),
                0.01,
            )
            available_i = float(precompliance_holdings[i]) if precompliance_holdings is not None else float(self.holdings[i])

            raw_coverage = available_i / obligation_i
            if raw_coverage <= 1.0:
                coverage_score = raw_coverage
            elif raw_coverage <= 1.1:
                coverage_score = 1.0 - (raw_coverage - 1.0) * 10.0
            else:
                coverage_score = -2.0 * (raw_coverage - 1.1)

            shaping += coverage_weight * coverage_score * self.shaping_weight

            # Extra penalty when an agent remains short while still bidding near the auction floor.
            bid_floor = float(self.config["auction"]["price_min"])
            bid_i = float(self._phase1_bid_prices[i]) if self._phase1_bid_prices is not None else bid_floor
            shortfall_i = max(0.0, obligation_i - available_i)

            if shortfall_i > 1e-6 and bid_i <= near_floor_threshold * bid_floor:
                shaping -= low_bid_shortfall_weight * min(shortfall_i / obligation_i, 1.0) * self.shaping_weight

            raw = -cost_norm - emissions_intensity - penalty_norm + shaping

            rewards[i] = float(raw)

        return rewards

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _compute_price_ma3(self) -> float:
        """P1: 3-year moving average of clearing price."""
        if not self._price_history:
            return self.last_clearing_price
        window = self._price_history[-3:]
        return float(np.mean(window))

    def _get_obs_phase1(self) -> np.ndarray:
        """Phase 1 observations for all agents.
        Base: 20D (P8: +2 secondary dims). With opponent modeling: 20 + 2*(N-1) dims.
        """
        cap_t = self.cap_schedule.get_cap(self.current_year)
        price_ma3 = self._compute_price_ma3()

        obs_list = []
        for i, c in enumerate(self.companies):
            if self._opponent_modeling and self.n_agents > 1:
                opp_parts = []
                for j in range(self.n_agents):
                    if j != i:
                        opp_parts.append(float(self._last_episode_bids[j]) / self._price_norm)
                        opp_parts.append(float(self._last_episode_greens[j]))
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
                last_secondary_volume=self.last_secondary_volume,  # P8
            )
            obs_list.append(obs_i)

        return np.stack(obs_list)
