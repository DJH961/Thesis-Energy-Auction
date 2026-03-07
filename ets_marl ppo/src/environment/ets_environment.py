"""
ets_environment.py
==================
Two-phase EU ETS environment for multi-agent RL with technology-specific
energy mix, real-data-grounded investment costs, and construction queues.

Each year is split into two decision phases:
  Phase 1 (Auction + Investment):
    - Agents observe market state (18+2*(N-1) dim with opponent modeling)
    - Decide: [bid_price, quantity, invest_frac, tech_choice_logit0..2]
    - Auction clears, investments are planned
    - Returns enriched observation (phase1_dim+3) with auction results

  Phase 2 (Secondary Market):
    - Agents observe auction results (phase1_dim+3 dim)
    - Decide: [secondary_price, secondary_quantity]
    - Secondary market clears, compliance is checked
    - Returns reward and next year's Phase 1 observation

Action space (Phase 1): 6D continuous
  [bid_price, quantity, invest_frac, tech_logit_onshore, tech_logit_offshore, tech_logit_solar]
  tech_choice is derived by argmax of the 3 logits (discrete from continuous)

Action space (Phase 2): 2D continuous
  [price_multiplier, quantity]

Roadmap improvements (P1-P4):
  P1: 3-year price MA in obs; reserve_price floor
  P2: gae_lambda/entropy handled in ppo_agent + train; n_years = 12
  P3: Green penalty floor per agent; per-agent reward normalisation in PPOAgent
  P4: Green investment reward shaping with episode-level decay weight
  Opponent modeling (Option C): Phase 1 obs augmented with last-episode
    (bid/200, green_frac) for each other agent (+6 dims for 4 agents).
  Agent cycling (Option A): handled in train.py.
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
        self._phase1_bid_prices = None    # P1: store bid prices for bid cost penalty

        # P4: Per-agent fossil fraction history within the episode (last 3 years)
        # Used for the fossil lock-in penalty (3 consecutive years of no greening).
        self._fossil_frac_history: List[List[float]] = [[] for _ in range(self.n_agents)]

        # Opponent modeling (Option C): store last episode's avg bid and green_frac
        # per agent; used to augment Phase 1 observations.
        opp_enabled = config.get("opponent_modeling", {}).get("enabled", False)
        self._opponent_modeling = opp_enabled
        self._last_episode_bids = np.zeros(self.n_agents)
        self._last_episode_greens = np.array([
            c.green_frac for c in self.companies
        ])
        self._current_episode_bid_sum = np.zeros(self.n_agents)
        self._current_episode_bid_count = np.zeros(self.n_agents, dtype=int)

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
        self.shaping_weight = max(0.0, 1.0 - episode / decay_ep)

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
        self._price_history = []
        self._last_gaps = np.zeros(self.n_agents)
        self.holdings = np.zeros(self.n_agents)
        self.episode_log = []
        self._fossil_frac_history = [[] for _ in range(self.n_agents)]

        self.cap_schedule.reset()

        initial_mixes = self.config["companies"]["initial_mix"]
        for i, company in enumerate(self.companies):
            company.reset(initial_mix=initial_mixes[i])
            company.rng = self.rng

        obs_phase1 = self._get_obs_phase1()
        return obs_phase1, {}

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
        obs_phase2 : np.ndarray, shape (N_agents, 21)
        year_info : dict
        """
        assert not self.episode_done, "Episode done. Call reset()."

        year = self.current_year
        log = {"year": year}

        # Capture holdings at start of year for diagnostics (before allocations)
        bank_start = self.holdings.copy()
        log["bank_start"] = bank_start.tolist()

        # 1. Apply matured investments + reset annual budget
        for company in self.companies:
            company.apply_matured_investments(year)
            company.reset_budget()

        # 2. Compute TNAC and auction volume
        cap_t = self.cap_schedule.get_cap(year)
        tnac = float(self.holdings.sum())
        auction_volume = self.cap_schedule.get_auction_volume(year, tnac)
        log["cap"] = cap_t
        log["tnac"] = tnac
        log["auction_volume"] = auction_volume
        log["msr_reserve"] = self.cap_schedule.msr_reserve()

        # 3. Auction
        bid_actions = auction_actions[:, :2].copy()
        bid_actions[:, 0] = np.clip(
            bid_actions[:, 0],
            self.config["auction"]["price_min"],
            self.config["auction"]["price_max"],
        )
        # Store bid prices for diagnostics and opponent modeling
        self._phase1_bid_prices = bid_actions[:, 0].copy()
        if self._opponent_modeling:
            self._current_episode_bid_sum += self._phase1_bid_prices
            self._current_episode_bid_count += 1

        bids = build_bids(bid_actions)

        clearing_price, allocations, payments, auction_stats = market_clearing_ets(
            bids=bids,
            q_cap=auction_volume,
            reserve_price=self.config["ets"].get("reserve_price", 0.0),
        )
        self.last_clearing_price = clearing_price
        self._phase1_clearing_price = clearing_price
        log["clearing_price"] = clearing_price
        log["auction_stats"] = auction_stats

        # 4. Green investments
        invest_costs = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            invest_frac = float(auction_actions[i, 2])
            # Derive discrete tech choice from logits (argmax of 3 values)
            tech_logits = auction_actions[i, 3:6]
            tech_choice = int(np.argmax(tech_logits))  # 0=onshore, 1=offshore, 2=solar
            invest_costs[i] = company.plan_investment(tech_choice, invest_frac, year)

        # Store for phase 2
        self._phase1_allocations = allocations
        self._phase1_payments = payments
        self._phase1_invest_costs = invest_costs
        self._phase1_log = log

        # 5. Build phase 2 observations
        emissions = np.array([c.compute_emissions() for c in self.companies])
        obs_phase1 = self._get_obs_phase1()

        obs_phase2 = np.stack([
            self.companies[i].get_observation_phase2(
                obs_phase1=obs_phase1[i],
                allocation=allocations[i],
                clearing_price=clearing_price,
                emissions=emissions[i],
                banked=self.holdings[i],
            )
            for i in range(self.n_agents)
        ])

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
        clearing_price = self._phase1_clearing_price
        log = self._phase1_log

        # 5. Secondary market
        secondary_prices = clearing_price * np.clip(secondary_actions[:, 0], 0.5, 2.0)
        secondary_qtys = np.clip(
            secondary_actions[:, 1],
            -self.config["auction"]["quantity_max"],
            self.config["auction"]["quantity_max"],
        )

        trade_costs, trade_qtys, secondary_clearing = self._settle_double_auction(
            allocations=allocations.copy(),
            secondary_prices=secondary_prices,
            secondary_qtys=secondary_qtys,
            clearing_price=clearing_price,
        )

        self.last_secondary_price = secondary_clearing

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

        emissions = np.array([c.compute_emissions() for c in self.companies])

        # 6. Compliance
        penalties = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            penalties[i] = company.settle_compliance(allowances_held=holdings[i])

        # Banking: surplus carry forward
        for i in range(self.n_agents):
            self.holdings[i] = max(0.0, holdings[i] - emissions[i])
        self._last_gaps = self.holdings.copy()

        # P4: Update fossil fraction history (for lock-in penalty)
        for i, company in enumerate(self.companies):
            self._fossil_frac_history[i].append(company.fossil_frac)
            if len(self._fossil_frac_history[i]) > 3:
                self._fossil_frac_history[i].pop(0)

        # 7. Rewards (raw — normalisation happens in PPOAgent)
        rewards = self._compute_rewards(
            payments, trade_costs, penalties, invest_costs,
            emissions, clearing_price
        )

        # Compute per-agent shortfall for diagnostics
        shortfalls = np.array([
            max(0.0, emissions[i] - max(0.0, self.holdings[i] + allocations[i] + trade_qtys[i]))
            for i in range(self.n_agents)
        ])

        # 8. Log
        log.update({
            "allocations": allocations.tolist(),
            "payments": payments.tolist(),
            "emissions": emissions.tolist(),
            "trade_costs": trade_costs.tolist(),
            "trade_qtys": trade_qtys.tolist(),
            "secondary_clearing": secondary_clearing,
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
        })
        self.episode_log.append(log)

        # 9. Advance year + AR(1) price with stochastic shock
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
    # Secondary market — double auction
    # ------------------------------------------------------------------

    def _settle_double_auction(self, allocations, secondary_prices,
                                secondary_qtys, clearing_price):
        cfg = self.config["trading"]
        trade_costs = np.zeros(self.n_agents)
        trade_qtys = np.zeros(self.n_agents)

        if not cfg["enabled"]:
            return trade_costs, trade_qtys, clearing_price

        tx_cost = cfg["transaction_cost"]

        buyers = []
        sellers = []

        for i in range(self.n_agents):
            qty = float(secondary_qtys[i])
            price = float(secondary_prices[i])
            if qty > 1e-6:
                buyers.append([i, price, qty])
            elif qty < -1e-6:
                max_sell = float(allocations[i])
                sell_qty = min(abs(qty), max_sell)
                if sell_qty > 1e-6:
                    sellers.append([i, price, sell_qty])

        if not buyers or not sellers:
            return trade_costs, trade_qtys, clearing_price

        buyers.sort(key=lambda x: -x[1])
        sellers.sort(key=lambda x: x[1])

        executed_trades = []
        b_idx, s_idx = 0, 0
        while b_idx < len(buyers) and s_idx < len(sellers):
            buyer_id, buyer_price, buy_qty_rem = buyers[b_idx]
            seller_id, seller_price, sell_qty_rem = sellers[s_idx]

            if buyer_price < seller_price:
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
            return trade_costs, trade_qtys, clearing_price

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
        return trade_costs, trade_qtys, sec_clearing

    # ------------------------------------------------------------------
    # Reward function (P3 + P4 improvements)
    # ------------------------------------------------------------------

    def _compute_rewards(self, payments, trade_costs, penalties,
                         invest_costs, emissions, clearing_price):
        """
        R_i = -(alpha x TotalCost_i + beta x EmissionsIntensity_i + gamma x Penalty_i)
              + Shaping_i

        TotalCost includes: auction payments, secondary market costs,
        investment costs, operational costs (all in M€).

        P3: emissions intensity capped at initial fossil floor so agents are
            not punished for structurally irreducible fossil fractions.
        P4: green shaping bonuses (progress + queue + lock-in penalty).

        Note: per-agent running normalisation is applied in PPOAgent.normalize_reward()
        AFTER this function returns raw rewards.
        """
        rewards = np.zeros(self.n_agents)
        non_compliance_mult = self.config["penalty"].get("non_compliance_multiplier", 3.0)
        reward_cfg = self.config.get("reward", {})

        # P3: green penalty floor per agent (initial fossil fractions)
        green_floor_fossil = reward_cfg.get("green_floor_fossil", [0.0] * self.n_agents)

        # P4: shaping parameters
        beta_shaping = reward_cfg.get("shaping_beta", 10.0)
        gamma_queue = reward_cfg.get("shaping_gamma", 1.0)
        lockin_penalty = reward_cfg.get("shaping_lockin_penalty", 2.0)
        lockin_start = reward_cfg.get("shaping_lockin_start_episode", 200)

        for i, company in enumerate(self.companies):
            auction_cost = float(payments[i])
            secondary_cost = float(trade_costs[i])
            penalty_cost = float(penalties[i])
            investment_cost = float(invest_costs[i])
            operational_cost = company.compute_operational_cost()

            secondary_cost = max(secondary_cost, -auction_cost)

            # Record spending (auction + secondary + investment)
            company.record_spending(auction_cost + max(0.0, secondary_cost) + investment_cost)
            budget_penalty = company.compute_budget_penalty()

            # Total cost in M€
            total_cost = (auction_cost + secondary_cost + investment_cost
                         + operational_cost + budget_penalty)

            # Normalize cost (typical range: 0-2000 M€/year for 10 TWh company)
            cost_norm = total_cost / 1000.0

            # P3: Emissions intensity capped so agents are not penalised for
            # their structurally irreducible fossil fraction.
            # Initial floor = initial fossil fraction per agent.
            fossil_floor_i = green_floor_fossil[i] if i < len(green_floor_fossil) else 0.0
            # Compute the emission factor at the initial fossil floor
            # (weighted EF that would correspond to max-penalisable fossil level)
            initial_ef_at_floor = fossil_floor_i * max(company.emission_factors[~company.is_green])
            # Actual emissions intensity, clipped so nothing below the floor contributes
            # (i.e., A4 gets zero green penalty if its fossil_frac ≤ floor)
            penalisable_ef = max(0.0, company.weighted_emission_factor - initial_ef_at_floor)
            emissions_intensity = penalisable_ef / 0.82  # normalised to [0,1] scale

            # Penalty component (compliance non-compliance, scaled harshly)
            penalty_norm = penalty_cost * non_compliance_mult / 1000.0

            # P4: Green investment shaping rewards
            # a) Progress bonus: immediate reward for any green fraction increase
            green_delta = max(0.0, company.green_frac - company.prev_green_frac)
            shaping = beta_shaping * green_delta * self.shaping_weight

            # b) Queue progress reward: small bonus per project under construction
            n_queue_active = sum(
                1 for item in company._construction_queue
                if item.get("frac_delta", 0) > 0
            )
            shaping += gamma_queue * n_queue_active * 0.1 * self.shaping_weight

            # c) Fossil lock-in penalty: activated after warm-up period.
            # If fossil_frac unchanged for 3 consecutive years, penalise stagnation.
            if self.current_episode >= lockin_start and self.shaping_weight > 0:
                hist = self._fossil_frac_history[i]
                if len(hist) >= 3:
                    unchanged = all(abs(f - hist[-1]) < 1e-4 for f in hist[-2:])
                    if unchanged:
                        shaping -= lockin_penalty * self.shaping_weight

            rewards[i] = -(
                company.w_cost * cost_norm
                + company.w_green * emissions_intensity
                + penalty_norm
            ) + shaping

        return rewards

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _compute_price_ma3(self) -> float:
        """
        P1: 3-year moving average of clearing price.
        Falls back to last_clearing_price when history is short.
        Provides a more stable price signal than a single noisy observation.
        """
        if not self._price_history:
            return self.last_clearing_price
        window = self._price_history[-3:]
        return float(np.mean(window))

    def _get_obs_phase1(self) -> np.ndarray:
        """Phase 1 observations for all agents.
        Base: 18D. With opponent modeling: 18 + 2*(N-1) dims.
        """
        cap_t = self.cap_schedule.get_cap(self.current_year)
        price_ma3 = self._compute_price_ma3()  # P1: 3-year MA price

        obs_list = []
        for i, c in enumerate(self.companies):
            # Build per-agent opponent obs: (bid_j/200, green_j) for all j != i
            if self._opponent_modeling and self.n_agents > 1:
                opp_parts = []
                for j in range(self.n_agents):
                    if j != i:
                        opp_parts.append(float(self._last_episode_bids[j]) / 200.0)
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
            )
            obs_list.append(obs_i)

        return np.stack(obs_list)
