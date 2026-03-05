"""
ets_environment.py
==================
Two-phase EU ETS environment for multi-agent RL.

Each year is split into two decision phases:
  Phase 1 (Auction):
    - Agents observe market state (12 dim)
    - Decide: [bid_price, quantity, delta_green]
    - Auction clears, investments are planned
    - Returns enriched observation (15 dim) with auction results

  Phase 2 (Secondary Market):
    - Agents observe auction results (15 dim)
    - Decide: [secondary_price, secondary_quantity]
    - Secondary market clears, compliance is checked
    - Returns reward and next year's Phase 1 observation

This two-phase structure ensures agents make secondary market
decisions AFTER seeing their auction allocation, enabling
informed arbitrage strategies.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
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
                initial_fossil_frac=initial_mixes[i][0], rng=self.rng,
            )
            for i in range(self.n_agents)
        ]

        # Episode state
        self.current_year = 0
        self.last_clearing_price = config["price"]["initial_expected"]
        self.expected_price = config["price"]["initial_expected"]
        self._price_history: List[float] = []
        self.last_secondary_price = config["price"]["initial_expected"]
        self._last_gaps = np.zeros(self.n_agents)
        self.holdings = np.zeros(self.n_agents)   # banked allowances carried across years
        self.episode_done = False

        # Secondary market profit tracking (EMA per agent)
        self._secondary_profit_ema = np.zeros(self.n_agents)  # €/Mt
        self._ema_alpha = 0.1  # smoothing factor

        # Auction results (stored between phase 1 and phase 2)
        self._phase1_allocations = None
        self._phase1_payments = None
        self._phase1_clearing_price = 0.0
        self._phase1_switching_costs = None
        self._phase1_obs = None  # phase 1 observations for logging

        # Logging
        self.episode_log: List[dict] = []

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
        self._price_history = []
        self._last_gaps = np.zeros(self.n_agents)
        self.holdings = np.zeros(self.n_agents)
        self.episode_log = []

        # Don't reset EMA — it carries profit signal across episodes
        # (this is intentional: the agent should learn from past episodes
        # that secondary market is profitable)

        self.cap_schedule.reset()

        initial_mixes = self.config["companies"]["initial_mix"]
        for i, company in enumerate(self.companies):
            company.reset(initial_fossil_frac=initial_mixes[i][0])
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
        auction_actions : np.ndarray, shape (N_agents, 3)
            [bid_price, quantity, delta_green] per agent.

        Returns
        -------
        obs_phase2 : np.ndarray, shape (N_agents, 15)
            Enriched observation including auction results.
        year_info : dict
            Auction results for logging.
        """
        assert not self.episode_done, "Episode done. Call reset()."

        year = self.current_year
        log = {"year": year}

        # 1. Apply matured investments + reset annual budget
        for company in self.companies:
            company.apply_matured_investments(year)
            company.reset_budget()

        # 2. Compute TNAC and auction volume
        # TNAC = total banked allowances in circulation (not surrendered yet)
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
        switching_costs = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            switching_costs[i] = company.plan_investment(auction_actions[i, 2])

        # Store for phase 2
        self._phase1_allocations = allocations
        self._phase1_payments = payments
        self._phase1_switching_costs = switching_costs
        self._phase1_log = log

        # 5. Build phase 2 observations
        emissions = np.array([c.compute_emissions() for c in self.companies])
        obs_phase1 = self._get_obs_phase1()  # current state

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
            price_multiplier in [0.5, 2.0] × clearing_price
            quantity: positive=buy, negative=sell

        Returns
        -------
        obs_next : np.ndarray, shape (N_agents, 12)
            Phase 1 observation for next year.
        rewards : np.ndarray, shape (N_agents,)
        terminated : bool
        truncated : bool
        info : dict
        """
        allocations = self._phase1_allocations
        payments = self._phase1_payments
        switching_costs = self._phase1_switching_costs
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
            if trade_qtys[i] < -1e-6:  # sold something
                profit_per_mt = -trade_costs[i] / abs(trade_qtys[i])  # revenue per Mt
                margin = profit_per_mt - clearing_price  # profit above auction price
                self._secondary_profit_ema[i] = (
                    self._ema_alpha * margin +
                    (1 - self._ema_alpha) * self._secondary_profit_ema[i]
                )
            # Buyers: update with negative signal (buying is a cost)
            elif trade_qtys[i] > 1e-6:
                cost_per_mt = trade_costs[i] / trade_qtys[i]
                margin = clearing_price - cost_per_mt  # savings vs penalty
                self._secondary_profit_ema[i] = (
                    self._ema_alpha * margin +
                    (1 - self._ema_alpha) * self._secondary_profit_ema[i]
                )

        # Holdings after secondary market (banked from last year + new allocation + trades)
        holdings = self.holdings + allocations + trade_qtys

        emissions = np.array([c.compute_emissions() for c in self.companies])

        # 6. Compliance
        penalties = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            penalties[i] = company.settle_compliance(allowances_held=holdings[i])

        # Banking: surplus allowances carry forward to next year
        for i in range(self.n_agents):
            self.holdings[i] = max(0.0, holdings[i] - emissions[i])
        self._last_gaps = self.holdings.copy()

        # 7. Rewards
        rewards = self._compute_rewards(payments, trade_costs, penalties, switching_costs)

        # 8. Log
        log.update({
            "allocations": allocations.tolist(),
            "payments": payments.tolist(),
            "emissions": emissions.tolist(),
            "trade_costs": trade_costs.tolist(),
            "trade_qtys": trade_qtys.tolist(),
            "secondary_clearing": secondary_clearing,
            "penalties": penalties.tolist(),
            "switching_costs": switching_costs.tolist(),
            "rewards": rewards.tolist(),
            "green_fracs": [c.green_frac for c in self.companies],
            "holdings": holdings.tolist(),
        })
        self.episode_log.append(log)

        # 9. Advance year + AR(1) price
        self._price_history.append(clearing_price)
        rho = self.config["price"].get("ar1_persistence", 0.85)
        price_floor = self.config["price"].get("price_floor", 42.0)
        if clearing_price > 0:
            self.expected_price = max(
                rho * clearing_price + (1.0 - rho) * price_floor,
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
        """Single-call step for backward compat. actions shape (N, 5)."""
        obs2, _ = self.step_auction(actions[:, :3])
        return self.step_secondary(actions[:, 3:])

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
    # Reward
    # ------------------------------------------------------------------

    def _compute_rewards(self, payments, trade_costs, penalties, switching_costs):
        rewards = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            auction_cost = float(payments[i])
            secondary_cost = float(trade_costs[i])  # negative = profit
            penalty_cost = float(penalties[i])
            invest_cost = float(switching_costs[i])

            secondary_cost = max(secondary_cost, -auction_cost)

            # Record actual carbon spending for budget tracking (no negative spending)
            company.record_spending(auction_cost + max(0.0, secondary_cost))
            budget_penalty = company.compute_budget_penalty()

            total_cost = auction_cost + secondary_cost + penalty_cost + invest_cost + budget_penalty
            cost_norm = total_cost / 300.0

            green_component = company.fossil_frac

            green_delta = company.green_frac - company.prev_green_frac
            green_bonus = company.w_green * max(0.0, green_delta) * 10.0

            rewards[i] = -(
                company.w_cost * cost_norm
                + company.w_green * green_component
            ) + green_bonus
        return rewards

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs_phase1(self) -> np.ndarray:
        """Phase 1 observations (12 dim) for all agents."""
        cap_t = self.cap_schedule.get_cap(self.current_year)
        obs = np.stack([
            c.get_observation_phase1(
                year=self.current_year,
                cap_t=cap_t,
                last_clearing_price=self.last_clearing_price,
                expected_price=self.expected_price,
                auction_gap=self._last_gaps[i],
                last_secondary_price=self.last_secondary_price,
                secondary_profit_signal=self._secondary_profit_ema[i],
            )
            for i, c in enumerate(self.companies)
        ])
        return obs
