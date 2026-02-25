"""
ets_environment.py
==================
Multi-year EU ETS environment (Gymnasium-compatible).

One episode = N years (default 10).
One step = one year:
  1. Apply matured green investments
  2. Compute TNAC → auction volume (LRF + MSR)
  3. Auction: agents submit (bid_price, quantity) → uniform-price clearing
  4. Green investment decision
  5. Secondary market: double auction (agents submit price + quantity)
  6. Compliance: surrender allowances, compute penalties
  7. Compute rewards
  8. Advance year

Action space per agent: 5-dimensional
    [0] bid_price      : absolute bid price in €/t, range [price_min, price_max]
    [1] quantity_bid   : Mt bid at auction, range [0, quantity_max]
    [2] delta_green    : p.p./year green investment, range [0, max_delta_green]
    [3] secondary_price: multiplier on clearing price for secondary market,
                         range [0.5, 2.0]  (0.5 = deep discount, 2.0 = high ask)
    [4] secondary_qty  : Mt to trade on secondary market,
                         range [-quantity_max, quantity_max]
                         positive = buy, negative = sell

Observation space per agent: 11-dimensional (see Company.get_observation)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

from src.auction.market_clearing_ets import market_clearing_ets, build_bids
from src.environment.cap_schedule import CapSchedule
from src.environment.company import Company


class ETSEnvironment(gym.Env):
    """
    Multi-agent EU ETS environment (single Gymnasium env wrapping N agents).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__()

        self.config = config
        self.n_agents = config["companies"]["n_agents"]
        self.n_years = config["simulation"]["n_years"]

        # Seeded RNG
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Sub-components
        self.cap_schedule = CapSchedule(config)

        initial_mixes = config["companies"]["initial_mix"]
        self.companies: List[Company] = [
            Company(
                agent_id=i,
                config=config,
                initial_fossil_frac=initial_mixes[i][0],
                rng=self.rng,
            )
            for i in range(self.n_agents)
        ]

        # Action/observation spaces
        self._build_spaces()

        # Episode state
        self.current_year = 0
        self.last_clearing_price = config["price"]["initial_expected"]
        self.expected_price = config["price"]["initial_expected"]
        self._price_history: List[float] = []
        self.last_secondary_price = config["price"]["initial_expected"]
        self._last_gaps = np.zeros(self.n_agents)
        self.episode_done = False

        # Per-episode logging
        self.episode_log: List[dict] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        """Reset environment for a new episode."""
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
        self.episode_log = []

        self.cap_schedule.reset()

        initial_mixes = self.config["companies"]["initial_mix"]
        for i, company in enumerate(self.companies):
            company.reset(initial_fossil_frac=initial_mixes[i][0])
            company.rng = self.rng

        obs = self._get_obs()
        return obs, {}

    def step(self, actions: np.ndarray):
        """
        Execute one year of the simulation.

        Parameters
        ----------
        actions : np.ndarray, shape (N_agents, 5)
            Each row: [bid_price, quantity_bid, delta_green,
                       secondary_price_multiplier, secondary_qty]
        """
        assert not self.episode_done, "Episode done. Call reset()."

        year = self.current_year
        log = {"year": year}

        # ----------------------------------------------------------------
        # 1. Apply matured investments
        # ----------------------------------------------------------------
        for company in self.companies:
            company.apply_matured_investments(year)

        # ----------------------------------------------------------------
        # 2. Compute TNAC and auction volume
        # ----------------------------------------------------------------
        total_estimate = sum(c.compute_estimate_need() for c in self.companies)
        cap_t = self.cap_schedule.get_cap(year)
        # TNAC = allowance surplus in circulation (cap − demand)
        # Positive → excess supply, triggers MSR withholding
        # Zero     → scarcity or balance, no MSR action
        tnac = max(0.0, cap_t - total_estimate)
        auction_volume = self.cap_schedule.get_auction_volume(year, tnac)
        log["cap"] = cap_t
        log["tnac"] = tnac
        log["auction_volume"] = auction_volume
        log["msr_reserve"] = self.cap_schedule.msr_reserve()

        # ----------------------------------------------------------------
        # 3. Auction (uniform-price sealed-bid)
        # action[:,0] = absolute bid price in €/t, range [price_min, price_max]
        # action[:,1] = quantity bid in Mt
        # ----------------------------------------------------------------
        auction_actions = actions[:, :2].copy()
        # action[0] is already the bid price — just clip to valid range
        auction_actions[:, 0] = np.clip(
            auction_actions[:, 0],
            self.config["auction"]["price_min"],
            self.config["auction"]["price_max"],
        )
        bids = build_bids(auction_actions)

        clearing_price, allocations, payments, auction_stats = market_clearing_ets(
            bids=bids,
            q_cap=auction_volume,
            reserve_price=self.config["ets"].get("reserve_price", 0.0),
        )
        self.last_clearing_price = clearing_price
        log["clearing_price"] = clearing_price
        log["auction_stats"] = auction_stats

        # Gap provvisorio post-asta (aggiornato dopo investimenti)
        provisional_needs = np.array([c.compute_emissions() for c in self.companies])
        self._last_gaps = allocations - provisional_needs

        # ----------------------------------------------------------------
        # 4. Green investments
        # ----------------------------------------------------------------
        switching_costs = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            switching_costs[i] = company.plan_investment(actions[i, 2])

        # ----------------------------------------------------------------
        # 5. Secondary market — double auction (RL decision)
        #
        # action[:,3] = price_multiplier in [0.5, 2.0]
        #   → secondary_price = clearing_price * price_multiplier
        # action[:,4] = quantity in [-quantity_max, +quantity_max]
        #   → positive = want to buy, negative = want to sell
        #
        # Matching: sort buyers DESC by price, sellers ASC by price.
        # Match if buyer_price >= seller_price → trade at midpoint price.
        # ----------------------------------------------------------------
        actual_needs = np.array([c.compute_emissions() for c in self.companies])
        secondary_prices = clearing_price * np.clip(actions[:, 3], 0.5, 2.0)
        secondary_qtys = np.clip(
            actions[:, 4],
            -self.config["auction"]["quantity_max"],
            self.config["auction"]["quantity_max"],
        )

        trade_costs, trade_qtys_executed, secondary_clearing = \
            self._settle_double_auction(
                allocations=allocations.copy(),
                secondary_prices=secondary_prices,
                secondary_qtys=secondary_qtys,
                clearing_price=clearing_price,
            )

        self.last_secondary_price = secondary_clearing

        # Holdings after secondary market
        holdings = allocations + trade_qtys_executed

        # ----------------------------------------------------------------
        # 6. Compliance and penalties
        # ----------------------------------------------------------------
        penalties = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            penalties[i] = company.settle_compliance(allowances_held=holdings[i])

        # ----------------------------------------------------------------
        # 7. Compute rewards
        # ----------------------------------------------------------------
        rewards = self._compute_rewards(
            payments=payments,
            trade_costs=trade_costs,
            penalties=penalties,
            switching_costs=switching_costs,
        )

        # ----------------------------------------------------------------
        # 8. Log year
        # ----------------------------------------------------------------
        log.update({
            "allocations": allocations.tolist(),
            "payments": payments.tolist(),
            "trade_costs": trade_costs.tolist(),
            "trade_qtys": trade_qtys_executed.tolist(),
            "secondary_clearing": secondary_clearing,
            "penalties": penalties.tolist(),
            "switching_costs": switching_costs.tolist(),
            "rewards": rewards.tolist(),
            "green_fracs": [c.green_frac for c in self.companies],
        })
        self.episode_log.append(log)

        # ----------------------------------------------------------------
        # 9. Advance year — update expected price via AR(1) process
        #    E[p_{t+1}] = rho * p_t + (1 - rho) * p_long_run
        #    where p_long_run = floor (equilibrium under scarcity)
        # ----------------------------------------------------------------
        self._price_history.append(clearing_price)
        rho = self.config["price"].get("ar1_persistence", 0.85)
        price_floor = self.config["price"].get("price_floor", 45.5)
        if clearing_price > 0:
            self.expected_price = max(
                rho * clearing_price + (1.0 - rho) * price_floor,
                price_floor,
            )
        # If clearing_price == 0 (no auction), keep previous expected_price

        self.current_year += 1
        terminated = self.current_year >= self.n_years
        truncated = False
        self.episode_done = terminated

        obs = self._get_obs()
        return obs, rewards, terminated, truncated, {"year_log": log}

    # ------------------------------------------------------------------
    # Secondary market — double auction
    # ------------------------------------------------------------------

    def _settle_double_auction(
        self,
        allocations: np.ndarray,
        secondary_prices: np.ndarray,
        secondary_qtys: np.ndarray,
        clearing_price: float,
    ):
        """
        Double auction for the secondary market.

        Buyers (secondary_qty > 0) submit a max willingness-to-pay price.
        Sellers (secondary_qty < 0) submit a min acceptable price.
        Matching: sort buyers DESC, sellers ASC.
        Trade when buyer_price >= seller_price at midpoint price.
        Transaction cost applied per unit traded.

        Returns
        -------
        trade_costs : np.ndarray
            Cost per agent (positive = paid, negative = received). M€ units.
        trade_qtys_executed : np.ndarray
            Actual quantity traded per agent (positive = bought, negative = sold).
        secondary_clearing : float
            Average price of executed trades (or clearing_price if no trades).
        """
        cfg = self.config["trading"]
        trade_costs = np.zeros(self.n_agents)
        trade_qtys_executed = np.zeros(self.n_agents)

        if not cfg["enabled"]:
            return trade_costs, trade_qtys_executed, clearing_price

        tx_cost = cfg["transaction_cost"]  # €/t

        # Separate buyers and sellers
        buyers = []   # (agent_id, price, qty_wanted)
        sellers = []  # (agent_id, price, qty_available)

        for i in range(self.n_agents):
            qty = float(secondary_qtys[i])
            price = float(secondary_prices[i])
            if qty > 1e-6:
                buyers.append([i, price, qty])
            elif qty < -1e-6:
                # Can't sell more than you hold from auction
                max_sell = float(allocations[i])
                sell_qty = min(abs(qty), max_sell)
                if sell_qty > 1e-6:
                    sellers.append([i, price, sell_qty])

        if not buyers or not sellers:
            return trade_costs, trade_qtys_executed, clearing_price

        # Sort: buyers highest price first, sellers lowest price first
        buyers.sort(key=lambda x: -x[1])
        sellers.sort(key=lambda x: x[1])

        executed_trades = []  # list of (buyer_id, seller_id, price, qty)

        b_idx, s_idx = 0, 0
        while b_idx < len(buyers) and s_idx < len(sellers):
            buyer_id, buyer_price, buy_qty_rem = buyers[b_idx]
            seller_id, seller_price, sell_qty_rem = sellers[s_idx]

            if buyer_price < seller_price:
                break  # no more matches possible

            # Trade at midpoint price
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
            return trade_costs, trade_qtys_executed, clearing_price

        # Aggregate results
        total_value = 0.0
        total_qty = 0.0

        for buyer_id, seller_id, trade_price, trade_qty in executed_trades:
            cost_buyer = trade_qty * (trade_price + tx_cost)    # buyer pays
            revenue_seller = trade_qty * (trade_price - tx_cost) # seller receives

            trade_costs[buyer_id] += cost_buyer
            trade_costs[seller_id] -= revenue_seller   # negative = received money

            trade_qtys_executed[buyer_id] += trade_qty
            trade_qtys_executed[seller_id] -= trade_qty

            total_value += trade_price * trade_qty
            total_qty += trade_qty

        secondary_clearing = total_value / total_qty if total_qty > 0 else clearing_price

        return trade_costs, trade_qtys_executed, secondary_clearing

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------

    def _compute_rewards(self, payments, trade_costs, penalties, switching_costs):
        """
        reward_i = -(w_cost * cost_norm + w_green * fossil_frac) + green_bonus

        UNITS — all already in M€:
          payments        : allocations [Mt] * clearing_price [€/t] = M€
          trade_costs     : trade_qty [Mt] * price [€/t] = M€ (negative = gain)
          penalties       : shortfall [Mt] * penalty_rate [€/t] = M€
          switching_costs : from plan_investment, defined in M€/p.p.

        Normalisation scale = 300 M€:
          worst case A1: ~226 M€ auction + ~67 M€ penalty + ~1.5 M€ invest ≈ 295 M€
          This keeps cost_norm in [0, ~1] for typical scenarios.

        green_bonus: small positive reward for increasing green_frac vs previous year.
          Incentivises steady decarbonisation. Scaled by w_green so green-driven
          agents benefit more.

        trade_costs gain is capped at auction spend to prevent reward explosion
        when a company sells large quantities on the secondary market.
        """
        rewards = np.zeros(self.n_agents)
        for i, company in enumerate(self.companies):
            auction_cost   = float(payments[i])         # M€
            secondary_cost = float(trade_costs[i])      # M€, negative = gain
            penalty_cost   = float(penalties[i])        # M€
            invest_cost    = float(switching_costs[i])  # M€

            # Cap gain from secondary: can't earn more than auction spend
            secondary_cost = max(secondary_cost, -auction_cost)

            total_cost = auction_cost + secondary_cost + penalty_cost + invest_cost

            # Normalise: ~300 M€ = realistic worst case for this micro-ETS
            cost_norm = total_cost / 300.0

            # Green component: 0 = fully green (good), 1 = fully fossil (bad)
            green_component = company.fossil_frac

            # Green bonus: reward for increasing green_frac since last year
            green_delta = company.green_frac - company.prev_green_frac
            green_bonus = company.w_green * max(0.0, green_delta) * 10.0  # scale ~0.3 pp → ~0.09

            rewards[i] = -(
                company.w_cost  * cost_norm
                + company.w_green * green_component
            ) + green_bonus
        return rewards

    # ------------------------------------------------------------------
    # Observation and spaces
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return observations for all agents.

        NOTE: After step(), current_year is already incremented, so
        agents observe the NEXT year's cap. This is intentional: agents
        need to plan ahead for the upcoming auction.
        """
        cap_t = self.cap_schedule.get_cap(self.current_year)
        obs = np.stack([
            c.get_observation(
                year=self.current_year,
                cap_t=cap_t,
                last_clearing_price=self.last_clearing_price,
                expected_price=self.expected_price,
                auction_gap=self._last_gaps[i],
                last_secondary_price=self.last_secondary_price,
            )
            for i, c in enumerate(self.companies)
        ])
        return obs

    def _build_spaces(self):
        """Define action and observation spaces."""
        aq = self.config["auction"]
        inv = self.config["investment"]

        # Action per agent: [bid_price, qty_bid, delta_green,
        #                     secondary_price_mult, secondary_qty]
        act_low = np.array([
            aq["price_min"],               # bid price min (€/t)
            0.0,                           # qty_bid min
            0.0,                           # delta_green min
            0.5,                           # secondary price mult min (50% of clearing)
            -aq["quantity_max"],           # secondary qty min (sell)
        ], dtype=np.float32)

        act_high = np.array([
            aq["price_max"],               # bid price max (€/t)
            aq["quantity_max"],            # qty_bid max
            inv["max_delta_green_pp"],     # delta_green max
            2.0,                           # secondary price mult max (200% of clearing)
            aq["quantity_max"],            # secondary qty max (buy)
        ], dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.tile(act_low, (self.n_agents, 1)),
            high=np.tile(act_high, (self.n_agents, 1)),
            dtype=np.float32,
        )

        obs_dim = self.companies[0].obs_dim  # 11
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, obs_dim),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode="human"):
        if not self.episode_log:
            print("No data yet.")
            return
        last = self.episode_log[-1]
        print(f"\n--- Year {last['year']} ---")
        print(f"  Cap:                {last['cap']:.3f} Mt")
        print(f"  Auction volume:     {last['auction_volume']:.3f} Mt")
        print(f"  TNAC:               {last['tnac']:.3f} Mt")
        print(f"  MSR reserve:        {last['msr_reserve']:.3f} Mt")
        print(f"  Clearing price:     {last['clearing_price']:.2f} €/t")
        print(f"  Secondary price:    {last['secondary_clearing']:.2f} €/t")
        print(f"  Trade qtys:         {[f'{q:.3f}' for q in last['trade_qtys']]}")
        print(f"  Green fracs:        {[f'{g:.2f}' for g in last['green_fracs']]}")
        print(f"  Penalties:          {[f'{p:.4f}' for p in last['penalties']]}")
        print(f"  Rewards:            {[f'{r:.2f}' for r in last['rewards']]}")