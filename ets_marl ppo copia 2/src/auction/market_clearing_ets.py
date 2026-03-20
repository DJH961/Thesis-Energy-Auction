"""
market_clearing_ets.py
======================
Uniform-price sealed-bid multi-unit buyer auction clearing for EU ETS.

Forked and adapted from:
    ckrk/bidding_learning (MIT License, 2024)
    https://github.com/ckrk/bidding_learning

Key differences from the original (energy seller market):
    - Bids are BUYER bids: agents bid the MAX price they are willing to pay.
    - Sort order is DESCENDING by price (highest willingness-to-pay first).
    - The clearing price is the LOWEST accepted bid (marginal buyer).
    - Every winner pays the uniform clearing price (not their own bid).
    - Supply side = fixed cap volume Q_cap offered by the auctioneer.

EU ETS reference:
    Single-round, sealed-bid, uniform-price auction.
    EU Auctioning Regulation (Commission Regulation 1031/2010).

Stylised simplifications vs real EU ETS:
    - ``reserve_price`` acts as a stylised *price floor* to prevent
      zero-price learning artefacts.  In the real EU ETS, the reserve
      price is computed via a different formula (CREG).
    - Under-subscription (total demand < supply) *optionally* cancels the
      auction, mirroring Article 7(6) of Regulation 1031/2010.  Disabled
      by default for training; enable via ``cancel_under_subscribed=True``.
    - Tie-break at the marginal price uses random ranking (not pro-rata):
      tied bids are shuffled randomly and filled sequentially; only the
      last successful bid in the random order receives a partial fill.
"""


import numpy as np


# ---------------------------------------------------------------------------
# Main clearing function
# ---------------------------------------------------------------------------

def market_clearing_ets(bids: np.ndarray, q_cap: float, reserve_price: float = 0.0,
                        max_agent_share: float = 1.0, rng=None,
                        cancel_under_subscribed: bool = False):
    """
    Uniform-price sealed-bid buyer auction clearing (EU ETS style).

    Parameters
    ----------
    bids : np.ndarray, shape (N, 3)
        Each row: [agent_id (int), quantity_bid (Mt), price_bid (EUR/t)]
    q_cap : float
        Total allowances offered by the auctioneer (Mt).
    reserve_price : float
        Stylised minimum accepted price (EUR/t).  Bids below this are
        discarded.  Not identical to the real EU ETS reserve price
        (CREG formula), but serves an analogous role.
    max_agent_share : float
        Maximum fraction of q_cap any single agent can receive (California-
        style holding limit).  Default 1.0 = no limit.  E.g. 0.25 means
        each agent can receive at most 25 % of the total auctioned volume.
    rng : np.random.Generator or None
        Random number generator for reproducible tie-breaking.  If None,
        a fresh default generator is created.
    cancel_under_subscribed : bool
        If True, the auction is cancelled when total valid demand < q_cap
        (Article 7(6) of Reg. 1031/2010).  If False (default), the auction
        proceeds and sells whatever is demanded — better for RL training
        where early-stage agents may bid insufficient quantities.

    Returns
    -------
    clearing_price : float
        Uniform price paid by all winners (EUR/t).
    allocations : np.ndarray, shape (n_agents,)
        Allowances allocated to each agent (indexed by agent_id).
    payments : np.ndarray, shape (n_agents,)
        Total payment by each agent = allocation * clearing_price.
    auction_stats : dict
        Diagnostic information (cover ratio, total demand, etc.).
    """
    bids = np.array(bids, dtype=float)

    # --- Safeguard: empty bids array ---
    if bids.ndim != 2 or len(bids) == 0:
        stats = {
            "clearing_price": reserve_price,
            "total_demand": 0.0,
            "total_allocated": 0.0,
            "cover_ratio": 0.0,
            "auction_failed": True,
            "fail_reason": "no_bids",
            "unsold": q_cap,
        }
        return reserve_price, np.zeros(0, dtype=float), np.zeros(0, dtype=float), stats

    n_agents = int(bids[:, 0].max()) + 1  # infer from max agent_id
    allocations = np.zeros(n_agents, dtype=float)
    payments = np.zeros(n_agents, dtype=float)

    # --- Filter: remove invalid bids ---
    valid_mask = (bids[:, 1] > 0) & (bids[:, 2] >= reserve_price)
    valid_bids = bids[valid_mask]

    if len(valid_bids) == 0:
        # No valid bids: auction fails, nothing sold
        stats = {
            "clearing_price": reserve_price,
            "total_demand": 0.0,
            "total_allocated": 0.0,
            "cover_ratio": 0.0,
            "auction_failed": True,
            "fail_reason": "all_below_reserve",
            "unsold": q_cap,
        }
        return reserve_price, allocations, payments, stats

    total_demand = valid_bids[:, 1].sum()

    # --- Under-subscription check (Article 7(6) of Reg. 1031/2010) ---
    # If total valid demand does not reach the supply, the auction is
    # cancelled and no allowances are sold.  Disabled by default for
    # RL training (early agents bid too little → uninformative signal).
    if cancel_under_subscribed and total_demand < q_cap - 1e-9:
        stats = {
            "clearing_price": reserve_price,
            "total_demand": total_demand,
            "total_allocated": 0.0,
            "cover_ratio": total_demand / q_cap if q_cap > 0 else 0.0,
            "auction_failed": True,
            "fail_reason": "under_subscribed",
            "unsold": q_cap,
        }
        return reserve_price, allocations, payments, stats

    # --- Sort DESCENDING by price; random tie-break at same price ---
    if rng is None:
        rng = np.random.default_rng()
    tiebreakers = rng.random(len(valid_bids))
    sort_idx = np.lexsort((tiebreakers, -valid_bids[:, 2]))
    sorted_bids = valid_bids[sort_idx]

    # --- Walk through sorted bids, fill up to q_cap ---
    # California-style holding limit: each agent can receive at most
    # max_agent_share * q_cap allowances across all their bids.
    per_agent_cap = max_agent_share * q_cap
    agent_cumul = np.zeros(n_agents, dtype=float)

    cumulative = 0.0
    clearing_price = reserve_price
    alloc_per_bid = np.zeros(len(sorted_bids), dtype=float)

    for i, (agent_id, qty, price) in enumerate(sorted_bids):
        if cumulative >= q_cap - 1e-9:
            break
        aid = int(agent_id)
        agent_room = per_agent_cap - agent_cumul[aid]
        if agent_room <= 1e-9:
            continue  # this agent already at holding limit
        take = min(qty, q_cap - cumulative, agent_room)
        alloc_per_bid[i] = take
        cumulative += take
        agent_cumul[aid] += take
        clearing_price = price

    # --- Aggregate allocations back to agents ---
    for i, (agent_id, _, _) in enumerate(sorted_bids):
        allocations[int(agent_id)] += alloc_per_bid[i]

    # --- Compute payments (uniform price) ---
    payments = allocations * clearing_price

    # --- Auction statistics ---
    total_allocated = allocations.sum()
    cover_ratio = total_demand / q_cap if q_cap > 0 else 0.0
    unsold = max(0.0, q_cap - total_allocated)

    stats = {
        "clearing_price": clearing_price,
        "total_demand": total_demand,
        "total_allocated": total_allocated,
        "cover_ratio": cover_ratio,
        "auction_failed": False,
        "unsold": unsold,
    }

    return clearing_price, allocations, payments, stats


# ---------------------------------------------------------------------------
# Convenience: build bids array from per-agent (price, quantity) actions
# ---------------------------------------------------------------------------

def build_bids(actions: np.ndarray) -> np.ndarray:
    """
    Convert agent actions to bids array for market_clearing_ets.

    Parameters
    ----------
    actions : np.ndarray, shape (N, 2)
        Each row: [price_bid (EUR/t), quantity_bid (Mt)] for agent i.

    Returns
    -------
    bids : np.ndarray, shape (N, 3)
        Each row: [agent_id, quantity_bid, price_bid]
    """
    n = len(actions)
    agent_ids = np.arange(n, dtype=float).reshape(-1, 1)
    # reorder: [agent_id, quantity, price]
    bids = np.hstack([agent_ids, actions[:, 1:2], actions[:, 0:1]])
    return bids
