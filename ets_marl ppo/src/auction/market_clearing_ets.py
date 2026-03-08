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
"""

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Main clearing function
# ---------------------------------------------------------------------------

def market_clearing_ets(bids: np.ndarray, q_cap: float, reserve_price: float = 0.0):
    """
    Uniform-price sealed-bid buyer auction clearing (EU ETS style).

    Parameters
    ----------
    bids : np.ndarray, shape (N, 3)
        Each row: [agent_id (int), quantity_bid (Mt), price_bid (€/t)]
    q_cap : float
        Total allowances offered by the auctioneer (Mt).
    reserve_price : float
        Minimum accepted price (€/t). Bids below this are invalid.

    Returns
    -------
    clearing_price : float
        Uniform price paid by all winners (€/t).
    allocations : np.ndarray, shape (N,)
        Allowances allocated to each agent (indexed by agent_id).
    payments : np.ndarray, shape (N,)
        Total payment by each agent = allocation * clearing_price.
    auction_stats : dict
        Diagnostic information (cover ratio, total demand, etc.).
    """
    bids = np.array(bids, dtype=float)

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
        }
        return reserve_price, allocations, payments, stats

    # --- Sort DESCENDING by price (highest willingness-to-pay first) ---
    # Tie-break: stable sort by agent_id (deterministic)
    sort_idx = np.lexsort((valid_bids[:, 0], -valid_bids[:, 2]))
    sorted_bids = valid_bids[sort_idx]

    total_demand = sorted_bids[:, 1].sum()

    # --- Walk through sorted bids, fill up to q_cap ---
    cumulative = 0.0
    clearing_price = reserve_price
    alloc_per_bid = np.zeros(len(sorted_bids), dtype=float)

    marginal_idx = None
    for i, (agent_id, qty, price) in enumerate(sorted_bids):
        if cumulative >= q_cap:
            break
        take = min(qty, q_cap - cumulative)
        alloc_per_bid[i] = take
        cumulative += take
        clearing_price = price
        marginal_idx = i

    # --- Tie-break at marginal price (pro-rata among tied marginal bidders) ---
    if marginal_idx is not None:
        marginal_price = sorted_bids[marginal_idx, 2]
        tied_mask = sorted_bids[:, 2] == marginal_price
        tied_indices = np.where(tied_mask)[0]

        if len(tied_indices) > 1:
            # Re-allocate the marginal quantity pro-rata by bid quantity
            # First, remove existing allocations for tied bidders
            already_allocated_above = alloc_per_bid.copy()
            for ti in tied_indices:
                already_allocated_above[ti] = 0.0

            remaining = q_cap - already_allocated_above.sum()
            tied_quantities = sorted_bids[tied_indices, 1]
            total_tied_qty = tied_quantities.sum()

            if total_tied_qty > 0:
                for ti, tq in zip(tied_indices, tied_quantities):
                    alloc_per_bid[ti] = remaining * (tq / total_tied_qty)

    # --- Aggregate allocations back to agents ---
    for i, (agent_id, _, _) in enumerate(sorted_bids):
        allocations[int(agent_id)] += alloc_per_bid[i]

    # --- Compute payments (uniform price) ---
    payments = allocations * clearing_price

    # --- Auction statistics ---
    total_allocated = allocations.sum()
    cover_ratio = total_demand / q_cap if q_cap > 0 else 0.0

    if q_cap > 0 and total_allocated < 0.5 * q_cap:
        warnings.warn(
            f"[market_clearing_ets] Under-allocation: {total_allocated:.2f} Mt allocated "
            f"of {q_cap:.2f} Mt cap ({100*total_allocated/q_cap:.0f}%). "
            "This usually means most bids are below reserve_price. "
            "Check that auction.price_min >= ets.reserve_price in config.",
            UserWarning,
            stacklevel=2,
        )

    stats = {
        "clearing_price": clearing_price,
        "total_demand": total_demand,
        "total_allocated": total_allocated,
        "cover_ratio": cover_ratio,
        "auction_failed": False,
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
        Each row: [price_bid (€/t), quantity_bid (Mt)] for agent i.

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
