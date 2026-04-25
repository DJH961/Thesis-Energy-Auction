"""
phantom_bidder.py
=================
PhantomBidder — financial intermediary demand in the primary ETS auction.

Represents the ~40% of EU ETS auction demand that comes from financial
participants (banks, hedge funds, proprietary traders) who acquire
allowances speculatively and do not surrender them for compliance.

Reference: European Commission Auction Report 2024; EEX/ICE data 2019-2023.

Mechanism:
    The phantom bids above the reserve price with a LogNormal-distributed
    price anchored to a fraction of the effective penalty rate and a Uniform-
    distributed quantity fraction of the total auction supply.  Because
    bids are ranked descending, the phantom is consumed first when it wins,
    shrinking the effective supply available to floor-bidding compliance
    agents.  This creates a stochastic supply squeeze that destroys the
    "floor is always safe" belief, without requiring any changes to the
    auction clearing mechanism.

    The phantom is stateless per year: it has no holdings, no compliance
    obligation, and does not appear in any reward computation.  Its
    allocated allowances represent supply genuinely consumed by the
    financial market and are therefore intentionally discarded.
"""

import numpy as np


class PhantomBidder:
    """
    Represents financial intermediary demand in the primary auction.
    Stateless per-year: no holdings, no compliance, no reward.

    Price drawn from LogNormal(log(anchor), sigma) each year.
    Quantity drawn from Uniform(qty_frac_lo, qty_frac_hi) × auction_supply.

    Rulebook compliance: participates in the same uniform-price sealed-bid
    format as all other agents.  Has no special information.  EU ETS
    Regulation 1031/2010 explicitly allows financial intermediaries to bid.
    """

    def __init__(self, config: dict, rng):
        cfg = config.get("phantom_bidder", {})
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.qty_frac_lo: float = float(cfg.get("qty_frac_lo", 0.05))
        self.qty_frac_hi: float = float(cfg.get("qty_frac_hi", 0.20))
        self.sigma: float = float(cfg.get("price_lognormal_sigma", 0.45))
        self.price_fundamental_frac: float = float(cfg.get("price_fundamental_frac", 0.60))
        self.min_above_reserve: float = float(cfg.get("price_min_above_reserve", 2.0))
        self.max_frac_penalty: float = float(cfg.get("price_max_frac_penalty", 0.65))
        self.min_below_reserve_buffer: float = float(cfg.get("price_min_below_reserve_buffer", 5.0))
        self.rng = rng

        # Logging attributes — updated by sample_bid(), read by ets_environment.py
        self.last_price: float = 0.0
        self.last_qty: float = 0.0
        self.last_active: bool = False  # True if bid >= reserve_price (bid accepted)

    def sample_bid(
        self,
        price_ma3: float,
        reserve_price: float,
        penalty_rate: float,
        auction_supply: float,
    ) -> tuple:
        """
        Sample a (bid_price, bid_qty) pair for this year's auction.

        The bid price may be below the reserve — the caller must handle
        rejection via the normal reserve-price filter in market_clearing_ets.
        The phantom's last_active flag is False in that case.

        Parameters
        ----------
        price_ma3 : float
            3-year moving average of auction clearing prices (EUR/t).
            Deprecated and ignored for anchor computation (kept only for
            backward-compatible call sites/logging; planned for removal in
            a future release.
        reserve_price : float
            Effective reserve price for this year (EUR/t).
        penalty_rate : float
            Effective non-compliance penalty rate (EUR/t).
        auction_supply : float
            Total allowances offered at auction this year (Mt).

        Returns
        -------
        bid_price : float
            Drawn bid price (EUR/t); may be below reserve.
        bid_qty : float
            Drawn bid quantity (Mt).
        """
        # Explicitly ignore MA3 in anchor construction (deprecated arg retained).
        _ = price_ma3

        # Anchor: max(fundamental_fraction * penalty_rate, reserve + min_above).
        # This decouples phantom pricing from MA3 so floor stickiness in the
        # learning population does not suppress phantom demand pressure.
        # Note: np.lognormal(mean=log(anchor), sigma) gives median=anchor.
        anchor = max(
            self.price_fundamental_frac * max(penalty_rate, 1.0),
            reserve_price + self.min_above_reserve,
        )
        price = float(self.rng.lognormal(mean=np.log(anchor), sigma=self.sigma))
        # Clip: allow draws up to min_below_reserve_buffer below reserve
        # (bid rejected by clearing) but cap at max_frac_penalty to prevent
        # unrealistic blow-ups.
        price = float(np.clip(
            price,
            reserve_price - self.min_below_reserve_buffer,
            self.max_frac_penalty * max(penalty_rate, 1.0),
        ))
        qty = (
            float(self.rng.uniform(self.qty_frac_lo, self.qty_frac_hi))
            * max(auction_supply, 0.0)
        )

        self.last_price = price
        self.last_qty = qty
        self.last_active = price >= reserve_price
        return price, qty
