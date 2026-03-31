"""
cap_schedule.py
===============
Implements the EU ETS cap trajectory:
  - Linear Reduction Factor (LRF): annual percentage decrease of the cap.
  - Market Stability Reserve (MSR): adjusts the volume put to auction
    based on the Total Number of Allowances in Circulation (TNAC).

EU ETS references:
  - LRF 4.3% (2024-2027), 4.4% (2028+): EU ETS Directive post-2023 reform.
  - MSR thresholds 833M / 400M EU-scale, scaled here to micro-ETS.
"""


class CapSchedule:
    """
    Manages the annual cap and auction volume for the micro-ETS.

    Parameters
    ----------
    config : dict
        Full YAML config dict.
    """

    def __init__(self, config: dict):
        ets_cfg = config["ets"]

        self.cap_year_0 = ets_cfg["cap_year_0"]           # Mt
        self.lrf_phase1 = ets_cfg["lrf_phase1"]           # e.g. 0.043
        self.lrf_phase2 = ets_cfg["lrf_phase2"]           # e.g. 0.044
        self.lrf_switch = ets_cfg["lrf_phase_switch"]     # e.g. year 5

        msr = ets_cfg["msr"]
        self.msr_enabled = msr["enabled"]
        self.tnac_upper = msr["tnac_upper"]               # Mt
        self.tnac_lower = msr["tnac_lower"]               # Mt
        self.withhold_rate = msr["withhold_rate"]         # fraction
        self.release_amount = msr["release_amount"]       # Mt/year
        self.min_auction_frac = msr.get("min_auction_frac", 0.10)

        self.reserve_price = ets_cfg.get("reserve_price", 0.0)

        # P9: Price-responsive MSR triggers (break procyclical hoarding loop)
        self.price_containment_trigger = msr.get("price_containment_trigger", 0.70)
        self.price_release_trigger = msr.get("price_release_trigger", 0.85)
        self.emergency_release_amount = msr.get("emergency_release_amount", 0.50)

        # Internal MSR reserve (starts empty)
        self._msr_reserve = 0.0

        # Separate accounting for unsold allowances absorbed into MSR
        # (distinct from normal TNAC-triggered withholding)
        self._unsold_absorbed = 0.0

        # Unsold volume pending rollover to next year's auction
        self._unsold_rollover_pending = 0.0

        # Total cancelled allowances (MSR cancellation mechanism)
        self._total_cancelled = 0.0

        # History for logging
        self.cap_history = []
        self.volume_history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cap(self, year: int) -> float:
        """
        Return the total cap for a given year (LRF applied, no MSR).
        Year 0 = initial year (no reduction yet).
        """
        cap = self.cap_year_0
        for t in range(1, year + 1):
            lrf = self.lrf_phase1 if t < self.lrf_switch else self.lrf_phase2
            cap = cap * (1.0 - lrf)
        return cap

    def get_auction_volume(self, year: int, tnac: float,
                          clearing_price: float = 0.0,
                          price_max: float = 120.0) -> float:
        """
        Return the actual volume put to auction after MSR adjustments.

        Parameters
        ----------
        year : int
            Current simulation year (0-indexed).
        tnac : float
            Total Number of Allowances in Circulation (Mt).
        clearing_price : float
            Last auction clearing price (EUR/t). Used for price-responsive
            MSR triggers that prevent procyclical supply withdrawal.
        price_max : float
            Maximum auction price (EUR/t). Used to compute price_ratio.

        Returns
        -------
        auction_volume : float
            Allowances available at auction this year (Mt).
        """
        cap_t = self.get_cap(year)
        auction_vol = cap_t  # baseline: 100% auctioning

        if self.msr_enabled:
            auction_vol = self._apply_msr(auction_vol, tnac,
                                          clearing_price, price_max)

        # Safety floor: no EU ETS equivalent but the Auctioning Regulation
        # (2023/2830) guarantees member state minimum volumes.
        # Prevents micro-ETS strangulation where MSR zeros out auctions.
        min_vol = self.min_auction_frac * cap_t
        auction_vol = max(auction_vol, min_vol)

        # Add any unsold volume rolled over from the previous year
        rollover = self._unsold_rollover_pending
        self._unsold_rollover_pending = 0.0
        auction_vol += rollover

        # Store for logging
        self.cap_history.append(cap_t)
        self.volume_history.append(auction_vol)

        return max(auction_vol, 0.0)

    def msr_reserve(self) -> float:
        """Return current MSR reserve level (Mt)."""
        return self._msr_reserve

    def absorb_unsold(self, amount: float):
        """
        Absorb unsold auction allowances into the MSR reserve.

        This is tracked separately from normal TNAC-triggered withholding
        so the two sources can be distinguished in accounting.
        """
        amount = max(0.0, amount)
        self._msr_reserve += amount
        self._unsold_absorbed += amount

    def rollover_unsold(self, amount: float):
        """
        Roll over unsold auction volume to the next year's auction supply.

        Unlike absorb_unsold (which feeds into MSR), this adds the volume
        directly to the next call to get_auction_volume().
        """
        self._unsold_rollover_pending += max(0.0, amount)

    # ------------------------------------------------------------------
    # Internal MSR logic
    # ------------------------------------------------------------------

    def _apply_msr(self, auction_vol: float, tnac: float,
                   clearing_price: float = 0.0,
                   price_max: float = 120.0) -> float:
        """
        Apply MSR rules to the auction volume.

        Rules (scaled from EU ETS + P9 price-responsive triggers):
          1. If price >= release_trigger × price_max: emergency release from
             reserve (breaks procyclical loop where high prices + high TNAC
             cause further supply withdrawal).
          2. If price >= containment_trigger × price_max: suppress normal
             withdrawal even if TNAC > upper threshold.
          3. Normal TNAC-based rules otherwise.
          4. MSR cancellation: Per EU ETS Directive post-2023: MSR holdings above
             previous year's auction volume are permanently cancelled. In this
             micro-ETS, cancellation rarely triggers due to short episodes and
             moderate TNAC, but is included for regulatory completeness.
        """
        # MSR cancellation: cancel holdings exceeding previous year's auction volume
        # This implements the EU ETS post-2023 reform where excess MSR holdings
        # are permanently removed from the system.
        prev_auction_vol = self.volume_history[-1] if self.volume_history else auction_vol
        excess = max(0, self._msr_reserve - prev_auction_vol)
        self._msr_reserve -= excess
        self._total_cancelled += excess

        price_ratio = clearing_price / max(price_max, 1.0)

        # P9: Emergency release when prices approach ceiling
        if price_ratio >= self.price_release_trigger:
            release = min(self.emergency_release_amount, self._msr_reserve)
            self._msr_reserve -= release
            auction_vol += release
            return auction_vol

        # P9: Suppress withdrawal when prices are already elevated
        if price_ratio >= self.price_containment_trigger:
            # No withdrawal even if TNAC > upper; only release if TNAC < lower
            if tnac < self.tnac_lower:
                release = min(self.release_amount, self._msr_reserve)
                self._msr_reserve -= release
                auction_vol += release
            return auction_vol

        # Normal MSR logic
        if tnac > self.tnac_upper:
            withheld = self.withhold_rate * tnac
            withheld = min(withheld, auction_vol)
            self._msr_reserve += withheld
            auction_vol -= withheld

        elif tnac < self.tnac_lower:
            release = min(self.release_amount, self._msr_reserve)
            self._msr_reserve -= release
            auction_vol += release

        return auction_vol

    def reset(self):
        """Reset schedule to initial state (call at episode start)."""
        self._msr_reserve = 0.0
        self._unsold_absorbed = 0.0
        self._unsold_rollover_pending = 0.0
        self._total_cancelled = 0.0
        self.cap_history = []
        self.volume_history = []
