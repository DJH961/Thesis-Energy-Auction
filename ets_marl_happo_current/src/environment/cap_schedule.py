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
        # Threshold calibration rationale: tnac_upper set to ~60% of annual
        # system emissions, consistent with EU ETS 833 Mt threshold relative
        # to ~1.4 Gt annual emissions (Decision 2015/1814).
        self.tnac_upper = msr["tnac_upper"]               # Mt
        self.tnac_lower = msr["tnac_lower"]               # Mt
        self.withhold_rate = msr["withhold_rate"]         # fraction
        self.release_amount = msr["release_amount"]       # Mt/year
        self.min_auction_frac = msr.get("min_auction_frac", 0.10)
        self.msr_activation_year = msr.get("activation_year", msr.get("msr_activation_year", 2))

        self.reserve_price = ets_cfg.get("reserve_price", 0.0)

        # P9: Price-responsive MSR triggers (break procyclical hoarding loop)
        self.price_containment_trigger = msr.get("price_containment_trigger", 0.70)
        self.price_release_trigger = msr.get("price_release_trigger", 0.85)
        self.emergency_release_amount = msr.get("emergency_release_amount", 0.50)

        # Absolute price thresholds (EUR/t) as alternative to ratio-based triggers
        # These provide more stable MSR behavior when penalty rates vary
        self.price_containment_absolute = msr.get("price_containment_absolute", 200.0)
        self.price_release_absolute = msr.get("price_release_absolute", 300.0)

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

        # Per-episode MSR trigger counters for compact training summaries.
        self._msr_event_counts = {
            "emergency_release": 0,
            "containment_release": 0,
            "withdrawal_suppressed": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cap(self, year: int) -> float:
        """
        Return the total cap for a given year (LRF applied, no MSR).
        Year 0 = initial year (no reduction yet).
        """
        if year < 0:
            # Burn-in support: extrapolate backward so caps are higher pre-year-0.
            return self.cap_year_0 / ((1.0 - self.lrf_phase1) ** abs(year))

        cap = self.cap_year_0
        for t in range(1, year + 1):
            lrf = self.lrf_phase1 if t < self.lrf_switch else self.lrf_phase2
            cap = cap * (1.0 - lrf)
        return cap

    def get_auction_volume(self, year: int, tnac: float,
                          clearing_price: float = 0.0,
                          price_max: float = 120.0,
                          penalty_rate: float = 0.0,
                          inflation_rate: float = 0.0,
                          force_msr: bool = False) -> float:
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
        penalty_rate : float
            Base non-compliance penalty rate (EUR/t) at year 0. This should
            be the base rate from config (not inflation-adjusted). The MSR
            logic computes the inflation-adjusted rate internally.
        inflation_rate : float
            Annual inflation rate (e.g., 0.020 for 2%). Used to compute
            the inflation-adjusted penalty rate.
        force_msr : bool
            If True, applies MSR logic regardless of activation_year.
            Used by hidden burn-in years.

        Returns
        -------
        auction_volume : float
            Allowances available at auction this year (Mt).
        """
        cap_t = self.get_cap(year)
        auction_vol = cap_t  # baseline: 100% auctioning

        if self.msr_enabled:
            auction_vol = self._apply_msr(year, auction_vol, tnac,
                                          clearing_price, price_max,
                                          penalty_rate, inflation_rate,
                                          force_msr=force_msr)

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

    def msr_event_counts(self) -> dict:
        """Return per-episode counts of price-triggered MSR interventions."""
        return dict(self._msr_event_counts)

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

    def _apply_msr(self, year: int, auction_vol: float, tnac: float,
                   clearing_price: float = 0.0,
                   price_max: float = 120.0,
                   penalty_rate: float = 0.0,
                   inflation_rate: float = 0.0,
                   force_msr: bool = False) -> float:
        """
        Apply MSR rules to the auction volume.

        Rules (scaled from EU ETS + P9 price-responsive triggers):
             0. Before activation_year, MSR is inactive (no cancellation,
                 withholding, release, or price-triggered intervention).
          1. If price >= release_threshold: emergency release from
             reserve (breaks procyclical loop where high prices + high TNAC
             cause further supply withdrawal).
          2. If price >= containment_threshold: suppress normal
             withdrawal even if TNAC > upper threshold.
          3. Normal TNAC-based rules otherwise.
          4. MSR cancellation: Per EU ETS Directive post-2023: MSR holdings above
             previous year's auction volume are permanently cancelled. In this
             micro-ETS, cancellation rarely triggers due to short episodes and
             moderate TNAC, but is included for regulatory completeness.

        Parameters
        ----------
        year : int
            Current simulation year (0-indexed).
        auction_vol : float
            Baseline auction volume before MSR adjustments (Mt).
        tnac : float
            Total Number of Allowances in Circulation (Mt).
        clearing_price : float
            Last auction clearing price (EUR/t).
        price_max : float
            Maximum auction price (EUR/t).
        penalty_rate : float
            Base penalty rate at year 0 (EUR/t). This is the base rate from
            config, not inflation-adjusted.
        inflation_rate : float
            Annual inflation rate (e.g., 0.020 for 2%).

        Returns
        -------
        float
            Adjusted auction volume after MSR interventions (Mt).
        """
        if not force_msr and year < self.msr_activation_year:
            return auction_vol  # MSR inactive during observation period

        # MSR cancellation: cancel holdings exceeding previous year's auction volume
        # This implements the EU ETS post-2023 reform where excess MSR holdings
        # are permanently removed from the system.
        prev_auction_vol = self.volume_history[-1] if self.volume_history else auction_vol
        excess = max(0, self._msr_reserve - prev_auction_vol)
        self._msr_reserve -= excess
        self._total_cancelled += excess

        # Compute inflation-adjusted penalty rate (effective penalty at this year)
        if penalty_rate > 0 and inflation_rate >= 0:
            eff_penalty = penalty_rate * ((1.0 + inflation_rate) ** year)
        else:
            eff_penalty = penalty_rate if penalty_rate > 0 else 138.75  # fallback base rate

        # Dynamic absolute thresholds based on inflation-adjusted penalty
        # Containment threshold: ~1.8× effective penalty (~250 EUR/t at year 0, rising with inflation)
        # Release threshold: ~2.5× effective penalty (~347 EUR/t at year 0), hard ceiling 450 EUR/t
        containment_threshold = eff_penalty * 1.8
        release_threshold = min(eff_penalty * 2.5, 450.0)

        # P9: Emergency release when prices approach ceiling
        if clearing_price >= release_threshold:
            release = min(self.emergency_release_amount, self._msr_reserve)
            self._msr_reserve -= release
            auction_vol += release
            self._msr_event_counts["emergency_release"] += 1
            return auction_vol

        # P9: Suppress withdrawal when prices are already elevated
        if clearing_price >= containment_threshold:
            # No withdrawal even if TNAC > upper; only release if TNAC < lower
            if tnac < self.tnac_lower:
                release = min(self.release_amount, self._msr_reserve)
                self._msr_reserve -= release
                auction_vol += release
                self._msr_event_counts["containment_release"] += 1
            else:
                self._msr_event_counts["withdrawal_suppressed"] += 1
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

    def update_calibration(self, cap_year_0, tnac_upper, tnac_lower,
                           release_amount, emergency_release_amount):
        """Update runtime calibration values and reset MSR internal state."""
        self.cap_year_0 = float(cap_year_0)
        self.tnac_upper = float(tnac_upper)
        self.tnac_lower = float(tnac_lower)
        self.release_amount = float(release_amount)
        self.emergency_release_amount = float(emergency_release_amount)

        # Reset MSR internals to avoid mixing reserves between calibrations.
        self._msr_reserve = 0.0
        self._unsold_absorbed = 0.0
        self._unsold_rollover_pending = 0.0
        self._total_cancelled = 0.0
        self._msr_event_counts = {
            "emergency_release": 0,
            "containment_release": 0,
            "withdrawal_suppressed": 0,
        }
        self.cap_history.clear()
        self.volume_history.clear()

    def reset(self):
        """Reset schedule to initial state (call at episode start)."""
        self._msr_reserve = 0.0
        self._unsold_absorbed = 0.0
        self._unsold_rollover_pending = 0.0
        self._total_cancelled = 0.0
        self._msr_event_counts = {
            "emergency_release": 0,
            "containment_release": 0,
            "withdrawal_suppressed": 0,
        }
        self.cap_history = []
        self.volume_history = []
