"""
cap_schedule.py
===============
Implements the EU ETS cap trajectory:
  - Linear Reduction Factor (LRF): true linear annual reduction of the cap
    (each year removes lrf_k × cap_year_0 from the running cap, matching
    EU ETS Directive post-2023 reform linear-decline mandate).
  - Market Stability Reserve (MSR): adjusts the volume put to auction
    based on the Total Number of Allowances in Circulation (TNAC) with a
    proper 1-year observation lag (Decision 2015/1814 Art 1(5)).

EU ETS references:
  - LRF 4.3% (2026–27), 4.4% (2028+): EU ETS Directive post-2023 reform.
  - MSR thresholds scaled to micro-ETS; upper≈36%, lower≈22% of cap_year_0.
  - MSR release amount: 6.4% of cap_year_0 per year (scaled from EU 100 Mt).
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
        self.lrf_switch = ets_cfg["lrf_phase_switch"]     # switch year (e.g. 2)

        msr = ets_cfg["msr"]
        self.msr_enabled = msr["enabled"]
        # Threshold calibration: upper≈36% of cap_year_0 (relaxed ratio for
        # micro-ETS; real EU ≈53%). Lower raised to ≈22% (closer to CMW
        # guide's 25%, balanced for 16-agent system).
        self.tnac_upper = msr["tnac_upper"]               # Mt
        self.tnac_lower = msr["tnac_lower"]               # Mt
        self.withhold_rate = msr["withhold_rate"]         # fraction
        self.release_amount = msr["release_amount"]       # Mt/year (≈6.4% of cap)
        self.min_auction_frac = msr.get("min_auction_frac", 0.10)
        # activation_year retained for backward-compat reading but no longer
        # drives MSR gate; 1-year TNAC lag is enforced via _prev_tnac instead.
        self.msr_activation_year = msr.get("activation_year", msr.get("msr_activation_year", 1))

        self.reserve_price = ets_cfg.get("reserve_price", 0.0)

        # Price-responsive MSR triggers (break procyclical hoarding loop)
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

        # A2: 1-year TNAC lag — store TNAC from end of previous year.
        # None = no prior year exists (year 0 with no burn-in → skip MSR).
        self._prev_tnac = None

        # A4: Smoothed price trigger — previous year's price MA3.
        # None = no prior MA3 history (triggers fall back to absolute-only).
        self._prev_ma3 = None

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
        Return the total cap for a given year (linear LRF applied, no MSR).

        EU ETS Directive post-2023 specifies a LINEAR annual reduction:
        each year removes `lrf_k × cap_year_0` from the running cap, so
        the cap declines in equal absolute steps (not compound/exponential).

        Year 0 = initial year (cap_year_0, no reduction applied yet).
        """
        if year < 0:
            # Burn-in support: extrapolate backward linearly so caps are
            # higher in pre-year-0 years (symmetric with forward direction).
            return self.cap_year_0 + abs(year) * self.lrf_phase1 * self.cap_year_0

        # Linear decline: subtract lrf_k × cap_year_0 per year
        reduction = 0.0
        for t in range(1, year + 1):
            lrf = self.lrf_phase1 if t < self.lrf_switch else self.lrf_phase2
            reduction += lrf * self.cap_year_0
        return max(0.0, self.cap_year_0 - reduction)

    def get_auction_volume(self, year: int, tnac: float,
                          clearing_price: float = 0.0,
                          price_max: float = 120.0,
                          penalty_rate: float = 0.0,
                          inflation_rate: float = 0.0,
                          force_msr: bool = False,
                          price_ma3: float = None) -> float:
        """
        Return the actual volume put to auction after MSR adjustments.

        Parameters
        ----------
        year : int
            Current simulation year (0-indexed).
        tnac : float
            Total Number of Allowances in Circulation at the START of this
            year (= end of previous year). Stored for next year's MSR lag.
        clearing_price : float
            Last auction clearing price (EUR/t). Used for price-responsive
            MSR triggers.
        price_max : float
            Maximum auction price (EUR/t). Used to compute price_ratio.
        penalty_rate : float
            Base non-compliance penalty rate (EUR/t) at year 0.
        inflation_rate : float
            Annual inflation rate (e.g., 0.020 for 2%).
        force_msr : bool
            If True, forces MSR even when _prev_tnac is None. Used by
            hidden burn-in years (which carry their own TNAC sequence).
        price_ma3 : float, optional
            3-year moving average of clearing price (EUR/t). Used in A4
            smoothed price trigger. None = no MA3 history yet.

        Returns
        -------
        auction_volume : float
            Allowances available at auction this year (Mt).
        """
        cap_t = self.get_cap(year)
        auction_vol = cap_t  # baseline: 100% auctioning

        if self.msr_enabled:
            auction_vol = self._apply_msr(year, auction_vol,
                                          current_tnac=float(tnac),
                                          clearing_price=clearing_price,
                                          price_max=price_max,
                                          penalty_rate=penalty_rate,
                                          inflation_rate=inflation_rate,
                                          force_msr=force_msr,
                                          price_ma3=price_ma3)

        # Safety floor: prevents micro-ETS strangulation where MSR zeros out
        # auctions (no direct EU ETS equivalent, but Auctioning Regulation
        # 2023/2830 guarantees member-state minimum volumes).
        min_vol = self.min_auction_frac * cap_t
        auction_vol = max(auction_vol, min_vol)

        # Add any unsold volume rolled over from the previous year
        rollover = self._unsold_rollover_pending
        self._unsold_rollover_pending = 0.0
        auction_vol += rollover

        # Store for logging
        self.cap_history.append(cap_t)
        self.volume_history.append(auction_vol)

        # A2: Update TNAC lag — store current TNAC as previous for next year
        self._prev_tnac = float(tnac)
        # A4: Update MA3 lag for smoothed price trigger
        if price_ma3 is not None:
            self._prev_ma3 = float(price_ma3)

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

    def _apply_msr(self, year: int, auction_vol: float,
                   current_tnac: float = 0.0,
                   clearing_price: float = 0.0,
                   price_max: float = 120.0,
                   penalty_rate: float = 0.0,
                   inflation_rate: float = 0.0,
                   force_msr: bool = False,
                   price_ma3: float = None) -> float:
        """
        Apply MSR rules to the auction volume.

        A2: 1-year TNAC lag — MSR uses self._prev_tnac (end of previous
        year) rather than current TNAC. If _prev_tnac is None (no prior
        year exists), MSR is skipped unless force_msr=True.

        Rules (EU ETS Decision 2015/1814 + A4 smoothed price trigger):
          1. If price meets combined emergency-release trigger: emergency
             release from reserve (breaks procyclical loop where high
             prices + high TNAC cause further supply withdrawal).
             Trigger requires BOTH:
               (a) absolute threshold: clearing_price >= release_threshold
               (b) smoothed spike OR no prior MA3 history:
                   price_ma3 > 2.5 × _prev_ma3, or year < 2 / no _prev_ma3
          2. If price >= containment_threshold: suppress normal withdrawal
             even if TNAC > upper threshold.
          3. Normal TNAC-based rules otherwise.
          4. MSR cancellation: MSR holdings above previous year's auction
             volume are permanently cancelled (EU ETS Directive post-2023).

        Parameters
        ----------
        year : int
            Current simulation year (0-indexed).
        auction_vol : float
            Baseline auction volume before MSR adjustments (Mt).
        current_tnac : float
            TNAC passed to get_auction_volume this call. Used when
            force_msr=True (burn-in) since no 1-year lag is available.
        clearing_price : float
            Last auction clearing price (EUR/t).
        price_max : float
            Maximum auction price (EUR/t).
        penalty_rate : float
            Base penalty rate at year 0 (EUR/t), not inflation-adjusted.
        inflation_rate : float
            Annual inflation rate (e.g., 0.020 for 2%).
        force_msr : bool
            If True, applies MSR even when _prev_tnac is None (burn-in
            years use current_tnac directly, no 1-year lag).
        price_ma3 : float, optional
            3-year moving average of clearing price (EUR/t).

        Returns
        -------
        float
            Adjusted auction volume after MSR interventions (Mt).
        """
        # A2: 1-year TNAC lag gate
        # MSR skipped if no prior TNAC exists AND force_msr not set.
        if not force_msr and self._prev_tnac is None:
            return auction_vol

        # Use lagged TNAC for normal MSR decisions (end of previous year).
        # During burn-in (force_msr=True), use the passed current_tnac directly
        # since the burn-in loop builds historical state without a true prior year.
        tnac = current_tnac if force_msr else self._prev_tnac

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
        # Containment threshold: ~1.8× effective penalty (~250 EUR/t at year 0)
        # Release threshold: ~2.5× effective penalty, hard ceiling 450 EUR/t
        containment_threshold = eff_penalty * 1.8
        release_threshold = min(eff_penalty * 2.5, 450.0)

        # A4: Combined emergency-release trigger.
        # Requires BOTH absolute threshold AND smoothed price spike (or no history).
        # Smoothed check prevents spurious firing on single-auction anomalies.
        absolute_trigger = clearing_price >= release_threshold
        if absolute_trigger:
            # Smoothed-spike check: price_ma3 > 2.5× prev_ma3, or no prior MA3
            no_prior_ma3 = (self._prev_ma3 is None or year < 2)
            smoothed_spike = (
                no_prior_ma3 or
                (price_ma3 is not None and self._prev_ma3 is not None
                 and price_ma3 > 2.5 * self._prev_ma3)
            )
            if smoothed_spike:
                release = min(self.emergency_release_amount, self._msr_reserve)
                self._msr_reserve -= release
                auction_vol += release
                self._msr_event_counts["emergency_release"] += 1
                return auction_vol

        # Containment: suppress withdrawal when prices are already elevated
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

        # Normal MSR logic (TNAC-based)
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
        self._prev_tnac = None
        self._prev_ma3 = None
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
        # A2: reset TNAC lag so year 0 starts with no prior TNAC
        self._prev_tnac = None
        # A4: reset smoothed price history
        self._prev_ma3 = None
        self._msr_event_counts = {
            "emergency_release": 0,
            "containment_release": 0,
            "withdrawal_suppressed": 0,
        }
        self.cap_history = []
        self.volume_history = []
