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
    - MSR TNAC bands preserve legislative proportions lower:mid:upper
        ~= 400:833:1096 when scaled to the micro-ETS cap.
  - MSR release amount: 6.4% of cap_year_0 per year (scaled from EU 100 Mt).
"""


import warnings

# Legislative TNAC thresholds from Decision (EU) 2015/1814 (as amended).
TNAC_LOWER_REF = 400.0
TNAC_MID_REF = 833.0
TNAC_UPPER_REF = 1096.0
TNAC_MID_OVER_UPPER = TNAC_MID_REF / TNAC_UPPER_REF


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
        # Threshold calibration: scaled from legislative bands so lower:mid:upper
        # keeps the 400:833:1096 proportions at micro-ETS scale.
        self.tnac_upper = float(msr["tnac_upper"])         # Mt
        self.tnac_mid = float(msr.get("tnac_mid", self.tnac_upper * TNAC_MID_OVER_UPPER))
        self.tnac_lower = float(msr["tnac_lower"])         # Mt
        self.withhold_rate = msr["withhold_rate"]         # fraction
        self.release_amount = msr["release_amount"]       # Mt/year (≈6.4% of cap)
        self.min_auction_frac = msr.get("min_auction_frac", 0.10)
        # activation_year retained for backward-compat reading but no longer
        # drives MSR gate; 1-year TNAC lag is enforced via _prev_tnac instead.
        self.msr_activation_year = msr.get("activation_year", msr.get("msr_activation_year", 1))
        # Maximum rollover multiplier: caps unsold-rollover so a single year's
        # auction volume cannot exceed cap_t × max_rollover_multiplier.
        self.max_rollover_multiplier = ets_cfg.get("max_rollover_multiplier", 1.5)

        self.reserve_price = ets_cfg.get("reserve_price", 0.0)

        # Price-responsive MSR triggers (break procyclical hoarding loop)
        self.price_containment_trigger = msr.get("price_containment_trigger", 0.70)
        self.price_release_trigger = msr.get("price_release_trigger", 0.85)
        self.emergency_release_amount = msr.get("emergency_release_amount", 0.50)

        # Absolute price thresholds (EUR/t) as alternative to ratio-based triggers
        # These provide more stable MSR behavior when penalty rates vary
        self.price_containment_absolute = msr.get("price_containment_absolute", 200.0)
        self.price_release_absolute = msr.get("price_release_absolute", 300.0)

        # Whether emergency MSR price-release is active (disable to suppress
        # procyclical reserve injection during low-price regimes)
        self.price_release_enabled = bool(msr.get("price_release_enabled", True))

        # Sanity-check: warn if absolute thresholds exceed the auction price ceiling
        price_max = float(config.get("auction", {}).get("price_max", 250.0))
        if self.price_containment_absolute >= price_max:
            warnings.warn(
                f"[CapSchedule] price_containment_absolute ({self.price_containment_absolute} EUR/t) "
                f">= price_max ({price_max} EUR/t); containment trigger is unreachable.",
                UserWarning, stacklevel=2,
            )
        if self.price_release_absolute >= price_max:
            warnings.warn(
                f"[CapSchedule] price_release_absolute ({self.price_release_absolute} EUR/t) "
                f">= price_max ({price_max} EUR/t); release trigger is unreachable.",
                UserWarning, stacklevel=2,
            )

        # Internal MSR reserve (starts empty)
        self._msr_reserve = 0.0

        # Separate accounting for unsold allowances absorbed into MSR
        # (distinct from normal TNAC-triggered withholding)
        self._unsold_absorbed = 0.0

        # Unsold volume pending rollover to next year's auction
        self._unsold_rollover_pending = 0.0

        # Last-call supply-flow telemetry (used by environment logging/debugging)
        self._last_unsold_rollover_in = 0.0
        self._last_msr_withheld = 0.0
        self._last_msr_released = 0.0

        # Total cancelled allowances (MSR cancellation mechanism)
        self._total_cancelled = 0.0

        # 1-year TNAC lag — store TNAC from end of previous year.
        # None = no prior year exists (year 0 with no burn-in → skip MSR).
        self._prev_tnac = None

        # Smoothed price trigger — previous year's price MA3.
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

    def _compute_tnac_withholding(self, tnac: float, auction_vol: float) -> float:
        """
        Compute TNAC-based MSR withholding before any price-trigger overrides.

        Regimes:
          - tnac > tnac_upper: withhold_rate * tnac (24% of TOTAL TNAC)
          - tnac_mid <= tnac <= tnac_upper: tnac - tnac_mid
          - tnac_lower <= tnac < tnac_mid: 0
          - tnac < tnac_lower: 0 (release branch handled by caller)
        """
        if tnac > self.tnac_upper:
            return min(self.withhold_rate * tnac, auction_vol)
        if self.tnac_mid <= tnac <= self.tnac_upper:
            return min(tnac - self.tnac_mid, auction_vol)
        return 0.0

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
        self._last_unsold_rollover_in = 0.0
        self._last_msr_withheld = 0.0
        self._last_msr_released = 0.0

        if self.msr_enabled:
            auction_vol = self._apply_msr(year, auction_vol,
                                          current_tnac=float(tnac),
                                          clearing_price=clearing_price,
                                          price_max=price_max,
                                          penalty_rate=penalty_rate,
                                          inflation_rate=inflation_rate,
                                          force_msr=force_msr,
                                          price_ma3=price_ma3)
            if self._last_msr_withheld > 0.0:
                assert auction_vol <= (cap_t - self._last_msr_withheld + 1e-9), (
                    "MSR withholding logged but not reflected in pre-floor auction volume."
                )

        # Safety floor: prevents micro-ETS strangulation where MSR zeros out
        # auctions (no direct EU ETS equivalent, but Auctioning Regulation
        # 2023/2830 guarantees member-state minimum volumes).
        min_vol = self.min_auction_frac * cap_t
        auction_vol = max(auction_vol, min_vol)

        # Add any unsold volume rolled over from the previous year.
        # Cap the rollover so the total auction volume cannot exceed
        # cap_t × max_rollover_multiplier (prevents runaway accumulation
        # of unsold rollovers across burn-in or low-demand years).
        rollover = self._unsold_rollover_pending
        self._last_unsold_rollover_in = float(rollover)
        self._unsold_rollover_pending = 0.0
        auction_vol += rollover
        auction_vol = min(auction_vol, cap_t * self.max_rollover_multiplier)

        # Store for logging
        self.cap_history.append(cap_t)
        self.volume_history.append(auction_vol)

        self._prev_tnac = float(tnac)
        # Update MA3 lag for smoothed price trigger
        if price_ma3 is not None:
            self._prev_ma3 = float(price_ma3)

        return max(auction_vol, 0.0)

    def msr_reserve(self) -> float:
        """Return current MSR reserve level (Mt)."""
        return self._msr_reserve

    def msr_event_counts(self) -> dict:
        """Return per-episode counts of price-triggered MSR interventions."""
        return dict(self._msr_event_counts)

    def preview_auction_volume(self, year: int, clearing_price: float = 0.0,
                               price_max: float = 120.0, penalty_rate: float = 0.0,
                               inflation_rate: float = 0.0,
                               price_ma3: float = None) -> float:
        """
        Read-only preview of this year's auction volume for Phase 1 observations.

        Mirrors _apply_msr logic using the stored _prev_tnac (1-year TNAC lag)
        without mutating any state.  Agents can therefore observe THIS year's
        MSR-adjusted supply *before* placing their bids.

        Parameters
        ----------
        year : int
            Current simulation year (0-indexed).
        clearing_price : float
            Last auction clearing price (EUR/t) — used for price-triggered rules.
        price_max : float
            Maximum auction price (EUR/t).
        penalty_rate : float
            Base non-compliance penalty rate at year 0.
        inflation_rate : float
            Annual inflation rate.
        price_ma3 : float, optional
            3-year moving average price (EUR/t) for smoothed-spike check.

        Returns
        -------
        float
            Estimated auction volume (Mt) after MSR adjustments.
        """
        cap_t = self.get_cap(year)
        if not self.msr_enabled or self._prev_tnac is None:
            return cap_t

        tnac = self._prev_tnac  # 1-year lag (already stored from last get_auction_volume)
        auction_vol = cap_t

        # Read-only snapshot: apply tnac_lower-based cancellation cap without mutation.
        # Mirrors the live _apply_msr cancellation logic.
        msr_snap = max(0.0, self._msr_reserve - max(0.0, self._msr_reserve - self.tnac_lower))

        # Inflation-adjusted penalty rate
        if penalty_rate > 0 and inflation_rate >= 0:
            eff_penalty = penalty_rate * ((1.0 + inflation_rate) ** year)
        else:
            eff_penalty = 138.75
        containment_threshold = eff_penalty * 1.8
        release_threshold = min(eff_penalty * 2.5, 450.0)

        # Emergency release check (same logic as _apply_msr, read-only)
        # Skipped entirely when price_release_enabled=False.
        if self.price_release_enabled and clearing_price >= release_threshold:
            # Keep preview parity with live _apply_msr: no special year<2 bypass.
            no_prior_ma3 = self._prev_ma3 is None
            smoothed_spike = (
                no_prior_ma3 or
                (price_ma3 is not None and self._prev_ma3 is not None
                 and price_ma3 > 2.5 * self._prev_ma3)
            )
            if smoothed_spike:
                auction_vol += min(self.emergency_release_amount, msr_snap)
                return max(auction_vol, self.min_auction_frac * cap_t, 0.0)

        # Containment: suppress withdrawal at high prices
        if clearing_price >= containment_threshold:
            if tnac < self.tnac_lower:
                auction_vol += min(self.release_amount, msr_snap)
            return max(auction_vol, self.min_auction_frac * cap_t, 0.0)

        # Normal TNAC-based rules
        withheld = self._compute_tnac_withholding(tnac, auction_vol)
        if withheld > 0.0:
            auction_vol -= withheld
        elif tnac < self.tnac_lower:
            auction_vol += min(self.release_amount, msr_snap)

        return max(auction_vol, self.min_auction_frac * cap_t, 0.0)

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
                   price_ma3 > 2.5 × _prev_ma3, or no _prev_ma3
                   (year < 2 bypass removed; _prev_ma3 is seeded during
                   burn-in so the guard is active from year 0)
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
        # MSR skipped if no prior TNAC exists AND force_msr not set (1-year lag gate).
        if not force_msr and self._prev_tnac is None:
            self._last_msr_withheld = 0.0
            self._last_msr_released = 0.0
            return auction_vol

        # Use lagged TNAC for normal MSR decisions (end of previous year).
        # During burn-in (force_msr=True), use the passed current_tnac directly
        # since the burn-in loop builds historical state without a true prior year.
        tnac = current_tnac if force_msr else self._prev_tnac

        # MSR cancellation: cancel holdings exceeding tnac_lower.
        # Anchoring to tnac_lower (not previous auction volume) prevents the
        # reserve from being inflated by rollover-distorted volumes and matches
        # the legislative intent of keeping MSR holdings below the lower TNAC band.
        excess = max(0.0, self._msr_reserve - self.tnac_lower)
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

        # Combined emergency-release trigger.
        # Requires BOTH absolute threshold AND smoothed price spike (or no history).
        # Smoothed check prevents spurious firing on single-auction anomalies.
        # Disabled entirely when price_release_enabled=False (config flag).
        if self.price_release_enabled:
            absolute_trigger = clearing_price >= release_threshold
            if absolute_trigger:
                # Smoothed-spike check: price_ma3 > 2.5× prev_ma3, or no prior MA3.
                # year < 2 bypass removed — _prev_ma3 is seeded during burn-in so
                # the guard is active from year 0 of the real episode.
                no_prior_ma3 = self._prev_ma3 is None
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
                    self._last_msr_withheld = 0.0
                    self._last_msr_released = float(release)
                    return auction_vol

        # Containment: suppress withdrawal when prices are already elevated
        if clearing_price >= containment_threshold:
            # No withdrawal even if TNAC > upper; only release if TNAC < lower
            release = 0.0
            if tnac < self.tnac_lower:
                release = min(self.release_amount, self._msr_reserve)
                self._msr_reserve -= release
                auction_vol += release
                self._msr_event_counts["containment_release"] += 1
            else:
                self._msr_event_counts["withdrawal_suppressed"] += 1
            self._last_msr_withheld = 0.0
            self._last_msr_released = float(release)
            return auction_vol

        # Normal MSR logic (TNAC-based)
        withheld = self._compute_tnac_withholding(tnac, auction_vol)
        release = 0.0
        if withheld > 0.0:
            self._msr_reserve += withheld
            auction_vol -= withheld
        elif tnac < self.tnac_lower:
            release = min(self.release_amount, self._msr_reserve)
            self._msr_reserve -= release
            auction_vol += release

        self._last_msr_withheld = float(withheld)
        self._last_msr_released = float(release)

        return auction_vol

    def update_calibration(self, cap_year_0, tnac_upper, tnac_lower,
                           release_amount, emergency_release_amount,
                           tnac_mid=None):
        """Update runtime calibration values and reset MSR internal state."""
        self.cap_year_0 = float(cap_year_0)
        self.tnac_upper = float(tnac_upper)
        if tnac_mid is None:
            self.tnac_mid = float(self.tnac_upper * TNAC_MID_OVER_UPPER)
        else:
            self.tnac_mid = float(tnac_mid)
        self.tnac_lower = float(tnac_lower)
        self.release_amount = float(release_amount)
        self.emergency_release_amount = float(emergency_release_amount)

        # Reset MSR internals to avoid mixing reserves between calibrations.
        self._msr_reserve = 0.0
        self._unsold_absorbed = 0.0
        self._unsold_rollover_pending = 0.0
        self._last_unsold_rollover_in = 0.0
        self._last_msr_withheld = 0.0
        self._last_msr_released = 0.0
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
        self._last_unsold_rollover_in = 0.0
        self._last_msr_withheld = 0.0
        self._last_msr_released = 0.0
        self._total_cancelled = 0.0
        self._prev_tnac = None   # reset TNAC lag so year 0 starts with no prior TNAC
        self._prev_ma3 = None    # reset smoothed price history
        self._msr_event_counts = {
            "emergency_release": 0,
            "containment_release": 0,
            "withdrawal_suppressed": 0,
        }
        self.cap_history = []
        self.volume_history = []
