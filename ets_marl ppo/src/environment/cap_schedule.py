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

        self.reserve_price = ets_cfg.get("reserve_price", 0.0)

        # Internal MSR reserve (starts empty)
        self._msr_reserve = 0.0

        # Separate accounting for unsold allowances absorbed into MSR
        # (distinct from normal TNAC-triggered withholding)
        self._unsold_absorbed = 0.0

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

    def get_auction_volume(self, year: int, tnac: float) -> float:
        """
        Return the actual volume put to auction after MSR adjustments.

        Parameters
        ----------
        year : int
            Current simulation year (0-indexed).
        tnac : float
            Total Number of Allowances in Circulation (Mt).

        Returns
        -------
        auction_volume : float
            Allowances available at auction this year (Mt).
        """
        cap_t = self.get_cap(year)
        auction_vol = cap_t  # baseline: 100% auctioning

        if self.msr_enabled:
            auction_vol = self._apply_msr(auction_vol, tnac)

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

    # ------------------------------------------------------------------
    # Internal MSR logic
    # ------------------------------------------------------------------

    def _apply_msr(self, auction_vol: float, tnac: float) -> float:
        """
        Apply MSR rules to the auction volume.

        Rules (scaled from EU ETS):
          - If TNAC > upper threshold: withhold 24% of excess into MSR reserve.
          - If TNAC < lower threshold: release fixed amount from reserve into auction.
          - Otherwise: no adjustment.
        """
        if tnac > self.tnac_upper:
            # Withhold excess into reserve
            excess = tnac - self.tnac_upper
            withheld = self.withhold_rate * excess
            withheld = min(withheld, auction_vol)   # can't withhold more than available
            self._msr_reserve += withheld
            auction_vol -= withheld

        elif tnac < self.tnac_lower:
            # FIX: release from reserve into auction
            # Prima era: release = min(release_amount, reserve) ma non veniva
            # aggiunto correttamente ad auction_vol
            release = min(self.release_amount, self._msr_reserve)
            self._msr_reserve -= release
            auction_vol += release  # aggiunge effettivamente al volume

        # else: TNAC nella banda — nessun aggiustamento

        return auction_vol

    def reset(self):
        """Reset schedule to initial state (call at episode start)."""
        self._msr_reserve = 0.0
        self._unsold_absorbed = 0.0
        self.cap_history = []
        self.volume_history = []
