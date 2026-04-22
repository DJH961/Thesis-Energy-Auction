# TNAC / MSR Simulation Guide

## Ratio-Based Reference for a Scaled EU ETS Environment

This guide describes every dynamic of the Total Number of Allowances in Circulation (TNAC) and the Market Stability Reserve (MSR) in terms of ratios and percentages relative to your simulation's own cap. No absolute numbers are used — every threshold, rate, and trigger is expressed as a proportion of your starting cap (`CAP_0`) or your annual cap (`CAP_t`), so you can plug in whatever scale your simulation requires.

All rules are derived from the EU ETS Directive (Phase 4, post-Fit for 55 revision 2023), your environment rulebook, and the CMW EU ETS 101 guide.

---

## 1. The Declining Cap

The cap is the total number of allowances the system issues in a given year. It declines linearly.

### 1.1 Linear Reduction Factor (LRF)

The LRF is expressed as a **percentage of the 2013 base cap** (the Phase 3 starting cap). In the real EU ETS, this base cap was approximately 2,084 million EUAs. The annual reduction in absolute terms equals `LRF × BASE_CAP`.

For your simulation, define a `BASE_CAP` (e.g. your year-zero cap) and apply the LRF to it:

```
CAP_t = CAP_0 - (t × LRF × BASE_CAP)
```

Where `CAP_0` is your starting cap and `BASE_CAP` is your reference cap (they may be identical if your simulation starts at year zero of the scheme).

**Real-world LRF values (as % of 2013 base cap per year):**

| Period | LRF | Approximate annual decline as % of base cap |
|--------|-----|----------------------------------------------|
| Phase 3 (2013–2020) | 1.74% | ~1.74% of BASE_CAP per year |
| Phase 4a (2021–2023) | 2.2% | ~2.2% of BASE_CAP per year |
| Phase 4b (2024–2027) | 4.3% | ~4.3% of BASE_CAP per year |
| Phase 4c (2028–2030) | 4.4% | ~4.4% of BASE_CAP per year |

**For your simulation:** You should support at least one LRF switch (e.g. start at 2.2% and switch to 4.3% at a configurable year) to model the Fit for 55 tightening. The year of the switch is a parameter.

### 1.2 Cap-to-Emissions Ratio at Start

The starting cap must be calibrated so that initial total emissions are at or slightly below the cap — a mild surplus. In the real EU ETS at the start of Phase 4, verified emissions were roughly **85–95%** of the cap. Your simulation should start with:

```
TOTAL_INITIAL_EMISSIONS ≈ 0.90 × CAP_0   (i.e. ~10% initial surplus)
```

Starting in structural deficit from year zero is not allowed — it creates pathological scarcity with no real-world analog.

### 1.3 What Gets Auctioned

Since your simulation covers power generation only (100% auctioning, no free allocation), the annual auction volume equals the cap minus any MSR intake withholding, plus any MSR release:

```
AUCTION_VOLUME_t = CAP_t - MSR_INTAKE_t + MSR_RELEASE_t
```

---

## 2. TNAC — Total Number of Allowances in Circulation

### 2.1 Definition

The TNAC is the cumulative difference between all allowances that have entered the market and all allowances that have left the market (through surrender for compliance or cancellation). In your simulation:

```
TNAC_t = Σ(all allowances auctioned up to year t)
        + Σ(all allowances released from MSR up to year t)
        - Σ(all allowances surrendered for compliance up to year t)
        - Σ(all allowances placed into MSR up to year t)
```

Equivalently, TNAC equals the total banked allowances held by all agents collectively:

```
TNAC_t = Σ(agent_i.allowance_balance) for all agents i
```

This second formulation is simpler to compute at each time step in your simulation.

### 2.2 When TNAC Is Calculated

The TNAC is calculated **once per year**, after compliance for that year has been settled (agents have surrendered allowances). The resulting TNAC value determines the MSR action for the **following year's** auction cycle. There is always a one-year lag between the TNAC observation and the MSR response.

```
TNAC observed at end of year t → MSR action applies to auctions in year t+1
```

---

## 3. MSR — Market Stability Reserve

The MSR is a buffer account that holds allowances. It is not owned by any agent. It has three operations: intake, release, and cancellation.

### 3.1 Thresholds (Scaled to Your Cap)

The real EU ETS thresholds, expressed as ratios of the approximate Phase 4 starting cap (~1,572 million in 2021):

| Threshold | Real value | As % of 2021 cap | Recommended simulation ratio |
|-----------|-----------|-------------------|------------------------------|
| Upper threshold (triggers intake) | 833 million | ~53% of CAP | **~53% of your CAP_0** |
| Lower threshold (triggers release) | 400 million | ~25% of CAP | **~25% of your CAP_0** |

**Critical design note from your rulebook:** These thresholds must be scaled appropriately for your simulation's agent count and cap size. They must not be perpetually active (always triggering intake) or perpetually dormant (never triggering). Validate this during calibration by checking that the TNAC crosses both thresholds during a typical training run.

For your simulation, define:

```
UPPER_THRESHOLD = 0.53 × CAP_0
LOWER_THRESHOLD = 0.25 × CAP_0
```

If you find the MSR is always active or never active, adjust these ratios. The key structural property is that there is a **comfortable band** (between 25% and 53% of CAP_0) where the MSR is dormant and the market operates freely.

### 3.2 MSR Intake (Removing Allowances from Auctions)

**When:** `TNAC_t > UPPER_THRESHOLD`

**How much:** A percentage of the TNAC is withheld from the following year's auction and placed into the MSR.

| Period | Intake rate |
|--------|-------------|
| Original MSR (2019–2023) | 12% of excess TNAC above upper threshold |
| Strengthened MSR (2024 onwards) | 24% of excess TNAC above upper threshold |

**For your simulation, use 24%** (the current and recommended rate):

```
if TNAC_t > UPPER_THRESHOLD:
    MSR_INTAKE_{t+1} = 0.24 × (TNAC_t - UPPER_THRESHOLD)
```

**Mechanism:** The intake does NOT recall allowances from agents. Instead, it reduces the number of allowances available at the next auction. The surplus stays in private hands — but future supply shrinks.

```
AUCTION_VOLUME_{t+1} = CAP_{t+1} - MSR_INTAKE_{t+1}
```

If `MSR_INTAKE_{t+1}` exceeds `CAP_{t+1}`, clamp the auction volume to zero (no negative auctions). Any excess intake that cannot be fulfilled is simply not placed.

**Unsold allowances:** If any allowances go unsold at auction (e.g. all bids are below the reserve price), those unsold allowances must be routed to the MSR or explicitly discarded. They must not vanish silently.

### 3.3 MSR Release (Injecting Allowances Back to Auctions)

**When:** `TNAC_t < LOWER_THRESHOLD`

**How much:** A fixed number of allowances is released from the MSR and added to the next year's auction. In the real EU ETS this is capped at 100 million, which is approximately:

```
RELEASE_AMOUNT = ~6.4% of CAP_0
```

For your simulation:

```
if TNAC_t < LOWER_THRESHOLD:
    MSR_RELEASE_{t+1} = min(0.064 × CAP_0, MSR_HOLDINGS_t)
```

The release cannot exceed what the MSR actually holds. If the MSR is empty, nothing is released.

```
AUCTION_VOLUME_{t+1} = CAP_{t+1} + MSR_RELEASE_{t+1}
```

### 3.4 The Dead Band (No MSR Action)

**When:** `LOWER_THRESHOLD ≤ TNAC_t ≤ UPPER_THRESHOLD`

No intake, no release. The auction volume is simply the cap:

```
AUCTION_VOLUME_{t+1} = CAP_{t+1}
```

This band represents a "healthy" market where the surplus is considered manageable. The market should spend a meaningful portion of the simulation in this band after initial oversupply is absorbed.

### 3.5 Price-Based Emergency Release (Optional)

The EU ETS Directive contains an additional release trigger: if the TNAC is not below the lower threshold, but the allowance price has been more than **3× the two-year rolling average price** for six consecutive months, 100 million allowances are also released.

For your simulation (adapted to annual time steps):

```
if TNAC_t ≥ LOWER_THRESHOLD:
    avg_price_2yr = mean(clearing_price_{t-1}, clearing_price_{t-2})
    if clearing_price_t > 3.0 × avg_price_2yr:
        MSR_RELEASE_{t+1} = min(0.064 × CAP_0, MSR_HOLDINGS_t)
```

This is a safety valve for price spikes. Implementation is optional but recommended for realism.

---

## 4. MSR Cancellation (Permanent Removal)

This is the most impactful feature of the strengthened MSR and the mechanism that permanently removes allowances from the system.

### 4.1 The Rule (Post-2023)

At the start of each year, any allowances held in the MSR **above the previous year's auction volume** are permanently cancelled (invalidated, deleted, gone forever).

```
CANCELLATION_t = max(0, MSR_HOLDINGS_t - AUCTION_VOLUME_{t-1})
MSR_HOLDINGS_t = MSR_HOLDINGS_t - CANCELLATION_t
```

**In ratio terms:** Since auction volumes decline with the cap, the cancellation threshold itself declines over time. This means the MSR's capacity to hold allowances shrinks each year, and any large MSR holdings get progressively wiped out.

### 4.2 Real-World Example (For Intuition)

At end of 2022, the MSR held approximately **190% of the annual cap** (3 billion vs. ~1.57 billion cap). After cancellation on 1 January 2023, it was cut to approximately **31% of the 2022 cap** (486 million, equal to the 2022 auction volume). Roughly **83%** of MSR holdings were cancelled in one step.

### 4.3 Timing Within the Annual Cycle

The order of operations within each simulation year matters. Here is the correct sequence:

```
START OF YEAR t:
  1. MSR CANCELLATION: Cancel MSR holdings above last year's auction volume
     CANCELLATION = max(0, MSR_HOLDINGS - AUCTION_VOLUME_{t-1})
     MSR_HOLDINGS -= CANCELLATION

  2. DETERMINE AUCTION VOLUME based on last year's TNAC:
     if TNAC_{t-1} > UPPER_THRESHOLD:
         INTAKE = 0.24 × (TNAC_{t-1} - UPPER_THRESHOLD)
         AUCTION_VOLUME_t = CAP_t - INTAKE
         (INTAKE goes to MSR after auction)
     elif TNAC_{t-1} < LOWER_THRESHOLD:
         RELEASE = min(0.064 × CAP_0, MSR_HOLDINGS)
         AUCTION_VOLUME_t = CAP_t + RELEASE
         MSR_HOLDINGS -= RELEASE
     else:
         AUCTION_VOLUME_t = CAP_t

  3. RUN AUCTION(S): Agents bid, uniform-price clearing, winners get allowances

  4. SECONDARY MARKET TRADING: Agents buy/sell among themselves

  5. EMISSIONS OCCUR: Each agent emits based on their energy mix

  6. COMPLIANCE: Each agent surrenders allowances equal to emissions
     Shortfall → penalty of €100/tCO₂ (inflation-adjusted) + carry-forward

  7. COMPUTE TNAC_t: Sum of all allowances held by all agents

  8. INVESTMENT DECISIONS: Agents may invest in new capacity (construction delays apply)

END OF YEAR t → proceed to year t+1
```

Note: MSR intake is conceptually "withheld from auction" — the allowances that are never auctioned go directly into the MSR. In your implementation, the simplest approach is to reduce the auction supply and add the difference to MSR holdings after the auction step.

---

## 5. Summary of All Ratios and Parameters

| Parameter | Symbol | Value / Formula |
|-----------|--------|-----------------|
| Starting cap | `CAP_0` | Your choice (simulation scale) |
| Base cap for LRF | `BASE_CAP` | = `CAP_0` (if starting at Phase 4 start) |
| LRF Phase 4a | `LRF_1` | 2.2% of `BASE_CAP` per year |
| LRF Phase 4b (Fit for 55) | `LRF_2` | 4.3% of `BASE_CAP` per year |
| LRF Phase 4c | `LRF_3` | 4.4% of `BASE_CAP` per year |
| Upper TNAC threshold | `T_upper` | 53% of `CAP_0` |
| Lower TNAC threshold | `T_lower` | 25% of `CAP_0` |
| MSR intake rate | `r_intake` | 24% of excess TNAC above `T_upper` |
| MSR release amount | `Q_release` | 6.4% of `CAP_0` (fixed quantity) |
| MSR cancellation threshold | — | max(previous year's auction volume, previous year's cap) |
| Initial emissions-to-cap ratio | — | ~90% (mild 10% surplus) |
| Price spike trigger | — | Current price > 3× two-year average |
| Non-compliance penalty | — | €100/tCO₂ (adjustable for inflation) |
| Penalty carry-forward | — | 100% of shortfall added to next year |

---

## 6. State Variables to Track

At each time step, your simulation needs to maintain:

```python
# System-level state
cap_t: float              # This year's cap (declining via LRF)
auction_volume_t: float   # Allowances offered at auction this year
msr_holdings: float       # Allowances currently in the MSR
tnac_t: float             # Computed after compliance each year
clearing_price_t: float   # Auction clearing price this year
total_cancelled: float    # Cumulative cancellations (for logging)

# Per-agent state
agent.allowance_balance: float    # Banked allowances held
agent.carry_forward: float        # Penalty shortfall from prior years
agent.verified_emissions: float   # This year's emissions
```

The TNAC is a derived quantity:
```python
tnac_t = sum(agent.allowance_balance for agent in agents)
```

---

## 7. Edge Cases and Gotchas

### 7.1 MSR Intake Exceeds Cap

If 24% of the excess TNAC above the upper threshold is larger than the annual cap (possible when the surplus is extreme relative to a shrinking cap), clamp the auction to zero. You cannot auction negative allowances.

```python
intake = 0.24 * (tnac - upper_threshold)
auction_volume = max(0, cap_t - intake)
actual_intake = cap_t - auction_volume  # Only what was actually withheld
msr_holdings += actual_intake
```

### 7.2 MSR Is Empty During Release

If the TNAC drops below the lower threshold but the MSR has already been depleted (by prior cancellations), nothing can be released. The market stays tight. This is a realistic scenario in the late years when the MSR has been mostly cancelled down.

```python
release = min(0.064 * cap_0, msr_holdings)  # Can be zero
```

### 7.3 Cancellation Wipes Out the MSR

After a large cancellation event, the MSR may hold very few allowances. This is intended — the cancellation mechanism is designed to permanently destroy surplus. If the MSR holds exactly last year's auction volume (the cancellation floor), it retains all of them. As auction volumes themselves decline with the cap, this floor also shrinks year over year, meaning the MSR gradually empties unless new intake occurs.

### 7.4 TNAC Can Be Negative (Conceptually)

If agents have collectively surrendered more than was ever issued (possible with carry-forward penalties creating future obligations), the TNAC can in theory go negative. In practice this signals extreme scarcity. The MSR should not act on a negative TNAC (neither intake nor release). Treat negative TNAC as zero for MSR decision purposes.

### 7.5 Interaction With Unsold Allowances

If the auction has a reserve price and some allowances go unsold, those unsold allowances must be handled explicitly. Your rulebook requires they either go to the MSR or are discarded:

```python
unsold = auction_volume - total_sold
msr_holdings += unsold  # Route unsold to MSR (recommended)
```

---

## 8. Calibration Checklist

Before running full training, verify:

- [ ] The TNAC crosses the upper threshold at least once during a simulation run (MSR intake fires)
- [ ] The TNAC crosses the lower threshold at least once, OR the simulation reaches a scarcity regime (MSR release fires or market is tight)
- [ ] MSR cancellation visibly reduces MSR holdings in at least one year
- [ ] The cap reaches zero or near-zero at a plausible year (with LRF 4.3%, the real EU ETS cap reaches zero around 2039)
- [ ] Initial surplus is mild (~10% of cap), not extreme
- [ ] Auction volumes decline over time and eventually become very small
- [ ] Agents can survive the transition period (penalties are bounded, carry-forward does not create death spirals under normal play)

---

## 9. Pseudocode — Complete Annual MSR Cycle

```python
def run_msr_cycle(year, cap_0, base_cap, lrf, tnac_prev, msr_holdings,
                  auction_volume_prev, clearing_prices, agents):
    """
    Complete MSR logic for one simulation year.
    Call this at the START of each year, before the auction.
    """

    # --- Step 0: Compute this year's cap ---
    cap_t = cap_0 - (year * lrf * base_cap)
    cap_t = max(0, cap_t)

    # --- Step 1: MSR Cancellation ---
    if year > 0:
        # The cancellation floor is the larger of (a) last year's actual auction
        # volume and (b) last year's cap. This prevents distorted auction volumes
        # (e.g. from large unsold rollovers) from causing premature cancellation.
        cap_prev = cap_0 - ((year - 1) * lrf * base_cap)
        cancellation_floor = max(auction_volume_prev, cap_prev)
        cancellation = max(0, msr_holdings - cancellation_floor)
        msr_holdings -= cancellation
    else:
        cancellation = 0

    # --- Step 2: Determine auction volume based on LAST year's TNAC ---
    upper_threshold = 0.53 * cap_0
    lower_threshold = 0.25 * cap_0
    release_amount  = 0.064 * cap_0

    intake = 0.0
    release = 0.0

    if tnac_prev > upper_threshold:
        # INTAKE: Withhold 24% of the EXCESS above the upper threshold from auction
        intake = 0.24 * (tnac_prev - upper_threshold)
        auction_volume = max(0, cap_t - intake)
        actual_intake = cap_t - auction_volume
        msr_holdings += actual_intake

    elif tnac_prev < lower_threshold and tnac_prev >= 0:
        # RELEASE: Inject allowances from MSR
        release = min(release_amount, msr_holdings)
        auction_volume = cap_t + release
        msr_holdings -= release

    else:
        # DEAD BAND or price-spike check
        auction_volume = cap_t

        # Optional: price-based emergency release
        if len(clearing_prices) >= 3:
            avg_2yr = (clearing_prices[-2] + clearing_prices[-3]) / 2
            if clearing_prices[-1] > 3.0 * avg_2yr and avg_2yr > 0:
                emergency_release = min(release_amount, msr_holdings)
                auction_volume += emergency_release
                msr_holdings -= emergency_release

    return cap_t, auction_volume, msr_holdings, cancellation
```

---

## 10. How the Pieces Fit Together

```
Year t-1 ends:
  TNAC_{t-1} computed (sum of all agent holdings)
       │
       ▼
Year t begins:
  MSR Cancellation ──► MSR shrinks (holdings > max(last auction vol, last cap) are destroyed)
       │
       ▼
  TNAC_{t-1} evaluated against thresholds:
       │
       ├─ TNAC > 53% of CAP_0  ──► INTAKE: 24% of (TNAC - upper_threshold) withheld from auction
       │                            Fewer allowances auctioned → MSR grows
       │
       ├─ TNAC < 25% of CAP_0  ──► RELEASE: ~6.4% of CAP_0 added to auction
       │                            More allowances auctioned → MSR shrinks
       │
       └─ 25% ≤ TNAC ≤ 53%     ──► NO ACTION: Auction = Cap
       │
       ▼
  Auction runs ──► Agents acquire allowances
       │
       ▼
  Secondary market ──► Agents trade among themselves
       │
       ▼
  Emissions occur ──► Based on energy mix
       │
       ▼
  Compliance ──► Surrender allowances = emissions
       │         Shortfall → penalty + carry-forward
       ▼
  TNAC_t computed ──► feeds into Year t+1 MSR decisions
```

---

## Sources

All ratios and rules in this document are derived from:

- EU ETS Directive (2003/87/EC, as amended through Fit for 55, 2023)
- ETS Handbook (European Commission)
- CMW EU ETS 101 Guide (Carbon Market Watch, 2024)
- Environment Rulebook (Master's Thesis, v1.0, March 2026)
