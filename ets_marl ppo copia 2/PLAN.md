# Implementation Plan: HAPPO + Reward Simplification

## Goal
Replace IPPO with HAPPO (Heterogeneous-Agent PPO) and simplify the reward from 10+ terms to 3 terms. This addresses the catastrophic green-regression observed at ep 3000-7000 in training.

## Why These Two Changes Together
1. **HAPPO** prevents simultaneous policy shifts (agents update sequentially, not in parallel)
2. **Reward simplification** makes the green investment signal visible to the critic (from 0.25% to ~15% of total reward)

---

## Change 1: Reward Simplification

### File: `src/environment/ets_environment.py` — `_compute_rewards()`

**Current (10+ terms):**
```
R_i = -cost_norm - emissions_intensity - penalty_norm + shaping
  where shaping = green_delta + queue + lock_in + trading + coverage + low_bid
```

**New (3 terms):**
```
R_i = -cost_norm - penalty_norm + beta_green * green_delta
```

Specifically:
- **`cost_norm`** = `(auction_cost + secondary_cost + investment_cost + operational_cost + budget_penalty + mac_cost) / 1000.0`
  - **Remove** `holding_cost` from total_cost (banking is free in real ETS; overbidding already costly through auction spend)
  - **Keep** revenue subtraction (electricity margin)
- **`penalty_norm`** = `log1p(penalty_cost / 100.0) * non_compliance_mult` (unchanged)
- **`green_bonus`** = `beta_green * max(0, green_frac - prev_green_frac)` where `beta_green = 25.0`
  - No `fossil_scale` multiplier (removed diminishing returns — all agents rewarded equally for greening)
  - No `shaping_weight` gating (always active)

**Remove entirely:**
- `emissions_intensity` (redundant — greening reduces cost_norm directly via lower emissions)
- `coverage_score` (penalty already handles non-compliance)
- `lock_in_penalty` (noise; agents have positive green incentive instead)
- `queue_bonus` (too small to matter: +0.05/project)
- `trading_profit_bonus` (distracts from main signal)
- `low_bid_shortfall_penalty` (penalty already handles this)
- `holding_cost` from reward (keep as diagnostic only, not in total_cost)

**Implementation:**
- The method signature stays the same (backward compat)
- Controlled by config flag `reward.simplified: true` (default false for backward compat)
- When `simplified: true`, skip all shaping computation and use the 3-term formula
- When `simplified: false`, existing behavior unchanged (tests pass)

### File: `configs/default.yaml`

Add under `reward:` section:
```yaml
reward:
  simplified: true        # HAPPO: use 3-term reward (cost + penalty + green)
  green_beta: 25.0        # green transition bonus scale (replaces shaping_beta)
```

---

## Change 2: HAPPO Sequential Updates

### Core Idea
Instead of all agents updating simultaneously (IPPO) or with soft cycling (active agent + reduced LR for others), HAPPO updates agents **one at a time** in sequence. Each subsequent agent's advantage is reweighted by the importance ratios of previously-updated agents. This guarantees monotonic improvement of the joint objective.

### File: `scripts/train.py` — Episode rollout collection

**What changes in the rollout loop:**
During the per-year loop (lines 508-612), we already collect `auction_raws[i]`, `auction_logps[i]`, `secondary_raws[i]`, `secondary_logps[i]` for each agent. Currently these are stored only in each agent's individual buffer.

For HAPPO, we need a **shared trajectory store** so that when updating agent `i`, we can recompute agent `j`'s log probs (for j < i) under agent j's UPDATED policy.

**New: `HAPPOTrajectory` class** (add to `scripts/train.py`):
```python
class HAPPOTrajectory:
    """Stores per-year trajectory data for all agents in one episode."""
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.obs1 = [[] for _ in range(n_agents)]    # phase1 obs per agent
        self.obs2 = [[] for _ in range(n_agents)]    # phase2 obs per agent
        self.auc_raw = [[] for _ in range(n_agents)]  # auction raw actions
        self.sec_raw = [[] for _ in range(n_agents)]  # secondary raw actions
        self.old_auc_lp = [[] for _ in range(n_agents)]  # old auction log probs
        self.old_sec_lp = [[] for _ in range(n_agents)]  # old secondary log probs

    def push(self, agent_idx, obs1, obs2, auc_raw, sec_raw, auc_lp, sec_lp):
        self.obs1[agent_idx].append(obs1.copy())
        self.obs2[agent_idx].append(obs2.copy())
        self.auc_raw[agent_idx].append(auc_raw.copy())
        self.sec_raw[agent_idx].append(sec_raw.copy())
        self.old_auc_lp[agent_idx].append(auc_lp.copy())
        self.old_sec_lp[agent_idx].append(sec_lp.copy())

    def clear(self):
        for i in range(self.n_agents):
            self.obs1[i].clear(); self.obs2[i].clear()
            self.auc_raw[i].clear(); self.sec_raw[i].clear()
            self.old_auc_lp[i].clear(); self.old_sec_lp[i].clear()
```

During rollout, after each year-step, push data for ALL agents into `HAPPOTrajectory` (in addition to each agent's own buffer, which is still needed for the PPO update).

### File: `scripts/train.py` — PPO Update section (lines 614-650)

**Current:** Soft cycling loop that updates each agent independently.

**New HAPPO update logic:**

```
At end of episode:
  1. Pick agent ordering (rotate: [ep%N, (ep+1)%N, ..., (ep+N-1)%N])
  2. Initialize advantage_factor = ones(T) for all timesteps T

  For each agent_i in the ordering:
    a. Multiply agent_i's advantages by advantage_factor
    b. Call agent_i.update(advantage_factor=advantage_factor)
    c. After update, recompute agent_i's new log probs:
       - new_auc_lp = agent_i.auction_policy.evaluate(happo_traj.obs1[i], happo_traj.auc_raw[i])
       - new_sec_lp = agent_i.secondary_policy.evaluate(happo_traj.obs2[i], happo_traj.sec_raw[i])
    d. Compute importance ratio:
       - old_lp = old_auc_lp[i] + old_sec_lp[i]
       - new_lp = new_auc_lp + new_sec_lp
       - ratio = exp(new_lp - old_lp)
       - clipped_ratio = clamp(ratio, 1-clip_eps, 1+clip_eps)
    e. Update advantage_factor *= min(ratio, clipped_ratio)  (element-wise)
```

This ensures agent 2 sees agent 1's policy change reflected in the advantage, agent 3 sees agents 1+2, etc.

### File: `src/agents/ppo_agent.py` — `update()` method

**Add `advantage_factor` parameter:**

The current method computes advantages internally via GAE. The HAPPO modification:
- Add parameter: `advantage_factor: Optional[torch.Tensor] = None`
- After computing GAE advantages and normalizing them, multiply element-wise:
  ```python
  if advantage_factor is not None:
      adv_t = adv_t * advantage_factor.unsqueeze(1)
  ```
- This is applied BEFORE the PPO clipped surrogate computation
- Everything else in the update method stays identical

### File: `configs/default.yaml`

Replace `agent_cycling` with HAPPO config:
```yaml
happo:
  enabled: true              # use HAPPO sequential updates
  rotate_order: true         # rotate agent update order each episode

# Keep agent_cycling for backward compat (ignored when happo.enabled=true)
agent_cycling:
  enabled: false
```

---

## Change 3: Config Flag for Backward Compatibility

Both changes are gated by config flags:
- `reward.simplified: true/false` — switches between 3-term and 10-term reward
- `happo.enabled: true/false` — switches between HAPPO and IPPO/soft-cycling

When both are false, the system behaves exactly as before. All 66 existing tests should pass unchanged because they don't modify these config flags.

---

## Files Modified (Summary)

| File | What Changes |
|------|-------------|
| `src/environment/ets_environment.py` | `_compute_rewards()` — add simplified 3-term path |
| `src/agents/ppo_agent.py` | `update()` — add `advantage_factor` parameter |
| `scripts/train.py` | Add `HAPPOTrajectory` class; rewrite update loop for HAPPO |
| `configs/default.yaml` | Add `happo:` section, add `reward.simplified`, `reward.green_beta` |

## Files NOT Modified

| File | Why |
|------|-----|
| `src/agents/actor_critic.py` | Networks unchanged — HAPPO uses same architecture |
| `src/environment/cap_schedule.py` | Cap mechanics unchanged |
| `src/environment/company.py` | Company state/investment mechanics unchanged |
| `tests/*` | All existing tests pass (changes gated by config flags) |

---

## Test Strategy

1. **Existing tests**: Run `pytest tests/ -v` — all 66 must pass (backward compat)
2. **New test: HAPPO update ordering**: Verify that advantage_factor is computed correctly with known dummy data
3. **New test: Simplified reward**: Verify 3-term reward matches expected values for a known episode
4. **Smoke test**: Run 100-episode training with HAPPO + simplified reward, verify no NaN/crashes and that green fracs increase

---

## Expected Outcome

With these changes:
- The critic has a clean 3-term reward target → can learn that investing reduces future costs
- The green signal (+0.75/step at max investment, vs cost_norm ~0.3-0.5) is strong enough to drive behavior
- HAPPO's sequential updates prevent the simultaneous policy shift at the curriculum boundary that caused the ep 3000-7000 regression
- Agent ordering rotation ensures no agent is systematically disadvantaged
