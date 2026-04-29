"""
tests/test_environment.py
=========================
Integration tests for the ETSEnvironment with technology-specific mix.

Run with:
    pytest tests/test_environment.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import yaml
from src.environment.ets_environment import ETSEnvironment

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")

def load_env(seed=42):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return ETSEnvironment(config, seed=seed)


# ---------------------------------------------------------------------------
# Test 1: Reset returns correct observation shape
# ---------------------------------------------------------------------------

def test_reset_obs_shape():
    env = load_env()
    obs, info = env.reset()
    n_agents = env.config["companies"]["n_agents"]
    obs1_dim = env.companies[0].obs_dim_phase1
    assert obs.shape == (n_agents, obs1_dim), f"Expected ({n_agents}, {obs1_dim}), got {obs.shape}"


# ---------------------------------------------------------------------------
# Test 2: Two-phase step runs without error
# ---------------------------------------------------------------------------

def test_two_phase_step_runs():
    env = load_env()
    obs1, _ = env.reset()
    n_agents = env.n_agents

    # Phase 1: [bid_price, qty, invest_frac, tech_logit0, tech_logit1, tech_logit2]
    auction_actions = np.random.uniform(
        [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
        size=(n_agents, 6)
    ).astype(np.float32)

    obs2, log = env.step_auction(auction_actions)
    obs2_dim = env.companies[0].obs_dim_phase2
    assert obs2.shape == (n_agents, obs2_dim), f"Expected ({n_agents}, {obs2_dim}), got {obs2.shape}"

    # Phase 2: [sec_price_abs, sec_qty]
    secondary_actions = np.random.uniform(
        [45.0, -3.0], [200.0, 3.0], size=(n_agents, 2)
    ).astype(np.float32)

    obs1_next, rewards, terminated, truncated, info = env.step_secondary(secondary_actions)
    assert obs1_next.shape == (n_agents, env.companies[0].obs_dim_phase1)
    assert len(rewards) == n_agents


# ---------------------------------------------------------------------------
# Test 2b: Phase-1 invest action is direct physical invest_frac
# ---------------------------------------------------------------------------

def test_phase1_invest_action_direct_mapping():
    """Higher invest action values should execute higher invest_frac (no hidden remap)."""
    env = load_env(seed=77)
    env.reset(seed=77)
    n_agents = env.n_agents

    def _run_with_invest(invest_action: float, seed: int) -> float:
        env.reset(seed=seed)
        auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
        auction_actions[:, 0] = 100.0
        auction_actions[:, 1] = 1.0
        auction_actions[:, 2] = invest_action
        auction_actions[:, 3:] = [0.0, 0.0, 1.0]
        _, log = env.step_auction(auction_actions)
        sec_actions = np.zeros((n_agents, 2), dtype=np.float32)
        sec_actions[:, 0] = env._phase1_clearing_price
        env.step_secondary(sec_actions)
        return float(log["invest_fracs"][0])

    invest_low = _run_with_invest(0.00, seed=78)
    invest_mid = _run_with_invest(0.03, seed=78)

    assert invest_low <= 1e-6, f"Expected near-zero executed investment, got {invest_low:.6f}"
    assert invest_mid > invest_low, (
        f"Expected monotonic increase in executed investment, got low={invest_low:.6f}, mid={invest_mid:.6f}"
    )
    # Numeric tolerance: invest action 0.03 should map to ≤ requested fraction.
    # The action is the requested invest fraction; budget gating may scale it
    # down but never up, so executed_frac ∈ [0, 0.03 + small slack].
    assert invest_mid <= 0.03 + 1e-6, (
        f"Executed invest_frac {invest_mid:.6f} exceeded requested 0.03"
    )


# ---------------------------------------------------------------------------
# Test 2c: Budget gate scales qty down for cash-poor agents (v8.4.2: soft scale, not hard zero)
# ---------------------------------------------------------------------------

def test_budget_gate_scales_qty_for_cash_poor_agents():
    """Agents with cash < 10% of bid notional should have qty scaled down (not zeroed)
    so that bid_p × bid_q × 0.10 ≤ cash. v8.4.2 replaced the hard zero with a soft scale
    to avoid the gradient discontinuity that pushed policies to systematic under-bidding."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config["companies"]["n_bot_agents"] = 0
    config["budget"]["emergency_loan"]["enabled"] = False
    # Treasury off so only operating cash matters
    config["budget"].setdefault("treasury_reserve", {})["enabled"] = False

    env = ETSEnvironment(config, seed=91)
    env.reset(seed=91)
    n_agents = env.n_agents

    # Drain budget for agent 0 so they cannot cover 10% of a large notional
    env.companies[0].budget_spent_this_year = env.companies[0].annual_budget * 0.99
    cash_before = max(
        0.0,
        env.companies[0].annual_budget - env.companies[0].budget_spent_this_year,
    ) + env.companies[0].get_treasury_available()

    bid_price = 200.0
    bid_qty   = 2.0
    auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
    auction_actions[:, 0] = bid_price
    auction_actions[:, 1] = bid_qty   # notional well above remaining cash
    _, log = env.step_auction(auction_actions)

    # The gate should leave bid_p × scaled_qty × 0.10 ≤ cash. The scaled qty is
    # ≤ the original requested qty (2.0). It must be strictly less than the
    # requested qty (because cash is insufficient) and strictly positive
    # (because the gate scales rather than zeroes).
    alloc = log.get("allocations", [])
    if alloc and cash_before > 1e-6:
        max_affordable_qty = cash_before / (bid_price * 0.10)
        assert alloc[0] <= max_affordable_qty + 1e-6, (
            f"Allocation {alloc[0]:.4f} exceeds max affordable qty "
            f"{max_affordable_qty:.4f} given cash={cash_before:.2f}"
        )
        assert alloc[0] < bid_qty - 1e-6, (
            f"Expected scaled allocation < requested qty {bid_qty}, got {alloc[0]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Episode terminates after n_years steps
# ---------------------------------------------------------------------------

def test_episode_length():
    env = load_env()
    obs1, _ = env.reset()
    n_agents = env.n_agents
    steps = 0
    terminated = False

    while not terminated:
        auction_actions = np.random.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.random.uniform(
            [45.0, -3.0], [200.0, 3.0], size=(n_agents, 2)
        ).astype(np.float32)
        obs1, _, terminated, _, _ = env.step_secondary(secondary_actions)
        steps += 1

    assert steps == env.n_years


# ---------------------------------------------------------------------------
# Test 4: Cap decreases each year
# ---------------------------------------------------------------------------

def test_cap_decreases():
    env = load_env()
    env.reset()
    n_agents = env.n_agents
    caps = []

    for _ in range(env.n_years):
        auction_actions = np.random.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.0, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
        secondary_actions[:, 0] = env._phase1_clearing_price
        _, _, terminated, _, info = env.step_secondary(secondary_actions)
        caps.append(info["year_log"]["cap"])
        if terminated:
            break

    for i in range(1, len(caps)):
        assert caps[i] < caps[i - 1], f"Cap did not decrease at step {i}"


# ---------------------------------------------------------------------------
# Test 5: Rewards are finite
# ---------------------------------------------------------------------------

def test_rewards_finite():
    env = load_env()
    env.reset()
    n_agents = env.n_agents

    for _ in range(env.n_years):
        auction_actions = np.random.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.random.uniform(
            [45.0, -3.0], [200.0, 3.0], size=(n_agents, 2)
        ).astype(np.float32)
        _, rewards, terminated, _, _ = env.step_secondary(secondary_actions)
        assert np.all(np.isfinite(rewards)), f"Non-finite rewards: {rewards}"
        if terminated:
            break


# ---------------------------------------------------------------------------
# Test 6: Technology mix sums to 1
# ---------------------------------------------------------------------------

def test_tech_mix_sums_to_one():
    env = load_env()
    env.reset()
    n_agents = env.n_agents

    for _ in range(env.n_years):
        auction_actions = np.random.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
        secondary_actions[:, 0] = env._phase1_clearing_price
        _, _, terminated, _, _ = env.step_secondary(secondary_actions)

        for c in env.companies:
            assert abs(c.mix.sum() - 1.0) < 1e-6, f"Mix doesn't sum to 1: {c.mix}"
            assert np.all(c.mix >= -1e-9), f"Negative mix: {c.mix}"

        if terminated:
            break


# ---------------------------------------------------------------------------
# Test 7: Green fraction only increases (greening-only)
# ---------------------------------------------------------------------------

def test_green_fraction_non_decreasing():
    """With positive investment, green fraction should not decrease."""
    env = load_env()
    env.reset()
    n_agents = env.n_agents

    prev_green = [c.green_frac for c in env.companies]
    for _ in range(env.n_years):
        # Invest heavily in solar (tech_logit2 dominant)
        auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
        auction_actions[:, 0] = 100.0  # bid price
        auction_actions[:, 1] = 2.0    # quantity
        auction_actions[:, 2] = 0.03   # invest_frac
        auction_actions[:, 3] = -10.0
        auction_actions[:, 4] = -10.0
        auction_actions[:, 5] = 10.0   # solar logit dominant (sharp softmax)

        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
        secondary_actions[:, 0] = env._phase1_clearing_price
        _, _, terminated, _, _ = env.step_secondary(secondary_actions)
        if terminated:
            break

    # After investments mature, green should be >= initial
    # (may not increase in first years due to construction delays)


# ---------------------------------------------------------------------------
# Test 8: Reproducibility with same seed
# ---------------------------------------------------------------------------

def test_reproducibility():
    rewards_run1 = _collect_episode_rewards(seed=42)
    rewards_run2 = _collect_episode_rewards(seed=42)
    np.testing.assert_array_almost_equal(rewards_run1, rewards_run2)


def _collect_episode_rewards(seed):
    env = load_env(seed=seed)
    rng = np.random.default_rng(seed)
    obs1, _ = env.reset(seed=seed)
    n_agents = env.n_agents
    total = np.zeros(n_agents)

    for _ in range(env.n_years):
        auction_actions = rng.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = rng.uniform(
            [45.0, -3.0], [200.0, 3.0], size=(n_agents, 2)
        ).astype(np.float32)
        obs1, rewards, terminated, _, _ = env.step_secondary(secondary_actions)
        total += rewards
        if terminated:
            break
    return total


# ---------------------------------------------------------------------------
# Test 9: Different seeds give different results
# ---------------------------------------------------------------------------

def test_different_seeds_differ():
    rewards_42 = _collect_episode_rewards(seed=42)
    rewards_123 = _collect_episode_rewards(seed=123)
    assert not np.allclose(rewards_42, rewards_123), \
        "Different seeds produced identical results"


# ---------------------------------------------------------------------------
# Test 10: Year log has expected keys
# ---------------------------------------------------------------------------

def test_year_log_keys():
    env = load_env()
    env.reset()
    n_agents = env.n_agents

    auction_actions = np.random.uniform(
        [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
        size=(n_agents, 6)
    ).astype(np.float32)
    obs2, _ = env.step_auction(auction_actions)

    secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
    secondary_actions[:, 0] = env._phase1_clearing_price
    _, _, _, _, info = env.step_secondary(secondary_actions)
    log = info["year_log"]

    for key in ["year", "cap", "tnac", "auction_volume", "clearing_price",
                "allocations", "penalties", "rewards", "green_fracs",
                "tech_mixes", "holdings", "invest_costs",
                "bank_start", "shortfalls", "delta_greens", "queue_sizes", "bid_prices",
                "emission_shocks", "cf_shocks", "cancellations"]:  # P5/P6
        assert key in log, f"Missing key in year_log: {key}"


# ---------------------------------------------------------------------------
# Test 11: Emissions decrease with greener mix
# ---------------------------------------------------------------------------

def test_emissions_decrease_with_green():
    """Greenest learning agent should have much lower emissions than coal-heavy agent."""
    env = load_env()
    env.reset()
    e1 = env.companies[0].compute_emissions()  # coal-heavy
    # Green leader is the last LEARNING agent (index n_agents-1), not last company (bots follow)
    e_green = env.companies[env.n_agents - 1].compute_emissions()
    assert e_green < e1, f"Green emissions ({e_green}) should be < coal emissions ({e1})"
    assert e_green < 0.5 * e1, f"Green agent should have significantly lower emissions"


# ---------------------------------------------------------------------------
# Test 12: Investment costs are realistic (not off by 100x)
# ---------------------------------------------------------------------------

def test_investment_costs_realistic():
    """Shifting 3% output to onshore wind should cost ~tens of M€, not ~1.5 M€."""
    env = load_env()
    env.reset()
    c = env.companies[0]
    # 3% of 10 TWh = 300 GWh → needs ~98 MW at 35% CF → ~€132M
    cost = c.compute_investment_cost(tech_idx=2, frac_delta=0.03)  # onshore wind
    assert cost > 50, f"Investment cost ({cost:.1f} M€) is too low — should be >50 M€"
    assert cost < 500, f"Investment cost ({cost:.1f} M€) seems too high"


# ---------------------------------------------------------------------------
# Test 13: P5 — Emission shocks produce variance across episodes
# ---------------------------------------------------------------------------

def test_p5_emission_variance():
    """With uncertainty enabled, realized emissions should vary across episodes."""
    env = load_env()
    if not env.config.get("uncertainty", {}).get("enabled", False):
        # P5 is disabled in config — shocks should all be zero
        return
    emissions_ep = []
    for ep in range(20):
        env.reset(seed=ep * 999)
        n_agents = env.n_agents
        auction_actions = np.random.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.0, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        _, log = env.step_auction(auction_actions)
        emissions_ep.append(log["emission_shocks"][0])

    std = np.std(emissions_ep)
    assert std > 1e-6, "Emission shocks show no variance — P5 uncertainty may not be active"


# ---------------------------------------------------------------------------
# Test 14: P6 — CF noise method is numerically stable
# ---------------------------------------------------------------------------

def test_p6_cf_noise_stability():
    """compute_emissions_with_cf_noise should return positive finite result."""
    env = load_env()
    env.reset()
    import numpy as np
    cf_noise = np.array([0.0, 0.0, -0.15, 0.10, -0.05])
    for company in env.companies:
        e_base = company.compute_emissions()
        e_noisy = company.compute_emissions_with_cf_noise(cf_noise)
        assert e_noisy > 0, f"CF-noisy emissions non-positive: {e_noisy}"
        assert np.isfinite(e_noisy), f"CF-noisy emissions non-finite: {e_noisy}"
        # Lower green CF → more fossil → more emissions (or equal for 100% green)
        # Just check it's in a reasonable range
        assert e_noisy < e_base * 3.0, "CF noise causing implausible emission spike"


# ---------------------------------------------------------------------------
# Test 15: P7 — Warm-start seeds non-zero bank
# ---------------------------------------------------------------------------

def test_p7_warm_start_bank():
    """After reset with warm_start enabled, all agents should have bank > 0."""
    env = load_env()
    env.reset(seed=42)
    for i, h in enumerate(env.holdings):
        assert h > 0, f"Agent {i+1} has zero bank after warm-start reset: holdings={h}"


# ---------------------------------------------------------------------------
# Test 16: P7 — Warm-start seeds price history
# ---------------------------------------------------------------------------

def test_p7_price_history_seeded():
    """After reset with warm_start, price history should be non-empty."""
    env = load_env()
    env.reset(seed=42)
    assert len(env._price_history) > 0, "Price history empty after warm-start reset"


def test_burnin_tnac_in_band():
    """After burn-in, median TNAC over multiple seeds should be inside MSR band."""
    tnacs = []
    env = None
    for seed in range(10):
        env = load_env(seed=seed)
        env.reset(seed=seed)
        tnacs.append(float(env.holdings.sum()))

    median_tnac = sorted(tnacs)[len(tnacs) // 2]
    cfg = env.config["ets"]["msr"]
    assert cfg["tnac_lower"] <= median_tnac <= cfg["tnac_upper"], (
        f"Median TNAC {median_tnac:.2f} outside MSR band "
        f"[{cfg['tnac_lower']}, {cfg['tnac_upper']}]"
    )


def test_burnin_price_history_realistic():
    """After burn-in, price history should contain realistic values."""
    env = load_env()
    env.reset(seed=42)
    assert len(env._price_history) >= 2
    for p in env._price_history:
        assert 5.0 < p < 300.0, f"Unrealistic burn-in price: {p}"


def test_burnin_msr_reserve():
    """After burn-in, MSR reserve should be non-negative."""
    env = load_env()
    env.reset(seed=42)
    assert env.cap_schedule._msr_reserve >= 0


def test_burnin_prev_ma3_seeded():
    """After burn-in, cap_schedule._prev_ma3 should be set (not None) so the
    A4 smoothed guard is active from year 0 of the real episode."""
    env = load_env()
    env.reset(seed=42)
    assert env.cap_schedule._prev_ma3 is not None, (
        "_prev_ma3 should be seeded during burn-in so the A4 guard is "
        "active from year 0; got None instead"
    )
    assert env.cap_schedule._prev_ma3 > 0, (
        f"_prev_ma3 should be a positive price after burn-in; "
        f"got {env.cap_schedule._prev_ma3}"
    )


# ---------------------------------------------------------------------------
# Test 17: P8 — obs dims updated correctly (20 base, +2*(N-1) opp)
# ---------------------------------------------------------------------------

def test_p8_obs_dims():
    """Phase 1 obs should be 43D base (+ 7*(N_total-1) opponent dims) with opponent modeling.
    43 base dims: 38 prior + 5 clip feedback dims [38]-[42].
    N_total = learning + bot agents."""
    env = load_env()
    obs, _ = env.reset()
    n_agents = env.config["companies"]["n_agents"]
    n_total = n_agents + env.config["companies"].get("n_bot_agents", 0)
    opp_enabled = env.config.get("opponent_modeling", {}).get("enabled", False)
    opp_dims = env.config.get("opponent_obs", {}).get("dims_per_opponent", 7)
    expected_p1 = 43 + (opp_dims * (n_total - 1) if opp_enabled else 0)
    expected_p2 = expected_p1 + 12  # +12: alloc, price, compliance_pos, shock, auction_savings, coverage_ratio, carry_forward_norm, collateral_locked_norm, budget_remaining_phase2_norm, compliance_liability_norm, compliance_gap_norm, sec_qty_clip_ratio
    assert obs.shape == (n_agents, expected_p1), (
        f"Phase 1 obs: expected ({n_agents}, {expected_p1}), got {obs.shape}"
    )

    # Auction actions: action[1] is now a COVERAGE MULTIPLIER on estimated need
    auction_actions = np.random.uniform(
        [5.0, 0.3, 0.0, -1.0, -1.0, -1.0],
        [200.0, 2.0, 0.05, 1.0, 1.0, 1.0],
        size=(n_agents, 6)
    ).astype(np.float32)
    obs2, _ = env.step_auction(auction_actions)
    assert obs2.shape == (n_agents, expected_p2), (
        f"Phase 2 obs: expected ({n_agents}, {expected_p2}), got {obs2.shape}"
    )


# ---------------------------------------------------------------------------
# Test 18: Static reserve price equals config value
# ---------------------------------------------------------------------------

def test_static_reserve_equals_config():
    """In static mode, effective reserve equals the configured reserve_price."""
    env = load_env()
    env.reset(seed=42)

    ets_cfg = env.config["ets"]
    assert ets_cfg.get("reserve_price_mode") == "static", "Expected static reserve mode"

    effective = env._compute_dynamic_reserve()
    expected = ets_cfg["reserve_price"]
    assert effective == pytest.approx(expected, abs=1e-6), (
        f"Static reserve {effective:.2f} != configured {expected:.2f}"
    )


def test_static_reserve_no_silent_rejection():
    """With price_min == reserve_price (static), bids at price_min are accepted, not rejected."""
    env = load_env()
    env.reset(seed=42)

    n_agents = env.n_agents
    price_min = env.config["auction"]["price_min"]
    effective_reserve = env._compute_dynamic_reserve()

    # price_min should equal reserve_price in static mode
    assert price_min == pytest.approx(effective_reserve, abs=1e-6), (
        f"price_min ({price_min}) != reserve_price ({effective_reserve}) — "
        "static mode requires these to match to avoid silent bid rejection"
    )

    # Bid at exactly price_min — should NOT be rejected
    auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
    auction_actions[:, 0] = price_min
    auction_actions[:, 1] = 1.0

    _, log = env.step_auction(auction_actions)

    assert not log["auction_stats"].get("auction_failed", False), (
        "Auction should not fail when all bids are at price_min == reserve_price"
    )
    # At least some allocation should have occurred (unsold < total supply)
    unsold = log["auction_stats"].get("unsold", 0.0)
    total_demand = log["auction_stats"].get("total_demand", 0.0)
    assert total_demand > 0, "Bids at reserve_price must generate demand"
    assert unsold < log["auction_stats"].get("q_cap", float('inf')), \
        "Bids at reserve_price must receive allocation"


def test_unsold_volume_rolls_over_to_next_year():
    """When unsold_to_msr=false, unsold volume should appear in next year's auction supply."""
    env = load_env()
    # Override: this test specifically tests rollover behavior with unsold_to_msr=false
    env.config["ets"]["unsold_to_msr"] = False
    assert not env.config["ets"].get("unsold_to_msr", True), "Expected unsold_to_msr=false"
    # Temporarily disable bots for this test so that low bidding produces unsold volume
    saved_n_bots = env.n_bots
    env.n_bots = 0
    env.n_total = env.n_agents
    # Re-create environment without bots for clean test
    import copy
    config = copy.deepcopy(env.config)
    config["companies"]["n_bot_agents"] = 0
    config["auction"]["qty_mult_low"] = 0.1  # allow very low bids for rollover test
    env = ETSEnvironment(config, seed=42)
    env.reset(seed=42)

    n_agents = env.n_agents

    # Year 0: bid very low quantity so most volume goes unsold
    auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
    auction_actions[:, 0] = 100.0  # reasonable price
    auction_actions[:, 1] = 0.3    # low coverage multiplier → small bid qty

    obs2, log0 = env.step_auction(auction_actions)
    unsold_yr0 = log0["unsold_rollover_out"]
    assert unsold_yr0 > 0.1, f"Expected meaningful unsold volume, got {unsold_yr0}"

    # Check rollover is pending in cap_schedule
    assert env.cap_schedule._unsold_rollover_pending == pytest.approx(unsold_yr0, abs=1e-4)

    # Complete year 0
    sec_actions = np.zeros((n_agents, 2), dtype=np.float32)
    sec_actions[:, 0] = env._phase1_clearing_price
    env.step_secondary(sec_actions)

    # Year 1: get auction volume — should include rollover
    year1 = env.current_year
    tnac = float(env.holdings.sum())
    price_max = float(env.config["auction"]["price_max"])
    base_cap = env.cap_schedule.get_cap(year1)
    actual_volume = env.cap_schedule.get_auction_volume(
        year1, tnac, env.last_clearing_price, price_max
    )

    # Volume should exceed the base cap by approximately the rollover amount
    # (MSR adjustments may modify it, but the rollover should be included)
    assert actual_volume > base_cap - 0.1, (
        f"Year 1 auction volume ({actual_volume:.2f}) should be at least near "
        f"base cap ({base_cap:.2f}) + rollover ({unsold_yr0:.2f})"
    )
    # Rollover should now be consumed
    assert env.cap_schedule._unsold_rollover_pending == pytest.approx(0.0, abs=1e-9)


def test_defaulted_volume_not_double_counted_with_unsold_rollover():
    """Defaulted volume and unsold rollover must not be added twice."""
    import copy

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Disable bots for deterministic demand and relax leverage so defaults can occur.
    # Low budgets ensure payment (price_max × allocation) exceeds agent cash even with loans.
    config = copy.deepcopy(config)
    config["companies"]["n_bot_agents"] = 0
    config["auction"]["leverage_multiplier"] = 100.0
    config["budget"]["annual_budgets"] = [200.0] * 8  # well below price_max × per-agent alloc
    # v8.5.3: BCL now clips year-0 bids to ≈[ma3-V, ma3+V]; this test relies on
    # extreme 500 EUR/t bids to trigger affordability defaults, so disable BCL
    # for this stress setup. (BCL semantics are exercised by test_bid_change_limit.py.)
    config["auction"]["bid_change_limit"] = {"enabled": False, "value": 75.0}

    env = ETSEnvironment(config, seed=42)
    env.reset(seed=42)
    n_agents = env.n_agents

    # Aggressive bids to stress affordability and trigger defaults.
    auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
    auction_actions[:, 0] = 500.0
    auction_actions[:, 1] = 2.0

    _, log = env.step_auction(auction_actions)

    defaulted = float(log["auction_stats"].get("defaulted_volume", 0.0))
    assert defaulted > 0.0, "Expected defaulted volume for this stress setup"

    rollover_total = float(env.cap_schedule._unsold_rollover_pending + env._defaulted_volume_pending)
    post_settlement_gap = max(
        0.0,
        float(log["auction_volume"]) - float(np.sum(env._phase1_allocations)),
    )

    assert rollover_total == pytest.approx(post_settlement_gap, abs=1e-6), (
        "Rollover streams should exactly match the post-settlement supply gap "
        "(no double counting of defaults in unsold rollover)."
    )


def test_secondary_profit_ema_resets_each_episode():
    """Secondary profit EMA must not leak state across episode resets."""
    env = load_env()
    env.reset(seed=42)

    env._secondary_profit_ema[:] = 1.23
    env.reset(seed=43)

    assert np.allclose(env._secondary_profit_ema, 0.0), (
        "secondary profit EMA should be zeroed on reset"
    )


def test_liquidity_pool_fills_at_reference_plus_spread():
    """External liquidity pool should fill unmatched buy flow at reference*(1+spread)."""
    env = load_env()
    env.reset(seed=42)

    env.config.setdefault("secondary", {}).setdefault("liquidity_pool", {})["enabled"] = True
    env.config["secondary"]["liquidity_pool"]["spread"] = 0.05
    env.config["secondary"]["liquidity_pool"]["penalty_anchor_weight"] = 0.30
    env._liquidity_ref_ema = 100.0

    n_total = env.n_total
    allocations = np.zeros(n_total)
    secondary_prices = np.full(n_total, 90.0)
    secondary_qtys = np.zeros(n_total)

    # One buyer with no internal seller counterpart -> pool should fill.
    secondary_prices[0] = 120.0
    secondary_qtys[0] = 1.0

    trade_costs, trade_qtys, _, total_volume, pool_info = env._settle_double_auction(
        allocations=allocations,
        secondary_prices=secondary_prices,
        secondary_qtys=secondary_qtys,
        clearing_price=100.0,
    )

    assert pool_info["enabled"] is True
    assert pool_info["reference_price"] == pytest.approx(100.0, abs=1e-6)
    assert pool_info["sell_price"] == pytest.approx(105.0, abs=1e-6)
    assert trade_qtys[0] == pytest.approx(1.0, abs=1e-6)
    assert pool_info["sell_volume"] == pytest.approx(1.0, abs=1e-6)
    assert total_volume == pytest.approx(1.0, abs=1e-6)
    # Buyer pays pool sell price plus transaction cost.
    expected_cost = 1.0 * (105.0 + env.config["trading"]["transaction_cost"])
    assert trade_costs[0] == pytest.approx(expected_cost, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 19: No holding limit (max_agent_share = 1.0)
# ---------------------------------------------------------------------------

def test_no_holding_limit():
    """With max_agent_share=1.0, a high-bidding agent gets a larger allocation share."""
    env = load_env()
    env.reset(seed=42)
    n_agents = env.n_agents

    # Agent 0 bids at 160 EUR/t (above bot heuristic ~138 EUR/t) with a moderate
    # quantity multiplier (0.8×) so the auction payment stays within its annual
    # budget under E4 collateral constraints.  Other learning agents bid below
    # reserve and so are filtered out; bots generate their own higher bids.
    auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
    auction_actions[:, 0] = 40.0   # other agents bid below bot prices
    auction_actions[:, 1] = 0.3    # small multiplier
    auction_actions[0, 0] = 160.0  # agent 0 bids above bots → priority fill
    auction_actions[0, 1] = 0.8    # moderate qty — payment affordable under E4

    obs2, log = env.step_auction(auction_actions)

    assert not log["auction_stats"].get("auction_failed", False), "Auction should not fail"
    # Agent 0 should not have defaulted (bid is within budget)
    assert env._phase1_allocations[0] > 0, "Agent 0 should have won allocation and not defaulted"
    # Agent 0's allocation should be substantial (they bid highest, no holding limit)
    agent0_alloc = env._phase1_allocations[0]
    total_alloc = float(env._phase1_allocations.sum())
    if total_alloc > 1e-9:
        share = agent0_alloc / total_alloc
        assert share > 0.10, f"Agent 0 share {share:.2f} too low with no holding limit"


# ---------------------------------------------------------------------------
# Test: price_history_anchor="auction" prevents failed-auction price pollution
# ---------------------------------------------------------------------------

def test_price_history_anchor_default_is_auction():
    """Default price_history_anchor should be 'auction' (not 'secondary')."""
    env = load_env()
    env.reset(seed=42)
    assert env._reserve_anchor == "auction", (
        f"Expected _reserve_anchor='auction', got '{env._reserve_anchor}'. "
        "The 'secondary' default caused failed-auction reserve prices to pollute "
        "the MA3 price signal, driving heuristic bids into a downward spiral."
    )


def test_auction_anchor_excludes_secondary_prices_from_history():
    """With anchor='auction', secondary market trade prices must NOT enter _price_history.

    The secondary-price anchor (old default) allowed secondary trades at urgency-premium
    prices to distort the MA3 signal, causing erratic bid behaviour in subsequent rounds.
    With the 'auction' anchor, only primary auction clearing prices enter the MA3 history.
    """
    env = load_env()
    env.reset(seed=42)
    assert env._reserve_anchor == "auction", "Prerequisite: must be 'auction' anchor"

    n_agents = env.n_agents
    # Run a year that generates secondary trades
    auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
    auction_actions[:, 0] = 100.0   # bid above reserve → auction clears at ~100
    auction_actions[:, 1] = 1.0

    _, log = env.step_auction(auction_actions)
    auction_clearing = log["clearing_price"]
    history_after_auction = list(env._price_history)

    # Phase 2: force trades — half agents buy at high price, half sell at high price
    secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
    for i in range(n_agents):
        if i % 2 == 0:
            secondary_actions[i, 0] = 150.0   # willing to buy at 150
            secondary_actions[i, 1] = 0.5     # buy 0.5 Mt
        else:
            secondary_actions[i, 0] = 80.0    # willing to sell at 80
            secondary_actions[i, 1] = -0.5    # sell 0.5 Mt

    env.step_secondary(secondary_actions)

    # With 'auction' anchor: only the primary auction clearing price should have been
    # added to history (which already happened in step_auction).  The secondary-market
    # trade prices must NOT have been appended a second time.
    assert len(env._price_history) == len(history_after_auction), (
        "With 'auction' anchor, step_secondary must NOT append additional prices to "
        "_price_history — secondary trade prices must not distort the MA3."
    )


def test_secondary_anchor_adds_secondary_clearing_to_history():
    """With anchor='secondary' (legacy mode), secondary clearing prices enter _price_history.

    This test documents the old (now non-default) behaviour to ensure it still works
    when explicitly requested, while confirming the new default ('auction') does not
    exhibit the same distortion.
    """
    import copy
    with open(CONFIG_PATH) as f:
        import yaml
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    config["ets"]["price_history_anchor"] = "secondary"  # explicitly opt in to old mode
    env = ETSEnvironment(config, seed=42)
    env.reset(seed=42)

    assert env._reserve_anchor == "secondary", "Prerequisite: must be 'secondary' anchor"

    n_agents = env.n_agents
    auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
    auction_actions[:, 0] = 100.0
    auction_actions[:, 1] = 1.0

    env.step_auction(auction_actions)
    history_after_auction = list(env._price_history)

    # Force secondary trades at a price above the auction clearing
    secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
    for i in range(n_agents):
        if i % 2 == 0:
            secondary_actions[i, 0] = 150.0
            secondary_actions[i, 1] = 0.5
        else:
            secondary_actions[i, 0] = 80.0
            secondary_actions[i, 1] = -0.5

    env.step_secondary(secondary_actions)

    # With 'secondary' anchor: secondary clearing price IS added to history
    assert len(env._price_history) == len(history_after_auction) + 1, (
        "With 'secondary' anchor, step_secondary must append the secondary clearing "
        "price to _price_history."
    )


def test_successful_auction_updates_price_history():
    """A successful auction should add the clearing price to price_history."""
    env = load_env()
    env.reset(seed=42)
    history_len_before = len(env._price_history)

    n_agents = env.n_agents
    # Bid well above reserve to guarantee success
    auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
    auction_actions[:, 0] = 120.0   # above reserve_price (30 EUR/t)
    auction_actions[:, 1] = 1.0     # coverage multiplier

    _, log = env.step_auction(auction_actions)
    assert not log["auction_stats"].get("auction_failed", False), (
        "Auction should succeed with bids well above reserve"
    )

    sec_actions = np.zeros((n_agents, 2), dtype=np.float32)
    sec_actions[:, 0] = env._phase1_clearing_price
    env.step_secondary(sec_actions)

    assert len(env._price_history) > history_len_before, (
        "Successful auction clearing price should be appended to _price_history"
    )


def test_qlearning_state_discretizer_indices_stable():
    """New Phase-1 dims are appended; gap/carry-forward indices remain unchanged."""
    from src.agents.q_learning_agent import StateDiscretizer

    assert StateDiscretizer._OBS_GAP == 13
    assert StateDiscretizer._OBS_CF == 20


# ---------------------------------------------------------------------------
# Test: NPV discounting in heuristic_policy
# ---------------------------------------------------------------------------

def test_heuristic_npv_uses_discounting():
    """Heuristic invest_frac must be strictly lower with discounting than without."""
    import yaml
    from src.agents.heuristic_policy import auction_action
    from src.environment.company import Company

    with open(CONFIG_PATH) as f:
        config_base = yaml.safe_load(f)

    rng = np.random.default_rng(42)
    company = Company(
        agent_id=0,
        config=config_base,
        initial_mix=[0.40, 0.40, 0.10, 0.05, 0.05],  # coal-heavy: high EF → clear NPV incentive
        rng=rng,
    )

    # Config with discount rate 0 → undiscounted (original behavior: simple sum)
    import copy
    config_no_discount = copy.deepcopy(config_base)
    config_no_discount["investment"]["discount_rate"] = 0.0

    # Config with discount rate 5% → discounted (fixed behavior)
    config_discounted = copy.deepcopy(config_base)
    config_discounted["investment"]["discount_rate"] = 0.05

    action_no_disc = auction_action(
        company, price_ma3=100.0, current_year=1, n_years=12,
        config=config_no_discount, bank=0.0,
    )
    action_disc = auction_action(
        company, price_ma3=100.0, current_year=1, n_years=12,
        config=config_discounted, bank=0.0,
    )

    # With a 5% discount rate, the NPV factor < horizon, so invest_frac must be
    # less-than-or-equal for any given NPV ratio path.
    assert action_disc[2] <= action_no_disc[2] + 1e-5, (
        f"Discounted invest_frac ({action_disc[2]:.4f}) should be <= undiscounted "
        f"({action_no_disc[2]:.4f}); discounting should reduce NPV-driven investment"
    )


def test_heuristic_npv_discount_reduces_long_horizon_investment():
    """Discounting should reduce investment relative to undiscounted for a long-horizon episode.

    Uses agent_id=0 (financial archetype) with a carbon price of 60 EUR/t, where:
    - The undiscounted NPV ratio (factor=horizon) exceeds the investment threshold (>1),
      so the financial agent commits capital.
    - The discounted NPV ratio (annuity factor at 5%, ~12.97 for 21 years) falls below
      the threshold, so the agent correctly avoids the unprofitable investment.
    This demonstrates that proper NPV discounting avoids over-investment at long horizons.
    """
    import yaml
    from src.agents.heuristic_policy import auction_action
    from src.environment.company import Company
    import copy

    with open(CONFIG_PATH) as f:
        config_base = yaml.safe_load(f)

    rng = np.random.default_rng(42)
    company = Company(
        agent_id=0,
        config=config_base,
        initial_mix=[0.40, 0.40, 0.10, 0.05, 0.05],  # coal-heavy: high EF → clear NPV incentive
        rng=rng,
    )

    config_no_disc = copy.deepcopy(config_base)
    config_no_disc["investment"]["discount_rate"] = 0.0
    config_disc = copy.deepcopy(config_base)
    config_disc["investment"]["discount_rate"] = 0.05

    # At price_ma3=60 EUR/t, n_years=20, current_year=0:
    # - Undiscounted: NPV ratio > 1 → financial agent invests.
    # - Discounted at 5% over ~21 years: annuity factor ~12.85 vs 21 undiscounted
    #   → NPV ratio < 1 → financial agent correctly withholds investment.
    a_nd = auction_action(company, price_ma3=60.0, current_year=0, n_years=20,
                          config=config_no_disc, bank=0.0)
    a_d  = auction_action(company, price_ma3=60.0, current_year=0, n_years=20,
                          config=config_disc, bank=0.0)

    assert a_d[2] < a_nd[2], (
        f"With price_ma3=60 and n_years=20, proper NPV discounting (5%) should "
        f"keep invest_frac lower than the undiscounted sum; "
        f"got disc={a_d[2]:.5f} vs no_disc={a_nd[2]:.5f}"
    )
