"""
test_company.py
===============
Unit tests for Company class — core business logic that drives the simulation.

Covers:
  - Emissions computation (deterministic and with CF noise)
  - MAC fuel-switching logic
  - Investment queue lifecycle (plan → mature, with risk)
  - Compliance and carry-forward obligations
  - Budget enforcement
  - Observation generation (phase 1 and phase 2)
  - Properties (green_frac, fossil_frac, weighted_emission_factor)
"""

import sys
import os
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.company import Company, N_TECHS, BUILDABLE_INDICES
from src.environment.ets_environment import ETSEnvironment

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


# ---------------------------------------------------------------------------
# Shared fixture: minimal config for Company instantiation
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Minimal config dict for Company."""
    return {
        "companies": {
            "n_agents": 4,
            "output_twh": 10.0,
            "initial_mix": [
                [0.40, 0.40, 0.10, 0.05, 0.05],
                [0.15, 0.45, 0.20, 0.10, 0.10],
                [0.05, 0.25, 0.35, 0.20, 0.15],
                [0.00, 0.10, 0.30, 0.35, 0.25],
            ],
            "reward_weights": [[0.75, 0.25]] * 4,
        },
        "technologies": {
            "names": ["coal", "gas", "onshore_wind", "offshore_wind", "solar"],
            "emission_factors": [0.820, 0.490, 0.011, 0.012, 0.048],
            "capex": [3000, 1150, 1350, 3250, 750],
            "capacity_factors": [0.65, 0.60, 0.35, 0.47, 0.17],
            "deploy_delays": [0, 0, 3, 5, 1],
            "operational_costs": [72.0, 55.0, 17.0, 47.0, 10.0],
            "decommission_costs": [200, 100, 0, 0, 0],
            "is_green": [False, False, True, True, True],
            "is_buildable": [False, False, True, True, True],
        },
        "investment": {"max_invest_frac": 0.10, "convexity_alpha": 0.10},
        "risk": {
            "p_fail_min": 0.08, "p_fail_max": 0.65,
            "p_fail_alpha": 0.7, "experience_discount": 0.10,
            "experience_threshold": 2,
        },
        "penalty": {
            "rate": 100.0, "carry_forward": True,
            "carry_forward_cap": 0.5,
        },
        "auction": {"price_max": 500.0, "price_min": 5.0, "quantity_max": 3.0},
        "budget": {
            "annual_budgets": [1500.0, 1200.0, 800.0, 500.0],
            "overspend_penalty_coef": 0.5,
            "contingency_zone": 0.10,
            "hard_cap_multiplier": 1.20,
            "contingency_penalty_coef": 0.05,
            "capex_throughputs": [200.0, 200.0, 200.0, 200.0],
        },
        "mac": {"enabled": True, "coal_to_gas_cost": 48.0, "max_switch_frac": 0.20},
        "opponent_modeling": {"enabled": False},
        "construction_jitter": {"enabled": False},
    }


def make_company(config, agent_id=0, seed=42):
    rng = np.random.default_rng(seed)
    mix = config["companies"]["initial_mix"][agent_id]
    return Company(agent_id=agent_id, config=config, initial_mix=mix, rng=rng)


def load_env_config(seed=42):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["simulation"]["n_years"] = 3
    cfg["ets"]["lrf_phase1"] = 0.20
    cfg["ets"]["lrf_phase2"] = 0.20
    cfg["companies"]["n_bot_agents"] = 0
    cfg["warm_start"]["enabled"] = False
    cfg["uncertainty"]["enabled"] = False
    cfg["construction_jitter"]["enabled"] = False
    cfg["simulation"]["seeds"] = [seed]
    return cfg


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_mix_sums_to_one(config):
    c = make_company(config, agent_id=0)
    assert abs(c.mix.sum() - 1.0) < 1e-6

def test_green_frac(config):
    c = make_company(config, agent_id=0)
    # Agent 0 mix: [0.40, 0.40, 0.10, 0.05, 0.05], green = 0.10+0.05+0.05 = 0.20
    assert abs(c.green_frac - 0.20) < 1e-6

def test_fossil_frac(config):
    c = make_company(config, agent_id=0)
    assert abs(c.fossil_frac - 0.80) < 1e-6

def test_green_plus_fossil_is_one(config):
    for i in range(4):
        c = make_company(config, agent_id=i)
        assert abs(c.green_frac + c.fossil_frac - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------

def test_emissions_deterministic(config):
    """Emissions = output_MWh * weighted_EF / 1e6 (Mt)."""
    c = make_company(config, agent_id=0)
    ef = np.dot(c.mix, c.emission_factors)
    expected = c.output_mwh * ef / 1e6
    assert abs(c.compute_emissions() - expected) < 1e-8

def test_emissions_positive(config):
    for i in range(4):
        c = make_company(config, agent_id=i)
        assert c.compute_emissions() > 0

def test_emissions_with_zero_cf_noise(config):
    """Zero CF noise → same as deterministic."""
    c = make_company(config, agent_id=0)
    noise = np.zeros(N_TECHS)
    assert abs(c.compute_emissions_with_cf_noise(noise) - c.compute_emissions()) < 1e-8

def test_emissions_with_positive_cf_noise_reduces(config):
    """Positive green CF noise → more green output → less fossil → lower emissions."""
    c = make_company(config, agent_id=0)
    noise = np.array([0.0, 0.0, 0.15, 0.15, 0.15])  # green techs produce more
    e_noisy = c.compute_emissions_with_cf_noise(noise)
    e_base = c.compute_emissions()
    assert e_noisy < e_base, f"Expected lower emissions with positive CF noise: {e_noisy} vs {e_base}"

def test_emissions_with_negative_cf_noise_increases(config):
    """Negative green CF noise → less green output → more fossil → higher emissions."""
    c = make_company(config, agent_id=0)
    noise = np.array([0.0, 0.0, -0.15, -0.15, -0.15])  # green techs produce less
    e_noisy = c.compute_emissions_with_cf_noise(noise)
    e_base = c.compute_emissions()
    assert e_noisy > e_base, f"Expected higher emissions with negative CF noise: {e_noisy} vs {e_base}"

def test_greener_company_emits_less(config):
    """Agent with more green capacity should emit less."""
    c_dirty = make_company(config, agent_id=0)   # 80% fossil
    c_clean = make_company(config, agent_id=3)    # 10% fossil
    assert c_dirty.compute_emissions() > c_clean.compute_emissions()


def test_estimate_need_matches_emissions_no_risk_buffer(config):
    c = make_company(config, agent_id=0)
    assert c.compute_estimate_need() == pytest.approx(c.compute_emissions(), rel=1e-9)


# ---------------------------------------------------------------------------
# MAC fuel-switching
# ---------------------------------------------------------------------------

def test_mac_no_switch_below_cost(config):
    """No switching when carbon price < MAC cost."""
    c = make_company(config, agent_id=0)
    reduction, cost = c.apply_mac_switching(carbon_price=40.0)  # below 48
    assert reduction == 0.0
    assert cost == 0.0

def test_mac_switch_above_cost(config):
    """Switching occurs when carbon price > MAC cost."""
    c = make_company(config, agent_id=0)
    reduction, cost = c.apply_mac_switching(carbon_price=80.0)  # above 48
    assert reduction > 0
    assert cost > 0

def test_mac_reduction_bounded(config):
    """Reduction cannot exceed max_switch_frac of coal capacity."""
    c = make_company(config, agent_id=0)
    reduction, cost = c.apply_mac_switching(carbon_price=200.0)
    # max switchable = min(coal_frac=0.40, max_switch=0.20) = 0.20
    max_switched_mwh = 0.20 * c.output_mwh
    max_ef_reduction = c.emission_factors[0] - c.emission_factors[1]
    max_reduction = max_switched_mwh * max_ef_reduction / 1e6
    assert abs(reduction - max_reduction) < 1e-6

def test_mac_no_coal_no_switch(config):
    """No switching possible when agent has no coal."""
    c = make_company(config, agent_id=3)  # 0% coal
    reduction, cost = c.apply_mac_switching(carbon_price=200.0)
    assert reduction == 0.0
    assert cost == 0.0

def test_mac_does_not_modify_mix(config):
    """MAC switching is temporary — should not change the permanent mix."""
    c = make_company(config, agent_id=0)
    mix_before = c.mix.copy()
    c.apply_mac_switching(carbon_price=100.0)
    np.testing.assert_array_equal(c.mix, mix_before)


# ---------------------------------------------------------------------------
# Investment queue lifecycle
# ---------------------------------------------------------------------------

def test_plan_investment_adds_to_queue(config):
    c = make_company(config, agent_id=0)
    cost = c.plan_investment(tech_choice=2, invest_frac=0.05, current_year=0)  # solar
    assert len(c._construction_queue) == 1
    assert cost > 0

def test_plan_investment_zero_frac_no_queue(config):
    c = make_company(config, agent_id=0)
    cost = c.plan_investment(tech_choice=0, invest_frac=0.0, current_year=0)
    assert len(c._construction_queue) == 0
    assert cost == 0.0

def test_plan_investment_capped_at_fossil_frac(config):
    """Cannot invest more than remaining fossil fraction."""
    c = make_company(config, agent_id=3)  # only 10% fossil
    c.plan_investment(tech_choice=2, invest_frac=0.10, current_year=0)
    # Should be clipped to fossil_frac = 0.10
    item = c._construction_queue[0]
    assert item["frac_delta"] <= c.fossil_frac + 1e-6

def test_matured_investment_changes_mix(config):
    """Solar investment (1-year delay) should mature and update mix."""
    c = make_company(config, agent_id=0, seed=1)
    initial_green = c.green_frac
    # Force success by seeding — try multiple seeds to find one that succeeds
    for seed in range(100):
        c = make_company(config, agent_id=0, seed=seed)
        c.plan_investment(tech_choice=2, invest_frac=0.05, current_year=0)
        if c._construction_queue[0]["success"]:
            break
    assert c._construction_queue[0]["success"], "Could not find successful seed"
    # Solar has deploy_delay=1, so completes at year 1
    c.apply_matured_investments(current_year=1)
    assert c.green_frac > initial_green, "Green frac should increase after matured investment"
    assert abs(c.mix.sum() - 1.0) < 1e-6, "Mix should still sum to 1.0"

def test_failed_investment_no_mix_change(config):
    """Failed investment adds zero frac_delta — mix unchanged after maturity."""
    c = make_company(config, agent_id=0, seed=42)
    # Force failure by finding a seed that fails
    for seed in range(100):
        c = make_company(config, agent_id=0, seed=seed)
        c.plan_investment(tech_choice=2, invest_frac=0.05, current_year=0)
        if not c._construction_queue[0]["success"]:
            break
    if not c._construction_queue[0]["success"]:
        mix_before = c.mix.copy()
        c.apply_matured_investments(current_year=1)
        np.testing.assert_array_almost_equal(c.mix, mix_before, decimal=6)


# ---------------------------------------------------------------------------
# Compliance and carry-forward
# ---------------------------------------------------------------------------

def test_compliance_no_shortfall(config):
    """Enough allowances → zero penalty."""
    c = make_company(config, agent_id=0)
    emissions = c.compute_emissions()
    penalty = c.settle_compliance(allowances_held=emissions + 1.0)
    assert penalty == 0.0

def test_compliance_shortfall_penalty(config):
    """Shortfall × penalty_rate = penalty."""
    c = make_company(config, agent_id=0)
    emissions = c.compute_emissions()
    shortfall = 0.5  # Mt
    penalty = c.settle_compliance(allowances_held=emissions - shortfall)
    assert abs(penalty - shortfall * 100.0) < 1e-6

def test_carry_forward_accumulates(config):
    """Realized shortfall carries forward to next year."""
    c = make_company(config, agent_id=0)
    emissions = 3.0
    allowances = 2.0  # shortfall = 1.0
    c.settle_compliance_realized(allowances_held=allowances, realized_emissions=emissions)
    assert c._carry_forward > 0, "Carry-forward should accumulate on shortfall"

def test_carry_forward_capped(config):
    """Carry-forward is capped at carry_forward_cap × base_emissions."""
    c = make_company(config, agent_id=0)
    # Create a huge shortfall
    c.settle_compliance_realized(allowances_held=0.0, realized_emissions=100.0)
    base_need = c.compute_estimate_need()
    cap = config["penalty"]["carry_forward_cap"] * base_need
    assert c._carry_forward <= cap + 1e-6, (
        f"Carry-forward {c._carry_forward} exceeds cap {cap}")

def test_carry_forward_adds_to_next_obligation(config):
    """Carry-forward increases next year's compliance obligation."""
    c = make_company(config, agent_id=0)
    # Year 1: shortfall
    c.settle_compliance_realized(allowances_held=1.0, realized_emissions=3.0)
    cf = c._carry_forward
    assert cf > 0
    # Year 2: total_need = emissions + carry_forward
    # If we have enough for emissions but not carry-forward, we still get penalty
    emissions_y2 = c.compute_emissions()
    penalty = c.settle_compliance_realized(
        allowances_held=emissions_y2,  # covers emissions but not carry-forward
        realized_emissions=emissions_y2)
    assert penalty > 0, "Carry-forward should cause additional penalty"

def test_no_carry_forward_when_disabled(config):
    """With carry_forward disabled, shortfall does not persist."""
    config_no_cf = {**config, "penalty": {**config["penalty"], "carry_forward": False}}
    c = make_company(config_no_cf, agent_id=0)
    c.settle_compliance_realized(allowances_held=0.0, realized_emissions=5.0)
    assert c._carry_forward == 0.0


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

def test_budget_no_penalty_within_budget(config):
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(500.0)  # well within 1500 budget
    assert c.compute_budget_penalty() == 0.0

def test_budget_penalty_on_overspend(config):
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(2000.0)  # 500 over 1500 budget
    penalty = c.compute_budget_penalty()
    assert penalty > 0, "Should penalize overspending"

def test_budget_penalty_contingency_zone(config):
    """5% overspend should be positive in the soft zone (tiered penalty)."""
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(1.05 * c.annual_budget)
    p = c.compute_budget_penalty()
    assert p > 0.0
    # With tiered formula: coef * (normalized²) * overshoot_abs
    # normalized = 0.05/0.15 ≈ 0.333, overshoot_abs = 0.05 * budget
    # Must be moderate (not catastrophic) relative to budget
    assert p < c.annual_budget * 0.05, "Soft-zone penalty should stay moderate"


def test_budget_penalty_quadratic_zone_larger(config):
    """15% overspend should trigger a larger penalty than 5% overspend."""
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(1.05 * c.annual_budget)
    p_5 = c.compute_budget_penalty()

    c.reset_budget()
    c.record_spending(1.15 * c.annual_budget)
    p_15 = c.compute_budget_penalty()
    assert p_15 > p_5

def test_budget_utilization(config):
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(750.0)
    assert abs(c.get_budget_utilization() - 0.5) < 1e-6


def test_investment_scaled_to_budget_hard_cap():
    """Overspend attempt above 20% should be clipped by hard_cap_multiplier."""
    cfg = load_env_config(seed=11)
    n_agents = cfg["companies"]["n_agents"]
    cfg["budget"]["mode"] = "fixed"  # prevent dynamic budget from overriding
    cfg["budget"]["annual_budgets"] = [100.0] * n_agents
    cfg["budget"]["bot_annual_budgets"] = [100.0] * n_agents
    cfg["budget"]["hard_cap_multiplier"] = 1.20
    cfg["budget"]["capex_throughputs"] = [1e9] * n_agents

    env = ETSEnvironment(cfg, seed=11)
    env.reset(seed=11)

    actions = np.zeros((env.n_agents, 10), dtype=np.float32)
    # 3-tranche: [p1, q1, p2, q2, p3, q3, invest_frac, tech0, tech1, tech2]
    actions[:, 0] = 120.0   # price tranche 1
    actions[:, 1] = 0.33    # qty mult tranche 1
    actions[:, 2] = 120.0   # price tranche 2
    actions[:, 3] = 0.33    # qty mult tranche 2
    actions[:, 4] = 120.0   # price tranche 3
    actions[:, 5] = 0.33    # qty mult tranche 3
    actions[:, 6] = 0.20    # invest_frac (maps via (0.20+1)/2 * max_invest_frac)
    actions[:, 7] = 1.0     # tech logit onshore
    env.step_auction(actions)
    sec = np.zeros((env.n_agents, 2), dtype=np.float32)
    sec[:, 0] = 80.0
    _, _, _, _, info = env.step_secondary(sec)

    year_log = info["year_log"]
    inv_costs = year_log["invest_costs"]
    budget_ceiling = 100.0 * 1.20
    assert max(inv_costs) <= budget_ceiling + 1e-6


def test_capex_throughput_exact_allowed_and_over_blocked():
    """Capex at throughput is allowed; larger request is clipped to throughput."""
    cfg = load_env_config(seed=22)
    n_agents = cfg["companies"]["n_agents"]
    cfg["budget"]["annual_budgets"] = [5000.0] * n_agents
    cfg["budget"]["hard_cap_multiplier"] = 10.0
    cfg["budget"]["hard_cap_fraction"] = 10.0  # disable hard gate for this test
    cfg["technologies"]["decommission_costs"] = [0, 0, 0, 0, 0]

    probe_env = ETSEnvironment(cfg, seed=22)
    probe_env.reset(seed=22)
    probe_company = probe_env.companies[0]
    target_frac = 0.08
    target_cost = probe_company.compute_investment_cost(2, target_frac, 0)

    cfg["budget"]["capex_throughputs"] = [target_cost] * n_agents
    env = ETSEnvironment(cfg, seed=22)
    env.reset(seed=22)

    # With v7.2.1: invest_frac = ((action + 1) / 2) * max_invest_frac
    # To get invest_frac=target_frac: action = 2 * target_frac / max_invest_frac - 1
    max_invest_frac = cfg["investment"]["max_invest_frac"]
    action_value = 2.0 * target_frac / max_invest_frac - 1.0

    actions = np.zeros((env.n_agents, 10), dtype=np.float32)
    # 3-tranche: [p1, q1, p2, q2, p3, q3, invest_frac, tech0, tech1, tech2]
    actions[:, 0] = 120.0   # price tranche 1
    actions[:, 1] = 0.33    # qty mult tranche 1
    actions[:, 2] = 120.0   # price tranche 2
    actions[:, 3] = 0.33    # qty mult tranche 2
    actions[:, 4] = 120.0   # price tranche 3
    actions[:, 5] = 0.33    # qty mult tranche 3
    actions[:, 6] = action_value  # invest_frac
    actions[:, 7] = 1.0     # tech logit onshore
    env.step_auction(actions)
    sec = np.zeros((env.n_agents, 2), dtype=np.float32)
    sec[:, 0] = 80.0
    _, _, _, _, info = env.step_secondary(sec)
    invest_exact = float(info["year_log"]["invest_costs"][0])
    assert invest_exact == pytest.approx(target_cost, rel=1e-4)

    cfg["budget"]["capex_throughputs"] = [0.5 * target_cost] * n_agents
    env2 = ETSEnvironment(cfg, seed=22)
    env2.reset(seed=22)
    env2.step_auction(actions)
    _, _, _, _, info2 = env2.step_secondary(sec)
    invest_blocked = float(info2["year_log"]["invest_costs"][0])
    assert invest_blocked <= 0.5 * target_cost + 1e-4


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def test_obs_phase1_shape(config):
    """Phase 1 obs should be 29D base after Phase G + safety + C1 dims (no opponent modeling)."""
    c = make_company(config, agent_id=0)
    obs = c.get_observation_phase1(
        year=0, cap_t=24.0, last_clearing_price=80.0,
        expected_price=80.0, auction_gap=1.0)
    assert obs.shape == (29,), f"Expected 29D (Phase G + safety + C1 dims), got {obs.shape}"
    assert obs.dtype == np.float32

def test_obs_phase1_with_opponents(config):
    """With opponent modeling, obs should have 29 + 5*(N-1) dims (Phase G + safety + C1)."""
    config_opp = {**config, "opponent_modeling": {"enabled": True}}
    c = make_company(config_opp, agent_id=0)
    opponent_obs = np.zeros(5 * 3, dtype=np.float32)  # 3 opponents
    obs = c.get_observation_phase1(
        year=0, cap_t=24.0, last_clearing_price=80.0,
        expected_price=80.0, opponent_obs=opponent_obs)
    assert obs.shape == (29 + 15,)

def test_obs_phase2_extends_phase1(config):
    """Phase 2 obs = phase1 + 7 standard + 6 D1/D2 + 1 collateral + 2 E1/E2 = phase1 + 16 dims."""
    c = make_company(config, agent_id=0)
    obs1 = c.get_observation_phase1(
        year=0, cap_t=24.0, last_clearing_price=80.0, expected_price=80.0)
    obs2 = c.get_observation_phase2(
        obs_phase1=obs1, allocation=2.0, clearing_price=80.0,
        emissions=3.0, banked=1.0, emission_shock=0.05, payment=160.0)
    # Phase G: 29 base + 16 extra (7 standard + 6 D1/D2 + 1 collateral + 2 E1/E2)
    assert obs2.shape == (29 + 16,), f"Expected {29+16}D, got {obs2.shape}"
    # First 29 dims should match phase1
    np.testing.assert_array_equal(obs2[:29], obs1)

def test_obs_values_finite(config):
    """All observation values should be finite."""
    c = make_company(config, agent_id=0)
    obs1 = c.get_observation_phase1(
        year=5, cap_t=20.0, last_clearing_price=60.0,
        expected_price=70.0, auction_gap=2.0)
    assert np.all(np.isfinite(obs1))
    obs2 = c.get_observation_phase2(
        obs_phase1=obs1, allocation=1.5, clearing_price=60.0,
        emissions=3.0, banked=0.5)
    assert np.all(np.isfinite(obs2))

def test_obs_price_normalization(config):
    """Prices in obs should be normalized by price_max. Phase G: MA3 at [2] only."""
    c = make_company(config, agent_id=0)
    price_max = config["auction"]["price_max"]  # 500
    clearing = 100.0
    obs = c.get_observation_phase1(
        year=0, cap_t=24.0, last_clearing_price=clearing,
        expected_price=200.0, price_ma3=clearing)
    # [2] = price_ma3 / price_max
    assert abs(obs[2] - clearing / price_max) < 1e-6, "obs[2] should be MA3/price_max"
    # [3] = green_frac (NOT expected_price after Phase G)
    green_expected = float(c.mix[2] + c.mix[3] + c.mix[4])
    assert abs(obs[3] - green_expected) < 1e-6, "obs[3] should be green_frac after Phase G"


# ---------------------------------------------------------------------------
# Public info (opponent modeling)
# ---------------------------------------------------------------------------

def test_public_info_keys(config):
    c = make_company(config, agent_id=0)
    info = c.get_public_info()
    assert set(info.keys()) == {"emissions", "carry_forward", "green_frac", "fossil_frac", "queue_total"}

def test_queue_capacity_shape(config):
    c = make_company(config, agent_id=0)
    qc = c.get_queue_capacity()
    assert qc.shape == (3,)  # onshore, offshore, solar
    assert np.all(qc >= 0)


# ---------------------------------------------------------------------------
# Capex throughput enforcement
# ---------------------------------------------------------------------------

def test_capex_no_penalty_within_cap(config):
    """Spend within capex_throughput → zero capex penalty."""
    config["budget"]["capex_throughputs"] = [100.0, 100.0, 100.0, 100.0]
    config["budget"]["capex_overspend_coef"] = 1.0
    c = make_company(config, agent_id=0)
    c.reset_capex_budget()
    c.record_capex_spending(80.0)  # within 100 cap
    assert c.compute_capex_penalty() == 0.0

def test_capex_penalty_on_overshoot(config):
    """Spend above capex_throughput → positive penalty."""
    config["budget"]["capex_throughputs"] = [100.0, 100.0, 100.0, 100.0]
    config["budget"]["capex_overspend_coef"] = 1.0
    c = make_company(config, agent_id=0)
    c.reset_capex_budget()
    c.record_capex_spending(150.0)  # 50 over 100 cap
    penalty = c.compute_capex_penalty()
    assert penalty > 0, "Should penalize capex overshoot"

def test_capex_penalty_quadratic(config):
    """Penalty = coef × (overshoot/cap)² × cap."""
    config["budget"]["capex_throughputs"] = [100.0, 100.0, 100.0, 100.0]
    config["budget"]["capex_overspend_coef"] = 1.0
    c = make_company(config, agent_id=0)
    c.reset_capex_budget()
    c.record_capex_spending(150.0)
    overshoot = 50.0
    ratio = overshoot / 100.0
    expected = 1.0 * (ratio ** 2) * 100.0
    assert abs(c.compute_capex_penalty() - expected) < 1e-6

def test_capex_independent_of_compliance_budget(config):
    """Capex within cap, compliance budget stressed → zero capex penalty."""
    config["budget"]["capex_throughputs"] = [100.0, 100.0, 100.0, 100.0]
    config["budget"]["capex_overspend_coef"] = 1.0
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.reset_capex_budget()
    c.record_spending(700.0)        # heavy compliance spending
    c.record_capex_spending(80.0)   # within capex cap
    assert c.compute_capex_penalty() == 0.0
    # Compliance budget may or may not be penalized — that's independent
    # Just verify capex penalty is zero

def test_investment_hits_both_budgets(config):
    """Investment cost should increment both budget_spent and capex_spent."""
    config["budget"]["capex_throughputs"] = [100.0, 100.0, 100.0, 100.0]
    config["budget"]["capex_overspend_coef"] = 1.0
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.reset_capex_budget()
    # Simulate what the environment does: record investment cost in both
    invest_cost = 50.0
    c.record_spending(invest_cost)
    c.record_capex_spending(invest_cost)
    assert c.budget_spent_this_year == 50.0
    assert c.capex_spent_this_year == 50.0


# ---------------------------------------------------------------------------
# H+I: Tiered budget penalty & investment hard gate tests
# ---------------------------------------------------------------------------

def test_tiered_penalty_zero_below_soft_zone(config):
    """No penalty when spending is below soft_zone_start."""
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(0.95 * c.annual_budget)
    assert c.compute_budget_penalty() == 0.0

def test_tiered_penalty_positive_in_soft_zone(config):
    """Positive penalty when spending is in [soft_zone_start, hard_cap_fraction]."""
    config["budget"]["soft_zone_start"] = 1.0
    config["budget"]["hard_cap_fraction"] = 1.15
    config["budget"]["tiered_penalty_coef"] = 2.0
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(1.10 * c.annual_budget)
    p = c.compute_budget_penalty()
    assert p > 0.0

def test_tiered_penalty_quadratic_growth(config):
    """Penalty at 10% overshoot > penalty at 5% overshoot (quadratic growth)."""
    config["budget"]["soft_zone_start"] = 1.0
    config["budget"]["hard_cap_fraction"] = 1.15
    config["budget"]["tiered_penalty_coef"] = 2.0
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(1.05 * c.annual_budget)
    p5 = c.compute_budget_penalty()
    c.reset_budget()
    c.record_spending(1.10 * c.annual_budget)
    p10 = c.compute_budget_penalty()
    assert p10 > p5, f"Should be larger: {p10} vs {p5}"

def test_tiered_penalty_steep_above_hard_cap(config):
    """Penalty above hard_cap_fraction should be large."""
    config["budget"]["soft_zone_start"] = 1.0
    config["budget"]["hard_cap_fraction"] = 1.15
    config["budget"]["tiered_penalty_coef"] = 2.0
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(1.20 * c.annual_budget)  # above 1.15 hard cap
    p_above = c.compute_budget_penalty()
    c.reset_budget()
    c.record_spending(1.14 * c.annual_budget)  # just inside soft zone
    p_inside = c.compute_budget_penalty()
    assert p_above > p_inside

def test_tiered_penalty_respects_config(config):
    """Custom config values should change penalty magnitude."""
    config["budget"]["soft_zone_start"] = 0.90
    config["budget"]["hard_cap_fraction"] = 1.10
    config["budget"]["tiered_penalty_coef"] = 5.0
    c = make_company(config, agent_id=0)
    c.reset_budget()
    c.record_spending(1.05 * c.annual_budget)
    p = c.compute_budget_penalty()
    assert p > 0.0

def test_investment_hard_gate_clips_spending():
    """Investment hard gate should prevent spending above hard_cap_fraction × budget."""
    cfg = load_env_config(seed=33)
    n = cfg["companies"]["n_agents"]
    cfg["budget"]["annual_budgets"] = [200.0] * n
    cfg["budget"]["hard_cap_fraction"] = 1.15
    cfg["budget"]["investment_hard_gate"] = True
    env = ETSEnvironment(cfg, seed=33)
    env.reset(seed=33)

    actions = np.zeros((env.n_agents, 10), dtype=np.float32)
    actions[:, 0] = 80.0
    actions[:, 1] = 0.1
    actions[:, 2] = 80.0
    actions[:, 3] = 0.1
    actions[:, 4] = 80.0
    actions[:, 5] = 0.1
    actions[:, 6] = 1.0  # max investment
    actions[:, 9] = 1.0  # solar
    env.step_auction(actions)
    sec = np.zeros((env.n_agents, 2), dtype=np.float32)
    sec[:, 0] = 80.0
    _, _, _, _, info = env.step_secondary(sec)
    for i in range(env.n_agents):
        invest = float(info["year_log"]["invest_costs"][i])
        hard_limit = 1.15 * max(env.companies[i].annual_budget, 1.0)
        assert invest <= hard_limit + 1.0, f"Agent {i}: invest {invest} > hard cap {hard_limit}"

def test_reward_channels_dict_populated():
    """After a full year, reward channels should be populated for all agents."""
    cfg = load_env_config(seed=44)
    env = ETSEnvironment(cfg, seed=44)
    env.reset(seed=44)
    n = env.n_agents
    actions = np.zeros((n, 10), dtype=np.float32)
    actions[:, 0] = 80.0
    actions[:, 1] = 0.17
    actions[:, 2] = 80.0
    actions[:, 3] = 0.17
    actions[:, 4] = 80.0
    actions[:, 5] = 0.17
    env.step_auction(actions)
    sec = np.zeros((n, 2), dtype=np.float32)
    sec[:, 0] = 80.0
    env.step_secondary(sec)
    assert len(env._last_reward_channels) >= env.n_agents
