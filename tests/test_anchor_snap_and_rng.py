"""
test_anchor_snap_and_rng.py
===========================
Tier-3 mitigations for seed-driven price-basin lock-in:

1. ``PPOAgent`` ε-greedy exploration uses ``self._rng`` (per-agent, derived
   from the master seed) instead of the global ``np.random`` namespace, so
   exploration draws are independent across agents and reproducible from
   the master seed alone.

2. ``snap_price_head_to_anchor`` / ``restore_price_head`` provide an
   episode-scoped intervention that temporarily overwrites a learned
   ``auction_policy.price_head.bias`` with the fundamental anchor and
   restores the original bias afterwards.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.ppo_agent import PPOAgent


# ---------------------------------------------------------------------------
# Shared minimal config (mirrors tests/test_anchors.py setup)
# ---------------------------------------------------------------------------

def _make_config():
    return {
        "ppo": {
            "hidden_size": 64, "lr": 0.0003, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_eps": 0.2, "entropy_coef": 0.02,
            "value_coef": 0.5, "max_grad_norm": 0.5, "n_epochs": 2,
            "mini_batch_size": 4, "log_std_min": -2.5, "log_std_max": 0.0,
            "centralized_critic": False, "critic_hidden_size": 64,
        },
        "auction": {
            "price_min": 45.0, "price_max": 250.0, "quantity_max": 3.0,
            "qty_mult_low": 0.5, "qty_mult_high": 2.0,
        },
        "investment": {"max_invest_frac": 0.20},
        "trading": {"sec_price_min": 45.0, "sec_price_max_mult": 2.0},
        "penalty": {"rate": 138.75, "inflation_rate": 0.02,
                    "carry_forward": True, "carry_forward_cap": 0.0},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 10.0},
        "companies": {"n_agents": 8, "output_twh": 10.0,
                      "initial_mix": [[0.25, 0.30, 0.20, 0.15, 0.10]] * 8,
                      "annual_budgets": [880.0] * 8},
        "mac": {"enabled": True, "coal_to_gas_cost": 48.0, "max_switch_frac": 0.20},
        "price": {"banking_premium_mult": 1.4, "ar1_persistence": 0.85,
                  "volatility_std": 0.15, "burnin_std": 10.0,
                  "initial_expected": 70.0},
        "ets": {
            "cap_year_0_override": None, "initial_bank_fraction": 0.10,
            "cap_overhead_pct": 0.02, "lrf_phase1": 0.043, "lrf_phase2": 0.044,
            "lrf_phase_switch": 2,
            "msr": {
                "enabled": True, "tnac_upper_ratio": 0.36, "tnac_mid_ratio": None,
                "tnac_lower_ratio": None, "withhold_rate": 0.24,
                "release_frac": 0.0638297872, "activation_year": 1,
                "price_containment_absolute": 350, "price_release_absolute": 450,
                "emergency_release_frac": 0.064, "min_auction_frac": 0.10,
            },
            "banking": True, "reserve_price": 45.0, "reserve_price_mode": "static",
            "unsold_to_msr": False, "max_rollover_multiplier": 1.5,
            "price_history_anchor": "auction",
        },
        "technologies": {
            "names": ["coal", "gas", "onshore_wind", "offshore_wind", "solar"],
            "emission_factors": [0.820, 0.490, 0.011, 0.012, 0.048],
            "capacity_factors": [0.65, 0.60, 0.35, 0.47, 0.17],
            "capex": [3000, 1150, 1350, 3250, 750],
            "deploy_delays": [0, 0, 4, 7, 2],
            "operational_costs": [72.0, 55.0, 17.0, 47.0, 10.0],
            "decommission_costs": [200, 100, 0, 0, 0],
            "is_green": [False, False, True, True, True],
            "is_buildable": [False, False, True, True, True],
        },
        "simulation": {"n_years": 12, "n_episodes": 100},
        "exploration": {"mode": "uniform", "auction_anchors": None,
                        "secondary_anchors": None, "anchor_boost": 1.0},
    }


def _make_agent(agent_id: int = 0, seed: int = 0) -> PPOAgent:
    config = _make_config()
    return PPOAgent(
        agent_id=agent_id,
        obs_dim_phase1=22, obs_dim_phase2=29,
        auction_action_low=np.array([45.0, 0.5, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([250.0, 2.0, 0.20, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([45.0, -3.0], dtype=np.float32),
        secondary_action_high=np.array([350.0, 3.0], dtype=np.float32),
        config=config, seed=seed,
    )


# ---------------------------------------------------------------------------
# Anchor-snap save / restore
# ---------------------------------------------------------------------------

def test_snap_overwrites_then_restore_returns_original_bias():
    """snap_price_head_to_anchor must replace the bias with the anchor value,
    and restore_price_head must put the original bias back exactly.
    """
    agent = _make_agent(seed=0)

    # Set a deliberately non-anchor bias (high-basin proxy: ~0.7 raw → ~218 EUR/t).
    with torch.no_grad():
        agent.auction_policy.price_head.bias.fill_(0.7)
    bias_before = agent.auction_policy.price_head.bias.detach().clone()

    saved = agent.snap_price_head_to_anchor(year=0)

    # Saved tensor must equal the pre-snap bias and be a separate object.
    assert torch.allclose(saved, bias_before)
    assert saved.data_ptr() != agent.auction_policy.price_head.bias.data_ptr(), (
        "snap_price_head_to_anchor must return an independent clone, not a view"
    )

    # Bias was actually overwritten (anchor ≈67 EUR/t at yr0; raw far from 0.7).
    bias_snapped = agent.auction_policy.price_head.bias.detach().clone()
    assert not torch.allclose(bias_snapped, bias_before, atol=1e-3), (
        "snap_price_head_to_anchor did not overwrite the bias"
    )

    # Restore must return to the exact pre-snap value.
    agent.restore_price_head(saved)
    bias_restored = agent.auction_policy.price_head.bias.detach().clone()
    assert torch.allclose(bias_restored, bias_before)


def test_snap_pulls_deterministic_action_toward_fundamental_anchor():
    """A high-basin policy (bias near +0.7 raw → ~218 EUR/t deterministic
    output) must, after snap, emit a deterministic price near the
    fundamental anchor (~67 EUR/t for default MAC=48 × mult=1.4).
    """
    agent = _make_agent(seed=0)
    device = next(agent.auction_policy.parameters()).device
    obs = torch.zeros(1, 22, device=device)

    with torch.no_grad():
        agent.auction_policy.price_head.bias.fill_(0.7)
        action_high, _, _ = agent.auction_policy.act(obs, deterministic=True)
    price_high = float(action_high[0, 0].item())

    saved = agent.snap_price_head_to_anchor(year=0)
    with torch.no_grad():
        action_snap, _, _ = agent.auction_policy.act(obs, deterministic=True)
    price_snap = float(action_snap[0, 0].item())

    expected_anchor = 48.0 * 1.4  # 67.2 EUR/t
    assert price_high > 180.0, (
        f"High-basin proxy bias did not produce a high deterministic price "
        f"({price_high:.1f}); the test setup is wrong"
    )
    assert abs(price_snap - expected_anchor) < 20.0, (
        f"After snap, deterministic price {price_snap:.1f} is too far from "
        f"the fundamental anchor {expected_anchor:.1f} EUR/t"
    )

    # Restore returns to high-basin price.
    agent.restore_price_head(saved)
    with torch.no_grad():
        action_restored, _, _ = agent.auction_policy.act(obs, deterministic=True)
    price_restored = float(action_restored[0, 0].item())
    assert abs(price_restored - price_high) < 1e-3, (
        f"Restored price {price_restored:.1f} != pre-snap price {price_high:.1f}"
    )


def test_restore_price_head_with_none_is_noop():
    agent = _make_agent(seed=0)
    bias_before = agent.auction_policy.price_head.bias.detach().clone()
    agent.restore_price_head(None)
    assert torch.allclose(agent.auction_policy.price_head.bias, bias_before)


# ---------------------------------------------------------------------------
# Per-agent RNG for ε-greedy exploration
# ---------------------------------------------------------------------------

def test_epsilon_greedy_uses_per_agent_rng_not_global_numpy():
    """With identical agents, identical observations, and ``epsilon=1`` (always
    explore), the ε-greedy exploration draw must be independent of the global
    ``np.random`` state — it must depend only on the per-agent ``self._rng``
    (and on torch's RNG, which is used only for the always-sampled policy
    distribution and is held fixed across calls below).
    """
    obs1 = np.zeros(22, dtype=np.float32)

    a0 = _make_agent(agent_id=0, seed=0)
    a0_repro = _make_agent(agent_id=0, seed=0)

    # Hold torch RNG fixed so the policy-sample portion of each call is
    # identical; any remaining variance must come from numpy RNG paths.
    torch.manual_seed(7)
    np.random.seed(11111)
    act_a, _, _ = a0.select_auction_action(obs1, deterministic=False, epsilon=1.0,
                                           current_year=0)

    torch.manual_seed(7)
    np.random.seed(99999)  # *different* global numpy state
    act_a_repro, _, _ = a0_repro.select_auction_action(
        obs1, deterministic=False, epsilon=1.0, current_year=0)

    assert np.allclose(act_a, act_a_repro), (
        "ε-greedy draw changed when global np.random state changed; some "
        "exploration call is still consuming from np.random.* rather than "
        "self._rng. Got {!r} vs {!r}".format(act_a, act_a_repro)
    )

    # Different per-agent seeds → different ε-greedy draws (basic sanity).
    a_other = _make_agent(agent_id=0, seed=999)
    torch.manual_seed(7)
    np.random.seed(11111)
    act_other, _, _ = a_other.select_auction_action(
        obs1, deterministic=False, epsilon=1.0, current_year=0)
    assert not np.allclose(act_a, act_other), (
        "Two agents with different per-agent seeds produced identical "
        "ε-greedy draws; exploration may not actually depend on self._rng"
    )


def test_epsilon_greedy_secondary_uses_per_agent_rng():
    """Same property as the auction case, for select_secondary_action."""
    obs2 = np.zeros(29, dtype=np.float32)

    a0 = _make_agent(agent_id=0, seed=0)
    a0_repro = _make_agent(agent_id=0, seed=0)

    torch.manual_seed(7)
    np.random.seed(11111)
    act_a, _, _ = a0.select_secondary_action(obs2, deterministic=False, epsilon=1.0)

    torch.manual_seed(7)
    np.random.seed(99999)  # different global state
    act_a_repro, _, _ = a0_repro.select_secondary_action(
        obs2, deterministic=False, epsilon=1.0)

    assert np.allclose(act_a, act_a_repro), (
        "Secondary ε-greedy draw changed under different global np.random "
        "state; some np.random call leaked into the code path. "
        "Got {!r} vs {!r}".format(act_a, act_a_repro)
    )

