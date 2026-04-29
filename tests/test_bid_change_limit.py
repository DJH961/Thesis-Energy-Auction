"""
test_bid_change_limit.py
========================
Tests for the bid price change limit (PCL) feature introduced in v8.4.

Covers:
  - bid_change_limit.enabled=false → no clipping applied
  - Year 0 always unconstrained even when enabled
  - Year 1+: bid price clamped to [price_ma3 - value, price_ma3 + value]
  - _last_bid_price_clip records signed deviation (negative if clipped down)
  - _last_bid_qty_clip_ratio, _last_invest_clip_ratio initialized to 1.0
  - Obs dim [38] = pcl_headroom_norm in [0, 1]
  - Obs dim [39] = PCL bid price clip signal, signed
  - Obs dim [40] = budget price clip signal, signed
  - Obs dim [41] = bid qty clip ratio in [0, 1]
  - Obs dim [42] = invest frac clip ratio in [0, 1]
  - Phase 2 obs dim [base+11] = sec qty clip ratio
  - obs_dim_phase1 = 43 (no opp modeling) or 43 + 7*(N-1)
  - obs_dim_phase2 = obs_dim_phase1 + 12
"""

import sys
import os
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def _load_config(**overrides):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["warm_start"]["enabled"] = False
    cfg["uncertainty"]["enabled"] = False
    cfg["construction_jitter"]["enabled"] = False
    cfg["companies"]["n_bot_agents"] = 0
    cfg["opponent_modeling"]["enabled"] = False
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _make_env(seed=42, bcl_enabled=True, bcl_value=75.0, **overrides):
    cfg = _load_config(**overrides)
    cfg["auction"]["bid_change_limit"] = {"enabled": bcl_enabled, "value": bcl_value}
    env = ETSEnvironment(cfg, seed=seed)
    env.reset(seed=seed)
    return env


def _run_auction_step(env, bid_price, qty_mult=1.0, invest_frac=0.0):
    """Run one auction step with identical bids for all agents."""
    n = env.n_agents
    aa = np.zeros((n, 6), dtype=np.float32)
    aa[:, 0] = bid_price
    aa[:, 1] = qty_mult
    aa[:, 2] = invest_frac
    aa[:, 3:] = [0.0, 0.0, 1.0]
    env.step_auction(aa)


def _run_full_year(env, bid_price=80.0, qty_mult=1.0):
    """Run one full year (auction + secondary) and return rewards."""
    _run_auction_step(env, bid_price, qty_mult)
    n = env.n_agents
    sa = np.zeros((n, 2), dtype=np.float32)
    sa[:, 0] = env._phase1_clearing_price
    _, rewards, _, _, _ = env.step_secondary(sa)
    return rewards


# ---------------------------------------------------------------------------
# Toggle: disabled → no clipping
# ---------------------------------------------------------------------------

class TestBCLDisabled:

    def test_disabled_no_clip_applied(self):
        """With enabled=false, extreme bids are not clipped at year 1+."""
        env = _make_env(bcl_enabled=False)
        # Run year 0 first
        _run_full_year(env, bid_price=80.0)
        # Now year 1: submit extreme high bid
        _run_auction_step(env, bid_price=240.0)
        # last_bid_price_clip should be zero (no clipping happened)
        for i in range(env.n_agents):
            assert env._last_bid_price_clip[i] == pytest.approx(0.0, abs=1e-6), (
                f"Agent {i}: expected no clip with BCL disabled, "
                f"got {env._last_bid_price_clip[i]:.4f}"
            )

    def test_disabled_pcl_ceiling_equals_price_max(self):
        """With BCL disabled, _pcl_ceiling is price_max."""
        env = _make_env(bcl_enabled=False)
        price_max = float(env.config["auction"]["price_max"])
        _run_full_year(env, bid_price=80.0)
        _run_auction_step(env, bid_price=80.0)
        assert env._pcl_ceiling == pytest.approx(price_max, abs=1e-6)


# ---------------------------------------------------------------------------
# Year 0: always unconstrained
# ---------------------------------------------------------------------------

class TestBCLYear0Anchored:
    """v8.5.3: BCL is now active at year 0, anchored on
    `max(price_ma3, fundamental_anchor) ± value`. Previously year 0 was
    unconstrained and policies routinely emitted price_max (250 EUR/t),
    polluting MA3/AR(1) for the rest of the episode."""

    def test_year0_extreme_bid_is_clipped(self):
        """An extreme year-0 bid (240 EUR/t) is clipped down to ma3+value."""
        env = _make_env(bcl_enabled=True, bcl_value=50.0)
        assert env.current_year == 0
        _run_auction_step(env, bid_price=240.0)
        # With warm_start disabled and no prior history, _compute_price_ma3
        # falls back to expected_price (= _price_initial = 70). Anchor is
        # ~57. ref = max(70, 57) = 70. hi = 70 + 50 = 120. So a 240 bid
        # is clipped down by ≈ 120.
        for i in range(env.n_agents):
            clip = float(env._last_bid_price_clip[i])
            assert clip < -50.0, (
                f"Agent {i}: BCL should clip extreme year-0 bid down by ≥50, "
                f"got clip={clip:.4f}"
            )

    def test_year0_in_range_bid_is_not_clipped(self):
        """A reasonable year-0 bid inside [ref-value, ref+value] is unchanged."""
        env = _make_env(bcl_enabled=True, bcl_value=50.0)
        assert env.current_year == 0
        _run_auction_step(env, bid_price=80.0)  # well within [20, 120]
        for i in range(env.n_agents):
            assert env._last_bid_price_clip[i] == pytest.approx(0.0, abs=1e-6), (
                f"Agent {i}: in-range year-0 bid should not be clipped, "
                f"got clip={env._last_bid_price_clip[i]:.4f}"
            )

    def test_year0_clip_arrays_initialized_to_zero_or_one(self):
        """After reset, bid_price_clip=0, qty and invest clip ratios=1."""
        env = _make_env()
        np.testing.assert_array_equal(
            env._last_bid_price_clip, np.zeros(env.n_total),
            err_msg="_last_bid_price_clip should be 0 after reset"
        )
        np.testing.assert_array_equal(
            env._last_bid_qty_clip_ratio, np.ones(env.n_total),
            err_msg="_last_bid_qty_clip_ratio should be 1 after reset"
        )
        np.testing.assert_array_equal(
            env._last_invest_clip_ratio, np.ones(env.n_total),
            err_msg="_last_invest_clip_ratio should be 1 after reset"
        )
        np.testing.assert_array_equal(
            env._last_sec_qty_clip_ratio, np.ones(env.n_total),
            err_msg="_last_sec_qty_clip_ratio should be 1 after reset"
        )


# ---------------------------------------------------------------------------
# Year 1+: bid price clamped to [ma3 - value, ma3 + value]
# ---------------------------------------------------------------------------

class TestBCLClippingYear1Plus:

    def test_bid_above_ceiling_clipped_down(self):
        """A bid price above ma3 + bcl_value is clipped down to the ceiling."""
        bcl_value = 50.0
        env = _make_env(bcl_enabled=True, bcl_value=bcl_value)
        # Establish a price history near 80 €/t
        _run_full_year(env, bid_price=80.0)
        # price_ma3 should be ~80; ceiling = min(80 + 50, price_max) = 130
        ma3 = env._compute_price_ma3()
        price_max = float(env.config["auction"]["price_max"])
        price_min = float(env.config["auction"]["price_min"])
        expected_hi = float(np.clip(ma3 + bcl_value, price_min, price_max))

        extreme_bid = expected_hi + 20.0  # above ceiling
        _run_auction_step(env, bid_price=extreme_bid)

        for i in range(env.n_agents):
            clip = env._last_bid_price_clip[i]
            # clip = clipped - orig; orig > ceiling → clip < 0
            assert clip <= 0.0 + 1e-6, (
                f"Agent {i}: bid clipped down, clip should be ≤ 0, got {clip:.4f}"
            )

    def test_bid_below_floor_clipped_up(self):
        """A bid price below ma3 - bcl_value is clipped up to the floor."""
        bcl_value = 50.0
        env = _make_env(bcl_enabled=True, bcl_value=bcl_value)
        _run_full_year(env, bid_price=120.0)  # establish history at ~120
        ma3 = env._compute_price_ma3()
        price_max = float(env.config["auction"]["price_max"])
        price_min = float(env.config["auction"]["price_min"])
        expected_lo = float(np.clip(ma3 - bcl_value, price_min, price_max))

        # Reserve price (price_min) might be above the BCL floor, so only test
        # if the low clamp is meaningfully above price_min.
        if expected_lo <= price_min + 1.0:
            pytest.skip("BCL floor equals price_min — test not meaningful in this regime")

        low_bid = expected_lo - 10.0
        _run_auction_step(env, bid_price=low_bid)
        for i in range(env.n_agents):
            clip = env._last_bid_price_clip[i]
            assert clip >= -1e-6, (
                f"Agent {i}: bid clipped up, clip should be ≥ 0, got {clip:.4f}"
            )

    def test_bid_within_window_not_clipped(self):
        """A bid inside the allowed window produces zero clip signal."""
        bcl_value = 50.0
        env = _make_env(bcl_enabled=True, bcl_value=bcl_value)
        _run_full_year(env, bid_price=80.0)
        ma3 = env._compute_price_ma3()
        # Bid exactly at ma3 (safe, no clip)
        _run_auction_step(env, bid_price=ma3)
        for i in range(env.n_agents):
            clip = env._last_bid_price_clip[i]
            assert clip == pytest.approx(0.0, abs=1e-6), (
                f"Agent {i}: bid at ma3 should produce zero clip, got {clip:.4f}"
            )

    def test_pcl_ceiling_set_correctly(self):
        """_pcl_ceiling == min(ma3 + value, price_max) after year 1 auction step."""
        bcl_value = 50.0
        env = _make_env(bcl_enabled=True, bcl_value=bcl_value)
        _run_full_year(env, bid_price=80.0)
        ma3 = env._compute_price_ma3()
        price_max = float(env.config["auction"]["price_max"])
        price_min = float(env.config["auction"]["price_min"])
        _run_auction_step(env, bid_price=80.0)
        expected = float(np.clip(ma3 + bcl_value, price_min, price_max))
        assert env._pcl_ceiling == pytest.approx(expected, abs=0.5), (
            f"Expected _pcl_ceiling={expected:.2f}, got {env._pcl_ceiling:.2f}"
        )


# ---------------------------------------------------------------------------
# Clip signal magnitude
# ---------------------------------------------------------------------------

class TestBCLClipSignalMagnitude:

    def test_clip_signal_equals_clipped_minus_orig(self):
        """_last_bid_price_clip[i] = clipped_bid - original_bid (exact)."""
        bcl_value = 50.0
        env = _make_env(bcl_enabled=True, bcl_value=bcl_value)
        _run_full_year(env, bid_price=80.0)
        ma3 = env._compute_price_ma3()
        price_max = float(env.config["auction"]["price_max"])
        price_min = float(env.config["auction"]["price_min"])
        ceiling = float(np.clip(ma3 + bcl_value, price_min, price_max))

        # Bid well above ceiling
        orig_bid = ceiling + 30.0
        _run_auction_step(env, bid_price=orig_bid)

        expected_clip = ceiling - orig_bid  # negative value
        for i in range(env.n_agents):
            assert env._last_bid_price_clip[i] == pytest.approx(expected_clip, abs=0.5), (
                f"Agent {i}: expected clip={expected_clip:.2f}, got {env._last_bid_price_clip[i]:.2f}"
            )

    def test_qty_clip_ratio_bounded_zero_to_one(self):
        """_last_bid_qty_clip_ratio is always in [0, 1] after auction step."""
        env = _make_env()
        for _ in range(3):
            _run_auction_step(env, bid_price=80.0, qty_mult=2.0)
            for i in range(env.n_agents):
                r = env._last_bid_qty_clip_ratio[i]
                assert 0.0 <= r <= 1.0 + 1e-6, (
                    f"Agent {i}: _last_bid_qty_clip_ratio={r:.4f} outside [0,1]"
                )
            if env.episode_done:
                break

    def test_invest_clip_ratio_bounded_zero_to_one(self):
        """_last_invest_clip_ratio is always in [0, 1] after auction step."""
        env = _make_env()
        for _ in range(3):
            _run_auction_step(env, bid_price=80.0, invest_frac=0.5)
            for i in range(env.n_agents):
                r = env._last_invest_clip_ratio[i]
                assert 0.0 <= r <= 1.0 + 1e-6, (
                    f"Agent {i}: _last_invest_clip_ratio={r:.4f} outside [0,1]"
                )
            if env.episode_done:
                break

    def test_sec_qty_clip_ratio_bounded(self):
        """_last_sec_qty_clip_ratio is always in [-1, 1] after secondary step."""
        env = _make_env()
        for _ in range(3):
            _run_full_year(env, bid_price=80.0, qty_mult=1.0)
            for i in range(env.n_agents):
                r = env._last_sec_qty_clip_ratio[i]
                assert -1.0 - 1e-6 <= r <= 1.0 + 1e-6, (
                    f"Agent {i}: _last_sec_qty_clip_ratio={r:.4f} outside [-1,1]"
                )
            if env.episode_done:
                break


# ---------------------------------------------------------------------------
# Observation dimensions
# ---------------------------------------------------------------------------

class TestObsDimensions:

    def test_obs_dim_phase1_base_43(self):
        """obs_dim_phase1 == 43 when opponent modeling is disabled."""
        env = _make_env()
        assert env.companies[0].obs_dim_phase1 == 43, (
            f"Expected obs_dim_phase1=43, got {env.companies[0].obs_dim_phase1}"
        )

    def test_obs_dim_phase1_with_opp_modeling(self):
        """obs_dim_phase1 == 43 + 7*(n_total-1) with opponent modeling enabled."""
        cfg = _load_config()
        cfg["opponent_modeling"]["enabled"] = True
        cfg["auction"]["bid_change_limit"] = {"enabled": True, "value": 50.0}
        env = ETSEnvironment(cfg, seed=42)
        env.reset(seed=42)
        n = env.n_total
        opp_dims = cfg.get("opponent_obs", {}).get("dims_per_opponent", 7)
        expected = 43 + opp_dims * (n - 1)
        assert env.companies[0].obs_dim_phase1 == expected, (
            f"Expected obs_dim_phase1={expected}, got {env.companies[0].obs_dim_phase1}"
        )

    def test_obs_dim_phase2_equals_phase1_plus_12(self):
        """obs_dim_phase2 == obs_dim_phase1 + 12."""
        env = _make_env()
        p1 = env.companies[0].obs_dim_phase1
        p2 = env.companies[0].obs_dim_phase2
        assert p2 == p1 + 12, (
            f"Expected obs_dim_phase2={p1 + 12}, got {p2}"
        )

    def test_phase1_obs_shape_matches_dim(self):
        """Actual phase 1 observation vector length matches obs_dim_phase1."""
        env = _make_env()
        _run_auction_step(env, bid_price=80.0)
        obs = env._get_obs_phase1()
        expected_dim = env.companies[0].obs_dim_phase1
        assert obs.shape[1] == expected_dim, (
            f"Phase 1 obs shape {obs.shape} does not match obs_dim_phase1={expected_dim}"
        )

    def test_phase2_obs_shape_matches_dim(self):
        """Actual phase 2 observation vector length matches obs_dim_phase2."""
        env = _make_env()
        _run_full_year(env, bid_price=80.0)
        # obs_phase2 generated during step_secondary — check via get_observation_phase2
        company = env.companies[0]
        obs_p1 = env._get_obs_phase1()[0]
        obs_p2 = company.get_observation_phase2(
            obs_p1,
            allocation=1.0,
            clearing_price=80.0,
            emissions=2.5,
            banked=0.5,
        )
        expected_dim = company.obs_dim_phase2
        assert len(obs_p2) == expected_dim, (
            f"Phase 2 obs length {len(obs_p2)} does not match obs_dim_phase2={expected_dim}"
        )


# ---------------------------------------------------------------------------
# Obs dim [38]-[42] values
# ---------------------------------------------------------------------------

class TestObsClipDimValues:

    def test_obs_38_pcl_headroom_norm_in_unit_interval(self):
        """Obs dim [38] (pcl_headroom_norm) should be in [0, 1]."""
        env = _make_env(bcl_enabled=True, bcl_value=50.0)
        _run_full_year(env, bid_price=80.0)
        _run_auction_step(env, bid_price=80.0)
        obs = env._get_obs_phase1()
        for i in range(env.n_agents):
            val = float(obs[i, 38])
            assert 0.0 <= val <= 1.0 + 1e-6, (
                f"Agent {i}: obs[38]={val:.4f} outside [0,1]"
            )

    def test_obs_39_bid_price_clip_signal_range(self):
        """Obs dim [39] (PCL bid price clip signal) is in [-1, 1]."""
        env = _make_env(bcl_enabled=True, bcl_value=50.0)
        _run_full_year(env, bid_price=80.0)
        _run_auction_step(env, bid_price=240.0)  # extreme bid → clipped
        obs = env._get_obs_phase1()
        for i in range(env.n_agents):
            val = float(obs[i, 39])
            assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6, (
                f"Agent {i}: obs[39]={val:.4f} outside [-1,1]"
            )

    def test_obs_39_negative_when_bid_clipped_down(self):
        """When bid is clipped downward, obs[39] should be ≤ 0."""
        env = _make_env(bcl_enabled=True, bcl_value=50.0)
        _run_full_year(env, bid_price=80.0)
        ma3 = env._compute_price_ma3()
        price_max = float(env.config["auction"]["price_max"])
        price_min = float(env.config["auction"]["price_min"])
        ceiling = float(np.clip(ma3 + 50.0, price_min, price_max))
        _run_auction_step(env, bid_price=ceiling + 30.0)
        obs = env._get_obs_phase1()
        for i in range(env.n_agents):
            val = float(obs[i, 39])
            assert val <= 0.0 + 1e-6, (
                f"Agent {i}: obs[39]={val:.4f} should be ≤ 0 when bid clipped down"
            )

    def test_obs_40_budget_price_clip_signal_range(self):
        """Obs dim [40] (budget price clip signal) is in [-1, 1]."""
        env = _make_env()
        _run_auction_step(env, bid_price=80.0)
        obs = env._get_obs_phase1()
        for i in range(env.n_agents):
            val = float(obs[i, 40])
            assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6, (
                f"Agent {i}: obs[40]={val:.4f} outside [-1,1]"
            )

    def test_obs_41_qty_clip_ratio_in_unit_interval(self):
        """Obs dim [41] (bid qty clip ratio) is in [0, 1]."""
        env = _make_env()
        _run_auction_step(env, bid_price=80.0, qty_mult=2.0)
        obs = env._get_obs_phase1()
        for i in range(env.n_agents):
            val = float(obs[i, 41])
            assert 0.0 <= val <= 1.0 + 1e-6, (
                f"Agent {i}: obs[41]={val:.4f} outside [0,1]"
            )

    def test_obs_42_invest_clip_ratio_in_unit_interval(self):
        """Obs dim [42] (invest frac clip ratio) is in [0, 1]."""
        env = _make_env()
        _run_auction_step(env, bid_price=80.0, invest_frac=0.5)
        obs = env._get_obs_phase1()
        for i in range(env.n_agents):
            val = float(obs[i, 42])
            assert 0.0 <= val <= 1.0 + 1e-6, (
                f"Agent {i}: obs[42]={val:.4f} outside [0,1]"
            )

    def test_obs_dim39_zero_when_no_clip(self):
        """When bid is within window, obs[39] = 0.0."""
        env = _make_env(bcl_enabled=True, bcl_value=50.0)
        _run_full_year(env, bid_price=80.0)
        ma3 = env._compute_price_ma3()
        _run_auction_step(env, bid_price=ma3)
        obs = env._get_obs_phase1()
        for i in range(env.n_agents):
            val = float(obs[i, 39])
            assert val == pytest.approx(0.0, abs=1e-5), (
                f"Agent {i}: obs[39]={val:.6f} should be 0 when bid not clipped"
            )

    def test_phase2_obs_dim_base_plus11_sec_clip(self):
        """Phase 2 obs[base+11] is in [-1, 1] after a secondary step."""
        env = _make_env()
        _run_full_year(env, bid_price=80.0)
        company = env.companies[0]
        base = company.obs_dim_phase1
        obs_p1 = env._get_obs_phase1()[0]
        obs_p2 = company.get_observation_phase2(
            obs_p1,
            allocation=1.0,
            clearing_price=80.0,
            emissions=2.5,
            banked=0.5,
            last_sec_qty_clip_ratio=env._last_sec_qty_clip_ratio[0],
        )
        val = float(obs_p2[base + 11])
        assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6, (
            f"Phase 2 obs[base+11]={val:.4f} outside [-1,1]"
        )


# ---------------------------------------------------------------------------
# Reset clears clip state
# ---------------------------------------------------------------------------

class TestBCLResetBehavior:

    def test_reset_clears_clip_arrays(self):
        """After reset(), all clip arrays return to their default (0 or 1) values."""
        env = _make_env(bcl_enabled=True, bcl_value=50.0)
        _run_full_year(env, bid_price=80.0)
        _run_auction_step(env, bid_price=240.0)  # force clip

        env.reset()
        np.testing.assert_array_equal(
            env._last_bid_price_clip, np.zeros(env.n_total)
        )
        np.testing.assert_array_equal(
            env._last_bid_qty_clip_ratio, np.ones(env.n_total)
        )
        np.testing.assert_array_equal(
            env._last_invest_clip_ratio, np.ones(env.n_total)
        )
        np.testing.assert_array_equal(
            env._last_sec_qty_clip_ratio, np.ones(env.n_total)
        )

    def test_reset_restores_pcl_ceiling_to_price_max(self):
        """After reset(), _pcl_ceiling returns to price_max."""
        env = _make_env(bcl_enabled=True, bcl_value=50.0)
        _run_full_year(env, bid_price=80.0)
        env.reset()
        price_max = float(env.config["auction"]["price_max"])
        assert env._pcl_ceiling == pytest.approx(price_max, abs=1e-6)
