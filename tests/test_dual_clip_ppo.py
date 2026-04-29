"""
tests/test_dual_clip_ppo.py
===========================
Tests for the dual-clip PPO surrogate (Ye et al. 2020) used by
``PPOAgent._ppo_clipped_loss``. The standard PPO clipped surrogate is
unbounded below for negative-advantage samples whenever the importance
ratio drifts above ``1+clip_eps``: ``min(r·A, clip(r)·A) = r·A``
becomes a large negative number, so ``-min`` blows up. Dual-clip caps
this with a ``c·A`` floor (c>1).

These tests pin down:
  - Standard PPO clipping is unchanged for adv >= 0.
  - For adv < 0 with extreme ratios, the loss is bounded by ``c·|adv|``.
  - The dual-clip can be disabled with ``dual_clip_c <= 1.0``.
"""

import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.ppo_agent import PPOAgent  # noqa: E402
from scripts.train import build_agents  # noqa: E402
from src.environment.ets_environment import ETSEnvironment  # noqa: E402


def _make_agent(dual_clip_c: float = 3.0) -> PPOAgent:
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Override budget/episodes to keep the smoke construction cheap.
    config["ppo"]["dual_clip_c"] = dual_clip_c
    env = ETSEnvironment(config, seed=0)
    agents = build_agents(env, config, seed=0)
    return agents[0]


def test_dual_clip_unchanged_for_positive_advantage():
    """Positive-advantage rows must use the vanilla PPO clipped surrogate."""
    agent = _make_agent(dual_clip_c=3.0)
    # Ratio above (1 + clip_eps); positive advantage. Clipped surr2 dominates.
    ratio = torch.tensor([[5.0]])
    adv = torch.tensor([[1.0]])
    loss = agent._ppo_clipped_loss(ratio, adv).item()
    expected = -(1.0 + agent.clip_eps) * 1.0  # = -min(5*1, (1+eps)*1) = -(1+eps)
    assert abs(loss - expected) < 1e-6, f"got {loss}, expected {expected}"


def test_dual_clip_bounds_negative_advantage_loss():
    """Negative advantage with huge ratio: loss must be bounded by c*|adv|."""
    agent = _make_agent(dual_clip_c=3.0)
    # Without dual-clip, this would yield loss = -ratio*adv = +1e6.
    ratio = torch.tensor([[1e6]])
    adv = torch.tensor([[-1.0]])
    loss = agent._ppo_clipped_loss(ratio, adv).item()
    # max(min(1e6*-1, (1+eps)*-1), 3*-1) = max(-1e6, -3) = -3
    # -mean = 3.0
    assert abs(loss - 3.0) < 1e-6, f"loss should be bounded by c*|adv|=3, got {loss}"


def test_dual_clip_disabled_reproduces_vanilla_ppo():
    """With dual_clip_c <= 1 the helper must match the standard clipped surrogate."""
    agent = _make_agent(dual_clip_c=0.0)
    ratio = torch.tensor([[1e3]])
    adv = torch.tensor([[-1.0]])
    loss = agent._ppo_clipped_loss(ratio, adv).item()
    # min(1e3*-1, (1+eps)*-1) = -1e3 → loss = 1e3
    assert abs(loss - 1e3) < 1e-3, f"vanilla PPO is unbounded below, got {loss}"


def test_dual_clip_mixed_batch_only_floors_negative_rows():
    """In a mixed-sign batch, only the adv<0 rows are floored."""
    agent = _make_agent(dual_clip_c=3.0)
    ratio = torch.tensor([[1e6], [1e6]])
    adv = torch.tensor([[1.0], [-1.0]])
    loss = agent._ppo_clipped_loss(ratio, adv).item()
    # Row 0 (adv>0): min(1e6, 1+eps) = 1+eps
    # Row 1 (adv<0): max(min(-1e6, -(1+eps)), -3) = -3
    # mean = ((1+eps) + (-3))/2; loss = -mean
    expected = -((1.0 + agent.clip_eps) + (-3.0)) / 2.0
    assert abs(loss - expected) < 1e-6, f"got {loss}, expected {expected}"


def test_log_ratio_clip_caps_displayed_loss():
    """Even if log_ratio is enormous, the displayed loss must remain finite."""
    agent = _make_agent(dual_clip_c=3.0)
    # Sanity: the configured clamp is > 0 and finite.
    assert 0 < agent.log_ratio_clip < 50
    # Loss with extreme ratio (post log_ratio clamp) must be finite.
    big_ratio = torch.tensor([[float(torch.exp(torch.tensor(agent.log_ratio_clip)))]])
    adv = torch.tensor([[-1.0]])
    loss = agent._ppo_clipped_loss(big_ratio, adv).item()
    assert torch.isfinite(torch.tensor(loss))
    # And bounded by the dual-clip floor.
    assert loss <= 3.0 + 1e-6, f"dual-clip floor exceeded: {loss}"
