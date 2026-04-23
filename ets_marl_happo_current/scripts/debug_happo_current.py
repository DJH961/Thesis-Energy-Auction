from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import heuristic_policy
from src.environment.ets_environment import ETSEnvironment
from scripts.train import build_agents

TECH_NAMES = ["coal", "gas", "onshore_wind", "offshore_wind", "solar"]
BUILDABLE_TECH_NAMES = ["onshore_wind", "offshore_wind", "solar"]

# Diagnostic thresholds for price realism checks.
MIN_PRICE_SLOPE_FOR_UPTREND = 0.1
MIN_POSITIVE_SLOPE_EP_SHARE = 0.5
MIN_YOY_UP_SHARE = 0.45
MAX_FLAT_FLOOR_EP_SHARE = 0.5

# Budget guardrails used in diagnostics.
MIN_BUDGET_FOR_HARD_CAP = 1.0


@dataclass
class DebugOutputs:
    detailed_year_summary: pd.DataFrame
    detailed_issues: pd.DataFrame
    scale_episode_summary: pd.DataFrame
    scale_issues: pd.DataFrame
    scale_conclusions: pd.DataFrame


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repeat_items_to_length(items, n):
    """Repeat template items with deep copies until length n is reached."""
    if n <= 0:
        return []
    if not items:
        raise ValueError(
            "Cannot repeat an empty template list. Ensure your config "
            "contains valid template entries for the requested agents."
        )
    return [copy.deepcopy(items[i % len(items)]) for i in range(n)]


def configure_simulation(cfg: dict, n_years: int, n_learning_agents: int, n_bots: int) -> dict:
    """Apply runtime participant/year overrides while preserving config structure."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("simulation", {})["n_years"] = int(n_years)

    companies = cfg.setdefault("companies", {})
    budget = cfg.setdefault("budget", {})
    bots = cfg.setdefault("bots", {})

    base_learn_mixes = companies.get("initial_mix", [])
    base_learn_weights = companies.get("reward_weights", [])
    companies["initial_mix"] = repeat_items_to_length(base_learn_mixes, n_learning_agents) if n_learning_agents > 0 else []
    companies["reward_weights"] = repeat_items_to_length(base_learn_weights, n_learning_agents) if n_learning_agents > 0 else []

    bot_mix_templates = companies.get("bot_initial_mix", []) or base_learn_mixes
    bot_weight_templates = companies.get("bot_reward_weights", []) or base_learn_weights
    companies["bot_initial_mix"] = repeat_items_to_length(bot_mix_templates, n_bots)
    companies["bot_reward_weights"] = repeat_items_to_length(bot_weight_templates, n_bots)

    annual_templates = budget.get("annual_budgets", []) or [800.0]
    capex_templates = budget.get("capex_throughputs", []) or [130.0]
    budget["annual_budgets"] = repeat_items_to_length(annual_templates, n_learning_agents)
    budget["capex_throughputs"] = repeat_items_to_length(capex_templates, n_learning_agents)
    budget["bot_annual_budgets"] = repeat_items_to_length(annual_templates, n_bots)
    budget["bot_capex_throughputs"] = repeat_items_to_length(capex_templates, n_bots)

    if "urgency_denominators" in bots and bots["urgency_denominators"]:
        bots["urgency_denominators"] = repeat_items_to_length(bots["urgency_denominators"], n_bots)

    companies["n_agents"] = int(n_learning_agents)
    companies["n_bot_agents"] = int(n_bots)
    return cfg


def participant_name(env: ETSEnvironment, idx: int) -> str:
    """Map participant index to display name (A1.. for learners, B1.. for bots)."""
    return f"A{idx + 1}" if idx < env.n_agents else f"B{idx - env.n_agents + 1}"


def participant_type(env: ETSEnvironment, idx: int) -> str:
    """Return participant class label used in debug tables."""
    return "learning" if idx < env.n_agents else "bot"


def compute_remaining_budgets(env: ETSEnvironment) -> np.ndarray:
    """Compute per-participant remaining annual budget (non-negative)."""
    return np.array(
        [max(0.0, float(c.annual_budget - c.budget_spent_this_year)) for c in env.companies],
        dtype=float,
    )


def load_checkpoint_agents(config: dict, seed: int, checkpoint_dir: Path):
    """Build learning agents and load per-agent checkpoints."""
    env = ETSEnvironment(config, seed=seed)
    env.reset(seed=seed)
    agents = build_agents(env, config, seed)
    missing = []
    for i, agent in enumerate(agents):
        ckpt = checkpoint_dir / f"agent_{i}_best.pt"
        if ckpt.exists():
            agent.load(str(ckpt))
        else:
            missing.append(i)
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint(s) in '{checkpoint_dir}': "
            f"{', '.join(f'agent_{m}_best.pt' for m in missing)}. "
            "Ensure all learning-agent checkpoints are trained and saved there."
        )
    return agents


def build_learning_auction_actions(
    env: ETSEnvironment,
    config: dict,
    policy_agents=None,
    deterministic: bool = True,
) -> tuple[np.ndarray, str]:
    """Build phase-1 actions for learning agents (heuristic or loaded actor policy)."""
    actions = np.zeros((env.n_agents, 6), dtype=np.float32)
    if env.n_agents == 0:
        return actions, "none"

    if policy_agents is not None:
        obs_phase1 = env._get_obs_phase1()
        for i in range(env.n_agents):
            act, _, _ = policy_agents[i].select_auction_action(
                obs_phase1[i], deterministic=deterministic
            )
            actions[i] = act
        return actions, "checkpoint_actor"

    for i in range(env.n_agents):
        company = env.companies[i]
        current_year = env.current_year
        price_ma3 = env._compute_price_ma3()
        cap_t = env.cap_schedule.get_cap(current_year)
        base_penalty_rate = float(config["penalty"]["rate"])
        inflation_rate = float(env._inflation_rate(current_year))
        this_year_auction_volume = env.cap_schedule.preview_auction_volume(
            current_year,
            clearing_price=env.last_clearing_price,
            price_max=float(config["auction"]["price_max"]),
            penalty_rate=base_penalty_rate,
            inflation_rate=inflation_rate,
            price_ma3=price_ma3,
        )
        unsold_pending = float(getattr(env.cap_schedule, "_unsold_rollover_pending", 0.0))
        defaulted_pending = float(env._defaulted_volume_pending)
        this_year_auction_volume += unsold_pending + defaulted_pending
        max_rollover_mult = float(getattr(env.cap_schedule, "max_rollover_multiplier", 1.5))
        this_year_auction_volume = min(this_year_auction_volume, cap_t * max_rollover_mult)

        actions[i] = heuristic_policy.auction_action(
            company,
            price_ma3,
            current_year,
            config["simulation"]["n_years"],
            config,
            bank=float(env.holdings[i]),
            reserve_price=env._compute_dynamic_reserve(),
            auction_volume=float(this_year_auction_volume),
            cap_t=float(cap_t),
            suspension_remaining=int(env._suspension_remaining[i]),
            suspension_length=int(config["auction"].get("suspension_length", 2)),
            collateral_load_last=float(env._last_collateral_load[i]),
        )
    return actions, "heuristic"


def build_learning_secondary_actions(
    env: ETSEnvironment,
    config: dict,
    obs_phase2: np.ndarray | None = None,
    policy_agents=None,
    deterministic: bool = True,
) -> tuple[np.ndarray, str]:
    """Build phase-2 actions for learning agents (heuristic or loaded actor policy)."""
    actions = np.zeros((env.n_agents, 2), dtype=np.float32)
    if env.n_agents == 0:
        return actions, "none"

    if policy_agents is not None:
        if obs_phase2 is None:
            raise ValueError(
                "obs_phase2 is required when policy_agents are provided. "
                "Call step_auction() first and pass its phase-2 observations."
            )
        for i in range(env.n_agents):
            act, _, _ = policy_agents[i].select_secondary_action(
                obs_phase2[i], deterministic=deterministic
            )
            actions[i] = act
        return actions, "checkpoint_actor"

    for i in range(env.n_agents):
        company = env.companies[i]
        actions[i] = heuristic_policy.secondary_action(
            company,
            bank=float(env.holdings[i]),
            allocation=float(env._phase1_allocations[i]),
            clearing_price=env._phase1_clearing_price,
            config=config,
            current_year=env.current_year,
            n_years=config["simulation"]["n_years"],
        )
    return actions, "heuristic"


def check_year_constraints(
    env: ETSEnvironment,
    config: dict,
    log: dict,
    carry_start: np.ndarray,
    pre_suspension: np.ndarray,
    pre_cash: np.ndarray,
) -> list[dict]:
    """Run year-level accounting/constraint checks and return structured diagnostics."""
    issues = []
    eps = 1e-6
    n = env.n_total

    price_min = float(config["auction"]["price_min"])
    price_max = float(config["auction"]["price_max"])
    qty_lo = float(config["auction"].get("qty_mult_low", 0.3))
    qty_hi = float(config["auction"].get("qty_mult_high", 2.0))
    lev_mult = float(config["auction"].get("leverage_multiplier", 3.0))

    bids_p = np.array(log.get("bid_prices", [0.0] * n), dtype=float)
    bid_qtys = np.array(log.get("bid_quantities", [0.0] * n), dtype=float)
    bid_mult = np.array(log.get("bid_qty_multipliers", [0.0] * n), dtype=float)
    alloc = np.array(log.get("allocations", [0.0] * n), dtype=float)
    payments = np.array(log.get("payments", [0.0] * n), dtype=float)
    trades = np.array(log.get("trade_qtys", [0.0] * n), dtype=float)
    emissions = np.array(log.get("emissions", [0.0] * n), dtype=float)
    shortfalls = np.array(log.get("shortfalls", [0.0] * n), dtype=float)
    end_bank = np.array(log.get("holdings", [0.0] * n), dtype=float)
    collateral_costs = np.array(log.get("collateral_costs", [0.0] * n), dtype=float)

    auction_volume = float(log.get("auction_volume", 0.0))
    defaulted_volume = float(log.get("auction_stats", {}).get("defaulted_volume", 0.0))
    unsold_out = float(log.get("unsold_rollover_out", 0.0))
    alloc_sum = float(np.sum(alloc))
    if abs((alloc_sum + defaulted_volume + unsold_out) - auction_volume) > 1e-3:
        issues.append(
            {
                "type": "market_mass_balance",
                "severity": "hard",
                "detail": (
                    f"alloc+default+unsold="
                    f"{alloc_sum + defaulted_volume + unsold_out:.4f} "
                    f"vs auction_volume={auction_volume:.4f}"
                ),
            }
        )

    for i in range(n):
        name = participant_name(env, i)
        if bids_p[i] < price_min - eps or bids_p[i] > price_max + eps:
            issues.append(
                {
                    "type": "bid_price_bounds",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"bid_price={bids_p[i]:.4f} not in [{price_min}, {price_max}]",
                }
            )
        if env._is_agent_active(i) and pre_suspension[i] <= 0:
            if bid_mult[i] < qty_lo - 1e-3 or bid_mult[i] > qty_hi + 1e-3:
                issues.append(
                    {
                        "type": "bid_multiplier_bounds",
                        "severity": "hard",
                        "participant": name,
                        "detail": f"qty_mult={bid_mult[i]:.4f} not in [{qty_lo}, {qty_hi}]",
                    }
                )
        if pre_suspension[i] > 0 and bid_qtys[i] > 1e-6:
            issues.append(
                {
                    "type": "suspension_not_enforced",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"pre_suspension={pre_suspension[i]}, bid_qty={bid_qtys[i]:.6f}",
                }
            )
        if lev_mult > 0 and bids_p[i] > 1e-6:
            max_q = lev_mult * max(0.0, pre_cash[i]) / bids_p[i]
            if bid_qtys[i] > max_q + 1e-3:
                issues.append(
                    {
                        "type": "leverage_gate_violation",
                        "severity": "warn",
                        "participant": name,
                        "detail": f"bid_qty={bid_qtys[i]:.4f} > max_qty={max_q:.4f}",
                    }
                )
        if alloc[i] - bid_qtys[i] > 1e-5:
            issues.append(
                {
                    "type": "allocation_exceeds_bid",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"allocation={alloc[i]:.6f} > bid_qty={bid_qtys[i]:.6f}",
                }
            )
        clearing = float(log.get("clearing_price", 0.0))
        expected_payment = alloc[i] * clearing
        if abs(expected_payment - payments[i]) > 1e-3:
            issues.append(
                {
                    "type": "payment_consistency",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"payment={payments[i]:.4f} vs alloc*clearing={expected_payment:.4f}",
                }
            )
        pre_comp = float(log["bank_start"][i]) + alloc[i] + trades[i]
        need = emissions[i] + float(carry_start[i])
        expected_shortfall = max(0.0, need - pre_comp)
        if abs(shortfalls[i] - expected_shortfall) > 1e-3:
            issues.append(
                {
                    "type": "shortfall_consistency",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"shortfall={shortfalls[i]:.4f} vs expected={expected_shortfall:.4f}",
                }
            )
        expected_end = max(0.0, pre_comp - need)
        if abs(end_bank[i] - expected_end) > 1e-3:
            issues.append(
                {
                    "type": "ending_bank_consistency",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"end_bank={end_bank[i]:.4f} vs expected={expected_end:.4f}",
                }
            )
        if end_bank[i] < -1e-6:
            issues.append(
                {
                    "type": "negative_holdings",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"end_bank={end_bank[i]:.6f}",
                }
            )
        c = env.companies[i]
        hard_cap_frac = float(config.get("budget", {}).get("hard_cap_fraction", 1.15))
        hard_cap_abs = hard_cap_frac * max(float(c.annual_budget), MIN_BUDGET_FOR_HARD_CAP)
        loan_interest_cost = float(c.compute_green_loan_cost())
        tracked_budget_spend = (
            float(payments[i])
            + float(log["trade_costs"][i])
            + float(log["invest_costs"][i])
            + float(log.get("mac_costs", [0.0] * n)[i])
            + float(log.get("collateral_costs", [0.0] * n)[i])
            + loan_interest_cost
        )
        budget_gap = float(c.budget_spent_this_year) - tracked_budget_spend
        if abs(budget_gap) > 1e-3:
            issues.append(
                {
                    "type": "budget_accounting_gap",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"budget_spent={c.budget_spent_this_year:.4f} vs tracked={tracked_budget_spend:.4f} (gap={budget_gap:.4f})",
                }
            )
        annual_overspend = float(c.budget_spent_this_year - c.annual_budget)
        if annual_overspend > 1e-3:
            issues.append(
                {
                    "type": "annual_budget_overspend",
                    "severity": "soft",
                    "participant": name,
                    "detail": f"spend={c.budget_spent_this_year:.4f} > annual_budget={c.annual_budget:.4f} by {annual_overspend:.4f}",
                }
            )
        if c.budget_spent_this_year > hard_cap_abs + 1e-3:
            issues.append(
                {
                    "type": "budget_hard_cap_exceeded",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"spent={c.budget_spent_this_year:.4f} > hard_cap={hard_cap_abs:.4f}",
                }
            )
        if c.capex_spent_this_year > c.capex_throughput + 1e-3:
            issues.append(
                {
                    "type": "capex_soft_overshoot",
                    "severity": "soft",
                    "participant": name,
                    "detail": f"capex_spent={c.capex_spent_this_year:.4f} > capex_limit={c.capex_throughput:.4f}",
                }
            )
        if collateral_costs[i] < -1e-9:
            issues.append(
                {
                    "type": "negative_collateral_cost",
                    "severity": "hard",
                    "participant": name,
                    "detail": f"collateral_cost={collateral_costs[i]:.6f}",
                }
            )
    return issues


def build_per_participant_df(env: ETSEnvironment, log: dict, carry_start: np.ndarray, learning_action_source: str) -> pd.DataFrame:
    """Create a per-agent debug table with bids, costs, compliance, and budget state."""
    rows = []
    n = env.n_total
    for i in range(n):
        c = env.companies[i]
        tech = int(log.get("invest_tech_choices", [0] * n)[i])
        tech_name = BUILDABLE_TECH_NAMES[tech] if 0 <= tech < len(BUILDABLE_TECH_NAMES) else str(tech)
        start_bank = float(log["bank_start"][i])
        allocation = float(log["allocations"][i])
        sec_trade = float(log["trade_qtys"][i])
        pre_comp = start_bank + allocation + sec_trade
        need = float(log["emissions"][i]) + float(carry_start[i])
        rows.append(
            {
                "participant": participant_name(env, i),
                "type": participant_type(env, i),
                "learning_action_source": learning_action_source if i < env.n_agents else "bot_heuristic",
                "start_bank_mt": start_bank,
                "carry_in_mt": float(carry_start[i]),
                "est_need_mt": float(log.get("estimate_needs", [0] * n)[i]),
                "bid_qty_mult": float(log.get("bid_qty_multipliers", [0] * n)[i]),
                "bid_price": float(log.get("bid_prices", [0] * n)[i]),
                "bid_qty_mt": float(log.get("bid_quantities", [0] * n)[i]),
                "alloc_mt": allocation,
                "payment_meur": float(log["payments"][i]),
                "sec_price": float(log.get("sec_price_mults", [0] * n)[i]),
                "sec_action_qty": float(log.get("sec_qty_actions", [0] * n)[i]),
                "sec_trade_mt": sec_trade,
                "sec_trade_cost": float(log["trade_costs"][i]),
                "invest_frac": float(log.get("invest_fracs", [0] * n)[i]),
                "invest_tech": tech_name,
                "invest_cost": float(log["invest_costs"][i]),
                "mac_reduction_mt": float(log.get("mac_reductions", [0] * n)[i]),
                "mac_cost_meur": float(log.get("mac_costs", [0] * n)[i]),
                "collateral_cost_meur": float(log.get("collateral_costs", [0] * n)[i]),
                "emissions_mt": float(log["emissions"][i]),
                "pre_compliance_allowances_mt": pre_comp,
                "total_need_mt": need,
                "shortfall_mt": float(log["shortfalls"][i]),
                "penalty_meur": float(log["penalties"][i]),
                "end_bank_mt": float(log["holdings"][i]),
                "carry_next_mt": float(c._carry_forward),
                "reward": float(log["rewards"][i]),
                "budget_spent": float(c.budget_spent_this_year),
                "annual_budget": float(c.annual_budget),
                "annual_budget_overspend": float(max(0.0, c.budget_spent_this_year - c.annual_budget)),
                "capex_spent": float(c.capex_spent_this_year),
                "capex_limit": float(c.capex_throughput),
                "green_frac_pct": 100.0 * float(c.green_frac),
            }
        )
    return pd.DataFrame(rows)


def run_detailed_episode(
    config: dict,
    seed: int = 42,
    print_output: bool = True,
    policy_agents=None,
    deterministic_policy: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute one episode with full tracing and return (year_summary_df, issues_df)."""
    env = ETSEnvironment(config, seed=seed)
    env.reset(seed=seed)
    issues_all = []
    summary_rows = []
    for year in range(config["simulation"]["n_years"]):
        pre_suspension = env._suspension_remaining.copy()
        pre_cash = compute_remaining_budgets(env)
        carry_start = np.array([float(c._carry_forward) for c in env.companies], dtype=float)

        auc_actions, learning_action_source = build_learning_auction_actions(
            env,
            config,
            policy_agents=policy_agents,
            deterministic=deterministic_policy,
        )
        obs_phase2, _ = env.step_auction(auc_actions)
        sec_actions, _ = build_learning_secondary_actions(
            env,
            config,
            obs_phase2=obs_phase2,
            policy_agents=policy_agents,
            deterministic=deterministic_policy,
        )
        _, _, terminated, truncated, info = env.step_secondary(sec_actions)
        log = info["year_log"]
        year_issues = check_year_constraints(env, config, log, carry_start, pre_suspension, pre_cash)
        issues_all.extend([{"year": year + 1, **x} for x in year_issues])
        summary_rows.append(
            {
                "year": year + 1,
                "clearing_price": float(log.get("clearing_price", np.nan)),
                "secondary_clearing": float(log.get("secondary_clearing", np.nan)),
                "auction_volume": float(log.get("auction_volume", np.nan)),
                "total_allocated": float(log.get("auction_stats", {}).get("total_allocated", np.nan)),
                "defaults": int(log.get("auction_stats", {}).get("defaults", 0)),
                "defaulted_volume": float(log.get("auction_stats", {}).get("defaulted_volume", 0.0)),
                "unsold_out": float(log.get("unsold_rollover_out", 0.0)),
                "learning_action_source": learning_action_source,
                "learning_bid_mean": float(np.mean(log.get("bid_prices", [])[:env.n_agents])) if env.n_agents > 0 else np.nan,
                "learning_bid_std": float(np.std(log.get("bid_prices", [])[:env.n_agents])) if env.n_agents > 0 else np.nan,
                "learning_invest_mean": float(np.mean(log.get("invest_fracs", [])[:env.n_agents])) if env.n_agents > 0 else np.nan,
                "learning_annual_budget_overspend_total": float(
                    np.sum([
                        max(0.0, env.companies[i].budget_spent_this_year - env.companies[i].annual_budget)
                        for i in range(env.n_agents)
                    ])
                ),
                "issues_this_year": int(len(year_issues)),
                "collateral_clip_events_total": int(sum(env._collateral_clip_events.values())),
            }
        )
        if print_output:
            year_summary = summary_rows[-1]
            print(
                f"[Year {year + 1:02d}] "
                f"clearing={year_summary['clearing_price']:.3f} "
                f"secondary={year_summary['secondary_clearing']:.3f} "
                f"defaults={year_summary['defaults']} issues={len(year_issues)}"
            )
            if year_issues:
                for i in year_issues:
                    who = i.get("participant", "GLOBAL")
                    print(f"  - [{i['type']}] {who}: {i['detail']}")
            else:
                print("  - no constraint issues")
            _df = build_per_participant_df(env, log, carry_start, learning_action_source=learning_action_source)
            print(_df.to_string(index=False))
            if env.n_agents > 0:
                reward_rows = []
                for ai in range(env.n_agents):
                    rc = env._last_reward_channels.get(ai, {})
                    arc = env._last_auction_reward_channels.get(ai, {})
                    reward_rows.append(
                        {
                            "agent": participant_name(env, ai),
                            "cost_norm": rc.get("cost_norm", np.nan),
                            "penalty_norm": rc.get("penalty_norm", np.nan),
                            "esg_signal": rc.get("esg_signal", np.nan),
                            "base_reward": rc.get("base_reward", np.nan),
                            "auc_cost": arc.get("auction_cost", np.nan),
                            "auc_collateral": arc.get("collateral_cost", np.nan),
                            "auc_invest": arc.get("investment_cost", np.nan),
                            "auc_opex_delta": arc.get("opex_delta", np.nan),
                            "auc_mac_cost": arc.get("mac_cost", np.nan),
                            "auc_reward": arc.get("auction_reward", np.nan),
                        }
                    )
                print(pd.DataFrame(reward_rows).to_string(index=False))
        if terminated or truncated:
            break
    return pd.DataFrame(summary_rows), pd.DataFrame(issues_all)


def run_scale_diagnostics(
    config: dict,
    n_episodes: int = 30,
    seed_start: int = 100,
    policy_agents=None,
    deterministic_policy: bool = True,
    learning_policy_mode: str = "heuristic",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run many episodes and return (episode_summary_df, issue_counts_df, conclusions_df)."""
    rows = []
    issue_counter = Counter()
    hard_issue_counter = Counter()
    for ep in range(n_episodes):
        env = ETSEnvironment(config, seed=seed_start + ep)
        env.reset(seed=seed_start + ep)
        ep_defaults = 0
        ep_fails = 0
        ep_prices = []
        ep_sec = []
        ep_issues_total = 0
        ep_issues_hard = 0
        learning_overspend_count = 0
        bot_overspend_count = 0
        learning_overspend_total = 0.0
        bot_overspend_total = 0.0
        for _ in range(config["simulation"]["n_years"]):
            pre_suspension = env._suspension_remaining.copy()
            pre_cash = compute_remaining_budgets(env)
            carry_start = np.array([float(c._carry_forward) for c in env.companies], dtype=float)
            auc_actions, _ = build_learning_auction_actions(
                env, config, policy_agents=policy_agents, deterministic=deterministic_policy
            )
            obs_phase2, _ = env.step_auction(auc_actions)
            sec_actions, _ = build_learning_secondary_actions(
                env,
                config,
                obs_phase2=obs_phase2,
                policy_agents=policy_agents,
                deterministic=deterministic_policy,
            )
            _, _, terminated, truncated, info = env.step_secondary(sec_actions)
            log = info["year_log"]
            issues = check_year_constraints(env, config, log, carry_start, pre_suspension, pre_cash)
            ep_issues_total += len(issues)
            for it in issues:
                issue_counter[it["type"]] += 1
                if it.get("severity", "hard") == "hard":
                    hard_issue_counter[it["type"]] += 1
                    ep_issues_hard += 1
            stats = log.get("auction_stats", {})
            ep_defaults += int(stats.get("defaults", 0))
            ep_fails += int(bool(stats.get("auction_failed", False)))
            ep_prices.append(float(log.get("clearing_price", np.nan)))
            ep_sec.append(float(log.get("secondary_clearing", np.nan)))
            for i, c in enumerate(env.companies):
                overs = max(0.0, float(c.budget_spent_this_year - c.annual_budget))
                if i < env.n_agents:
                    learning_overspend_total += overs
                    if overs > 1e-6:
                        learning_overspend_count += 1
                else:
                    bot_overspend_total += overs
                    if overs > 1e-6:
                        bot_overspend_count += 1
            if terminated or truncated:
                break
        prices = np.array(ep_prices, dtype=float)
        sec = np.array(ep_sec, dtype=float)
        years = len(ep_prices)
        if years > 1:
            x = np.arange(years, dtype=float)
            price_slope = float(np.polyfit(x, prices, 1)[0])
            yoy = np.diff(prices)
            yoy_abs_max = float(np.max(np.abs(yoy))) if len(yoy) > 0 else 0.0
            yoy_up_share = float(np.mean(yoy > 0.0)) if len(yoy) > 0 else 0.0
        else:
            price_slope = 0.0
            yoy_abs_max = 0.0
            yoy_up_share = 0.0
        rows.append(
            {
                "episode": ep + 1,
                "seed": seed_start + ep,
                "learning_policy_mode": learning_policy_mode,
                "years": len(ep_prices),
                "defaults_total": ep_defaults,
                "auction_fail_years": ep_fails,
                "price_mean": float(np.nanmean(prices)) if len(prices) else np.nan,
                "price_min": float(np.nanmin(prices)) if len(prices) else np.nan,
                "price_max": float(np.nanmax(prices)) if len(prices) else np.nan,
                "secondary_price_mean": float(np.nanmean(sec)) if len(sec) else np.nan,
                "price_slope_per_year": price_slope,
                "price_yoy_abs_max": yoy_abs_max,
                "price_yoy_up_share": yoy_up_share,
                "issues_total": ep_issues_total,
                "issues_hard": ep_issues_hard,
                "learning_annual_overspend_total": learning_overspend_total,
                "learning_annual_overspend_count": learning_overspend_count,
                "bot_annual_overspend_total": bot_overspend_total,
                "bot_annual_overspend_count": bot_overspend_count,
                "collateral_clip_events_total": int(sum(env._collateral_clip_events.values())),
            }
        )
    ep_df = pd.DataFrame(rows)
    issue_df = (
        pd.DataFrame([{"issue_type": k, "count": v} for k, v in issue_counter.items()])
        .sort_values("count", ascending=False)
        if issue_counter
        else pd.DataFrame(columns=["issue_type", "count"])
    )
    conclusions = []
    if len(ep_df) > 0:
        defaults_sum = int(ep_df["defaults_total"].sum())
        fails_sum = int(ep_df["auction_fail_years"].sum())
        issues_sum = int(ep_df["issues_total"].sum())
        issues_hard_sum = int(ep_df["issues_hard"].sum())
        total_collateral_clip_events = int(ep_df["collateral_clip_events_total"].sum())
        learning_overspend_total = float(ep_df["learning_annual_overspend_total"].sum())
        bot_overspend_total = float(ep_df["bot_annual_overspend_total"].sum())
        mean_price_slope = float(ep_df["price_slope_per_year"].mean())
        positive_slope_share = float(np.mean(ep_df["price_slope_per_year"] > 0.0))
        mean_yoy_up_share = float(ep_df["price_yoy_up_share"].mean())
        pmin = float(ep_df["price_min"].min())
        pmax = float(ep_df["price_max"].max())
        if defaults_sum == 0:
            conclusions.append(("works", "No auction defaults observed across scale diagnostics."))
        else:
            conclusions.append(("does_not_work", f"Observed {defaults_sum} auction defaults across scale diagnostics."))
        if fails_sum == 0:
            conclusions.append(("works", "No auction-failed years observed across scale diagnostics."))
        else:
            conclusions.append(("does_not_work", f"Observed {fails_sum} auction-failed years across scale diagnostics."))
        if pmin >= float(config["auction"]["price_min"]) - 1e-6 and pmax <= float(config["auction"]["price_max"]) + 1e-6:
            conclusions.append(("works", f"Clearing prices stayed within configured bounds [{config['auction']['price_min']}, {config['auction']['price_max']}]."))
        else:
            conclusions.append(("does_not_work", f"Price out-of-bounds observed (min={pmin:.3f}, max={pmax:.3f})."))
        flat_floor_mask = (
            (ep_df["price_max"] - ep_df["price_min"] <= 1e-6)
            & (ep_df["price_min"] <= float(config["auction"]["price_min"]) + 1e-6)
        )
        flat_floor_share = float(np.mean(flat_floor_mask))
        if (
            mean_price_slope > MIN_PRICE_SLOPE_FOR_UPTREND
            and positive_slope_share >= MIN_POSITIVE_SLOPE_EP_SHARE
        ):
            conclusions.append(("works", f"Price trend is generally upward (mean slope={mean_price_slope:.3f}, positive-slope episodes={positive_slope_share:.1%})."))
        else:
            conclusions.append(("needs_review", f"Price rise is not consistently upward (mean slope={mean_price_slope:.3f}, positive-slope episodes={positive_slope_share:.1%})."))
        if mean_yoy_up_share >= MIN_YOY_UP_SHARE:
            conclusions.append(("works", f"Year-to-year price increases appear active (mean up-share={mean_yoy_up_share:.1%})."))
        else:
            conclusions.append(("needs_review", f"Year-to-year price increases are sparse (mean up-share={mean_yoy_up_share:.1%})."))
        if flat_floor_share > MAX_FLAT_FLOOR_EP_SHARE:
            conclusions.append(
                (
                    "does_not_work",
                    (
                        "Price is stuck at the auction floor in "
                        f"{flat_floor_share:.1%} of episodes; this is likely "
                        "not realistic market price formation."
                    ),
                )
            )
        if learning_overspend_total <= 1e-6 and bot_overspend_total <= 1e-6:
            conclusions.append(("works", "No annual-budget overspend observed for either learning agents or bots."))
        else:
            conclusions.append(("needs_review", f"Annual-budget overspend observed: learning={learning_overspend_total:.3f}, bots={bot_overspend_total:.3f}."))
        if issues_hard_sum == 0:
            conclusions.append(("works", "No hard-constraint violations were detected."))
        else:
            conclusions.append(("does_not_work", f"Detected {issues_hard_sum} hard-constraint violations."))
        if issues_sum - issues_hard_sum > 0:
            conclusions.append(("needs_review", f"Observed {issues_sum - issues_hard_sum} soft/warning constraint flags (inspect scale_issues table)."))
        if total_collateral_clip_events == 0:
            conclusions.append(("works", "No collateral clip events observed."))
        else:
            conclusions.append(("needs_review", f"Collateral clip events observed ({total_collateral_clip_events}); review heuristic/environment alignment."))
        if learning_policy_mode == "checkpoint_actor":
            conclusions.append(("works", "Learning-agent decisions came from loaded actor checkpoints (not heuristic fallback)."))
        else:
            conclusions.append(("needs_review", "Learning-agent decisions used heuristics; load checkpoints to directly evaluate learned actor behavior."))
    conc_df = pd.DataFrame(conclusions, columns=["status", "conclusion"])
    return ep_df, issue_df, conc_df


def run_debug_session(
    config_path: Path,
    seed: int = 42,
    n_years: int = 12,
    n_learning_agents: int = 8,
    n_bots: int = 8,
    scale_episodes: int = 30,
    scale_seed_start: int = 100,
    print_detailed: bool = True,
    checkpoint_dir: Path | None = None,
    deterministic_policy: bool = True,
    require_learning_agents: bool = True,
) -> DebugOutputs:
    """Run detailed + scale diagnostics and package all outputs."""
    raw = load_config(config_path)
    cfg = configure_simulation(raw, n_years=n_years, n_learning_agents=n_learning_agents, n_bots=n_bots)
    if require_learning_agents and cfg["companies"].get("n_agents", 0) <= 0:
        raise ValueError(
            "This debug session requires learning agents; set n_learning_agents > 0 "
            "or pass --allow-no-learning-agents."
        )

    policy_agents = None
    learning_policy_mode = "heuristic"
    if checkpoint_dir is not None:
        policy_agents = load_checkpoint_agents(cfg, seed=seed, checkpoint_dir=checkpoint_dir)
        learning_policy_mode = "checkpoint_actor"

    detailed_year_summary, detailed_issues = run_detailed_episode(
        cfg,
        seed=seed,
        print_output=print_detailed,
        policy_agents=policy_agents,
        deterministic_policy=deterministic_policy,
    )
    scale_episode_summary, scale_issues, scale_conclusions = run_scale_diagnostics(
        cfg,
        n_episodes=scale_episodes,
        seed_start=scale_seed_start,
        policy_agents=policy_agents,
        deterministic_policy=deterministic_policy,
        learning_policy_mode=learning_policy_mode,
    )
    return DebugOutputs(
        detailed_year_summary=detailed_year_summary,
        detailed_issues=detailed_issues,
        scale_episode_summary=scale_episode_summary,
        scale_issues=scale_issues,
        scale_conclusions=scale_conclusions,
    )


def main():
    parser = argparse.ArgumentParser(description="Detailed debugging environment for ets_marl_happo_current.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--years", type=int, default=12)
    parser.add_argument("--learning-agents", type=int, default=8)
    parser.add_argument("--bots", type=int, default=8)
    parser.add_argument("--scale-episodes", type=int, default=30)
    parser.add_argument("--scale-seed-start", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Path containing agent_<i>_best.pt checkpoints for learning agents.")
    parser.add_argument("--stochastic-policy", action="store_true", help="Use stochastic actor actions (default is deterministic).")
    parser.add_argument("--allow-no-learning-agents", action="store_true", help="Allow runs with n_learning_agents=0.")
    parser.add_argument("--quiet-detailed", action="store_true", help="Disable detailed per-year prints.")
    parser.add_argument("--export-dir", type=Path, default=None)
    args = parser.parse_args()

    outputs = run_debug_session(
        config_path=args.config,
        seed=args.seed,
        n_years=args.years,
        n_learning_agents=args.learning_agents,
        n_bots=args.bots,
        scale_episodes=args.scale_episodes,
        scale_seed_start=args.scale_seed_start,
        print_detailed=not args.quiet_detailed,
        checkpoint_dir=args.checkpoint_dir,
        deterministic_policy=not args.stochastic_policy,
        require_learning_agents=not args.allow_no_learning_agents,
    )

    print("\n===== SCALE EPISODE SUMMARY =====")
    print(outputs.scale_episode_summary.to_string(index=False))
    if len(outputs.scale_issues) > 0:
        print("\n===== SCALE ISSUES =====")
        print(outputs.scale_issues.to_string(index=False))
    print("\n===== WORKS / DOESN'T =====")
    print(outputs.scale_conclusions.to_string(index=False))

    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        outputs.detailed_year_summary.to_csv(args.export_dir / "detailed_year_summary.csv", index=False)
        outputs.detailed_issues.to_csv(args.export_dir / "detailed_issues.csv", index=False)
        outputs.scale_episode_summary.to_csv(args.export_dir / "scale_episode_summary.csv", index=False)
        outputs.scale_issues.to_csv(args.export_dir / "scale_issues.csv", index=False)
        outputs.scale_conclusions.to_csv(args.export_dir / "scale_conclusions.csv", index=False)
        print(f"\nExports written to: {args.export_dir}")


if __name__ == "__main__":
    main()
