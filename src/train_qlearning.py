"""
train_qlearning.py
==================
Tabular Q-learning training loop for the ETS MARL environment.

Mirrors the env-interaction surface of the PPO/HAPPO trainer
(``scripts/train.py``) so the two algorithms can be compared
apples-to-apples on the same default config:

* Same ``ETSEnvironment`` instance, same warm-start, same Phase-1 →
  Phase-2 cadence per simulated year.
* Same per-seed reset semantics (``seed + episode * 1000``) and
  optional ``--run-tag`` filename infix used by the sweep launcher.
* Same year-log fields exposed in the cumulative CSV (per-agent
  allocations, emissions, shortfalls, penalties, rewards, holdings,
  bid prices, MAC reductions, terminal values, …).
* Same per-episode anchor-invariant ``quality_score`` (and the five
  ``Q_*`` components) computed via ``src.utils.quality_metric`` —
  the same metric the PPO trainer logs.

Usage:
    python src/train_qlearning.py --config configs/default.yaml \\
                                  --ql-config configs/qlearning.yaml \\
                                  [--seed 42 | --seeds 42 123 456] \\
                                  [--run-tag <name>]

If neither ``--seed`` nor ``--seeds`` is given, the loop iterates
over ``simulation.seeds`` from the merged config.
"""

import argparse
import csv
import os
import sys
import time
import datetime
import io
import yaml
import numpy as np

# UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.agents.q_learning_agent import (
    QLearningAgent, StateDiscretizer, ActionProfileMapper)
from src.utils.quality_metric import compute_episode_quality


# Short labels for the auction/secondary action profiles. Used in the
# console table so that "a1/a2 = 1/0" reads as "Mod/Hold" — closer to
# how a HAPPO column communicates strategy than bare profile indices.
_A1_SHORT = ["Cons", "Mod", "Aggr", "Grn", "Fin", "Pnc"]
_A2_SHORT = ["Hold", "Sell", "Buy", "BuyA"]


def _a1_label(idx: int) -> str:
    if idx < 0 or idx >= len(_A1_SHORT):
        return "?"
    return _A1_SHORT[idx]


def _a2_label(idx: int) -> str:
    if idx < 0 or idx >= len(_A2_SHORT):
        return "?"
    return _A2_SHORT[idx]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge override into base (override wins on conflicts)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def _format_hms(seconds: float) -> str:
    return str(datetime.timedelta(seconds=max(0, int(round(float(seconds))))))


def _trend_arrow(seq) -> str:
    """Trailing trend arrow used by HAPPO console: ↑/↓/→ depending on the
    relative change between the first and last value of the per-year series."""
    if not seq or len(seq) < 2:
        return ""
    a, b = float(seq[0]), float(seq[-1])
    if abs(a) < 1e-9:
        rel = b
    else:
        rel = (b - a) / abs(a)
    if rel > 0.05:
        return "  ↑"
    if rel < -0.05:
        return "  ↓"
    return "  →"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ep_fields(n_total_agents: int) -> list:
    """Episode-level CSV header. Mirrors the subset of
    ``training_log_*.csv`` that's environment-derived (i.e. comparable
    between PPO and Q-learning); learner-internal fields like
    ``actor_loss_*``, ``entropy_coef`` etc. are intentionally omitted."""
    fields = [
        "episode", "epsilon",
        "clearing_price_last", "cap_last", "tnac",
        "ep_mean_clearing_price", "mean_price",
        "secondary_volume", "secondary_avg_price",
        "price_start", "price_peak", "price_std",
        # Anchor-invariant quality metric (same as PPO trainer)
        "quality_score", "Q_compliance", "Q_price_realism",
        "Q_saved_carbon", "Q_cost_eff", "Q_volatility",
    ]
    for i in range(n_total_agents):
        fields += [
            f"reward_A{i+1}", f"green_frac_A{i+1}", f"delta_green_A{i+1}",
            f"compliance_rate_A{i+1}", f"shortfall_A{i+1}", f"penalty_A{i+1}",
            f"bid_price_A{i+1}", f"queue_size_A{i+1}",
            f"a1_profile_A{i+1}", f"a2_profile_A{i+1}",
            f"invest_cost_A{i+1}",
        ]
    return fields


def _yr_fields(n_total_agents: int) -> list:
    """Year-level CSV header. Superset overlaps with the PPO
    ``year_log_*.csv`` so analysis notebooks can load either log into
    the same dataframe shape."""
    fields = [
        "episode", "year",
        "cap", "auction_volume", "tnac",
        "clearing_price", "secondary_price", "secondary_volume",
        "msr_reserve", "msr_total_cancelled",
        "msr_withhold_this_year", "msr_release_this_year",
        "inflation_rate", "inflation_factor",
    ]
    for i in range(n_total_agents):
        fields += [
            f"bank_start_A{i+1}", f"alloc_A{i+1}", f"emissions_A{i+1}",
            f"trade_qty_A{i+1}", f"trade_cost_A{i+1}",
            f"green_frac_A{i+1}", f"delta_green_A{i+1}",
            f"shortfall_A{i+1}", f"penalty_A{i+1}",
            f"reward_A{i+1}",
            f"holdings_A{i+1}", f"bank_end_A{i+1}",
            f"invest_cost_A{i+1}", f"collateral_cost_A{i+1}",
            f"bid_price_A{i+1}", f"queue_size_A{i+1}",
            f"mac_reduction_A{i+1}", f"mac_cost_A{i+1}",
            f"terminal_bank_value_A{i+1}", f"terminal_queue_value_A{i+1}",
            f"sec_price_mult_A{i+1}", f"sec_qty_action_A{i+1}",
            f"sec_action_side_A{i+1}",
            f"bid_qty_mult_A{i+1}",
        ]
    return fields


def _yr_row(yl: dict, episode: int, year: int, n_total_agents: int) -> dict:
    """Flatten the env year_log dict into the CSV row format."""
    row = {
        "episode": episode, "year": year,
        "cap": yl.get("cap", 0),
        "auction_volume": yl.get("auction_volume", 0),
        "tnac": yl.get("tnac", 0),
        "clearing_price": yl.get("clearing_price", 0),
        "secondary_price": yl.get("secondary_clearing", 0),
        "secondary_volume": yl.get("secondary_volume", 0),
        "msr_reserve": yl.get("msr_reserve", 0),
        "msr_total_cancelled": yl.get("msr_total_cancelled", 0),
        "msr_withhold_this_year": yl.get("msr_withhold_this_year", 0),
        "msr_release_this_year": yl.get("msr_release_this_year", 0),
        "inflation_rate": yl.get("inflation_rate", 0),
        "inflation_factor": yl.get("inflation_factor", 1.0),
    }

    def _g(key, default=0.0):
        vals = yl.get(key, [default] * n_total_agents)
        return vals if isinstance(vals, list) else [default] * n_total_agents

    bank_start = _g("bank_start")
    allocations = _g("allocations")
    emissions = _g("emissions")
    trade_qtys = _g("trade_qtys")
    trade_costs = _g("trade_costs")
    green_fracs = _g("green_fracs")
    delta_greens = _g("delta_greens")
    shortfalls = _g("shortfalls")
    penalties = _g("penalties")
    rewards = _g("rewards")
    holdings = _g("holdings")
    invest_costs = _g("invest_costs")
    collateral_costs = _g("collateral_costs")
    bid_prices = _g("bid_prices")
    queue_sizes = _g("queue_sizes")
    mac_reductions = _g("mac_reductions")
    mac_costs = _g("mac_costs")
    term_bank = _g("terminal_bank_values")
    term_queue = _g("terminal_queue_values")
    sec_price_mults = _g("sec_price_mults")
    sec_qty_actions = _g("sec_qty_actions")
    sec_action_sides = _g("sec_action_sides")
    bid_qty_mults = _g("bid_qty_multipliers")

    def _at(arr, i, default=0):
        return arr[i] if i < len(arr) else default

    for i in range(n_total_agents):
        bs = float(_at(bank_start, i))
        alloc_i = float(_at(allocations, i))
        emiss_i = float(_at(emissions, i))
        # bank_end ≈ bank_start + alloc - emiss (post-compliance, before
        # secondary settlement is folded back in via `holdings`).
        # For the analysis we want the realised post-everything bank,
        # which the env publishes via `holdings`.
        row[f"bank_start_A{i+1}"] = bs
        row[f"alloc_A{i+1}"] = alloc_i
        row[f"emissions_A{i+1}"] = emiss_i
        row[f"trade_qty_A{i+1}"] = float(_at(trade_qtys, i))
        row[f"trade_cost_A{i+1}"] = float(_at(trade_costs, i))
        row[f"green_frac_A{i+1}"] = float(_at(green_fracs, i))
        row[f"delta_green_A{i+1}"] = float(_at(delta_greens, i))
        row[f"shortfall_A{i+1}"] = float(_at(shortfalls, i))
        row[f"penalty_A{i+1}"] = float(_at(penalties, i))
        row[f"reward_A{i+1}"] = float(_at(rewards, i))
        row[f"holdings_A{i+1}"] = float(_at(holdings, i))
        row[f"bank_end_A{i+1}"] = float(_at(holdings, i))
        row[f"invest_cost_A{i+1}"] = float(_at(invest_costs, i))
        row[f"collateral_cost_A{i+1}"] = float(_at(collateral_costs, i))
        row[f"bid_price_A{i+1}"] = float(_at(bid_prices, i))
        row[f"queue_size_A{i+1}"] = int(_at(queue_sizes, i))
        row[f"mac_reduction_A{i+1}"] = float(_at(mac_reductions, i))
        row[f"mac_cost_A{i+1}"] = float(_at(mac_costs, i))
        row[f"terminal_bank_value_A{i+1}"] = float(_at(term_bank, i))
        row[f"terminal_queue_value_A{i+1}"] = float(_at(term_queue, i))
        row[f"sec_price_mult_A{i+1}"] = float(_at(sec_price_mults, i))
        row[f"sec_qty_action_A{i+1}"] = float(_at(sec_qty_actions, i))
        row[f"sec_action_side_A{i+1}"] = int(_at(sec_action_sides, i))
        row[f"bid_qty_mult_A{i+1}"] = float(_at(bid_qty_mults, i))
    return row


# ---------------------------------------------------------------------------
# Training loop (one seed)
# ---------------------------------------------------------------------------

def train_qlearning(config: dict, ql_config: dict, seed: int,
                    run_tag: str | None = None):
    """Run Q-learning training for a single seed."""

    # Merge Q-learning overrides into base config
    config = merge_configs(config, ql_config)

    ql_params = config.get("qlearning", {})
    n_episodes = ql_params.get("n_episodes", 5000)
    alpha = ql_params.get("alpha", 0.1)
    gamma = ql_params.get("gamma", 0.95)
    eps_start = ql_params.get("epsilon_start", 1.0)
    eps_end = ql_params.get("epsilon_end", 0.05)
    eps_decay_frac = ql_params.get("epsilon_decay_frac", 0.7)
    # ``max(1, ...)`` keeps the linear-decay denominator well-defined
    # even on micro smoke runs (e.g. n_episodes=1, eps_decay_frac=0.7
    # → int rounds to 0). In that degenerate case the loop spends one
    # episode at eps_start and no later episodes exist.
    eps_decay_episodes = max(1, int(n_episodes * eps_decay_frac))

    n_agents = config["companies"]["n_agents"]
    n_bot_agents = config["companies"].get("n_bot_agents", 0)
    n_total_agents = n_agents + n_bot_agents
    n_years = config["simulation"]["n_years"]

    tag_part = f"_{run_tag}" if run_tag else ""

    # Read log_interval up-front so the legend banner can reference it.
    log_interval = config.get("logging", {}).get("log_interval", 50)

    print(f"\n{'═'*70}")
    print(f"  Q-Learning Baseline Training — seed {seed}"
          f"{' [tag=' + run_tag + ']' if run_tag else ''}")
    print(f"  Episodes: {n_episodes}  |  Alpha: {alpha}  |  Gamma: {gamma}")
    print(f"  Epsilon: {eps_start:.2f} → {eps_end:.2f} over {eps_decay_episodes} episodes")
    print(f"  Learning agents: {n_agents}  |  Bots: {n_bot_agents}"
          f"  |  Years/episode: {n_years}")
    print(f"  States: {StateDiscretizer.N_STATES}  |  Actions: 6×4 profiles")
    print(f"  Console log every {log_interval} episodes")
    print(f"{'═'*70}")
    print(f"  Console legend (HAPPO-mirrored, per log_interval episode):")
    print(f"  │  Ep / elapsed / ETA   Episode number · wall-clock · estimated remaining")
    print(f"  │  ε                    Current epsilon-greedy exploration rate")
    print(f"  │  Q                    Anchor-invariant quality_score in [-5, +5]")
    print(f"  │  Price/yr             Auction clearing price per simulated year (€/t)")
    print(f"  │  Emiss/yr             Total system emissions per year (Mt)")
    print(f"  │  Bid/yr               Total agent bid demand per year (Mt)")
    print(f"  │  Auct/yr              Auction supply after cap+rollover+MSR (Mt)")
    print(f"  │  Per-agent table:")
    print(f"  │     Green             Green fraction at start → end of episode (Δpp)")
    print(f"  │     Emiss/Alloc       Mean per-year emissions / allocations (Mt)")
    print(f"  │     Sf                Shortfall years / total years")
    print(f"  │     Why(U/D/M/B/C)    Per-year compliance attribution (HAPPO-aligned):")
    print(f"  │                       U=under-bid, D=debt-cascade, M=both,")
    print(f"  │                       B=bank-rescued (compliant via bank), C=sec-buy-rescued")
    print(f"  │     yr1€/yrN€/avg€    First-year, last-year, qty-weighted average bid (€/t)")
    print(f"  │     lo€/hi€           Lowest / highest bid price across the episode (€/t)")
    print(f"  │     BidMt             Mean bid quantity per year (Mt)")
    print(f"  │     Rew               Episode total reward (M€-equivalent)")
    print(f"  │     a1/a2             Last-year auction / secondary profile (short name)")
    print(f"  │  Q-learning columns (replace HAPPO's policy-loss columns):")
    print(f"  │     |TD|              Mean |Bellman residual| over this log window")
    print(f"  │     maxQ              Mean of greedy Q-value across visited states")
    print(f"  │     Cov%              Q-table state coverage (% of {StateDiscretizer.N_STATES} states)")
    print(f"  │     ε%                Share of action picks taken from the ε-random branch")
    print(f"  │  Q-Learning Board     Per-agent: states-visited, distinct greedy (a1,a2)")
    print(f"  │                       pairs across visited states, top auction profile,")
    print(f"  │                       mean |TD|, mean maxQ, # Bellman updates this window")
    print(f"  │  Event Board          Defaults, treasury draws, emergency loans (years/peak)")
    print(f"{'═'*70}\n")


    # Create environment (handles bots internally via heuristic_policy)
    env = ETSEnvironment(config, seed=seed)

    # Q-learning agents — only the learning agents need a Q-table; bots
    # are driven by the env's internal heuristic policy.
    agents = [
        QLearningAgent(agent_id=i, alpha=alpha, gamma=gamma, seed=seed + i)
        for i in range(n_agents)
    ]

    # Setup logging
    results_dir = config.get("logging", {}).get("results_dir", "results/qlearning/")
    os.makedirs(results_dir, exist_ok=True)

    ep_path = os.path.join(results_dir, f"ql_training_log{tag_part}_s{seed}.csv")
    ep_fields = _ep_fields(n_total_agents)
    ep_csv = open(ep_path, "w", newline="")
    ep_writer = csv.DictWriter(ep_csv, fieldnames=ep_fields, extrasaction="ignore")
    ep_writer.writeheader()

    yr_path = os.path.join(results_dir, f"ql_year_log{tag_part}_s{seed}.csv")
    yr_fields = _yr_fields(n_total_agents)
    yr_csv = open(yr_path, "w", newline="")
    yr_writer = csv.DictWriter(yr_csv, fieldnames=yr_fields, extrasaction="ignore")
    yr_writer.writeheader()

    # Tracking buffers for console output (learning agents only)
    reward_buffer = np.zeros((log_interval, n_agents))
    price_buffer = []
    green_buffer = np.zeros((log_interval, n_agents))
    compliance_buffer = np.zeros((log_interval, n_agents))
    shortfall_buffer = np.zeros((log_interval, n_agents))

    train_t0 = time.time()

    for episode in range(n_episodes):
        # Linear epsilon decay
        if episode < eps_decay_episodes:
            epsilon = eps_start + (eps_end - eps_start) * (episode / eps_decay_episodes)
        else:
            epsilon = eps_end

        env.set_episode(episode)
        obs1, _ = env.reset(seed=seed + episode * 1000)

        total_rewards_learning = np.zeros(n_agents)
        total_rewards_total = np.zeros(n_total_agents)  # incl. bots, from env logs

        year_states = [[] for _ in range(n_agents)]
        year_a1 = [[] for _ in range(n_agents)]
        year_a2 = [[] for _ in range(n_agents)]
        episode_shortfalls = np.zeros(n_total_agents)
        episode_compliant_years = np.zeros(n_total_agents)
        episode_invest_costs = np.zeros(n_total_agents)
        last_clearing_price = 0.0
        last_cap = 0.0
        last_tnac = 0.0
        episode_sec_volume = 0.0
        episode_sec_value = 0.0   # for secondary_avg_price
        episode_prices = []
        last_a1_profiles = np.zeros(n_agents, dtype=int)
        last_a2_profiles = np.zeros(n_agents, dtype=int)

        for year in range(n_years):
            price_ma3 = env._compute_price_ma3()

            # === PHASE 1: Auction ===
            auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
            a1_indices = np.zeros(n_agents, dtype=int)

            for i in range(n_agents):
                state = agents[i].discretizer.discretize(obs1[i], env.companies[i])
                action_vec, a1_idx = agents[i].select_auction_action(
                    obs1[i], env.companies[i], price_ma3, config,
                    epsilon=epsilon, current_year=year)
                auction_actions[i] = action_vec
                a1_indices[i] = a1_idx
                year_states[i].append(state)
                year_a1[i].append(a1_idx)

            obs2, _ = env.step_auction(auction_actions)

            # === PHASE 2: Secondary Market ===
            secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
            a2_indices = np.zeros(n_agents, dtype=int)

            for i in range(n_agents):
                action_vec, a2_idx = agents[i].select_secondary_action(
                    obs2[i], env.companies[i], env._phase1_clearing_price,
                    config, a1_idx=a1_indices[i], epsilon=epsilon,
                    current_year=year)
                secondary_actions[i] = action_vec
                a2_indices[i] = a2_idx
                year_a2[i].append(a2_idx)

            obs1_next, rewards, terminated, _, info = env.step_secondary(secondary_actions)
            total_rewards_learning += rewards

            # Q-learning update (one per learning agent)
            for i in range(n_agents):
                state = year_states[i][-1]
                next_state = agents[i].discretizer.discretize(
                    obs1_next[i], env.companies[i])
                agents[i].update(state, a1_indices[i], a2_indices[i],
                                 rewards[i], next_state, terminated)

            # Year-level logging — flatten env year_log into CSV row
            yl = info.get("year_log", {})
            last_clearing_price = yl.get("clearing_price", 0)
            last_cap = yl.get("cap", 0)
            last_tnac = yl.get("tnac", 0)
            sec_vol_y = float(yl.get("secondary_volume", 0) or 0)
            sec_clr_y = float(yl.get("secondary_clearing", 0) or 0)
            episode_prices.append(last_clearing_price)
            episode_sec_volume += sec_vol_y
            episode_sec_value += sec_clr_y * sec_vol_y
            last_a1_profiles = a1_indices.copy()
            last_a2_profiles = a2_indices.copy()

            shortfalls_y = yl.get("shortfalls", [0] * n_total_agents)
            invest_costs_y = yl.get("invest_costs", [0] * n_total_agents)
            rewards_y = yl.get("rewards", [0] * n_total_agents)
            for i in range(n_total_agents):
                sf = float(shortfalls_y[i]) if i < len(shortfalls_y) else 0.0
                episode_shortfalls[i] += sf
                if sf < 1e-6:
                    episode_compliant_years[i] += 1
                if i < len(invest_costs_y):
                    episode_invest_costs[i] += float(invest_costs_y[i] or 0.0)
                if i < len(rewards_y):
                    total_rewards_total[i] += float(rewards_y[i] or 0.0)

            yr_writer.writerow(_yr_row(yl, episode, year, n_total_agents))

            obs1 = obs1_next
            if terminated:
                break

        # Episode-level rollups
        buf_idx = episode % log_interval
        reward_buffer[buf_idx] = total_rewards_learning
        price_buffer.append(np.mean(episode_prices) if episode_prices else 0.0)
        green_buffer[buf_idx] = [env.companies[i].green_frac for i in range(n_agents)]
        compliance_buffer[buf_idx] = episode_compliant_years[:n_agents] / max(n_years, 1)
        shortfall_buffer[buf_idx] = episode_shortfalls[:n_agents]

        # Quality metric (anchor-invariant) — same util the PPO trainer uses
        quality = compute_episode_quality(
            env.episode_log, config, env.cap_schedule.get_cap,
            n_total_agents, env.n_years,
        )

        ep_mean_clearing = float(np.mean(episode_prices)) if episode_prices else 0.0
        price_start = float(episode_prices[0]) if episode_prices else 0.0
        price_peak = float(max(episode_prices)) if episode_prices else 0.0
        price_std = float(np.std(episode_prices)) if len(episode_prices) > 1 else 0.0
        sec_avg_price = (episode_sec_value / episode_sec_volume) if episode_sec_volume > 1e-9 else 0.0

        ep_row = {
            "episode": episode,
            "epsilon": round(float(epsilon), 4),
            "clearing_price_last": round(float(last_clearing_price), 2),
            "cap_last": round(float(last_cap), 4),
            "tnac": round(float(last_tnac), 4),
            "ep_mean_clearing_price": round(ep_mean_clearing, 2),
            "mean_price": round(ep_mean_clearing, 2),  # alias
            "secondary_volume": round(float(episode_sec_volume), 4),
            "secondary_avg_price": round(float(sec_avg_price), 2),
            "price_start": round(price_start, 2),
            "price_peak": round(price_peak, 2),
            "price_std": round(price_std, 2),
            "quality_score": (
                round(quality["quality_score"], 3)
                if not np.isnan(quality["quality_score"]) else None
            ),
            "Q_compliance":    None if np.isnan(quality["Q_compliance"]) else round(quality["Q_compliance"], 4),
            "Q_price_realism": None if np.isnan(quality["Q_price_realism"]) else round(quality["Q_price_realism"], 4),
            "Q_saved_carbon":  None if np.isnan(quality["Q_saved_carbon"]) else round(quality["Q_saved_carbon"], 4),
            "Q_cost_eff":      None if np.isnan(quality["Q_cost_eff"]) else round(quality["Q_cost_eff"], 4),
            "Q_volatility":    None if np.isnan(quality["Q_volatility"]) else round(quality["Q_volatility"], 4),
        }
        # Precompute per-agent total penalty across the episode in a
        # single pass over env.episode_log (avoids re-walking the log
        # n_total_agents times inside the row-building loop below).
        episode_penalties = np.zeros(n_total_agents, dtype=np.float64)
        for yl in env.episode_log:
            pens = yl.get("penalties", [])
            for i in range(min(n_total_agents, len(pens))):
                episode_penalties[i] += float(pens[i] or 0.0)
        for i in range(n_total_agents):
            company_i = env.companies[i] if i < len(env.companies) else None
            if i < n_agents:
                ep_row[f"reward_A{i+1}"] = round(float(total_rewards_learning[i]), 3)
                ep_row[f"a1_profile_A{i+1}"] = int(last_a1_profiles[i])
                ep_row[f"a2_profile_A{i+1}"] = int(last_a2_profiles[i])
            else:
                # Bots: use the env-reported total reward, no profile index
                ep_row[f"reward_A{i+1}"] = round(float(total_rewards_total[i]), 3)
                ep_row[f"a1_profile_A{i+1}"] = -1
                ep_row[f"a2_profile_A{i+1}"] = -1
            ep_row[f"green_frac_A{i+1}"] = round(
                float(company_i.green_frac if company_i is not None else 0.0), 4
            )
            ep_row[f"delta_green_A{i+1}"] = round(
                float((company_i.green_frac - company_i.prev_green_frac)
                      if company_i is not None else 0.0), 4
            )
            ep_row[f"compliance_rate_A{i+1}"] = round(
                float(episode_compliant_years[i] / max(n_years, 1)), 3
            )
            ep_row[f"shortfall_A{i+1}"] = round(float(episode_shortfalls[i]), 4)
            ep_row[f"penalty_A{i+1}"] = round(float(episode_penalties[i]), 3)
            ep_row[f"bid_price_A{i+1}"] = round(float(last_clearing_price), 2)
            ep_row[f"queue_size_A{i+1}"] = int(
                len(company_i._construction_queue) if company_i is not None else 0
            )
            ep_row[f"invest_cost_A{i+1}"] = round(float(episode_invest_costs[i]), 3)
        ep_writer.writerow(ep_row)

        # Periodic flush
        flush_interval = config.get("logging", {}).get("csv_flush_interval", 500)
        if (episode + 1) % flush_interval == 0:
            ep_csv.flush()
            yr_csv.flush()

        # Console output — HAPPO-style heavy box
        if (episode + 1) % log_interval == 0:
            elapsed = time.time() - train_t0
            eps_per_sec = (episode + 1) / max(elapsed, 1e-6)
            eta = (n_episodes - episode - 1) / max(eps_per_sec, 0.01)
            avg_ep_s = elapsed / max(episode + 1, 1)
            n_buf = min(episode + 1, log_interval)

            avg_reward = reward_buffer[:n_buf].mean(axis=0)
            avg_green = green_buffer[:n_buf].mean(axis=0)
            avg_compliance = compliance_buffer[:n_buf].mean(axis=0)
            avg_shortfall = shortfall_buffer[:n_buf].mean(axis=0)

            # Per-year market trajectories of the most recent episode.
            yr_price = []
            yr_emiss_total = []
            yr_bid_total = []
            yr_auct_vol = []
            for yl in env.episode_log:
                yr_price.append(float(yl.get("clearing_price", 0.0) or 0.0))
                emi = yl.get("emissions", []) or []
                yr_emiss_total.append(float(sum(emi[: n_total_agents])))
                bqs = yl.get("bid_quantities", []) or []
                yr_bid_total.append(float(sum(max(0.0, q) for q in bqs[: n_total_agents])))
                yr_auct_vol.append(float(yl.get("auction_volume", yl.get("cap", 0.0)) or 0.0))

            # Per-agent bid trajectories (yr1/yrN/avg/lo/hi/BidMt) — same
            # structure HAPPO prints. Walk env.episode_log once per agent.
            yr1_bid = np.zeros(n_total_agents)
            yrN_bid = np.zeros(n_total_agents)
            min_bid = np.zeros(n_total_agents)
            max_bid = np.zeros(n_total_agents)
            avg_bid = np.zeros(n_total_agents)
            avg_bidqty = np.zeros(n_total_agents)
            ep_mean_emiss = np.zeros(n_total_agents)
            ep_mean_alloc = np.zeros(n_total_agents)
            ep_shortfall_years = np.zeros(n_total_agents, dtype=int)
            for i in range(n_total_agents):
                bids = []
                qtys = []
                emis = []
                allocs = []
                sfy = 0
                for yl in env.episode_log:
                    bp = yl.get("bid_prices", []) or []
                    bq = yl.get("bid_quantities", []) or []
                    em = yl.get("emissions", []) or []
                    al = yl.get("allocations", []) or []
                    sf = yl.get("shortfalls", []) or []
                    if i < len(bp):
                        bids.append(float(bp[i]))
                    if i < len(bq):
                        qtys.append(max(0.0, float(bq[i])))
                    if i < len(em):
                        emis.append(float(em[i]))
                    if i < len(al):
                        allocs.append(float(al[i]))
                    if i < len(sf) and float(sf[i]) > 1e-6:
                        sfy += 1
                if bids:
                    yr1_bid[i] = bids[0]
                    yrN_bid[i] = bids[-1]
                    min_bid[i] = min(bids)
                    max_bid[i] = max(bids)
                    if qtys and sum(qtys) > 1e-9:
                        avg_bid[i] = float(np.dot(bids, qtys) / sum(qtys))
                    else:
                        avg_bid[i] = float(np.mean(bids))
                if qtys:
                    avg_bidqty[i] = float(np.mean(qtys))
                if emis:
                    ep_mean_emiss[i] = float(np.mean(emis))
                if allocs:
                    ep_mean_alloc[i] = float(np.mean(allocs))
                ep_shortfall_years[i] = sfy

            # Secondary stats (per agent + system-level) and compliance
            # attribution. Buckets mirror HAPPO's per-agent attribution
            # (Underbid / Debt-cascade / Mixed / sec-Buy-rescued /
            # sec-non-Compliant) so the two consoles read identically:
            #   U: alloc < emiss this year, no inherited carry-forward
            #   D: alloc ≥ emiss this year, but inherited carry-forward
            #      created a residual shortfall (debt cascade)
            #   M: both — under-bid this year AND inherited debt
            #   B: emiss + cf > alloc but settled compliant via secondary
            #      *without* a net buy this year (used bank only)
            #   C: emiss + cf > alloc, settled compliant via a net
            #      secondary buy this year
            per_agent_sec = [
                {"buy_y": 0, "buy_v": 0.0, "buy_px": 0.0,
                 "sell_y": 0, "sell_v": 0.0, "sell_px": 0.0}
                for _ in range(n_total_agents)
            ]
            per_agent_attr = [
                {"u": 0, "d": 0, "m": 0, "b": 0, "c": 0}
                for _ in range(n_total_agents)
            ]
            total_sec_vol = 0.0
            years_with_trades = 0
            for yl in env.episode_log:
                tq = yl.get("trade_qtys", []) or []
                tc = yl.get("trade_costs", []) or []
                allocs_yl = yl.get("allocations", []) or []
                emiss_yl = yl.get("emissions", []) or []
                cf_yl = yl.get("old_carry_forward", []) or []
                shorts_yl = yl.get("shortfalls", []) or []
                year_had_trade = False
                for i in range(n_total_agents):
                    q = float(tq[i]) if i < len(tq) else 0.0
                    c = float(tc[i]) if i < len(tc) else 0.0
                    if q > 1e-6:
                        per_agent_sec[i]["buy_y"] += 1
                        per_agent_sec[i]["buy_v"] += q
                        per_agent_sec[i]["buy_px"] += abs(c)
                        total_sec_vol += q
                        year_had_trade = True
                    elif q < -1e-6:
                        per_agent_sec[i]["sell_y"] += 1
                        per_agent_sec[i]["sell_v"] += -q
                        per_agent_sec[i]["sell_px"] += abs(c)
                        year_had_trade = True

                    # Compliance attribution per (agent, year)
                    if i < len(allocs_yl) and i < len(emiss_yl):
                        alloc_i = float(allocs_yl[i])
                        emiss_i = float(emiss_yl[i])
                        cf_i = float(cf_yl[i]) if i < len(cf_yl) else 0.0
                        short_i = float(shorts_yl[i]) if i < len(shorts_yl) else 0.0
                        underbid = alloc_i + 1e-6 < emiss_i
                        debt = cf_i > 1e-6
                        gap = (emiss_i + cf_i) > alloc_i + 1e-6
                        non_comp = short_i > 1e-6
                        net_buy = q > 1e-6
                        if non_comp:
                            if underbid and not debt:
                                per_agent_attr[i]["u"] += 1
                            elif (not underbid) and debt:
                                per_agent_attr[i]["d"] += 1
                            elif underbid and debt:
                                per_agent_attr[i]["m"] += 1
                            # else: residual — skip (should not happen)
                        else:
                            if gap:
                                if net_buy:
                                    per_agent_attr[i]["c"] += 1
                                else:
                                    per_agent_attr[i]["b"] += 1
                            # else: no obligation gap, compliant — quiet year
                if year_had_trade:
                    years_with_trades += 1
            sec_match_rate = years_with_trades / max(len(env.episode_log), 1)
            for s in per_agent_sec:
                if s["buy_v"] > 1e-9:
                    s["buy_px"] = s["buy_px"] / s["buy_v"]
                if s["sell_v"] > 1e-9:
                    s["sell_px"] = s["sell_px"] / s["sell_v"]

            # Event Board: defaults / treasury draws / emergency loans
            default_counts = {}
            treasury_counts = {}
            loan_counts = {}
            loan_peak = {}
            for yl in env.episode_log:
                aud = yl.get("auction_stats", {}) or {}
                for di in aud.get("defaults_agents", []) or []:
                    idx = int(di)
                    default_counts[idx] = default_counts.get(idx, 0) + 1
                td = yl.get("treasury_drawn", []) or []
                for idx, t in enumerate(td):
                    if float(t or 0.0) > 0:
                        treasury_counts[idx] = treasury_counts.get(idx, 0) + 1
                lo = yl.get("loan_outstanding", []) or []
                for idx, v in enumerate(lo):
                    v = float(v or 0.0)
                    if v > 0:
                        loan_counts[idx] = loan_counts.get(idx, 0) + 1
                        loan_peak[idx] = max(loan_peak.get(idx, 0.0), v)

            qs = quality["quality_score"]
            qs_str = "n/a" if np.isnan(qs) else f"{qs:+.2f}"

            avg_emiss_year = (
                float(np.mean(yr_emiss_total)) if yr_emiss_total else 0.0
            )

            W = 195
            sep = "═" * W
            thin = "─" * W

            # ── Control plane ────────────────────────────────────────────
            print(sep)
            print(f"  Ep {episode+1:5d}/{n_episodes} │ "
                  f"{_format_hms(elapsed)} elapsed  ETA {_format_hms(eta)}"
                  f"  ({avg_ep_s:.2f} s/ep)"
                  f" │ ε={epsilon:.3f} │ Q={qs_str}")

            # ── Market trajectories ──────────────────────────────────────
            price_traj = " ".join(f"{p:5.0f}" for p in yr_price)
            emiss_traj = " ".join(f"{e:5.1f}" for e in yr_emiss_total)
            bid_traj   = " ".join(f"{b:5.1f}" for b in yr_bid_total)
            auct_traj  = " ".join(f"{c:5.1f}" for c in yr_auct_vol)
            price_std_ep = float(np.std(yr_price)) if len(yr_price) > 1 else 0.0
            print(f"  {'Price/yr':<8}: {price_traj}   (σ={price_std_ep:.0f}){_trend_arrow(yr_price)}")
            print(f"  {'Emiss/yr':<8}: {emiss_traj}   (avg {avg_emiss_year:.1f} Mt/yr){_trend_arrow(yr_emiss_total)}")
            print(f"  {'Bid/yr':<8}: {bid_traj}   (agents' total auction demand, Mt){_trend_arrow(yr_bid_total)}")
            print(f"  {'Auct/yr':<8}: {auct_traj}   (auction supply after cap+rollover+MSR, TNAC={last_tnac:.1f} Mt){_trend_arrow(yr_auct_vol)}")

            # ── Health snapshot ──────────────────────────────────────────
            sys_comply = float(np.mean(avg_compliance)) * 100
            avg_secp = (
                episode_sec_value / episode_sec_volume
                if episode_sec_volume > 1e-9 else 0.0
            )
            print(f"  Health: comply={sys_comply:.0f}%"
                  f"  green(learning avg)={float(np.mean(avg_green))*100:.0f}%"
                  f" │ Sec: {total_sec_vol:.1f}Mt  match={sec_match_rate*100:.0f}%"
                  f"  avg={avg_secp:.0f}€")

            # ── Per-agent table ──────────────────────────────────────────
            # Mirrors HAPPO's table but the policy-internal columns
            # (DiagPts/ALoss/CLoss in HAPPO) are replaced by Q-learning
            # analogs: |TD| (mean absolute Bellman residual this window),
            # maxQ (mean of greedy Q across visited states), Cov% (Q-table
            # state coverage), and ε% (share of ε-random picks this
            # window). a1/a2 are rendered as profile names.
            print(thin)
            print(f"  {'':4}  {'Green':>16}  {'Emiss':>5} {'Alloc':>5}"
                  f" {'Sf':>5} {'Why(U/D/M/B/C)':>16}  "
                  f"{'yr1€':>5} {'yrN€':>5} {'avg€':>5} {'lo€':>5} {'hi€':>5} {'BidMt':>6}"
                  f"  {'Rew':>8}  {'a1/a2':>11}"
                  f"  {'|TD|':>7} {'maxQ':>7} {'Cov%':>5} {'ε%':>4}"
                  f"  Secondary")

            for i in range(n_agents):
                g0 = float(env.companies[i].prev_green_frac) * 100
                g1 = float(env.companies[i].green_frac) * 100
                dg = g1 - g0
                grn_str = f"{g0:3.0f}→{g1:3.0f}%({dg:+3.0f}pp)"
                sfy = ep_shortfall_years[i]
                ny = max(len(env.episode_log), 1)
                sf_str = f"{int(sfy):2d}/{ny:<2d}"
                attr = per_agent_attr[i]
                why_str = f"{attr['u']:2d}/{attr['d']:2d}/{attr['m']:2d}/{attr['b']:2d}/{attr['c']:2d}"

                ss = per_agent_sec[i]
                sec_parts = []
                if ss["buy_y"] > 0:
                    sec_parts.append(f"B{ss['buy_y']}y/{ss['buy_v']:.1f}Mt@{ss['buy_px']:.0f}€")
                if ss["sell_y"] > 0:
                    sec_parts.append(f"S{ss['sell_y']}y/{ss['sell_v']:.1f}Mt@{ss['sell_px']:.0f}€")
                sec_str = " ".join(sec_parts) if sec_parts else "HOLD"

                # Q-learning diagnostics for this agent / window
                td_abs = agents[i].window_td_mean_abs()
                max_q = agents[i].mean_max_q()
                cov_pct = agents[i].coverage_fraction() * 100.0
                eps_pct = agents[i].window_explore_share() * 100.0
                a1_lbl = _a1_label(int(last_a1_profiles[i]))
                a2_lbl = _a2_label(int(last_a2_profiles[i]))
                action_str = f"{a1_lbl:>4}/{a2_lbl:<4}"

                print(f"  A{i+1} : {grn_str:>16}"
                      f"  {ep_mean_emiss[i]:5.2f} {ep_mean_alloc[i]:5.2f}"
                      f" {sf_str:>5} {why_str:>16}  "
                      f"{yr1_bid[i]:5.0f} {yrN_bid[i]:5.0f} {avg_bid[i]:5.0f} {min_bid[i]:5.0f} {max_bid[i]:5.0f} {avg_bidqty[i]:6.2f}"
                      f"  {avg_reward[i]:8.1f}"
                      f"  {action_str:>11}"
                      f"  {td_abs:7.3f} {max_q:7.2f} {cov_pct:4.0f}% {eps_pct:3.0f}%"
                      f"  {sec_str}")

            # ── Bot rows (driven by heuristic policy) ────────────────────
            if n_bot_agents > 0:
                print(thin)
                for b in range(n_bot_agents):
                    j = n_agents + b
                    g0 = float(env.companies[j].prev_green_frac) * 100
                    g1 = float(env.companies[j].green_frac) * 100
                    dg = g1 - g0
                    grn_str = f"{g0:3.0f}→{g1:3.0f}%({dg:+3.0f}pp)"
                    sfy = ep_shortfall_years[j]
                    ny = max(len(env.episode_log), 1)
                    sf_str = f"{int(sfy):2d}/{ny:<2d}"
                    attr_b = per_agent_attr[j]
                    why_str = (f"{attr_b['u']:2d}/{attr_b['d']:2d}/"
                               f"{attr_b['m']:2d}/{attr_b['b']:2d}/{attr_b['c']:2d}")
                    ss = per_agent_sec[j]
                    sec_parts = []
                    if ss["buy_y"] > 0:
                        sec_parts.append(f"B{ss['buy_y']}y/{ss['buy_v']:.1f}Mt@{ss['buy_px']:.0f}€")
                    if ss["sell_y"] > 0:
                        sec_parts.append(f"S{ss['sell_y']}y/{ss['sell_v']:.1f}Mt@{ss['sell_px']:.0f}€")
                    sec_str = " ".join(sec_parts) if sec_parts else "HOLD"
                    # Bots have no Q-table — pad the QL columns with dashes
                    # so the table still aligns vertically.
                    print(f"  B{b+1} : {grn_str:>16}"
                          f"  {ep_mean_emiss[j]:5.2f} {ep_mean_alloc[j]:5.2f}"
                          f" {sf_str:>5} {why_str:>16}  "
                          f"{yr1_bid[j]:5.0f} {yrN_bid[j]:5.0f} {avg_bid[j]:5.0f} {min_bid[j]:5.0f} {max_bid[j]:5.0f} {avg_bidqty[j]:6.2f}"
                          f"  {total_rewards_total[j]:8.1f}"
                          f"  {'  bot/bot':>11}"
                          f"  {'    -':>7} {'    -':>7} {'  - ':>5} {' - ':>4}"
                          f"  {sec_str}")

            # ── Q-Learning Board (sibling of HAPPO's Event Board) ────────
            # Surfaces the per-agent learning state of the Q-tables so the
            # "Q-learning-ness" is visible at a glance: state coverage,
            # mean |TD|, mean max-Q, and how many distinct (a1,a2) pairs
            # the greedy policy currently uses across visited states.
            print(thin)
            print("  Q-Learning Board:  α={:.3f}  γ={:.3f}  states={}  ε={:.3f}".format(
                alpha, gamma, StateDiscretizer.N_STATES, epsilon))
            for i in range(n_agents):
                cov_states = agents[i].coverage_states()
                cov_pct = 100.0 * cov_states / StateDiscretizer.N_STATES
                td_abs = agents[i].window_td_mean_abs()
                max_q = agents[i].mean_max_q()
                diversity = agents[i].greedy_policy_diversity()
                top_a1 = agents[i].top_auction_profile()
                n_updates = agents[i].window_update_count()
                print(f"    A{i+1}: states {cov_states:3d}/{StateDiscretizer.N_STATES} ({cov_pct:4.0f}%)"
                      f"  greedy-pairs={diversity:2d}/{ActionProfileMapper.N_AUCTION_PROFILES * ActionProfileMapper.N_SECONDARY_PROFILES}"
                      f"  top-a1={_a1_label(top_a1):<4}"
                      f"  |TD|={td_abs:7.3f}"
                      f"  maxQ={max_q:7.2f}"
                      f"  updates={n_updates:5d}")

            # Reset rolling Q-learning diagnostics for the next window
            for ag in agents:
                ag.reset_window_stats()

            # ── Event board ──────────────────────────────────────────────
            print(thin)
            print("  Event Board:")
            if default_counts:
                items = []
                for idx in sorted(default_counts):
                    tag = f"A{idx+1}" if idx < n_agents else f"B{idx-n_agents+1}"
                    items.append(f"{tag}:{default_counts[idx]}")
                print(f"  Defaults by agent (year-count): {'  '.join(items)}")
            else:
                print("  Defaults by agent: none")
            if treasury_counts:
                items = []
                for idx in sorted(treasury_counts):
                    tag = f"A{idx+1}" if idx < n_agents else f"B{idx-n_agents+1}"
                    items.append(f"{tag}:{treasury_counts[idx]}yr")
                print(f"  Treasury drawn (years): {'  '.join(items)}")
            if loan_counts:
                items = []
                for idx in sorted(loan_counts):
                    tag = f"A{idx+1}" if idx < n_agents else f"B{idx-n_agents+1}"
                    items.append(f"{tag}:{loan_counts[idx]}yr/pk{loan_peak[idx]:.0f}M€")
                print(f"  Emergency loans (years/peak outstanding): {'  '.join(items)}")
            print(sep)


        # Periodic Q-table checkpoint
        if (episode + 1) % 100 == 0:
            ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            qtable_path = os.path.join(
                ckpt_dir, f"qtables{tag_part}_s{seed}_ep{episode+1}.npz")
            qtables = {f"agent_{i}": agents[i].q_table.copy() for i in range(n_agents)}
            np.savez(qtable_path, **qtables)

            if (episode + 1) % 500 == 0:
                print(f"--- Top-5 Q-values at episode {episode+1} ---")
                for i in range(n_agents):
                    top = agents[i].get_top_q_values(5)
                    print(f"  A{i+1}:", end="")
                    for s, a1, a2, qval in top:
                        print(f"  (s={s},a1={a1},a2={a2},Q={qval:+.3f})", end="")
                    print()
                print()

    # Final checkpoint
    final_path = os.path.join(results_dir, f"qtables{tag_part}_s{seed}_final.npz")
    qtables = {f"agent_{i}": agents[i].q_table.copy() for i in range(n_agents)}
    np.savez(final_path, **qtables)

    ep_csv.close()
    yr_csv.close()

    print(f"\n{'═'*70}")
    print(f"  Q-Learning training complete — seed {seed}")
    print(f"  Final Q-tables saved to {final_path}")
    print(f"  Episode log: {ep_path}")
    print(f"  Year log:    {yr_path}")
    print(f"{'═'*70}")

    return agents, config


# ---------------------------------------------------------------------------
# Greedy evaluation
# ---------------------------------------------------------------------------

def evaluate_qlearning(agents, config, seed, n_eval_episodes=100,
                       run_tag: str | None = None):
    """Run greedy evaluation episodes and log results."""
    n_agents = config["companies"]["n_agents"]
    n_bot_agents = config["companies"].get("n_bot_agents", 0)
    n_total_agents = n_agents + n_bot_agents
    n_years = config["simulation"]["n_years"]
    results_dir = config.get("logging", {}).get("results_dir", "results/qlearning/")
    tag_part = f"_{run_tag}" if run_tag else ""

    print(f"\n{'='*70}")
    print(f"Q-Learning Evaluation — {n_eval_episodes} greedy episodes (seed {seed})")
    print(f"{'='*70}")

    env = ETSEnvironment(config, seed=seed + 99999)

    eval_path = os.path.join(results_dir, f"ql_eval_log{tag_part}_s{seed}.csv")
    eval_fields = ["episode", "clearing_price_last", "tnac", "secondary_volume",
                   "mean_price", "quality_score"]
    for i in range(n_total_agents):
        eval_fields += [f"reward_A{i+1}", f"green_frac_A{i+1}",
                        f"compliance_rate_A{i+1}", f"shortfall_A{i+1}"]
    eval_csv = open(eval_path, "w", newline="")
    eval_writer = csv.DictWriter(eval_csv, fieldnames=eval_fields, extrasaction="ignore")
    eval_writer.writeheader()

    all_rewards = np.zeros((n_eval_episodes, n_agents))
    all_green = np.zeros((n_eval_episodes, n_agents))
    all_compliance = np.zeros((n_eval_episodes, n_agents))
    all_shortfall = np.zeros((n_eval_episodes, n_agents))
    all_prices = []
    all_quality = []

    for ep in range(n_eval_episodes):
        env.set_episode(99999)  # disable shaping decay
        obs1, _ = env.reset(seed=seed + 99999 + ep * 1000)
        total_rewards = np.zeros(n_agents)
        compliant_years = np.zeros(n_total_agents)
        total_shortfall = np.zeros(n_total_agents)
        total_rewards_all = np.zeros(n_total_agents)
        last_price = 0.0
        last_tnac = 0.0
        ep_sec_vol = 0.0
        ep_prices = []

        for year in range(n_years):
            price_ma3 = env._compute_price_ma3()

            auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
            a1_indices = np.zeros(n_agents, dtype=int)
            for i in range(n_agents):
                action_vec, a1_idx = agents[i].select_auction_action(
                    obs1[i], env.companies[i], price_ma3, config,
                    epsilon=0.0, current_year=year)
                auction_actions[i] = action_vec
                a1_indices[i] = a1_idx

            obs2, _ = env.step_auction(auction_actions)

            secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
            for i in range(n_agents):
                action_vec, _ = agents[i].select_secondary_action(
                    obs2[i], env.companies[i], env._phase1_clearing_price,
                    config, a1_idx=a1_indices[i], epsilon=0.0,
                    current_year=year)
                secondary_actions[i] = action_vec

            obs1, rewards, terminated, _, info = env.step_secondary(secondary_actions)
            total_rewards += rewards

            yl = info.get("year_log", {})
            last_price = yl.get("clearing_price", 0)
            last_tnac = yl.get("tnac", 0)
            ep_sec_vol += float(yl.get("secondary_volume", 0) or 0)
            ep_prices.append(last_price)
            shortfalls = yl.get("shortfalls", [0] * n_total_agents)
            rewards_y = yl.get("rewards", [0] * n_total_agents)
            for i in range(n_total_agents):
                sf = float(shortfalls[i]) if i < len(shortfalls) else 0.0
                total_shortfall[i] += sf
                if sf < 1e-6:
                    compliant_years[i] += 1
                if i < len(rewards_y):
                    total_rewards_all[i] += float(rewards_y[i] or 0.0)

            if terminated:
                break

        all_rewards[ep] = total_rewards
        all_green[ep] = [env.companies[i].green_frac for i in range(n_agents)]
        all_compliance[ep] = compliant_years[:n_agents] / max(n_years, 1)
        all_shortfall[ep] = total_shortfall[:n_agents]
        all_prices.append(np.mean(ep_prices) if ep_prices else 0.0)

        quality = compute_episode_quality(
            env.episode_log, config, env.cap_schedule.get_cap,
            n_total_agents, env.n_years,
        )
        qs = quality["quality_score"]
        all_quality.append(qs if not np.isnan(qs) else 0.0)

        row = {"episode": ep,
               "clearing_price_last": round(float(last_price), 2),
               "tnac": round(float(last_tnac), 4),
               "secondary_volume": round(float(ep_sec_vol), 4),
               "mean_price": round(float(np.mean(ep_prices)) if ep_prices else 0.0, 2),
               "quality_score": None if np.isnan(qs) else round(qs, 3)}
        for i in range(n_total_agents):
            if i < n_agents:
                row[f"reward_A{i+1}"] = round(float(total_rewards[i]), 3)
            else:
                row[f"reward_A{i+1}"] = round(float(total_rewards_all[i]), 3)
            row[f"green_frac_A{i+1}"] = round(float(env.companies[i].green_frac), 4)
            row[f"compliance_rate_A{i+1}"] = round(
                float(compliant_years[i] / max(n_years, 1)), 3)
            row[f"shortfall_A{i+1}"] = round(float(total_shortfall[i]), 4)
        eval_writer.writerow(row)

    eval_csv.close()

    print(f"\nEvaluation Results (mean ± std over {n_eval_episodes} episodes):")
    print(f"  Avg clearing price: {np.mean(all_prices):.1f} ± {np.std(all_prices):.1f} €/t")
    print(f"  Avg quality_score:  {np.mean(all_quality):+.2f} ± {np.std(all_quality):.2f}")
    for i in range(n_agents):
        print(f"  A{i+1}: reward={all_rewards[:,i].mean():+.2f}±{all_rewards[:,i].std():.2f}  "
              f"green={all_green[:,i].mean()*100:.1f}%  "
              f"comply={all_compliance[:,i].mean()*100:.1f}%  "
              f"short={all_shortfall[:,i].mean():.3f}Mt")
    print(f"\n  Eval log saved to {eval_path}")

    return {
        "rewards": all_rewards,
        "green_fracs": all_green,
        "compliance": all_compliance,
        "shortfalls": all_shortfall,
        "prices": np.array(all_prices),
        "quality": np.array(all_quality),
    }


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Q-Learning baseline training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Base environment config (e.g. configs/default.yaml)")
    parser.add_argument("--ql-config", type=str, default="configs/qlearning.yaml",
                        help="Q-learning specific config overrides")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (overrides simulation.seeds)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds (overrides simulation.seeds)")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Filename infix for sweep variants (mirrors scripts/train.py)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load Q-tables and evaluate")
    parser.add_argument("--qtable-path", type=str, default=None,
                        help="Path to Q-table .npz file (for --eval-only)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip post-training evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    ql_config = load_config(args.ql_config)
    merged = merge_configs(config, ql_config)

    # Resolve seed list
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = list(args.seeds)
    else:
        seeds = list(merged.get("simulation", {}).get("seeds", [42]))
    print(f"\nQ-Learning baseline run — seeds: {seeds}"
          f"{' [tag=' + args.run_tag + ']' if args.run_tag else ''}\n")

    if args.eval_only:
        if args.qtable_path is None and len(seeds) != 1:
            raise SystemExit(
                "--eval-only without --qtable-path requires a single --seed")
        n_agents = merged["companies"]["n_agents"]
        ql_params = merged.get("qlearning", {})
        eval_cfg = merged.get("evaluation", {})

        for seed in seeds:
            qtable_path = args.qtable_path
            if qtable_path is None:
                results_dir = merged.get("logging", {}).get(
                    "results_dir", "results/qlearning/")
                tag_part = f"_{args.run_tag}" if args.run_tag else ""
                qtable_path = os.path.join(
                    results_dir, f"qtables{tag_part}_s{seed}_final.npz")
            print(f"Loading Q-tables from {qtable_path}")
            qtables = dict(np.load(qtable_path))
            agents = [
                QLearningAgent(agent_id=i,
                               alpha=ql_params.get("alpha", 0.1),
                               gamma=ql_params.get("gamma", 0.95),
                               seed=seed + i)
                for i in range(n_agents)
            ]
            for i in range(n_agents):
                agents[i].q_table = qtables[f"agent_{i}"]
            evaluate_qlearning(agents, merged, seed,
                               n_eval_episodes=eval_cfg.get("n_episodes", 100),
                               run_tag=args.run_tag)
        return

    # Train (and optionally evaluate) each seed sequentially
    for seed in seeds:
        agents, merged_cfg = train_qlearning(config, ql_config, seed,
                                             run_tag=args.run_tag)
        if not args.no_eval:
            eval_cfg = merged_cfg.get("evaluation", {})
            evaluate_qlearning(agents, merged_cfg, seed,
                               n_eval_episodes=eval_cfg.get("n_episodes", 100),
                               run_tag=args.run_tag)


if __name__ == "__main__":
    main()
