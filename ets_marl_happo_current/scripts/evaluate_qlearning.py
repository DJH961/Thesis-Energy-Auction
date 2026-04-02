"""
evaluate_qlearning.py
=====================
Load trained Q-learning agents and run deterministic evaluation episodes.
Logs the same metrics as PPO evaluation for direct comparison.

Usage:
    python scripts/evaluate_qlearning.py \
        --config configs/default.yaml \
        --ql-config configs/qlearning.yaml \
        --qtable-path results/qlearning/qtables_s42_final.pkl \
        [--n-episodes 100] [--seed 42]
"""

import argparse
import csv
import os
import sys
import io
import yaml
import numpy as np

if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.agents.q_learning_agent import QLearningAgent


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base, override):
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Q-learning agents")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ql-config", type=str, default="configs/qlearning.yaml")
    parser.add_argument("--qtable-path", type=str, required=True,
                        help="Path to Q-table pickle file")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for eval results (default: same as qtable dir)")
    args = parser.parse_args()

    config = load_config(args.config)
    ql_config = load_config(args.ql_config)
    config = merge_configs(config, ql_config)

    n_agents = config["companies"]["n_agents"]
    n_years = config["simulation"]["n_years"]
    ql_params = config.get("qlearning", {})
    tech_names = config["technologies"]["names"]

    # Load Q-tables
    print(f"Loading Q-tables from {args.qtable_path}")
    qtables = dict(np.load(args.qtable_path))

    agents = [
        QLearningAgent(agent_id=i,
                       alpha=ql_params.get("alpha", 0.1),
                       gamma=ql_params.get("gamma", 0.95),
                       seed=args.seed + i)
        for i in range(n_agents)
    ]
    for i in range(n_agents):
        agents[i].q_table = qtables[f"agent_{i}"]
        print(f"  Agent {i}: Q-table loaded, "
              f"non-zero={np.count_nonzero(agents[i].q_table)}/{agents[i].q_table.size}")

    # Output directory
    output_dir = args.output_dir or os.path.dirname(args.qtable_path)
    os.makedirs(output_dir, exist_ok=True)

    # CSV for eval results (PPO-compatible format)
    eval_path = os.path.join(output_dir, f"ql_eval_results_s{args.seed}.csv")
    eval_fields = ["episode", "clearing_price_last", "cap_last", "tnac"]
    for i in range(n_agents):
        eval_fields += [f"reward_A{i+1}", f"green_frac_A{i+1}",
                        f"shortfall_A{i+1}", f"penalty_A{i+1}",
                        f"compliance_rate_A{i+1}"]
    eval_fields += ["secondary_volume", "mean_price"]
    eval_csv = open(eval_path, "w", newline="")
    eval_writer = csv.DictWriter(eval_csv, fieldnames=eval_fields)
    eval_writer.writeheader()

    # Year-level CSV for detailed analysis
    yr_path = os.path.join(output_dir, f"ql_eval_year_log_s{args.seed}.csv")
    yr_fields = ["episode", "year", "cap", "auction_volume", "tnac",
                 "clearing_price", "secondary_price", "msr_reserve"]
    for i in range(n_agents):
        yr_fields += [f"alloc_A{i+1}", f"emissions_A{i+1}",
                      f"green_frac_A{i+1}", f"shortfall_A{i+1}",
                      f"penalty_A{i+1}", f"reward_A{i+1}",
                      f"bid_price_A{i+1}", f"holdings_A{i+1}"]
    yr_csv = open(yr_path, "w", newline="")
    yr_writer = csv.DictWriter(yr_csv, fieldnames=yr_fields)
    yr_writer.writeheader()

    env = ETSEnvironment(config, seed=args.seed)

    print(f"\n{'='*70}")
    print(f"Q-LEARNING EVALUATION — {args.n_episodes} greedy episodes")
    print(f"{'='*70}")

    all_rewards = np.zeros((args.n_episodes, n_agents))
    all_green = np.zeros((args.n_episodes, n_agents))
    all_compliance = np.zeros((args.n_episodes, n_agents))
    all_shortfall = np.zeros((args.n_episodes, n_agents))
    all_prices = []

    for ep in range(args.n_episodes):
        env.set_episode(99999)  # disable shaping
        obs1, _ = env.reset(seed=args.seed + 99999 + ep * 1000)
        total_rewards = np.zeros(n_agents)
        compliant_years = np.zeros(n_agents)
        total_shortfall = np.zeros(n_agents)
        total_penalty = np.zeros(n_agents)
        episode_prices = []
        last_price = 0.0
        last_cap = 0.0
        last_tnac = 0.0
        episode_sec_volume = 0.0

        for year in range(n_years):
            price_ma3 = env._compute_price_ma3()

            # Phase 1 (greedy)
            auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
            a1_indices = np.zeros(n_agents, dtype=int)
            for i in range(n_agents):
                action_vec, a1_idx = agents[i].select_auction_action(
                    obs1[i], env.companies[i], price_ma3, config, epsilon=0.0)
                auction_actions[i] = action_vec
                a1_indices[i] = a1_idx

            obs2, _ = env.step_auction(auction_actions)

            # Phase 2 (greedy)
            secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
            for i in range(n_agents):
                action_vec, _ = agents[i].select_secondary_action(
                    obs2[i], env.companies[i], env._phase1_clearing_price,
                    config, a1_idx=a1_indices[i], epsilon=0.0)
                secondary_actions[i] = action_vec

            obs1, rewards, terminated, _, info = env.step_secondary(secondary_actions)
            total_rewards += rewards

            yl = info.get("year_log", {})
            last_price = yl.get("clearing_price", 0)
            last_cap = yl.get("cap", 0)
            last_tnac = yl.get("tnac", 0)
            episode_prices.append(last_price)
            episode_sec_volume += yl.get("secondary_volume", 0)

            shortfalls = yl.get("shortfalls", [0] * n_agents)
            penalties = yl.get("penalties", [0] * n_agents)
            for i in range(n_agents):
                sf = shortfalls[i] if i < len(shortfalls) else 0
                pn = penalties[i] if i < len(penalties) else 0
                total_shortfall[i] += sf
                total_penalty[i] += pn
                if sf < 1e-6:
                    compliant_years[i] += 1

            # Year-level CSV
            yr_row = {
                "episode": ep, "year": year,
                "cap": yl.get("cap", 0),
                "auction_volume": yl.get("auction_volume", 0),
                "tnac": yl.get("tnac", 0),
                "clearing_price": yl.get("clearing_price", 0),
                "secondary_price": yl.get("secondary_clearing", 0),
                "msr_reserve": yl.get("msr_reserve", 0),
            }
            for i in range(n_agents):
                def _get(key, default=0):
                    vals = yl.get(key, [default] * n_agents)
                    return vals[i] if i < len(vals) else default
                yr_row[f"alloc_A{i+1}"] = _get("allocations")
                yr_row[f"emissions_A{i+1}"] = _get("emissions")
                yr_row[f"green_frac_A{i+1}"] = _get("green_fracs")
                yr_row[f"shortfall_A{i+1}"] = _get("shortfalls")
                yr_row[f"penalty_A{i+1}"] = _get("penalties")
                yr_row[f"reward_A{i+1}"] = _get("rewards")
                yr_row[f"bid_price_A{i+1}"] = _get("bid_prices")
                yr_row[f"holdings_A{i+1}"] = _get("holdings")
            yr_writer.writerow(yr_row)

            if terminated:
                break

        all_rewards[ep] = total_rewards
        all_green[ep] = [env.companies[i].green_frac for i in range(n_agents)]
        all_compliance[ep] = compliant_years / n_years
        all_shortfall[ep] = total_shortfall
        all_prices.append(np.mean(episode_prices) if episode_prices else 0)

        # Episode CSV
        row = {
            "episode": ep,
            "clearing_price_last": f"{last_price:.1f}",
            "cap_last": f"{last_cap:.2f}",
            "tnac": f"{last_tnac:.2f}",
            "secondary_volume": f"{episode_sec_volume:.2f}",
            "mean_price": f"{np.mean(episode_prices):.1f}" if episode_prices else "0",
        }
        for i in range(n_agents):
            row[f"reward_A{i+1}"] = f"{total_rewards[i]:.2f}"
            row[f"green_frac_A{i+1}"] = f"{env.companies[i].green_frac:.3f}"
            row[f"shortfall_A{i+1}"] = f"{total_shortfall[i]:.3f}"
            row[f"penalty_A{i+1}"] = f"{total_penalty[i]:.2f}"
            row[f"compliance_rate_A{i+1}"] = f"{compliant_years[i]/n_years:.2f}"
        eval_writer.writerow(row)

        # Print per-episode summary
        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"Ep {ep+1:3d}/{args.n_episodes} | "
                f"price={last_price:.1f} | "
                f"rewards=[{', '.join(f'{r:.1f}' for r in total_rewards)}] | "
                f"green=[{', '.join(f'{g*100:.0f}%' for g in all_green[ep])}]"
            )

    eval_csv.close()
    yr_csv.close()

    # Final summary
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY ({args.n_episodes} episodes, greedy policy)")
    print(f"{'='*70}")
    print(f"{'Agent':<12s} {'Reward':>10s} {'Green%':>8s} {'Comply%':>9s} "
          f"{'Shortfall':>10s} {'Penalty':>9s}")
    print("-" * 60)

    archetypes = ["Coal/Fin", "Coal/Grn", "Gas/Fin", "Gas/Grn",
                  "Trans/Fin", "Trans/Grn", "Green/Fin", "Green/Grn"]
    for i in range(n_agents):
        print(f"A{i+1} {archetypes[i]:<8s} "
              f"{all_rewards[:,i].mean():+10.2f} "
              f"{all_green[:,i].mean()*100:8.1f} "
              f"{all_compliance[:,i].mean()*100:9.1f} "
              f"{all_shortfall[:,i].mean():10.3f} "
              f"{all_shortfall[:,i].mean()*100:9.1f}")

    print(f"\nAvg clearing price: {np.mean(all_prices):.1f} +/- {np.std(all_prices):.1f} EUR/t")
    print(f"\nFinal tech mixes:")
    for i, c in enumerate(env.companies[:n_agents]):
        mix_str = " | ".join([f"{tech_names[t]}:{c.mix[t]*100:.1f}%" for t in range(5)])
        print(f"  A{i+1}: {mix_str}")

    print(f"\nResults saved to:")
    print(f"  Episode log: {eval_path}")
    print(f"  Year log:    {yr_path}")


if __name__ == "__main__":
    main()
