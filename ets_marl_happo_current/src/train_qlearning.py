"""
train_qlearning.py
==================
Tabular Q-learning training loop for the ETS MARL environment.

Usage:
    python src/train_qlearning.py --config configs/default.yaml \
                                  --ql-config configs/qlearning.yaml \
                                  [--seed 42]

Instantiates the same ETSEnvironment as PPO training, but replaces
neural-network agents with tabular Q-learning agents that discretize
the state space and use predefined action profiles.
"""

import argparse
import csv
import os
import pickle
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
from src.agents.q_learning_agent import QLearningAgent, StateDiscretizer


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


def train_qlearning(config: dict, ql_config: dict, seed: int):
    """Run Q-learning training loop."""

    # Merge Q-learning overrides into base config
    config = merge_configs(config, ql_config)

    ql_params = config.get("qlearning", {})
    n_episodes = ql_params.get("n_episodes", 5000)
    alpha = ql_params.get("alpha", 0.1)
    gamma = ql_params.get("gamma", 0.95)
    eps_start = ql_params.get("epsilon_start", 1.0)
    eps_end = ql_params.get("epsilon_end", 0.05)
    eps_decay_frac = ql_params.get("epsilon_decay_frac", 0.7)
    eps_decay_episodes = int(n_episodes * eps_decay_frac)

    n_agents = config["companies"]["n_agents"]
    n_years = config["simulation"]["n_years"]

    print(f"\n{'='*70}")
    print(f"Q-Learning Baseline Training — seed {seed}")
    print(f"  Episodes: {n_episodes}  |  Alpha: {alpha}  |  Gamma: {gamma}")
    print(f"  Epsilon: {eps_start:.2f} -> {eps_end:.2f} over {eps_decay_episodes} episodes")
    print(f"  Agents: {n_agents}  |  Years/episode: {n_years}")
    print(f"  States: {StateDiscretizer.N_STATES}  |  Actions: 6×4 profiles")
    print(f"{'='*70}\n")

    # Create environment
    env = ETSEnvironment(config, seed=seed)

    # Create Q-learning agents
    agents = [
        QLearningAgent(agent_id=i, alpha=alpha, gamma=gamma, seed=seed + i)
        for i in range(n_agents)
    ]

    # Setup logging
    results_dir = config.get("logging", {}).get("results_dir", "results/qlearning/")
    os.makedirs(results_dir, exist_ok=True)
    log_interval = config.get("logging", {}).get("log_interval", 50)

    # CSV episode logger
    ep_path = os.path.join(results_dir, f"ql_training_log_s{seed}.csv")
    ep_fields = ["episode", "epsilon", "clearing_price_last", "cap_last"]
    for i in range(n_agents):
        ep_fields += [f"reward_A{i+1}", f"green_frac_A{i+1}",
                      f"compliance_rate_A{i+1}", f"shortfall_A{i+1}",
                      f"a1_profile_A{i+1}", f"a2_profile_A{i+1}"]
    ep_fields += ["secondary_volume", "mean_price", "tnac"]
    ep_csv = open(ep_path, "w", newline="")
    ep_writer = csv.DictWriter(ep_csv, fieldnames=ep_fields)
    ep_writer.writeheader()

    # CSV year-level logger
    yr_path = os.path.join(results_dir, f"ql_year_log_s{seed}.csv")
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

    # Tracking buffers for console output
    reward_buffer = np.zeros((log_interval, n_agents))
    price_buffer = []
    green_buffer = np.zeros((log_interval, n_agents))
    compliance_buffer = np.zeros((log_interval, n_agents))
    shortfall_buffer = np.zeros((log_interval, n_agents))

    train_t0 = time.time()

    for episode in range(n_episodes):
        # Epsilon schedule: linear decay
        if episode < eps_decay_episodes:
            epsilon = eps_start + (eps_end - eps_start) * (episode / eps_decay_episodes)
        else:
            epsilon = eps_end

        # Disable reward shaping in environment
        env.set_episode(episode)

        obs1, _ = env.reset(seed=seed + episode * 1000)
        total_rewards = np.zeros(n_agents)

        # Per-year tracking
        year_states = [[] for _ in range(n_agents)]
        year_a1 = [[] for _ in range(n_agents)]
        year_a2 = [[] for _ in range(n_agents)]
        episode_shortfalls = np.zeros(n_agents)
        episode_compliant_years = np.zeros(n_agents)
        last_clearing_price = 0.0
        last_cap = 0.0
        last_tnac = 0.0
        episode_sec_volume = 0.0
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
                    obs1[i], env.companies[i], price_ma3, config, epsilon=epsilon)
                auction_actions[i] = action_vec
                a1_indices[i] = a1_idx
                year_states[i].append(state)
                year_a1[i].append(a1_idx)

            obs2, auction_info = env.step_auction(auction_actions)

            # === PHASE 2: Secondary Market ===
            secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
            a2_indices = np.zeros(n_agents, dtype=int)

            for i in range(n_agents):
                action_vec, a2_idx = agents[i].select_secondary_action(
                    obs2[i], env.companies[i], env._phase1_clearing_price,
                    config, a1_idx=a1_indices[i], epsilon=epsilon)
                secondary_actions[i] = action_vec
                a2_indices[i] = a2_idx
                year_a2[i].append(a2_idx)

            obs1_next, rewards, terminated, _, info = env.step_secondary(secondary_actions)
            total_rewards += rewards

            # Q-learning update for each agent
            for i in range(n_agents):
                state = year_states[i][-1]
                next_state = agents[i].discretizer.discretize(
                    obs1_next[i], env.companies[i])
                agents[i].update(state, a1_indices[i], a2_indices[i],
                                 rewards[i], next_state, terminated)

            # Year-level logging
            yl = info.get("year_log", {})
            last_clearing_price = yl.get("clearing_price", 0)
            last_cap = yl.get("cap", 0)
            last_tnac = yl.get("tnac", 0)
            episode_prices.append(last_clearing_price)
            episode_sec_volume += yl.get("secondary_volume", 0)
            last_a1_profiles = a1_indices.copy()
            last_a2_profiles = a2_indices.copy()

            # Track compliance
            shortfalls = yl.get("shortfalls", [0] * n_agents)
            for i in range(n_agents):
                sf = shortfalls[i] if i < len(shortfalls) else 0
                episode_shortfalls[i] += sf
                if sf < 1e-6:
                    episode_compliant_years[i] += 1

            # Year CSV
            yr_row = {
                "episode": episode, "year": year,
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

            obs1 = obs1_next
            if terminated:
                break

        # Episode-level logging
        buf_idx = episode % log_interval
        reward_buffer[buf_idx] = total_rewards
        price_buffer.append(np.mean(episode_prices) if episode_prices else 0)
        green_buffer[buf_idx] = [env.companies[i].green_frac for i in range(n_agents)]
        compliance_buffer[buf_idx] = episode_compliant_years / n_years
        shortfall_buffer[buf_idx] = episode_shortfalls

        ep_row = {
            "episode": episode, "epsilon": f"{epsilon:.3f}",
            "clearing_price_last": f"{last_clearing_price:.1f}",
            "cap_last": f"{last_cap:.2f}",
            "secondary_volume": f"{episode_sec_volume:.2f}",
            "mean_price": f"{np.mean(episode_prices):.1f}" if episode_prices else "0",
            "tnac": f"{last_tnac:.2f}",
        }
        for i in range(n_agents):
            ep_row[f"reward_A{i+1}"] = f"{total_rewards[i]:.2f}"
            ep_row[f"green_frac_A{i+1}"] = f"{env.companies[i].green_frac:.3f}"
            ep_row[f"compliance_rate_A{i+1}"] = f"{episode_compliant_years[i]/n_years:.2f}"
            ep_row[f"shortfall_A{i+1}"] = f"{episode_shortfalls[i]:.3f}"
            ep_row[f"a1_profile_A{i+1}"] = str(last_a1_profiles[i])
            ep_row[f"a2_profile_A{i+1}"] = str(last_a2_profiles[i])
        ep_writer.writerow(ep_row)

        # Flush CSVs periodically
        flush_interval = config.get("logging", {}).get("csv_flush_interval", 500)
        if (episode + 1) % flush_interval == 0:
            ep_csv.flush()
            yr_csv.flush()

        # Console output
        if (episode + 1) % log_interval == 0:
            elapsed = time.time() - train_t0
            eps_per_sec = (episode + 1) / elapsed
            eta = (n_episodes - episode - 1) / max(eps_per_sec, 0.01)
            n_buf = min(episode + 1, log_interval)

            avg_reward = reward_buffer[:n_buf].mean(axis=0)
            avg_green = green_buffer[:n_buf].mean(axis=0)
            avg_compliance = compliance_buffer[:n_buf].mean(axis=0)
            avg_shortfall = shortfall_buffer[:n_buf].mean(axis=0)
            avg_price = np.mean(price_buffer[-n_buf:]) if price_buffer else 0

            print(f"Ep {episode+1:5d}/{n_episodes} | "
                  f"eps={epsilon:.3f} | "
                  f"price={avg_price:.1f} | "
                  f"elapsed={_format_hms(elapsed)} ETA={_format_hms(eta)}")

            # Per-agent summary
            for i in range(n_agents):
                print(f"  A{i+1}: rew={avg_reward[i]:+7.2f}  "
                      f"green={avg_green[i]*100:5.1f}%  "
                      f"comply={avg_compliance[i]*100:5.1f}%  "
                      f"short={avg_shortfall[i]:.3f}Mt")

            print()

        # Dump Q-tables every 100 episodes
        if (episode + 1) % 100 == 0:
            ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            qtable_path = os.path.join(ckpt_dir, f"qtables_s{seed}_ep{episode+1}.pkl")
            qtables = {f"agent_{i}": agents[i].q_table.copy() for i in range(n_agents)}
            with open(qtable_path, "wb") as f:
                pickle.dump(qtables, f)

            # Print top-5 Q-values per agent
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
    final_path = os.path.join(results_dir, f"qtables_s{seed}_final.pkl")
    qtables = {f"agent_{i}": agents[i].q_table.copy() for i in range(n_agents)}
    with open(final_path, "wb") as f:
        pickle.dump(qtables, f)

    ep_csv.close()
    yr_csv.close()

    print(f"\n{'='*70}")
    print(f"Q-Learning training complete — seed {seed}")
    print(f"  Final Q-tables saved to {final_path}")
    print(f"  Episode log: {ep_path}")
    print(f"  Year log: {yr_path}")
    print(f"{'='*70}")

    return agents, config


def evaluate_qlearning(agents, config, seed, n_eval_episodes=100):
    """
    Run greedy evaluation episodes and log results.

    Parameters
    ----------
    agents : list of QLearningAgent
    config : dict
    seed : int
    n_eval_episodes : int
    """
    n_agents = config["companies"]["n_agents"]
    n_years = config["simulation"]["n_years"]
    results_dir = config.get("logging", {}).get("results_dir", "results/qlearning/")

    print(f"\n{'='*70}")
    print(f"Q-Learning Evaluation — {n_eval_episodes} greedy episodes")
    print(f"{'='*70}")

    env = ETSEnvironment(config, seed=seed + 99999)

    # CSV for eval results
    eval_path = os.path.join(results_dir, f"ql_eval_log_s{seed}.csv")
    eval_fields = ["episode", "clearing_price_last"]
    for i in range(n_agents):
        eval_fields += [f"reward_A{i+1}", f"green_frac_A{i+1}",
                        f"compliance_rate_A{i+1}", f"shortfall_A{i+1}"]
    eval_fields += ["tnac"]
    eval_csv = open(eval_path, "w", newline="")
    eval_writer = csv.DictWriter(eval_csv, fieldnames=eval_fields)
    eval_writer.writeheader()

    all_rewards = np.zeros((n_eval_episodes, n_agents))
    all_green = np.zeros((n_eval_episodes, n_agents))
    all_compliance = np.zeros((n_eval_episodes, n_agents))
    all_shortfall = np.zeros((n_eval_episodes, n_agents))
    all_prices = []

    for ep in range(n_eval_episodes):
        env.set_episode(99999)  # no shaping decay
        obs1, _ = env.reset(seed=seed + 99999 + ep * 1000)
        total_rewards = np.zeros(n_agents)
        compliant_years = np.zeros(n_agents)
        total_shortfall = np.zeros(n_agents)
        last_price = 0.0
        last_tnac = 0.0

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
            last_tnac = yl.get("tnac", 0)
            shortfalls = yl.get("shortfalls", [0] * n_agents)
            for i in range(n_agents):
                sf = shortfalls[i] if i < len(shortfalls) else 0
                total_shortfall[i] += sf
                if sf < 1e-6:
                    compliant_years[i] += 1

            if terminated:
                break

        all_rewards[ep] = total_rewards
        all_green[ep] = [env.companies[i].green_frac for i in range(n_agents)]
        all_compliance[ep] = compliant_years / n_years
        all_shortfall[ep] = total_shortfall
        all_prices.append(last_price)

        row = {"episode": ep, "clearing_price_last": f"{last_price:.1f}",
               "tnac": f"{last_tnac:.2f}"}
        for i in range(n_agents):
            row[f"reward_A{i+1}"] = f"{total_rewards[i]:.2f}"
            row[f"green_frac_A{i+1}"] = f"{env.companies[i].green_frac:.3f}"
            row[f"compliance_rate_A{i+1}"] = f"{compliant_years[i]/n_years:.2f}"
            row[f"shortfall_A{i+1}"] = f"{total_shortfall[i]:.3f}"
        eval_writer.writerow(row)

    eval_csv.close()

    # Summary
    print(f"\nEvaluation Results (mean ± std over {n_eval_episodes} episodes):")
    print(f"  Avg clearing price: {np.mean(all_prices):.1f} ± {np.std(all_prices):.1f} €/t")
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
    }


def main():
    parser = argparse.ArgumentParser(description="Q-Learning baseline training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Base environment config")
    parser.add_argument("--ql-config", type=str, default="configs/qlearning.yaml",
                        help="Q-learning specific config overrides")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load Q-tables and evaluate")
    parser.add_argument("--qtable-path", type=str, default=None,
                        help="Path to Q-table pickle (for --eval-only)")
    args = parser.parse_args()

    config = load_config(args.config)
    ql_config = load_config(args.ql_config)

    if args.eval_only:
        merged = merge_configs(config, ql_config)
        n_agents = merged["companies"]["n_agents"]
        ql_params = merged.get("qlearning", {})

        # Load Q-tables
        qtable_path = args.qtable_path
        if qtable_path is None:
            results_dir = merged.get("logging", {}).get("results_dir", "results/qlearning/")
            qtable_path = os.path.join(results_dir, f"qtables_s{args.seed}_final.pkl")

        print(f"Loading Q-tables from {qtable_path}")
        with open(qtable_path, "rb") as f:
            qtables = pickle.load(f)

        agents = [
            QLearningAgent(agent_id=i,
                           alpha=ql_params.get("alpha", 0.1),
                           gamma=ql_params.get("gamma", 0.95),
                           seed=args.seed + i)
            for i in range(n_agents)
        ]
        for i in range(n_agents):
            agents[i].q_table = qtables[f"agent_{i}"]

        eval_cfg = merged.get("evaluation", {})
        evaluate_qlearning(agents, merged, args.seed,
                           n_eval_episodes=eval_cfg.get("n_episodes", 100))
    else:
        agents, merged_config = train_qlearning(config, ql_config, args.seed)

        # Run evaluation after training
        eval_cfg = merged_config.get("evaluation", {})
        evaluate_qlearning(agents, merged_config, args.seed,
                           n_eval_episodes=eval_cfg.get("n_episodes", 100))


if __name__ == "__main__":
    main()
