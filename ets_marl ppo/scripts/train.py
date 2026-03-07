"""
train.py
========
Two-phase training loop for ETS MARL with PPO agents.

Each year:
  1. Phase 1: agents see market obs(18) → choose [bid, qty, invest_frac, tech_logits×3]
  2. Auction clears → agents see enriched obs(21)
  3. Phase 2: agents see auction results → choose [sec_price, sec_qty]
  4. Secondary market, compliance, rewards

PPO update happens at the end of each episode (on-policy).

Roadmap improvements wired here:
  P2: Entropy decay schedule — entropy_coef linearly decays from initial to final
      between decay_start and decay_end episodes.
  P3: Reward normalisation — per-agent RewardNormalizer called before buffer storage.
  P4: Shaping weight decay — communicated to environment via env.set_episode().
"""

import argparse
import csv
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.agents.ppo_agent import PPOAgent


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_agents(env: ETSEnvironment, config: dict, seed: int):
    """Instantiate one PPO agent per company."""
    obs1_dim = env.companies[0].obs_dim_phase1  # 18
    obs2_dim = env.companies[0].obs_dim_phase2  # 21

    aq = config["auction"]
    inv = config["investment"]

    # Phase 1: [bid_price, qty, invest_frac, tech_logit0, tech_logit1, tech_logit2]
    auction_low = np.array([
        aq["price_min"], 0.0, 0.0, -1.0, -1.0, -1.0
    ], dtype=np.float32)
    auction_high = np.array([
        aq["price_max"], aq["quantity_max"], inv["max_invest_frac"], 1.0, 1.0, 1.0
    ], dtype=np.float32)

    # Phase 2: [sec_price_multiplier, sec_qty]
    secondary_low = np.array([0.5, -aq["quantity_max"]], dtype=np.float32)
    secondary_high = np.array([2.0, aq["quantity_max"]], dtype=np.float32)

    agents = []
    for i in range(config["companies"]["n_agents"]):
        agent = PPOAgent(
            agent_id=i,
            obs_dim_phase1=obs1_dim,
            obs_dim_phase2=obs2_dim,
            auction_action_low=auction_low,
            auction_action_high=auction_high,
            secondary_action_low=secondary_low,
            secondary_action_high=secondary_high,
            config=config,
            seed=seed + i,
        )
        agents.append(agent)
    return agents


def _get_entropy_coef(episode: int, ppo_cfg: dict) -> float:
    """
    P2: Linearly decay entropy_coef from initial to final between
    decay_start and decay_end episodes.
    """
    coef_init = ppo_cfg.get("entropy_coef", 0.05)
    coef_final = ppo_cfg.get("entropy_coef_final", 0.005)
    decay_start = ppo_cfg.get("entropy_decay_start", 2000)
    decay_end = ppo_cfg.get("entropy_decay_end", 4000)

    if episode <= decay_start:
        return coef_init
    if episode >= decay_end:
        return coef_final
    frac = (episode - decay_start) / max(decay_end - decay_start, 1)
    return coef_init + frac * (coef_final - coef_init)


def train_one_seed(config: dict, seed: int):
    n_agents = config["companies"]["n_agents"]
    n_episodes = config["simulation"]["n_episodes"]
    n_years = config["simulation"]["n_years"]

    print(f"\n{'='*60}")
    print(f"Training — seed {seed}, {n_agents} agents, PPO, two-phase")
    print(f"Technology-specific mix | Real CapEx | Construction queues")
    print(f"P1-P4 roadmap improvements active")
    print(f"{'='*60}")

    env = ETSEnvironment(config, seed=seed)
    agents = build_agents(env, config, seed)

    ppo_cfg = config["ppo"]

    # --- CSV loggers ---
    results_dir = config["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Episode-level
    ep_path = os.path.join(results_dir, f"training_log_s{seed}.csv")
    ep_fields = ["episode", "clearing_price_last", "cap_last", "entropy_coef", "shaping_weight"]
    for i in range(n_agents):
        ep_fields += [f"reward_A{i+1}", f"green_frac_A{i+1}",
                      f"penalty_A{i+1}", f"actor_loss_A{i+1}", f"critic_loss_A{i+1}",
                      f"bid_price_A{i+1}"]
    ep_fields += ["secondary_volume", "secondary_avg_price"]
    ep_csv = open(ep_path, "w", newline="")
    ep_writer = csv.DictWriter(ep_csv, fieldnames=ep_fields)
    ep_writer.writeheader()

    # Year-level
    yr_path = os.path.join(results_dir, f"year_log_s{seed}.csv")
    yr_fields = ["episode", "year", "cap", "auction_volume", "tnac",
                 "clearing_price", "secondary_price", "msr_reserve"]
    for i in range(n_agents):
        yr_fields += [f"alloc_A{i+1}", f"emissions_A{i+1}", f"trade_qty_A{i+1}",
                      f"trade_cost_A{i+1}", f"green_frac_A{i+1}", f"penalty_A{i+1}",
                      f"reward_A{i+1}", f"holdings_A{i+1}", f"invest_cost_A{i+1}",
                      f"bid_price_A{i+1}"]
    yr_csv = open(yr_path, "w", newline="")
    yr_writer = csv.DictWriter(yr_csv, fieldnames=yr_fields)
    yr_writer.writeheader()

    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]
    best_total_reward = -np.inf

    for episode in range(n_episodes):
        # P2: Update entropy coefficients for all agents
        entropy_coef = _get_entropy_coef(episode, ppo_cfg)
        for agent in agents:
            agent.set_entropy_coef(entropy_coef)

        # P4: Communicate episode to environment for shaping weight + lock-in activation
        env.set_episode(episode)

        obs1, _ = env.reset(seed=seed + episode * 1000)
        total_rewards = np.zeros(n_agents)

        for year in range(n_years):
            # === PHASE 1: Auction + Investment ===
            auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
            auction_raws = []
            auction_logps = []

            for i in range(n_agents):
                action, raw, logp = agents[i].select_auction_action(obs1[i])
                auction_actions[i] = action
                auction_raws.append(raw)
                auction_logps.append(logp)

            obs2, auction_info = env.step_auction(auction_actions)

            # === PHASE 2: Secondary Market ===
            secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
            secondary_raws = []
            secondary_logps = []

            for i in range(n_agents):
                action, raw, logp = agents[i].select_secondary_action(obs2[i])
                secondary_actions[i] = action
                secondary_raws.append(raw)
                secondary_logps.append(logp)

            obs1_next, rewards, terminated, truncated, info = env.step_secondary(
                secondary_actions)

            # P3: Normalise rewards per-agent before storing in buffer
            normalised_rewards = np.array([
                agents[i].normalize_reward(rewards[i]) for i in range(n_agents)
            ], dtype=np.float32)

            # Store transitions with normalised rewards
            for i in range(n_agents):
                value = agents[i].estimate_value(obs2[i])
                agents[i].store_transition(
                    obs1=obs1[i], obs2=obs2[i],
                    auc_raw=auction_raws[i], sec_raw=secondary_raws[i],
                    auc_lp=auction_logps[i], sec_lp=secondary_logps[i],
                    reward=normalised_rewards[i], done=terminated, value=value,
                )

            total_rewards += rewards  # log RAW rewards for diagnostics

            # --- Year-level logging ---
            yl = info.get("year_log", {})
            bid_prices_this_year = yl.get("bid_prices", [0] * n_agents)
            yr_row = {
                "episode": episode, "year": year,
                "cap": yl.get("cap", 0), "auction_volume": yl.get("auction_volume", 0),
                "tnac": yl.get("tnac", 0), "clearing_price": yl.get("clearing_price", 0),
                "secondary_price": yl.get("secondary_clearing", 0),
                "msr_reserve": yl.get("msr_reserve", 0),
            }
            for i in range(n_agents):
                for key, log_key in [("alloc", "allocations"), ("emissions", "emissions"),
                                     ("trade_qty", "trade_qtys"), ("trade_cost", "trade_costs"),
                                     ("green_frac", "green_fracs"), ("penalty", "penalties"),
                                     ("reward", "rewards"), ("holdings", "holdings"),
                                     ("invest_cost", "invest_costs")]:
                    vals = yl.get(log_key, [0]*n_agents)
                    yr_row[f"{key}_A{i+1}"] = vals[i] if i < len(vals) else 0
                yr_row[f"bid_price_A{i+1}"] = (
                    bid_prices_this_year[i] if i < len(bid_prices_this_year) else 0
                )
            yr_writer.writerow(yr_row)

            obs1 = obs1_next
            if terminated:
                break

        # === PPO Update (end of episode) ===
        latest_losses = []
        for i in range(n_agents):
            loss = agents[i].update(last_value=0.0)
            latest_losses.append(loss)

        # --- Episode-level logging ---
        last_log = env.episode_log[-1] if env.episode_log else {}

        total_sec_vol = sum(
            sum(q for q in yl.get("trade_qtys", []) if q > 0)
            for yl in env.episode_log
        )
        total_sec_value = sum(
            sum(abs(c) for c in yl.get("trade_costs", []))
            for yl in env.episode_log
        )
        avg_sec_price = total_sec_value / max(total_sec_vol, 1e-6) / 2

        # Average bid price per agent across the episode
        avg_bid_per_agent = []
        for i in range(n_agents):
            bids_this_ep = [
                yl["bid_prices"][i]
                for yl in env.episode_log
                if "bid_prices" in yl and i < len(yl["bid_prices"])
            ]
            avg_bid_per_agent.append(np.mean(bids_this_ep) if bids_this_ep else 0.0)

        ep_row = {
            "episode": episode,
            "clearing_price_last": last_log.get("clearing_price", 0),
            "cap_last": last_log.get("cap", 0),
            "entropy_coef": round(entropy_coef, 5),
            "shaping_weight": round(env.shaping_weight, 4),
            "secondary_volume": round(total_sec_vol, 4),
            "secondary_avg_price": round(avg_sec_price, 2),
        }
        for i in range(n_agents):
            ep_row[f"reward_A{i+1}"] = round(total_rewards[i], 4)
            ep_row[f"green_frac_A{i+1}"] = round(
                last_log.get("green_fracs", [0]*n_agents)[i], 4)
            ep_row[f"penalty_A{i+1}"] = round(
                sum(yl.get("penalties", [0]*n_agents)[i] for yl in env.episode_log), 6)
            ep_row[f"bid_price_A{i+1}"] = round(avg_bid_per_agent[i], 2)
            if latest_losses[i]:
                ep_row[f"actor_loss_A{i+1}"] = round(latest_losses[i]["actor_loss"], 6)
                ep_row[f"critic_loss_A{i+1}"] = round(latest_losses[i]["critic_loss"], 6)
            else:
                ep_row[f"actor_loss_A{i+1}"] = 0.0
                ep_row[f"critic_loss_A{i+1}"] = 0.0

        ep_writer.writerow(ep_row)

        # Console — include bid prices so we can monitor P1 fix
        if episode % log_interval == 0:
            r_str = " ".join([f"A{i+1}:{total_rewards[i]:7.2f}" for i in range(n_agents)])
            g_str = " ".join([f"{last_log.get('green_fracs', [0]*n_agents)[i]*100:.0f}%"
                             for i in range(n_agents)])
            b_str = " ".join([f"A{i+1}:{avg_bid_per_agent[i]:.0f}" for i in range(n_agents)])
            price = last_log.get("clearing_price", 0)
            print(f"Ep {episode:5d} | price={price:6.1f} | bids=[{b_str}] | "
                  f"green=[{g_str}] | {r_str} | ent={entropy_coef:.4f}")

        # Checkpointing
        if episode % save_interval == 0:
            ckpt_dir = os.path.join(results_dir, f"checkpoints_s{seed}")
            os.makedirs(ckpt_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                agent.save(os.path.join(ckpt_dir, f"agent_{i}_ep{episode}.pt"))

        ep_total = total_rewards.sum()
        if ep_total > best_total_reward:
            best_total_reward = ep_total
            ckpt_dir = os.path.join(results_dir, f"checkpoints_s{seed}")
            os.makedirs(ckpt_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                agent.save(os.path.join(ckpt_dir, f"agent_{i}_best.pt"))

    ep_csv.close()
    yr_csv.close()
    print(f"\nDone — seed {seed}. Logs: {ep_path}, {yr_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ETS MARL (PPO, two-phase)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    config = load_config(args.config)
    for seed in args.seed:
        train_one_seed(config, seed)


if __name__ == "__main__":
    main()
