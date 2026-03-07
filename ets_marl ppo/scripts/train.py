"""
train.py
========
Two-phase training loop for ETS MARL with PPO agents.

Each year:
  1. Phase 1: agents see market obs → choose [bid, qty, invest_frac, tech_logits x3]
  2. Auction clears → agents see enriched obs
  3. Phase 2: agents see auction results → choose [sec_price, sec_qty]
  4. Secondary market, compliance, rewards

PPO update happens at the end of each episode (on-policy).

Roadmap improvements wired here:
  P2: Condition-based entropy decay — begins only after price stabilises
      (price_std < 40 over last 100 eps) AND green fracs are non-decreasing
      for 50 consecutive episodes.
  P3: Reward normalisation — per-agent RewardNormalizer called before buffer storage.
  P4: Shaping weight decay — communicated to environment via env.set_episode().
  Agent cycling (Option A): only the active agent (episode % n_agents) calls
      update() each episode; others collect experience.
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


class EntropyConditionTracker:
    """
    P2: Condition-based entropy decay.

    Decay begins only after BOTH conditions are met:
      1. price_std < 40 €/t over the last `price_window` episodes.
      2. No agent green fraction has decreased for `green_window` consecutive episodes.

    Once conditions trigger, entropy decays linearly from coef_init to coef_final
    over `decay_window` episodes.
    """

    def __init__(self, ppo_cfg: dict, n_agents: int):
        self.coef_init = ppo_cfg.get("entropy_coef", 0.05)
        self.coef_final = ppo_cfg.get("entropy_coef_final", 0.005)
        self.decay_window = ppo_cfg.get("entropy_decay_window", 2000)
        self.price_window = 100
        self.green_window = 50
        self.price_std_threshold = 40.0
        self.n_agents = n_agents

        self._price_history = []           # last N clearing prices
        self._green_stable_count = 0       # consecutive episodes with non-decreasing green
        self._prev_green_fracs = None
        self._decay_start_episode = None   # set when conditions first met

    def update(self, episode: int, clearing_price: float, green_fracs: list) -> float:
        """Update trackers and return current entropy coef."""
        self._price_history.append(clearing_price)
        if len(self._price_history) > self.price_window:
            self._price_history.pop(0)

        # Green stability: check no agent decreased
        if self._prev_green_fracs is not None:
            all_stable = all(
                green_fracs[i] >= self._prev_green_fracs[i] - 1e-4
                for i in range(self.n_agents)
            )
            if all_stable:
                self._green_stable_count += 1
            else:
                self._green_stable_count = 0
        self._prev_green_fracs = list(green_fracs)

        # Check if conditions are met for the first time
        if self._decay_start_episode is None:
            price_cond = (
                len(self._price_history) >= self.price_window and
                float(np.std(self._price_history)) < self.price_std_threshold
            )
            green_cond = self._green_stable_count >= self.green_window
            if price_cond and green_cond:
                self._decay_start_episode = episode

        # Compute coef
        if self._decay_start_episode is None:
            return self.coef_init
        episodes_since = episode - self._decay_start_episode
        if episodes_since >= self.decay_window:
            return self.coef_final
        frac = episodes_since / max(self.decay_window, 1)
        return self.coef_init + frac * (self.coef_final - self.coef_init)

    @property
    def decay_triggered(self) -> bool:
        return self._decay_start_episode is not None


def train_one_seed(config: dict, seed: int):
    n_agents = config["companies"]["n_agents"]
    n_episodes = config["simulation"]["n_episodes"]
    n_years = config["simulation"]["n_years"]

    print(f"\n{'='*60}")
    print(f"Training — seed {seed}, {n_agents} agents, PPO, two-phase")
    print(f"Technology-specific mix | Real CapEx | Construction queues")
    print(f"P1-P8 roadmap improvements active")
    print(f"{'='*60}")

    env = ETSEnvironment(config, seed=seed)
    agents = build_agents(env, config, seed)

    ppo_cfg = config["ppo"]
    cycling_cfg = config.get("agent_cycling", {})
    cycling_enabled = cycling_cfg.get("enabled", False)
    cycling_soft = cycling_cfg.get("soft", False)
    cycling_lr_scale = cycling_cfg.get("soft_lr_scale", 0.1)

    # Condition-based entropy tracker
    entropy_tracker = EntropyConditionTracker(ppo_cfg, n_agents)

    # --- CSV loggers ---
    results_dir = config["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Episode-level
    ep_path = os.path.join(results_dir, f"training_log_s{seed}.csv")
    ep_fields = ["episode", "clearing_price_last", "cap_last", "entropy_coef",
                 "shaping_weight", "entropy_decay_triggered", "active_agent"]
    for i in range(n_agents):
        ep_fields += [f"reward_A{i+1}", f"green_frac_A{i+1}", f"delta_green_A{i+1}",
                      f"penalty_A{i+1}", f"shortfall_A{i+1}", f"queue_size_A{i+1}",
                      f"actor_loss_A{i+1}", f"critic_loss_A{i+1}", f"bid_price_A{i+1}"]
    ep_fields += ["secondary_volume", "secondary_avg_price", "secondary_match_rate"]
    ep_csv = open(ep_path, "w", newline="")
    ep_writer = csv.DictWriter(ep_csv, fieldnames=ep_fields)
    ep_writer.writeheader()

    # Year-level
    yr_path = os.path.join(results_dir, f"year_log_s{seed}.csv")
    yr_fields = ["episode", "year", "cap", "auction_volume", "tnac",
                 "clearing_price", "secondary_price", "msr_reserve"]
    for i in range(n_agents):
        yr_fields += [f"bank_start_A{i+1}", f"alloc_A{i+1}", f"emissions_A{i+1}",
                      f"trade_qty_A{i+1}", f"trade_cost_A{i+1}", f"green_frac_A{i+1}",
                      f"delta_green_A{i+1}", f"shortfall_A{i+1}", f"penalty_A{i+1}",
                      f"reward_A{i+1}", f"holdings_A{i+1}", f"invest_cost_A{i+1}",
                      f"bid_price_A{i+1}", f"queue_size_A{i+1}",
                      f"emission_shock_A{i+1}", f"cf_shock_A{i+1}",  # P5/P6
                      f"cancellation_A{i+1}"]                         # P6
    yr_csv = open(yr_path, "w", newline="")
    yr_writer = csv.DictWriter(yr_csv, fieldnames=yr_fields)
    yr_writer.writeheader()

    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]
    best_total_reward = -np.inf

    for episode in range(n_episodes):
        # Agent cycling: which agent updates this episode
        active_agent_idx = episode % n_agents if cycling_enabled else None

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
            yr_row = {
                "episode": episode, "year": year,
                "cap": yl.get("cap", 0), "auction_volume": yl.get("auction_volume", 0),
                "tnac": yl.get("tnac", 0), "clearing_price": yl.get("clearing_price", 0),
                "secondary_price": yl.get("secondary_clearing", 0),
                "msr_reserve": yl.get("msr_reserve", 0),
            }
            for i in range(n_agents):
                def _get(log_key, default=0):
                    vals = yl.get(log_key, [default] * n_agents)
                    return vals[i] if i < len(vals) else default
                yr_row[f"bank_start_A{i+1}"] = _get("bank_start")
                yr_row[f"alloc_A{i+1}"] = _get("allocations")
                yr_row[f"emissions_A{i+1}"] = _get("emissions")
                yr_row[f"trade_qty_A{i+1}"] = _get("trade_qtys")
                yr_row[f"trade_cost_A{i+1}"] = _get("trade_costs")
                yr_row[f"green_frac_A{i+1}"] = _get("green_fracs")
                yr_row[f"delta_green_A{i+1}"] = _get("delta_greens")
                yr_row[f"shortfall_A{i+1}"] = _get("shortfalls")
                yr_row[f"penalty_A{i+1}"] = _get("penalties")
                yr_row[f"reward_A{i+1}"] = _get("rewards")
                yr_row[f"holdings_A{i+1}"] = _get("holdings")
                yr_row[f"invest_cost_A{i+1}"] = _get("invest_costs")
                yr_row[f"bid_price_A{i+1}"] = _get("bid_prices")
                yr_row[f"queue_size_A{i+1}"] = _get("queue_sizes")
                yr_row[f"emission_shock_A{i+1}"] = _get("emission_shocks")   # P5
                yr_row[f"cf_shock_A{i+1}"] = _get("cf_shocks")               # P6
                yr_row[f"cancellation_A{i+1}"] = _get("cancellations")       # P6
            yr_writer.writerow(yr_row)

            obs1 = obs1_next
            if terminated:
                break

        # === PPO Update (end of episode) ===
        # Condition-based entropy decay: update tracker with last year's data
        last_log_ep = env.episode_log[-1] if env.episode_log else {}
        last_price = last_log_ep.get("clearing_price", 0.0)
        last_greens = last_log_ep.get("green_fracs", [0.0] * n_agents)
        entropy_coef = entropy_tracker.update(episode, last_price, last_greens)
        for agent in agents:
            agent.set_entropy_coef(entropy_coef)

        latest_losses = []
        for i in range(n_agents):
            if cycling_enabled:
                if i == active_agent_idx:
                    # Active agent: full update
                    loss = agents[i].update(last_value=0.0)
                elif cycling_soft:
                    # Soft cycling: update with reduced lr
                    orig_lr = agents[i].optimizer.param_groups[0]["lr"]
                    for pg in agents[i].optimizer.param_groups:
                        pg["lr"] = orig_lr * cycling_lr_scale
                    loss = agents[i].update(last_value=0.0)
                    for pg in agents[i].optimizer.param_groups:
                        pg["lr"] = orig_lr
                else:
                    # Hard freeze: discard buffer without updating
                    agents[i].buffer.clear()
                    loss = None
            else:
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

        # Secondary match rate: years where at least one trade occurred
        years_with_trades = sum(
            1 for yl in env.episode_log
            if any(abs(q) > 1e-6 for q in yl.get("trade_qtys", []))
        )
        sec_match_rate = years_with_trades / max(n_years, 1)

        # Average bid price per agent across the episode
        avg_bid_per_agent = []
        for i in range(n_agents):
            bids_this_ep = [
                yl["bid_prices"][i]
                for yl in env.episode_log
                if "bid_prices" in yl and i < len(yl["bid_prices"])
            ]
            avg_bid_per_agent.append(np.mean(bids_this_ep) if bids_this_ep else 0.0)

        # Per-agent episode aggregates
        ep_total_shortfalls = [
            sum(yl.get("shortfalls", [0]*n_agents)[i] for yl in env.episode_log)
            for i in range(n_agents)
        ]
        ep_total_penalties = [
            sum(yl.get("penalties", [0]*n_agents)[i] for yl in env.episode_log)
            for i in range(n_agents)
        ]
        ep_delta_greens = [
            sum(yl.get("delta_greens", [0]*n_agents)[i] for yl in env.episode_log)
            for i in range(n_agents)
        ]
        ep_avg_queue = [
            np.mean([yl.get("queue_sizes", [0]*n_agents)[i] for yl in env.episode_log])
            for i in range(n_agents)
        ]

        ep_row = {
            "episode": episode,
            "clearing_price_last": last_log.get("clearing_price", 0),
            "cap_last": last_log.get("cap", 0),
            "entropy_coef": round(entropy_coef, 5),
            "shaping_weight": round(env.shaping_weight, 4),
            "entropy_decay_triggered": int(entropy_tracker.decay_triggered),
            "active_agent": active_agent_idx if cycling_enabled else -1,
            "secondary_volume": round(total_sec_vol, 4),
            "secondary_avg_price": round(avg_sec_price, 2),
            "secondary_match_rate": round(sec_match_rate, 3),
        }
        for i in range(n_agents):
            ep_row[f"reward_A{i+1}"] = round(total_rewards[i], 4)
            ep_row[f"green_frac_A{i+1}"] = round(
                last_log.get("green_fracs", [0]*n_agents)[i], 4)
            ep_row[f"delta_green_A{i+1}"] = round(ep_delta_greens[i], 5)
            ep_row[f"penalty_A{i+1}"] = round(ep_total_penalties[i], 6)
            ep_row[f"shortfall_A{i+1}"] = round(ep_total_shortfalls[i], 6)
            ep_row[f"queue_size_A{i+1}"] = round(ep_avg_queue[i], 2)
            ep_row[f"bid_price_A{i+1}"] = round(avg_bid_per_agent[i], 2)
            if latest_losses[i]:
                ep_row[f"actor_loss_A{i+1}"] = round(latest_losses[i]["actor_loss"], 6)
                ep_row[f"critic_loss_A{i+1}"] = round(latest_losses[i]["critic_loss"], 6)
            else:
                ep_row[f"actor_loss_A{i+1}"] = 0.0
                ep_row[f"critic_loss_A{i+1}"] = 0.0

        ep_writer.writerow(ep_row)

        # Console diagnostics
        if episode % log_interval == 0:
            price  = last_log.get("clearing_price", 0)
            cap    = last_log.get("cap", 0)
            tnac   = last_log.get("tnac", 0)
            sec_p  = last_log.get("secondary_clearing", 0)

            # Last-year per-agent arrays (most recent state)
            ly_bank   = last_log.get("bank_start",   [0.0] * n_agents)
            ly_alloc  = last_log.get("allocations",  [0.0] * n_agents)
            ly_emiss  = last_log.get("emissions",    [0.0] * n_agents)
            ly_secnet = last_log.get("trade_qtys",   [0.0] * n_agents)
            ly_hold   = last_log.get("holdings",     [0.0] * n_agents)
            ly_short  = last_log.get("shortfalls",   [0.0] * n_agents)
            ly_pen    = last_log.get("penalties",    [0.0] * n_agents)
            ly_green  = last_log.get("green_fracs",  [0.0] * n_agents)
            ly_dg     = last_log.get("delta_greens", [0.0] * n_agents)

            cyc_str   = f" [cyc=A{active_agent_idx+1}]" if cycling_enabled else ""
            decay_str = " [ENT-DECAY]" if entropy_tracker.decay_triggered else ""

            sep = "─" * 112
            print(sep)
            print(f"Ep {episode:5d} │ price={price:6.1f}  cap={cap:5.0f}  TNAC={tnac:6.0f} │ "
                  f"sec_p={sec_p:5.1f}  sec_vol={total_sec_vol:7.1f}  match={sec_match_rate*100:.0f}% │ "
                  f"ent={entropy_coef:.4f}  shaping={env.shaping_weight:.3f}"
                  f"{decay_str}{cyc_str}")
            # Header: last-year columns | episode aggregate columns
            print(f"  {'':4}  {'BankSt':>7} {'Alloc':>7} {'Emiss':>7} {'SecNet':>7} "
                  f"{'Hold':>7} {'Short/yr':>8} {'Pen/yr':>9} "
                  f"{'Green':>6} {'ΔGreen':>7} {'Bid':>5}  "
                  f"│ {'EpRew':>7} {'EpShort':>8} {'EpPen':>9} {'aLoss':>7} {'cLoss':>7}")
            for i in range(n_agents):
                act_mark = "*" if (cycling_enabled and i == active_agent_idx) else " "
                dg_str   = f"{ly_dg[i] * 100:+.1f}%"
                loss_i   = latest_losses[i]
                al_str   = f"{loss_i['actor_loss']:.4f}"  if loss_i else "  n/a "
                cl_str   = f"{loss_i['critic_loss']:.4f}" if loss_i else "  n/a "
                print(
                    f"  A{i+1}{act_mark}: "
                    f"{ly_bank[i]:7.1f} {ly_alloc[i]:7.1f} {ly_emiss[i]:7.1f} "
                    f"{ly_secnet[i]:+7.1f} {ly_hold[i]:7.1f} "
                    f"{ly_short[i]:8.2f} {ly_pen[i]:9.1f} "
                    f"{ly_green[i]*100:5.1f}% {dg_str:>7} {avg_bid_per_agent[i]:5.1f}  "
                    f"│ {total_rewards[i]:7.2f} {ep_total_shortfalls[i]:8.2f} "
                    f"{ep_total_penalties[i]:9.1f} {al_str:>7} {cl_str:>7}"
                )
            print(sep)

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
