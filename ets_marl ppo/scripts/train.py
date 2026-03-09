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
import io
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure UTF-8 output on Windows (box-drawing characters in console)
# hasattr guard: Jupyter notebooks use OutStream which has no .buffer attribute
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.agents.ppo_agent import PPOAgent
import src.agents.heuristic_policy as heuristic_policy


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_agents(env: ETSEnvironment, config: dict, seed: int):
    """Instantiate one PPO agent per company."""
    obs1_dim = env.companies[0].obs_dim_phase1  # 18
    obs2_dim = env.companies[0].obs_dim_phase2  # 21

    aq = config["auction"]
    inv = config["investment"]

    # Phase 1: [bid_price, qty_multiplier, invest_frac, tech_logit0, tech_logit1, tech_logit2]
    # qty_multiplier is a coverage ratio on estimated need — the policy learns HOW MUCH
    # of its compliance need to cover at auction rather than an arbitrary absolute volume.
    auction_low = np.array([
        aq["price_min"], aq.get("qty_mult_low", 0.3), 0.0, -1.0, -1.0, -1.0
    ], dtype=np.float32)
    auction_high = np.array([
        aq["price_max"], aq.get("qty_mult_high", 2.0), inv["max_invest_frac"], 1.0, 1.0, 1.0
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


# ---------------------------------------------------------------------------
# Behavioral cloning warm-start (heuristic pre-training)
# ---------------------------------------------------------------------------

def _to_pretanh(physical_action: np.ndarray, policy, device) -> np.ndarray:
    """
    Convert a physical action to pre-tanh (raw) space for MSE supervision.

    raw = atanh( clamp( (action - bias) / scale, -1+ε, 1-ε ) )
    """
    scale = policy.action_scale  # registered buffer on device
    bias = policy.action_bias
    h_t = torch.FloatTensor(physical_action).to(device)
    normalized = torch.clamp((h_t - bias) / (scale + 1e-8), -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.atanh(normalized).cpu().numpy()


def _run_bc_epoch(policy, obs_arr, tgt_arr, bc_opt, batch_size: int) -> float:
    """One epoch of mini-batch MSE on the policy mean head. Returns mean loss."""
    perm = torch.randperm(len(obs_arr), device=obs_arr.device)
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(obs_arr), batch_size):
        mb = perm[start:start + batch_size]
        dist = policy.forward(obs_arr[mb])
        loss = nn.MSELoss()(dist.mean, tgt_arr[mb])
        bc_opt.zero_grad()
        loss.backward()
        bc_opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def pretrain_behavioral_cloning(agents, env, config: dict,
                                 pretrain_cfg: dict, seed: int):
    """
    Behavioural cloning warm-start for both AuctionPolicy and SecondaryPolicy.

    Collection phase
    ----------------
    Run `pretrain.episodes` full episodes using the heuristic policy from
    heuristic_policy.py for BOTH phases.  At each year-step store:
      - (obs_phase1_i, auction_raw_i)    for AuctionPolicy BC
      - (obs_phase2_i, secondary_raw_i)  for SecondaryPolicy BC

    Training phase
    --------------
    For each agent, run `pretrain.epochs` epochs of mini-batch MSE loss on
    the respective policy mean heads using dedicated BC optimisers.  The
    PPO optimiser and value network are untouched.

    After this function returns, the main PPO loop resumes with the
    pretrained weights as its starting point.
    """
    n_agents = config["companies"]["n_agents"]
    n_years = config["simulation"]["n_years"]
    n_episodes = pretrain_cfg.get("episodes", 300)
    n_epochs = pretrain_cfg.get("epochs", 10)
    bc_lr = pretrain_cfg.get("lr", 0.001)
    batch_size = 64

    print(f"\n{'─'*60}")
    print(f"Behavioral cloning pre-training")
    print(f"  Episodes: {n_episodes}  |  Epochs: {n_epochs}  |  LR: {bc_lr}")
    print(f"  Policies: AuctionPolicy + SecondaryPolicy")
    print(f"{'─'*60}")

    # per-agent dataset: list of (obs1, auc_raw) and (obs2, sec_raw)
    auc_data = [[] for _ in range(n_agents)]
    sec_data = [[] for _ in range(n_agents)]

    # ----------------------------------------------------------------
    # Collection phase
    # ----------------------------------------------------------------
    for ep in range(n_episodes):
        obs1, _ = env.reset(seed=seed + ep * 997)

        for _year in range(n_years):
            price_ma3 = env._compute_price_ma3()
            current_year = env.current_year

            auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
            for i in range(n_agents):
                company = env.companies[i]
                h_auc = heuristic_policy.auction_action(
                    company, price_ma3, current_year, n_years, config)
                auction_actions[i] = h_auc

                auc_raw = _to_pretanh(h_auc, agents[i].auction_policy,
                                      agents[i].device)
                auc_data[i].append((obs1[i].copy(), auc_raw))

            obs2, _ = env.step_auction(auction_actions)

            # Secondary heuristic uses env state set by step_auction
            secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
            for i in range(n_agents):
                company = env.companies[i]
                h_sec = heuristic_policy.secondary_action(
                    company,
                    bank=float(env.holdings[i]),
                    allocation=float(env._phase1_allocations[i]),
                    clearing_price=env._phase1_clearing_price,
                    config=config,
                )
                secondary_actions[i] = h_sec

                sec_raw = _to_pretanh(h_sec, agents[i].secondary_policy,
                                      agents[i].device)
                sec_data[i].append((obs2[i].copy(), sec_raw))

            obs1_next, _, terminated, _, _ = env.step_secondary(secondary_actions)
            obs1 = obs1_next
            if terminated:
                break

    # ----------------------------------------------------------------
    # Training phase — AuctionPolicy then SecondaryPolicy, per agent
    # ----------------------------------------------------------------
    for agent_idx in range(n_agents):
        agent = agents[agent_idx]
        dev = agent.device

        # --- AuctionPolicy ---
        a_data = auc_data[agent_idx]
        if a_data:
            obs_a = torch.FloatTensor(np.array([d[0] for d in a_data])).to(dev)
            tgt_a = torch.FloatTensor(np.array([d[1] for d in a_data])).to(dev)
            bc_opt_a = optim.Adam(agent.auction_policy.parameters(), lr=bc_lr)

            for epoch in range(n_epochs):
                loss_a = _run_bc_epoch(agent.auction_policy, obs_a, tgt_a,
                                       bc_opt_a, batch_size)
                if epoch == 0 or epoch == n_epochs - 1:
                    print(f"  A{agent_idx+1} [auction]    "
                          f"epoch {epoch+1:2d}/{n_epochs}  loss={loss_a:.4f}")

        # --- SecondaryPolicy ---
        s_data = sec_data[agent_idx]
        if s_data:
            obs_s = torch.FloatTensor(np.array([d[0] for d in s_data])).to(dev)
            tgt_s = torch.FloatTensor(np.array([d[1] for d in s_data])).to(dev)
            bc_opt_s = optim.Adam(agent.secondary_policy.parameters(), lr=bc_lr)

            for epoch in range(n_epochs):
                loss_s = _run_bc_epoch(agent.secondary_policy, obs_s, tgt_s,
                                       bc_opt_s, batch_size)
                if epoch == 0 or epoch == n_epochs - 1:
                    print(f"  A{agent_idx+1} [secondary]  "
                          f"epoch {epoch+1:2d}/{n_epochs}  loss={loss_s:.4f}")

    print(f"{'─'*60}")
    print("Behavioral cloning pre-training complete.\n")


class EntropyConditionTracker:
    """
    P2: Time-based entropy decay.

    Entropy stays at coef_init for the first `decay_start` episodes
    (allowing BC warm-start and critic warmup to settle), then decays
    linearly from coef_init to coef_final over `decay_window` episodes.
    """

    def __init__(self, ppo_cfg: dict, n_agents: int):
        self.coef_init = ppo_cfg.get("entropy_coef", 0.02)
        self.coef_final = ppo_cfg.get("entropy_coef_final", 0.005)
        self.decay_window = ppo_cfg.get("entropy_decay_window", 3000)
        self.decay_start = ppo_cfg.get("entropy_decay_start", 500)

    def update(self, episode: int) -> float:
        """Return current entropy coefficient based on episode number."""
        if episode < self.decay_start:
            return self.coef_init
        episodes_since = episode - self.decay_start
        if episodes_since >= self.decay_window:
            return self.coef_final
        frac = episodes_since / max(self.decay_window, 1)
        return self.coef_init + frac * (self.coef_final - self.coef_init)

    @property
    def decay_triggered(self) -> bool:
        return True  # always available for logging compatibility


def _print_training_legend():
    """Print field guide once at the start of each training run."""
    leg = "─" * 82
    print(f"\n{leg}")
    print("TRAINING CONSOLE — COLUMN GUIDE")
    print(leg)
    print("Market header  (printed every log_interval episodes)")
    print("  price X→Y (peak Z) : ETS clearing price yr-0 → yr-N and peak  (€/t)")
    print("  cap                : Cap (Mt) in the final year")
    print("  TNAC               : Total allowances in circulation, final year")
    print("  sec_price/vol/match: Secondary market stats")
    print("  entropy/shaping    : PPO entropy coef & green-shaping weight")
    print("  [ENT-DECAY]        : Entropy decay triggered")
    print("  [cyc=Ax]           : Active agent (soft cycling)")
    print()
    print("Per-agent columns")
    print("  Green(0→N)  : Green fraction trajectory")
    print("  ΔGreen      : Net green change (pp)")
    print("  AvgEmiss    : Mean emissions (Mt)")
    print("  AvgAlloc    : Mean allocation (Mt)")
    print("  SfYrs       : Shortfall years")
    print("  AvgBid€     : Mean bid price (€/t)")
    print("  BidMt       : Mean bid volume (Mt) — after coverage-multiplier expansion")
    print("  InvFr       : Mean invest_frac action (how aggressively agent invests)")
    print("  TotRew      : Total raw reward")
    print("  TotShort    : Total shortfall (Mt)")
    print("  TotPenalty  : Total penalty (M\u20ac)")
    print("  ActLoss     : Actor loss")
    print("  CriLoss     : Critic loss")
    print("  MAC_Mt      : MAC fuel-switching reduction (Mt)")
    print("  SecMl       : Mean secondary price multiplier (action[0]; 0.5=discount 2.0=premium)")
    print("  SqAct       : Mean secondary qty action (+buy intent / -sell intent, Mt)")
    print(leg)


def train_one_seed(config: dict, seed: int):
    n_agents = config["companies"]["n_agents"]
    n_episodes = config["simulation"]["n_episodes"]
    n_years = config["simulation"]["n_years"]

    print(f"\n{'='*60}")
    print(f"Training — seed {seed}, {n_agents} agents, PPO, two-phase")
    print(f"v5.0: MAC switching | Electricity revenue | Carry-forward")
    print(f"P1-P8 + structural improvements active")
    print(f"{'='*60}")
    _print_training_legend()

    env = ETSEnvironment(config, seed=seed)
    agents = build_agents(env, config, seed)

    ppo_cfg = config["ppo"]

    pretrain_cfg = config.get("pretrain", {})
    bc_ran = False
    if pretrain_cfg.get("enabled", False):
        pretrain_behavioral_cloning(agents, env, config, pretrain_cfg, seed)
        bc_ran = True

    # Snapshot BC-trained weights as frozen KL anchors (only when BC was run)
    kl_beta_init = ppo_cfg.get("kl_anchor_beta", 0.0)
    if bc_ran and kl_beta_init > 0.0:
        for agent in agents:
            agent.set_bc_anchor()
        print(f"KL anchor: frozen BC policies captured for all {n_agents} agents "
              f"(β₀={kl_beta_init}, decay={ppo_cfg.get('kl_anchor_decay_episodes', 2000)} eps).")

    critic_warmup_eps = ppo_cfg.get("critic_warmup_episodes", 0)
    kl_decay_eps     = ppo_cfg.get("kl_anchor_decay_episodes", 2000)
    if critic_warmup_eps > 0:
        print(f"Critic-warmup: actor gradients frozen for first {critic_warmup_eps} episodes.")

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
    ep_fields += ["price_start", "price_peak", "price_std"]  # episode price trajectory
    for i in range(n_agents):  # allocation + P5/P6/P8/MAC episode aggregates
        ep_fields += [f"mean_alloc_A{i+1}",
                      f"mean_shock_A{i+1}", f"max_shock_A{i+1}",
                      f"mean_cf_shock_A{i+1}", f"total_cancels_A{i+1}",
                      f"total_holding_cost_A{i+1}", f"total_mac_reduction_A{i+1}"]
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
                      f"cancellation_A{i+1}", f"holding_cost_A{i+1}",  # P6/P8
                      f"auction_cost_A{i+1}", f"secondary_net_A{i+1}",  # cost breakdown
                      f"compliance_surplus_A{i+1}", f"bank_end_A{i+1}",  # compliance
                      f"mac_reduction_A{i+1}", f"mac_cost_A{i+1}"]  # MAC
    yr_csv = open(yr_path, "w", newline="")
    yr_writer = csv.DictWriter(yr_csv, fieldnames=yr_fields)
    yr_writer.writeheader()

    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]
    best_total_reward = -np.inf

    # ── Broken-policy detection state ────────────────────────────────────────
    _diag            = config.get("diagnostics", {})
    _broken_window   = _diag.get("market_broken_window", 20)
    _ceil_thresh     = _diag.get("market_broken_bid_threshold", 0.99)  # fraction of price_max
    _floor_thresh    = _diag.get("floor_bid_threshold", 1.02)          # fraction of price_min
    _zero_qty_thresh = _diag.get("zero_qty_threshold", 0.01)           # fraction of quantity_max
    _price_max       = config["auction"]["price_max"]
    _price_min       = config["auction"]["price_min"]
    _qty_max         = config["auction"]["quantity_max"]
    _streak_ceil  = np.zeros(n_agents, dtype=int)  # avg bid >= ceil_thresh * price_max
    _streak_floor = np.zeros(n_agents, dtype=int)  # avg bid <= floor_thresh * price_min
    _streak_qty   = np.zeros(n_agents, dtype=int)  # avg bid qty <= zero_qty_thresh * qty_max

    for episode in range(n_episodes):
        # Agent cycling: which agent updates this episode
        active_agent_idx = episode % n_agents if cycling_enabled else None

        # Critic-warmup: disable actor gradients for the first N episodes
        actor_update = (episode >= critic_warmup_eps)

        # KL anchor beta: linear decay from beta_init to 0 over kl_decay_eps
        if kl_beta_init > 0.0:
            kl_beta_now = kl_beta_init * max(0.0, 1.0 - episode / max(kl_decay_eps, 1))
            for agent in agents:
                agent.set_kl_beta(kl_beta_now)

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
                yr_row[f"holding_cost_A{i+1}"] = _get("holding_costs")       # P8
                # Derived cost-breakdown and compliance fields
                _alloc    = _get("allocations")
                _price    = yl.get("clearing_price", 0.0)
                _payment  = _get("payments")           # actual auction payment
                _tqty     = _get("trade_qtys")
                _tcost    = _get("trade_costs")
                _emiss    = _get("emissions")
                _bstart   = _get("bank_start")
                _holdings = _get("holdings")            # post-compliance bank
                yr_row[f"auction_cost_A{i+1}"]      = round(_payment, 4)
                yr_row[f"secondary_net_A{i+1}"]     = round(-_tcost, 4)  # +ve = revenue
                yr_row[f"compliance_surplus_A{i+1}"] = round(
                    _bstart + _alloc + _tqty - _emiss, 4)  # pre-compliance surplus
                yr_row[f"bank_end_A{i+1}"]          = round(_holdings, 4)  # post-compliance
                yr_row[f"mac_reduction_A{i+1}"]     = _get("mac_reductions")
                yr_row[f"mac_cost_A{i+1}"]          = _get("mac_costs")
            yr_writer.writerow(yr_row)

            obs1 = obs1_next
            if terminated:
                break

        # === PPO Update (end of episode) ===
        # Time-based entropy decay
        entropy_coef = entropy_tracker.update(episode)
        for agent in agents:
            agent.set_entropy_coef(entropy_coef)

        latest_losses = []
        for i in range(n_agents):
            if cycling_enabled:
                if i == active_agent_idx:
                    # Active agent: full update
                    loss = agents[i].update(last_value=0.0, actor_update=actor_update)
                elif cycling_soft:
                    # Soft cycling: update with reduced lr
                    orig_lr = agents[i].optimizer.param_groups[0]["lr"]
                    for pg in agents[i].optimizer.param_groups:
                        pg["lr"] = orig_lr * cycling_lr_scale
                    loss = agents[i].update(last_value=0.0, actor_update=actor_update)
                    for pg in agents[i].optimizer.param_groups:
                        pg["lr"] = orig_lr
                else:
                    # Hard freeze: discard buffer without updating
                    agents[i].buffer.clear()
                    loss = None
            else:
                loss = agents[i].update(last_value=0.0, actor_update=actor_update)
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

        # Average bid quantity (Mt) per agent — Phase 1 action[1] after multiplier expansion
        avg_bid_qty_per_agent = []
        for i in range(n_agents):
            qtys_this_ep = [
                yl["bid_quantities"][i]
                for yl in env.episode_log
                if "bid_quantities" in yl and i < len(yl["bid_quantities"])
            ]
            avg_bid_qty_per_agent.append(np.mean(qtys_this_ep) if qtys_this_ep else 0.0)

        # ── Broken-policy detection ───────────────────────────────────────────
        # 1. NaN reward: numerical explosion → immediate halt
        if np.isnan(total_rewards).any():
            ep_csv.close(); yr_csv.close()
            _nan_agents = [f"A{i+1}" for i in range(n_agents) if np.isnan(total_rewards[i])]
            raise RuntimeError(
                f"[BROKEN – NaN REWARD] ep {episode}: NaN reward for "
                f"{', '.join(_nan_agents)}. Numerical instability. Aborting."
            )
        # 2-4. Streak-based structural collapses (only counted after actor starts updating)
        if not actor_update:
            _streak_ceil[:] = 0; _streak_floor[:] = 0; _streak_qty[:] = 0
        else:
            for _i in range(n_agents):
                _streak_ceil[_i]  = (_streak_ceil[_i]  + 1) if avg_bid_per_agent[_i]     >= _ceil_thresh * _price_max  else 0
                _streak_floor[_i] = (_streak_floor[_i] + 1) if avg_bid_per_agent[_i]     <= _floor_thresh * _price_min else 0
                _streak_qty[_i]   = (_streak_qty[_i]   + 1) if avg_bid_qty_per_agent[_i] <= _zero_qty_thresh * _qty_max else 0

                if _streak_ceil[_i] >= _broken_window:
                    ep_csv.close(); yr_csv.close()
                    raise RuntimeError(
                        f"[BROKEN – CEILING BID] A{_i+1}: avg bid "
                        f">={_ceil_thresh*100:.0f}% of price_max ({_price_max:.0f} €/t) for "
                        f"{_broken_window} consecutive episodes (ep {episode}). "
                        f"Policy collapsed to price ceiling. Aborting."
                    )
                if _streak_floor[_i] >= _broken_window:
                    ep_csv.close(); yr_csv.close()
                    raise RuntimeError(
                        f"[BROKEN – FLOOR BID] A{_i+1}: avg bid "
                        f"<={_floor_thresh*100:.0f}% of price_min ({_price_min:.0f} €/t) for "
                        f"{_broken_window} consecutive episodes (ep {episode}). "
                        f"Policy stuck at price floor. Aborting."
                    )
                if _streak_qty[_i] >= _broken_window:
                    ep_csv.close(); yr_csv.close()
                    raise RuntimeError(
                        f"[BROKEN – ZERO QUANTITY] A{_i+1}: avg bid qty "
                        f"<={_zero_qty_thresh*100:.0f}% of qty_max ({_qty_max:.1f} Mt) for "
                        f"{_broken_window} consecutive episodes (ep {episode}). "
                        f"Agent opted out of auction. Aborting."
                    )

        # Average invest_frac action per agent — Phase 1 action[2]
        avg_invest_frac_per_agent = []
        for i in range(n_agents):
            frac_this_ep = [
                yl["invest_fracs"][i]
                for yl in env.episode_log
                if "invest_fracs" in yl and i < len(yl["invest_fracs"])
            ]
            avg_invest_frac_per_agent.append(np.mean(frac_this_ep) if frac_this_ep else 0.0)

        # Average secondary price multiplier per agent — Phase 2 action[0]
        avg_sec_mult_per_agent = []
        for i in range(n_agents):
            mults_this_ep = [
                yl["sec_price_mults"][i]
                for yl in env.episode_log
                if "sec_price_mults" in yl and i < len(yl["sec_price_mults"])
            ]
            avg_sec_mult_per_agent.append(np.mean(mults_this_ep) if mults_this_ep else 1.0)

        # Average secondary qty action per agent — Phase 2 action[1] (+ve=buy -ve=sell)
        avg_sec_qty_per_agent = []
        for i in range(n_agents):
            sqt_this_ep = [
                yl["sec_qty_actions"][i]
                for yl in env.episode_log
                if "sec_qty_actions" in yl and i < len(yl["sec_qty_actions"])
            ]
            avg_sec_qty_per_agent.append(np.mean(sqt_this_ep) if sqt_this_ep else 0.0)

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

        # P5/P6/P8 episode aggregates for diagnostics
        ep_mean_shock = [                                              # mean |ε| over years
            np.mean([abs(yl.get("emission_shocks", [0]*n_agents)[i])
                     for yl in env.episode_log])
            for i in range(n_agents)
        ]
        ep_max_shock = [                                               # worst-case ε
            max(yl.get("emission_shocks", [0]*n_agents)[i]
                for yl in env.episode_log)
            for i in range(n_agents)
        ]
        ep_mean_cf_shock = [                                           # mean |CF noise|
            np.mean([abs(yl.get("cf_shocks", [0]*n_agents)[i])
                     for yl in env.episode_log])
            for i in range(n_agents)
        ]
        ep_total_cancels = [                                           # total project cancellations
            sum(int(yl.get("cancellations", [0]*n_agents)[i]) for yl in env.episode_log)
            for i in range(n_agents)
        ]
        ep_total_holding_cost = [                                      # total P8 holding cost (M€)
            sum(yl.get("holding_costs", [0.0]*n_agents)[i] for yl in env.episode_log)
            for i in range(n_agents)
        ]
        ep_total_mac_reduction = [                                     # total MAC abatement (Mt)
            sum(yl.get("mac_reductions", [0.0]*n_agents)[i] for yl in env.episode_log)
            for i in range(n_agents)
        ]

        # Episode trajectory stats (across all years) — used in console only
        first_log   = env.episode_log[0] if env.episode_log else {}
        n_years_ep  = len(env.episode_log)
        prices_ep   = [yl.get("clearing_price", 0.0) for yl in env.episode_log]
        price_start = prices_ep[0]  if prices_ep else 0.0
        price_final = prices_ep[-1] if prices_ep else 0.0
        price_peak  = max(prices_ep) if prices_ep else 0.0

        ep_green_start = [
            first_log.get("green_fracs", [0.0]*n_agents)[i] for i in range(n_agents)
        ]
        ep_green_end = [
            last_log.get("green_fracs", [0.0]*n_agents)[i] for i in range(n_agents)
        ]
        ep_mean_emiss = [
            np.mean([yl.get("emissions", [0.0]*n_agents)[i] for yl in env.episode_log])
            for i in range(n_agents)
        ]
        ep_mean_alloc = [
            np.mean([yl.get("allocations", [0.0]*n_agents)[i] for yl in env.episode_log])
            for i in range(n_agents)
        ]
        ep_shortfall_years = [                       # count of years where agent had shortfall
            sum(1 for yl in env.episode_log
                if yl.get("shortfalls", [0.0]*n_agents)[i] > 1e-6)
            for i in range(n_agents)
        ]

        price_std = float(np.std(prices_ep)) if len(prices_ep) > 1 else 0.0

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
            "price_start": round(price_start, 2),
            "price_peak": round(price_peak, 2),
            "price_std": round(price_std, 2),
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
            ep_row[f"mean_alloc_A{i+1}"]          = round(ep_mean_alloc[i], 4)
            ep_row[f"mean_shock_A{i+1}"]         = round(ep_mean_shock[i], 5)
            ep_row[f"max_shock_A{i+1}"]          = round(ep_max_shock[i], 5)
            ep_row[f"mean_cf_shock_A{i+1}"]      = round(ep_mean_cf_shock[i], 5)
            ep_row[f"total_cancels_A{i+1}"]      = ep_total_cancels[i]
            ep_row[f"total_holding_cost_A{i+1}"] = round(ep_total_holding_cost[i], 4)
            ep_row[f"total_mac_reduction_A{i+1}"] = round(ep_total_mac_reduction[i], 4)
            if latest_losses[i]:
                ep_row[f"actor_loss_A{i+1}"] = round(latest_losses[i]["actor_loss"], 6)
                ep_row[f"critic_loss_A{i+1}"] = round(latest_losses[i]["critic_loss"], 6)
            else:
                ep_row[f"actor_loss_A{i+1}"] = 0.0
                ep_row[f"critic_loss_A{i+1}"] = 0.0

        ep_writer.writerow(ep_row)

        # Console diagnostics
        if episode % log_interval == 0:
            cap    = last_log.get("cap", 0)
            tnac   = last_log.get("tnac", 0)
            sec_p  = last_log.get("secondary_clearing", 0)

            cyc_str    = f" [cyc=A{active_agent_idx+1}]" if cycling_enabled else ""
            decay_str  = " [ENT-DECAY]" if entropy_tracker.decay_triggered else ""
            warmup_str = f" [WARMUP {episode+1}/{critic_warmup_eps}]" if not actor_update else ""

            # Compact warning summary — only non-zero counters
            _warn_labels = {
                "under_alloc": "under", "price_floor": "floor", "price_ceiling": "ceil",
                "auction_failed": "fail", "cover_below_one": "cov<1", "zero_invest": "0inv",
                "chronic_short": "chron", "bid_cluster": "bclust", "bank_hoard": "hoard",
                "sec_one_sided": "1side", "sec_zero_vol": "0vol",
            }
            _warn_parts = [
                f"{_warn_labels.get(k, k)}={v}"
                for k, v in env._warnings.items() if v > 0
            ]
            warn_str = "  │ warn: " + " ".join(_warn_parts) if _warn_parts else ""

            sep = "─" * 152
            print(sep)
            print(f"Ep {episode:5d} │ price {price_start:.0f}→{price_final:.0f} (peak {price_peak:.0f})"
                  f"  cap={cap:5.0f}  TNAC={tnac:5.0f} │ "
                  f"sec={sec_p:5.1f}€ vol={total_sec_vol:5.1f} match={sec_match_rate*100:.0f}% │ "
                  f"ent={entropy_coef:.4f}  shp={env.shaping_weight:.3f}"
                  f"{decay_str}{cyc_str}{warmup_str}{warn_str}")
            # Compact header covering all 8 action dimensions
            print(f"  {'':4}  {'Grn':>9} {'ΔG':>6} {'Emiss':>6} {'Alloc':>6} "
                  f"{'Sf':>5} {'Bid€':>6} {'BidMt':>6} {'InvFr':>5} "
                  f"│ {'Rew':>7} {'Short':>6} {'Pen':>7} {'ALoss':>7} {'CLoss':>7} "
                  f"│ {'MAC_Mt':>7} │ {'SecMl':>5} {'SqAct':>6}")
            for i in range(n_agents):
                act_mark  = "*" if (cycling_enabled and i == active_agent_idx) else " "
                grn_str   = f"{ep_green_start[i]*100:.0f}→{ep_green_end[i]*100:.0f}%"
                dgrn_str  = f"{(ep_green_end[i]-ep_green_start[i])*100:+.1f}"
                sf_str    = f"{ep_shortfall_years[i]}/{n_years_ep}"
                loss_i    = latest_losses[i]
                al_str    = f"{loss_i['actor_loss']:.4f}"  if loss_i else "   n/a"
                cl_str    = f"{loss_i['critic_loss']:.4f}" if loss_i else "   n/a"
                print(
                    f"  A{i+1}{act_mark}: "
                    f"{grn_str:>9} {dgrn_str:>6} {ep_mean_emiss[i]:6.2f} {ep_mean_alloc[i]:6.2f} "
                    f"{sf_str:>5} {avg_bid_per_agent[i]:6.0f} {avg_bid_qty_per_agent[i]:6.2f} {avg_invest_frac_per_agent[i]:5.3f} "
                    f"│ {total_rewards[i]:7.1f} {ep_total_shortfalls[i]:6.2f} "
                    f"{ep_total_penalties[i]:7.0f} {al_str:>7} {cl_str:>7} "
                    f"│ {ep_total_mac_reduction[i]:7.3f} │ {avg_sec_mult_per_agent[i]:5.2f} {avg_sec_qty_per_agent[i]:6.2f}"
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
