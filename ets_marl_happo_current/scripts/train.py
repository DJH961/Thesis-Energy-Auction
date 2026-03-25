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
import collections
import copy
import csv
import datetime
import io
import os
import sys
import time
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
    """Instantiate one PPO agent per learning company (not bots)."""
    obs1_dim = env.companies[0].obs_dim_phase1
    obs2_dim = env.companies[0].obs_dim_phase2
    n_agents = config["companies"]["n_agents"]  # learning agents only

    # MAPPO: global state = concatenation of learning agents' phase2 obs
    centralized = config["ppo"].get("centralized_critic", False)
    global_state_dim = n_agents * obs2_dim if centralized else 0

    aq = config["auction"]
    inv = config["investment"]

    # Phase 1: [bid_price, qty_multiplier, invest_frac, tech_logit0, tech_logit1, tech_logit2]
    # bid_price is direct in [price_min, price_max] €/t (no markup mechanism).
    # qty_multiplier is a coverage ratio on estimated need.
    auction_low = np.array([
        aq["price_min"], aq.get("qty_mult_low", 0.3), 0.0, -1.0, -1.0, -1.0
    ], dtype=np.float32)
    auction_high = np.array([
        aq["price_max"], aq.get("qty_mult_high", 1.3), inv["max_invest_frac"], 1.0, 1.0, 1.0
    ], dtype=np.float32)

    # Phase 2: [sec_price_multiplier, sec_qty]
    trading_cfg = config.get("trading", {})
    sec_mult_low = trading_cfg.get("sec_mult_low", 0.8)
    sec_mult_high = trading_cfg.get("sec_mult_high", 1.3)
    secondary_low = np.array([sec_mult_low, -aq["quantity_max"]], dtype=np.float32)
    secondary_high = np.array([sec_mult_high, aq["quantity_max"]], dtype=np.float32)

    agents = []
    for i in range(n_agents):
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
            global_state_dim=global_state_dim,
        )
        agents.append(agent)
    return agents


# ---------------------------------------------------------------------------
# Behavioral cloning warm-start (heuristic pre-training)
# ---------------------------------------------------------------------------

def _to_raw(physical_action: np.ndarray, policy, device) -> np.ndarray:
    """
    Convert a physical action to raw (normalised) space for MSE supervision.

    With clipped Gaussian: raw = clamp((action - bias) / scale, -1, 1).
    No atanh needed — the raw→physical mapping is linear within bounds.
    """
    scale = policy.action_scale  # registered buffer on device
    bias = policy.action_bias
    h_t = torch.FloatTensor(physical_action).to(device)
    return torch.clamp((h_t - bias) / (scale + 1e-8), -1.0, 1.0).cpu().numpy()


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
                    company, price_ma3, current_year, n_years, config,
                    reserve_price=env._compute_dynamic_reserve())
                auction_actions[i] = h_auc

                auc_raw = _to_raw(h_auc, agents[i].auction_policy,
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

                sec_raw = _to_raw(h_sec, agents[i].secondary_policy,
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
    P2: Time-based entropy decay with auto-scaling.

    Entropy stays at coef_init for the first `decay_start` episodes
    (allowing BC warm-start and critic warmup to settle), then decays
    linearly from coef_init to coef_final over `decay_window` episodes.

    If decay_start or decay_window are 0 in config, they auto-scale
    to 5% and 80% of n_episodes respectively.  This ensures the
    schedule adapts to both short Colab runs and full 15000-ep training.
    """

    def __init__(self, ppo_cfg: dict, n_agents: int, n_episodes: int = 15000):
        self.coef_init = ppo_cfg.get("entropy_coef", 0.02)
        self.coef_final = ppo_cfg.get("entropy_coef_final", 0.005)

        raw_window = ppo_cfg.get("entropy_decay_window", 0)
        raw_start = ppo_cfg.get("entropy_decay_start", 0)

        # Auto-scale: 0 means "compute from n_episodes"
        self.decay_start = raw_start if raw_start > 0 else int(0.05 * n_episodes)
        self.decay_window = raw_window if raw_window > 0 else int(0.80 * n_episodes)

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
    print("  entropy/shaping    : PPO entropy coef & green-shaping weight")
    print("  [ENT-DECAY]        : Entropy decay triggered")
    print("  [cyc=Ax]           : Active agent (soft cycling)")
    print()
    print("Secondary market line")
    print("  N sellers (-X Mt)  : Avg sellers/yr and total sold volume")
    print("  N buyers (+X Mt)   : Avg buyers/yr and total bought volume")
    print("  vol/match/avg_px   : Total volume, match rate, avg trade price")
    print()
    print("Market dynamics line")
    print("  avg_emiss          : Avg annual emissions across all participants (Mt/yr)")
    print("  compliance         : % of (agent, year) pairs with zero shortfall")
    print("  avg_green          : Avg green fraction across all participants at episode end")
    print("  TNAC               : Total allowances in circulation, final year")
    print()
    print("Bot summary line (if bots enabled)")
    print("  avg_bid/emiss      : Mean bot bid price (€/t) and emissions (Mt)")
    print("  compliant          : Bot compliant agent-years / total bot agent-years")
    print("  sec: Ns Nb         : Bot secondary roles (s=seller, b=buyer, h=hold)")
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
    print("  Role        : Secondary market role: SELL / BUY / HOLD")
    print("  SecMl       : Mean secondary price multiplier (action[0]; 0.5=discount 2.0=premium)")
    print("  SqAct       : Mean secondary qty action (+buy intent / -sell intent, Mt)")
    print("  elapsed/ETA : Wall-clock elapsed time and estimated remaining time")
    print()
    print("Inline warnings  (appear as │ warn: key=N at end of market header line)")
    print("  These count how many year-steps within the episode triggered each condition.")
    print("  Low counts (1-2) are normal early in training. Persistent high counts signal problems.")
    print()
    print("  lowAlloc=N   : Total allocation < 30% of auction volume. Agents badly under-bidding.")
    print("                 Common early; should fade by ep ~2000.")
    print("  priceFloor=N : Clearing price hit the reserve price floor. Market not competitive —")
    print("                 agents bid near minimum. Often pairs with lowAlloc.")
    print("  priceCeil=N  : Clearing price ≥90% of price_max. Agents overbidding — unlikely to")
    print("                 be optimal in uniform-price auction. May indicate reward miscalibration.")
    print("  auctFail=N   : Auction cancelled (cancel_under_subscribed=true, currently disabled).")
    print("  lowDemand=N  : Total demand < 70% of supply. Agents under-bidding on quantity.")
    print("                 Early exploration noise; persistent = weak quantity signal.")
    print("  noInvest=N   : All agents chose invest_frac ≈ 0 — nobody investing in green.")
    print("                 Occasional is fine; persistent = investment signal too weak.")
    print("  debtSpiral=N : Agent had 3+ consecutive shortfall years (only counted from year 3+).")
    print("                 Agent stuck in carry-forward debt spiral. CF-cap should limit this.")
    print("  bidCluster=N : Bid price std < €5 — agents converged to near-identical bids.")
    print("                 Not always bad if the resulting price is reasonable.")
    print("  overBank=N   : TNAC > 2× total emissions — massive over-banking. Stockpiling")
    print("                 allowances suppresses price signals and delays scarcity.")
    print("  1sideSec=N   : All agents buying OR all selling on secondary market.")
    print("                 No natural counterparty exists — expected in thin compliance markets.")
    print("                 Informational, not pathological (pool disabled, pure agent-to-agent).")
    print("  noTrade=N    : Secondary market volume ≈ 0. No trading this year-step.")
    print("                 Expected in compliance-only market without liquidity pool.")
    print("                 Signals thin market, not a bug. Watch for trends, not individual counts.")
    print("  cornering=N  : An agent's alloc share > 2× its emissions share AND > 30% of total,")
    print("                 with meaningful auction volume (>30% of cap). Market concentration risk.")
    print("  rsvReject=N  : Rejected bid volume > 25% of total bid volume (bids below dynamic")
    print("                 reserve). Normal when reserve rises; persistent = agents can't adapt.")
    print()
    print("Streak warnings  (printed as separate ⚠ lines between log intervals)")
    print("  ⚠ CEILING BID  : Agent's avg bid ≥99% of price_max for 200+ consecutive episodes.")
    print("  ⚠ FLOOR BID    : Agent's avg bid ≤102% of price_min for 200+ consecutive episodes.")
    print("  ⚠ ZERO QUANTITY : Agent's avg bid qty ≤1% of qty_max for 200+ consecutive episodes.")
    print("  🔄 ENTROPY BOOST: Agent stuck at floor/ceiling → entropy temporarily boosted to")
    print("                    force re-exploration. Resets once agent moves away from extreme.")
    print("  These indicate an agent's policy has collapsed to an extreme. The entropy boost")
    print("  mechanism tries to rescue it; if warnings persist, check reward signal or hyperparams.")
    print(leg)


def _format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(datetime.timedelta(seconds=max(0, int(round(float(seconds))))))


def train_one_seed(config: dict, seed: int, on_log=None):
    n_agents = config["companies"]["n_agents"]
    n_episodes = config["simulation"]["n_episodes"]
    n_years = config["simulation"]["n_years"]

    happo_enabled = config["ppo"].get("happo", False)
    clip_eps = config["ppo"].get("clip_eps", 0.2)

    if happo_enabled:
        algo = "HAPPO (sequential update, centralized critic)"
    elif config["ppo"].get("centralized_critic", False):
        algo = "MAPPO (centralized critic)"
    else:
        algo = "IPPO (independent critic)"

    curric_cfg = config.get("curriculum", {})
    curric_str = (f" | Curriculum {curric_cfg.get('start_years',4)}→{n_years}yr"
                  if curric_cfg.get("enabled", False) else "")
    cf_cap = config.get("penalty", {}).get("carry_forward_cap", 0.0)
    cf_str = f" | CF-cap={cf_cap}×" if cf_cap > 0 else ""
    explore_cfg_banner = config.get("exploration", {})
    eps_str = (f" | ε-greedy {explore_cfg_banner.get('epsilon_start',0):.0%}→"
               f"{explore_cfg_banner.get('epsilon_final',0):.0%}"
               if explore_cfg_banner.get("epsilon_start", 0) > 0 else "")

    n_bot_agents = config["companies"].get("n_bot_agents", 0)
    bot_str = f" + {n_bot_agents} bots" if n_bot_agents > 0 else ""

    print(f"\n{'='*60}")
    print(f"Training — seed {seed}, {n_agents} learning agents{bot_str}, {algo}, two-phase")
    print(f"v5.3: MAC switching | Electricity revenue | Carry-forward{cf_str} | Permanent coverage signal")
    print(f"Clipped Gaussian (no tanh) + P1-P8 active{curric_str}{eps_str}")
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

    # Curriculum learning (disabled — kept for config compat)
    curriculum_cfg = config.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    if curriculum_enabled:
        curriculum_start_years = curriculum_cfg.get("start_years", 4)
        curriculum_ramp_episodes = curriculum_cfg.get("ramp_episodes", 3000)
        print(f"Curriculum: {curriculum_start_years}yr → {n_years}yr "
              f"over {curriculum_ramp_episodes} episodes.")

    # Epsilon-greedy exploration schedule (physical-space)
    explore_cfg = config.get("exploration", {})
    eps_start = explore_cfg.get("epsilon_start", 0.0)
    eps_final = explore_cfg.get("epsilon_final", 0.0)
    eps_decay_episodes = explore_cfg.get("epsilon_decay_episodes", 5000)
    if eps_start > 0.0:
        print(f"Epsilon-greedy: {eps_start:.2f} → {eps_final:.2f} "
              f"over {eps_decay_episodes} episodes (physical-space).")

    # Historical Policy Pool (HPP) — anti-regression safety net
    hpp_cfg = config.get("hpp", {})
    hpp_enabled = hpp_cfg.get("enabled", False)
    hpp_pool_size = hpp_cfg.get("pool_size", 10)
    hpp_save_interval = hpp_cfg.get("save_interval", 500)
    hpp_swap_prob = hpp_cfg.get("swap_prob", 0.20)
    hpp_warmup = hpp_cfg.get("warmup_episodes", 1000)
    # Per-agent FIFO pool of actor state_dicts (auction + secondary only)
    hpp_pools = [collections.deque(maxlen=hpp_pool_size) for _ in range(n_agents)]
    if hpp_enabled:
        print(f"HPP: pool={hpp_pool_size}, save every {hpp_save_interval} eps, "
              f"swap_prob={hpp_swap_prob:.0%}, warmup={hpp_warmup} eps.")

    # Condition-based entropy tracker (auto-scales to n_episodes)
    entropy_tracker = EntropyConditionTracker(ppo_cfg, n_agents, n_episodes)

    # --- CSV loggers ---
    results_dir = config["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Episode-level
    ep_path = os.path.join(results_dir, f"training_log_s{seed}.csv")
    ep_fields = ["episode", "clearing_price_last", "cap_last", "entropy_coef",
                 "shaping_weight", "entropy_decay_triggered", "active_agent",
                 "epsilon"]
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
                      f"total_mac_reduction_A{i+1}"]
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
                      f"cancellation_A{i+1}",  # P6
                      f"auction_cost_A{i+1}", f"secondary_net_A{i+1}",  # cost breakdown
                      f"compliance_surplus_A{i+1}", f"bank_end_A{i+1}",  # compliance
                      f"mac_reduction_A{i+1}", f"mac_cost_A{i+1}"]  # MAC
    yr_csv = open(yr_path, "w", newline="")
    yr_writer = csv.DictWriter(yr_csv, fieldnames=yr_fields)
    yr_writer.writeheader()

    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]
    # Flush logs every N episodes to reduce data loss if training aborts early.
    csv_flush_interval = int(
        config["logging"].get("csv_flush_interval", 1000)
    )
    csv_flush_interval = max(1, csv_flush_interval)
    best_total_reward = -np.inf

    train_t0 = time.time()
    recent_ep_durations = collections.deque(maxlen=200)

    def _flush_csv_logs(current_episode: int, force: bool = False) -> None:
        if force or ((current_episode + 1) % csv_flush_interval == 0):
            ep_csv.flush()
            yr_csv.flush()
            # fsync keeps data safer on abrupt termination.
            os.fsync(ep_csv.fileno())
            os.fsync(yr_csv.fileno())

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

    # Per-agent entropy boost: when an agent's bid is stuck at floor/ceiling
    # for _stuck_boost_window consecutive episodes, temporarily set its
    # entropy to max(_stuck_boost_coef, base_coef).
    _stuck_boost_window = _diag.get("stuck_boost_window", 500)
    _stuck_boost_coef   = _diag.get("stuck_boost_coef", 0.025)

    for episode in range(n_episodes):
        episode_t0 = time.time()
        # Agent cycling: which agent updates this episode
        active_agent_idx = episode % n_agents if cycling_enabled else None

        # Critic-warmup: disable actor gradients for the first N episodes
        actor_update = (episode >= critic_warmup_eps)

        # KL anchor beta: linear decay from beta_init to 0 over kl_decay_eps
        if kl_beta_init > 0.0:
            kl_beta_now = kl_beta_init * max(0.0, 1.0 - episode / max(kl_decay_eps, 1))
            for agent in agents:
                agent.set_kl_beta(kl_beta_now)

        # Always use configured n_years (curriculum disabled — 12-year episodes
        # are short enough for direct training with terminal value rewards).
        effective_n_years = n_years

        # P4: Communicate episode to environment for shaping weight + lock-in activation
        env.set_episode(episode)

        # Epsilon-greedy schedule: linear decay
        if eps_start > 0.0:
            eps_frac = min(1.0, episode / max(eps_decay_episodes, 1))
            current_epsilon = eps_start + eps_frac * (eps_final - eps_start)
        else:
            current_epsilon = 0.0

        obs1, _ = env.reset(seed=seed + episode * 1000)
        total_rewards = np.zeros(n_agents)

        # HPP: swap some agents to historical policies for this episode's rollout
        hpp_swapped = {}  # agent_idx → saved (auc_sd, sec_sd)
        if hpp_enabled and episode >= hpp_warmup:
            for i in range(n_agents):
                if hpp_pools[i] and np.random.random() < hpp_swap_prob:
                    # Save current actor weights
                    hpp_swapped[i] = (
                        copy.deepcopy(agents[i].auction_policy.state_dict()),
                        copy.deepcopy(agents[i].secondary_policy.state_dict()),
                    )
                    # Load random historical policy for action selection
                    hist_auc, hist_sec = hpp_pools[i][
                        np.random.randint(len(hpp_pools[i]))]
                    agents[i].auction_policy.load_state_dict(hist_auc)
                    agents[i].secondary_policy.load_state_dict(hist_sec)

        for year in range(effective_n_years):
            # === PHASE 1: Auction + Investment ===
            auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
            auction_raws = []
            auction_logps = []

            for i in range(n_agents):
                action, raw, logp = agents[i].select_auction_action(
                    obs1[i], epsilon=current_epsilon)
                auction_actions[i] = action
                auction_raws.append(raw)
                auction_logps.append(logp)

            obs2, auction_info = env.step_auction(auction_actions)

            # MAPPO: construct global state from all agents' phase2 obs
            _centralized = config["ppo"].get("centralized_critic", False)
            global_state = obs2.flatten() if _centralized else None

            # === PHASE 2: Secondary Market ===
            secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
            secondary_raws = []
            secondary_logps = []

            for i in range(n_agents):
                action, raw, logp = agents[i].select_secondary_action(
                    obs2[i], epsilon=current_epsilon)
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
                value = agents[i].estimate_value(
                    global_state if _centralized else obs2[i])
                agents[i].store_transition(
                    obs1=obs1[i], obs2=obs2[i],
                    auc_raw=auction_raws[i], sec_raw=secondary_raws[i],
                    auc_lp=auction_logps[i], sec_lp=secondary_logps[i],
                    reward=normalised_rewards[i], done=terminated, value=value,
                    global_state=global_state,
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

        # HPP: restore current policies (swapped agents used historical for rollout only)
        if hpp_swapped:
            for i, (saved_auc, saved_sec) in hpp_swapped.items():
                agents[i].auction_policy.load_state_dict(saved_auc)
                agents[i].secondary_policy.load_state_dict(saved_sec)
            # Swapped agents collected experience under historical policy —
            # discard their buffers so they don't corrupt the current policy update.
            for i in hpp_swapped:
                agents[i].buffer.clear()

        # HPP: periodically snapshot current actors into the pool
        if hpp_enabled and episode > 0 and episode % hpp_save_interval == 0:
            for i in range(n_agents):
                hpp_pools[i].append((
                    copy.deepcopy(agents[i].auction_policy.state_dict()),
                    copy.deepcopy(agents[i].secondary_policy.state_dict()),
                ))

        # === PPO Update (end of episode) ===
        # Time-based entropy decay
        entropy_coef = entropy_tracker.update(episode)
        for i, agent in enumerate(agents):
            # Per-agent entropy boost: if agent has been stuck at floor or
            # ceiling for a long streak, temporarily boost its entropy to
            # force re-exploration.  This prevents premature convergence
            # where an agent (e.g. A1) locks into floor-bidding and entropy
            # decay prevents it from ever discovering that bidding higher
            # in a uniform-price auction is nearly costless.
            agent_ent = entropy_coef
            if _stuck_boost_window > 0 and actor_update:
                if _streak_floor[i] >= _stuck_boost_window or _streak_ceil[i] >= _stuck_boost_window:
                    agent_ent = max(entropy_coef, _stuck_boost_coef)
            agent.set_entropy_coef(agent_ent)

        latest_losses = []
        if happo_enabled:
            # === HAPPO: Sequential update with cumulative importance ratios ===
            # 1. Compute GAE advantages per agent (using their own centralized critics)
            gae_data = []
            for i in range(n_agents):
                adv, ret, buf = agents[i].compute_gae(last_value=0.0)
                gae_data.append((adv, ret, buf))

            # 2. Sequential update in random order
            order = np.random.permutation(n_agents).tolist()
            # Find T from first agent with non-empty buffer
            T = 0
            for _gd in gae_data:
                if _gd[2] is not None:
                    T = _gd[2]["T"]
                    break
            cumulative_ratio = torch.ones(T, 1) if T > 0 else None

            for j in order:
                adv_j, ret_j, buf_j = gae_data[j]
                if buf_j is None:
                    latest_losses.append(None)
                    continue

                loss = agents[j].update_happo(
                    adv_t=adv_j,
                    ret_t=ret_j,
                    buf_tensors=buf_j,
                    advantage_weights=cumulative_ratio,
                    actor_update=actor_update,
                )
                latest_losses.append(loss)

                # Compute post-update importance ratio for this agent
                if actor_update and cumulative_ratio is not None:
                    ratio_j = agents[j].compute_post_update_ratio(buf_j)
                    clipped_j = torch.min(
                        ratio_j,
                        torch.clamp(ratio_j, 1 - clip_eps, 1 + clip_eps)
                    )
                    cumulative_ratio = cumulative_ratio * clipped_j.detach().cpu()

            # Reorder losses to match agent index (not update order)
            ordered_losses = latest_losses
            latest_losses = [None] * n_agents
            for idx_in_order, j in enumerate(order):
                latest_losses[j] = ordered_losses[idx_in_order]
        else:
            # === IPPO / cycling update (fallback) ===
            for i in range(n_agents):
                # HPP: skip update for agents whose buffer was cleared (swapped)
                if i in hpp_swapped:
                    latest_losses.append(None)
                    continue
                if cycling_enabled:
                    if i == active_agent_idx:
                        loss = agents[i].update(last_value=0.0, actor_update=actor_update)
                    elif cycling_soft:
                        orig_lr = agents[i].optimizer.param_groups[0]["lr"]
                        for pg in agents[i].optimizer.param_groups:
                            pg["lr"] = orig_lr * cycling_lr_scale
                        loss = agents[i].update(last_value=0.0, actor_update=actor_update)
                        for pg in agents[i].optimizer.param_groups:
                            pg["lr"] = orig_lr
                    else:
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
        sec_match_rate = years_with_trades / max(effective_n_years, 1)

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
            _flush_csv_logs(episode, force=True)
            ep_csv.close(); yr_csv.close()
            _nan_agents = [f"A{i+1}" for i in range(n_agents) if np.isnan(total_rewards[i])]
            raise RuntimeError(
                f"[BROKEN – NaN REWARD] ep {episode}: NaN reward for "
                f"{', '.join(_nan_agents)}. Numerical instability. Aborting."
            )
        # 2-4. Streak-based structural collapses (warnings only — training continues)
        if not actor_update:
            _streak_ceil[:] = 0; _streak_floor[:] = 0; _streak_qty[:] = 0
        else:
            for _i in range(n_agents):
                _streak_ceil[_i]  = (_streak_ceil[_i]  + 1) if avg_bid_per_agent[_i]     >= _ceil_thresh * _price_max  else 0
                _streak_floor[_i] = (_streak_floor[_i] + 1) if avg_bid_per_agent[_i]     <= _floor_thresh * _price_min else 0
                _streak_qty[_i]   = (_streak_qty[_i]   + 1) if avg_bid_qty_per_agent[_i] <= _zero_qty_thresh * _qty_max else 0

                if _streak_ceil[_i] >= _broken_window and _streak_ceil[_i] % _broken_window == 0:
                    print(
                        f"⚠ [WARN – CEILING BID] A{_i+1}: avg bid "
                        f">={_ceil_thresh*100:.0f}% of price_max ({_price_max:.0f} €/t) for "
                        f"{_streak_ceil[_i]} consecutive episodes (ep {episode})."
                    )
                if _streak_floor[_i] >= _broken_window and _streak_floor[_i] % _broken_window == 0:
                    print(
                        f"⚠ [WARN – FLOOR BID] A{_i+1}: avg bid "
                        f"<={_floor_thresh*100:.0f}% of price_min ({_price_min:.0f} €/t) for "
                        f"{_streak_floor[_i]} consecutive episodes (ep {episode})."
                    )
                # Per-agent entropy boost logging
                if _stuck_boost_window > 0:
                    if (_streak_floor[_i] == _stuck_boost_window or
                            _streak_ceil[_i] == _stuck_boost_window):
                        side = "FLOOR" if _streak_floor[_i] >= _stuck_boost_window else "CEILING"
                        print(
                            f"🔄 [ENTROPY BOOST] A{_i+1}: stuck at {side} for "
                            f"{_stuck_boost_window} episodes → entropy boosted "
                            f"to {_stuck_boost_coef:.3f} (ep {episode})."
                        )

                if _streak_qty[_i] >= _broken_window and _streak_qty[_i] % _broken_window == 0:
                    print(
                        f"⚠ [WARN – ZERO QUANTITY] A{_i+1}: avg bid qty "
                        f"<={_zero_qty_thresh*100:.0f}% of qty_max ({_qty_max:.1f} Mt) for "
                        f"{_streak_qty[_i]} consecutive episodes (ep {episode})."
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
            "epsilon": round(current_epsilon, 4),
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
            ep_row[f"total_mac_reduction_A{i+1}"] = round(ep_total_mac_reduction[i], 4)
            if latest_losses[i]:
                ep_row[f"actor_loss_A{i+1}"] = round(latest_losses[i]["actor_loss"], 6)
                ep_row[f"critic_loss_A{i+1}"] = round(latest_losses[i]["critic_loss"], 6)
            else:
                ep_row[f"actor_loss_A{i+1}"] = 0.0
                ep_row[f"critic_loss_A{i+1}"] = 0.0

        ep_writer.writerow(ep_row)

        # Track runtime and flush logs periodically for crash resilience.
        recent_ep_durations.append(time.time() - episode_t0)
        _flush_csv_logs(episode)

        # Console diagnostics
        if episode % log_interval == 0:
            cap    = last_log.get("cap", 0)
            tnac   = last_log.get("tnac", 0)
            sec_p  = last_log.get("secondary_clearing", 0)
            elapsed_s = time.time() - train_t0
            avg_ep_s = float(np.mean(recent_ep_durations)) if recent_ep_durations else 0.0
            episodes_left = max(0, n_episodes - (episode + 1))
            eta_s = avg_ep_s * episodes_left

            cyc_str    = f" [cyc=A{active_agent_idx+1}]" if cycling_enabled else ""
            decay_str  = " [ENT-DECAY]" if entropy_tracker.decay_triggered else ""
            warmup_str = f" [WARMUP {episode+1}/{critic_warmup_eps}]" if not actor_update else ""

            # Compact warning summary — only non-zero counters
            _warn_labels = {
                "low_alloc": "lowAlloc", "price_floor": "priceFloor", "price_ceil": "priceCeil",
                "auct_fail": "auctFail", "low_demand": "lowDemand", "no_invest": "noInvest",
                "debt_spiral": "debtSpiral", "bid_cluster": "bidCluster", "over_bank": "overBank",
                "one_side_sec": "1sideSec", "no_trade": "noTrade",
                "cornering": "cornering", "rsv_reject": "rsvReject",
            }
            _warn_parts = [
                f"{_warn_labels.get(k, k)}={v}"
                for k, v in env._warnings.items() if v > 0
            ]
            warn_str = "  │ warn: " + " ".join(_warn_parts) if _warn_parts else ""

            # ── Enhanced secondary market breakdown ────────────────────
            n_total = env.n_agents  # learning + bots
            sec_sellers = 0; sec_buyers = 0; sec_holders = 0
            sec_sell_vol = 0.0; sec_buy_vol = 0.0
            for _yi in env.episode_log:
                for _si in range(n_total):
                    tq = _yi.get("trade_qtys", [0.0] * n_total)
                    if _si < len(tq):
                        if tq[_si] < -1e-6:
                            sec_sellers += 1; sec_sell_vol += abs(tq[_si])
                        elif tq[_si] > 1e-6:
                            sec_buyers += 1; sec_buy_vol += tq[_si]
                        else:
                            sec_holders += 1
            sec_avg_sellers = sec_sellers / max(n_years_ep, 1)
            sec_avg_buyers = sec_buyers / max(n_years_ep, 1)
            # Per-agent secondary role for learning agents
            agent_sec_role = []
            for _ai in range(n_agents):
                avg_q = avg_sec_qty_per_agent[_ai]
                if avg_q < -0.01:
                    agent_sec_role.append("SELL")
                elif avg_q > 0.01:
                    agent_sec_role.append("BUY")
                else:
                    agent_sec_role.append("HOLD")

            # ── Market dynamics aggregates ─────────────────────────────
            total_emiss_ep = sum(
                sum(yl.get("emissions", [0.0] * n_total))
                for yl in env.episode_log
            )
            avg_annual_emiss = total_emiss_ep / max(n_years_ep, 1)
            # Compliance rate: fraction of (agent, year) with no shortfall
            total_agent_years = n_total * n_years_ep
            compliant_ay = sum(
                sum(1 for s in yl.get("shortfalls", [0.0] * n_total) if s < 1e-6)
                for yl in env.episode_log
            )
            compliance_rate = compliant_ay / max(total_agent_years, 1)
            avg_green_all = float(np.mean(
                last_log.get("green_fracs", [0.0] * n_total)[:n_total]))

            # ── Bot summary ────────────────────────────────────────────
            bot_lines = []
            n_bot_agents = config["companies"].get("n_bot_agents", 0)
            if n_bot_agents > 0:
                bot_avg_bid = []
                bot_avg_emiss = []
                bot_compliant = 0
                bot_sec_roles = {"SELL": 0, "BUY": 0, "HOLD": 0}
                for b in range(n_bot_agents):
                    bidx = n_agents + b  # bots indexed after learning agents in env arrays
                    # Since bots are at indices n_agents..n_total-1 in env arrays,
                    # we access them via episode_log which stores n_total-length arrays
                    b_bids = [yl["bid_prices"][bidx] for yl in env.episode_log
                              if "bid_prices" in yl and bidx < len(yl["bid_prices"])]
                    b_emiss = [yl["emissions"][bidx] for yl in env.episode_log
                               if "emissions" in yl and bidx < len(yl["emissions"])]
                    b_short = [yl["shortfalls"][bidx] for yl in env.episode_log
                               if "shortfalls" in yl and bidx < len(yl["shortfalls"])]
                    b_tq = [yl["trade_qtys"][bidx] for yl in env.episode_log
                            if "trade_qtys" in yl and bidx < len(yl["trade_qtys"])]
                    bot_avg_bid.append(np.mean(b_bids) if b_bids else 0.0)
                    bot_avg_emiss.append(np.mean(b_emiss) if b_emiss else 0.0)
                    bot_compliant += sum(1 for s in b_short if s < 1e-6)
                    avg_tq = np.mean(b_tq) if b_tq else 0.0
                    if avg_tq < -0.01:
                        bot_sec_roles["SELL"] += 1
                    elif avg_tq > 0.01:
                        bot_sec_roles["BUY"] += 1
                    else:
                        bot_sec_roles["HOLD"] += 1
                bot_compliant_yrs = bot_compliant
                bot_total_yrs = n_bot_agents * n_years_ep
                bot_sec_str = " ".join(
                    f"{v}{k[0].lower()}" for k, v in bot_sec_roles.items() if v > 0)

            sep = "─" * 152
            print(sep)
            print(f"Time elapsed: {_format_hms(elapsed_s)} | ETA: {_format_hms(eta_s)} | speed={avg_ep_s:.2f}s/ep")
            print(f"Ep {episode:5d} │ price {price_start:.0f}→{price_final:.0f} (peak {price_peak:.0f})"
                  f"  cap={cap:5.0f}  TNAC={tnac:5.0f} │ "
                  f"ent={entropy_coef:.4f}  shp={env.shaping_weight:.3f}"
                  f"  eps={current_epsilon:.3f}"
                  f"{decay_str}{cyc_str}{warmup_str}{warn_str}")
            # Enhanced secondary market line
            print(f"  Secondary: {sec_avg_sellers:.0f} sellers (-{sec_sell_vol:.1f} Mt) "
                  f"{sec_avg_buyers:.0f} buyers (+{sec_buy_vol:.1f} Mt) "
                  f"│ vol={total_sec_vol:.1f} Mt  match={sec_match_rate*100:.0f}%  "
                  f"avg_px={avg_sec_price:.1f}€  sec_clear={sec_p:.1f}€")
            # Market dynamics line
            print(f"  Market: avg_emiss={avg_annual_emiss:.1f} Mt/yr  "
                  f"compliance={compliance_rate*100:.0f}%  "
                  f"avg_green={avg_green_all*100:.0f}%  "
                  f"TNAC={tnac:.1f} Mt")
            # Bot summary
            if n_bot_agents > 0:
                print(f"  Bots ({n_bot_agents}): "
                      f"avg_bid={np.mean(bot_avg_bid):.0f}€  "
                      f"avg_emiss={np.mean(bot_avg_emiss):.2f} Mt  "
                      f"compliant={bot_compliant_yrs}/{bot_total_yrs} yr  "
                      f"sec: {bot_sec_str}")
            # Compact header covering all 8 action dimensions
            print(f"  {'':4}  {'Grn':>9} {'ΔG':>6} {'Emiss':>6} {'Alloc':>6} "
                  f"{'Sf':>5} {'Bid€':>6} {'BidMt':>6} {'InvFr':>5} "
                  f"│ {'Rew':>7} {'Short':>6} {'Pen':>7} {'ALoss':>7} {'CLoss':>7} "
                  f"│ {'MAC_Mt':>7} │ {'Role':>4} {'SecMl':>5} {'SqAct':>6}")
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
                    f"│ {ep_total_mac_reduction[i]:7.3f} │ {agent_sec_role[i]:>4} {avg_sec_mult_per_agent[i]:5.2f} {avg_sec_qty_per_agent[i]:6.2f}"
                )
            print(sep)

            # Optional callback for live plotting (e.g. from notebook)
            if on_log is not None:
                on_log(episode, ep_path)

        # Checkpointing
        if episode % save_interval == 0:
            ckpt_dir = os.path.join(results_dir, f"checkpoints_s{seed}")
            os.makedirs(ckpt_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                agent.save(os.path.join(ckpt_dir, f"agent_{i}_ep{episode}.pt"))
            _flush_csv_logs(episode, force=True)

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
