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

import atexit
import argparse
import collections
import copy
import csv
import datetime
import io
import math
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

    # Phase 1: 3-tranche bid ladder
    # [p1, q1, p2, q2, p3, q3, invest_frac, tech_logit0, tech_logit1, tech_logit2]
    # Each (p_k, q_k) pair: price in [price_min, price_max], qty_multiplier on need.
    p_lo, p_hi = aq["price_min"], aq["price_max"]
    q_lo = aq.get("qty_mult_low", 0.3)
    q_hi = aq.get("qty_mult_high", 1.3)
    auction_low = np.array([
        p_lo, q_lo, p_lo, q_lo, p_lo, q_lo,
        0.0, -1.0, -1.0, -1.0
    ], dtype=np.float32)
    auction_high = np.array([
        p_hi, q_hi, p_hi, q_hi, p_hi, q_hi,
        inv["max_invest_frac"], 1.0, 1.0, 1.0
    ], dtype=np.float32)

    # Phase 2: [sec_price (absolute EUR/t), sec_qty]
    # Secondary price range: [sec_price_min, sec_price_max_mult × max_penalty_rate]
    # Use worst-case penalty rate (final year inflation) for action space bounds.
    trading_cfg = config.get("trading", {})
    sec_price_min = trading_cfg.get("sec_price_min", 30.0)
    sec_price_max_mult = trading_cfg.get("sec_price_max_mult", 2.0)
    pen_cfg = config.get("penalty", {})
    base_penalty = pen_cfg.get("rate", 138.75)
    infl_rate = pen_cfg.get("inflation_rate", 0.02)
    n_yrs = config["simulation"]["n_years"]
    max_penalty = base_penalty * (1.0 + infl_rate) ** n_yrs
    sec_price_high = sec_price_max_mult * max_penalty
    secondary_low = np.array([sec_price_min, -aq["quantity_max"]], dtype=np.float32)
    secondary_high = np.array([sec_price_high, aq["quantity_max"]], dtype=np.float32)

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
            cap_t = env.cap_schedule.get_cap(current_year)
            # Use MSR preview for THIS year's supply (matches obs[20] = auction_volume_ratio)
            base_penalty_rate = float(config["penalty"]["rate"])
            inflation_rate = float(env._inflation_rate(current_year))
            this_year_auction_volume = env.cap_schedule.preview_auction_volume(
                current_year,
                clearing_price=env.last_clearing_price,
                price_max=float(config["auction"]["price_max"]),
                penalty_rate=base_penalty_rate,
                inflation_rate=inflation_rate,
                price_ma3=price_ma3,
            ) + env._defaulted_volume_pending
            suspension_length = int(config["auction"].get("suspension_length", 2))

            auction_actions = np.zeros((n_agents, 10), dtype=np.float32)
            for i in range(n_agents):
                company = env.companies[i]
                h_auc = heuristic_policy.auction_action(
                    company, price_ma3, current_year, n_years, config,
                    bank=float(env.holdings[i]),
                    reserve_price=env._compute_dynamic_reserve(),
                    auction_volume=float(this_year_auction_volume),
                    cap_t=float(cap_t),
                    suspension_remaining=int(env._suspension_remaining[i]),
                    suspension_length=suspension_length,
                    collateral_load_last=float(env._last_collateral_load[i]),
                    loan_outstanding_norm=company.get_loan_outstanding_norm(),
                )
                p, q = h_auc[0], h_auc[1] / 3.0
                auction_actions[i] = np.array([p, q, p, q, p, q,
                                               h_auc[2], h_auc[3], h_auc[4], h_auc[5]],
                                              dtype=np.float32)

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
                    current_year=current_year,
                    n_years=n_years,
                    loan_outstanding_norm=company.get_loan_outstanding_norm(),
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
    W   = 100
    leg = "═" * W
    print(f"\n{leg}")
    print("  TRAINING CONSOLE — FIELD GUIDE")
    print(leg)
    print()
    print("  ┌─ CONTROL PLANE")
    print("  │  Ep / elapsed / ETA    Episode number · wall-clock time · estimated remaining")
    print("  │  ent / shp / ε         Entropy coef · shaping weight · ε-greedy rate")
    print("  │  [ENT-DECAY]           Entropy coefficient is being annealed down")
    print("  │  [cyc=Ax]              Soft cycling — only agent Ax updates its actor this interval")
    print("  │  [WARMUP]              Critic-only warmup phase (actor frozen)")
    print()
    print("  ┌─ MARKET TRAJECTORY  (one value per simulated year, oldest → newest)")
    print("  │  Price    Auction clearing price (€/t) · ↑/↓/→ = trend across episode years")
    print("  │  Emiss    Total fleet emissions (Mt/yr)")
    print("  │  Auct     Auction supply volume after MSR withholding/release (Mt)")
    print("  │  TNAC     Total Number of Allowances in Circulation at episode end (Mt)")
    print("  │  MSR      Shown only when cumulative cancellation > 0 — reserve & cancelled stock")
    print()
    print("  ┌─ HEALTH SNAPSHOT")
    print("  │  comply   % of (agent × year) pairs with zero shortfall (full compliance)")
    print("  │  green    Fleet-avg green energy fraction: start→end (Δpp = percentage-point change)")
    print("  │  sec      Secondary market: total volume · match rate · avg trade price")
    print("  │           ↓sell / ↑buy = mean sellers / buyers participating per year")
    print()
    print("  ┌─ PER-AGENT TABLE  (* = active cycling agent)")
    print("  │  Green      Green energy share: start→end %(Δpp)")
    print("  │  Emiss      Mean annual emissions (Mt)")
    print("  │  BidAmt     Mean annual bid quantity submitted to auction (Mt)")
    print("  │  Alloc      Mean annual allowances received from auction (Mt)")
    print("  │  Sf         Shortfall-years / total-years (compliance failures)")
    print("  │  T1/T2/T3   Final-year tranche bid prices (€/t) — low/mid/high price tiers")
    print("  │  Rew        Total reward summed over all years (raw RL signal)")
    print("  │  DiagPts    Sfin/Sgrn/Scomp shown as points on a 0–100 scale")
    print("  │               Sfin  = cost efficiency score       [0–100 pts]")
    print("  │               Sgrn  = green progress score        [0–100 pts]")
    print("  │               Scomp = weighted composite score    [0–100 pts]")
    print("  │  ALoss/CLoss  Actor / Critic loss from most recent PPO update")
    print("  │  Secondary    BNy/VMt@P€ = bought N years, V Mt at avg P €  │  HOLD = no trades")
    print()
    print("  ┌─ BOT TABLE  (heuristic agents — one row per bot)")
    print("  │  Same core columns as learning agents, including T1/T2/T3 prices")
    print("  │  Event board is printed as a separate section below the bot rows")
    print()
    print("  ┌─ EVENT BOARD  (separate visual block)")
    print("  │  Defaults/Suspensions by agent: A#/B#:<count> (how many year-events)")
    print("  │  Stuck states (learning agents): one line only if any condition occurs")
    print("  │  Format: ceiling=A#:<streak_ep> | floor=A#:<streak_ep> | zeroQty=A#:<streak_ep>")
    print("  │  Warnings (year-step counts) are printed here, not on the Health line")
    print("  │  Non-zero warning counters:")
    print("  │     lowAlloc    Allocation < 30% of auction volume (severe under-bidding)")
    print("  │     priceFloor  Clearing price hit the reserve floor (~30 €/t)")
    print("  │     priceCeil   Clearing price ≥ 90% of price_max (near hard cap)")
    print("  │     auctFail    Auction failed entirely (no valid bids or all below reserve)")
    print("  │     lowDemand   Total bid demand < 70% of supply (weak market)")
    print("  │     noInvest    All agents chose invest_frac ≈ 0 (no green investment)")
    print("  │     debtSpiral  Agent had 3+ consecutive shortfall years (chronic non-compliance)")
    print("  │     bidCluster  Bid price std dev < €5 (all agents bidding nearly the same)")
    print("  │     overBank    TNAC > 2× total annual emissions (excessive allowance banking)")
    print("  │     1sideSec    All secondary-market participants on same side (no counterparty)")
    print("  │     noTrade     Zero secondary-market volume this year")
    print("  │     cornering   One agent holds disproportionate share of allocations (HHI risk)")
    print("  │     rsvReject   > 25% of bids fell below the reserve price (rejected)")
    print(leg)


def _format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(datetime.timedelta(seconds=max(0, int(round(float(seconds))))))


def _trend_arrow(values: list) -> str:
    """Return a trend arrow by comparing 2nd-half mean to 1st-half mean."""
    if len(values) < 4:
        return ""
    mid = len(values) // 2
    a = float(sum(values[:mid]) / mid)
    b = float(sum(values[mid:]) / (len(values) - mid))
    if abs(a) < 1e-9:
        return ""
    pct = (b - a) / abs(a)
    if pct > 0.03:
        return " ↑"
    if pct < -0.03:
        return " ↓"
    return " →"


def _resolve_auto_episode_count(raw_value, n_episodes: int,
                                frac: float, min_count: int, max_count: int):
    """Resolve integer schedule values that may be set to auto/0."""
    auto = False
    if isinstance(raw_value, str):
        auto = (raw_value.strip().lower() == "auto")
    elif raw_value is None:
        auto = True
    else:
        try:
            auto = int(raw_value) <= 0
        except Exception:
            auto = True

    if auto:
        val = int(round(n_episodes * frac))
        val = max(min_count, min(max_count, val))
        return val, True

    return int(raw_value), False


def _apply_ppo_run_profile(config: dict, n_episodes: int) -> dict:
    """
    Resolve PPO settings by run length.

    Long runs (n_episodes > threshold) keep configured long-run PPO values.
    Short runs (n_episodes <= threshold) apply short_run_overrides when provided.
    """
    ppo_cfg = config.setdefault("ppo", {})
    threshold = int(ppo_cfg.get("long_run_episode_threshold", 20000))
    short_cfg = ppo_cfg.get("short_run_overrides", {}) or {}

    long_lr = float(ppo_cfg.get("lr", 0.0002))
    long_entropy_final = float(ppo_cfg.get("entropy_coef_final", 0.03))
    long_epu = int(ppo_cfg.get("episodes_per_update", 1))

    profile = "long-run" if n_episodes > threshold else "short-run"
    if profile == "short-run":
        ppo_cfg["lr"] = float(short_cfg.get("lr", long_lr))
        ppo_cfg["entropy_coef_final"] = float(
            short_cfg.get("entropy_coef_final", long_entropy_final)
        )
        ppo_cfg["episodes_per_update"] = int(
            short_cfg.get("episodes_per_update", long_epu)
        )

    return {
        "profile": profile,
        "threshold": threshold,
        "lr": float(ppo_cfg["lr"]),
        "entropy_coef_final": float(ppo_cfg["entropy_coef_final"]),
        "episodes_per_update": int(ppo_cfg["episodes_per_update"]),
        "long_defaults": {
            "lr": long_lr,
            "entropy_coef_final": long_entropy_final,
            "episodes_per_update": long_epu,
        },
    }


def train_one_seed(config: dict, seed: int, on_log=None):
    # Isolate per-run auto-resolved schedule values (e.g. shaping decay)
    # so earlier short runs do not mutate config used by later long runs.
    config = copy.deepcopy(config)

    # Reproducibility: seed all RNGs before any stochastic operation
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tr_cfg = config.get("tabula_rasa", {})
    if tr_cfg.get("enabled", False):
        n_ep = config["simulation"]["n_episodes"]
        # Disable BC and KL anchor
        config.setdefault("pretrain", {})["enabled"] = False
        config.setdefault("ppo", {})["kl_anchor_beta"] = 0.0

        # Override exploration
        explore_cfg = config.setdefault("exploration", {})
        explore_cfg["epsilon_start"] = tr_cfg["epsilon_start"]
        explore_cfg["epsilon_final"] = tr_cfg["epsilon_final"]
        explore_cfg["epsilon_decay_episodes"] = int(tr_cfg["epsilon_decay_frac"] * n_ep)
        explore_cfg["mode"] = tr_cfg.get("exploration_mode", "uniform")

        # Override entropy + critic warmup
        ppo_cfg = config.setdefault("ppo", {})
        ppo_cfg["entropy_coef"] = tr_cfg["entropy_coef"]
        ppo_cfg["entropy_decay_start"] = int(tr_cfg["entropy_decay_start_frac"] * n_ep)
        ppo_cfg["entropy_decay_window"] = int(tr_cfg["entropy_decay_window_frac"] * n_ep)
        ppo_cfg["critic_warmup_episodes"] = int(tr_cfg["critic_warmup_frac"] * n_ep)

        # Override shaping and HPP
        config.setdefault("reward", {})["shaping_decay_episode"] = int(
            tr_cfg["shaping_decay_frac"] * n_ep
        )
        config.setdefault("hpp", {})["warmup_episodes"] = int(
            tr_cfg["hpp_warmup_frac"] * n_ep
        )

        # Remove explicit action anchors (auction price still initializes
        # near expected market price via PPOAgent fallback).
        explore_cfg["auction_anchors"] = None
        explore_cfg["secondary_anchors"] = None
        print(
            "TABULA RASA: BC off, KL off, anchors off, side-balanced price exploration, "
            f"ε={tr_cfg['epsilon_start']}→{tr_cfg['epsilon_final']}, "
            f"entropy={tr_cfg['entropy_coef']}"
        )

    n_agents = config["companies"]["n_agents"]
    n_episodes = config["simulation"]["n_episodes"]
    n_years = config["simulation"]["n_years"]
    run_profile = _apply_ppo_run_profile(config, n_episodes)

    # Reward shaping decay schedule: allow auto-scaling from n_episodes.
    reward_cfg = config.setdefault("reward", {})
    shaping_decay_eps, shaping_decay_auto = _resolve_auto_episode_count(
        reward_cfg.get("shaping_decay_episode", 3000), n_episodes,
        frac=0.12, min_count=300, max_count=8000,
    )
    reward_cfg["shaping_decay_episode"] = shaping_decay_eps

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
    print(f"v8.3.1: OPEX delta + /annual_budget norm + phase-aware GAE + calibration tune | Carry-forward{cf_str}")
    print(f"Clipped Gaussian (no tanh) + P1-P8 active{curric_str}{eps_str}")
    print(
        f"PPO profile: {run_profile['profile']} "
        f"(threshold>{run_profile['threshold']} episodes) "
        f"| lr={run_profile['lr']} "
        f"| entropy_final={run_profile['entropy_coef_final']} "
        f"| episodes_per_update={run_profile['episodes_per_update']}"
    )
    print(f"{'='*60}")
    _print_training_legend()

    if shaping_decay_auto:
        print(f"Reward shaping decay: → 0 at episode {shaping_decay_eps} [auto] "
              f"(12% of {n_episodes}, clamp=[300, 8000]).")

    env = ETSEnvironment(config, seed=seed)
    agents = build_agents(env, config, seed)
    n_total_agents = env.n_total

    ppo_cfg = config["ppo"]

    pretrain_cfg = config.get("pretrain", {})
    eff_pretrain_cfg = dict(pretrain_cfg)
    pretrain_eps_eff, pretrain_eps_auto = _resolve_auto_episode_count(
        pretrain_cfg.get("episodes", 0), n_episodes,
        frac=0.04, min_count=40, max_count=2000,
    )
    pretrain_epochs_eff, pretrain_epochs_auto = _resolve_auto_episode_count(
        pretrain_cfg.get("epochs", 0), n_episodes,
        frac=0.0002, min_count=4, max_count=12,
    )
    eff_pretrain_cfg["episodes"] = pretrain_eps_eff
    eff_pretrain_cfg["epochs"] = pretrain_epochs_eff

    bc_ran = False
    if pretrain_cfg.get("enabled", False):
        if pretrain_eps_auto or pretrain_epochs_auto:
            print("Pretrain auto-scale: "
                  f"episodes={pretrain_eps_eff}, epochs={pretrain_epochs_eff} "
                  f"(n_episodes={n_episodes}).")
        pretrain_behavioral_cloning(agents, env, config, eff_pretrain_cfg, seed)
        bc_ran = True

    # Snapshot BC-trained weights as frozen KL anchors (only when BC was run)
    kl_beta_init = ppo_cfg.get("kl_anchor_beta", 0.0)
    kl_decay_eps, kl_decay_auto = _resolve_auto_episode_count(
        ppo_cfg.get("kl_anchor_decay_episodes", 0), n_episodes,
        frac=0.50, min_count=500, max_count=40000,
    )
    if bc_ran and kl_beta_init > 0.0:
        for agent in agents:
            agent.set_bc_anchor()
        auto_note = " [auto]" if kl_decay_auto else ""
        print(f"KL anchor: frozen BC policies captured for all {n_agents} agents "
              f"(β₀={kl_beta_init}, decay={kl_decay_eps} eps{auto_note}).")

    critic_warmup_eps, critic_warmup_auto = _resolve_auto_episode_count(
        ppo_cfg.get("critic_warmup_episodes", 0), n_episodes,
        frac=0.06, min_count=60, max_count=1200,
    )
    if critic_warmup_eps > 0:
        auto_note = " [auto]" if critic_warmup_auto else ""
        print(f"Critic-warmup: actor gradients frozen for first {critic_warmup_eps} episodes{auto_note}.")

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
    eps_decay_episodes, eps_decay_auto = _resolve_auto_episode_count(
        explore_cfg.get("epsilon_decay_episodes", 0), n_episodes,
        frac=0.35, min_count=300, max_count=12000,
    )
    if eps_start > 0.0:
        auto_note = " [auto]" if eps_decay_auto else ""
        print(f"Epsilon-greedy: {eps_start:.2f} → {eps_final:.2f} "
              f"over {eps_decay_episodes} episodes (physical-space){auto_note}.")

    # Historical Policy Pool (HPP) — anti-regression safety net
    hpp_cfg = config.get("hpp", {})
    hpp_enabled = hpp_cfg.get("enabled", False)
    hpp_pool_size = hpp_cfg.get("pool_size", 10)
    hpp_save_interval, hpp_save_auto = _resolve_auto_episode_count(
        hpp_cfg.get("save_interval", 0), n_episodes,
        frac=0.03, min_count=50, max_count=1000,
    )
    hpp_swap_prob = hpp_cfg.get("swap_prob", 0.20)
    hpp_warmup, hpp_warmup_auto = _resolve_auto_episode_count(
        hpp_cfg.get("warmup_episodes", 0), n_episodes,
        frac=0.20, min_count=200, max_count=6000,
    )
    # Per-agent FIFO pool of actor state_dicts (auction + secondary only)
    hpp_pools = [collections.deque(maxlen=hpp_pool_size) for _ in range(n_agents)]
    if hpp_enabled:
        auto_note = " [auto]" if (hpp_save_auto or hpp_warmup_auto) else ""
        print(
            f"HPP: pool={hpp_pool_size}, save every {hpp_save_interval} eps, "
            f"swap_prob={hpp_swap_prob:.0%}, warmup={hpp_warmup} eps{auto_note}."
        )

    # Batch accumulation: collect N episodes before triggering PPO update
    episodes_per_update = ppo_cfg.get("episodes_per_update", 1)
    if episodes_per_update > 1:
        print(f"Batch accumulation: {episodes_per_update} episodes per update "
              f"(~{episodes_per_update * n_years} transitions per batch).")

    # Cosine LR decay setup
    lr_decay_mode = ppo_cfg.get("lr_decay", "none")
    lr_min = ppo_cfg.get("lr_min", 0.0)
    actor_lr_init = ppo_cfg["lr"]
    critic_lr_init = ppo_cfg.get("critic_lr", ppo_cfg["lr"])
    if lr_decay_mode == "cosine":
        print(f"Cosine LR decay: actor {actor_lr_init}→{lr_min}, "
              f"critic {critic_lr_init}→{lr_min} over {n_episodes} episodes.")

    # Condition-based entropy tracker (auto-scales to n_episodes)
    entropy_tracker = EntropyConditionTracker(ppo_cfg, n_agents, n_episodes)

    # Print consolidated decay-to-zero schedules
    print(f"\nDecay-to-zero schedules (episode → 0):")
    print(f"  Shaping weight:   → 0 at ep {shaping_decay_eps}")
    print(f"  Entropy coef:     {entropy_tracker.coef_init:.4f} → {entropy_tracker.coef_final:.4f} "
          f"(start={entropy_tracker.decay_start}, window={entropy_tracker.decay_window})")
    if kl_beta_init > 0.0:
        print(f"  KL anchor beta:   {kl_beta_init} → 0 at ep {kl_decay_eps}")
    if eps_start > 0.0:
        print(f"  Epsilon-greedy:   {eps_start:.2f} → {eps_final:.2f} over {eps_decay_episodes} eps")
    if critic_warmup_eps > 0:
        print(f"  Critic warmup:    actor frozen for {critic_warmup_eps} eps")

    # --- CSV loggers ---
    results_dir = config["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Episode-level
    ep_path = os.path.join(results_dir, f"training_log_s{seed}.csv")
    ep_fields = ["episode", "clearing_price_last", "cap_last", "entropy_coef",
                 "shaping_weight", "entropy_decay_triggered", "active_agent",
                 "epsilon"]
    for i in range(n_total_agents):
        ep_fields += [f"reward_A{i+1}", f"reward_base_A{i+1}", f"reward_shaping_A{i+1}",
                  f"green_frac_A{i+1}", f"delta_green_A{i+1}",
                      f"penalty_A{i+1}", f"shortfall_A{i+1}", f"queue_size_A{i+1}",
                      f"actor_loss_A{i+1}", f"critic_loss_A{i+1}", f"bid_price_A{i+1}"]
    ep_fields += ["secondary_volume", "secondary_avg_price", "secondary_match_rate"]
    for i in range(n_total_agents):
        ep_fields += [f"sec_buy_vol_A{i+1}", f"sec_sell_vol_A{i+1}",
                      f"sec_buy_avg_px_A{i+1}", f"sec_sell_avg_px_A{i+1}",
                      f"sec_buy_years_A{i+1}", f"sec_sell_years_A{i+1}",
                      f"avg_sec_mult_A{i+1}", f"avg_sec_qty_A{i+1}",
                      f"avg_bid_mult_A{i+1}", f"avg_bid_coverage_A{i+1}",
                      f"sec_buy_intent_share_A{i+1}", f"sec_sell_intent_share_A{i+1}",
                      f"inv_onshore_share_A{i+1}", f"inv_offshore_share_A{i+1}",
                      f"inv_solar_share_A{i+1}"]
    ep_fields += ["price_start", "price_peak", "price_std"]  # episode price trajectory
    for i in range(n_total_agents):  # post-warmstart initial bank per agent
        ep_fields += [f"ep_start_bank_A{i+1}"]
    for i in range(n_total_agents):  # allocation + P5/P6/P8/MAC episode aggregates
        ep_fields += [f"mean_alloc_A{i+1}",
                      f"mean_shock_A{i+1}", f"max_shock_A{i+1}",
                      f"mean_cf_shock_A{i+1}", f"total_cancels_A{i+1}",
                      f"total_mac_reduction_A{i+1}"]
    ep_fields += [
        "warn_lowAlloc", "warn_priceFloor", "warn_priceCeil", "warn_auctFail",
        "warn_lowDemand", "warn_noInvest", "warn_debtSpiral", "warn_bidCluster",
        "warn_overBank", "warn_1sideSec", "warn_noTrade", "warn_cornering",
        "warn_rsvReject",
        "warn_agents_stuck_ceiling", "warn_agents_stuck_floor", "warn_agents_stuck_zeroQty",
    ]
    for i in range(n_agents):
        ep_fields += [f"streak_ceil_A{i+1}", f"streak_floor_A{i+1}", f"streak_zeroqty_A{i+1}"]
    # F: Episode-mean diagnostic scores per learning agent (F3)
    for i in range(n_agents):
        ep_fields += [f"diag_S_financial_A{i+1}", f"diag_S_green_A{i+1}",
                      f"diag_S_composite_A{i+1}"]
    ep_csv = open(ep_path, "w", newline="")
    ep_writer = csv.DictWriter(ep_csv, fieldnames=ep_fields)
    ep_writer.writeheader()

    # Year-level
    yr_path = os.path.join(results_dir, f"year_log_s{seed}.csv")
    yr_fields = ["episode", "year", "cap", "auction_volume", "tnac",
                 "clearing_price", "secondary_price", "msr_reserve",
                 "msr_total_cancelled", "msr_withhold_this_year", "msr_release_this_year",
                 "inflation_rate", "inflation_factor", "marginal_ef_used"]
    for i in range(n_total_agents):
        yr_fields += [f"bank_start_A{i+1}", f"alloc_A{i+1}", f"emissions_A{i+1}",
                      f"trade_qty_A{i+1}", f"trade_cost_A{i+1}", f"green_frac_A{i+1}",
                      f"delta_green_A{i+1}", f"shortfall_A{i+1}", f"penalty_A{i+1}",
                      f"reward_A{i+1}", f"reward_base_A{i+1}", f"reward_shaping_A{i+1}",
                      f"holdings_A{i+1}", f"invest_cost_A{i+1}",
                      f"collateral_cost_A{i+1}",
                      f"bid_price_A{i+1}",
                      # 3-tranche bid ladder
                      f"tranche_price_1_A{i+1}", f"tranche_qty_1_A{i+1}",
                      f"tranche_price_2_A{i+1}", f"tranche_qty_2_A{i+1}",
                      f"tranche_price_3_A{i+1}", f"tranche_qty_3_A{i+1}",
                      f"queue_size_A{i+1}",
                      f"emission_shock_A{i+1}", f"cf_shock_A{i+1}",  # P5/P6
                      f"cancellation_A{i+1}",  # P6
                      f"auction_cost_A{i+1}", f"secondary_net_A{i+1}",  # cost breakdown
                      f"compliance_surplus_A{i+1}", f"bank_end_A{i+1}",  # compliance
                      f"mac_reduction_A{i+1}", f"mac_cost_A{i+1}",  # MAC
                      f"terminal_bank_value_A{i+1}",
                      f"terminal_queue_value_A{i+1}",
                      f"terminal_liquidation_value_A{i+1}",
                      f"sec_price_mult_A{i+1}",   # Phase 2 action[0] per year
                      f"sec_qty_action_A{i+1}",   # Phase 2 action[1] per year (+buy/-sell)
                      f"sec_action_side_A{i+1}",  # -1=sell, 0=hold, 1=buy intent
                      f"bid_qty_mult_A{i+1}",
                      f"estimate_need_A{i+1}",
                      f"bid_coverage_A{i+1}",
                      f"bid_to_reserve_A{i+1}",
                      f"invest_tech_choice_A{i+1}",
                      # F: Diagnostic scores (F2)
                      f"diag_S_financial_A{i+1}",
                      f"diag_S_green_A{i+1}",
                      f"diag_S_composite_A{i+1}",
                      f"wtp_economic_A{i+1}",
                      f"wtp_budget_A{i+1}",
                      f"wtp_binding_A{i+1}",
                      f"invest_frac_pre_clip_A{i+1}",
                      f"invest_frac_post_clip_A{i+1}",
                      f"available_budget_A{i+1}",
                      f"compliance_share_of_available_A{i+1}"]
    yr_csv = open(yr_path, "w", newline="")
    yr_writer = csv.DictWriter(yr_csv, fieldnames=yr_fields)
    yr_writer.writeheader()

    # Register CSV cleanup so files are closed even on crash
    def _close_csvs():
        if not ep_csv.closed:
            ep_csv.close()
        if not yr_csv.closed:
            yr_csv.close()
    atexit.register(_close_csvs)

    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]
    # Flush logs every N episodes to reduce data loss if training aborts early.
    csv_flush_interval = int(
        config["logging"].get("csv_flush_interval", 1000)
    )
    csv_flush_interval = max(1, csv_flush_interval)
    best_total_reward = -np.inf
    # Persist most recent available update losses so sparse update schedules
    # still show losses at coarse log intervals (e.g. every 50 episodes).
    last_available_losses = [None] * n_agents
    last_loss_episode = [None] * n_agents

    train_t0 = time.time()
    recent_ep_durations = collections.deque(maxlen=200)
    msr_event_totals = {
        "emergency_release": 0,
        "containment_release": 0,
        "withdrawal_suppressed": 0,
    }

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

        episode_seed = seed + episode * 1000
        episode_rng = np.random.default_rng(episode_seed)

        obs1, _ = env.reset(seed=episode_seed)
        total_rewards = np.zeros(n_agents)
        # F3: Accumulate per-year diagnostic scores for episode-level summary
        ep_diag_accumulator = [{
            "S_financial": 0.0, "S_green": 0.0, "S_composite": 0.0, "count": 0
        } for _ in range(n_agents)]

        # HPP: swap some agents to historical policies for this episode's rollout
        hpp_swapped = {}  # agent_idx → saved (auc_sd, sec_sd)
        if hpp_enabled and episode >= hpp_warmup:
            for i in range(n_agents):
                if hpp_pools[i] and episode_rng.random() < hpp_swap_prob:
                    # Save current actor weights
                    hpp_swapped[i] = (
                        copy.deepcopy(agents[i].auction_policy.state_dict()),
                        copy.deepcopy(agents[i].secondary_policy.state_dict()),
                    )
                    # Load random historical policy for action selection
                    hist_auc, hist_sec = hpp_pools[i][
                        int(episode_rng.integers(len(hpp_pools[i])))]
                    agents[i].auction_policy.load_state_dict(hist_auc)
                    agents[i].secondary_policy.load_state_dict(hist_sec)

        for year in range(effective_n_years):
            # === PHASE 1: Auction + Investment ===
            auction_actions = np.zeros((n_agents, 10), dtype=np.float32)
            auction_raws = []
            auction_logps = []

            for i in range(n_agents):
                action, raw, logp = agents[i].select_auction_action(
                    obs1[i], epsilon=current_epsilon)
                auction_actions[i] = action
                auction_raws.append(raw)
                auction_logps.append(logp)

            obs2, auction_info = env.step_auction(auction_actions)

            # v8.3: Split rewards — compute auction-phase intermediate reward
            # (collapses all tranche costs into single auction-phase reward)
            r_auction = env.compute_auction_rewards()

            # MAPPO: construct global state from all agents' phase2 obs
            _centralized = config["ppo"].get("centralized_critic", False)
            global_state = obs2.flatten() if _centralized else None

            # Store auction-phase transition (done=False: episode continues)
            for i in range(n_agents):
                value_auc = agents[i].estimate_value(
                    global_state if _centralized else obs2[i])
                agents[i].store_transition(
                    obs1=obs1[i], obs2=obs2[i],
                    auc_raw=auction_raws[i], sec_raw=np.zeros(2, dtype=np.float32),
                    auc_lp=auction_logps[i], sec_lp=np.zeros(1, dtype=np.float32),
                    reward=float(r_auction[i]), done=False, value=value_auc,
                    global_state=global_state,
                    phase='auction',  # auction policy trained on these rows
                )

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

            # v8.3: Split rewards — secondary reward = total reward - auction reward
            r_secondary = rewards - r_auction[:n_agents]

            # v8.3: Store raw rewards directly in buffer — batch normalization
            # happens inside compute_gae(). RewardNormalizer kept for monitoring only.
            for i in range(n_agents):
                agents[i].normalize_reward(rewards[i])  # update stats for logging only

            # Store secondary-phase transition
            for i in range(n_agents):
                value_sec = agents[i].estimate_value(
                    global_state if _centralized else obs2[i])
                agents[i].store_transition(
                    obs1=obs1[i], obs2=obs2[i],
                    auc_raw=auction_raws[i], sec_raw=secondary_raws[i],
                    auc_lp=auction_logps[i], sec_lp=secondary_logps[i],
                    reward=float(r_secondary[i]), done=terminated, value=value_sec,
                    global_state=global_state,
                    phase='secondary',  # secondary policy trained on these rows
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
                "msr_total_cancelled": yl.get("msr_total_cancelled", 0),
                "msr_withhold_this_year": yl.get("msr_withhold_this_year", 0),
                "msr_release_this_year": yl.get("msr_release_this_year", 0),
                "inflation_rate": yl.get("inflation_rate", 0),
                "inflation_factor": yl.get("inflation_factor", 1.0),
            }
            for i in range(n_total_agents):
                def _get(log_key, default=0):
                    vals = yl.get(log_key, [default] * n_total_agents)
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
                yr_row[f"reward_base_A{i+1}"] = _get("rewards_base")
                yr_row[f"reward_shaping_A{i+1}"] = _get("rewards_shaping")
                yr_row[f"holdings_A{i+1}"] = _get("holdings")
                yr_row[f"invest_cost_A{i+1}"] = _get("invest_costs")
                yr_row[f"collateral_cost_A{i+1}"] = _get("collateral_costs")
                yr_row[f"bid_price_A{i+1}"] = _get("bid_prices")
                # 3-tranche bid ladder data
                tranche_prices_all = yl.get("tranche_prices_sorted", [])
                tranche_qtys_all = yl.get("tranche_quantities_sorted", [])
                if i < len(tranche_prices_all):
                    for t in range(3):  # 3 tranches
                        if t < len(tranche_prices_all[i]):
                            yr_row[f"tranche_price_{t+1}_A{i+1}"] = round(tranche_prices_all[i][t], 2)
                            yr_row[f"tranche_qty_{t+1}_A{i+1}"] = round(tranche_qtys_all[i][t], 4)
                        else:
                            yr_row[f"tranche_price_{t+1}_A{i+1}"] = 0.0
                            yr_row[f"tranche_qty_{t+1}_A{i+1}"] = 0.0
                else:
                    for t in range(3):
                        yr_row[f"tranche_price_{t+1}_A{i+1}"] = 0.0
                        yr_row[f"tranche_qty_{t+1}_A{i+1}"] = 0.0
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
                yr_row[f"terminal_bank_value_A{i+1}"] = _get("terminal_bank_values")
                yr_row[f"terminal_queue_value_A{i+1}"] = _get("terminal_queue_values")
                yr_row[f"terminal_liquidation_value_A{i+1}"] = _get("terminal_liquidation_values")
                yr_row[f"sec_price_mult_A{i+1}"] = round(_get("sec_price_mults", default=1.0), 4)
                yr_row[f"sec_qty_action_A{i+1}"] = round(_get("sec_qty_actions", default=0.0), 4)
                yr_row[f"sec_action_side_A{i+1}"] = int(_get("sec_action_sides", default=0))
                yr_row[f"bid_qty_mult_A{i+1}"] = round(_get("bid_qty_multipliers", default=0.0), 4)
                yr_row[f"estimate_need_A{i+1}"] = round(_get("estimate_needs", default=0.0), 4)
                yr_row[f"bid_coverage_A{i+1}"] = round(_get("bid_coverages", default=0.0), 4)
                yr_row[f"bid_to_reserve_A{i+1}"] = round(_get("bid_to_reserve_ratio", default=0.0), 4)
                yr_row[f"invest_tech_choice_A{i+1}"] = int(_get("invest_tech_choices", default=-1))
                # Per-bot compliance diagnostics
                pad = yl.get("per_agent_diag", {})
                agent_diag = pad.get(i, {})
                yr_row[f"wtp_economic_A{i+1}"] = round(float(agent_diag.get("wtp_economic", agent_diag.get("wtp", float("nan")))), 4) if not np.isnan(agent_diag.get("wtp_economic", agent_diag.get("wtp", float("nan")))) else None
                yr_row[f"wtp_budget_A{i+1}"] = round(float(agent_diag.get("wtp_budget", float("nan"))), 4) if not np.isnan(float(agent_diag.get("wtp_budget", float("nan")))) else None
                yr_row[f"wtp_binding_A{i+1}"] = str(agent_diag.get("wtp_binding", ""))
                yr_row[f"invest_frac_pre_clip_A{i+1}"] = round(float(agent_diag.get("invest_frac_pre_compliance_clip", float("nan"))), 6) if not np.isnan(float(agent_diag.get("invest_frac_pre_compliance_clip", float("nan")))) else None
                yr_row[f"invest_frac_post_clip_A{i+1}"] = round(float(agent_diag.get("invest_frac_post_compliance_clip", float("nan"))), 6) if not np.isnan(float(agent_diag.get("invest_frac_post_compliance_clip", float("nan")))) else None
                yr_row[f"available_budget_A{i+1}"] = round(float(agent_diag.get("available_budget", float("nan"))), 2) if not np.isnan(float(agent_diag.get("available_budget", float("nan")))) else None
                yr_row[f"compliance_share_of_available_A{i+1}"] = round(float(agent_diag.get("compliance_cost_share_of_budget", float("nan"))), 4) if not np.isnan(float(agent_diag.get("compliance_cost_share_of_budget", float("nan")))) else None
            # F3: Diagnostic scores (one set per learning agent)
            try:
                diag_scores = env.compute_diagnostic_score()
                for ds in diag_scores:
                    aid = ds["agent_id"]
                    yr_row[f"diag_S_financial_A{aid+1}"] = ds["S_financial"]
                    yr_row[f"diag_S_green_A{aid+1}"] = ds["S_green"]
                    yr_row[f"diag_S_composite_A{aid+1}"] = ds["S_composite"]
                    # Accumulate for episode-level console display
                    if aid < len(ep_diag_accumulator):
                        ep_diag_accumulator[aid]["S_financial"] += ds["S_financial"]
                        ep_diag_accumulator[aid]["S_green"] += ds["S_green"]
                        ep_diag_accumulator[aid]["S_composite"] += ds["S_composite"]
                        ep_diag_accumulator[aid]["count"] += 1
            except Exception:
                pass  # diagnostic scoring is non-critical
            yr_row["marginal_ef_used"] = round(float(yl.get("marginal_ef_used", getattr(env, "_last_marginal_ef", 0.0))), 4)
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

        # === PPO Update (batched: every episodes_per_update episodes) ===
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

        # Batch accumulation: only update every episodes_per_update episodes.
        # Do NOT clear buffers between episodes within a batch.
        is_update_episode = ((episode + 1) % episodes_per_update == 0) or (episode == n_episodes - 1)

        latest_losses = []
        if is_update_episode:
            if happo_enabled:
                # === HAPPO: Sequential update with cumulative importance ratios ===
                # 1. Compute GAE advantages per agent (using their own centralized critics)
                gae_data = []
                for i in range(n_agents):
                    adv, ret, buf = agents[i].compute_gae(last_value=0.0)
                    gae_data.append((adv, ret, buf))

                # 2. Sequential update ordered by initial emission intensity (highest first).
                # v8.3: Fixed ordering ensures highest emitters get the cleanest
                # advantages (before ratio drift from earlier updates).
                order = sorted(range(n_agents), key=lambda i: env.companies[i].initial_ef, reverse=True)
                # Determine expected trajectory length for HAPPO ratio chain.
                # v8.3: With split rewards, T = 2 × n_years per episode.
                # Agents with mismatched T (e.g. HPP-cleared buffers) are
                # excluded from the M-factor accumulation chain entirely.
                expected_T = episodes_per_update * n_years * 2
                cumulative_ratio = torch.ones(expected_T, 1)

                for j in order:
                    adv_j, ret_j, buf_j = gae_data[j]
                    if buf_j is None:
                        latest_losses.append(None)
                        continue

                    T_j = int(buf_j["T"])
                    # Skip agents with mismatched T from HAPPO ratio chain
                    # (e.g. HPP-cleared buffers give shorter trajectories)
                    if T_j != expected_T:
                        loss = agents[j].update_happo(
                            adv_t=adv_j,
                            ret_t=ret_j,
                            buf_tensors=buf_j,
                            advantage_weights=None,  # standard PPO, no M-factor
                            actor_update=actor_update,
                        )
                        latest_losses.append(loss)
                        continue
                    ratio_weight_j = cumulative_ratio

                    loss = agents[j].update_happo(
                        adv_t=adv_j,
                        ret_t=ret_j,
                        buf_tensors=buf_j,
                        advantage_weights=ratio_weight_j,
                        actor_update=actor_update,
                    )
                    latest_losses.append(loss)

                    # Compute post-update importance ratio for this agent
                    if actor_update and T_j == expected_T:
                        ratio_j = agents[j].compute_post_update_ratio(buf_j)
                        clipped_j = torch.clamp(ratio_j, 1 - clip_eps, 1 + clip_eps)
                        cumulative_ratio = (
                            cumulative_ratio * clipped_j.detach().cpu()
                        )

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
                            orig_lr = agents[i].actor_optimizer.param_groups[0]["lr"]
                            for pg in agents[i].actor_optimizer.param_groups:
                                pg["lr"] = orig_lr * cycling_lr_scale
                            loss = agents[i].update(last_value=0.0, actor_update=actor_update)
                            for pg in agents[i].actor_optimizer.param_groups:
                                pg["lr"] = orig_lr
                        else:
                            agents[i].buffer.clear()
                            loss = None
                    else:
                        loss = agents[i].update(last_value=0.0, actor_update=actor_update)
                    latest_losses.append(loss)

            # --- Cosine LR decay (applied after each PPO update batch) ---
            if lr_decay_mode == "cosine":
                frac = episode / max(n_episodes - 1, 1)
                actor_lr_now = lr_min + 0.5 * (actor_lr_init - lr_min) * (1 + math.cos(math.pi * frac))
                critic_lr_now = lr_min + 0.5 * (critic_lr_init - lr_min) * (1 + math.cos(math.pi * frac))
                for agent in agents:
                    for pg in agent.actor_optimizer.param_groups:
                        pg["lr"] = actor_lr_now
                    for pg in agent.critic_optimizer.param_groups:
                        pg["lr"] = critic_lr_now
        else:
            # Non-update episode: keep buffers, report no losses
            latest_losses = [None] * n_agents

        for i in range(n_agents):
            if i < len(latest_losses) and latest_losses[i]:
                last_available_losses[i] = latest_losses[i]
                last_loss_episode[i] = episode

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

        # Average bid price per agent across the episode.
        # Use quantity-weighted average when possible so Bid€ reflects submitted volume.
        avg_bid_per_agent = []
        for i in range(n_total_agents):
            bids_this_ep = [
                yl["bid_prices"][i]
                for yl in env.episode_log
                if "bid_prices" in yl and i < len(yl["bid_prices"])
            ]
            qtys_this_ep_for_price = [
                max(0.0, yl["bid_quantities"][i])
                for yl in env.episode_log
                if "bid_quantities" in yl and i < len(yl["bid_quantities"])
            ]
            if bids_this_ep and qtys_this_ep_for_price and len(bids_this_ep) == len(qtys_this_ep_for_price):
                total_qty = float(np.sum(qtys_this_ep_for_price))
                if total_qty > 1e-9:
                    weighted_px = float(np.dot(bids_this_ep, qtys_this_ep_for_price) / total_qty)
                    avg_bid_per_agent.append(weighted_px)
                else:
                    avg_bid_per_agent.append(float(np.mean(bids_this_ep)))
            else:
                avg_bid_per_agent.append(float(np.mean(bids_this_ep)) if bids_this_ep else 0.0)

        # Average bid quantity (Mt) per agent — Phase 1 action[1] after multiplier expansion
        avg_bid_qty_per_agent = []
        for i in range(n_total_agents):
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
                    print(f"⚠ STUCK CEILING: A{_i+1} ({_streak_ceil[_i]}ep)  "
                          f"avg bid ≥{_ceil_thresh*100:.0f}% of {_price_max:.0f}€  (ep {episode})")
                if _streak_floor[_i] >= _broken_window and _streak_floor[_i] % _broken_window == 0:
                    print(f"⚠ STUCK FLOOR: A{_i+1} ({_streak_floor[_i]}ep)  "
                          f"avg bid ≤{_price_min:.0f}€  (ep {episode})")
                # Per-agent entropy boost — one-time at threshold
                if _stuck_boost_window > 0:
                    if (_streak_floor[_i] == _stuck_boost_window or
                            _streak_ceil[_i] == _stuck_boost_window):
                        side = "FLOOR" if _streak_floor[_i] >= _stuck_boost_window else "CEILING"
                        print(f"🔄 ENTROPY BOOST: A{_i+1} stuck {side} ({_stuck_boost_window}ep)"
                              f"  → ent={_stuck_boost_coef:.3f}  (ep {episode})")
                if _streak_qty[_i] >= _broken_window and _streak_qty[_i] % _broken_window == 0:
                    print(f"⚠ STUCK ZERO-QTY: A{_i+1} ({_streak_qty[_i]}ep)  "
                          f"avg qty ≤{_zero_qty_thresh*100:.0f}% of {_qty_max:.1f}Mt  (ep {episode})")

        # Average invest_frac action per agent — Phase 1 action[2]
        avg_invest_frac_per_agent = []
        for i in range(n_total_agents):
            frac_this_ep = [
                yl["invest_fracs"][i]
                for yl in env.episode_log
                if "invest_fracs" in yl and i < len(yl["invest_fracs"])
            ]
            avg_invest_frac_per_agent.append(np.mean(frac_this_ep) if frac_this_ep else 0.0)

        # Average secondary price multiplier per agent — Phase 2 action[0]
        avg_sec_mult_per_agent = []
        for i in range(n_total_agents):
            mults_this_ep = [
                yl["sec_price_mults"][i]
                for yl in env.episode_log
                if "sec_price_mults" in yl and i < len(yl["sec_price_mults"])
            ]
            avg_sec_mult_per_agent.append(np.mean(mults_this_ep) if mults_this_ep else 1.0)

        # Average secondary qty action per agent — Phase 2 action[1] (+ve=buy -ve=sell)
        avg_sec_qty_per_agent = []
        for i in range(n_total_agents):
            sqt_this_ep = [
                yl["sec_qty_actions"][i]
                for yl in env.episode_log
                if "sec_qty_actions" in yl and i < len(yl["sec_qty_actions"])
            ]
            avg_sec_qty_per_agent.append(np.mean(sqt_this_ep) if sqt_this_ep else 0.0)

        # Average Phase-1 bid multiplier + coverage ratio per agent
        avg_bid_mult_per_agent = []
        avg_bid_coverage_per_agent = []
        for i in range(n_total_agents):
            mult_this_ep = [
                yl["bid_qty_multipliers"][i]
                for yl in env.episode_log
                if "bid_qty_multipliers" in yl and i < len(yl["bid_qty_multipliers"])
            ]
            cov_this_ep = [
                yl["bid_coverages"][i]
                for yl in env.episode_log
                if "bid_coverages" in yl and i < len(yl["bid_coverages"])
            ]
            avg_bid_mult_per_agent.append(np.mean(mult_this_ep) if mult_this_ep else 0.0)
            avg_bid_coverage_per_agent.append(np.mean(cov_this_ep) if cov_this_ep else 0.0)

        # Secondary intent shares and investment-tech usage shares per agent
        sec_buy_intent_share = []
        sec_sell_intent_share = []
        inv_onshore_share = []
        inv_offshore_share = []
        inv_solar_share = []
        for i in range(n_total_agents):
            sec_sides = [
                yl["sec_action_sides"][i]
                for yl in env.episode_log
                if "sec_action_sides" in yl and i < len(yl["sec_action_sides"])
            ]
            tech_choices = [
                yl["invest_tech_choices"][i]
                for yl in env.episode_log
                if "invest_tech_choices" in yl and i < len(yl["invest_tech_choices"])
            ]

            if sec_sides:
                sec_sides_arr = np.array(sec_sides)
                sec_buy_intent_share.append(float(np.mean(sec_sides_arr > 0)))
                sec_sell_intent_share.append(float(np.mean(sec_sides_arr < 0)))
            else:
                sec_buy_intent_share.append(0.0)
                sec_sell_intent_share.append(0.0)

            if tech_choices:
                tech_arr = np.array(tech_choices)
                inv_onshore_share.append(float(np.mean(tech_arr == 0)))
                inv_offshore_share.append(float(np.mean(tech_arr == 1)))
                inv_solar_share.append(float(np.mean(tech_arr == 2)))
            else:
                inv_onshore_share.append(0.0)
                inv_offshore_share.append(0.0)
                inv_solar_share.append(0.0)

        # Per-agent secondary buy/sell breakdown across the episode
        per_agent_sec_stats = []
        for i in range(n_total_agents):
            buy_vol = 0.0; sell_vol = 0.0
            buy_cost = 0.0; sell_rev = 0.0
            buy_years = 0; sell_years = 0
            for yl in env.episode_log:
                tq = yl.get("trade_qtys", [0.0] * n_total_agents)
                tc = yl.get("trade_costs", [0.0] * n_total_agents)
                if i < len(tq):
                    if tq[i] > 1e-6:
                        buy_vol += tq[i]
                        buy_cost += tc[i]
                        buy_years += 1
                    elif tq[i] < -1e-6:
                        sell_vol += abs(tq[i])
                        sell_rev += abs(tc[i])
                        sell_years += 1
            buy_avg_px = buy_cost / buy_vol if buy_vol > 1e-6 else 0.0
            sell_avg_px = sell_rev / sell_vol if sell_vol > 1e-6 else 0.0
            per_agent_sec_stats.append({
                "buy_vol": buy_vol, "sell_vol": sell_vol,
                "buy_avg_px": buy_avg_px, "sell_avg_px": sell_avg_px,
                "buy_years": buy_years, "sell_years": sell_years,
            })

        # Per-agent episode aggregates
        ep_total_shortfalls = [
            sum(yl.get("shortfalls", [0] * n_total_agents)[i] for yl in env.episode_log)
            for i in range(n_total_agents)
        ]
        ep_total_penalties = [
            sum(yl.get("penalties", [0] * n_total_agents)[i] for yl in env.episode_log)
            for i in range(n_total_agents)
        ]
        ep_delta_greens = [
            sum(yl.get("delta_greens", [0] * n_total_agents)[i] for yl in env.episode_log)
            for i in range(n_total_agents)
        ]
        ep_avg_queue = [
            np.mean([yl.get("queue_sizes", [0] * n_total_agents)[i] for yl in env.episode_log])
            for i in range(n_total_agents)
        ]

        # P5/P6/P8 episode aggregates for diagnostics
        ep_mean_shock = [                                              # mean |ε| over years
            np.mean([abs(yl.get("emission_shocks", [0] * n_total_agents)[i])
                     for yl in env.episode_log])
            for i in range(n_total_agents)
        ]
        ep_max_shock = [                                               # worst-case ε
            max(yl.get("emission_shocks", [0] * n_total_agents)[i]
                for yl in env.episode_log)
            for i in range(n_total_agents)
        ]
        ep_mean_cf_shock = [                                           # mean |CF noise|
            np.mean([abs(yl.get("cf_shocks", [0] * n_total_agents)[i])
                     for yl in env.episode_log])
            for i in range(n_total_agents)
        ]
        ep_total_cancels = [                                           # total project cancellations
            sum(int(yl.get("cancellations", [0] * n_total_agents)[i]) for yl in env.episode_log)
            for i in range(n_total_agents)
        ]
        ep_total_mac_reduction = [                                     # total MAC abatement (Mt)
            sum(yl.get("mac_reductions", [0.0] * n_total_agents)[i] for yl in env.episode_log)
            for i in range(n_total_agents)
        ]

        ep_total_rewards_all = [
            sum(yl.get("rewards", [0.0] * n_total_agents)[i] for yl in env.episode_log)
            for i in range(n_total_agents)
        ]
        ep_total_rewards_base_all = [
            sum(yl.get("rewards_base", [0.0] * n_total_agents)[i] for yl in env.episode_log)
            for i in range(n_total_agents)
        ]
        ep_total_rewards_shaping_all = [
            sum(yl.get("rewards_shaping", [0.0] * n_total_agents)[i] for yl in env.episode_log)
            for i in range(n_total_agents)
        ]

        # Episode trajectory stats (across all years) — used in console only
        first_log   = env.episode_log[0] if env.episode_log else {}
        n_years_ep  = len(env.episode_log)
        prices_ep   = [yl.get("clearing_price", 0.0) for yl in env.episode_log]
        price_start = prices_ep[0]  if prices_ep else 0.0
        price_final = prices_ep[-1] if prices_ep else 0.0
        price_peak  = max(prices_ep) if prices_ep else 0.0

        ep_green_start = [
            first_log.get("green_fracs", [0.0] * n_total_agents)[i] for i in range(n_total_agents)
        ]
        # Post-warmstart initial holdings (bank_start at year 0), logged for the
        # "Holdings by Year" plot so it reflects the state after burn-in seeding.
        ep_start_bank = [
            first_log.get("bank_start", [0.0] * n_total_agents)[i] for i in range(n_total_agents)
        ]
        ep_green_end = [
            last_log.get("green_fracs", [0.0] * n_total_agents)[i] for i in range(n_total_agents)
        ]
        ep_mean_emiss = [
            np.mean([yl.get("emissions", [0.0] * n_total_agents)[i] for yl in env.episode_log])
            for i in range(n_total_agents)
        ]
        ep_mean_alloc = [
            np.mean([yl.get("allocations", [0.0] * n_total_agents)[i] for yl in env.episode_log])
            for i in range(n_total_agents)
        ]
        ep_shortfall_years = [                       # count of years where agent had shortfall
            sum(1 for yl in env.episode_log
                if yl.get("shortfalls", [0.0] * n_total_agents)[i] > 1e-6)
            for i in range(n_total_agents)
        ]

        price_std = float(np.std(prices_ep)) if len(prices_ep) > 1 else 0.0

        # ── Per-episode summary diagnostics (validation sequence, Run 1/2) ──
        # Coal bot indices: bots B1=n_agents, B2=n_agents+1 (coal-heavy mix)
        coal_bot_indices = [n_agents, n_agents + 1] if n_bot_agents >= 2 else []
        ep_mean_clearing = float(np.mean(prices_ep)) if prices_ep else 0.0

        # Mean coverage ratio for coal bots (post-compliance, from per_agent_diag)
        coal_coverages = []
        for yl in env.episode_log:
            pad = yl.get("per_agent_diag", {})
            for ci in coal_bot_indices:
                if ci in pad and "coverage_ratio_post_compliance" in pad[ci]:
                    coal_coverages.append(float(pad[ci]["coverage_ratio_post_compliance"]))
        ep_mean_coal_coverage = float(np.mean(coal_coverages)) if coal_coverages else float("nan")

        # Default count for this episode
        ep_default_count = sum(
            yl.get("auction_stats", {}).get("defaults", 0) for yl in env.episode_log
        )

        # Mean bid qty_mult across all agents
        all_bid_mults = []
        for yl in env.episode_log:
            bmults = yl.get("bid_qty_multipliers", [])
            all_bid_mults.extend([float(v) for v in bmults if v > 0])
        ep_mean_bid_qty_mult = float(np.mean(all_bid_mults)) if all_bid_mults else float("nan")

        # Mean coal-bot budget headroom after compliance (available budget at end of year)
        coal_headrooms = []
        for i, company in enumerate(env.companies):
            if i in coal_bot_indices:
                coal_headrooms.append(
                    max(0.0, float(company.annual_budget - company.budget_spent_this_year))
                )
        ep_mean_coal_budget_headroom = float(np.mean(coal_headrooms)) if coal_headrooms else float("nan")

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
            # Per-episode summary diagnostics
            "ep_mean_clearing_price": round(ep_mean_clearing, 2),
            "ep_mean_coal_coverage_ratio": round(ep_mean_coal_coverage, 4) if not np.isnan(ep_mean_coal_coverage) else None,
            "ep_default_count": ep_default_count,
            "ep_mean_bid_qty_mult": round(ep_mean_bid_qty_mult, 4) if not np.isnan(ep_mean_bid_qty_mult) else None,
            "ep_mean_coal_budget_headroom": round(ep_mean_coal_budget_headroom, 2) if not np.isnan(ep_mean_coal_budget_headroom) else None,
            "secondary_avg_price": round(avg_sec_price, 2),
            "secondary_match_rate": round(sec_match_rate, 3),
            "price_start": round(price_start, 2),
            "price_peak": round(price_peak, 2),
            "price_std": round(price_std, 2),
        }
        for i in range(n_total_agents):
            # Post-warmstart initial bank for "Holdings by Year" plot
            ep_row[f"ep_start_bank_A{i+1}"] = round(ep_start_bank[i], 4)
        for i in range(n_total_agents):
            ep_row[f"reward_A{i+1}"] = round(ep_total_rewards_all[i], 4)
            ep_row[f"reward_base_A{i+1}"] = round(ep_total_rewards_base_all[i], 4)
            ep_row[f"reward_shaping_A{i+1}"] = round(ep_total_rewards_shaping_all[i], 4)
            ep_row[f"green_frac_A{i+1}"] = round(
                last_log.get("green_fracs", [0] * n_total_agents)[i], 4)
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
            if i < n_agents and latest_losses[i]:
                ep_row[f"actor_loss_A{i+1}"] = round(latest_losses[i]["actor_loss"], 6)
                ep_row[f"critic_loss_A{i+1}"] = round(latest_losses[i]["critic_loss"], 6)
            else:
                ep_row[f"actor_loss_A{i+1}"] = 0.0
                ep_row[f"critic_loss_A{i+1}"] = 0.0
            ss = per_agent_sec_stats[i]
            ep_row[f"sec_buy_vol_A{i+1}"]     = round(ss["buy_vol"], 4)
            ep_row[f"sec_sell_vol_A{i+1}"]    = round(ss["sell_vol"], 4)
            ep_row[f"sec_buy_avg_px_A{i+1}"]  = round(ss["buy_avg_px"], 2)
            ep_row[f"sec_sell_avg_px_A{i+1}"] = round(ss["sell_avg_px"], 2)
            ep_row[f"sec_buy_years_A{i+1}"]   = ss["buy_years"]
            ep_row[f"sec_sell_years_A{i+1}"]  = ss["sell_years"]
            ep_row[f"avg_sec_mult_A{i+1}"]    = round(avg_sec_mult_per_agent[i], 4)
            ep_row[f"avg_sec_qty_A{i+1}"]     = round(avg_sec_qty_per_agent[i], 4)
            ep_row[f"avg_bid_mult_A{i+1}"]    = round(avg_bid_mult_per_agent[i], 4)
            ep_row[f"avg_bid_coverage_A{i+1}"] = round(avg_bid_coverage_per_agent[i], 4)
            ep_row[f"sec_buy_intent_share_A{i+1}"] = round(sec_buy_intent_share[i], 4)
            ep_row[f"sec_sell_intent_share_A{i+1}"] = round(sec_sell_intent_share[i], 4)
            ep_row[f"inv_onshore_share_A{i+1}"] = round(inv_onshore_share[i], 4)
            ep_row[f"inv_offshore_share_A{i+1}"] = round(inv_offshore_share[i], 4)
            ep_row[f"inv_solar_share_A{i+1}"] = round(inv_solar_share[i], 4)

        # F3: Episode-mean diagnostic scores per learning agent
        for i in range(n_agents):
            acc = ep_diag_accumulator[i]
            n_years_diag = max(acc["count"], 1)
            ep_row[f"diag_S_financial_A{i+1}"] = round(acc["S_financial"] / n_years_diag, 4)
            ep_row[f"diag_S_green_A{i+1}"] = round(acc["S_green"] / n_years_diag, 4)
            ep_row[f"diag_S_composite_A{i+1}"] = round(acc["S_composite"] / n_years_diag, 4)

        ep_row["warn_lowAlloc"] = int(env._warnings.get("low_alloc", 0))
        ep_row["warn_priceFloor"] = int(env._warnings.get("price_floor", 0))
        ep_row["warn_priceCeil"] = int(env._warnings.get("price_ceil", 0))
        ep_row["warn_auctFail"] = int(env._warnings.get("auct_fail", 0))
        ep_row["warn_lowDemand"] = int(env._warnings.get("low_demand", 0))
        ep_row["warn_noInvest"] = int(env._warnings.get("no_invest", 0))
        ep_row["warn_debtSpiral"] = int(env._warnings.get("debt_spiral", 0))
        ep_row["warn_bidCluster"] = int(env._warnings.get("bid_cluster", 0))
        ep_row["warn_overBank"] = int(env._warnings.get("over_bank", 0))
        ep_row["warn_1sideSec"] = int(env._warnings.get("one_side_sec", 0))
        ep_row["warn_noTrade"] = int(env._warnings.get("no_trade", 0))
        ep_row["warn_cornering"] = int(env._warnings.get("cornering", 0))
        ep_row["warn_rsvReject"] = int(env._warnings.get("rsv_reject", 0))
        ep_row["warn_agents_stuck_ceiling"] = int(np.sum(_streak_ceil >= _broken_window))
        ep_row["warn_agents_stuck_floor"] = int(np.sum(_streak_floor >= _broken_window))
        ep_row["warn_agents_stuck_zeroQty"] = int(np.sum(_streak_qty >= _broken_window))
        for i in range(n_agents):
            ep_row[f"streak_ceil_A{i+1}"] = int(_streak_ceil[i])
            ep_row[f"streak_floor_A{i+1}"] = int(_streak_floor[i])
            ep_row[f"streak_zeroqty_A{i+1}"] = int(_streak_qty[i])

        ep_writer.writerow(ep_row)

        # Aggregate MSR intervention frequencies for a single run-level summary.
        ep_msr_events = env.cap_schedule.msr_event_counts()
        for k in msr_event_totals:
            msr_event_totals[k] += int(ep_msr_events.get(k, 0))

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
            warn_detail = " ".join(_warn_parts) if _warn_parts else "none"

            # ── Enhanced secondary market breakdown ────────────────────
            n_total = env.n_total  # learning + bots
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

            # ── Market dynamics aggregates ─────────────────────────────
            total_emiss_ep = sum(
                sum(yl.get("emissions", [0.0] * n_total))
                for yl in env.episode_log
            )
            avg_annual_emiss = total_emiss_ep / max(n_years_ep, 1)
            total_agent_years = n_total * n_years_ep
            compliant_ay = sum(
                sum(1 for s in yl.get("shortfalls", [0.0] * n_total) if s < 1e-6)
                for yl in env.episode_log
            )
            compliance_rate = compliant_ay / max(total_agent_years, 1)
            avg_green_start_all = float(np.mean(ep_green_start[:n_total]))
            avg_green_end_all = float(np.mean(ep_green_end[:n_total]))

            # ── Year-by-year trajectories ──────────────────────────────
            yr_emiss = [sum(yl.get("emissions", [0.0] * n_total)[j] for j in range(n_total))
                        for yl in env.episode_log]
            yr_bid_total = [sum(yl.get("bid_quantities", [0.0] * n_total)[j] for j in range(n_total))
                            for yl in env.episode_log]
            yr_auct_vol = [yl.get("auction_volume", yl.get("cap", 0.0)) for yl in env.episode_log]

            # MSR status tracking
            msr_cancelled_total = last_log.get("msr_total_cancelled", 0.0)
            msr_str = (f"  MSR: rsv={last_log.get('msr_reserve', 0.0):.1f}Mt"
                       f"  canc={msr_cancelled_total:.2f}Mt") if msr_cancelled_total > 0.01 else ""

            n_bot_agents = config["companies"].get("n_bot_agents", 0)

            # ── Tranche prices from final year (for agent rows) ────────
            _last_tp = []
            if env.episode_log:
                _last_yl = env.episode_log[-1]
                _last_tp = _last_yl.get("tranche_prices_sorted", [])

            # ═══════════════════════════════════════════════════════════
            #  PRINT
            # ═══════════════════════════════════════════════════════════
            W = 130
            sep = "═" * W
            thin = "─" * W
            print(sep)

            # ── Control plane ──────────────────────────────────────────
            print(f"  Ep {episode:5d} │ {_format_hms(elapsed_s)} elapsed  ETA {_format_hms(eta_s)}"
                  f"  ({avg_ep_s:.2f} s/ep)"
                  f" │ ent={entropy_coef:.4f}  shp={env.shaping_weight:.3f}"
                  f"  ε={current_epsilon:.3f}"
                  f"{decay_str}{cyc_str}{warmup_str}")

            # ── Market trajectories ────────────────────────────────────
            price_traj = " ".join(f"{p:5.0f}" for p in prices_ep)
            emiss_traj = " ".join(f"{e:5.1f}" for e in yr_emiss)
            bid_traj = " ".join(f"{b:5.1f}" for b in yr_bid_total)
            auct_traj  = " ".join(f"{c:5.1f}" for c in yr_auct_vol)
            print(f"  {'Price/yr':<8}: {price_traj}   (σ={price_std:.0f}){_trend_arrow(prices_ep)}")
            print(f"  {'Emiss/yr':<8}: {emiss_traj}   (avg {avg_annual_emiss:.1f} Mt/yr){_trend_arrow(yr_emiss)}")
            print(f"  {'Bid/yr':<8}: {bid_traj}   (agents' total auction demand, Mt){_trend_arrow(yr_bid_total)}")
            print(f"  {'Auct/yr':<8}: {auct_traj}   (auction supply after cap+rollover+MSR, TNAC={tnac:.1f} Mt){msr_str}{_trend_arrow(yr_auct_vol)}")

            # ── Health snapshot ─────────────────────────────────────────
            _gs = avg_green_start_all * 100
            _ge = avg_green_end_all * 100
            _dg = _ge - _gs
            print(f"  Health: comply={compliance_rate*100:.0f}%"
                  f"  green={_gs:3.0f}→{_ge:3.0f}%({_dg:+.0f}pp)"
                  f" │ Sec: {total_sec_vol:.1f}Mt  match={sec_match_rate*100:.0f}%"
                  f"  avg={avg_sec_price:.0f}€  clear={sec_p:.0f}€"
                f"  (↓{sec_avg_sellers:.0f}sell/↑{sec_avg_buyers:.0f}buy)")

            # ── Per-agent table ─────────────────────────────────────────
            print(thin)
            print(f"  {'':4}  {'Green':>16}  {'Emiss':>5} {'BidAmt':>6} {'Alloc':>5}"
                  f" {'Sf':>4}  {'T1/T2/T3€':>11}"
                f"  {'Rew':>8}  {'DiagPts(F/G/C)':>16}"
                  f"  {'ALoss':>7} {'CLoss':>7}"
                  f"  Secondary")

            for i in range(n_agents):
                act_mark = "*" if (cycling_enabled and i == active_agent_idx) else " "
                g0 = ep_green_start[i] * 100
                g1 = ep_green_end[i] * 100
                dg = g1 - g0
                grn_str = f"{g0:3.0f}→{g1:3.0f}%({dg:+3.0f}pp)"
                sf_str = f"{ep_shortfall_years[i]}/{n_years_ep}"

                # Tranche prices from final year
                if i < len(_last_tp) and len(_last_tp[i]) >= 3:
                    t_str = f"{_last_tp[i][0]:3.0f}/{_last_tp[i][1]:3.0f}/{_last_tp[i][2]:3.0f}"
                else:
                    t_str = f"{avg_bid_per_agent[i]:3.0f}/  -/  -"

                # Diagnostic scores (inline)
                acc = ep_diag_accumulator[i]
                nd = max(acc["count"], 1)
                s_fin  = float(np.nan_to_num(acc["S_financial"] / nd, nan=0.0, posinf=1.0, neginf=0.0))
                s_grn  = float(np.nan_to_num(acc["S_green"] / nd, nan=0.0, posinf=1.0, neginf=0.0))
                s_comp = float(np.nan_to_num(acc["S_composite"] / nd, nan=0.0, posinf=1.0, neginf=0.0))
                s_fin_pts = int(np.clip(round(s_fin * 100.0), 0, 100))
                s_grn_pts = int(np.clip(round(s_grn * 100.0), 0, 100))
                s_cmp_pts = int(np.clip(round(s_comp * 100.0), 0, 100))
                diag_str = f"{s_fin_pts:3d}/{s_grn_pts:3d}/{s_cmp_pts:3d}"

                # Losses
                loss_i = latest_losses[i] if latest_losses[i] else last_available_losses[i]
                al_str = f"{loss_i['actor_loss']:.4f}" if loss_i else "    n/a"
                cl_str = f"{loss_i['critic_loss']:.4f}" if loss_i else "    n/a"

                # Secondary detail
                ss_i = per_agent_sec_stats[i]
                sec_parts = []
                if ss_i["buy_years"] > 0:
                    sec_parts.append(f"B{ss_i['buy_years']}y/{ss_i['buy_vol']:.1f}Mt@{ss_i['buy_avg_px']:.0f}€")
                if ss_i["sell_years"] > 0:
                    sec_parts.append(f"S{ss_i['sell_years']}y/{ss_i['sell_vol']:.1f}Mt@{ss_i['sell_avg_px']:.0f}€")
                sec_str = " ".join(sec_parts) if sec_parts else "HOLD"

                print(f"  A{i+1}{act_mark}: {grn_str:>16}"
                        f"  {ep_mean_emiss[i]:5.2f} {avg_bid_qty_per_agent[i]:6.2f} {ep_mean_alloc[i]:5.2f}"
                      f" {sf_str:>4}  {t_str:>11}"
                        f"  {ep_total_rewards_all[i]:8.1f}  {diag_str:>16}"
                      f"  {al_str:>7} {cl_str:>7}"
                      f"  {sec_str}")

            # ── Bot table (per-bot rows) ────────────────────────────────
            if n_bot_agents > 0:
                print(thin)
                print(f"  {'':4}  {'Green':>16}  {'Emiss':>5} {'BidAmt':>6} {'Alloc':>5}"
                      f" {'Sf':>4}  {'T1/T2/T3€':>11}"
                      f"  {'Rew':>8}  Secondary")
                for b in range(n_bot_agents):
                    j = n_agents + b
                    g0 = ep_green_start[j] * 100
                    g1 = ep_green_end[j] * 100
                    dg = g1 - g0
                    grn_str = f"{g0:3.0f}→{g1:3.0f}%({dg:+3.0f}pp)"
                    sf_str = f"{ep_shortfall_years[j]}/{n_years_ep}"
                    if j < len(_last_tp) and len(_last_tp[j]) >= 3:
                        t_bot_str = f"{_last_tp[j][0]:3.0f}/{_last_tp[j][1]:3.0f}/{_last_tp[j][2]:3.0f}"
                    else:
                        t_bot_str = f"{avg_bid_per_agent[j]:3.0f}/  -/  -"

                    # Secondary detail
                    ss_j = per_agent_sec_stats[j]
                    sec_parts_b = []
                    if ss_j["buy_years"] > 0:
                        sec_parts_b.append(f"B{ss_j['buy_years']}y/{ss_j['buy_vol']:.1f}Mt@{ss_j['buy_avg_px']:.0f}€")
                    if ss_j["sell_years"] > 0:
                        sec_parts_b.append(f"S{ss_j['sell_years']}y/{ss_j['sell_vol']:.1f}Mt@{ss_j['sell_avg_px']:.0f}€")
                    sec_str_b = " ".join(sec_parts_b) if sec_parts_b else "HOLD"

                    print(f"  B{b+1} : {grn_str:>16}"
                          f"  {ep_mean_emiss[j]:5.2f} {avg_bid_qty_per_agent[j]:6.2f} {ep_mean_alloc[j]:5.2f}"
                          f" {sf_str:>4}  {t_bot_str:>11}"
                          f"  {ep_total_rewards_all[j]:8.1f}  {sec_str_b}")

            # ── Event board (separate visual block) ────────────────────
            print(thin)
            print("  Event Board:")

            _default_counts = {}
            for yl in env.episode_log:
                _as = yl.get("auction_stats", {})
                _def_list = _as.get("defaults_agents", [])
                for _di in _def_list:
                    _idx = int(_di)
                    _default_counts[_idx] = _default_counts.get(_idx, 0) + 1

            if _default_counts:
                _def_items = []
                for _idx in sorted(_default_counts):
                    _tag = f"A{_idx + 1}" if _idx < n_agents else f"B{_idx - n_agents + 1}"
                    _def_items.append(f"{_tag}:{_default_counts[_idx]}")
                print(f"  Defaults/Suspensions by agent (count): {'  '.join(_def_items)}")
            else:
                print("  Defaults/Suspensions by agent (count): none")

            _stuck_ceil_agents = [
                f"A{i+1}:{int(v)}ep" for i, v in enumerate(_streak_ceil) if v >= _broken_window
            ]
            _stuck_floor_agents = [
                f"A{i+1}:{int(v)}ep" for i, v in enumerate(_streak_floor) if v >= _broken_window
            ]
            _stuck_qty_agents = [
                f"A{i+1}:{int(v)}ep" for i, v in enumerate(_streak_qty) if v >= _broken_window
            ]
            _stuck_parts = []
            if _stuck_ceil_agents:
                _stuck_parts.append("ceiling=" + "  ".join(_stuck_ceil_agents))
            if _stuck_floor_agents:
                _stuck_parts.append("floor=" + "  ".join(_stuck_floor_agents))
            if _stuck_qty_agents:
                _stuck_parts.append("zeroQty=" + "  ".join(_stuck_qty_agents))
            if _stuck_parts:
                print(f"  Stuck states (learning, >= {_broken_window}ep): "
                      f"{' | '.join(_stuck_parts)}")
            print(f"  Warnings (year-step counts): {warn_detail}")

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

    print(
        "MSR summary (run total): "
        f"emergency_release={msr_event_totals['emergency_release']}  "
        f"containment_release={msr_event_totals['containment_release']}  "
        f"withdrawal_suppressed={msr_event_totals['withdrawal_suppressed']}"
    )

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
