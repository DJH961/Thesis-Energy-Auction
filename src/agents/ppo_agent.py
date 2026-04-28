"""
ppo_agent.py
============
PPO agent with two-phase decision making for EU ETS:
  Phase 1 (Auction):   obs(18) → [bid_price, qty, invest_frac, tech_logits×3]
  Phase 2 (Secondary): obs(21) → [sec_price_abs, sec_qty]

On-policy: collects full episode rollout, then updates via
clipped surrogate objective with GAE advantage estimation.

Supports **MAPPO** (Multi-Agent PPO) via centralized critic:
  When `ppo.centralized_critic = true`, the value network V(s) receives the
  global state (concatenation of all agents' phase-2 observations) instead of
  the local observation.  Actors remain decentralized (CTDE paradigm).

"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence
from typing import Optional

from src.agents.actor_critic import AuctionPolicy, SecondaryPolicy, ValueNetwork


# ---------------------------------------------------------------------------
# Per-agent running reward normaliser
# ---------------------------------------------------------------------------

class RewardNormalizer:
    """
    Online running mean/std normaliser with adaptive warmup.

    reward_norm = (reward - mu) / (std + eps)

    Uses an adaptive alpha schedule: alpha_eff = max(alpha, 1/(n+1)).
    This gives alpha=1.0 on the first sample (mu = reward exactly),
    alpha=0.1 after 10 samples, converging to the steady-state alpha
    after ~1/alpha samples.  Eliminates the cold-start bias that causes
    wild normalised values in early training.

    Parameters
    ----------
    alpha : float
        Steady-state EMA decay rate. alpha=0.01 ≈ window of 100 samples.
    eps : float
        Numerical stability floor for std.
    """

    def __init__(self, alpha: float = 0.01, eps: float = 1e-8):
        self.alpha = alpha
        self.eps = eps
        self.mu = 0.0
        self.var = 1.0     # initialise to 1 so first normalised value ≈ raw reward
        self._n_samples = 0

    def update_and_normalize(self, reward: float) -> float:
        """Update running stats and return normalised reward (NOT clipped)."""
        # Adaptive alpha: fast warmup, stable long-term
        self._n_samples += 1
        effective_alpha = max(self.alpha, 1.0 / self._n_samples)

        # EMA mean (save old mean for unbiased variance update)
        old_mu = self.mu
        self.mu = (1.0 - effective_alpha) * old_mu + effective_alpha * reward
        # EMA variance (use old_mu to avoid double-counting the mean shift)
        self.var = (1.0 - effective_alpha) * self.var + effective_alpha * (reward - old_mu) ** 2
        std = max(self.var ** 0.5, self.eps)
        return (reward - self.mu) / std

    def reset(self):
        """Optionally reset stats (not called by default — stats persist across episodes)."""
        self.mu = 0.0
        self.var = 1.0
        self._n_samples = 0


# ---------------------------------------------------------------------------
# Observation index constants
# ---------------------------------------------------------------------------
OBS1_EXPECTED_PRICE_IDX = 3   # Phase-1 obs dim 3: normalized expected price (×price_max)
OBS1_NEED_IDX = 10             # Phase-1 obs dim 10: estimated_need / 10.0 (Mt)
OBS1_BUDGET_HEADROOM_IDX = 27  # Phase-1 obs dim 27: budget headroom (1.0=fresh)
OBS1_COVER_RATIO_IDX = 33      # Phase-1 obs dim 33: last auction cover_ratio / 3.0
OBS1_OWN_SEC_BUY_PRICE_IDX = 34   # Phase-1 obs dim 34: own last secondary buy price / price_max
OBS1_CUM_COVERAGE_RATIO_IDX = 35  # Phase-1 obs dim 35: cumulative alloc/emissions ratio / 2.0
# Phase-2 appends 10 dims to phase1; clearing_price_norm is extra dim [base+1].
# Relative index from end keeps this robust across opponent-modeling sizes.
OBS2_CLEARING_PRICE_IDX_FROM_END = -9

# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one episode of transitions for on-policy update."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs1 = []
        self.obs2 = []
        self.global_states = []  # MAPPO: centralized critic input
        self.auction_raw = []
        self.secondary_raw = []
        self.auction_logp = []
        self.secondary_logp = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.phases = []  # 'auction' or 'secondary' per transition (for phase-split gradients)

    def push(self, obs1, obs2, auc_raw, sec_raw, auc_lp, sec_lp, reward, done, value,
             global_state=None, phase='secondary'):
        """
        phase : str, either 'auction' or 'secondary'
            Tags which phase produced this transition so update_happo() and
            compute_post_update_ratio() can route each policy loss to the
            correct observation space.
        """
        self.obs1.append(obs1)
        self.obs2.append(obs2)
        if global_state is not None:
            self.global_states.append(global_state)
        self.auction_raw.append(auc_raw)
        self.secondary_raw.append(sec_raw)
        self.auction_logp.append(auc_lp)
        self.secondary_logp.append(sec_lp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.phases.append(phase)

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:

    def __init__(self, agent_id, obs_dim_phase1, obs_dim_phase2,
                 auction_action_low, auction_action_high,
                 secondary_action_low, secondary_action_high,
                 config, seed=None, global_state_dim=0):
        self.agent_id = agent_id
        self.config = config
        self.obs_dim_phase1 = obs_dim_phase1
        # Seeded RNG for reproducible mini-batch shuffling
        self._rng = np.random.default_rng(seed if seed is not None else 0 + agent_id)
        ppo = config["ppo"]

        self.gamma = ppo["gamma"]
        self.gae_lambda = ppo["gae_lambda"]
        self.clip_eps = ppo["clip_eps"]
        self.entropy_coef = ppo["entropy_coef"]  # updated per-episode by train.py via set_entropy_coef()
        self.value_coef = ppo["value_coef"]
        self.max_grad_norm = ppo["max_grad_norm"]
        self.n_epochs = ppo["n_epochs"]
        self.mini_batch_size = ppo["mini_batch_size"]
        self.normalize_advantages = ppo.get("normalize_advantages", True)
        reward_cfg = config.get("reward", {})
        self.gae_min_std = float(max(1e-8, reward_cfg.get("gae_min_std", 0.1)))
        self.target_kl = ppo.get("target_kl", 0.0)  # KL early stopping; 0 = disabled

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = ppo["hidden_size"]
        log_std_min = ppo.get("log_std_min", -2.0)
        log_std_max = ppo.get("log_std_max", 1.0)

        auction_dim = len(auction_action_low)
        secondary_dim = len(secondary_action_low)

        a_low = torch.FloatTensor(auction_action_low).to(self.device)
        a_high = torch.FloatTensor(auction_action_high).to(self.device)
        s_low = torch.FloatTensor(secondary_action_low).to(self.device)
        s_high = torch.FloatTensor(secondary_action_high).to(self.device)
        self.auction_price_max = float(auction_action_high[0])

        # Action anchors: realistic initial targets in physical space.
        # These shift the policy's initial mean output toward historically
        # plausible actions, giving agents a sensible starting point without
        # limiting what they can learn.
        explore_cfg = config.get("exploration", {})
        self.exploration_mode = explore_cfg.get("mode", "anchored")
        auction_anchors = explore_cfg.get("auction_anchors", None)
        secondary_anchors = explore_cfg.get("secondary_anchors", None)

        # Fundamental price anchor: MAC-based economically derived initial price.
        # Falls back to config price.initial_expected or 80.0 if anchor fails.
        try:
            from src.utils.price_anchor import compute_fundamental_anchor
            self.expected_price_fallback = compute_fundamental_anchor(0, config)
        except Exception:
            self.expected_price_fallback = float(
                config.get("price", {}).get("initial_expected", 80.0)
            )

        # WTP-anchored exploration parameters (Approach C+D: escape floor-price trap)
        penalty_cfg = config.get("penalty", {})
        self._wtp_penalty_rate = float(penalty_cfg.get("rate", 138.75))
        companies_cfg = config.get("companies", {})
        budgets = companies_cfg.get("annual_budgets", [880.0])
        idx = min(agent_id, len(budgets) - 1)
        self._wtp_annual_budget = float(budgets[idx])
        self._wtp_mac = float(config.get("mac", {}).get("coal_to_gas_cost", 48.0))

        # No explicit anchors: keep neutral defaults for non-price dimensions,
        # but initialize bid-price near the expected market level.
        if auction_anchors is None:
            auction_init_anchors_t = 0.5 * (a_low + a_high)
            auction_init_anchors_t[0] = float(np.clip(
                self.expected_price_fallback,
                float(a_low[0].item()),
                float(a_high[0].item()),
            ))
            auction_init_anchors = auction_init_anchors_t.detach().cpu().tolist()
        else:
            auction_init_anchors = auction_anchors

        self.auction_policy = AuctionPolicy(
            obs_dim_phase1, auction_dim, hidden, a_low, a_high,
            log_std_min=log_std_min, log_std_max=log_std_max,
            action_anchors=auction_init_anchors,
        ).to(self.device)

        self.secondary_policy = SecondaryPolicy(
            obs_dim_phase2, secondary_dim, hidden, s_low, s_high,
            log_std_min=log_std_min, log_std_max=log_std_max,
            action_anchors=secondary_anchors,
        ).to(self.device)

        # MAPPO: centralized critic sees global state (all agents' obs2 concatenated)
        self.centralized_critic = ppo.get("centralized_critic", False)
        critic_hidden = ppo.get("critic_hidden_size", hidden)

        if self.centralized_critic and global_state_dim > 0:
            self.value_net = ValueNetwork(global_state_dim, critic_hidden).to(self.device)
        else:
            self.value_net = ValueNetwork(obs_dim_phase2, hidden).to(self.device)

        # Separate actor/critic optimizers for independent learning rates
        actor_params = (
            list(self.auction_policy.parameters()) +
            list(self.secondary_policy.parameters())
        )
        critic_params = list(self.value_net.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=ppo["lr"])
        critic_lr = ppo.get("critic_lr", ppo["lr"])
        self.critic_optimizer = optim.Adam(critic_params, lr=critic_lr)
        # Backwards-compat alias used by cycling code in train.py
        self.optimizer = self.actor_optimizer

        self.buffer = RolloutBuffer()
        self.actor_loss_history = []
        self.critic_loss_history = []

        # Critic training enhancements
        self.critic_extra_epochs = ppo.get("critic_extra_epochs", 0)
        self.critic_huber = ppo.get("critic_huber", False)
        self.critic_huber_delta = ppo.get("critic_huber_delta", 10.0)
        self.normalize_returns = ppo.get("normalize_returns", True)
        self.clip_value = ppo.get("clip_value", False)

        # Initialize critic loss function
        if self.critic_huber:
            self.critic_loss_fn = nn.SmoothL1Loss(beta=self.critic_huber_delta)
        else:
            self.critic_loss_fn = nn.MSELoss()

        reward_cfg = config.get("reward", {})
        norm_alpha = reward_cfg.get("normalizer_alpha", 0.01)
        self._reward_normalizer = RewardNormalizer(alpha=norm_alpha)
        self._reward_clip_min = reward_cfg.get("clip_min", -10.0)
        self._reward_clip_max = reward_cfg.get("clip_max", 10.0)

        # KL anchor: frozen snapshot of BC-trained policy (set after BC pretraining)
        self._bc_auction_policy = None
        self._bc_secondary_policy = None
        self.kl_beta = 0.0

    def set_entropy_coef(self, coef: float):
        """Update entropy coefficient (decay schedule lives in train.py)."""
        self.entropy_coef = float(coef)

    def set_kl_beta(self, beta: float):
        """Update KL anchor penalty weight (decayed by train.py)."""
        self.kl_beta = float(beta)

    def inject_fundamental_anchor(self, year: int = 0) -> None:
        """Overwrite price_head.bias so the initial policy mean ≈ fundamental anchor.

        Called once from train.py after build_agents() (and after BC if enabled,
        so BC doesn't undo the calibration).  Safe to call multiple times.
        """
        import numpy as np
        from src.utils.price_anchor import compute_fundamental_anchor
        anchor = compute_fundamental_anchor(year, self.config)
        scale  = float(self.auction_policy.action_scale[0].item())
        bias   = float(self.auction_policy.action_bias[0].item())
        raw    = float(np.clip((anchor - bias) / (scale + 1e-8), -0.95, 0.95))
        with torch.no_grad():
            self.auction_policy.price_head.bias.fill_(raw)

    def set_bc_anchor(self):
        """
        Snapshot current auction and secondary policy weights as a frozen
        BC anchor.  Called by train.py immediately after BC pretraining.
        The anchor networks receive no gradient updates.
        """
        self._bc_auction_policy = copy.deepcopy(self.auction_policy).eval()
        for p in self._bc_auction_policy.parameters():
            p.requires_grad_(False)

        self._bc_secondary_policy = copy.deepcopy(self.secondary_policy).eval()
        for p in self._bc_secondary_policy.parameters():
            p.requires_grad_(False)

    def normalize_reward(self, reward: float) -> float:
        """Normalise reward with per-agent running stats, then clip."""
        r_norm = self._reward_normalizer.update_and_normalize(reward)
        return float(np.clip(r_norm, self._reward_clip_min, self._reward_clip_max))

    def _critic_loss(self, v_pred, v_target, old_values=None):
        """Huber or MSE loss, with optional value-function clipping."""
        if self.clip_value and old_values is not None:
            # Clip value predictions to old_values ± clip_eps
            v_pred_clipped = old_values + torch.clamp(
                v_pred - old_values, -self.clip_eps, self.clip_eps
            )
            # Compute loss for both clipped and unclipped, take max
            loss_unclipped = self.critic_loss_fn(v_pred, v_target)
            loss_clipped = self.critic_loss_fn(v_pred_clipped, v_target)
            return torch.max(loss_unclipped, loss_clipped)
        else:
            return self.critic_loss_fn(v_pred, v_target)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_auction_action(self, obs1: np.ndarray, deterministic=False,
                              epsilon: float = 0.0,
                              last_secondary_buy_price: float = 0.0,
                              current_year: int = 0):
        """Phase 1: obs(18) → (action[6], raw[6], logp[1]).

        When ``epsilon > 0`` and not deterministic, with probability *epsilon*
        an epsilon-random action in physical space replaces the policy sample.
        The raw action and log_prob are still computed under the current policy
        so that the PPO importance ratio remains correct.

        ``last_secondary_buy_price``: price this agent paid per Mt on the
        secondary market last year. Used for Approach C+D — shifts the WTP
        exploration anchor upward so agents learn to bid above floor when
        the secondary market is expensive.
        """
        obs_t = torch.FloatTensor(obs1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, raw, log_prob = self.auction_policy.act(obs_t, deterministic)

        if not deterministic and epsilon > 0.0 and np.random.random() < epsilon:
            with torch.no_grad():
                low = self.auction_policy.action_bias - self.auction_policy.action_scale
                high = self.auction_policy.action_bias + self.auction_policy.action_scale
                if self.exploration_mode == "uniform":
                    rand_action = low + (high - low) * torch.rand_like(action)

                    # WTP-anchored exploration (Approach C+D): anchor exploration
                    # on willingness-to-pay instead of expected price.
                    # This breaks the floor-price self-reinforcement loop where
                    # expected_price = floor → exploration samples near floor →
                    # clearing price = floor → expected_price stays at floor.
                    price_min = float(low[0].item())
                    price_max = float(high[0].item())

                    # Compute WTP anchor from agent's own state
                    need_mt = float(obs1[OBS1_NEED_IDX]) * 10.0 if len(obs1) > OBS1_NEED_IDX else 1.0
                    need_mt = max(need_mt, 0.01)
                    budget_headroom = float(obs1[OBS1_BUDGET_HEADROOM_IDX]) if len(obs1) > OBS1_BUDGET_HEADROOM_IDX else 0.5
                    available_budget = max(budget_headroom, 0.0) * self._wtp_annual_budget
                    # Year-adjusted fundamental anchor (yr1≈80, grows with year)
                    from src.utils.price_anchor import compute_fundamental_anchor
                    _boost = float(self.config.get("exploration", {}).get("anchor_boost", 1.14))
                    wtp_economic = compute_fundamental_anchor(current_year, self.config) * _boost
                    # WTP budget: what can the agent afford per tonne
                    wtp_budget = available_budget / need_mt if need_mt > 0.01 else price_max

                    # Approach C: shift anchor to secondary buy price if higher
                    # "Last year I paid 280 on secondary → I should bid at least
                    # that much at auction to avoid overpaying again."
                    wtp_base = min(wtp_economic, wtp_budget)
                    if last_secondary_buy_price > 0.0:
                        wtp_base = max(wtp_base, last_secondary_buy_price)

                    # Approach D: blend — bid floor = 0.5 * sec_price + 0.5 * econ
                    # This softens the shift so agents don't jump to full sec price
                    if last_secondary_buy_price > wtp_economic and last_secondary_buy_price > 0.0:
                        wtp_base = 0.5 * last_secondary_buy_price + 0.5 * wtp_economic

                    wtp_anchor = float(np.clip(wtp_base, price_min, price_max))

                    if wtp_anchor <= price_min + 1e-9:
                        sampled_price = np.random.uniform(price_min, price_max * 0.5)
                    elif wtp_anchor >= price_max - 1e-9:
                        sampled_price = np.random.uniform(price_min, wtp_anchor)
                    elif np.random.random() < 0.5:
                        sampled_price = np.random.uniform(price_min, wtp_anchor)
                    else:
                        sampled_price = np.random.uniform(wtp_anchor, price_max)
                    rand_action[0, 0] = float(sampled_price)
                else:
                    # Anchored exploration: sample each dim from Gaussian around
                    # realistic company expectations instead of uniform.
                    # This gives exploration a sensible starting distribution that
                    # reflects how real companies would initially behave.
                    rand_action = torch.zeros_like(action)
                    price_max = high[0].item()
                    price_min = low[0].item()

                    # [0] bid_price: Gaussian around year-adjusted fundamental anchor
                    need_mt = float(obs1[OBS1_NEED_IDX]) * 10.0 if len(obs1) > OBS1_NEED_IDX else 1.0
                    need_mt = max(need_mt, 0.01)
                    bh = float(obs1[OBS1_BUDGET_HEADROOM_IDX]) if len(obs1) > OBS1_BUDGET_HEADROOM_IDX else 0.5
                    avail = max(bh, 0.0) * self._wtp_annual_budget
                    from src.utils.price_anchor import compute_fundamental_anchor
                    _boost = float(self.config.get("exploration", {}).get("anchor_boost", 1.14))
                    wtp_e = compute_fundamental_anchor(current_year, self.config) * _boost
                    wtp_b = avail / need_mt if need_mt > 0.01 else price_max
                    wtp_base = min(wtp_e, wtp_b)
                    # Approach C: shift anchor to secondary buy price if higher
                    if last_secondary_buy_price > 0.0:
                        wtp_base = max(wtp_base, last_secondary_buy_price)
                    # Approach D: blend when secondary > economic
                    if last_secondary_buy_price > wtp_e and last_secondary_buy_price > 0.0:
                        wtp_base = 0.5 * last_secondary_buy_price + 0.5 * wtp_e
                    wtp_anc = float(np.clip(wtp_base, price_min, price_max))
                    rand_action[0, 0] = np.clip(
                        np.random.normal(wtp_anc, max(wtp_anc * 0.3, 15.0)),
                        price_min, price_max)

                    # [1] qty_multiplier: Gaussian around 1.0 (cover full need)
                    rand_action[0, 1] = np.clip(
                        np.random.normal(1.0, 0.15),
                        low[1].item(), high[1].item())

                    # [2] invest_frac: Gaussian around 0.03 (moderate investment)
                    rand_action[0, 2] = np.clip(
                        np.random.normal(0.03, 0.02),
                        low[2].item(), high[2].item())

                    # [3-5] tech logits: slight solar/onshore preference, moderate spread
                    # onshore=0.3, offshore=-0.5, solar=0.5 (reflects cost/speed reality)
                    rand_action[0, 3] = np.clip(np.random.normal(0.3, 0.5), -1.0, 1.0)
                    rand_action[0, 4] = np.clip(np.random.normal(-0.5, 0.5), -1.0, 1.0)
                    rand_action[0, 5] = np.clip(np.random.normal(0.5, 0.5), -1.0, 1.0)

                # Convert to raw (normalised) space
                rand_raw = torch.clamp(
                    (rand_action - self.auction_policy.action_bias) /
                    (self.auction_policy.action_scale + 1e-8), -1.0, 1.0)
                # Log-prob under current policy (for PPO importance ratio)
                dist = self.auction_policy.forward(obs_t)
                rand_lp = dist.log_prob(rand_raw).sum(dim=-1, keepdim=True)
            return (rand_action.cpu().numpy().squeeze(0),
                    rand_raw.cpu().numpy().squeeze(0),
                    rand_lp.cpu().numpy().squeeze(0))

        return (action.cpu().numpy().squeeze(0),
                raw.cpu().numpy().squeeze(0),
                log_prob.cpu().numpy().squeeze(0))

    def select_secondary_action(self, obs2: np.ndarray, deterministic=False,
                                epsilon: float = 0.0):
        """Phase 2: obs(21) → (action[2], raw[2], logp[1]).

        Epsilon-greedy in physical space (same approach as auction actions).
        """
        obs_t = torch.FloatTensor(obs2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, raw, log_prob = self.secondary_policy.act(obs_t, deterministic)

        if not deterministic and epsilon > 0.0 and np.random.random() < epsilon:
            with torch.no_grad():
                low = self.secondary_policy.action_bias - self.secondary_policy.action_scale
                high = self.secondary_policy.action_bias + self.secondary_policy.action_scale
                if self.exploration_mode == "uniform":
                    rand_action = low + (high - low) * torch.rand_like(action)
                else:
                    rand_action = torch.zeros_like(action)

                    # [0] sec_price (absolute EUR/t): sample around observed auction
                    # clearing price from obs2 (normalized by auction price_max).
                    sec_price_low = low[0].item()
                    sec_price_high = high[0].item()
                    clearing_ref = self.expected_price_fallback
                    if len(obs2) >= 10:
                        clearing_norm = float(obs2[OBS2_CLEARING_PRICE_IDX_FROM_END])
                        if np.isfinite(clearing_norm):
                            clearing_ref = clearing_norm * self.auction_price_max
                    if not np.isfinite(clearing_ref):
                        clearing_ref = self.expected_price_fallback
                    clearing_ref = float(np.clip(clearing_ref, sec_price_low, sec_price_high))
                    rand_action[0, 0] = np.clip(
                        np.random.normal(clearing_ref, max(10.0, 0.15 * clearing_ref)),
                        sec_price_low, sec_price_high)

                    # [1] sec_qty: Gaussian around 0 with moderate spread.
                    # Positive = buy, negative = sell; neutral center lets both be explored.
                    rand_action[0, 1] = np.clip(
                        np.random.normal(0.0, 1.0),
                        low[1].item(), high[1].item())

                rand_raw = torch.clamp(
                    (rand_action - self.secondary_policy.action_bias) /
                    (self.secondary_policy.action_scale + 1e-8), -1.0, 1.0)
                dist = self.secondary_policy.forward(obs_t)
                rand_lp = dist.log_prob(rand_raw).sum(dim=-1, keepdim=True)
            return (rand_action.cpu().numpy().squeeze(0),
                    rand_raw.cpu().numpy().squeeze(0),
                    rand_lp.cpu().numpy().squeeze(0))

        return (action.cpu().numpy().squeeze(0),
                raw.cpu().numpy().squeeze(0),
                log_prob.cpu().numpy().squeeze(0))

    def estimate_value(self, obs: np.ndarray) -> float:
        """V(s) — from local obs2 (IPPO) or global state (MAPPO)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.value_net(obs_t).cpu().item()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_transition(self, obs1, obs2, auc_raw, sec_raw,
                         auc_lp, sec_lp, reward, done, value,
                         global_state=None, phase='secondary'):
        self.buffer.push(obs1, obs2, auc_raw, sec_raw,
                         auc_lp, sec_lp, reward, done, value,
                         global_state=global_state, phase=phase)

    # ------------------------------------------------------------------
    # PPO Update (end of episode)
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0, actor_update: bool = True) -> Optional[dict]:
        """
        PPO update for one episode rollout.

        Parameters
        ----------
        last_value : float
            Bootstrap value for the last step (0 if terminal).
        actor_update : bool
            When False (critic-warmup phase) only the value network is trained;
            actor gradients are not computed or applied.
        """
        if len(self.buffer) < 2:
            return None

        obs1_np = np.nan_to_num(np.array(self.buffer.obs1), nan=0.0, posinf=1e6, neginf=-1e6)
        obs2_np = np.nan_to_num(np.array(self.buffer.obs2), nan=0.0, posinf=1e6, neginf=-1e6)

        obs1 = torch.FloatTensor(obs1_np).to(self.device)
        obs2 = torch.FloatTensor(obs2_np).to(self.device)

        # MAPPO: use global states for centralized critic if available
        if self.centralized_critic and len(self.buffer.global_states) > 0:
            critic_input = torch.FloatTensor(
                np.array(self.buffer.global_states)).to(self.device)
        else:
            critic_input = obs2

        auc_raw = torch.FloatTensor(np.array(self.buffer.auction_raw)).to(self.device)
        sec_raw = torch.FloatTensor(np.array(self.buffer.secondary_raw)).to(self.device)
        old_auc_lp = torch.FloatTensor(np.array(self.buffer.auction_logp)).to(self.device)
        old_sec_lp = torch.FloatTensor(np.array(self.buffer.secondary_logp)).to(self.device)

        # Phase mask: True = auction transition (obs1-space), False = secondary (obs2-space).
        # Auction policy is trained only on auction rows; secondary only on secondary rows.
        is_auction = torch.BoolTensor(
            [p == 'auction' for p in self.buffer.phases]).to(self.device)

        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)

        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        dones = np.nan_to_num(dones, nan=1.0, posinf=1.0, neginf=1.0)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        adv_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        ret_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)

        # Return normalization
        if self.normalize_returns and T > 1:
            ret_mean = ret_t.mean()
            ret_std = ret_t.std()
            if ret_std > 1e-8:
                ret_t = (ret_t - ret_mean) / (ret_std + 1e-8)

        if self.normalize_advantages and T > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        adv_t = torch.nan_to_num(adv_t, nan=0.0, posinf=0.0, neginf=0.0)
        ret_t = torch.nan_to_num(ret_t, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert old values to tensor for value clipping
        old_values_t = torch.FloatTensor(values).to(self.device).unsqueeze(1) if self.clip_value else None

        # Critic extra epochs: warm up critic before main PPO loop
        if self.critic_extra_epochs > 0:
            for _extra_epoch in range(self.critic_extra_epochs):
                idx = np.arange(T)
                self._rng.shuffle(idx)
                for start in range(0, T, self.mini_batch_size):
                    end = min(start + self.mini_batch_size, T)
                    mb = idx[start:end]
                    v_pred = self.value_net(critic_input[mb])
                    critic_loss = self._critic_loss(
                        v_pred, ret_t[mb],
                        old_values_t[mb] if self.clip_value else None
                    )
                    if not torch.isfinite(critic_loss):
                        continue
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.value_net.parameters()), self.max_grad_norm)
                    bad_crit = any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in self.value_net.parameters())
                    if not bad_crit:
                        self.critic_optimizer.step()

        # PPO epochs
        total_a_loss = 0.0
        total_v_loss = 0.0
        n_up = 0

        for _epoch in range(self.n_epochs):
            idx = np.arange(T)
            self._rng.shuffle(idx)

            epoch_kl_sum = 0.0
            epoch_kl_count = 0

            for start in range(0, T, self.mini_batch_size):
                end = min(start + self.mini_batch_size, T)
                mb = idx[start:end]

                v_pred = self.value_net(critic_input[mb])
                value_loss = self._critic_loss(
                    v_pred, ret_t[mb],
                    old_values_t[mb] if self.clip_value else None
                )

                if actor_update:
                    # Phase-split: auction policy only on auction rows, secondary only on secondary rows.
                    is_auc_mb = is_auction[mb]   # [mb_size] bool
                    sec_mb = ~is_auc_mb          # [mb_size] bool

                    auc_policy_loss = torch.tensor(0.0, device=self.device)
                    sec_policy_loss = torch.tensor(0.0, device=self.device)
                    auc_ent = torch.zeros(1, device=self.device)
                    sec_ent = torch.zeros(1, device=self.device)
                    auc_log_ratio = torch.zeros(1, 1, device=self.device)
                    sec_log_ratio = torch.zeros(1, 1, device=self.device)
                    auc_ratio = torch.ones(1, 1, device=self.device)
                    sec_ratio = torch.ones(1, 1, device=self.device)

                    # Auction policy: obs1-space, auction-phase rows only
                    if is_auc_mb.any():
                        auc_lp_new, auc_ent = self.auction_policy.evaluate(
                            obs1[mb][is_auc_mb], auc_raw[mb][is_auc_mb])
                        auc_log_ratio = torch.clamp(
                            auc_lp_new - old_auc_lp[mb][is_auc_mb], -2.0, 2.0)
                        auc_ratio = torch.exp(auc_log_ratio)
                        auc_adv = adv_t[mb][is_auc_mb]
                        auc_surr1 = auc_ratio * auc_adv
                        auc_surr2 = torch.clamp(
                            auc_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * auc_adv
                        auc_policy_loss = -torch.min(auc_surr1, auc_surr2).mean()

                    # Secondary policy: obs2-space, secondary-phase rows only
                    if sec_mb.any():
                        sec_lp_new, sec_ent = self.secondary_policy.evaluate(
                            obs2[mb][sec_mb], sec_raw[mb][sec_mb])
                        sec_log_ratio = torch.clamp(
                            sec_lp_new - old_sec_lp[mb][sec_mb], -2.0, 2.0)
                        sec_ratio = torch.exp(sec_log_ratio)
                        sec_adv = adv_t[mb][sec_mb]
                        sec_surr1 = sec_ratio * sec_adv
                        sec_surr2 = torch.clamp(
                            sec_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * sec_adv
                        sec_policy_loss = -torch.min(sec_surr1, sec_surr2).mean()

                    policy_loss = auc_policy_loss + sec_policy_loss

                    with torch.no_grad():
                        kl_auc = ((auc_ratio - 1.0) - auc_log_ratio).mean()
                        kl_sec = ((sec_ratio - 1.0) - sec_log_ratio).mean()
                        mb_kl = 0.5 * (kl_auc + kl_sec)
                        epoch_kl_sum += mb_kl.item() * len(mb)
                        epoch_kl_count += len(mb)

                    entropy = (auc_ent.mean() + sec_ent.mean()) * 0.5

                    # KL anchor penalty against frozen BC policy
                    kl_pen = torch.tensor(0.0, device=self.device)
                    if self._bc_auction_policy is not None and self.kl_beta > 0.0:
                        kl_auc_bc = torch.tensor(0.0, device=self.device)
                        kl_sec_bc = torch.tensor(0.0, device=self.device)
                        if is_auc_mb.any():
                            curr_auc_dist = self.auction_policy.forward(obs1[mb][is_auc_mb])
                            with torch.no_grad():
                                bc_auc_dist = self._bc_auction_policy.forward(obs1[mb][is_auc_mb])
                            kl_auc_bc = kl_divergence(curr_auc_dist, bc_auc_dist).mean()
                        if sec_mb.any():
                            curr_sec_dist = self.secondary_policy.forward(obs2[mb][sec_mb])
                            with torch.no_grad():
                                bc_sec_dist = self._bc_secondary_policy.forward(obs2[mb][sec_mb])
                            kl_sec_bc = kl_divergence(curr_sec_dist, bc_sec_dist).mean()
                        kl_pen = self.kl_beta * (kl_auc_bc + kl_sec_bc) * 0.5

                    actor_loss_total = (policy_loss
                                        - self.entropy_coef * entropy
                                        + kl_pen)
                    critic_loss_total = self.value_coef * value_loss
                else:
                    # Critic-warmup: train value network only
                    policy_loss = torch.tensor(0.0, device=self.device)
                    actor_loss_total = None
                    critic_loss_total = self.value_coef * value_loss

                # --- Critic step ---
                if not torch.isfinite(critic_loss_total):
                    continue
                self.critic_optimizer.zero_grad()
                critic_loss_total.backward()
                nn.utils.clip_grad_norm_(
                    list(self.value_net.parameters()), self.max_grad_norm)
                bad_crit = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.value_net.parameters())
                if bad_crit:
                    self.critic_optimizer.zero_grad()
                else:
                    self.critic_optimizer.step()

                # --- Actor step ---
                if actor_loss_total is not None:
                    if not torch.isfinite(actor_loss_total):
                        total_v_loss += value_loss.item()
                        n_up += 1
                        continue
                    self.actor_optimizer.zero_grad()
                    actor_loss_total.backward()
                    actor_params = (list(self.auction_policy.parameters()) +
                                    list(self.secondary_policy.parameters()))
                    nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                    bad_actor = any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in actor_params)
                    if bad_actor:
                        self.actor_optimizer.zero_grad()
                    else:
                        self.actor_optimizer.step()

                total_a_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                n_up += 1

            if actor_update and self.target_kl > 0 and epoch_kl_count > 0:
                avg_kl = epoch_kl_sum / epoch_kl_count
                if avg_kl > self.target_kl:
                    break

        self.buffer.clear()

        avg_a = total_a_loss / max(n_up, 1)
        avg_v = total_v_loss / max(n_up, 1)
        self.actor_loss_history.append(avg_a)
        self.critic_loss_history.append(avg_v)
        return {"actor_loss": avg_a, "critic_loss": avg_v}

    # ------------------------------------------------------------------
    # HAPPO: Sequential multi-agent update
    # ------------------------------------------------------------------

    def compute_gae(self, last_value: float = 0.0):
        """
        Extract GAE advantages and returns from the rollout buffer.
        Rewards are normalized per-phase (auction vs secondary) before GAE
        to account for their different reward distributions.

        Returns
        -------
        adv_t : Tensor [T, 1]
            Normalised GAE advantages.
        ret_t : Tensor [T, 1]
            GAE returns (advantages + values).
        buf_tensors : dict
            Pre-processed buffer tensors for reuse in update_happo / compute_post_update_ratio.
        """
        if len(self.buffer) < 2:
            return None, None, None

        obs1_np = np.nan_to_num(np.array(self.buffer.obs1), nan=0.0, posinf=1e6, neginf=-1e6)
        obs2_np = np.nan_to_num(np.array(self.buffer.obs2), nan=0.0, posinf=1e6, neginf=-1e6)

        obs1 = torch.FloatTensor(obs1_np).to(self.device)
        obs2 = torch.FloatTensor(obs2_np).to(self.device)

        if self.centralized_critic and len(self.buffer.global_states) > 0:
            critic_input = torch.FloatTensor(
                np.array(self.buffer.global_states)).to(self.device)
        else:
            critic_input = obs2

        auc_raw = torch.FloatTensor(np.array(self.buffer.auction_raw)).to(self.device)
        sec_raw = torch.FloatTensor(np.array(self.buffer.secondary_raw)).to(self.device)
        old_auc_lp = torch.FloatTensor(np.array(self.buffer.auction_logp)).to(self.device)
        old_sec_lp = torch.FloatTensor(np.array(self.buffer.secondary_logp)).to(self.device)

        # Phase mask: True = auction transition (obs1-space), False = secondary (obs2-space).
        # Exposed in buf_tensors so update_happo() can apply each policy loss to the
        # correct rows without cross-contaminating obs dimensions.
        is_auction_t = torch.BoolTensor(
            [p == 'auction' for p in self.buffer.phases]).to(self.device)

        rewards = np.nan_to_num(np.array(self.buffer.rewards, dtype=np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
        dones = np.nan_to_num(np.array(self.buffer.dones, dtype=np.float32),
                              nan=1.0, posinf=1.0, neginf=1.0)
        values = np.nan_to_num(np.array(self.buffer.values, dtype=np.float32),
                               nan=0.0, posinf=0.0, neginf=0.0)

        # Phase-aware reward normalization for GAE computation.
        # Auction and secondary rewards have different distributions; normalizing
        # each phase separately preserves learning signal in both heads.
        phase_is_auction = np.array([p == 'auction' for p in self.buffer.phases], dtype=bool)
        phase_is_secondary = ~phase_is_auction
        _EPS_STD = self.gae_min_std

        if phase_is_auction.any():
            auc_rewards = rewards[phase_is_auction]
            auc_mu = auc_rewards.mean()
            auc_std = max(auc_rewards.std(), _EPS_STD)
            rewards[phase_is_auction] = np.clip((auc_rewards - auc_mu) / auc_std, -10.0, 10.0)

        if phase_is_secondary.any():
            sec_rewards = rewards[phase_is_secondary]
            sec_mu = sec_rewards.mean()
            sec_std = max(sec_rewards.std(), _EPS_STD)
            rewards[phase_is_secondary] = np.clip((sec_rewards - sec_mu) / sec_std, -10.0, 10.0)

        # GAE
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        adv_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        ret_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)

        # Return normalization
        if self.normalize_returns and T > 1:
            ret_mean = ret_t.mean()
            ret_std = ret_t.std()
            if ret_std > 1e-8:
                ret_t = (ret_t - ret_mean) / (ret_std + 1e-8)

        if self.normalize_advantages and T > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        adv_t = torch.nan_to_num(adv_t, nan=0.0, posinf=0.0, neginf=0.0)
        ret_t = torch.nan_to_num(ret_t, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert old values to tensor for value clipping
        old_values_t = torch.FloatTensor(values).to(self.device).unsqueeze(1) if self.clip_value else None

        buf_tensors = {
            "obs1": obs1, "obs2": obs2, "critic_input": critic_input,
            "auc_raw": auc_raw, "sec_raw": sec_raw,
            "old_auc_lp": old_auc_lp, "old_sec_lp": old_sec_lp,
            "old_values": old_values_t,  # for value clipping in update_happo
            "T": T,
            "is_auction": is_auction_t,  # [T] bool: route policy losses to correct obs space
        }
        return adv_t, ret_t, buf_tensors

    def update_happo(self, adv_t, ret_t, buf_tensors,
                     advantage_weights=None, actor_update: bool = True) -> Optional[dict]:
        """
        HAPPO update: PPO with externally-weighted advantages.

        Parameters
        ----------
        adv_t : Tensor [T, 1]
            Pre-computed normalised advantages.
        ret_t : Tensor [T, 1]
            Pre-computed returns.
        buf_tensors : dict
            Buffer tensors from compute_gae().
        advantage_weights : Tensor [T, 1] or None
            Cumulative clipped importance ratio from prior agents (HAPPO M factor).
            When None, equivalent to standard PPO.
        actor_update : bool
            When False, only train the critic.
        """
        if buf_tensors is None:
            self.buffer.clear()
            return None

        obs1 = buf_tensors["obs1"]
        obs2 = buf_tensors["obs2"]
        critic_input = buf_tensors["critic_input"]
        auc_raw = buf_tensors["auc_raw"]
        sec_raw = buf_tensors["sec_raw"]
        old_auc_lp = buf_tensors["old_auc_lp"]
        old_sec_lp = buf_tensors["old_sec_lp"]
        old_values_t = buf_tensors.get("old_values", None)  # for value clipping
        is_auction = buf_tensors["is_auction"]  # [T] bool: auction rows use obs1-space
        T = buf_tensors["T"]

        # Apply HAPPO advantage weighting
        if advantage_weights is not None:
            weighted_adv = adv_t * advantage_weights.to(self.device)
            # HAPPO weighted-advantage re-normalization
            w_std = weighted_adv.std()
            if w_std > 1e-6:
                weighted_adv = (weighted_adv - weighted_adv.mean()) / (w_std + 1e-8)
        else:
            weighted_adv = adv_t

        total_a_loss = 0.0
        total_v_loss = 0.0
        n_up = 0

        for _epoch in range(self.n_epochs):
            idx = np.arange(T)
            self._rng.shuffle(idx)

            epoch_kl_sum = 0.0
            epoch_kl_count = 0

            for start in range(0, T, self.mini_batch_size):
                end = min(start + self.mini_batch_size, T)
                mb = idx[start:end]

                v_pred = self.value_net(critic_input[mb])
                value_loss = self._critic_loss(
                    v_pred, ret_t[mb],
                    old_values_t[mb] if self.clip_value else None
                )

                if actor_update:
                    # Phase-split: auction policy only on auction rows, secondary only on secondary rows.
                    is_auc_mb = is_auction[mb]   # [mb_size] bool
                    sec_mb = ~is_auc_mb          # [mb_size] bool

                    auc_policy_loss = torch.tensor(0.0, device=self.device)
                    sec_policy_loss = torch.tensor(0.0, device=self.device)
                    auc_ent = torch.zeros(1, device=self.device)
                    sec_ent = torch.zeros(1, device=self.device)
                    auc_log_ratio = torch.zeros(1, 1, device=self.device)
                    sec_log_ratio = torch.zeros(1, 1, device=self.device)
                    auc_ratio = torch.ones(1, 1, device=self.device)
                    sec_ratio = torch.ones(1, 1, device=self.device)

                    # Auction policy: obs1-space, auction-phase rows only
                    if is_auc_mb.any():
                        auc_lp_new, auc_ent = self.auction_policy.evaluate(
                            obs1[mb][is_auc_mb], auc_raw[mb][is_auc_mb])
                        auc_log_ratio = torch.clamp(
                            auc_lp_new - old_auc_lp[mb][is_auc_mb], -2.0, 2.0)
                        auc_ratio = torch.exp(auc_log_ratio)
                        auc_adv = weighted_adv[mb][is_auc_mb]
                        auc_surr1 = auc_ratio * auc_adv
                        auc_surr2 = torch.clamp(
                            auc_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * auc_adv
                        auc_policy_loss = -torch.min(auc_surr1, auc_surr2).mean()

                    # Secondary policy: obs2-space, secondary-phase rows only
                    if sec_mb.any():
                        sec_lp_new, sec_ent = self.secondary_policy.evaluate(
                            obs2[mb][sec_mb], sec_raw[mb][sec_mb])
                        sec_log_ratio = torch.clamp(
                            sec_lp_new - old_sec_lp[mb][sec_mb], -2.0, 2.0)
                        sec_ratio = torch.exp(sec_log_ratio)
                        sec_adv = weighted_adv[mb][sec_mb]
                        sec_surr1 = sec_ratio * sec_adv
                        sec_surr2 = torch.clamp(
                            sec_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * sec_adv
                        sec_policy_loss = -torch.min(sec_surr1, sec_surr2).mean()

                    policy_loss = auc_policy_loss + sec_policy_loss

                    with torch.no_grad():
                        kl_auc = ((auc_ratio - 1.0) - auc_log_ratio).mean()
                        kl_sec = ((sec_ratio - 1.0) - sec_log_ratio).mean()
                        mb_kl = 0.5 * (kl_auc + kl_sec)
                        epoch_kl_sum += mb_kl.item() * len(mb)
                        epoch_kl_count += len(mb)

                    entropy = (auc_ent.mean() + sec_ent.mean()) * 0.5

                    kl_pen = torch.tensor(0.0, device=self.device)
                    if self._bc_auction_policy is not None and self.kl_beta > 0.0:
                        kl_auc_bc = torch.tensor(0.0, device=self.device)
                        kl_sec_bc = torch.tensor(0.0, device=self.device)
                        if is_auc_mb.any():
                            curr_auc_dist = self.auction_policy.forward(obs1[mb][is_auc_mb])
                            with torch.no_grad():
                                bc_auc_dist = self._bc_auction_policy.forward(obs1[mb][is_auc_mb])
                            kl_auc_bc = kl_divergence(curr_auc_dist, bc_auc_dist).mean()
                        if sec_mb.any():
                            curr_sec_dist = self.secondary_policy.forward(obs2[mb][sec_mb])
                            with torch.no_grad():
                                bc_sec_dist = self._bc_secondary_policy.forward(obs2[mb][sec_mb])
                            kl_sec_bc = kl_divergence(curr_sec_dist, bc_sec_dist).mean()
                        kl_pen = self.kl_beta * (kl_auc_bc + kl_sec_bc) * 0.5

                    actor_loss_total = (policy_loss
                                        - self.entropy_coef * entropy
                                        + kl_pen)
                    critic_loss_total = self.value_coef * value_loss
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)
                    actor_loss_total = None
                    critic_loss_total = self.value_coef * value_loss

                # --- Critic step ---
                if not torch.isfinite(critic_loss_total):
                    continue
                self.critic_optimizer.zero_grad()
                critic_loss_total.backward()
                nn.utils.clip_grad_norm_(
                    list(self.value_net.parameters()), self.max_grad_norm)
                bad_crit = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.value_net.parameters())
                if bad_crit:
                    self.critic_optimizer.zero_grad()
                else:
                    self.critic_optimizer.step()

                # --- Actor step ---
                if actor_loss_total is not None:
                    if not torch.isfinite(actor_loss_total):
                        total_v_loss += value_loss.item()
                        n_up += 1
                        continue
                    self.actor_optimizer.zero_grad()
                    actor_loss_total.backward()
                    actor_params = (list(self.auction_policy.parameters()) +
                                    list(self.secondary_policy.parameters()))
                    nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                    bad_actor = any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in actor_params)
                    if bad_actor:
                        self.actor_optimizer.zero_grad()
                    else:
                        self.actor_optimizer.step()

                total_a_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                n_up += 1

            if actor_update and self.target_kl > 0 and epoch_kl_count > 0:
                avg_kl = epoch_kl_sum / epoch_kl_count
                if avg_kl > self.target_kl:
                    break

        self.buffer.clear()

        avg_a = total_a_loss / max(n_up, 1)
        avg_v = total_v_loss / max(n_up, 1)
        self.actor_loss_history.append(avg_a)
        self.critic_loss_history.append(avg_v)
        return {"actor_loss": avg_a, "critic_loss": avg_v}

    def compute_post_update_ratio(self, buf_tensors) -> torch.Tensor:
        """
        Compute the per-row importance ratio after a HAPPO update.

        Phase-masked: auction rows use the auction policy (obs1-space),
        secondary rows use the secondary policy (obs2-space). This prevents
        cross-obs-space contamination in the HAPPO cumulative M-factor chain.

        Returns
        -------
        ratio : Tensor [T, 1]
            Per-timestep importance ratio (clamped for stability).
        """
        obs1 = buf_tensors["obs1"]
        obs2 = buf_tensors["obs2"]
        auc_raw = buf_tensors["auc_raw"]
        sec_raw = buf_tensors["sec_raw"]
        old_auc_lp = buf_tensors["old_auc_lp"]
        old_sec_lp = buf_tensors["old_sec_lp"]
        is_auction = buf_tensors["is_auction"]   # [T] bool
        T = buf_tensors["T"]

        with torch.no_grad():
            joint_log_ratio = torch.zeros(T, 1, device=self.device)

            # Auction rows: ratio from auction policy only (obs1-space)
            if is_auction.any():
                new_auc_lp, _ = self.auction_policy.evaluate(
                    obs1[is_auction], auc_raw[is_auction])
                auc_lr = torch.clamp(
                    new_auc_lp - old_auc_lp[is_auction], -2.0, 2.0)
                joint_log_ratio[is_auction] = auc_lr

            # Secondary rows: ratio from secondary policy only (obs2-space)
            sec_mask = ~is_auction
            if sec_mask.any():
                new_sec_lp, _ = self.secondary_policy.evaluate(
                    obs2[sec_mask], sec_raw[sec_mask])
                sec_lr = torch.clamp(
                    new_sec_lp - old_sec_lp[sec_mask], -2.0, 2.0)
                joint_log_ratio[sec_mask] = sec_lr

            ratio = torch.exp(joint_log_ratio)

        return ratio

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            "auction_policy": self.auction_policy.state_dict(),
            "secondary_policy": self.secondary_policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.auction_policy.load_state_dict(ckpt["auction_policy"])
        self.secondary_policy.load_state_dict(ckpt["secondary_policy"])
        self.value_net.load_state_dict(ckpt["value_net"])
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
