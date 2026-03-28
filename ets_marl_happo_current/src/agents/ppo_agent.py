"""
ppo_agent.py
============
PPO agent with two-phase decision making for EU ETS:
  Phase 1 (Auction):   obs(18) → [bid_price, qty, invest_frac, tech_logits×3]
  Phase 2 (Secondary): obs(21) → [sec_price_mult, sec_qty]

On-policy: collects full episode rollout, then updates via
clipped surrogate objective with GAE advantage estimation.

Supports **MAPPO** (Multi-Agent PPO) via centralized critic:
  When `ppo.centralized_critic = true`, the value network V(s) receives the
  global state (concatenation of all agents' phase-2 observations) instead of
  the local observation.  Actors remain decentralized (CTDE paradigm).

Roadmap improvements:
  P2: entropy_coef is updated externally via set_entropy_coef() (decay schedule
      lives in train.py so the agent stays stateless w.r.t. episode count).
  P3: RewardNormalizer tracks per-agent running mean/std with EMA.
      normalize_reward() normalises and clips before storing in buffer.
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
# P3: Per-agent running reward normaliser
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

        # EMA mean
        self.mu = (1.0 - effective_alpha) * self.mu + effective_alpha * reward
        # EMA variance
        self.var = (1.0 - effective_alpha) * self.var + effective_alpha * (reward - self.mu) ** 2
        std = max(self.var ** 0.5, self.eps)
        return (reward - self.mu) / std

    def reset(self):
        """Optionally reset stats (not called by default — stats persist across episodes)."""
        self.mu = 0.0
        self.var = 1.0
        self._n_samples = 0


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

    def push(self, obs1, obs2, auc_raw, sec_raw, auc_lp, sec_lp, reward, done, value,
             global_state=None):
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
        ppo = config["ppo"]

        self.gamma = ppo["gamma"]
        self.gae_lambda = ppo["gae_lambda"]
        self.clip_eps = ppo["clip_eps"]
        self.entropy_coef = ppo["entropy_coef"]  # P2: updated per-episode by train.py
        self.value_coef = ppo["value_coef"]
        self.max_grad_norm = ppo["max_grad_norm"]
        self.n_epochs = ppo["n_epochs"]
        self.mini_batch_size = ppo["mini_batch_size"]
        self.normalize_advantages = ppo.get("normalize_advantages", True)
        # P11: KL-based early stopping — abort PPO epochs if policy drifts too far
        self.target_kl = ppo.get("target_kl", 0.0)  # 0 = disabled

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

        # Action anchors: realistic initial targets in physical space.
        # These shift the policy's initial mean output toward historically
        # plausible actions, giving agents a sensible starting point without
        # limiting what they can learn.  None = use midpoint (legacy behavior).
        explore_cfg = config.get("exploration", {})
        auction_anchors = explore_cfg.get("auction_anchors", None)
        secondary_anchors = explore_cfg.get("secondary_anchors", None)

        self.auction_policy = AuctionPolicy(
            obs_dim_phase1, auction_dim, hidden, a_low, a_high,
            log_std_min=log_std_min, log_std_max=log_std_max,
            action_anchors=auction_anchors,
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

        # Critic improvements: Huber loss, value clipping, return normalisation
        self.critic_huber = ppo.get("critic_huber", True)
        self.critic_huber_delta = ppo.get("critic_huber_delta", 10.0)
        self.clip_value = ppo.get("clip_value", True)
        self.clip_value_eps = ppo.get("clip_value_eps", self.clip_eps)
        self.normalize_returns = ppo.get("normalize_returns", True)
        self._huber_loss = nn.SmoothL1Loss(beta=self.critic_huber_delta) if self.critic_huber else None

        self.buffer = RolloutBuffer()
        self.actor_loss_history = []
        self.critic_loss_history = []

        # P3: per-agent reward normaliser
        reward_cfg = config.get("reward", {})
        norm_alpha = reward_cfg.get("normalizer_alpha", 0.01)
        self._reward_normalizer = RewardNormalizer(alpha=norm_alpha)
        self._reward_clip_min = reward_cfg.get("clip_min", -10.0)
        self._reward_clip_max = reward_cfg.get("clip_max", 2.0)

        # KL anchor: frozen snapshot of BC-trained policy (set after BC pretraining)
        self._bc_auction_policy = None
        self._bc_secondary_policy = None
        self.kl_beta = 0.0

    # ------------------------------------------------------------------
    # P2: Entropy coefficient update (called by train.py)
    # ------------------------------------------------------------------

    def set_entropy_coef(self, coef: float):
        """Update entropy coefficient for this training step (P2 decay schedule)."""
        self.entropy_coef = float(coef)

    def set_kl_beta(self, beta: float):
        """Update KL anchor penalty weight (decayed by train.py)."""
        self.kl_beta = float(beta)

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

    # ------------------------------------------------------------------
    # P3: Reward normalisation
    # ------------------------------------------------------------------

    def normalize_reward(self, reward: float) -> float:
        """
        Normalise reward with per-agent running stats, then clip.
        Called by the training loop before storing transitions.
        """
        r_norm = self._reward_normalizer.update_and_normalize(reward)
        return float(np.clip(r_norm, self._reward_clip_min, self._reward_clip_max))

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_auction_action(self, obs1: np.ndarray, deterministic=False,
                              epsilon: float = 0.0):
        """Phase 1: obs(18) → (action[6], raw[6], logp[1]).

        When ``epsilon > 0`` and not deterministic, with probability *epsilon*
        a uniform random action in physical space replaces the policy sample.
        The raw action and log_prob are still computed under the current policy
        so that the PPO importance ratio remains correct.
        """
        obs_t = torch.FloatTensor(obs1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, raw, log_prob = self.auction_policy.act(obs_t, deterministic)

        if not deterministic and epsilon > 0.0 and np.random.random() < epsilon:
            with torch.no_grad():
                low = self.auction_policy.action_bias - self.auction_policy.action_scale
                high = self.auction_policy.action_bias + self.auction_policy.action_scale
                # Anchored exploration: sample each dim from Gaussian around
                # realistic company expectations instead of uniform.
                # This gives exploration a sensible starting distribution that
                # reflects how real companies would initially behave.
                rand_action = torch.zeros_like(action)
                price_max = high[0].item()
                price_min = low[0].item()

                # [0] bid_price: Gaussian around expected_price (obs[3] × price_max)
                expected_price = float(obs1[3]) * price_max
                rand_action[0, 0] = np.clip(
                    np.random.normal(expected_price, max(expected_price * 0.3, 15.0)),
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
                rand_action[0, 3] = np.clip(np.random.normal( 0.3, 0.5), -1.0, 1.0)
                rand_action[0, 4] = np.clip(np.random.normal(-0.5, 0.5), -1.0, 1.0)
                rand_action[0, 5] = np.clip(np.random.normal( 0.5, 0.5), -1.0, 1.0)

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
                rand_action = torch.zeros_like(action)

                # [0] sec_price_mult: Gaussian around 1.05 (trade near clearing price)
                rand_action[0, 0] = np.clip(
                    np.random.normal(1.05, 0.10),
                    low[0].item(), high[0].item())

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
                         global_state=None):
        self.buffer.push(obs1, obs2, auc_raw, sec_raw,
                         auc_lp, sec_lp, reward, done, value,
                         global_state=global_state)

    # ------------------------------------------------------------------
    # Critic loss helper (Huber + value clipping)
    # ------------------------------------------------------------------

    def _critic_loss(self, v_pred, ret_target, old_values=None):
        """
        Compute critic loss with optional Huber loss and value clipping.

        Value clipping (Schulman 2017, Engstrom et al. 2020): constrains
        the value function update to a trust region around old predictions,
        preventing catastrophic over-correction on outlier returns.

        Huber loss: reduces sensitivity to return outliers from penalty
        spikes, terminal values, and stochastic shocks.
        """
        if self.clip_value and old_values is not None:
            # Clipped value prediction: restrict update to ±clip_value_eps
            v_clipped = old_values + torch.clamp(
                v_pred - old_values,
                -self.clip_value_eps, self.clip_value_eps
            )
            if self._huber_loss is not None:
                loss_unclipped = self._huber_loss(v_pred, ret_target)
                loss_clipped = self._huber_loss(v_clipped, ret_target)
            else:
                loss_unclipped = (v_pred - ret_target).pow(2).mean()
                loss_clipped = (v_clipped - ret_target).pow(2).mean()
            return torch.max(loss_unclipped, loss_clipped)
        else:
            if self._huber_loss is not None:
                return self._huber_loss(v_pred, ret_target)
            return nn.MSELoss()(v_pred, ret_target)

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

        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)

        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        dones = np.nan_to_num(dones, nan=1.0, posinf=1.0, neginf=1.0)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # GAE (P2: gae_lambda = 0.97 in config for longer credit assignment)
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

        if self.normalize_advantages and T > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Return normalization: stabilises critic regression targets
        if self.normalize_returns and T > 1:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        adv_t = torch.nan_to_num(adv_t, nan=0.0, posinf=0.0, neginf=0.0)
        ret_t = torch.nan_to_num(ret_t, nan=0.0, posinf=0.0, neginf=0.0)

        # Old value predictions for value clipping
        old_values_t = torch.FloatTensor(values).to(self.device).unsqueeze(1)
        if self.normalize_returns and T > 1:
            # Normalise old values with same stats so clipping is consistent
            ret_mean = torch.FloatTensor(returns).to(self.device).mean()
            ret_std = torch.FloatTensor(returns).to(self.device).std() + 1e-8
            old_values_t = (old_values_t - ret_mean) / ret_std

        # PPO epochs
        total_a_loss = 0.0
        total_v_loss = 0.0
        n_up = 0

        for _epoch in range(self.n_epochs):
            idx = np.arange(T)
            np.random.shuffle(idx)

            # P11: KL tracking for early stopping
            epoch_kl_sum = 0.0
            epoch_kl_count = 0

            for start in range(0, T, self.mini_batch_size):
                end = min(start + self.mini_batch_size, T)
                mb = idx[start:end]

                v_pred = self.value_net(critic_input[mb])
                value_loss = self._critic_loss(v_pred, ret_t[mb], old_values_t[mb])

                if actor_update:
                    # Re-evaluate current policy
                    auc_lp_new, auc_ent = self.auction_policy.evaluate(obs1[mb], auc_raw[mb])
                    sec_lp_new, sec_ent = self.secondary_policy.evaluate(obs2[mb], sec_raw[mb])

                    # Per-policy PPO clipping: decomposes the joint ratio so each
                    # policy's gradient update is independently clipped.  Prevents
                    # a profitable secondary trade from incorrectly reinforcing
                    # bad auction bids (and vice versa).
                    # P11: Tighter log-ratio clamp — max ratio e^2≈7.4 (was e^10≈22026).
                    # Prevents catastrophic loss from rare high-ratio mini-batches.
                    auc_log_ratio = torch.clamp(auc_lp_new - old_auc_lp[mb], -2.0, 2.0)
                    auc_ratio = torch.exp(auc_log_ratio)
                    auc_surr1 = auc_ratio * adv_t[mb]
                    auc_surr2 = torch.clamp(auc_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                    auc_policy_loss = -torch.min(auc_surr1, auc_surr2).mean()

                    sec_log_ratio = torch.clamp(sec_lp_new - old_sec_lp[mb], -2.0, 2.0)
                    sec_ratio = torch.exp(sec_log_ratio)
                    sec_surr1 = sec_ratio * adv_t[mb]
                    sec_surr2 = torch.clamp(sec_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                    sec_policy_loss = -torch.min(sec_surr1, sec_surr2).mean()

                    policy_loss = auc_policy_loss + sec_policy_loss

                    # P11: Approximate KL divergence for early stopping
                    # Schulman (2020): approx_kl ≈ (ratio - 1) - log(ratio)
                    with torch.no_grad():
                        mb_kl = 0.5 * (
                            ((auc_ratio - 1.0) - auc_log_ratio).mean()
                            + ((sec_ratio - 1.0) - sec_log_ratio).mean()
                        )
                        epoch_kl_sum += mb_kl.item() * len(mb)
                        epoch_kl_count += len(mb)

                    # P2: entropy_coef updated externally via set_entropy_coef()
                    entropy = (auc_ent + sec_ent).mean()

                    # KL anchor penalty against frozen BC policy
                    kl_pen = torch.tensor(0.0, device=self.device)
                    if self._bc_auction_policy is not None and self.kl_beta > 0.0:
                        curr_auc_dist = self.auction_policy.forward(obs1[mb])
                        curr_sec_dist = self.secondary_policy.forward(obs2[mb])
                        with torch.no_grad():
                            bc_auc_dist = self._bc_auction_policy.forward(obs1[mb])
                            bc_sec_dist = self._bc_secondary_policy.forward(obs2[mb])
                        kl_auc = kl_divergence(curr_auc_dist, bc_auc_dist).mean()
                        kl_sec = kl_divergence(curr_sec_dist, bc_sec_dist).mean()
                        kl_pen = self.kl_beta * (kl_auc + kl_sec) * 0.5

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
                critic_loss_total.backward(retain_graph=(actor_loss_total is not None))
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

            # P11: KL early stopping — abort remaining epochs if policy
            # has already drifted significantly from data-collection policy.
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

        rewards = np.nan_to_num(np.array(self.buffer.rewards, dtype=np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
        dones = np.nan_to_num(np.array(self.buffer.dones, dtype=np.float32),
                              nan=1.0, posinf=1.0, neginf=1.0)
        values = np.nan_to_num(np.array(self.buffer.values, dtype=np.float32),
                               nan=0.0, posinf=0.0, neginf=0.0)

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

        if self.normalize_advantages and T > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Return normalization and old values for value clipping
        old_values_t = torch.FloatTensor(values).to(self.device).unsqueeze(1)
        if self.normalize_returns and T > 1:
            ret_mean = ret_t.mean()
            ret_std = ret_t.std() + 1e-8
            ret_t = (ret_t - ret_mean) / ret_std
            old_values_t = (old_values_t - ret_mean) / ret_std

        adv_t = torch.nan_to_num(adv_t, nan=0.0, posinf=0.0, neginf=0.0)
        ret_t = torch.nan_to_num(ret_t, nan=0.0, posinf=0.0, neginf=0.0)

        buf_tensors = {
            "obs1": obs1, "obs2": obs2, "critic_input": critic_input,
            "auc_raw": auc_raw, "sec_raw": sec_raw,
            "old_auc_lp": old_auc_lp, "old_sec_lp": old_sec_lp,
            "old_values": old_values_t,
            "T": T,
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
        old_values_t = buf_tensors.get("old_values", None)
        T = buf_tensors["T"]

        # Apply HAPPO advantage weighting
        if advantage_weights is not None:
            weighted_adv = adv_t * advantage_weights.to(self.device)
        else:
            weighted_adv = adv_t

        total_a_loss = 0.0
        total_v_loss = 0.0
        n_up = 0

        for _epoch in range(self.n_epochs):
            idx = np.arange(T)
            np.random.shuffle(idx)

            epoch_kl_sum = 0.0
            epoch_kl_count = 0

            for start in range(0, T, self.mini_batch_size):
                end = min(start + self.mini_batch_size, T)
                mb = idx[start:end]

                v_pred = self.value_net(critic_input[mb])
                old_v_mb = old_values_t[mb] if old_values_t is not None else None
                value_loss = self._critic_loss(v_pred, ret_t[mb], old_v_mb)

                if actor_update:
                    auc_lp_new, auc_ent = self.auction_policy.evaluate(obs1[mb], auc_raw[mb])
                    sec_lp_new, sec_ent = self.secondary_policy.evaluate(obs2[mb], sec_raw[mb])

                    auc_log_ratio = torch.clamp(auc_lp_new - old_auc_lp[mb], -2.0, 2.0)
                    auc_ratio = torch.exp(auc_log_ratio)
                    auc_surr1 = auc_ratio * weighted_adv[mb]
                    auc_surr2 = torch.clamp(auc_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * weighted_adv[mb]
                    auc_policy_loss = -torch.min(auc_surr1, auc_surr2).mean()

                    sec_log_ratio = torch.clamp(sec_lp_new - old_sec_lp[mb], -2.0, 2.0)
                    sec_ratio = torch.exp(sec_log_ratio)
                    sec_surr1 = sec_ratio * weighted_adv[mb]
                    sec_surr2 = torch.clamp(sec_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * weighted_adv[mb]
                    sec_policy_loss = -torch.min(sec_surr1, sec_surr2).mean()

                    policy_loss = auc_policy_loss + sec_policy_loss

                    with torch.no_grad():
                        mb_kl = 0.5 * (
                            ((auc_ratio - 1.0) - auc_log_ratio).mean()
                            + ((sec_ratio - 1.0) - sec_log_ratio).mean()
                        )
                        epoch_kl_sum += mb_kl.item() * len(mb)
                        epoch_kl_count += len(mb)

                    entropy = (auc_ent + sec_ent).mean()

                    kl_pen = torch.tensor(0.0, device=self.device)
                    if self._bc_auction_policy is not None and self.kl_beta > 0.0:
                        curr_auc_dist = self.auction_policy.forward(obs1[mb])
                        curr_sec_dist = self.secondary_policy.forward(obs2[mb])
                        with torch.no_grad():
                            bc_auc_dist = self._bc_auction_policy.forward(obs1[mb])
                            bc_sec_dist = self._bc_secondary_policy.forward(obs2[mb])
                        kl_auc = kl_divergence(curr_auc_dist, bc_auc_dist).mean()
                        kl_sec = kl_divergence(curr_sec_dist, bc_sec_dist).mean()
                        kl_pen = self.kl_beta * (kl_auc + kl_sec) * 0.5

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
                critic_loss_total.backward(retain_graph=(actor_loss_total is not None))
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
        Compute the joint importance ratio after a HAPPO update.

        ratio_i = exp((new_auc_lp - old_auc_lp) + (new_sec_lp - old_sec_lp))

        Returns
        -------
        ratio : Tensor [T, 1]
            Per-timestep joint importance ratio (clamped for stability).
        """
        obs1 = buf_tensors["obs1"]
        obs2 = buf_tensors["obs2"]
        auc_raw = buf_tensors["auc_raw"]
        sec_raw = buf_tensors["sec_raw"]
        old_auc_lp = buf_tensors["old_auc_lp"]
        old_sec_lp = buf_tensors["old_sec_lp"]

        with torch.no_grad():
            new_auc_lp, _ = self.auction_policy.evaluate(obs1, auc_raw)
            new_sec_lp, _ = self.secondary_policy.evaluate(obs2, sec_raw)

            # Joint log-ratio (clamped for numerical stability)
            joint_log_ratio = torch.clamp(
                (new_auc_lp - old_auc_lp) + (new_sec_lp - old_sec_lp),
                -2.0, 2.0
            )
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
        ckpt = torch.load(path, map_location=self.device)
        self.auction_policy.load_state_dict(ckpt["auction_policy"])
        self.secondary_policy.load_state_dict(ckpt["secondary_policy"])
        self.value_net.load_state_dict(ckpt["value_net"])
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
