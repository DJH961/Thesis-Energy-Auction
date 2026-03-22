"""
ppo_agent.py
============
PPO agent with two-phase decision making for EU ETS:
  Phase 1 (Auction):   obs(18) → [bid_price, qty, invest_frac, tech_logits×3]
  Phase 2 (Secondary): obs(21) → [sec_price_mult, sec_qty]

On-policy: collects full episode rollout, then updates via
clipped surrogate objective with GAE advantage estimation.

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
    Online running mean/std normaliser using exponential moving average.

    reward_norm = (reward - mu) / (std + eps)

    Parameters
    ----------
    alpha : float
        EMA decay rate. alpha=0.01 ≈ window of 100 samples.
    eps : float
        Numerical stability floor for std.
    """

    def __init__(self, alpha: float = 0.01, eps: float = 1e-8):
        self.alpha = alpha
        self.eps = eps
        self.mu = 0.0
        self.var = 1.0     # initialise to 1 so first normalised value ≈ raw reward

    def update_and_normalize(self, reward: float) -> float:
        """Update running stats and return normalised reward (NOT clipped)."""
        # EMA mean
        self.mu = (1.0 - self.alpha) * self.mu + self.alpha * reward
        # EMA variance (using current reward before updating mean for stability)
        self.var = (1.0 - self.alpha) * self.var + self.alpha * (reward - self.mu) ** 2
        std = max(self.var ** 0.5, self.eps)
        return (reward - self.mu) / std

    def reset(self):
        """Optionally reset stats (not called by default — stats persist across episodes)."""
        self.mu = 0.0
        self.var = 1.0


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
        self.auction_raw = []
        self.secondary_raw = []
        self.auction_logp = []
        self.secondary_logp = []
        self.rewards = []
        self.dones = []
        self.values = []

    def push(self, obs1, obs2, auc_raw, sec_raw, auc_lp, sec_lp, reward, done, value):
        self.obs1.append(obs1)
        self.obs2.append(obs2)
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
                 config, seed=None):
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

        self.auction_policy = AuctionPolicy(
            obs_dim_phase1, auction_dim, hidden, a_low, a_high,
            log_std_min=log_std_min, log_std_max=log_std_max,
        ).to(self.device)

        self.secondary_policy = SecondaryPolicy(
            obs_dim_phase2, secondary_dim, hidden, s_low, s_high,
            log_std_min=log_std_min, log_std_max=log_std_max,
        ).to(self.device)

        self.value_net = ValueNetwork(obs_dim_phase2, hidden).to(self.device)

        all_params = (
            list(self.auction_policy.parameters()) +
            list(self.secondary_policy.parameters()) +
            list(self.value_net.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=ppo["lr"])

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

    def select_auction_action(self, obs1: np.ndarray, deterministic=False):
        """Phase 1: obs(18) → (action[6], raw[6], logp[1])."""
        obs_t = torch.FloatTensor(obs1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, raw, log_prob = self.auction_policy.act(obs_t, deterministic)
        return (action.cpu().numpy().squeeze(0),
                raw.cpu().numpy().squeeze(0),
                log_prob.cpu().numpy().squeeze(0))

    def select_secondary_action(self, obs2: np.ndarray, deterministic=False):
        """Phase 2: obs(21) → (action[2], raw[2], logp[1])."""
        obs_t = torch.FloatTensor(obs2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, raw, log_prob = self.secondary_policy.act(obs_t, deterministic)
        return (action.cpu().numpy().squeeze(0),
                raw.cpu().numpy().squeeze(0),
                log_prob.cpu().numpy().squeeze(0))

    def estimate_value(self, obs2: np.ndarray) -> float:
        """V(s) from phase 2 observation."""
        obs_t = torch.FloatTensor(obs2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.value_net(obs_t).cpu().item()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_transition(self, obs1, obs2, auc_raw, sec_raw,
                         auc_lp, sec_lp, reward, done, value):
        self.buffer.push(obs1, obs2, auc_raw, sec_raw,
                         auc_lp, sec_lp, reward, done, value)

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

        obs1 = torch.FloatTensor(np.array(self.buffer.obs1)).to(self.device)
        obs2 = torch.FloatTensor(np.array(self.buffer.obs2)).to(self.device)
        auc_raw = torch.FloatTensor(np.array(self.buffer.auction_raw)).to(self.device)
        sec_raw = torch.FloatTensor(np.array(self.buffer.secondary_raw)).to(self.device)
        old_auc_lp = torch.FloatTensor(np.array(self.buffer.auction_logp)).to(self.device)
        old_sec_lp = torch.FloatTensor(np.array(self.buffer.secondary_logp)).to(self.device)
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)

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
        adv_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        ret_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)

        if self.normalize_advantages and T > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO epochs
        total_a_loss = 0.0
        total_v_loss = 0.0
        n_up = 0

        for _ in range(self.n_epochs):
            idx = np.arange(T)
            np.random.shuffle(idx)

            for start in range(0, T, self.mini_batch_size):
                end = min(start + self.mini_batch_size, T)
                mb = idx[start:end]

                v_pred = self.value_net(obs2[mb])
                value_loss = nn.MSELoss()(v_pred, ret_t[mb])

                if actor_update:
                    # Re-evaluate current policy
                    auc_lp_new, auc_ent = self.auction_policy.evaluate(obs1[mb], auc_raw[mb])
                    sec_lp_new, sec_ent = self.secondary_policy.evaluate(obs2[mb], sec_raw[mb])

                    new_lp = auc_lp_new + sec_lp_new
                    old_lp = old_auc_lp[mb] + old_sec_lp[mb]

                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()

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

                    loss = (policy_loss
                            + self.value_coef * value_loss
                            - self.entropy_coef * entropy
                            + kl_pen)
                else:
                    # Critic-warmup: train value network only
                    policy_loss = torch.tensor(0.0, device=self.device)
                    loss = self.value_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.auction_policy.parameters()) +
                    list(self.secondary_policy.parameters()) +
                    list(self.value_net.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_a_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                n_up += 1

        self.buffer.clear()

        avg_a = total_a_loss / max(n_up, 1)
        avg_v = total_v_loss / max(n_up, 1)
        self.actor_loss_history.append(avg_a)
        self.critic_loss_history.append(avg_v)
        return {"actor_loss": avg_a, "critic_loss": avg_v}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            "auction_policy": self.auction_policy.state_dict(),
            "secondary_policy": self.secondary_policy.state_dict(),
            "value_net": self.value_net.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.auction_policy.load_state_dict(ckpt["auction_policy"])
        self.secondary_policy.load_state_dict(ckpt["secondary_policy"])
        self.value_net.load_state_dict(ckpt["value_net"])
