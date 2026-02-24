"""
ddpg_agent.py
=============
DDPG agent for one energy company in the ETS environment.

Each agent maintains its own Actor, Critic, and target networks,
plus a separate replay buffer.

Reference:
    Di Persio, Garbelli, Giordano (2024) — Algorithm 1
    Lillicrap et al. (2015) — Continuous control with deep RL
    ckrk/bidding_learning — agent_ddpg.py (adapted)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from src.agents.actor_critic import Actor, Critic
from src.agents.noise import OUNoise
from src.utils.replay_buffer import ReplayBuffer


class DDPGAgent:
    """
    DDPG agent for a single energy company.

    Parameters
    ----------
    agent_id : int
    obs_dim : int
    action_dim : int
    action_low : np.ndarray
    action_high : np.ndarray
    config : dict
        Full YAML config dict.
    seed : int, optional
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: dict,
        seed: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        ddpg = config["ddpg"]
        self.gamma = ddpg["gamma"]
        self.tau = ddpg["tau"]
        self.batch_size = ddpg["batch_size"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Action bounds as tensors ---
        act_low  = torch.FloatTensor(action_low).to(self.device)
        act_high = torch.FloatTensor(action_high).to(self.device)

        # --- Networks ---
        hidden = ddpg["hidden_size"]
        self.actor  = Actor(obs_dim, action_dim, hidden, act_low, act_high).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden).to(self.device)

        # Target networks (initialised to same weights)
        self.actor_target  = Actor(obs_dim, action_dim, hidden, act_low, act_high).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, hidden).to(self.device)
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        # --- Optimisers ---
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=ddpg["lr_actor"])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=ddpg["lr_critic"],
                                       weight_decay=1e-4)  # L2 regularisation

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(
            capacity=ddpg["replay_buffer_size"],
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=seed,
        )

        # --- Exploration noise ---
        self.noise = OUNoise(
            action_dim=action_dim,
            theta=ddpg["ou_theta"],
            mu=ddpg["ou_mu"],
            sigma=ddpg["ou_sigma"],
            seed=seed,
        )

        self.action_low_np  = action_low
        self.action_high_np = action_high

        # Training metrics
        self.actor_loss_history  = []
        self.critic_loss_history = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action given observation.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)
        explore : bool
            If True, add OU noise for exploration.

        Returns
        -------
        action : np.ndarray, shape (action_dim,)
            In physical units (not normalised).
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().squeeze(0)
        self.actor.train()

        if explore:
            noise = self.noise.sample()
            action = action + noise

        action = np.clip(action, self.action_low_np, self.action_high_np)
        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> Optional[dict]:
        """
        Sample a mini-batch and update Actor and Critic.

        Returns
        -------
        dict with actor_loss and critic_loss, or None if buffer not ready.
        """
        if not self.buffer.ready:
            return None

        obs, actions, rewards, next_obs, dones = [
            torch.FloatTensor(x).to(self.device)
            for x in self.buffer.sample(self.batch_size)
        ]
        # Normalizza reward nel batch: media 0, std 1
        # Evita che gradienti grandi nelle prime fasi blocchino l'apprendimento
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)


        # ---- Update Critic ----
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q = rewards + self.gamma * (1 - dones) * self.critic_target(next_obs, next_actions)

        current_q = self.critic(obs, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # ---- Update Actor ----
        predicted_actions = self.actor(obs)
        actor_loss = -self.critic(obs, predicted_actions).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ---- Soft update target networks ----
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        a_loss = actor_loss.item()
        c_loss = critic_loss.item()
        self.actor_loss_history.append(a_loss)
        self.critic_loss_history.append(c_loss)

        return {"actor_loss": a_loss, "critic_loss": c_loss}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Polyak soft update: target = tau*source + (1-tau)*target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def reset_noise(self):
        self.noise.reset()

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
