"""
lstm_bot_policy.py
==================
LSTM-based co-evolving bot policy for the ETS MARL environment.

Replaces the static heuristic bots with adaptive opponents that periodically
retrain on recent episode data.  Each bot uses a unidirectional LSTM that
carries hidden state across years within an episode, enabling temporal
strategy adaptation (e.g. adjusting bids based on price trends, banking
positions, and competitive dynamics).

Architecture
------------
Two separate LSTM networks (mirroring the two-phase agent design):
  - **LSTMAuctionNet** : obs_dim_phase1 → 6  (bid_price, qty_mult, invest_frac, tech_logits)
  - **LSTMSecondaryNet**: obs_dim_phase2 → 2  (sec_price_mult, sec_qty)

Co-evolution modes (``target_mode`` in config)
----------------------------------------------
  - ``heuristic``:     Always imitate the heuristic policy (stable, non-adaptive).
  - ``imitate_best``:  Imitate actions of the top-K performing agents in recent
                       episodes, matched by archetype parity (recommended).

Stability safeguards
--------------------
  - Warmup period: LSTM bots fall back to heuristic during early training.
  - Soft blend:  α ramps linearly from 0→1 over a configurable window, blending
                 LSTM output with heuristic actions for a smooth transition.
  - Action clipping: all outputs clamped to physical bounds.
  - Gradient clipping: max_norm=1.0 during retraining.
  - NaN fallback: if any LSTM output is NaN, heuristic is used for that bot.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple


# ======================================================================
# Network Architectures
# ======================================================================

class LSTMAuctionNet(nn.Module):
    """LSTM policy for Phase-1 auction actions (6D output)."""

    def __init__(self, obs_dim: int, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 6),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        obs: torch.Tensor,          # (batch, seq_len, obs_dim) or (batch, obs_dim)
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)   # (batch, 1, obs_dim)
        lstm_out, hidden = self.lstm(obs, hidden)
        # Use last time-step output
        action = self.head(lstm_out[:, -1, :])  # (batch, 6)
        return action, hidden


class LSTMSecondaryNet(nn.Module):
    """LSTM policy for Phase-2 secondary market actions (2D output)."""

    def __init__(self, obs_dim: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        lstm_out, hidden = self.lstm(obs, hidden)
        action = self.head(lstm_out[:, -1, :])
        return action, hidden


# ======================================================================
# Main Policy Class
# ======================================================================

class LSTMBotPolicy:
    """Co-evolving LSTM bot policy manager.

    Parameters
    ----------
    obs_dim_phase1 : int
        Phase-1 observation dimension.
    obs_dim_phase2 : int
        Phase-2 observation dimension.
    n_bots : int
        Number of bot agents.
    config : dict
        Full training configuration (reads ``lstm_bots`` section).
    device : str
        Torch device.
    """

    def __init__(
        self,
        obs_dim_phase1: int,
        obs_dim_phase2: int,
        n_bots: int,
        config: dict,
        device: str = "cpu",
    ):
        self.n_bots = n_bots
        self.device = torch.device(device)
        self.config = config

        lcd = config.get("lstm_bots", {})
        hidden_auc = lcd.get("hidden_size_auction", 128)
        hidden_sec = lcd.get("hidden_size_secondary", 64)
        num_layers = lcd.get("num_layers", 1)
        lr = lcd.get("lr", 3e-4)
        self.noise_std = lcd.get("noise_std", 0.02)
        self.target_mode = lcd.get("target_mode", "imitate_best")
        self.top_k_fraction = lcd.get("top_k_fraction", 0.25)

        # Action bounds (physical space)
        aq = config["auction"]
        tr = config.get("trading", {})
        self.auc_low = np.array([
            aq["price_min"], aq.get("qty_mult_low", 0.3),
            0.0, -2.0, -2.0, -2.0,
        ], dtype=np.float32)
        self.auc_high = np.array([
            aq["price_max"], aq.get("qty_mult_high", 2.0),
            config.get("investment", {}).get("max_invest_frac", 0.20),
            2.0, 2.0, 2.0,
        ], dtype=np.float32)
        self.sec_low = np.array([
            tr.get("sec_mult_low", 0.8),
            -aq.get("quantity_max", 3.0),
        ], dtype=np.float32)
        self.sec_high = np.array([
            tr.get("sec_mult_high", 1.4),
            aq.get("quantity_max", 3.0),
        ], dtype=np.float32)

        # Networks
        self.auction_net = LSTMAuctionNet(
            obs_dim_phase1, hidden_auc, num_layers,
        ).to(self.device)
        self.secondary_net = LSTMSecondaryNet(
            obs_dim_phase2, hidden_sec, num_layers,
        ).to(self.device)

        # Optimizers
        self.auc_optimizer = optim.Adam(self.auction_net.parameters(), lr=lr)
        self.sec_optimizer = optim.Adam(self.secondary_net.parameters(), lr=lr)

        # LSTM hidden states per bot (reset each episode)
        self._h_auc = None   # (num_layers, n_bots, hidden_auc)
        self._c_auc = None
        self._h_sec = None
        self._c_sec = None

        # Readiness flag (set after first training pass)
        self._is_trained = False

        # Training stats
        self.last_auc_loss = 0.0
        self.last_sec_loss = 0.0
        self.retrain_count = 0

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset_hidden(self):
        """Zero LSTM hidden states at the start of each episode."""
        nl = self.auction_net.num_layers
        self._h_auc = torch.zeros(nl, self.n_bots, self.auction_net.hidden_size,
                                  device=self.device)
        self._c_auc = torch.zeros_like(self._h_auc)

        nl_s = self.secondary_net.num_layers
        self._h_sec = torch.zeros(nl_s, self.n_bots, self.secondary_net.hidden_size,
                                  device=self.device)
        self._c_sec = torch.zeros_like(self._h_sec)

    def is_ready(self) -> bool:
        """True once the LSTM has been trained at least once."""
        return self._is_trained

    # ------------------------------------------------------------------
    # Action generation (called by environment)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def auction_action(self, obs_batch: np.ndarray) -> np.ndarray:
        """Generate Phase-1 actions for all bots.

        Parameters
        ----------
        obs_batch : (n_bots, obs_dim_phase1)

        Returns
        -------
        actions : (n_bots, 6) in physical space, clipped to bounds.
        """
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        raw_action, (self._h_auc, self._c_auc) = self.auction_net(
            obs_t, (self._h_auc, self._c_auc),
        )
        actions = raw_action.cpu().numpy()

        # Add small noise for diversity
        if self.noise_std > 0:
            noise = np.random.randn(*actions.shape).astype(np.float32) * self.noise_std
            # Scale noise per dimension
            action_range = self.auc_high - self.auc_low
            actions += noise * action_range

        actions = np.clip(actions, self.auc_low, self.auc_high)
        return actions

    @torch.no_grad()
    def secondary_action(self, obs_batch: np.ndarray) -> np.ndarray:
        """Generate Phase-2 actions for all bots.

        Parameters
        ----------
        obs_batch : (n_bots, obs_dim_phase2)

        Returns
        -------
        actions : (n_bots, 2) in physical space, clipped to bounds.
        """
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        raw_action, (self._h_sec, self._c_sec) = self.secondary_net(
            obs_t, (self._h_sec, self._c_sec),
        )
        actions = raw_action.cpu().numpy()

        if self.noise_std > 0:
            noise = np.random.randn(*actions.shape).astype(np.float32) * self.noise_std
            action_range = self.sec_high - self.sec_low
            actions += noise * action_range

        actions = np.clip(actions, self.sec_low, self.sec_high)
        return actions

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------
    def train_on_buffer(
        self,
        buffer,   # BotDataBuffer
        epochs: int = 3,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Retrain LSTM networks on data from the buffer.

        Parameters
        ----------
        buffer : BotDataBuffer
        epochs : int
            Number of passes over sampled data.
        batch_size : int
            Number of episodes per training batch.
        verbose : bool
            Print loss information.

        Returns
        -------
        dict with ``auc_loss`` and ``sec_loss``.
        """
        if len(buffer) < batch_size:
            return {"auc_loss": 0.0, "sec_loss": 0.0}

        n_bots = self.n_bots
        total_auc_loss = 0.0
        total_sec_loss = 0.0
        n_updates = 0

        self.auction_net.train()
        self.secondary_net.train()

        for epoch in range(epochs):
            episodes = buffer.sample_episodes(batch_size)

            for ep_data in episodes:
                n_years = ep_data["obs1"].shape[0]
                n_total = ep_data["obs1"].shape[1]
                n_learning = n_total - n_bots

                # Get target actions based on mode
                if self.target_mode == "imitate_best":
                    auc_targets, sec_targets = buffer.get_top_k_targets(
                        ep_data, n_bots, self.top_k_fraction,
                    )
                else:  # "heuristic" — use the bot's own recorded actions
                    auc_targets = ep_data["auc"][:, n_learning:, :]
                    sec_targets = ep_data["sec"][:, n_learning:, :]

                # Get bot observations
                obs1_bots = ep_data["obs1"][:, n_learning:, :]  # (T, n_bots, D1)
                obs2_bots = ep_data["obs2"][:, n_learning:, :]  # (T, n_bots, D2)

                # --- Auction LSTM training ---
                obs1_t = torch.as_tensor(
                    obs1_bots, dtype=torch.float32, device=self.device,
                )  # (T, n_bots, D1)
                tgt_auc = torch.as_tensor(
                    auc_targets, dtype=torch.float32, device=self.device,
                )

                # Forward through full sequence
                # Reshape: treat each bot as a batch element
                # obs1_t: (T, n_bots, D1) -> (n_bots, T, D1)
                obs1_seq = obs1_t.permute(1, 0, 2)
                tgt_auc_seq = tgt_auc.permute(1, 0, 2)  # (n_bots, T, 6)

                auc_pred, _ = self.auction_net(obs1_seq, None)
                # auc_pred is (n_bots, 6) for last timestep only.
                # We need all timesteps — unroll manually
                auc_loss = self._sequence_loss(
                    self.auction_net, obs1_seq, tgt_auc_seq,
                )

                self.auc_optimizer.zero_grad()
                auc_loss.backward()
                nn.utils.clip_grad_norm_(self.auction_net.parameters(), 1.0)
                self.auc_optimizer.step()

                # --- Secondary LSTM training ---
                obs2_t = torch.as_tensor(
                    obs2_bots, dtype=torch.float32, device=self.device,
                )
                tgt_sec = torch.as_tensor(
                    sec_targets, dtype=torch.float32, device=self.device,
                )

                obs2_seq = obs2_t.permute(1, 0, 2)
                tgt_sec_seq = tgt_sec.permute(1, 0, 2)

                sec_loss = self._sequence_loss(
                    self.secondary_net, obs2_seq, tgt_sec_seq,
                )

                self.sec_optimizer.zero_grad()
                sec_loss.backward()
                nn.utils.clip_grad_norm_(self.secondary_net.parameters(), 1.0)
                self.sec_optimizer.step()

                total_auc_loss += auc_loss.item()
                total_sec_loss += sec_loss.item()
                n_updates += 1

        avg_auc = total_auc_loss / max(n_updates, 1)
        avg_sec = total_sec_loss / max(n_updates, 1)

        self.last_auc_loss = avg_auc
        self.last_sec_loss = avg_sec
        self._is_trained = True
        self.retrain_count += 1

        if verbose:
            print(f"  LSTM bot retrain #{self.retrain_count}: "
                  f"auc_loss={avg_auc:.4f}  sec_loss={avg_sec:.4f}")

        self.auction_net.eval()
        self.secondary_net.eval()

        return {"auc_loss": avg_auc, "sec_loss": avg_sec}

    def _sequence_loss(
        self,
        net: nn.Module,
        obs_seq: torch.Tensor,    # (n_bots, T, D)
        tgt_seq: torch.Tensor,    # (n_bots, T, A)
    ) -> torch.Tensor:
        """Compute MSE loss over a full sequence, unrolling the LSTM step by step."""
        n_bots, T, _ = obs_seq.shape
        hidden = None
        total_loss = torch.tensor(0.0, device=self.device)

        for t in range(T):
            obs_t = obs_seq[:, t, :]          # (n_bots, D)
            tgt_t = tgt_seq[:, t, :]          # (n_bots, A)
            pred_t, hidden = net(obs_t, hidden)  # (n_bots, A)
            total_loss = total_loss + nn.functional.mse_loss(pred_t, tgt_t)

        return total_loss / T

    # ------------------------------------------------------------------
    # Blend with heuristic
    # ------------------------------------------------------------------
    def compute_blend_alpha(self, episode: int, config: dict) -> float:
        """Compute blending coefficient α for LSTM vs heuristic.

        During warmup: α = 0  (pure heuristic).
        After warmup, ramps linearly from 0→1 over blend_window episodes.
        """
        lcd = config.get("lstm_bots", {})
        n_episodes = config["simulation"]["n_episodes"]
        warmup = lcd.get("warmup_episodes", 0)
        if warmup <= 0:
            warmup = int(0.15 * n_episodes)
        blend_window = lcd.get("blend_window", 1000)

        if episode < warmup:
            return 0.0
        elapsed = episode - warmup
        if elapsed >= blend_window:
            return 1.0
        return elapsed / blend_window

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save both LSTM networks to disk."""
        torch.save({
            "auction_net": self.auction_net.state_dict(),
            "secondary_net": self.secondary_net.state_dict(),
            "auc_optimizer": self.auc_optimizer.state_dict(),
            "sec_optimizer": self.sec_optimizer.state_dict(),
            "retrain_count": self.retrain_count,
            "is_trained": self._is_trained,
        }, path)

    def load(self, path: str):
        """Load both LSTM networks from disk."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.auction_net.load_state_dict(ckpt["auction_net"])
        self.secondary_net.load_state_dict(ckpt["secondary_net"])
        self.auc_optimizer.load_state_dict(ckpt["auc_optimizer"])
        self.sec_optimizer.load_state_dict(ckpt["sec_optimizer"])
        self.retrain_count = ckpt.get("retrain_count", 0)
        self._is_trained = ckpt.get("is_trained", True)
        self.auction_net.eval()
        self.secondary_net.eval()
