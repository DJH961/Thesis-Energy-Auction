"""
logger.py
=========
Training metrics logger. Saves episode statistics to CSV and prints
summary to console.
"""

import os
import csv
import numpy as np
from datetime import datetime


class Logger:
    """
    Logs episode-level statistics for all agents.

    Parameters
    ----------
    results_dir : str
    n_agents : int
    seed : int
    """

    def __init__(self, results_dir: str, n_agents: int, seed: int):
        self.n_agents = n_agents
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(results_dir, f"seed_{seed}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # CSV header
        self.csv_path = os.path.join(self.run_dir, "training_log.csv")
        self._init_csv()

        self.episode_rewards = [[] for _ in range(n_agents)]

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            agent_cols = []
            for i in range(self.n_agents):
                agent_cols += [
                    f"reward_A{i+1}",
                    f"green_frac_A{i+1}",
                    f"bank_A{i+1}",
                    f"penalty_A{i+1}",
                    f"actor_loss_A{i+1}",
                    f"critic_loss_A{i+1}",
                ]
            header = ["episode", "clearing_price_last_year", "cap_last_year"] + agent_cols
            writer.writerow(header)

    def log_episode(
        self,
        episode: int,
        episode_log: list,
        agent_rewards: list,
        agent_losses: list,
        log_interval: int = 10,
    ):
        """
        Log one episode to CSV and optionally print to console.

        Parameters
        ----------
        episode : int
        episode_log : list of dict (one per year, from ETSEnvironment)
        agent_rewards : list of float (total reward per agent this episode)
        agent_losses : list of dict (latest actor/critic loss per agent)
        log_interval : int
        """
        last_year = episode_log[-1] if episode_log else {}
        clearing_price = last_year.get("clearing_price", 0.0)
        cap = last_year.get("cap", 0.0)
        green_fracs = last_year.get("green_fracs", [0.0] * self.n_agents)
        banks = last_year.get("banks", [0.0] * self.n_agents)
        penalties = last_year.get("penalties", [0.0] * self.n_agents)

        row = [episode, round(clearing_price, 2), round(cap, 3)]
        for i in range(self.n_agents):
            losses = agent_losses[i] or {"actor_loss": 0.0, "critic_loss": 0.0}
            row += [
                round(agent_rewards[i], 4),
                round(green_fracs[i], 4),
                round(banks[i], 4),
                round(penalties[i] / 1e6, 4),
                round(losses.get("actor_loss", 0.0), 6),
                round(losses.get("critic_loss", 0.0), 6),
            ]

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Store for running average
        for i in range(self.n_agents):
            self.episode_rewards[i].append(agent_rewards[i])

        if episode % log_interval == 0:
            avg_rewards = [
                np.mean(r[-log_interval:]) for r in self.episode_rewards
            ]
            print(
                f"Ep {episode:4d} | "
                f"Price: {clearing_price:6.1f} €/t | "
                f"Cap: {cap:.3f} Mt | "
                f"Avg rewards: {[f'{r:.1f}' for r in avg_rewards]}"
            )

    def checkpoint_dir(self) -> str:
        path = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)
        return path
