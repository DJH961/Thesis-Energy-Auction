"""
train.py
========
Main training entry point for ETS MARL.

Usage:
    python scripts/train.py --config configs/default.yaml --seed 42
    python scripts/train.py --config configs/default.yaml --seed 42 123 456
"""

import argparse
import os
import sys
import yaml
import numpy as np

# Make src importable from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.agents.ddpg_agent import DDPGAgent
from src.utils.logger import Logger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_agents(env: ETSEnvironment, config: dict, seed: int):
    """Instantiate one DDPG agent per company."""
    obs_dim    = env.companies[0].obs_dim     # 11
    action_dim = 5                            # [bid_price, qty, delta_green,
                                              #  secondary_price_mult, secondary_qty]

    aq  = config["auction"]
    inv = config["investment"]

    action_low  = np.array([
        aq["price_min"],               # bid_price min (€/t)
        0.0,                           # qty_bid min
        0.0,                           # delta_green min
        0.5,                           # secondary_price_mult min
        -aq["quantity_max"],           # secondary_qty min
    ], dtype=np.float32)

    action_high = np.array([
        aq["price_max"],               # bid_price max (€/t)
        aq["quantity_max"],            # qty_bid max
        inv["max_delta_green_pp"],     # delta_green max
        2.0,                           # secondary_price_mult max
        aq["quantity_max"],            # secondary_qty max
    ], dtype=np.float32)

    agents = []
    for i in range(config["companies"]["n_agents"]):
        agent = DDPGAgent(
            agent_id=i,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            config=config,
            seed=seed + i,
        )
        agents.append(agent)
    return agents


def train_one_seed(config: dict, seed: int):
    print(f"\n{'='*60}")
    print(f"Training — seed {seed}")
    print(f"{'='*60}")

    # --- Environment ---
    env = ETSEnvironment(config, seed=seed)

    # --- Agents ---
    agents = build_agents(env, config, seed)

    # --- Logger ---
    logger = Logger(
        results_dir=config["logging"]["results_dir"],
        n_agents=config["companies"]["n_agents"],
        seed=seed,
    )

    n_episodes   = config["simulation"]["n_episodes"]
    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]

    best_total_reward = -np.inf

    for episode in range(n_episodes):

        # Reset environment and noise
        obs, _ = env.reset(seed=seed + episode * 1000)
        for agent in agents:
            agent.reset_noise()
            agent.decay_noise(episode)

        total_rewards = np.zeros(config["companies"]["n_agents"])
        latest_losses = [None] * config["companies"]["n_agents"]

        # --- Episode: one step = one year ---
        for year in range(config["simulation"]["n_years"]):

            # Each agent selects action based on its own observation
            actions = np.stack([
                agents[i].select_action(obs[i], explore=True)
                for i in range(len(agents))
            ])

            # Environment step
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            # Store transitions and update agents
            for i, agent in enumerate(agents):
                agent.store_transition(
                    obs=obs[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_obs=next_obs[i],
                    done=terminated,
                )
                loss = agent.update()
                if loss is not None:
                    latest_losses[i] = loss

            total_rewards += rewards
            obs = next_obs

            if terminated:
                break

        # --- Logging ---
        logger.log_episode(
            episode=episode,
            episode_log=env.episode_log,
            agent_rewards=total_rewards.tolist(),
            agent_losses=latest_losses,
            log_interval=log_interval,
        )

        # --- Checkpointing ---
        if episode % save_interval == 0:
            ckpt_dir = logger.checkpoint_dir()
            for i, agent in enumerate(agents):
                agent.save(os.path.join(ckpt_dir, f"agent_{i}_ep{episode}.pt"))

        # Save best model
        episode_total = total_rewards.sum()
        if episode_total > best_total_reward:
            best_total_reward = episode_total
            ckpt_dir = logger.checkpoint_dir()
            for i, agent in enumerate(agents):
                agent.save(os.path.join(ckpt_dir, f"agent_{i}_best.pt"))

    print(f"\nTraining complete — seed {seed}. Results in: {logger.run_dir}")
    return logger


def main():
    parser = argparse.ArgumentParser(description="Train ETS MARL agents")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed",   type=int, nargs="+", default=[42])
    args = parser.parse_args()

    config = load_config(args.config)

    for seed in args.seed:
        train_one_seed(config, seed)


if __name__ == "__main__":
    main()