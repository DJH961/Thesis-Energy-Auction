"""
evaluate.py
===========
Load trained DDPG agents and run a deterministic evaluation episode.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
                                --checkpoint results/seed_42_*/checkpoints/
"""

import argparse
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.agents.ddpg_agent import DDPGAgent


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Directory containing agent_0_best.pt ... agent_3_best.pt")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--render",     action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    env = ETSEnvironment(config, seed=args.seed)

    # Build agents and load weights
    aq  = config["auction"]
    inv = config["investment"]
    action_low  = np.array([0.0, 0.0, 0.0, -aq["quantity_max"]], dtype=np.float32)
    action_high = np.array([aq["price_max"], aq["quantity_max"],
                            inv["max_delta_green_pp"], aq["quantity_max"]], dtype=np.float32)

    agents = []
    for i in range(config["companies"]["n_agents"]):
        agent = DDPGAgent(
            agent_id=i,
            obs_dim=env.companies[0].obs_dim,
            action_dim=4,
            action_low=action_low,
            action_high=action_high,
            config=config,
            seed=args.seed + i,
        )
        ckpt_path = os.path.join(args.checkpoint, f"agent_{i}_best.pt")
        if os.path.exists(ckpt_path):
            agent.load(ckpt_path)
            print(f"Loaded agent {i} from {ckpt_path}")
        else:
            print(f"WARNING: checkpoint not found for agent {i} — using random policy")
        agents.append(agent)

    # Run one evaluation episode (no exploration noise)
    obs, _ = env.reset(seed=args.seed)
    total_rewards = np.zeros(config["companies"]["n_agents"])

    print("\n" + "="*60)
    print("EVALUATION EPISODE")
    print("="*60)

    for year in range(config["simulation"]["n_years"]):
        actions = np.stack([
            agents[i].select_action(obs[i], explore=False)
            for i in range(len(agents))
        ])
        obs, rewards, terminated, _, info = env.step(actions)
        total_rewards += rewards

        if args.render:
            env.render()

        log = info["year_log"]
        print(
            f"Year {year+1:2d} | "
            f"Cap={log['cap']:.2f}Mt | "
            f"Vol={log['auction_volume']:.2f}Mt | "
            f"TNAC={log['tnac']:.2f}Mt | "
            f"MSR={log['msr_reserve']:.3f}Mt | "
            f"Price={log['clearing_price']:.1f}€/t | "
            f"Green%={[f'{g*100:.1f}' for g in log['green_fracs']]}"
        )

        if terminated:
            break

    print("\n" + "-"*60)
    print("Total rewards per agent:", [f"{r:.2f}" for r in total_rewards])
    print("Final green fracs:      ", [f"{c.green_frac*100:.1f}%" for c in env.companies])
    print("Final banks (Mt):       ", [f"{c.bank:.3f}" for c in env.companies])


if __name__ == "__main__":
    main()
