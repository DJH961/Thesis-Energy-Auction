"""
evaluate.py
===========
Load trained PPO agents and run a deterministic evaluation episode.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
                                --checkpoint results/checkpoints_s42/
"""

import argparse
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from scripts.train import build_agents


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Directory containing agent_0_best.pt ... agent_3_best.pt")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    env = ETSEnvironment(config, seed=args.seed)
    n_agents = config["companies"]["n_agents"]

    # Build agents and load weights
    agents = build_agents(env, config, args.seed)
    for i, agent in enumerate(agents):
        ckpt_path = os.path.join(args.checkpoint, f"agent_{i}_best.pt")
        if os.path.exists(ckpt_path):
            agent.load(ckpt_path)
            print(f"Loaded agent {i} from {ckpt_path}")
        else:
            print(f"WARNING: checkpoint not found for agent {i} — using random policy")

    # Run one evaluation episode (deterministic)
    obs1, _ = env.reset(seed=args.seed)
    total_rewards = np.zeros(n_agents)

    print("\n" + "="*70)
    print("EVALUATION EPISODE — Technology-Specific Mix")
    print("="*70)

    tech_names = config["technologies"]["names"]

    for year in range(config["simulation"]["n_years"]):
        # Phase 1
        auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
        for i in range(n_agents):
            action, _, _ = agents[i].select_auction_action(obs1[i], deterministic=True)
            auction_actions[i] = action

        obs2, auction_info = env.step_auction(auction_actions)

        # Phase 2
        secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
        for i in range(n_agents):
            action, _, _ = agents[i].select_secondary_action(obs2[i], deterministic=True)
            secondary_actions[i] = action

        obs1, rewards, terminated, _, info = env.step_secondary(secondary_actions)
        total_rewards += rewards

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

    print("\n" + "-"*70)
    print("Total rewards per agent:", [f"{r:.2f}" for r in total_rewards])
    print("Final green fracs:      ", [f"{c.green_frac*100:.1f}%" for c in env.companies])
    print("Final tech mixes:")
    for i, c in enumerate(env.companies):
        mix_str = " | ".join([f"{tech_names[t]}:{c.mix[t]*100:.1f}%" for t in range(5)])
        print(f"  A{i+1}: {mix_str}")
    print("Final holdings (Mt):    ", [f"{h:.3f}" for h in env.holdings])


if __name__ == "__main__":
    main()
