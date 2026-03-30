"""
qlearning_analysis.py
=====================
Analysis and visualization tools for Q-learning baseline results.

Produces:
  1. Q-table heatmaps (dominant action per state)
  2. Strategy frequency histograms per agent archetype
  3. Comparison plots vs PPO (reward curves, prices, green fracs, compliance)

Usage:
    python src/analysis/qlearning_analysis.py \
        --ql-results results/qlearning/ \
        --ppo-results results/ \
        --seed 42
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.agents.q_learning_agent import (
    StateDiscretizer, ActionProfileMapper, QLearningAgent
)

# Agent archetype labels
ARCHETYPES = [
    "Coal-heavy/Fin", "Coal-heavy/Green",
    "Gas-dom/Fin", "Gas-dom/Green",
    "Transit/Fin", "Transit/Green",
    "Green-lead/Fin", "Green-lead/Green",
]
FINANCIAL_AGENTS = [0, 2, 4, 6]  # even indices
GREEN_AGENTS = [1, 3, 5, 7]       # odd indices


def load_qtables(results_dir: str, seed: int, episode: str = "final") -> dict:
    """Load Q-tables from pickle file."""
    if episode == "final":
        path = os.path.join(results_dir, f"qtables_s{seed}_final.pkl")
    else:
        path = os.path.join(results_dir, "checkpoints",
                            f"qtables_s{seed}_ep{episode}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_training_log(results_dir: str, seed: int, prefix: str = "ql") -> pd.DataFrame:
    """Load episode-level training log CSV."""
    path = os.path.join(results_dir, f"{prefix}_training_log_s{seed}.csv")
    return pd.read_csv(path)


def load_year_log(results_dir: str, seed: int, prefix: str = "ql") -> pd.DataFrame:
    """Load year-level training log CSV."""
    path = os.path.join(results_dir, f"{prefix}_year_log_s{seed}.csv")
    return pd.read_csv(path)


def load_eval_log(results_dir: str, seed: int) -> pd.DataFrame:
    """Load evaluation log CSV."""
    path = os.path.join(results_dir, f"ql_eval_log_s{seed}.csv")
    return pd.read_csv(path)


# =====================================================================
# 1. Q-Table Heatmaps
# =====================================================================

def plot_qtable_heatmaps(qtables: dict, n_agents: int = 8,
                         save_path: str = None):
    """
    Plot heatmap of dominant auction profile per state for each agent.

    Each cell in the 243×1 grid is colored by the argmax auction profile.
    """
    discretizer = StateDiscretizer()
    n_states = StateDiscretizer.N_STATES
    profile_names = ActionProfileMapper.auction_profile_names()

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Q-Table: Dominant Auction Profile per State", fontsize=14, y=1.02)

    cmap = plt.cm.get_cmap("Set2", 6)

    for idx in range(n_agents):
        ax = axes[idx // 4, idx % 4]
        key = f"agent_{idx}"
        if key not in qtables:
            ax.set_title(f"A{idx+1} (missing)")
            continue

        qt = qtables[key]
        # For each state, find best auction profile (marginalized over secondary)
        best_a1 = qt.max(axis=2).argmax(axis=1)  # shape (243,)

        # Reshape to 27×9 grid for visualization (3^3 × 3^2)
        grid = best_a1.reshape(27, 9)

        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=5)
        ax.set_title(f"A{idx+1}: {ARCHETYPES[idx]}", fontsize=10)
        ax.set_xlabel("State (low dims)")
        ax.set_ylabel("State (high dims)")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                      norm=plt.Normalize(0, 5)),
                      cax=cbar_ax)
    cb.set_ticks(np.arange(6) + 0.5)
    cb.set_ticklabels(profile_names)

    plt.tight_layout(rect=[0, 0, 0.90, 1.0])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()


def plot_secondary_heatmaps(qtables: dict, n_agents: int = 8,
                             save_path: str = None):
    """Plot dominant secondary profile per state for each agent."""
    profile_names = ActionProfileMapper.secondary_profile_names()

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Q-Table: Dominant Secondary Profile per State", fontsize=14, y=1.02)

    cmap = plt.cm.get_cmap("Set1", 4)

    for idx in range(n_agents):
        ax = axes[idx // 4, idx % 4]
        key = f"agent_{idx}"
        if key not in qtables:
            ax.set_title(f"A{idx+1} (missing)")
            continue

        qt = qtables[key]
        # Best secondary profile given best auction profile
        best_a1 = qt.max(axis=2).argmax(axis=1)
        best_a2 = np.array([qt[s, best_a1[s], :].argmax()
                            for s in range(StateDiscretizer.N_STATES)])

        grid = best_a2.reshape(27, 9)
        ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=3)
        ax.set_title(f"A{idx+1}: {ARCHETYPES[idx]}", fontsize=10)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                      norm=plt.Normalize(0, 3)),
                      cax=cbar_ax)
    cb.set_ticks(np.arange(4) + 0.5)
    cb.set_ticklabels(profile_names)

    plt.tight_layout(rect=[0, 0, 0.90, 1.0])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()


# =====================================================================
# 2. Strategy Frequency Histograms
# =====================================================================

def plot_strategy_frequency(df: pd.DataFrame, n_agents: int = 8,
                            last_n: int = 500, save_path: str = None):
    """
    Bar chart of action profile frequency during the last N episodes.

    Groups agents by archetype: Financial vs Green objective.
    """
    profile_names_a1 = ActionProfileMapper.auction_profile_names()
    profile_names_a2 = ActionProfileMapper.secondary_profile_names()

    tail = df.tail(last_n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Strategy Profile Frequency (Last {last_n} Episodes)", fontsize=14)

    # Financial agents — auction profiles
    ax = axes[0, 0]
    counts = np.zeros(6)
    for i in FINANCIAL_AGENTS:
        col = f"a1_profile_A{i+1}"
        if col in tail.columns:
            for p in range(6):
                counts[p] += (tail[col].astype(int) == p).sum()
    ax.bar(range(6), counts, color="steelblue", alpha=0.8)
    ax.set_xticks(range(6))
    ax.set_xticklabels(profile_names_a1, rotation=30, ha="right", fontsize=8)
    ax.set_title("Financial Agents — Auction Profiles")
    ax.set_ylabel("Count")

    # Green agents — auction profiles
    ax = axes[0, 1]
    counts = np.zeros(6)
    for i in GREEN_AGENTS:
        col = f"a1_profile_A{i+1}"
        if col in tail.columns:
            for p in range(6):
                counts[p] += (tail[col].astype(int) == p).sum()
    ax.bar(range(6), counts, color="forestgreen", alpha=0.8)
    ax.set_xticks(range(6))
    ax.set_xticklabels(profile_names_a1, rotation=30, ha="right", fontsize=8)
    ax.set_title("Green Agents — Auction Profiles")

    # Financial agents — secondary profiles
    ax = axes[1, 0]
    counts = np.zeros(4)
    for i in FINANCIAL_AGENTS:
        col = f"a2_profile_A{i+1}"
        if col in tail.columns:
            for p in range(4):
                counts[p] += (tail[col].astype(int) == p).sum()
    ax.bar(range(4), counts, color="steelblue", alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(profile_names_a2, fontsize=9)
    ax.set_title("Financial Agents — Secondary Profiles")
    ax.set_ylabel("Count")

    # Green agents — secondary profiles
    ax = axes[1, 1]
    counts = np.zeros(4)
    for i in GREEN_AGENTS:
        col = f"a2_profile_A{i+1}"
        if col in tail.columns:
            for p in range(4):
                counts[p] += (tail[col].astype(int) == p).sum()
    ax.bar(range(4), counts, color="forestgreen", alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(profile_names_a2, fontsize=9)
    ax.set_title("Green Agents — Secondary Profiles")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()


# =====================================================================
# 3. Comparison Plots: Q-Learning vs PPO
# =====================================================================

def _smooth(arr, window=50):
    """Simple moving average for smoothing curves."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_reward_comparison(ql_df: pd.DataFrame, ppo_df: pd.DataFrame = None,
                            n_agents: int = 8, save_path: str = None):
    """Overlay reward curves for Q-learning and PPO."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    fig.suptitle("Reward Comparison: Q-Learning vs PPO", fontsize=14)

    for idx in range(n_agents):
        ax = axes[idx // 4, idx % 4]
        col = f"reward_A{idx+1}"

        if col in ql_df.columns:
            ql_rewards = ql_df[col].astype(float).values
            ql_smooth = _smooth(ql_rewards)
            ax.plot(ql_smooth, label="Q-Learning", color="tab:blue", alpha=0.8)

        if ppo_df is not None and col in ppo_df.columns:
            ppo_rewards = ppo_df[col].astype(float).values
            ppo_smooth = _smooth(ppo_rewards)
            ax.plot(ppo_smooth, label="PPO/HAPPO", color="tab:orange", alpha=0.8)

        ax.set_title(f"A{idx+1}: {ARCHETYPES[idx]}", fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()


def plot_price_comparison(ql_df: pd.DataFrame, ppo_df: pd.DataFrame = None,
                           save_path: str = None):
    """Overlay clearing price trajectories."""
    fig, ax = plt.subplots(figsize=(12, 5))

    if "clearing_price_last" in ql_df.columns:
        ql_prices = ql_df["clearing_price_last"].astype(float).values
        ax.plot(_smooth(ql_prices), label="Q-Learning", color="tab:blue", alpha=0.8)

    if ppo_df is not None and "clearing_price_last" in ppo_df.columns:
        ppo_prices = ppo_df["clearing_price_last"].astype(float).values
        ax.plot(_smooth(ppo_prices), label="PPO/HAPPO", color="tab:orange", alpha=0.8)

    ax.set_title("Clearing Price Trajectory", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Price (€/t)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()


def plot_green_comparison(ql_df: pd.DataFrame, ppo_df: pd.DataFrame = None,
                           n_agents: int = 8, save_path: str = None):
    """Overlay green fraction trajectories."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Average across agents
    ql_cols = [f"green_frac_A{i+1}" for i in range(n_agents)]
    available_ql = [c for c in ql_cols if c in ql_df.columns]
    if available_ql:
        ql_avg = ql_df[available_ql].astype(float).mean(axis=1).values
        ax.plot(_smooth(ql_avg), label="Q-Learning (avg)", color="tab:blue", alpha=0.8)

    if ppo_df is not None:
        available_ppo = [c for c in ql_cols if c in ppo_df.columns]
        if available_ppo:
            ppo_avg = ppo_df[available_ppo].astype(float).mean(axis=1).values
            ax.plot(_smooth(ppo_avg), label="PPO/HAPPO (avg)",
                    color="tab:orange", alpha=0.8)

    ax.set_title("Average Green Fraction Trajectory", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Green Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()


def plot_compliance_comparison(ql_df: pd.DataFrame, ppo_df: pd.DataFrame = None,
                                n_agents: int = 8, save_path: str = None):
    """Overlay compliance rate trajectories."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ql_cols = [f"compliance_rate_A{i+1}" for i in range(n_agents)]
    available_ql = [c for c in ql_cols if c in ql_df.columns]
    if available_ql:
        ql_avg = ql_df[available_ql].astype(float).mean(axis=1).values
        ax.plot(_smooth(ql_avg), label="Q-Learning (avg)", color="tab:blue", alpha=0.8)

    if ppo_df is not None:
        # PPO log may not have compliance_rate directly; compute from shortfall
        ppo_cols_sf = [f"shortfall_A{i+1}" for i in range(n_agents)]
        available_ppo = [c for c in ppo_cols_sf if c in ppo_df.columns]
        if available_ppo:
            # compliance ≈ 1 - (shortfall > 0)
            ppo_comply = (ppo_df[available_ppo].astype(float) < 0.001).mean(axis=1).values
            ax.plot(_smooth(ppo_comply), label="PPO/HAPPO (avg)",
                    color="tab:orange", alpha=0.8)

    ax.set_title("Average Compliance Rate", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Compliance Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()


# =====================================================================
# Main CLI
# =====================================================================

def run_all_analysis(ql_dir: str, ppo_dir: str = None, seed: int = 42,
                     n_agents: int = 8):
    """Run all analysis and save plots."""
    plots_dir = os.path.join(ql_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nQ-Learning Analysis — seed {seed}")
    print(f"  Results dir: {ql_dir}")
    print(f"  Plots dir:   {plots_dir}")

    # Load Q-tables
    try:
        qtables = load_qtables(ql_dir, seed)
        plot_qtable_heatmaps(qtables, n_agents,
                             save_path=os.path.join(plots_dir, "qtable_auction_heatmap.png"))
        plot_secondary_heatmaps(qtables, n_agents,
                                 save_path=os.path.join(plots_dir, "qtable_secondary_heatmap.png"))
    except FileNotFoundError as e:
        print(f"  Q-tables not found: {e}")

    # Load training log
    try:
        ql_df = load_training_log(ql_dir, seed)
        plot_strategy_frequency(ql_df, n_agents,
                                save_path=os.path.join(plots_dir, "strategy_frequency.png"))
    except FileNotFoundError:
        ql_df = None
        print("  Training log not found — skipping strategy frequency plot.")

    # Load PPO log for comparison
    ppo_df = None
    if ppo_dir:
        try:
            ppo_df = load_training_log(ppo_dir, seed, prefix="training")
        except FileNotFoundError:
            print(f"  PPO log not found in {ppo_dir} — comparison plots will show Q-learning only.")

    # Comparison plots
    if ql_df is not None:
        plot_reward_comparison(ql_df, ppo_df, n_agents,
                               save_path=os.path.join(plots_dir, "reward_comparison.png"))
        plot_price_comparison(ql_df, ppo_df,
                              save_path=os.path.join(plots_dir, "price_comparison.png"))
        plot_green_comparison(ql_df, ppo_df, n_agents,
                              save_path=os.path.join(plots_dir, "green_comparison.png"))
        plot_compliance_comparison(ql_df, ppo_df, n_agents,
                                   save_path=os.path.join(plots_dir, "compliance_comparison.png"))

    print("\nAnalysis complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Q-Learning analysis")
    parser.add_argument("--ql-results", type=str, default="results/qlearning/",
                        help="Q-learning results directory")
    parser.add_argument("--ppo-results", type=str, default=None,
                        help="PPO results directory (for comparison)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-agents", type=int, default=8)
    args = parser.parse_args()

    run_all_analysis(args.ql_results, args.ppo_results, args.seed, args.n_agents)


if __name__ == "__main__":
    main()
