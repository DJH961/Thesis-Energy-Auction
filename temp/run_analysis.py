"""
Full diagnostic analysis of v7.11.0 training run (seed 42, 70k episodes).
Produces figures and statistics for the analysis bundle.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

BASE = "/Users/alessiodesideri/Desktop/CBS/tesi/Thesis-Energy-Auction-main 5/ets_marl_happo_current copia"
RESULTS = f"{BASE}/results"
OUTDIR = f"{BASE}/temp/analysis-output/figures"

AGENTS = [f"A{i}" for i in range(1, 9)]

# ── 1. Load training log (episode-level) ─────────────────────────────────
print("Loading training_log_s42.csv...")
tr_cols = ['episode', 'clearing_price_last', 'cap_last', 'entropy_coef', 'epsilon',
           'warn_agents_stuck_floor', 'warn_agents_stuck_ceiling']
for a in AGENTS:
    tr_cols += [f'reward_{a}', f'reward_base_{a}', f'reward_shaping_{a}',
                f'green_frac_{a}', f'delta_green_{a}', f'penalty_{a}',
                f'shortfall_{a}', f'queue_size_{a}', f'actor_loss_{a}',
                f'critic_loss_{a}', f'bid_price_{a}',
                f'streak_floor_{a}', f'streak_ceil_{a}',
                f'diag_S_financial_{a}', f'diag_S_green_{a}', f'diag_S_composite_{a}']

df_tr = pd.read_csv(f"{RESULTS}/training_log_s42.csv", usecols=lambda c: c in tr_cols)
df_tr = df_tr.sort_values('episode').reset_index(drop=True)
N_EPS = len(df_tr)
print(f"  Loaded {N_EPS} episodes")

# ── 2. Load year log (year-level) — sample for speed ────────────────────
print("Loading year_log_s42.csv (sampled)...")
yr_cols = ['episode', 'year', 'cap', 'auction_volume', 'clearing_price',
           'secondary_price', 'tnac', 'msr_reserve', 'inflation_factor']
for a in AGENTS:
    yr_cols += [f'bank_start_{a}', f'alloc_{a}', f'emissions_{a}',
                f'trade_qty_{a}', f'trade_cost_{a}', f'green_frac_{a}',
                f'shortfall_{a}', f'penalty_{a}', f'reward_{a}',
                f'holdings_{a}', f'invest_cost_{a}', f'collateral_cost_{a}',
                f'bid_price_{a}', f'bid_qty_mult_{a}', f'estimate_need_{a}',
                f'bid_coverage_{a}', f'bid_to_reserve_{a}',
                f'wtp_economic_{a}', f'wtp_budget_{a}', f'wtp_binding_{a}',
                f'available_budget_{a}', f'mac_cost_{a}',
                f'terminal_bank_value_{a}', f'terminal_queue_value_{a}',
                f'invest_frac_post_clip_{a}', f'sec_action_side_{a}']

df_yr = pd.read_csv(f"{RESULTS}/year_log_s42.csv", usecols=lambda c: c in yr_cols)
df_yr = df_yr.sort_values(['episode', 'year']).reset_index(drop=True)
print(f"  Loaded {len(df_yr)} year-rows ({len(df_yr)//12} episodes × 12 years)")


# ── Helper: rolling window ──────────────────────────────────────────────
def rolling_mean(series, window=500):
    return series.rolling(window, min_periods=1).mean()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Clearing Price Trajectory
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Clearing price trajectory...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

ax = axes[0]
cp = df_tr['clearing_price_last']
ax.plot(df_tr['episode'], rolling_mean(cp, 1000), color='#2196F3', lw=1.5, label='Clearing price (MA-1000)')
ax.axhline(45.0, color='red', ls='--', lw=1, alpha=0.7, label='Price floor (45 EUR)')
ax.axhline(138.75, color='orange', ls='--', lw=1, alpha=0.5, label='Penalty rate (138.75 EUR)')
ax.set_ylabel('Clearing Price (EUR/tCO2)')
ax.set_title('Figure 1: Auction Clearing Price Over Training')
ax.legend(loc='upper right')
ax.set_xlim(0, N_EPS)

# Epsilon overlay
ax2 = ax.twinx()
ax2.fill_between(df_tr['episode'], df_tr['epsilon'], alpha=0.1, color='gray', label='Epsilon')
ax2.set_ylabel('Epsilon', color='gray')
ax2.set_ylim(0, 0.35)

ax = axes[1]
# Floor-stuck agents count
for a in AGENTS:
    col = f'streak_floor_{a}'
    if col in df_tr.columns:
        ax.plot(df_tr['episode'], rolling_mean(df_tr[col].clip(upper=50), 2000),
                lw=0.8, alpha=0.6, label=a)
ax.set_ylabel('Floor streak (MA-2k)')
ax.set_xlabel('Episode')
ax.legend(ncol=8, fontsize=8)
ax.set_xlim(0, N_EPS)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-01-clearing-price.png")
plt.close()
print("  Saved figure-01-clearing-price.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Per-Agent Bid Price Evolution
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 2: Per-agent bid prices...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
colors = plt.cm.tab10(range(8))

for idx, a in enumerate(AGENTS):
    ax = axes[idx // 4, idx % 4]
    col = f'bid_price_{a}'
    if col in df_tr.columns:
        bp = df_tr[col]
        ax.plot(df_tr['episode'], rolling_mean(bp, 2000), color=colors[idx], lw=1.2)
        ax.axhline(45.0, color='red', ls=':', lw=0.8, alpha=0.5)
        # Show raw scatter for last 5k
        mask = df_tr['episode'] >= N_EPS - 5000
        ax.scatter(df_tr.loc[mask, 'episode'], bp[mask], s=0.3, alpha=0.15, color=colors[idx])
    ax.set_title(f'{a}', fontsize=11)
    ax.set_ylim(30, 200)
    if idx >= 4:
        ax.set_xlabel('Episode')
    if idx % 4 == 0:
        ax.set_ylabel('Bid Price (EUR/t)')

plt.suptitle('Figure 2: Per-Agent Mean Bid Price (MA-2000) + Raw (last 5k)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-02-bid-prices.png")
plt.close()
print("  Saved figure-02-bid-prices.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Reward Decomposition
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 3: Reward decomposition...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, a in enumerate(AGENTS):
    ax = axes[idx // 4, idx % 4]
    for comp, color, label in [
        (f'reward_base_{a}', '#1976D2', 'Base'),
        (f'reward_shaping_{a}', '#4CAF50', 'Shaping'),
        (f'reward_{a}', '#FF5722', 'Total'),
    ]:
        if comp in df_tr.columns:
            ax.plot(df_tr['episode'], rolling_mean(df_tr[comp], 2000),
                    color=color, lw=1.0, label=label, alpha=0.8)
    ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.3)
    ax.set_title(f'{a}', fontsize=11)
    ax.legend(fontsize=7, loc='best')
    if idx >= 4:
        ax.set_xlabel('Episode')
    if idx % 4 == 0:
        ax.set_ylabel('Reward (MA-2k)')

plt.suptitle('Figure 3: Per-Agent Reward Decomposition', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-03-reward-decomposition.png")
plt.close()
print("  Saved figure-03-reward-decomposition.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Green Transition
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 4: Green fraction evolution...")
fig, ax = plt.subplots(figsize=(16, 8))

for idx, a in enumerate(AGENTS):
    col = f'green_frac_{a}'
    if col in df_tr.columns:
        ax.plot(df_tr['episode'], rolling_mean(df_tr[col], 2000),
                color=colors[idx], lw=1.5, label=a)

ax.set_ylabel('Green Fraction (end-of-episode)')
ax.set_xlabel('Episode')
ax.set_title('Figure 4: Green Transition Progress per Agent')
ax.legend(ncol=4)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, N_EPS)
ax.axhline(1.0, color='green', ls='--', lw=0.8, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-04-green-fraction.png")
plt.close()
print("  Saved figure-04-green-fraction.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Penalty & Shortfall
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 5: Penalty and shortfall...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
for idx, a in enumerate(AGENTS):
    col = f'penalty_{a}'
    if col in df_tr.columns:
        ax.plot(df_tr['episode'], rolling_mean(df_tr[col], 2000),
                color=colors[idx], lw=1.2, label=a)
ax.set_ylabel('Cumulative Penalty (M EUR, MA-2k)')
ax.set_xlabel('Episode')
ax.set_title('Penalty Costs per Agent')
ax.legend(ncol=4, fontsize=8)

ax = axes[1]
for idx, a in enumerate(AGENTS):
    col = f'shortfall_{a}'
    if col in df_tr.columns:
        ax.plot(df_tr['episode'], rolling_mean(df_tr[col], 2000),
                color=colors[idx], lw=1.2, label=a)
ax.set_ylabel('Shortfall (Mt, MA-2k)')
ax.set_xlabel('Episode')
ax.set_title('Compliance Shortfall per Agent')
ax.legend(ncol=4, fontsize=8)

plt.suptitle('Figure 5: Penalty Burden & Compliance Shortfall', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-05-penalty-shortfall.png")
plt.close()
print("  Saved figure-05-penalty-shortfall.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Year-Level Market Dynamics (last 5k episodes)
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 6: Year-level market dynamics...")
last_5k = df_yr[df_yr['episode'] >= N_EPS - 5000].copy()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 6a: Clearing price by year
ax = axes[0, 0]
for yr in range(12):
    sub = last_5k[last_5k['year'] == yr]
    bp = ax.boxplot(sub['clearing_price'].dropna(), positions=[yr], widths=0.6,
                    patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(plt.cm.viridis(yr / 11))
ax.axhline(45.0, color='red', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Clearing Price (EUR/t)')
ax.set_title('6a: Clearing Price by Year (last 5k eps)')

# 6b: Total emissions by year
ax = axes[0, 1]
total_em = sum(last_5k[f'emissions_{a}'] for a in AGENTS if f'emissions_{a}' in last_5k.columns)
em_by_year = last_5k.groupby('year').apply(
    lambda g: sum(g[f'emissions_{a}'].mean() for a in AGENTS if f'emissions_{a}' in g.columns))
cap_by_year = last_5k.groupby('year')['cap'].mean()
ax.bar(em_by_year.index, em_by_year.values, color='#FF7043', alpha=0.7, label='Total Emissions')
ax.plot(cap_by_year.index, cap_by_year.values, 'k--', lw=2, label='Cap')
ax.set_xlabel('Year')
ax.set_ylabel('Emissions / Cap (Mt)')
ax.set_title('6b: Total Emissions vs Cap (last 5k eps)')
ax.legend()

# 6c: Secondary price by year
ax = axes[1, 0]
for yr in range(12):
    sub = last_5k[last_5k['year'] == yr]
    vals = sub['secondary_price'].dropna()
    if len(vals) > 0:
        bp = ax.boxplot(vals, positions=[yr], widths=0.6,
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor(plt.cm.plasma(yr / 11))
ax.set_xlabel('Year')
ax.set_ylabel('Secondary Price (EUR/t)')
ax.set_title('6c: Secondary Market Price by Year (last 5k eps)')

# 6d: Bid coverage ratio
ax = axes[1, 1]
for idx, a in enumerate(AGENTS):
    col = f'bid_coverage_{a}'
    if col in last_5k.columns:
        by_yr = last_5k.groupby('year')[col].mean()
        ax.plot(by_yr.index, by_yr.values, color=colors[idx], lw=1.2, marker='o', ms=3, label=a)
ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Bid Coverage Ratio')
ax.set_title('6d: Bid Coverage Ratio by Year (last 5k eps)')
ax.legend(ncol=4, fontsize=8)

plt.suptitle('Figure 6: Year-Level Market Dynamics (Converged Phase)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-06-year-dynamics.png")
plt.close()
print("  Saved figure-06-year-dynamics.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Actor/Critic Loss
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 7: Training losses...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
for idx, a in enumerate(AGENTS):
    col = f'actor_loss_{a}'
    if col in df_tr.columns:
        ax.plot(df_tr['episode'], rolling_mean(df_tr[col].clip(-1, 1), 2000),
                color=colors[idx], lw=0.8, alpha=0.7, label=a)
ax.set_ylabel('Actor Loss (MA-2k, clipped)')
ax.set_xlabel('Episode')
ax.set_title('Actor Loss')
ax.legend(ncol=4, fontsize=8)

ax = axes[1]
for idx, a in enumerate(AGENTS):
    col = f'critic_loss_{a}'
    if col in df_tr.columns:
        ax.plot(df_tr['episode'], rolling_mean(df_tr[col].clip(0, 50), 2000),
                color=colors[idx], lw=0.8, alpha=0.7, label=a)
ax.set_ylabel('Critic Loss (MA-2k)')
ax.set_xlabel('Episode')
ax.set_title('Critic Loss')
ax.legend(ncol=4, fontsize=8)

plt.suptitle('Figure 7: Actor & Critic Training Loss', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-07-training-losses.png")
plt.close()
print("  Saved figure-07-training-losses.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Investment & Queue
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 8: Investment behavior...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
for idx, a in enumerate(AGENTS):
    col = f'queue_size_{a}'
    if col in df_tr.columns:
        ax.plot(df_tr['episode'], rolling_mean(df_tr[col], 2000),
                color=colors[idx], lw=1.2, label=a)
ax.set_ylabel('Queue Size (MA-2k)')
ax.set_xlabel('Episode')
ax.set_title('Construction Queue Size')
ax.legend(ncol=4, fontsize=8)

# Investment fraction from year log
ax = axes[1]
last_10k = df_yr[df_yr['episode'] >= N_EPS - 10000]
for idx, a in enumerate(AGENTS):
    col = f'invest_frac_post_clip_{a}'
    if col in last_10k.columns:
        by_yr = last_10k.groupby('year')[col].mean()
        ax.plot(by_yr.index, by_yr.values, color=colors[idx], lw=1.5, marker='o', ms=4, label=a)
ax.set_xlabel('Year')
ax.set_ylabel('Invest Frac (post-clip)')
ax.set_title('Investment Fraction by Year (last 10k eps)')
ax.legend(ncol=4, fontsize=8)

plt.suptitle('Figure 8: Investment Behavior', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-08-investment.png")
plt.close()
print("  Saved figure-08-investment.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 9: Diagnostic Scores
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 9: Diagnostic scores...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, metric, title in zip(axes,
    ['diag_S_financial', 'diag_S_green', 'diag_S_composite'],
    ['S_financial', 'S_green', 'S_composite']):
    for idx, a in enumerate(AGENTS):
        col = f'{metric}_{a}'
        if col in df_tr.columns:
            ax.plot(df_tr['episode'], rolling_mean(df_tr[col], 3000),
                    color=colors[idx], lw=1.0, label=a)
    ax.set_ylabel(f'{title} (MA-3k)')
    ax.set_xlabel('Episode')
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=7)

plt.suptitle('Figure 9: Diagnostic Scores', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-09-diagnostic-scores.png")
plt.close()
print("  Saved figure-09-diagnostic-scores.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 10: Bid-to-Reserve Ratio & WTP Analysis (year-level, last 5k)
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 10: Bid-to-reserve and WTP...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
for idx, a in enumerate(AGENTS):
    col = f'bid_to_reserve_{a}'
    if col in last_5k.columns:
        by_yr = last_5k.groupby('year')[col].mean()
        ax.plot(by_yr.index, by_yr.values, color=colors[idx], lw=1.5, marker='o', ms=4, label=a)
ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label='At reserve')
ax.set_xlabel('Year')
ax.set_ylabel('Bid / Reserve Price')
ax.set_title('10a: Bid-to-Reserve Ratio (last 5k eps)')
ax.legend(ncol=4, fontsize=8)

ax = axes[1]
for idx, a in enumerate(AGENTS):
    wtp_e = f'wtp_economic_{a}'
    bid_p = f'bid_price_{a}'
    if wtp_e in last_5k.columns and bid_p in last_5k.columns:
        wtp_vals = pd.to_numeric(last_5k[wtp_e], errors='coerce')
        bid_vals = pd.to_numeric(last_5k[bid_p], errors='coerce')
        ratio = (bid_vals / wtp_vals.clip(lower=1)).mean()
        by_yr_bid = last_5k.groupby('year')[bid_p].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
        by_yr_wtp = last_5k.groupby('year')[wtp_e].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
        ax.plot(by_yr_bid.index, by_yr_bid.values, color=colors[idx], lw=1.5, ls='-', marker='o', ms=3, label=f'{a} bid')
        ax.plot(by_yr_wtp.index, by_yr_wtp.values, color=colors[idx], lw=1.0, ls='--', marker='s', ms=3, alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Price (EUR/t)')
ax.set_title('10b: Bid Price vs WTP (solid=bid, dashed=WTP)')
ax.legend(ncol=4, fontsize=8)

plt.suptitle('Figure 10: Bid Strategy Analysis', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/figure-10-bid-strategy.png")
plt.close()
print("  Saved figure-10-bid-strategy.png")


# ══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("STATISTICS SUMMARY")
print("="*70)

# ── Phase windows ─────────────────────────────────────────────────────
windows = {
    'Early (0-5k)': (0, 5000),
    'Mid (20k-30k)': (20000, 30000),
    'Late (60k-70k)': (60000, 70000),
    'Last 5k': (N_EPS-5000, N_EPS),
}

print("\n── Clearing Price by Phase ──")
for label, (lo, hi) in windows.items():
    mask = (df_tr['episode'] >= lo) & (df_tr['episode'] < hi)
    cp_win = df_tr.loc[mask, 'clearing_price_last']
    print(f"  {label:20s}: mean={cp_win.mean():.2f}, std={cp_win.std():.2f}, "
          f"median={cp_win.median():.2f}, min={cp_win.min():.2f}, max={cp_win.max():.2f}, "
          f"pct_at_floor={100*(cp_win <= 45.5).mean():.1f}%")

print("\n── Per-Agent Bid Price (Last 5k episodes) ──")
mask_last = df_tr['episode'] >= N_EPS - 5000
for a in AGENTS:
    col = f'bid_price_{a}'
    if col in df_tr.columns:
        bp = df_tr.loc[mask_last, col]
        print(f"  {a}: mean={bp.mean():.2f}, std={bp.std():.2f}, "
              f"median={bp.median():.2f}, [p5={bp.quantile(0.05):.1f}, p95={bp.quantile(0.95):.1f}]")

print("\n── Per-Agent Green Fraction (Last 5k episodes, end-of-episode) ──")
for a in AGENTS:
    col = f'green_frac_{a}'
    if col in df_tr.columns:
        gf = df_tr.loc[mask_last, col]
        print(f"  {a}: mean={gf.mean():.4f}, std={gf.std():.4f}")

print("\n── Per-Agent Total Reward (Last 5k episodes) ──")
for a in AGENTS:
    col = f'reward_{a}'
    if col in df_tr.columns:
        rw = df_tr.loc[mask_last, col]
        print(f"  {a}: mean={rw.mean():.4f}, std={rw.std():.4f}, "
              f"base_mean={df_tr.loc[mask_last, f'reward_base_{a}'].mean():.4f}")

print("\n── Per-Agent Penalty (Last 5k episodes, episode total M EUR) ──")
for a in AGENTS:
    col = f'penalty_{a}'
    if col in df_tr.columns:
        pen = df_tr.loc[mask_last, col]
        print(f"  {a}: mean={pen.mean():.2f}, std={pen.std():.2f}")

print("\n── Market Scarcity (Year-level, last 5k episodes) ──")
yr_last = df_yr[df_yr['episode'] >= N_EPS - 5000]
for yr in [0, 3, 6, 9, 11]:
    sub = yr_last[yr_last['year'] == yr]
    total_em = sum(sub[f'emissions_{a}'].mean() for a in AGENTS if f'emissions_{a}' in sub.columns)
    cap_val = sub['cap'].mean()
    cp_val = sub['clearing_price'].mean()
    print(f"  Year {yr:2d}: emissions={total_em:.2f} Mt, cap={cap_val:.2f} Mt, "
          f"ratio={total_em/cap_val:.3f}, clearing_price={cp_val:.2f}")

print("\n── Bid-to-Reserve Ratio (Year-level, last 5k episodes) ──")
for a in AGENTS:
    col = f'bid_to_reserve_{a}'
    if col in yr_last.columns:
        btr = pd.to_numeric(yr_last[col], errors='coerce')
        print(f"  {a}: mean={btr.mean():.4f}, std={btr.std():.4f}")

print("\n── Floor Streak (Last 5k episodes) ──")
for a in AGENTS:
    col = f'streak_floor_{a}'
    if col in df_tr.columns:
        strk = df_tr.loc[mask_last, col]
        print(f"  {a}: mean_streak={strk.mean():.1f}, max_streak={strk.max():.0f}")

print("\n── Diagnostic Scores (Last 5k episodes) ──")
for a in AGENTS:
    fin = f'diag_S_financial_{a}'
    grn = f'diag_S_green_{a}'
    cmp = f'diag_S_composite_{a}'
    if all(c in df_tr.columns for c in [fin, grn, cmp]):
        print(f"  {a}: S_fin={df_tr.loc[mask_last, fin].mean():.4f}, "
              f"S_green={df_tr.loc[mask_last, grn].mean():.4f}, "
              f"S_comp={df_tr.loc[mask_last, cmp].mean():.4f}")

# ── Convergence test: is clearing price improving? ─────────────────────
print("\n── Convergence Test: Clearing Price ──")
for (l1, (lo1, hi1)), (l2, (lo2, hi2)) in [
    (('Early (0-5k)', (0, 5000)), ('Mid (20k-30k)', (20000, 30000))),
    (('Mid (20k-30k)', (20000, 30000)), ('Late (60k-70k)', (60000, 70000))),
]:
    cp1 = df_tr.loc[(df_tr['episode'] >= lo1) & (df_tr['episode'] < hi1), 'clearing_price_last']
    cp2 = df_tr.loc[(df_tr['episode'] >= lo2) & (df_tr['episode'] < hi2), 'clearing_price_last']
    from scipy import stats
    t_stat, p_val = stats.mannwhitneyu(cp1, cp2, alternative='two-sided')
    print(f"  {l1} vs {l2}: U={t_stat:.0f}, p={p_val:.4e}, "
          f"Δmean={cp2.mean()-cp1.mean():.3f} EUR")

# ── Secondary market usage ──────────────────────────────────────────────
print("\n── Secondary Market (last 5k eps) ──")
for a in AGENTS:
    side_col = f'sec_action_side_{a}'
    qty_col = f'trade_qty_{a}'
    if side_col in yr_last.columns:
        sides = yr_last[side_col]
        buys = (sides > 0).mean()
        sells = (sides < 0).mean()
        idle = (sides == 0).mean()
        trade_vol = yr_last[qty_col].abs().mean() if qty_col in yr_last.columns else 0
        print(f"  {a}: buy={100*buys:.1f}%, sell={100*sells:.1f}%, idle={100*idle:.1f}%, "
              f"avg_vol={trade_vol:.3f} Mt")

print("\n\nAnalysis complete. All figures saved to temp/analysis-output/figures/")
