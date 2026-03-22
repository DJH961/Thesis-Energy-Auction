# Thesis-Energy-Auction

Multi-agent reinforcement learning experiments for emissions trading and energy-market transition behavior.

## Start Here

- Current flagship project (recommended): [ets_marl_happo_current](ets_marl_happo_current)
- Main project documentation: [ets_marl_happo_current/README.md](ets_marl_happo_current/README.md)

## Repository Structure

- [ets_marl_happo_current](ets_marl_happo_current)
Purpose: Active HAPPO-based implementation and analysis.
Status: Current and maintained.
Docs: [ets_marl_happo_current/README.md](ets_marl_happo_current/README.md)

- [ets_marl_legacy_ppo](ets_marl_legacy_ppo)
Purpose: Archived legacy PPO-era project snapshot.
Status: Legacy/archive.
Docs: No dedicated README yet in this folder.

- [ets_marl_legacy_test](ets_marl_legacy_test)
Purpose: Archived legacy test/scratch project.
Status: Legacy/archive.
Docs: [ets_marl_legacy_test/README.md](ets_marl_legacy_test/README.md)

## Documentation Map

- Primary methodology and usage: [ets_marl_happo_current/README.md](ets_marl_happo_current/README.md)
- Legacy test notes: [ets_marl_legacy_test/README.md](ets_marl_legacy_test/README.md)
- Legacy PPO archive folder: [ets_marl_legacy_ppo](ets_marl_legacy_ppo)

## Recommended Workflow

1. Use only [ets_marl_happo_current](ets_marl_happo_current) for new runs and comparisons.
2. Treat legacy folders as read-only reference material.
3. Keep results and model artifacts out of git history (already covered by [.gitignore](.gitignore)).

## Notes For External Readers

- If you are evaluating the thesis codebase, begin with [ets_marl_happo_current/README.md](ets_marl_happo_current/README.md).
- Legacy folders are retained for reproducibility and historical comparison.
- Folder names intentionally mark active vs archived code to reduce ambiguity.
