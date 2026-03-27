## Plan: Budget Hardening and Realistic Bid Path

Recommended approach: keep your intended flexibility by allowing overspend up to +15%, but make anything above that infeasible through a true hard cap in the decision path (not only post-hoc penalties). In parallel, align heuristic penalty-rate usage to inflation-consistent values and enforce realistic year-1 bot bids near 80 EUR/t while softly guiding RL toward 180-230 EUR/t by year 10.

**Steps**
1. Phase 1 - Lock policy parameters and scope.
2. Define budget regime as soft zone up to +15% overspend, then hard ceiling beyond that.
3. Define bid targets as year 1 near 80 EUR/t (bots hard-enforced, RL soft-guided) and year 10 centered in 180-230 EUR/t.
4. Keep all changes inside ets_marl_happo_current only.
5. Phase 2 - Implement budget hardening in company/environment flow (depends on 1).
6. Add explicit soft/hard budget controls to config parsing and internal state.
7. Replace pure post-hoc-only budget behavior with tiered penalty near cap plus true hard-stop investment feasibility check above cap.
8. Preserve independence between annual budget and capex throughput constraints, with clear precedence when both bind.
9. Phase 3 - Bot realism and inflation consistency (depends on 2; parts parallel with 4).
10. Add budget-feasibility checks into heuristic decision logic so bots cannot propose over-hard-cap investments.
11. Calibrate heuristic year-1 pricing logic to remain close to realistic market start levels.
12. Ensure heuristic auction and secondary branches use consistent inflation-effective penalty semantics.
13. Phase 4 - Training glue and fallback cleanup (parallel where non-overlapping).
14. Update training callsites so heuristic auction path receives episode inflation factors.
15. Remove remaining old 100 EUR fallback defaults where they still influence behavior.
16. Phase 5 - RL soft-guidance trajectory tuning (depends on stabilized 2-4).
17. Reconfirm anchor and exploration settings to keep initial RL bids near 80 EUR/t distribution.
18. Add/adjust gentle early-year overbidding discouragement without hard RL price ceilings.
19. Validate progression by year toward 180-230 EUR/t around year 10.
20. Phase 6 - Tests and docs (depends on 2-5).
21. Add unit and integration coverage for budget boundary behavior, hard-cap prevention, inflation-consistent heuristics, and year-conditioned bid realism.
22. Update design documentation with new budget regime rationale and expected market effects.

**Relevant files**
- [ets_marl_happo_current/configs/default.yaml](ets_marl_happo_current/configs/default.yaml)
- [ets_marl_happo_current/src/environment/company.py](ets_marl_happo_current/src/environment/company.py)
- [ets_marl_happo_current/src/environment/ets_environment.py](ets_marl_happo_current/src/environment/ets_environment.py)
- [ets_marl_happo_current/src/agents/heuristic_policy.py](ets_marl_happo_current/src/agents/heuristic_policy.py)
- [ets_marl_happo_current/scripts/train.py](ets_marl_happo_current/scripts/train.py)
- [ets_marl_happo_current/src/agents/q_learning_agent.py](ets_marl_happo_current/src/agents/q_learning_agent.py)
- [ets_marl_happo_current/tests/test_company.py](ets_marl_happo_current/tests/test_company.py)
- [ets_marl_happo_current/tests/test_heuristic.py](ets_marl_happo_current/tests/test_heuristic.py)
- [ets_marl_happo_current/tests/test_environment.py](ets_marl_happo_current/tests/test_environment.py)
- [ets_marl_happo_current/docs/design.md](ets_marl_happo_current/docs/design.md)

**Verification**
1. Run targeted tests for budget, heuristic, and environment behavior in ets_marl_happo_current.
2. Run a short train/eval smoke pass and inspect year logs for no above-hard-cap spend, bots near ~80 EUR/t in year 1, and upward trajectory toward 180-230 EUR/t by year 10.
3. Run a stochastic-seed check to verify inflation-consistent heuristic penalty handling in both auction and secondary branches.
4. Compare pre/post compliance and total-cost behavior to ensure hardening does not create pathological under-compliance.

If this matches your intent, approve and I’ll hand this off for implementation exactly as specified.
