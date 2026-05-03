# Research Questions

The questions below frame the thesis. They are reproduced verbatim in the
top-level `README.md`; this document keeps them alongside the other
methodological notes (`data_dictionary.md`, `data_science_analysis.md`,
`reward_function.md`) and points the reader at the artefacts that address
each one.

`docs/design.md` is environment-design-only — research questions are
intentionally **not** mirrored there.

---

## Main RQ

> **How can multi-agent reinforcement learning be used to simulate the EU
> Emissions Trading System, and what does the resulting simulation reveal
> about the system's regulatory mechanisms and the strategic behavior of
> power-sector participants balancing financial and environmental
> objectives?**

The Main RQ is answered cumulatively by the three sub-questions below.
The synthesis lives in the discussion chapter of the thesis.

---

## Sub-RQ 1 — Simulation credibility

> **What environment and algorithm design choices are required to build a
> stable and behaviorally credible simulation of a carbon market?**

Concerns the *artefact*: convergence, cross-seed reproducibility, and
whether emergent market behaviour is qualitatively recognisable as an
EU-ETS-style auction (price band, compliance plateau, no pathological
warning regimes dominating the converged window).

| Where it is addressed | Artefact |
|---|---|
| Default-config training dynamics & seed stability | `notebooks/ets_marl - Default RQ Analysis.ipynb`, §RQ1 |
| Per-seed deep dives (single-run sanity) | `notebooks/ets_marl - Full Run & Analysis.ipynb` |
| Methodology mapping (PELT, plateau detection, fan charts) | `data_science_analysis.md` §C1, §C2 |

---

## Sub-RQ 2 — Regulatory-mechanism sensitivity

> **How sensitive are market outcomes such as price stability, compliance,
> and the pace of decarbonization to the regulatory mechanisms of the EU
> ETS?**

True *sensitivity* requires comparison across regulatory variants
(cap LRF, MSR on/off, penalty level, …). The default-only notebook
therefore restricts itself to a **descriptive** characterisation of how
the regulatory mechanisms operate inside the default config (cap-vs-price
linkage, TNAC / MSR trajectory, penalty incidence, decarbonization
pace). The cross-variant comparison that actually answers RQ2 lives in a
separate experiments notebook (planned, not in scope for the
default-only analysis).

| Where it is addressed | Artefact |
|---|---|
| Within-default regulatory behaviour (descriptive baseline) | `notebooks/ets_marl - Default RQ Analysis.ipynb`, §RQ2 |
| Cross-variant sensitivity (LRF / MSR / penalty sweeps) | `notebooks/ets_marl - Thesis Experiments Analysis.ipynb`; sweep configs under `configs/sweeps/` |
| Methodology mapping (descriptive comparison + cluster shifts + UDBC pathways) | `data_science_analysis.md` §R1, §R2 |

---

## Sub-RQ 3 — Strategic behaviour and aggregation

> **How do financial and environmental objectives shape the strategies of
> power companies in this market, and how do these individual strategies
> aggregate into market-level outcomes?**

Concerns *what the agents do once they have learned*. The default config
runs four archetypes (coal-heavy, gas-dominant, transitioner,
green-leader) crossed with two reward weightings — pure financial
(`[w_cost, w_green] = [1.0, 0.0]`, even agent indices A1/A3/A5/A7) and
balanced ESG (`[0.5, 0.5]`, odd indices A2/A4/A6/A8). That gives us the
financial-vs-environmental contrast directly inside one config.

| Where it is addressed | Artefact |
|---|---|
| Per-archetype bidding, investment, secondary-market role, UDBC pathway, reward decomposition | `notebooks/ets_marl - Default RQ Analysis.ipynb`, §RQ3 |
| Strategy clustering & theory tests (LCOE crossover, archetype heterogeneity) | `data_science_analysis.md` §S1, §S2, §S3 |

---

## Notebook layout summary

* `ets_marl - Default RQ Analysis.ipynb` — **default-config only**, multi-sweep loader; addresses RQ1 in full, RQ2 descriptively, RQ3 in full.
* `ets_marl - Thesis Experiments Analysis.ipynb` — cross-config sweep analysis; the home of the actual RQ2 sensitivity claims.
* `ets_marl - Sweep Analysis.ipynb` — single-sweep operational diagnostics (convergence dashboards, deep-dive picker).
* `ets_marl - Full Run & Analysis.ipynb` — single-seed deep dive on a fully-trained run.
* `ets_marl - Data Science Analysis.ipynb` — methodology home (clustering, regression, change-point detection); maps every method back to a thesis claim per `data_science_analysis.md`.
* `ets_marl - Q-Learning Baseline.ipynb` and `ets_marl - Bots-Only Baseline.ipynb` — credibility floors used in RQ1 cross-algorithm comparisons.
