# Journal Evidence Build — v0.3 Exploratory Phase

## Target journal

**Primary target:** IEEE Transactions on Power Systems (TPWRS)

Publication-cost constraint:

- submit via the **Traditional** route;
- no Open Access APC;
- first submission must remain at or below **10 IEEE pages**;
- final accepted paper should remain at or below **12 published pages** to avoid mandatory overlength charges.

The journal manuscript is not the January-2024 data-audit preprint. The journal target is the larger forecasting study:

> **Cross-Task Transfer in Türkiye's Electricity System: Multi-Horizon Forecasting and Calibrated Uncertainty from EPİAŞ Data**

## Scientific boundary

This v0.3 branch is **exploratory only**.

The confirmatory holdout is still:

`UNDEFINED_AND_NOT_REQUESTED`

The exploratory acquisition hard-stops before `2026-01-01T00:00:00+03:00`. The default historical availability scope is 2021-01-01 through 2025-12-31. That interval may be narrowed later if source availability or schema changes require it, but no 2026 data are requested by the v0.3 acquisition runner.

## Why v0.3 is needed

The current repository already contains:

- authenticated EPİAŞ ingestion;
- a provenance-preserved January-2024 data audit;
- persistence and seasonal-naive helper functions;
- a compact shared Transformer with Gaussian mean/scale heads;
- chronological splitting;
- training-only scaler reuse in the exploratory research runner;
- point metrics and split-conformal utilities.

However, the current exploratory runner is not yet journal evidence because it:

1. evaluates only the shared Transformer;
2. reports one overall error across nine outputs with different physical units;
3. does not produce per-target × per-horizon tables;
4. does not persist aligned per-origin predictions for paired inference;
5. does not benchmark the 24-hour and 168-hour seasonal baselines in one journal table;
6. does not yet implement the fair isolated-task learned comparator;
7. does not perform a compute-budget audit;
8. does not perform regime analysis;
9. does not run a frozen multi-seed experiment;
10. is explicitly exploratory and must not be confused with a confirmatory test.

## Phase gates

### Gate A — Historical availability audit

Acquire and preserve exploratory EPİAŞ history for 2021–2025 in monthly chunks.

Required outputs:

- raw JSON per month and dataset, kept local;
- processed monthly aligned CSVs, kept local;
- SHA-256 manifest;
- monthly coverage table;
- full-period aligned exploratory CSV;
- no 2026 request.

**Pass condition:** identify a contiguous multi-season historical interval with sufficient complete hourly coverage to support exploratory model development.

### Gate B — Strong temporal baselines

Run, per target and per horizon:

- persistence;
- 24-hour seasonal naive;
- 168-hour seasonal naive.

Required metrics:

- MAE;
- RMSE;
- sMAPE;
- skill relative to the strongest temporal baseline where appropriate.

Metrics must be reported separately for:

- consumption @ +1 h, +6 h, +24 h;
- renewable generation @ +1 h, +6 h, +24 h;
- MCP/PTF @ +1 h, +6 h, +24 h.

No cross-unit averaging is allowed for the primary tables.

### Gate C — Learned isolated-task comparator

Implement a task-specific sequence model with architecture and capacity chosen to make the joint-vs-isolated comparison fair.

The preferred first comparator is:

- same temporal encoder family;
- one model per target;
- three horizons per target;
- same training window;
- same training-only preprocessing;
- same exploratory origins;
- same seed set.

### Gate D — Shared Transformer

Evaluate the shared multi-task Transformer on the exact same forecast origins.

Primary exploratory estimand:

`MAE(shared) - MAE(strongest isolated comparator)`

reported per target and horizon.

### Gate E — Uncertainty

For learned models:

- native Gaussian scale diagnostics;
- split-conformal calibration using the validation/calibration segment only;
- empirical interval coverage;
- mean interval width;
- no holdout-derived calibration threshold.

### Gate F — Regime analysis

Training-derived regime thresholds only.

Candidate regimes:

- renewable-share quantiles;
- high / normal / low demand;
- absolute one-hour demand ramps;
- MCP/PTF extreme-price indicator.

The objective is to identify where transfer changes sign, not merely where average error is lowest.

### Gate G — Compute audit

Record:

- training time per model and seed;
- inference time;
- number of trainable parameters;
- peak storage/checkpoint footprint where available;
- total projected confirmatory cost.

The final protocol must be reduced before preregistration if the measured no-cost compute envelope is insufficient.

### Gate H — Protocol freeze

Only after exploratory evidence is complete, freeze:

- primary target;
- primary horizon;
- comparator set;
- model hyperparameters;
- seed list;
- metrics;
- missing-data policy;
- clipping/exclusion rules;
- regime thresholds;
- calibration method;
- paired inference method;
- confirmatory holdout definition.

Then create a **separate OSF confirmatory preregistration**.

## Planned journal tables

1. Dataset availability and integrity by year/month.
2. Baseline forecasting metrics by target and horizon.
3. Shared vs isolated learned models by target and horizon.
4. Paired transfer effect estimates with uncertainty intervals.
5. Calibration coverage and interval width.
6. Regime-specific transfer effects.
7. Compute budget and model size.

## Planned journal figures

1. Study design and evidence firewall.
2. Multi-year target time series with exploratory-only boundary.
3. Shared-vs-isolated skill by target/horizon.
4. Transfer benefit by regime.
5. Calibration reliability / empirical coverage.
6. Compute-vs-skill tradeoff.

## Negative-result policy

A null or negative transfer result remains publishable evidence if the experiment is controlled and the preregistered confirmatory analysis is preserved. The primary target, horizon, comparator, or seed policy must not be changed after viewing confirmatory outcomes.
