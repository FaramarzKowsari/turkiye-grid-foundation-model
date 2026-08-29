# Exploratory Learned-Model Protocol — Journal Evidence v0.3

**Status:** pre-result exploratory protocol for learned-model evidence.

**Confirmatory holdout:** `UNDEFINED_AND_NOT_REQUESTED`.

This document is committed before the learned-model results are inspected.

## Data boundary

- EPİAŞ exploratory data: 2021-01-01 through 2025-12-31.
- 43,824 aligned hourly timestamps were established by the preceding data gate.
- No 2026+ data are requested or read by this learned-model runner.
- The confirmatory holdout remains undefined and untouched.

## Forecast construction

- Context: 168 hourly observations.
- Horizons: +1 h, +6 h, +24 h.
- Targets:
  - electricity consumption;
  - renewable generation;
  - MCP/PTF.
- Features are historical only and terminate before the first target timestamp.
- Train/validation/exploratory-evaluation partitions are chronological by rows.
- Target horizons are required to remain inside their partition, preventing
  target overlap across split boundaries.
- Feature and target standardization is fitted on training rows only.

## Learned comparison

The central controlled comparison uses the same Transformer encoder family.

### Isolated models

One model per target, each predicting that target at +1 h, +6 h and +24 h.

### Shared model

One shared temporal encoder predicting all three targets at all three horizons.

All isolated and shared models use:

- the same historical feature set;
- the same context length;
- the same chronological origins;
- the same seed set;
- the same optimization family;
- the same early-stopping rule.

The primary exploratory transfer quantity is:

`Delta MAE = MAE(shared) - MAE(isolated)`

Negative values indicate positive transfer.

## Seeds

Exploratory seeds:

- 2026
- 2027
- 2028

These seeds are exploratory. The final confirmatory seed policy is not yet frozen.

## Point metrics

Reported separately for every target × horizon:

- MAE
- RMSE
- sMAPE

No cross-unit averaging is permitted for the primary evidence tables.

## Uncertainty

- Native Gaussian scale head is retained.
- Split conformal calibration is fitted on validation predictions only.
- Nominal conformal coverage: 90%.
- Report empirical coverage and mean interval width separately for each output.

## Transfer inference

Exploratory paired inference uses:

- identical evaluation origins;
- absolute-error differences;
- 24-hour block bootstrap;
- fixed bootstrap RNG seed 20260828;
- 95% bootstrap intervals.

This is exploratory inference and is not a substitute for the later
confirmatory preregistered procedure.

## Regime analysis

Thresholds are estimated from training rows only.

Candidate regimes:

- renewable share: low / middle / high;
- demand level: low / middle / high;
- absolute one-hour demand ramp: normal / high;
- MCP/PTF level: normal / extreme.

Regime analysis asks where cross-task transfer changes sign.

## Information-set boundary

The official EPİAŞ normal Day-Ahead Market process states that bids for the
following day are submitted until 12:30, optimization is performed afterward,
and final next-day PTF values are announced at 14:00.

Implications for this project:

- MCP +1 h and +6 h delivery-price prediction is not presented as an
  operational forecast, because the relevant day-ahead price is ordinarily
  already public before delivery.
- MCP +24 h is reported both as an all-origin representation benchmark and
  on information-set-aware subsets:
  - issue hour before 14:00 local time (pre-publication);
  - issue hour at or before 12:00 local time (hourly pre-gate-close proxy).
- Consumption and renewable-generation forecasts retain their usual
  rolling-origin interpretation, subject to the limitation that exact
  real-time publication latency is not modeled in this exploratory phase.

Official process reference:
https://www.epias.com.tr/gun-oncesi-piyasasi/surecler/

## Compute audit

Record for every run:

- device;
- trainable parameters;
- epochs completed;
- best epoch;
- training seconds;
- inference seconds;
- checkpoint size.

## Stop rule

This automated phase stops after producing a freeze-ready evidence report.

It MUST NOT:

- define the confirmatory holdout;
- request 2026+ data;
- submit an OSF confirmatory preregistration;
- run the confirmatory analysis.

Those actions occur only after exploratory evidence, modern-baseline review,
compute-budget review and protocol freeze.
