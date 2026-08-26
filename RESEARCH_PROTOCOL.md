# Research Protocol — Draft Before Preregistration

**Status:** confirmatory design draft; lifecycle state registered on OSF, but the confirmatory protocol is not yet preregistered.  
**Repository version:** v0.2.1 empirical open-science milestone.  
**Date:** 26 August 2026.


## Open-science lifecycle record

- OSF Registration DOI: https://doi.org/10.17605/OSF.IO/FMCYQ
- Associated OSF project: https://osf.io/tz5pw
- Zenodo v0.2.1 DOI: https://doi.org/10.5281/zenodo.22102736
- Zenodo Concept DOI: https://doi.org/10.5281/zenodo.22102735

This lifecycle registration is **not** the final confirmatory preregistration. The exact confirmatory target/horizon, model set, seed policy, inference procedure and holdout remain to be frozen before confirmatory evaluation.

## Working paper title

**Cross-Task Transfer in Türkiye's Electricity System: Reproducible Multi-Horizon Forecasting of Load, Renewable Generation and Day-Ahead Price**

## Central question

Does a shared temporal encoder trained jointly on electricity consumption, renewable generation and market-clearing price improve out-of-time predictive performance and calibrated uncertainty relative to strong isolated-task baselines?

## Primary estimand concept

The intended primary estimand is the paired difference in forecast error between the frozen joint model and the strongest frozen non-joint comparator on aligned holdout forecast origins. The exact target/horizon chosen as primary will be fixed only after data availability and exploratory validation are complete.

## Candidate hypotheses

These are **candidate hypotheses**, not registered hypotheses.

- **H1:** joint multi-task learning reduces out-of-time MAE for at least one prespecified primary target/horizon relative to the strongest frozen comparator.
- **H2:** transfer benefit differs by horizon.
- **H3:** transfer benefit is associated with prespecified system regimes such as renewable share, demand ramps and price extremes.
- **H4:** conformalized intervals achieve coverage close to the prespecified nominal level on the untouched holdout.

The final protocol should avoid claiming all of these as co-primary. A single primary test with a clearly defined secondary family is preferable.

## Data

Primary source: EPİAŞ Transparency Platform REST API.

Implemented series:

1. hourly real-time consumption;
2. hourly source-level real-time generation;
3. hourly day-ahead market clearing price.

Renewable generation is the sum of available solar, wind, run-of-river, dammed hydro, geothermal and biomass fields. This definition must be frozen before confirmatory analysis and reviewed against source semantics.

## Prediction task

Context: previous 168 complete hourly observations.  
Candidate horizons: +1 h, +6 h, +24 h.  
Candidate targets: consumption MWh, renewable MWh, MCP/PTF TRY/MWh.

## Comparators

Minimum comparator set:

- persistence;
- 24-hour seasonal naive;
- 168-hour seasonal naive;
- a task-specific compact neural sequence model;
- the shared multi-task Transformer.

Additional sophisticated baselines may be added during the exploratory phase only if the compute budget remains feasible.

## Splitting

Primary inference must use chronological separation. `configs/research.yaml` currently contains provisional 70/15/15 fractions only as an engineering default. Exact timestamps will replace fractions after the availability audit.

No random shuffling across calendar time is allowed for the final train/validation/holdout definition.

## Preprocessing

- timestamps normalized and deduplicated;
- calendar features computed in Europe/Istanbul;
- scalers fitted on training data only;
- missing-data policy frozen before confirmatory evaluation;
- no target leakage from future observations;
- no holdout-derived clipping thresholds.

## Metrics

Candidate point metrics: MAE, RMSE, sMAPE.  
Candidate uncertainty metrics: empirical coverage and interval width at a frozen nominal coverage.

## Statistical plan to freeze later

Preferred approach:

1. aggregate errors into a prespecified independent-enough unit such as forecast-origin day;
2. compute paired differences for the primary comparison;
3. use a paired bootstrap or other justified paired test;
4. report an effect estimate and interval, not only a p-value;
5. correct the secondary family for multiplicity if multiple horizons/targets are inferentially tested.

The final choice must be made before viewing holdout comparisons.

## Regime analysis

Candidate precomputable regimes:

- renewable-share quantiles;
- high/normal/low demand;
- absolute one-hour demand ramp;
- MCP/PTF extreme-price indicator.

Regime thresholds must come from training data, not the holdout distribution.

## Negative-result policy

If the shared model does not outperform the comparator, the result remains part of the study. The repository must not switch the primary horizon, replace the comparator or retune on holdout results.

## Required artifacts for paper release

- exact source/data manifest;
- frozen configuration files;
- model checkpoints where redistributable;
- per-origin predictions;
- metric tables;
- statistical-analysis outputs;
- environment/package lock or frozen dependency report;
- SHA-256 manifest;
- manuscript and supplementary methods.
