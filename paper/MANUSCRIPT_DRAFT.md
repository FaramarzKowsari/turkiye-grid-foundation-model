# Cross-Task Transfer in Türkiye's Electricity System

## Reproducible Multi-Horizon Forecasting of Load, Renewable Generation and Day-Ahead Price

**Faramarz Kowsari**

> **Manuscript status:** methods scaffold only. Results placeholders are intentionally not populated until a real-data confirmatory analysis exists.

## Abstract

Electricity demand, renewable production and market price interact within the same physical and market system, yet they are commonly forecast as separate tasks. This study is designed to test whether a shared temporal representation can transfer useful information across these signals in Türkiye's electricity system without sacrificing out-of-time robustness or uncertainty calibration. We construct a reproducible pipeline around official EPİAŞ Transparency Platform hourly data and compare a compact multi-task Transformer with strong temporal and task-specific baselines at multiple horizons. The protocol emphasizes chronological evaluation, training-only preprocessing, paired inference, explicit uncertainty assessment and preservation of negative findings. **Empirical results will be inserted only after the data audit, model-selection phase and frozen holdout analysis are completed.**

## 1. Introduction

The motivation is not that every electricity series should share one model. The stronger question is whether common temporal structure is sufficiently stable to create transferable representations across demand, renewable output and market price, and whether that transfer survives regime change. Türkiye offers an informative setting because consumption, generation mix and day-ahead market prices are observable through a common official transparency infrastructure.

Three tensions motivate the study. First, variables are coupled through the power system but respond differently to weather, operational constraints and market conditions. Second, apparent gains under random validation can disappear under chronological shift. Third, a model that reduces mean error may still be unsafe as a research conclusion if its uncertainty is badly calibrated during unusual periods.

The study therefore treats cross-task transfer as an empirical hypothesis rather than a design assumption.

## 2. Data

**To be completed after the frozen data audit.** This section will report exact source endpoints, retrieval dates, temporal coverage, exclusions, missingness, schema changes and source hashes. Raw third-party data will not be represented as authored by this repository.

## 3. Methods

### 3.1 Prediction problem

A one-week hourly context is mapped to forecasts of consumption, renewable generation and market-clearing price at +1, +6 and +24 hours. Final horizons and the primary estimand remain subject to the pre-confirmatory freeze described in `RESEARCH_PROTOCOL.md`.

### 3.2 Shared model

The initial shared model uses a linear input projection, sinusoidal positional encoding and Transformer encoder. The final context representation feeds parallel parameter heads producing a mean and positive scale for each target-horizon output.

### 3.3 Baselines

The study will include persistence and seasonal-naive baselines and at least one frozen task-specific learned comparator. Comparator choice will be finalized using training and validation periods only.

### 3.4 Uncertainty

Native Gaussian scale predictions provide a model-based uncertainty estimate. A separate calibration period can be used for split-conformal residual calibration. Both coverage and interval width will be reported.

### 3.5 Evaluation

The main evaluation is chronological. Forecasts are aligned by origin and compared pairwise. Exact statistical units and multiplicity handling will be frozen before confirmatory inspection.

## 4. Results

**No empirical paper-level results yet.**

Planned tables:

- Table 1 — data coverage and missingness;
- Table 2 — model complexity and compute budget;
- Table 3 — primary holdout errors by target/horizon;
- Table 4 — paired effect estimates and uncertainty;
- Table 5 — prediction-interval calibration;
- Table 6 — prespecified regime analysis.

## 5. Discussion

To be written after results. The discussion will explicitly distinguish evidence of positive transfer, negative transfer, equivalence-like uncertainty, and inconclusive comparisons.

## 6. Reproducibility

The paper release will point to a frozen repository commit, exact configurations, source manifests, predictions and statistical outputs. If an OSF preregistration is created before confirmatory execution, its identifier will be added here without retroactively describing earlier exploratory work as preregistered.
