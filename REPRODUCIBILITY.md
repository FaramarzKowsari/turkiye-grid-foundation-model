# Reproducibility Contract

This file states the rules intended to separate software development, exploratory analysis and future confirmatory inference.

## Phase A — software validation

Synthetic data may be used to test schemas, tensor shapes, training loops and metrics. Results from synthetic data have no empirical interpretation about Türkiye's electricity system.

## Phase B — data audit

Real EPİAŞ data may be inspected for availability, missingness, schema consistency and runtime planning. Summary plots may be used to design quality-control rules. This phase is exploratory and must not be presented as confirmatory evidence.

## Phase C — exploratory model selection

Candidate architectures, learning rates, context lengths and baseline implementations may be compared on training and validation periods only. Every model retained for confirmatory evaluation must have a frozen configuration.

## Phase D — confirmatory freeze

Before the holdout is evaluated, preserve:

- raw-data manifest and hashes;
- processed-data hash;
- split timestamps;
- feature list;
- target definitions;
- exclusion rules;
- model configs;
- seeds;
- primary and secondary hypotheses;
- statistical tests;
- multiplicity correction;
- runtime budget;
- exact software commit.

An OSF registration may be added at this point. Until such a registration exists, the repository must not call the protocol “preregistered.”

## Phase E — confirmatory run

The holdout is evaluated once under the frozen plan. Technical reruns are permitted only for documented execution failures that do not reveal or alter scientific results. Any deviation must be logged.

## Randomness

All experiment seeds must be explicit. Deterministic settings should be enabled where practical; nondeterministic kernels must be documented if used.

## Reporting

Report all frozen primary analyses. A non-significant or adverse result is retained. Exploratory analyses performed after holdout inspection must be labeled exploratory.

## Compute budget

Before freeze, benchmark total expected cost across every model × seed × horizon. If the projected compute exceeds the available budget, reduce the experiment *before* confirmatory evaluation rather than silently changing the plan later.
