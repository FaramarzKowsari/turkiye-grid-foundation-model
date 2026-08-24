# Data Contract

## Scope

The empirical study is designed around official EPİAŞ Transparency Platform electricity data. This repository stores code and provenance instructions; it does not automatically redistribute raw third-party records.

## Implemented source endpoints

| Internal name | Official REST path | Frequency | Intended use |
|---|---|---:|---|
| `consumption` | `/v1/consumption/data/realtime-consumption` | hourly | load target and historical context |
| `generation` | `/v1/generation/data/realtime-generation` | hourly | source mix, renewable target, regime features |
| `mcp` | `/v1/markets/dam/data/mcp` | hourly | day-ahead price target |

Base service: `https://seffaflik.epias.com.tr/electricity-service`.

Authentication uses a Ticket Granting Ticket (TGT) obtained from `https://cas.epias.com.tr/cas/v1/tickets`. Credentials must be provided through environment variables and must never be committed.

## Canonical derived variables

- `consumption_mwh`: EPİAŞ real-time consumption.
- `renewable_mwh`: sum of available `sun`, `wind`, `river`, `dammedHydro`, `geothermal`, and `biomass` generation fields.
- `total_generation_mwh`: reported total generation.
- `renewable_share`: `renewable_mwh / total_generation_mwh` when denominator is positive.
- `mcp_tl_mwh`: day-ahead market clearing price in TRY/MWh.

Thermal context fields currently retained: `naturalGas`, `importCoal`, `lignite`.

## Time handling

EPİAŞ request/response dates are expected to carry an explicit offset. Raw timestamps are parsed to UTC internally. Calendar features are constructed in the `Europe/Istanbul` timezone.

## Data audit required before confirmatory freeze

The data-audit release must report:

1. first and last usable timestamp for each endpoint;
2. duplicate timestamps;
3. missing hours and missing values by field;
4. schema changes across years;
5. physically implausible or obviously malformed values;
6. overlap available to all three primary targets;
7. API retrieval failures and successful retries;
8. cryptographic hashes for preserved raw snapshots;
9. exact exclusion rules;
10. license/terms review for the chosen research snapshot.

No confirmatory period should be chosen until this audit is complete.

## Leakage rule

Forward filling across the train/validation/holdout boundary is forbidden. Any imputation rule used in the final analysis must operate with information available at the forecast origin and must be fitted or selected without inspecting confirmatory outcomes.
