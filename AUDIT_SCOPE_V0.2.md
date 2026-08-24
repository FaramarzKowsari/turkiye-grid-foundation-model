# v0.2a Real-Data Audit Scope Lock

**Status:** exploratory data audit only — not confirmatory.

This file is intentionally committed **before** the first empirical EPİAŞ acquisition.

## Locked first audit window

- Start: `2024-01-01T00:00:00+03:00`
- End: `2024-01-31T23:00:00+03:00`
- Expected hourly timestamps: **744**
- Datasets: real-time consumption, source-level real-time generation, day-ahead MCP/PTF.

The audit runner does not expose command-line date overrides. Its first empirical run is hard-limited to this window.

## Evidence firewall

No final confirmatory holdout dates have been selected. The v0.2a runner requests no observations outside the January 2024 audit window. The purpose is to inspect data availability, schema behavior, missingness, duplicate timestamps, source consistency, numeric ranges, alignment and provenance before any confirmatory protocol is frozen.

No model-comparison hypothesis is tested in this phase. Descriptive correlations, outlier flags and other summaries are explicitly exploratory and cannot be promoted to confirmatory findings later without an independently frozen protocol.

## Raw-data policy

Raw and processed EPİAŞ records remain excluded from Git by `.gitignore`. The repository may preserve acquisition metadata, aggregate data-quality summaries, schema field names, SHA-256 hashes of local source snapshots, and the generated scientific audit report.

Credentials are never written to a file by the audit runner.
