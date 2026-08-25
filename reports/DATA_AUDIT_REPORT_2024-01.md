# First Empirical EPİAŞ Data Audit — January 2024

**Audit ID:** `2024-01-v0.2a`  
**Scientific status:** Exploratory data-quality audit only; **not confirmatory**.  
**Retrieval time (UTC):** `2026-08-25T13:44:01.208005+00:00`

## 1. Evidence firewall

This run requested **only** `2024-01-01T00:00:00+03:00` through `2024-01-31T23:00:00+03:00` (744 expected hourly timestamps). No final confirmatory holdout has been defined, and this runner did not request observations outside the locked January 2024 interval.

## 2. Official source contract

- CAS/TGT: `https://giris.epias.com.tr/cas/v1/tickets`
- API base: `https://seffaflik.epias.com.tr/electricity-service`
- Consumption: `/v1/consumption/data/realtime-consumption`
- Generation: `/v1/generation/data/realtime-generation`
- MCP/PTF: `/v1/markets/dam/data/mcp`

Raw source records remain local and gitignored; aggregate audit outputs and hashes are preserved.

## 3. Retrieval and hourly coverage

| Dataset | Raw rows | Unique hours | Duplicates | Missing expected | Coverage | Key nulls | Outside scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| consumption | 744 | 744 | 0 | 0 | 100.000% | 0 | 0 |
| generation | 744 | 744 | 0 | 0 | 100.000% | 0 | 0 |
| mcp | 744 | 744 | 0 | 0 | 100.000% | 0 | 0 |

Aligned union rows: **744**  
Complete consumption + renewables + PTF rows: **744/744 (100.000%)**

## 4. Schema observations
- **consumption:** `consumption, date, time`; timezone suffixes `['+03:00']`; maximum gap **1.0 h**.
- **generation:** `asphaltiteCoal, biomass, blackCoal, dammedHydro, date, fueloil, geothermal, hour, importCoal, importExport, lignite, lng, naphta, naturalGas, river, sun, total, wasteheat, wind`; timezone suffixes `['+03:00']`; maximum gap **1.0 h**.
- **mcp:** `date, hour, price, priceEur, priceUsd`; timezone suffixes `['+03:00']`; maximum gap **1.0 h**.

## 5. Descriptive numeric audit

| Variable | N | Mean | SD | Min | P05 | Median | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| consumption_mwh | 744 | 38,873.309 | 5,767.913 | 24,610.500 | 30,137.229 | 39,592.630 | 46,900.831 | 48,394.390 |
| renewable_mwh | 744 | 19,081.383 | 3,440.859 | 10,398.190 | 12,745.366 | 19,116.315 | 24,406.332 | 26,093.290 |
| total_generation_mwh | 744 | 37,898.719 | 5,400.053 | 24,536.760 | 29,426.151 | 38,574.885 | 45,589.784 | 47,992.460 |
| renewable_share | 744 | 0.505 | 0.068 | 0.350 | 0.396 | 0.503 | 0.625 | 0.673 |
| mcp_tl_mwh | 744 | 1,942.905 | 626.953 | 102.260 | 900.000 | 2,180.980 | 2,689.000 | 2,700.000 |

These are data-quality descriptions, not forecasting metrics.

## 6. Generation-source consistency

Available source fields: `fueloil, blackCoal, lignite, geothermal, naturalGas, river, dammedHydro, lng, biomass, naphta, importCoal, asphaltiteCoal, wind, sun, wasteheat`

- Rows checked: **744**
- Mean residual (reported total − listed-source sum): **-88.346 MWh**
- Mean absolute residual: **121.439 MWh**
- Maximum absolute residual: **380.370 MWh**

Residuals are diagnostics, not automatic evidence of bad data; EPİAŞ definitions must be considered.

## 7. Exploratory anomaly flags — no deletion performed

- Non-positive consumption: **0**
- Negative renewable aggregate: **0**
- Renewable share outside [0,1]: **0**
- Negative PTF: **0**
- Zero PTF: **0**

| Variable | N | IQR lower fence | IQR upper fence | Flagged |
| --- | --- | --- | --- | --- |
| consumption_mwh | 744 | 18,419.056 | 59,057.206 | 0 |
| renewable_mwh | 744 | 9,531.396 | 29,262.186 | 0 |
| mcp_tl_mwh | 744 | -248.331 | 4,013.859 | 0 |

## 8. Descriptive correlation matrix

| Variable | consumption_mwh | renewable_mwh | mcp_tl_mwh |
| --- | --- | --- | --- |
| consumption_mwh | 1.0000 | 0.6755 | 0.6778 |
| renewable_mwh | 0.6755 | 1.0000 | 0.1896 |
| mcp_tl_mwh | 0.6778 | 0.1896 | 1.0000 |

Exploratory only; not causal and not a confirmatory hypothesis.

## 9. Cryptographic provenance

| Dataset | Local raw snapshot | SHA-256 |
| --- | --- | --- |
| consumption | consumption.json | 49a528ee6ffbd3c1a2717cddc7bbbb311a5c4a52a6a0379ea05dc1f2fc9ddc18 |
| generation | generation.json | 63afec4aed0c73bef2b374ae356e3e1914d85be23f167a90b7db8682fb7ff344 |
| mcp | mcp.json | 1853a77aa7ef96d7ecbc7bfef0896e151954251e2dae54318b06043fde452513 |

Processed aligned CSV SHA-256: `d0cc98d066fafcd71868b876a2b44dc292a5c597f6c0df3661a0e4620741b512`

Machine-readable summary: `metadata/audits/2024-01-v0.2a-summary.json`

## 10. First scientific interpretation

This checkpoint asks a narrower question than forecasting: **is the selected EPİAŞ data layer internally usable enough to justify a larger exploratory modelling stage?** The decision is based on timestamp coverage, key-field completeness, schema stability, cross-dataset alignment and unexplained structural inconsistencies—not on whether a Transformer appears accurate.

### Next allowed actions

1. Investigate any missing timestamps, duplicates, numeric nulls or source-total residuals.
2. If structurally clean enough, expand only the exploratory/training-side availability audit under a separately committed scope extension.
3. Benchmark strong baselines and runtime before choosing final seeds/model families.
4. Keep the confirmatory holdout undefined and unseen until data and compute audits are complete.
5. Freeze hypotheses, exclusions, metrics, seeds and eventual holdout before confirmatory evaluation.

---
Generated automatically by Türkiye Grid Foundation Model v0.2a.
