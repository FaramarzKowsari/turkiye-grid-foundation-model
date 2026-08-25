# v0.2a.1 Corrective Generation Audit — January 2024

**Status:** exploratory corrective audit only; **not confirmatory**.  
**New EPİAŞ requests:** **0**  
**Confirmatory holdout:** `UNDEFINED_AND_NOT_REQUESTED`

## Why this correction exists
The first audit had complete coverage, but its source-total closure diagnostic excluded `importExport` although that field is present in the EPİAŞ generation schema. The metadata also showed two negative `sun` observations.

This correction reuses the exact local January 2024 generation snapshot and tests alternative `importExport` accounting conventions without assuming the sign in advance.

## Snapshot
SHA-256: `63afec4aed0c73bef2b374ae356e3e1914d85be23f167a90b7db8682fb7ff344`

## Closure comparison
| Convention | N | Mean residual | MAE residual | P95 abs residual | Max abs residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| `excluding_importExport` | 744 | -88.345793 | 121.439449 | 274.696500 | 380.370000 |
| `adding_importExport` | 744 | -0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `subtracting_importExport` | 744 | -176.691586 | 242.878898 | 549.393000 | 760.740000 |

Lowest internal MAE: **`adding_importExport`**

## Negative solar diagnostic
- Count: **2**
- Minimum sun value: **-0.01 MWh**
- Total absolute magnitude of negative observations: **0.02 MWh**
- Timestamps: `2024-01-04T18:00:00+03:00, 2024-01-13T18:00:00+03:00`

No value was removed or clipped.

## Scientific boundary
This is a data-quality correction only. It makes no forecasting, causal, or confirmatory claim and touches no future holdout.
