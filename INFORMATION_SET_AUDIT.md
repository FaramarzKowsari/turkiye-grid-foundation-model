# Operational Information-Set Audit

## EPİAŞ Day-Ahead Market timing

EPİAŞ describes the normal daily process as follows:

- next-day bids are submitted by 12:30;
- bid validation occurs from 12:30 to 13:00;
- optimization occurs from 13:00 to 13:30;
- objections are handled afterward;
- final next-day prices and matches are announced at 14:00.

Source:
https://www.epias.com.tr/gun-oncesi-piyasasi/surecler/

## Consequence for MCP/PTF evaluation

A delivery-hour MCP/PTF value is a day-ahead market outcome, not a value first
revealed at the delivery hour.

Therefore:

1. MCP +1 h and +6 h results remain useful as temporal-representation
   benchmarks, but they are not labeled operational price forecasts.
2. MCP +24 h is evaluated on all exploratory origins for representation
   comparability.
3. Two information-set-aware MCP +24 h subsets are additionally reported:
   - pre-publication: forecast issue local hour < 14;
   - pre-gate proxy: forecast issue local hour <= 12.
4. No target values beyond the forecast origin are passed into model inputs.

## Historical-timing limitation

The current official normal-process schedule is used as the operational
reference. Before confirmatory preregistration, the project must verify whether
material schedule changes occurred during 2021–2025 and document any such
changes. This exploratory audit does not silently assume historical invariance.

## Scientific boundary

This is an exploratory information-set audit.

Confirmatory holdout: `UNDEFINED_AND_NOT_REQUESTED`.
