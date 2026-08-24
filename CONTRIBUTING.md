# Contributing

Contributions are welcome when they preserve the separation between software validation, exploratory analysis and confirmatory evidence.

## Pull requests

- include tests for new data transformations or models;
- do not commit credentials or raw third-party datasets;
- do not label synthetic results as empirical evidence;
- keep chronological evaluation leakage-resistant;
- document any new dependency and why it is needed;
- update the research protocol before changing a frozen confirmatory definition.

Run before opening a PR:

```bash
ruff check src tests scripts
pytest
python scripts/train_smoke.py
```
