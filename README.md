# Volatility forecasting: EWMA, GARCH and a small TCN

Daily market data, chronological evaluation, and a deliberately small neural model.
The target is next-day squared return; this is a forecasting study, not a trading claim.

```bash
../NAS/venv/bin/python run.py --tickers SPY QQQ --epochs 8
../NAS/venv/bin/python -m unittest -v
```

Data comes from Stooq CSV downloads and is cached with hashes. The report records
the download URLs, dates and metrics. The GARCH implementation is a simple
Gaussian maximum-likelihood fit; no `arch` dependency is needed.
