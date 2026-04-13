![Volatility Forecasting banner](assets/banner.svg)

# Volatility Forecasting

A chronological comparison of three next-day volatility forecasting approaches:

- Exponentially weighted moving average
- Gaussian GARCH
- A small temporal convolutional network

The target is next-day squared return. The project evaluates forecasting error and does not convert forecasts into a trading strategy.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --tickers SPY QQQ --epochs 8
python -m unittest -v
```

## Data and evaluation

Daily prices are downloaded from Stooq, cached locally, and recorded with source URLs and hashes. All splits are chronological. Normalization and model fitting use training data only.

The GARCH model uses a direct Gaussian maximum-likelihood implementation, so the project does not require the `arch` package.

## Scope

This is a small forecasting benchmark. Results depend on the selected assets, time period, loss function, and neural training budget. Forecast accuracy alone does not establish economic value.
