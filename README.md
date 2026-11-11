# Moving Average Crossover Trading Bot (Alpaca paper trading)

A Python bot that watches ~1,000 US stocks and trades a simple moving average
crossover strategy on an **Alpaca paper trading account** (simulated money only;
the client is hard-coded to `paper=True`).

## Strategy

For each stock in `config.json` the bot:

1. Downloads the last 100 days of daily closes from Yahoo Finance (`yfinance`).
2. Computes the 20-day and 50-day simple moving averages.
3. Compares the last two completed days:
   - 20-day moves from at/below to above the 50-day: **buy**
   - 20-day moves from at/above to below the 50-day: **sell** (only shares already held; never shorts)
4. Submits market orders to Alpaca, sells first, then buys.

Rules applied before buying:

- `max_share_price`: skip stocks whose last close is above this (default $2,000).
- `max_margin`: skip buys that would push borrowing past this (default $5,000).
- Signals from out-of-date price data, and stocks too new for a 50-day average, are skipped.

## Stock lists

`lists/nasdaq_top500.csv` and `lists/nyse_top500.csv` hold the top 500 common
stocks on each exchange by market cap (one share class per company), built from
the NASDAQ screener and cross-checked against the NASDAQ Trader symbol directory,
Alpaca's asset list, and yfinance. `config.json` contains these plus a few
hand-picked symbols.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then add your Alpaca PAPER API key and secret
```

## Usage

```bash
.venv/bin/python bot.py --dry-run            # signals only, no Alpaca connection
.venv/bin/python bot.py --preview --report   # what the next run would do, no orders
.venv/bin/python bot.py --report             # place orders, then print a summary
```

Each live or preview run writes a full log to `logs/` (kept 90 days) and appends
to a rolling `bot.log`. Neither is committed.

## Disclaimer

For learning and paper trading only. Not financial advice.
