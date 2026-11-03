# Rotating trend-following portfolio (long-only, no leverage)

Code requested for review. Three independent parts, all plain Python (pandas/numpy; yfinance only for the real-data checks).

## 1. `rotativo/` — RSI rotation with correlation filter
`python portfolio_rotativo.py` (from `rotativo/`) reads the 11 sleeve CSVs in `data/csv_sleeves/` and prints Sharpe / CAGR / MaxDD per window, with and without a 10 bp rotation cost. Config block at the top of the script (sleeves, TOP_N, correlation filter on/off, threshold, window, lag, refuge, cost).
`rotacion.py` is the single implementation of the rotation (ported 1:1 from my original backtest; 0 differing days out of 3223).
Rule: rank by RSI(60) of the cumulative buy&hold curve (shifted `LAG`), keep RSI > 50; #1 always in; each next slot is the next asset whose max correlation vs the picks is <= 0.8 (120 weekdays of buy&hold returns, data up to the previous weekday); empty slot -> GLD if its sleeve is on and RSI > 50, else cash. Weight 1/TOP_N times the sleeve's own size.

## 2. `tqqq_ensemble/` — 60-system TQQQ ensemble
Each system = 3 EMA crosses + MACD confirmation + optional ATR trailing stop (long/flat). 45 "slow" systems (`frozen_TQQQ.json`) + 15 "new" (list inside `tqqq_motor.py`); size = 0.5*mean(slow on/off) + 0.5*mean(new on/off).
Reproduces my notebook output on real OHLC: 99.95% of 2209 bars identical with `delay=1`. `python diag2.py --json frozen_TQQQ.json --csv ../data/csv_sleeves/TQQQ_10bp_daily.csv` prints the delay grid.

## 3. `sleeves/` — the 10 other sleeve systems
`sleeves.py` (definitions), `indicadores.py`, `sizing.py` (vol sizing), `verificar.py` (`python verificar.py --csv-dir ../data/csv_sleeves --yahoo`): reproduces my grid CSVs 100% on BTC, ETH, SOL, AAPL, NVDA, AMD, GLD, DBC, QQQ, SPY.

## Known caveats
- The 0.8 correlation threshold looks tuned to my data (0.7 gives almost the same Sharpe).
- Rotation turnover is high (~50-55x/year summed |dw|); real costs are not measured yet (paper trading pending).
- Live convention (calculate after close, trade next close) costs a little Sharpe vs the lag-1 backtest.
