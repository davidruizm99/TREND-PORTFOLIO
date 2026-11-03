# Prop firm (FTMO 2-Step Swing) work — rotation portfolio + trend-following CFD portfolio

Rules simulated: +10% / +5% targets, 5% daily loss (from the day's reference balance, floating included), 10% static max loss,
minimum 4 trading days, no time limit. Funded account: payouts every ~30 days at 80%. Rules come from secondary 2026 sources
and still need to be confirmed on ftmo.com.

## 1. `notebooks/rotativo_prop_simulador_v4.ipynb` — the rotation portfolio (main work)
Takes the daily net-return series of the RSI rotation (TOP3 fractional, GLD refuge, 0.8 correlation filter) and applies a
portfolio-level sizing layer **without touching selection or timing**:
- **A**: fixed scale k
- **B**: vol-target, k = tau / EWMA(30d) vol, capped at 1
- **C**: B + drawdown brake (size shrinks toward the -10% limit) + daily-loss cushion (2.5 sigma of a bad day)

Inputs: `data/rotativo_daily_9sleeves.csv` (9 sleeves that exist as FTMO CFDs: BTC, ETH, SOL, QQQ, SPY, TQQQ, GLD, NVDA, AAPL)
and `data/rotativo_daily_10sleeves_AMD.csv` (adds AMD). Both include rotation costs (10 bp on turnover).
Copy the CSV you want to the notebook folder as `rotativo_daily.csv` and set `COSTE_EXTRA_ANUAL = 0.04` (CFD swing, assumption).
If the CSV is missing the notebook falls back to synthetic data and prints a warning: those numbers are worthless.

Assumptions to check: swap 4%/yr on exposure, worst intraday point = 1.5x the close-to-close loss, no OHLC for the portfolio,
no opening gaps for stock CFDs, no 1:1 margin limit for stocks/crypto in Swing accounts.

## 2. `montecarlo/montecarlo.py` — whole process (phase 1 -> phase 2 -> 12 months funded)
Stationary block bootstrap (~30-day blocks), 20,000 paths per scenario, with the historical edge kept at 100%, 50% and 0%
(mean daily return shifted). Plans: C tau 0.15 / C tau 0.10 / unscaled k=1 / fixed k 0.35, funded at fixed k 0.35.
Outputs in `results_charts/`.

## 3. `notebooks/prop_portfolio_simulador_v3.ipynb` — earlier trend-following CFD portfolio (Carver-style)
12 CFDs, Rick-style vs IDM sizing, long-only vs long-short, 8 scenarios. Results were slow (median 350-500 business days) and
IDM sizing blew up, which is why the work moved to the rotation portfolio. Kept for reference.

## Headline results (rotation, 9 sleeves, historical start dates, with rotation costs)
| Method | Pass both phases | Median days |
|---|---|---|
| Unscaled (k=1) | 46% | 80 |
| C tau 0.15 | 73% | 119 |
| C tau 0.10 | 82% | 194 |
| Fixed k 0.35 | 89% | 312 |

Monte Carlo, plan C tau 0.15 -> k 0.35: 62% / 39% / 14% to get funded with 100% / 50% / 0% of the historical edge.
The historical result leans heavily on 2021 (+284%).
