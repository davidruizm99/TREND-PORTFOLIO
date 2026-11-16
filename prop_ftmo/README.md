# FTMO 2-Step Swing portfolio (prop-firm version of the rotating trend-following portfolio)

Long-only rotation over 10 sleeves, scaled by a volatility-target overlay so the account can pass the FTMO 2-Step Swing challenge (CFDs, 100k account assumed).
**Status (2026-11-16):** research finished; forward test on a demo account starts now; no challenge has been bought.

## Layout
| Folder | Content |
|---|---|
| `data/csv_sleeves/` | Daily CSV of each sleeve used (grid notebook export): BTC, ETH, SOL, QQQ (US100), SPY (US500), GLD (XAU), NVDA, AAPL, AMD, JNJ |
| `data/ftmo_simbolos.csv` | Real FTMO contract conditions exported from the demo account (swap, spread, leverage) |
| `notebooks/` | `ftmo_rotativo.py` (engine + research modes), `senal_diaria.py` (daily signal for the forward test), `sleeves_motor.py` (sleeve rules computed live from Yahoo prices, copied 1:1 from the grid code) |
| `montecarlo/` | `mc_prop.py` (Monte Carlo of the 2 phases + funded account), `ftmo_daily_REF.csv` (reference return series at full exposure) and `results/` (tables for the reference and three variants) |
| `results_charts/` | Frontier pass-probability vs days, pass vs burn by volatility target, equity and drawdown of the reference |

## Rules
- Basket: BTC, ETH, SOL, US100, US500, XAU, NVDA, AAPL + AMD + JNJ. Each asset has a grid sleeve (0/1), defined in `sleeves_motor.SLEEVES`.
- Eligible = sleeve long AND RSI(60) of buy&hold (Wilder, lag 1) > 50. Rank by RSI; 3 slots at 1/3; max 2 crypto; correlation filter 0.8 on 120 days; XAU fills an empty slot if its sleeve is long. Long only.
- Costs: 10 bp per weight change + real FTMO spreads and swaps. Crypto swap is an assumption (9%/yr, the 6-9% quoted for the Swing account; the Standard demo shows 30%).
- Risk overlay: `k = min(1, tau / EWMA30 vol)` x drawdown brake `max(0.25, (E-0.9)/0.1)` x daily cushion (2.5 sigma, min 1%). Evaluation tau = 10% (12% alternative); funded gear fixed `k = 0.25`.

## Results (backtest 2018-2026, in-sample selection; see caveats)
Full exposure, no overlay: Sharpe 2.03, CAGR 75%, max DD -38%, worst day -9.9% (8 of the 10 worst days are crypto). See `results_charts/equity_referencia.png`.
Monte Carlo (stationary block bootstrap, 30-day blocks, 6,000 paths; edge 100% / 50% of the historical mean):

| Eval vol target | Pass both phases | Burn in eval | Median days |
|---|---|---|---|
| 10% | 81% / 58% | 18% / 29% | 196 / 302 |
| 12% | 73% / 52% | 27% / 43% | 155 / 224 |
| 15% | 60% / 42% | 40% / 57% | 110 / 150 |
| 18% | 52% / 35% | 48% / 65% | 87 / 106 |

Funded gear `k = 0.25`: burns 0.6% / 5% of those who pass. EV/day (the criterion of the Monte Carlo post) rises only slightly with risk (10% to 22% vol: 19 to 22 per day at 100% edge) while EV per attempt halves. The 75% CAGR is at full exposure; with tau = 10% the portfolio runs at about a third of the size, which is why the median time to pass is ~200 days.

## Levers tested (none improves the reference)
Earnings-gap blackout, 1/vol weights, hysteresis +1/+2/+3, RSI window 30/45/90/120 (60 sits on a plateau), 12-1 momentum ranking (with/without 3m filter), Nasdaq > MA200 market filter, GARCH(1,1) volatility filter (p90/p80), cap of 1 crypto, 4 positions. Lower historical drawdown does not move the pass-probability vs days frontier (`results_charts/frontera_pasa_vs_dias.png`): the Sharpe does.

## Usage
```
pip install yfinance arch pandas numpy
cd notebooks
python senal_diaria.py --equity 100450          # after the US close; add --ref3 <max of last 3 equity closes> and --tau 0.12 if needed
python ftmo_rotativo.py --sleeves-csv --top 3 --cap 2 --ref AMD,JNJ --mejoras --tag run1
cd ../montecarlo
python mc_prop.py ftmo_daily_REF.csv results/mc_new.csv
```
The scripts find `../data` by themselves. At every run the signal script recomputes the sleeves from prices, compares them with the CSVs in `data/csv_sleeves/` (expect ~100% match) and compares past logged signals with today's recalculation. `--solo-csv` uses only the CSVs.

## Caveats
- Sleeves and AMD/JNJ were selected on the same 2018-2026 history (AMD and JNJ among 7 candidates). The JNJ sleeve does NOT pass the grid filters (Sharpe ALL 1.01 < 1.35, DSR 0.29). True out-of-sample starts 2026-11-16.
- Sleeve rules reproduce the grid CSVs 100% offline for BTC, AAPL, NVDA, QQQ, SPY and JNJ; ETH, SOL, AMD and GLD (Donchian/Keltner) need real OHLC and are checked at runtime.
- Crypto swap (9% vs 30% on the Standard demo), stock/index commissions and the FTMO rules/fee (~540 EUR, refundable at first payout) come from secondary sources: confirm on ftmo.com.
- Live execution differs from the lag-1 close-to-close backtest (stocks fill at the next open); the forward test is meant to measure that gap.
- Funded phase simulated for 365 days only. Monte Carlo euros are relative, not a forecast.
