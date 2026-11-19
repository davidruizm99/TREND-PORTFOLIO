# Review points 1-10: what was changed, how it was checked, what is still open

Numbers below marked **PROXY** were produced in a sandbox without Yahoo access: prices were rebuilt from the `bh_return` column of the sleeve CSVs
(correlation 0.92 with my real reference series, mean daily return lower). They show the *direction and size* of each fix, **not** final numbers.
The final numbers need one run on a machine with Yahoo (commands at the bottom).

| # | Point | Fix (file) | Checked how | Open |
|---|---|---|---|---|
| 1 | Price cache froze | `descargar()`: every run re-downloads from *last bar − 7 days* and **replaces** the overlap; today's (provisional) bar is never stored; `comprobar_frescura()` aborts if any asset of the basket has no complete bar (crypto ≤ 2 d, others ≤ 5 d). Daily signal aborts instead of printing weights on old prices. | `tests/test_datos.py` (6 checks, no internet): cache extends, overlap replaced, today's bar dropped, stale ETH aborts, Monday with Friday bar passes. | Real Yahoo run not tested from here. |
| 2 | OOS helped choose the system | JNJ **out of the default reference** (it fails the grid's own filter; on the proxy each addition lowers pass / raises burn: base8 87.4 % → +AMD 83.2 % → +AMD+JNJ 81.3 %). `OOS_START` split kept only as a label; every OOS number is printed as contaminated; the clean OOS is the forward demo from 2026-11-16. Config + hashes are logged each run. | Proxy MC, 3 baskets. | I **cannot re-select the sleeves on IS only**: the grid and generator notebook are not in the repo. AMD and the 8 base sleeves were still chosen with OOS in sight. |
| 3 | The clock | One convention for backtest and live: signal uses closes ≤ t, fill at t+1, position earns from t+2. QQQ/SPY CSVs (exported with delay 1) get +1 day (`EXTRA_DELAY`); `sleeves_motor.py` uses delay 2 for all sleeves; correlation window ends the day *before* the decision. Stocks/indices/gold cannot be opened or closed on a day with no real bar (weekend, holiday): the position stays as it was. Live: run in the morning with yesterday's closes, trade the same day. | Live-computed sleeves vs CSV on the proxy: BTC, QQQ, SPY, NVDA, AAPL, JNJ **100 %**; ETH 98.9, SOL 91.1, GLD 98.8, AMD 99.8 (the proxy has no real High/Low, so Donchian/Keltner differ). | Check the "coincide %" line on your first real run. Not two literal calendars: one 7-day calendar with the trading lock above. |
| 4 | FTMO daily loss guessed | Intraday low = **worst of** (a) Brownian bridge conditioned on the close (gives green days a drawdown) and (b) 1.5 × red close. Same model in the Monte Carlo and in the historical evaluator. Drawdown brake now goes to **0** at the 10 % limit (was floor 25 %), also in the live tool. `--ref3` is **mandatory** with `--equity` (no default). | MC ablation in `montecarlo/results/ablacion_MC/`. **Lesson:** the bridge *alone* understates tail days and inflates pass 80 % → 96 %; I first shipped that by mistake and corrected it to worst-of-two. | Still approximations: no real floating P&L (needs intraday OHLC), no midnight CE(S)T reset (daily bars are UTC closes), `ref3` must come from the broker. |
| 5 | Pass-rate windows | `resumen_eval()`: IS windows end before 2023-01-01, OOS windows start after it, ≥ 365 days inside the split, max 1100 days. Reports **passed / burned (daily, max) / unfinished** separately, `n` starts and `n_ef` ≈ independent windows. | Grid on the proxy: τ=10 %, IS pass 86-87 %, burned 0 %, unfinished 13 %; **n_ef ≈ 10 (IS) and ≈ 6 (OOS)**. | With ~10 independent windows the historical pass rates say very little; the MC is the better guide. |
| 6 | Missing data became calm | Price forward-fill capped (crypto 2 d, others 5 d); beyond the cap the asset becomes ineligible instead of getting return 0. Sleeve CSVs: the **last row is no longer invented** ("hold"), fill capped, backtest cut at the last **common** date (2026-10-15 today, set by QQQ/SPY). Live: a stale CSV with no live rule takes its asset out; a stale price aborts. | Cut date and abort paths exercised in tests / smoke runs. | Refresh the sleeve CSVs from the grid notebook so the common end date moves forward. |
| 7 | Sizes | Weight = 1/3 × sleeve size (ETH/NVDA 0.25-1) in backtest **and** live; both come from the same `rotar()` weight matrix. | Proxy, base8+AMD+JNJ: sizes on vs off → max DD **−27 % vs −39 %**, CAGR 44 % vs 49 %. The old backtest ran ETH/NVDA at full size. | |
| 8 | Costs | The prop engine never reads `strategy_return`; it takes only `position/size` from the CSVs and charges commission + half spread (from your demo file) + swap once, so there is no double cost there. **Your `rotativo/portfolio_rotativo.py` did double count** (sleeve costs + 10 bp, and exits vanished): now gross returns × effective size with a single cost. Crypto swap default is now **30 %** inside the code (`SWAP_CRIPTO`, printed at start and stored in `run_log.jsonl`); before, the default was 9 % and my "30 %" run was an adjustment on top of the series. Live output now also gives lots (rounded down) and a margin estimate (assumes USD account). | Rotativo with LAG 2 + single cost: Sharpe 2.25 → 2.23 net (small change). | Commissions are still assumptions; Swing-demo swap unverified. |
| 9 | TQQQ 60-voter ensemble | `tqqq_motor.py`: systems are clustered by correlation of their positions (≥ 0.90 = same opinion) and each cluster counts once. Default is now clusters + delay 2; `pesos="conteo"`, `delay=1` kept only to reproduce the notebook. | On the proxy the 60 systems collapse to **14 independent groups**. | Re-run with real OHLC: `python tqqq_motor.py --json frozen_TQQQ.json --csv <TQQQ csv> --yahoo`. The prop basket does not use TQQQ. |
| 10 | Cannot be rebuilt | `run_log.jsonl`: one line per run with script sha256, parameters, first/last date per asset, price hash, sleeve-CSV hashes, seeds and library versions (live signal, backtests and Monte Carlo). Daily signal also saves `snapshots/senal_<date>.csv` (input snapshot) and creates `fills_log.csv` for the real fills. | Log lines inspected after smoke runs. | **Not fixable by me:** the generator notebook, raw data and the grid-search record are not in my hands, and I do not know what file "A" is. They have to come from you. |

## Effect on the headline numbers (τ = 10 %, one account)

| Series / engine | Monte Carlo | Pass 2 phases (edge 100 / 50 %) | Burn | Unfinished | Median days |
|---|---|---|---|---|---|
| Reference series, previous engine, swap 30 % adjusted | previous (sent to Rick) | 80.5 / 56.5 % | 18.9 / 29.7 % | 0.6 / 13.8 % | 203 / 305 |
| same series | **new** (worst-of-two low, brake → 0) | **80.2 / 55.5 %** | 18.9 / 20.7 % | 1.0 / 23.7 % | 203 / 308 |
| **PROXY**, new engine, base 8 | new | 87.4 / 58.3 % | 11.4 / 13.6 % | 1.1 / 28.1 % | 253 / 369 |
| **PROXY**, new engine, base 8 + AMD | new | 83.2 / 57.0 % | 16.0 / 18.5 % | 0.8 / 24.5 % | 240 / 351 |
| **PROXY**, new engine, base 8 + AMD + JNJ | new | 81.3 / 53.8 % | 17.6 / 20.9 % | 1.1 / 25.3 % | 246 / 341 |

Reading: the stricter simulator does not move the headline (80.5 → 80.2 %); the brake to 0 trades burns for "unfinished". The engine fixes (sizes) lower drawdown and
slow the pass (≈ 240-250 days). None of this removes the selection bias. Pass rates remain a simplified game until the forward demo says otherwise.

## Re-run on a machine with Yahoo
```
python tests/test_datos.py
cd notebooks && python ftmo_rotativo.py --sleeves-csv --mejoras --tag v2 --ref AMD      # writes ftmo_daily_mejoras_v2_REF.csv
python ../montecarlo/mc_prop.py ftmo_daily_mejoras_v2_REF.csv ../montecarlo/results/mc_REAL_v2.csv
python senal_diaria.py                                                                  # morning, closes of yesterday
```
